// Procedural celestial sphere: a starfield and a galaxy band, looked up with
// the escape direction of each ray and prefiltered by the ray's footprint.
//
// Near the photon sphere a pixel can map to a long, thin sliver of sky (or,
// on the far side of the shadow, to many degrees of it). Point-sampling the
// sky there sparkles, because each frame a different sub-pixel point lands on
// or off a star. Instead each pixel integrates the sky against a Gaussian
// whose covariance is the pixel's footprint on the celestial sphere
// (Σ = k² J Jᵀ, J = ∂(escape dir)/∂(pixel), see geodesic.wgsl), in the spirit
// of Igehy, "Tracing Ray Differentials" (1999), and EWA filtering (Heckbert
// 1989). Lensing conserves surface brightness, so this is also what a real
// camera would record: a magnified star gets brighter, a demagnified patch of
// sky averages out.

struct Footprint {
  t1: vec3f,
  t2: vec3f,
  inv: vec3f,       // Σ⁻¹ as (xx, xy, yy) in the (t1, t2) tangent basis
  norm: f32,        // 1 / (2π √det Σ): the Gaussian integrates to 1 over solid angle
  sigmaMax: f32,    // √(largest eigenvalue of Σ), radians
  cov: vec3f,       // Σ itself
};

fn makeFootprint(D: vec3f, Jx: vec3f, Jy: vec3f, extra: f32) -> Footprint {
  var fp: Footprint;
  let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(D.y) > 0.9);
  fp.t1 = normalize(cross(helper, D));
  fp.t2 = cross(D, fp.t1);
  let k = F.env.w;
  let ax = vec2f(dot(Jx, fp.t1), dot(Jx, fp.t2)) * k;
  let ay = vec2f(dot(Jy, fp.t1), dot(Jy, fp.t2)) * k;
  let s2 = extra * extra;
  let sxx = ax.x * ax.x + ay.x * ay.x + s2;
  let syy = ax.y * ax.y + ay.y * ay.y + s2;
  let sxy = ax.x * ax.y + ay.x * ay.y;
  let det = max(sxx * syy - sxy * sxy, 1e-30);
  fp.cov = vec3f(sxx, sxy, syy);
  fp.inv = vec3f(syy, -sxy, sxx) / det;
  fp.norm = 1.0 / (TAU * sqrt(det));
  let h = 0.5 * (sxx + syy);
  fp.sigmaMax = sqrt(h + sqrt(max(h * h - det, 0.0)));
  return fp;
}

// Adds an isotropic variance (e.g. a finite star size) to a footprint.
fn widen(fp: Footprint, extra: f32) -> Footprint {
  var o = fp;
  let s2 = extra * extra;
  let c = fp.cov + vec3f(s2, 0.0, s2);
  let det = max(c.x * c.z - c.y * c.y, 1e-30);
  o.cov = c;
  o.inv = vec3f(c.z, -c.y, c.x) / det;
  o.norm = 1.0 / (TAU * sqrt(det));
  let h = 0.5 * (c.x + c.z);
  o.sigmaMax = sqrt(h + sqrt(max(h * h - det, 0.0)));
  return o;
}

// Filter weight (per steradian) of a point source at direction S.
fn kernelAt(fp: Footprint, D: vec3f, S: vec3f) -> f32 {
  if (dot(S, D) <= 0.0) { return 0.0; }
  let d = vec2f(dot(S, fp.t1), dot(S, fp.t2));
  let q = fp.inv.x * d.x * d.x + 2.0 * fp.inv.y * d.x * d.y + fp.inv.z * d.y * d.y;
  if (q > 24.0) { return 0.0; }
  return fp.norm * exp(-0.5 * q);
}

// ---- Equi-angular cube map for the star grid -------------------------------
// Plain cube-map cells vary 5× in solid angle; the tan warp brings that down
// to about 1.4×, and the unresolved-star mean below uses the exact cell area.

struct CubePos {
  face: u32,
  st: vec2f,    // [0, 1]² on the face
  uv: vec2f,    // gnomonic face coordinates in [-1, 1]²
};

fn cubePos(d: vec3f) -> CubePos {
  let a = abs(d);
  var c: CubePos;
  if (a.x >= a.y && a.x >= a.z) {
    c.face = select(0u, 1u, d.x < 0.0);
    c.uv = d.yz / a.x;
  } else if (a.y >= a.z) {
    c.face = select(2u, 3u, d.y < 0.0);
    c.uv = d.zx / a.y;
  } else {
    c.face = select(4u, 5u, d.z < 0.0);
    c.uv = d.xy / a.z;
  }
  c.st = atan(c.uv) * (2.0 / PI) + 0.5;
  return c;
}

fn cubeDir(face: u32, st: vec2f) -> vec3f {
  let uv = tan((st * 2.0 - 1.0) * (PI * 0.25));
  let s = 1.0 - 2.0 * f32(face & 1u);
  let axis = face >> 1u;
  var d: vec3f;
  if (axis == 0u) {
    d = vec3f(s, uv.x, uv.y);
  } else if (axis == 1u) {
    d = vec3f(uv.y, s, uv.x);
  } else {
    d = vec3f(uv.x, uv.y, s);
  }
  return normalize(d);
}

// Solid angle of an equi-angular cell (N cells per face edge) at face coords uv.
fn cellSolidAngle(uv: vec2f, n: f32) -> f32 {
  let a = 1.0 + uv.x * uv.x;
  let b = 1.0 + uv.y * uv.y;
  let c = 1.0 + uv.x * uv.x + uv.y * uv.y;
  let ds = 2.0 / n;
  return ds * ds * (PI * PI / 16.0) * a * b / (c * sqrt(c));
}

// ---- Galaxy ----------------------------------------------------------------

const BAND_WIDTH: f32 = 0.11;   // Gaussian σ of the band in sin(latitude)
const BULGE_VAR: f32 = 0.02;    // bulge variance in angle² (σ ≈ 8°)

// The smooth large-scale profiles are Gaussians, so they are prefiltered
// exactly: a Gaussian of variance v blurred by the footprint σ is a Gaussian
// of variance v + σ² (scaled to keep its integral). Once the footprint spans
// the whole band (next to the photon ring) this spreads the band over the
// sky instead of aliasing it. Small-angle approximation on the sphere.
fn bandProfile(lat: f32, v: f32, sigma: f32) -> f32 {
  let vs = v + sigma * sigma;
  return sqrt(v / vs) * exp(-0.5 * lat * lat / vs);
}

fn bulgeProfile(cosC: f32, sigma: f32) -> f32 {
  let vs = BULGE_VAR + sigma * sigma;
  // 1 - cos θ ≈ θ²/2; a 2D Gaussian keeps its integral with v / (v + σ²).
  return (BULGE_VAR / vs) * exp(-(1.0 - cosC) / vs);
}

// Smooth relative star density: a uniform halo plus the band and the bulge.
fn starDensity(d: vec3f, sigma: f32) -> f32 {
  let band = bandProfile(dot(d, F.galaxyN), BAND_WIDTH * BAND_WIDTH, sigma);
  let bulge = bulgeProfile(dot(d, F.galaxyC), sigma);
  return 0.25 + 0.75 * max(band, bulge);
}

fn galaxy(d: vec3f, sigma: f32, g: f32) -> vec3f {
  let lat = dot(d, F.galaxyN);
  let band = bandProfile(lat, BAND_WIDTH * BAND_WIDTH, sigma);
  let bulge = bulgeProfile(dot(d, F.galaxyC), sigma);
  // Noise over the direction itself, so there is no seam anywhere on the
  // sphere; its octaves fade out with the footprint (their mean is zero).
  let structure = fbmFiltered(d * 5.0, 6u, sigma * 5.0, 11u);
  let dustNoise = fbmFiltered(d * 11.0 + 3.0, 5u, sigma * 11.0, 23u);
  let gasNoise = fbmFiltered(d * 7.0 - 5.0, 4u, sigma * 7.0, 41u);
  // Dust lanes absorb along the mid-plane; emission nebulae glow pink (Hα) near it.
  let dust = smoothstep(-0.05, 0.4, dustNoise) * bandProfile(lat, 0.5 * BAND_WIDTH * BAND_WIDTH, sigma);
  let absorb = exp(-2.2 * dust);
  let glow = (band * (0.5 + 0.9 * max(structure, -0.45)) + 1.2 * bulge) * absorb;
  let hII = smoothstep(0.25, 0.6, gasNoise) * band * absorb;
  // Old stars (bulge) are warm, the disc bluer. Blackbody components so the
  // observer's blueshift g moves their colour consistently with the stars.
  let warm = blackbody(4200.0 * g) / blackbody(4200.0).g;
  let cool = blackbody(11000.0 * g) / blackbody(11000.0).g;
  let colour = mix(cool, warm, clamp(0.35 + 0.8 * bulge + 0.4 * structure, 0.0, 1.0));
  // (Line emission shifts in wavelength, not temperature: g³ at fixed frequency.)
  return glow * colour + hII * vec3f(1.0, 0.28, 0.42) * (0.35 * g * g * g) + 0.012 * cool;
}

// ---- Stars -----------------------------------------------------------------

const STAR_LEVELS: u32 = 9u;
const STAR_N0: f32 = 4.0;            // cells per face edge on the coarsest level
const STAR_FLUX_STEP: f32 = 0.3;     // flux ratio between levels (4× more stars each)
const STAR_PRESENCE: f32 = 0.8;
const STAR_FLUX_MEAN: f32 = 1.3525;  // E[2^(4h-2)], h uniform

fn bbLog2Lum(T: f32) -> f32 {
  let x = clamp(log2(T / BB_T_MIN) / log2(BB_T_MAX / BB_T_MIN) * 255.0, 0.0, 255.0);
  let i = min(u32(x), 254u);
  return mix(bb[i].a, bb[i + 1u].a, x - f32(i));
}

// Colour of a star of temperature T seen with blueshift g, normalised so its
// luminance is 1 at g = 1.
fn starColour(T: f32, g: f32) -> vec3f {
  return blackbody(T * g) * exp2(-bbLog2Lum(T));
}

fn starLevel(level: u32, d: vec3f, cp: CubePos, fp: Footprint, g: f32, meanColour: vec3f, density: f32) -> vec3f {
  let n = STAR_N0 * f32(1u << level);
  let cellAngle = (0.5 * PI) / n;
  let flux = pow(STAR_FLUX_STEP, f32(level)) * 2.0e-5;
  // Resolved while the kernel's 3σ reach fits inside the 2×2 cells searched;
  // past that the level has become sub-footprint "haze": use its mean
  // radiance, which is what a pixel would average (the level's top mip).
  let resolved = 1.0 - smoothstep(0.08, 0.16, fp.sigmaMax / cellAngle);
  let mean = flux * STAR_FLUX_MEAN * STAR_PRESENCE * density / cellSolidAngle(cp.uv, n);
  var sum = (1.0 - resolved) * mean * meanColour;
  if (resolved <= 0.0) { return sum; }

  let f = cp.st * n - 0.5;
  let base = floor(f);
  var stars = vec3f(0.0);
  for (var k = 0u; k < 4u; k++) {
    let c = base + vec2f(f32(k & 1u), f32(k >> 1u));
    if (any(c < vec2f(0.0)) || any(c > vec2f(n - 1.0))) { continue; }
    let h = hash33(vec3u(u32(c.x), u32(c.y), cp.face + 6u * level + 101u));
    let centre = cubeDir(cp.face, (c + 0.5) / n);
    if (h.x > STAR_PRESENCE * starDensity(centre, 0.0)) { continue; }
    let h2 = hash33(pcg3d(vec3u(u32(c.x), u32(c.y), cp.face + 6u * level + 977u)));
    // Keep stars in cells on a face edge at least half a cell from the edge,
    // so a star is never needed from across a cube seam.
    let lo = select(vec2f(0.1), vec2f(0.5), c == vec2f(0.0));
    let hi = select(vec2f(0.9), vec2f(0.5), c == vec2f(n - 1.0));
    let s = cubeDir(cp.face, (c + mix(lo, hi, h2.xy)) / n);
    let w = kernelAt(fp, d, s);
    if (w <= 0.0) { continue; }
    let T = 2600.0 + 26000.0 * h.y * h.y * h.y;
    stars += (flux * exp2(4.0 * h2.z - 2.0) * w) * starColour(T, g);
  }
  return sum + resolved * stars;
}

fn starfield(d: vec3f, fp: Footprint, g: f32) -> vec3f {
  let cp = cubePos(d);
  let meanColour = starColour(5200.0, g);
  let density = starDensity(d, fp.sigmaMax);
  var sum = vec3f(0.0);
  for (var level = 0u; level < STAR_LEVELS; level++) {
    sum += starLevel(level, d, cp, fp, g, meanColour, density);
  }
  return sum;
}

// Radiance of the sky in direction d. g: blueshift of light from infinity at
// the observer (gravitational and, for a moving observer, Doppler).
fn environment(d: vec3f, Jx: vec3f, Jy: vec3f, g: f32) -> vec3f {
  let fp = makeFootprint(d, Jx, Jy, 0.0);
  var c = vec3f(0.0);
  if (F.sky.z > 0.5) {
    c += F.env.x * starfield(d, widen(fp, F.sky.x), g);
  }
  if (F.sky.w > 0.5) {
    c += F.env.y * galaxy(d, fp.sigmaMax, g);
  }
  if (F.beaconFlux > 0.0) {
    let bfp = widen(fp, F.sky.y);
    c += F.beaconFlux * kernelAt(bfp, d, F.beaconDir) * starColour(9000.0, g);
  }
  // Specific intensity scales as g³ at fixed frequency and the spectrum shifts:
  // for blackbodies that is exactly T → gT, done above. Surface brightness
  // is otherwise conserved along the ray, so lensing needs no extra factor.
  return c;
}
