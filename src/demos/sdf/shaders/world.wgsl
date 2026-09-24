// =============================================================================
//  SDF WORLD: an entire scene from signed distance functions, sphere traced in
//  ONE fullscreen fragment shader. One render pipeline, one draw of a
//  fullscreen triangle, no textures, no compute, no extra passes and no
//  history buffers: every pixel rebuilds the world from maths every frame.
//  The CPU uploads a single uniform block (camera, time, GUI values). The
//  only other binding is a tiny atomic counter buffer for the cost meter.
//
//  Sections
//     1. Uniforms and constants
//     2. Hashes and noise (value noise with analytic derivatives)
//     3. SDF primitives
//     4. SDF operators: smooth booleans with material blending, rounding,
//        onion, elongation, repetition, mirroring, twist, bend
//     5. Terrain: warped fBm height field, Lipschitz-safe distance, LOD
//     6. Architecture: temple, rotunda, plaza + stairs, aqueduct, bridge
//     7. Character: smooth-blended capsules and ellipsoids, procedural walk
//     8. Abstract sculpture: warped gyroid shell, metaballs, twisted ribbon
//     9. Scene map with bounding volumes
//    10. Ray marching: over-relaxed sphere tracing (Keinert et al. 2014)
//    11. Normals, soft shadows, ambient occlusion, thickness
//    12. Materials and BRDF
//    13. Water
//    14. Fog and the path loop with reflection bounces
//    15. Debug views, tone mapping, entry points
//
//  Requires src/shared/sky.wgsl (skyRadiance), which reads F.camPos,
//  F.sunDir, F.sunColor, F.sunIntensity and F.sky.
// =============================================================================


// -----------------------------------------------------------------------------
// 1. Uniforms and constants. Keep the layout in sync with index.ts.
// -----------------------------------------------------------------------------

struct Frame {
  invViewProj: mat4x4f,
  camPos: vec3f,
  time: f32,
  sunDir: vec3f,
  exposure: f32,
  sunColor: vec3f,       // sun irradiance at the ground
  sunIntensity: f32,     // sun irradiance at the top of the atmosphere
  ambient: vec3f,        // cosine-weighted average sky radiance
  debugView: f32,
  resolution: vec2f,     // internal (resolution-scaled) size in pixels
  pixelAngle: f32,       // angular size of one pixel (radians)
  statsOn: f32,          // 1 on frames whose cost is measured
  sky: vec4f,            // turbidity, rayleigh multiplier, mie g, multiple-scattering boost
  march: vec4f,          // max steps, over-relaxation omega, epsilon (pixels), max distance (m)
  light: vec4f,          // shadow steps, penumbra k, AO strength, reflection bounces
  toggles: vec4f,        // shadows, AO, fBm LOD, bounding volumes (0 or 1)
  fog: vec4f,            // density at water level (1/m), height falloff (1/m), SSS strength, water murk
  slice: vec4f,          // axis (0 = y, 1 = z, 2 = x), offset (m), contour spacing (m), supersampling
  anim: vec4f,           // walk speed (m/s), breathing, sculpture spin, sculpture warp
};

@group(0) @binding(0) var<uniform> F: Frame;
// Cost meter: [pixels, primary steps, map calls, groups, octaves, shadow steps, reflection steps, maxed-out rays]
@group(0) @binding(1) var<storage, read_write> stats: array<atomic<u32>, 8>;

const PI: f32 = 3.14159265;
const TAU: f32 = 6.28318531;

// World layout (metres). The temple faces +z, the river runs along x.
const WATER_Y: f32 = 0.0;
const PLAT_Y: f32 = 1.5;                         // plateau and plaza level
const STY_Y: f32 = 2.55;                         // top of the temple's stepped base
const TERRAIN_MAX: f32 = 112.0;                  // highest possible terrain point
const SCENE_TOP: f32 = 114.0;                    // nothing in the world is higher
const PLAZA_C: vec3f = vec3f(0.0, 1.5, 21.4);
const ROT_C: vec3f = vec3f(-34.0, 1.5, -4.0);    // rotunda centre
const AQ_Z: f32 = -46.0;                         // aqueduct axis (runs along x forever)
const BR_C: vec3f = vec3f(22.0, 4.1, 43.5);      // bridge deck centre

// Bounding boxes (centre, half size) of the expensive groups.
const TEMPLE_BC: vec3f = vec3f(1.5, 7.2, -1.0);
const TEMPLE_BH: vec3f = vec3f(9.3, 5.8, 14.8);
const PLAZA_BC: vec3f = vec3f(0.0, 2.45, 23.8);
const PLAZA_BH: vec3f = vec3f(15.6, 6.1, 11.8);
const BRIDGE_BC: vec3f = vec3f(22.0, 3.1, 43.5);
const BRIDGE_BH: vec3f = vec3f(2.3, 2.3, 13.4);

// Material ids. Hit.w blends m -> m2 (smooth unions blend materials too).
const MAT_TERRAIN: f32 = 1.0;
const MAT_MARBLE: f32 = 2.0;
const MAT_FLOOR: f32 = 3.0;      // polished marble tiles
const MAT_SANDSTONE: f32 = 4.0;
const MAT_GOLD: f32 = 5.0;
const MAT_CHROME: f32 = 6.0;
const MAT_SKIN: f32 = 7.0;
const MAT_CLOTH: f32 = 8.0;
const MAT_WATER: f32 = 9.0;      // analytic plane, never returned by map()
const MAT_EYE: f32 = 10.0;
const MAT_ROOF: f32 = 11.0;
const MAT_STRAW: f32 = 12.0;

// Per-pixel cost counters (reset in fs, summed into `stats`).
var<private> gEvals: u32 = 0u;
var<private> gGroups: u32 = 0u;
var<private> gOct: u32 = 0u;
var<private> gShadowSteps: u32 = 0u;
var<private> gReflSteps: u32 = 0u;


// -----------------------------------------------------------------------------
// 2. Hashes and noise
// -----------------------------------------------------------------------------

// Integer lattice hashes (no sin(): stable on every GPU at large coordinates).
fn hash2i(c: vec2i) -> f32 {
  var h = (bitcast<u32>(c.x) * 0x27d4eb2du) ^ (bitcast<u32>(c.y) * 0x165667b1u);
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = (h ^ (h >> 12u)) * 0x297a2d39u;
  h = h ^ (h >> 15u);
  return f32(h) * (1.0 / 4294967296.0);
}

fn hash3i(c: vec3i) -> f32 {
  var h = (bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u) ^ (bitcast<u32>(c.z) * 0xcb1ab31fu);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) * (1.0 / 4294967296.0);
}

// 2D value noise in [-1, 1] with its analytic gradient: (value, d/dx, d/dy).
// The gradient is what makes the terrain's Lipschitz step (section 5) cheap.
fn noised(x: vec2f) -> vec3f {
  let i = floor(x);
  let f = x - i;
  let u = f * f * (3.0 - 2.0 * f);
  let du = 6.0 * f * (1.0 - f);
  let c = vec2i(i);
  let a = hash2i(c);
  let b = hash2i(c + vec2i(1, 0));
  let cc = hash2i(c + vec2i(0, 1));
  let d = hash2i(c + vec2i(1, 1));
  let k1 = b - a;
  let k2 = cc - a;
  let k3 = a - b - cc + d;
  let v = a + k1 * u.x + k2 * u.y + k3 * u.x * u.y;
  let g = du * vec2f(k1 + k3 * u.y, k2 + k3 * u.x);
  return vec3f(2.0 * v - 1.0, 2.0 * g);
}

// 3D value noise in [0, 1], used only for surface patterns at shading time.
fn noise3(x: vec3f) -> f32 {
  let i = floor(x);
  let f = x - i;
  let u = f * f * (3.0 - 2.0 * f);
  let c = vec3i(i);
  let a = mix(hash3i(c), hash3i(c + vec3i(1, 0, 0)), u.x);
  let b = mix(hash3i(c + vec3i(0, 1, 0)), hash3i(c + vec3i(1, 1, 0)), u.x);
  let e = mix(hash3i(c + vec3i(0, 0, 1)), hash3i(c + vec3i(1, 0, 1)), u.x);
  let g = mix(hash3i(c + vec3i(0, 1, 1)), hash3i(c + vec3i(1, 1, 1)), u.x);
  return mix(mix(a, b, u.y), mix(e, g, u.y), u.z);
}

fn fbm3(x: vec3f) -> f32 {
  return 0.5 * noise3(x) + 0.3 * noise3(x * 2.03 + 7.1) + 0.2 * noise3(x * 4.1 + 3.3);
}

// smoothstep and its derivative with respect to x.
fn smoothstepD(e0: f32, e1: f32, x: f32) -> vec2f {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return vec2f(t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t) / (e1 - e0));
}


// -----------------------------------------------------------------------------
// 3. SDF primitives (after Inigo Quilez's catalogue)
// -----------------------------------------------------------------------------

fn sdSphere(p: vec3f, r: f32) -> f32 {
  return length(p) - r;
}

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdBox2(p: vec2f, b: vec2f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
}

// Rounding operator applied to a box: shrink by r, then inflate by r.
fn sdRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
  return sdBox(p, b - vec3f(r)) - r;
}

// Capped cylinder along y, radius r, half height h.
fn sdCylY(p: vec3f, r: f32, h: f32) -> f32 {
  let d = abs(vec2f(length(p.xz), p.y)) - vec2f(r, h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sdCapsule(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

fn sdTorus(p: vec3f, R: f32, r: f32) -> f32 {
  return length(vec2f(length(p.xz) - R, p.y)) - r;
}

// Ellipsoid (a bound, not exact; good enough for sphere tracing).
fn sdEllipsoid(p: vec3f, r: vec3f) -> f32 {
  let k0 = length(p / r);
  let k1 = length(p / (r * r));
  return k0 * (k0 - 1.0) / max(k1, 1e-6);
}

// Capped cone centred on the origin: radius r1 at y = -h, r2 at y = +h (exact).
fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(r2, h);
  let k2 = vec2f(r2 - r1, 2.0 * h);
  let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
  let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

// Cone with rounded ends along y (requires |r1 - r2| < h): radius r1 at y = 0, r2 at y = h.
fn sdRoundCone(p: vec3f, r1: f32, r2: f32, h: f32) -> f32 {
  let b = (r1 - r2) / h;
  let a = sqrt(1.0 - b * b);
  let q = vec2f(length(p.xz), p.y);
  let k = dot(q, vec2f(-b, a));
  if (k < 0.0) { return length(q) - r1; }
  if (k > a * h) { return length(q - vec2f(0.0, h)) - r2; }
  return dot(q, vec2f(a, b)) - r1;
}


// -----------------------------------------------------------------------------
// 4. SDF operators
// -----------------------------------------------------------------------------

// A distance plus a material that may be a blend of two ids.
struct Hit {
  d: f32,
  m: f32,
  m2: f32,
  w: f32,     // material = mix(m, m2, w)
};

fn hitOf(d: f32, m: f32) -> Hit {
  return Hit(d, m, m, 0.0);
}

fn dominant(h: Hit) -> f32 {
  return select(h.m, h.m2, h.w > 0.5);
}

fn opU(a: Hit, b: Hit) -> Hit {
  if (b.d < a.d) { return b; }
  return a;
}

// Polynomial smooth minimum (quadratic, Quilez). Returns (distance, weight of a).
fn sminH(a: f32, b: f32, k: f32) -> vec2f {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return vec2f(mix(b, a, h) - k * h * (1.0 - h), h);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
  return sminH(a, b, k).x;
}

// Smooth intersection; smooth subtraction is smax(a, -b, k).
fn smax(a: f32, b: f32, k: f32) -> f32 {
  return -smin(-a, -b, k);
}

// Smooth union that also blends the materials by the same weight, so the seam
// between skin and cloth (or chrome and gold) is a gradient, not a line.
fn opSmoothU(a: Hit, b: Hit, k: f32) -> Hit {
  let s = sminH(a.d, b.d, k);
  return Hit(s.x, dominant(a), dominant(b), 1.0 - s.y);
}

// Onion: turns any solid into a shell of thickness 2t.
fn opOnion(d: f32, t: f32) -> f32 {
  return abs(d) - t;
}

// Elongation (exact, Quilez): stretches a primitive by h along each axis by
// inserting a straight section. An elongated circle is a stadium, which is
// exactly the profile of a round-headed arch.
fn sdArch(v: vec2f, r: f32, e: f32) -> f32 {
  let q = abs(v) - vec2f(0.0, e);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Infinite repetition with period s.
fn opRep(x: f32, s: f32) -> f32 {
  return x - s * round(x / s);
}

// Limited repetition: cell ids clamped to [lo, hi].
fn opRepLim(x: f32, s: f32, lo: f32, hi: f32) -> f32 {
  return x - s * clamp(round(x / s), lo, hi);
}

// Polar (angular) repetition: folds the plane into one of n sectors around
// the origin. Returns (radial, tangential) coordinates in that sector.
fn opPolar(p: vec2f, n: f32) -> vec2f {
  let an = TAU / n;
  let id = round(atan2(p.y, p.x) / an);
  let c = cos(id * an);
  let s = sin(id * an);
  return vec2f(c * p.x + s * p.y, -s * p.x + c * p.y);
}

fn rot2(v: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Twist around y by k radians per metre. Not an isometry: a point at radius r
// moves sideways k·r per metre of height, so distances must be scaled by
// 1/sqrt(1 + (k·r_max)^2) to stay a lower bound.
fn opTwistY(p: vec3f, k: f32) -> vec3f {
  let xz = rot2(p.xz, k * p.y);
  return vec3f(xz.x, p.y, xz.y);
}

// Cheap bend: bends the z axis into an arc in the zy plane (k = 1/radius).
// With k < 0 the middle rises, which turns a straight slab into an arched deck.
fn opBendZY(p: vec3f, k: f32) -> vec3f {
  let c = cos(k * p.z);
  let s = sin(k * p.z);
  return vec3f(p.x, -s * p.z + c * p.y, c * p.z + s * p.y);
}


// -----------------------------------------------------------------------------
// 5. Terrain
//
// A height field h(x, z) gives a vertical offset p.y - h, not a distance. On a
// slope of gradient g the true distance is at most (p.y - h) / sqrt(1 + |g|^2)
// (the distance to the tangent plane), and a sphere tracer that uses the raw
// offset oversteps through hillsides. The Lipschitz-safe distance divides by
// a slope bound L: d = (p.y - h) / sqrt(1 + L^2).
//
// Using the global bound (cliff faces up to ~5:1, so d ≈ 0.2 Δy) everywhere wastes
// most of the steps on flat ground at grazing angles, which is where the
// majority of terrain pixels are. So L is local near the surface: the analytic
// gradient of the fBm, the warp Jacobian and the cliff profile, plus a margin
// for curvature. It blends to the global bound once the point is more than a
// few metres above the ground, where a steep cliff might be closer sideways
// than the ground below it is vertically.
//
// fBm level of detail: octaves whose wavelength is below ~4 pixel footprints
// are dropped (the last one fades out fractionally, so there is no popping),
// and the loop stops early when the point is so far above the partial sum
// that the remaining octaves (at most ±2·amplitude) cannot reach it.
// -----------------------------------------------------------------------------

const TER_FREQ: f32 = 0.018;     // first octave: 55 m wavelength
const TER_AMP: f32 = 3.6;        // first octave amplitude (m); the sum is within ±2·TER_AMP
const TER_OCT: f32 = 9.0;
const WARP_F: f32 = 0.011;
const WARP_A: f32 = 12.0;        // domain warp amplitude (m)
const TER_K_GLOBAL: f32 = 0.2;   // 1/sqrt(1 + L^2) for the steepest cliffs (L ≈ 5)
const ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);

// Returns (height, dh/dx, dh/dz, slack). slack > 0 means the fBm stopped early
// and the true surface may be up to `slack` metres higher.
fn terrainField(xz: vec2f, footprint: f32, probeY: f32) -> vec4f {
  // River: a meandering channel along x, in world space.
  let zc = 42.0 + 5.0 * sin(0.025 * xz.x);
  let dz = xz.y - zc;
  let rv = smoothstepD(5.0, 13.0, abs(dz));
  let rt = 1.0 - rv.x;
  // Plateau under the temple, plaza and rotunda.
  let pb = sdBox2(xz - vec2f(-10.0, -1.0), vec2f(40.0, 29.0));
  let pt = 1.0 - smoothstep(0.0, 14.0, pb);
  if (pt >= 1.0 && rt <= 0.0) {
    return vec4f(PLAT_Y - 0.2, 0.0, 0.0, 0.0);   // flat: skip all the noise
  }

  // Low-frequency domain warp with its analytic Jacobian.
  let n1 = noised(xz * WARP_F + vec2f(3.1, 7.7));
  let n2 = noised(xz * WARP_F + vec2f(-5.2, 1.3));
  let q = xz + WARP_A * vec2f(n1.x, n2.x);
  let jx = vec2f(1.0, 0.0) + (WARP_A * WARP_F) * n1.yz;   // gradient of q.x
  let jz = vec2f(0.0, 1.0) + (WARP_A * WARP_F) * n2.yz;   // gradient of q.y (warped z)

  // Valley profile across the (warped) valley axis: floor, cliffs, mountains.
  let vz = q.y - 20.0;
  let s = abs(vz);
  let gs = select(-1.0, 1.0, vz >= 0.0) * jz;
  var h = 1.7;
  var g = vec2f(0.0);
  // The cliff line wanders with two octaves of noise stretched along it,
  // which carves buttresses and gullies into the faces.
  let c1 = noised(vec2f(q.x * 0.07, q.y * 0.03));
  let c2 = noised(vec2f(q.x * 0.23, q.y * 0.1) + 17.0);
  let sc = s + 4.0 * c1.x + 1.3 * c2.x;
  let gsc = gs + 4.0 * (0.07 * c1.y * jx + 0.03 * c1.z * jz) + 1.3 * (0.23 * c2.y * jx + 0.1 * c2.z * jz);
  let cl = smoothstepD(78.0, 87.0, sc);      // lower cliff band
  h += 14.0 * cl.x;
  g += 14.0 * cl.y * gsc;
  let cu = smoothstepD(93.0, 101.0, sc);     // upper band above a grassy ledge
  h += 12.0 * cu.x;
  g += 12.0 * cu.y * gsc;
  let mt = smoothstepD(130.0, 330.0, s);
  h += 72.0 * mt.x;
  g += 72.0 * mt.y * gs;

  // fBm hills, gentle on the valley floor and rugged on cliffs and mountains.
  let rm = smoothstepD(35.0, 100.0, s);
  let rough = 0.22 + 0.78 * rm.x;
  var octF = TER_OCT;
  if (F.toggles.z > 0.5) {
    octF = clamp(log2(1.0 / (TER_FREQ * 4.0 * footprint)), 2.0, TER_OCT);
  }
  let nOct = u32(ceil(octF));
  var pp = q * TER_FREQ;
  var M = mat2x2f(TER_FREQ, 0.0, 0.0, TER_FREQ);   // transpose of d(pp)/dq
  var amp = TER_AMP;
  var sum = 0.0;
  var gq = vec2f(0.0);
  var slack = 0.0;
  for (var i = 0u; i < nOct; i++) {
    let n = noised(pp);
    let w = clamp(octF - f32(i), 0.0, 1.0);
    sum += amp * w * n.x;
    gq += amp * w * (M * n.yz);
    gOct += 1u;
    amp *= 0.5;
    pp = 2.0 * (ROT * pp) + vec2f(1.7, 9.2);
    M = 2.0 * (M * transpose(ROT));
    let rest = 2.0 * amp * rough;
    if (probeY - (h + rough * sum) - rest > 3.0) {
      slack = rest;
      break;
    }
  }
  h += rough * sum;
  g += rough * (gq.x * jx + gq.y * jz) + sum * 0.78 * rm.y * gs;

  // Apply the plateau (its own small gradient term is left to the margin),
  // then carve the river through everything.
  g *= 1.0 - pt;
  h = mix(h, PLAT_Y - 0.2, pt);
  let grt = -rv.y * select(-1.0, 1.0, dz >= 0.0) * vec2f(-0.125 * cos(0.025 * xz.x), 1.0);
  g = g * (1.0 - rt) + (-2.6 - h) * grt;
  h = mix(h, -2.6, rt);
  return vec4f(h, g, slack);
}

fn sdTerrain(p: vec3f) -> f32 {
  if (p.y > TERRAIN_MAX + 10.0) { return p.y - TERRAIN_MAX; }
  let footprint = F.pixelAngle * max(distance(p, F.camPos), 0.5);
  let T = terrainField(p.xz, footprint, p.y);
  let dy = p.y - T.x - T.w;
  let L = length(T.yz) + 0.3;
  let kLocal = inverseSqrt(1.0 + L * L);
  return dy * mix(kLocal, TER_K_GLOBAL, smoothstep(1.0, 16.0, dy));
}


// -----------------------------------------------------------------------------
// 6. Architecture
// -----------------------------------------------------------------------------

// A fluted column with its base at the origin: plinth and torus, a shaft with
// entasis and 20 flutes carved by subtraction in polar-repeated space, and an
// echinus smoothly blended into the shaft under a square abacus.
fn sdColumn(q: vec3f, h: f32) -> f32 {
  let yN = clamp(q.y / h, 0.0, 1.0);
  let r = 0.44 * (1.0 - 0.12 * yN + 0.05 * sin(PI * yN));
  var d = max(length(q.xz) - r, abs(q.y - h * 0.5) - h * 0.5);
  if (d < 0.12) {
    let fp = opPolar(q.xz, 20.0);
    let flute = max(length(vec2f(fp.x - r - 0.03, fp.y)) - 0.075, abs(q.y - h * 0.5) - (h * 0.5 - 0.55));
    d = max(d, -flute);
  }
  d = min(d, sdRoundBox(q - vec3f(0.0, 0.09, 0.0), vec3f(0.6, 0.09, 0.6), 0.02));
  d = min(d, sdTorus(q - vec3f(0.0, 0.25, 0.0), 0.47, 0.08));
  d = smin(d, sdEllipsoid(q - vec3f(0.0, h - 0.28, 0.0), vec3f(0.6, 0.16, 0.6)), 0.08);
  d = min(d, sdRoundBox(q - vec3f(0.0, h - 0.07, 0.0), vec3f(0.66, 0.07, 0.66), 0.015));
  return d;
}

// The collapsed corner of the temple: a lumpy ellipsoid subtracted from the
// columns, entablature, roof and cella (sine lumps with a bounded gradient,
// scaled to stay Lipschitz).
fn sdCollapse(p: vec3f) -> f32 {
  let c = p - vec3f(5.0, 10.8, -10.5);
  let n = sin(c.x * 2.3 + sin(c.z * 1.7)) * sin(c.y * 2.1 + sin(c.x * 1.3)) * sin(c.z * 1.9 + c.y);
  return (length(c * vec3f(1.0, 0.75, 1.0)) - 4.3 + 0.3 * n) * 0.6;
}

// A hexastyle temple, mirrored in x and z, with a partly collapsed back corner.
fn sdTemple(p: vec3f) -> Hit {
  // Stepped base (crepidoma): three rounded boxes; the top one is polished.
  var dBase = sdRoundBox(p - vec3f(0.0, PLAT_Y + 0.175, 0.0), vec3f(7.6, 0.175, 13.6), 0.03);
  dBase = min(dBase, sdRoundBox(p - vec3f(0.0, PLAT_Y + 0.525, 0.0), vec3f(7.2, 0.175, 13.2), 0.03));
  var res = hitOf(dBase, MAT_MARBLE);
  res = opU(res, hitOf(sdRoundBox(p - vec3f(0.0, PLAT_Y + 0.875, 0.0), vec3f(6.8, 0.175, 12.8), 0.03), MAT_FLOOR));

  let a = vec3f(abs(p.x), p.y - STY_Y, abs(p.z));   // mirrored coordinates
  let cz = sdCollapse(p);

  // Colonnade: limited repetition along each side, only evaluated inside the
  // rectangular band that holds the columns (a cheap sub-bound).
  let band = max(sdBox(vec3f(p.x, p.y - STY_Y - 3.2, p.z), vec3f(6.75, 3.25, 12.75)),
                 -sdBox(p, vec3f(5.25, 100.0, 11.25)));
  if (band < res.d) {
    let side = vec3f(a.x - 6.0, a.y, opRepLim(p.z, 2.4, -5.0, 5.0));           // x = ±6, 11 per side
    let front = vec3f(opRepLim(p.x - 1.2, 2.4, -2.0, 1.0), a.y, a.z - 12.0);   // z = ±12, x = ±1.2, ±3.6
    let dc = max(min(sdColumn(side, 6.4), sdColumn(front, 6.4)), -cz);
    res = opU(res, hitOf(dc, MAT_MARBLE));
  }

  // Entablature with triglyph grooves (limited repetition, subtracted) and a cornice.
  let ey = STY_Y + 6.4;
  var de = sdBox(vec3f(p.x, p.y - ey - 0.65, p.z), vec3f(6.75, 0.65, 12.75));
  de = min(de, sdBox(vec3f(p.x, p.y - ey - 1.36, p.z), vec3f(6.95, 0.08, 12.95)));
  if (de < 0.3) {
    let fy = p.y - ey - 0.95;
    let g1 = sdBox(vec3f(opRepLim(p.x, 0.6, -11.0, 11.0), fy, a.z - 12.75), vec3f(0.035, 0.24, 0.06));
    let g2 = sdBox(vec3f(a.x - 6.75, fy, opRepLim(p.z, 0.6, -21.0, 21.0)), vec3f(0.06, 0.24, 0.035));
    de = max(de, -min(g1, g2));
  }
  res = opU(res, hitOf(max(de, -cz), MAT_MARBLE));

  // Roof: a triangular prism (two sloped planes, a floor, two gable ends),
  // with the tympanum recess carved into each gable.
  let ry = ey + 1.44;
  let slope = normalize(vec2f(2.05, 7.0));
  var dr = max(max(dot(vec2f(a.x, p.y - ry), slope) - 7.0 * slope.x, ry - p.y), a.z - 13.0);
  let tymp = max(max(dot(vec2f(a.x, p.y - ry - 0.22), slope) - 5.9 * slope.x, ry + 0.22 - p.y), abs(a.z - 13.0) - 0.25);
  dr = max(max(dr, -tymp), -cz);
  res = opU(res, hitOf(dr, MAT_ROOF));

  // Cella: an onion (shell) of a box, with an arched door and arched windows
  // (elongated circles) carved through it.
  var dcel = opOnion(sdBox(vec3f(p.x, p.y - STY_Y - 3.3, p.z + 1.0), vec3f(4.2, 3.3, 8.6)), 0.3);
  let door = max(max(sdArch(vec2f(p.x, p.y - STY_Y - 2.2), 1.1, 2.2), STY_Y - p.y), abs(p.z - 7.6) - 0.6);
  let win = max(sdArch(vec2f(opRepLim(p.z + 1.0, 3.6, -1.0, 1.0), p.y - STY_Y - 3.4), 0.45, 0.7), abs(a.x - 4.2) - 0.6);
  dcel = max(max(dcel, -min(door, win)), -cz);
  res = opU(res, hitOf(dcel, MAT_MARBLE));

  // Rubble from the collapse: fallen drums (cylinders on their side) and a capital.
  var dr2 = 1e9;
  let q1 = p - vec3f(8.9, PLAT_Y + 0.44, -10.2);
  let r1 = rot2(q1.xz, 0.5);
  dr2 = min(dr2, sdCylY(vec3f(q1.y, r1.x, r1.y), 0.44, 0.5));
  let q2 = p - vec3f(8.0, PLAT_Y + 0.44, -13.9);
  let r2 = rot2(q2.xz, -0.35);
  dr2 = min(dr2, sdCylY(vec3f(q2.y, r2.x, r2.y), 0.44, 0.45));
  let q3 = p - vec3f(3.4, STY_Y + 0.44, -10.4);
  let r3 = rot2(q3.xz, 1.2);
  dr2 = min(dr2, sdCylY(vec3f(q3.y, r3.x, r3.y), 0.42, 0.55));
  let q4 = p - vec3f(9.6, PLAT_Y + 0.22, -6.9);
  let r4 = rot2(q4.xz, 0.3);
  dr2 = min(dr2, sdRoundBox(vec3f(r4.x, q4.y, r4.y), vec3f(0.66, 0.2, 0.66), 0.04));
  res = opU(res, hitOf(dr2, MAT_MARBLE));
  return res;
}

// A tholos: circular stepped base, 14 columns by polar repetition, a drum
// (onion of a cylinder) with arched windows, and a gilded dome (onion of a
// sphere) with an oculus and marble ribs.
fn sdRotunda(p: vec3f) -> Hit {
  let q = p - vec3f(ROT_C.x, PLAT_Y, ROT_C.z);
  let rxz = length(q.xz);
  var base = sdCylY(q - vec3f(0.0, 0.15, 0.0), 9.4, 0.15);
  base = min(base, sdCylY(q - vec3f(0.0, 0.45, 0.0), 9.0, 0.15));
  var res = hitOf(base, MAT_MARBLE);
  res = opU(res, hitOf(sdCylY(q - vec3f(0.0, 0.75, 0.0), 8.6, 0.15), MAT_FLOOR));
  let y0 = 0.9;

  if (abs(rxz - 7.6) - 0.8 < res.d) {
    let pc = opPolar(q.xz, 14.0);
    res = opU(res, hitOf(sdColumn(vec3f(pc.x - 7.6, q.y - y0, pc.y), 5.2), MAT_MARBLE));
  }

  let ring = max(abs(rxz - 7.55) - 0.75, abs(q.y - y0 - 5.55) - 0.35);
  let roof = max(abs(rxz - 6.9) - 1.5, abs(q.y - y0 - 6.05) - 0.15);
  var drum = max(opOnion(rxz - 5.4, 0.3), abs(q.y - y0 - 3.6) - 3.6);
  let wp = opPolar(q.xz, 8.0);
  let win = max(sdArch(vec2f(wp.y, q.y - y0 - 4.2), 0.55, 0.8), abs(wp.x - 5.4) - 0.6);
  let door = max(max(sdArch(vec2f(q.x, q.y - y0 - 1.3), 1.0, 1.3), -q.z), y0 - q.y);
  drum = max(drum, -min(win, door));
  res = opU(res, hitOf(min(min(ring, roof), drum), MAT_MARBLE));

  let dc = q - vec3f(0.0, y0 + 7.2, 0.0);
  let dl = length(dc);
  let dxz = length(dc.xz);
  let dome = max(max(opOnion(dl - 5.6, 0.2), -dc.y), 1.0 - dxz);
  res = opU(res, hitOf(dome, MAT_GOLD));
  let rp = opPolar(dc.xz, 16.0);
  let rib = max(max(max(abs(dl - 5.75) - 0.12, abs(rp.y) - 0.1), -dc.y), 1.15 - dxz);
  let lantern = sdTorus(dc - vec3f(0.0, 5.52, 0.0), 1.12, 0.14);
  res = opU(res, hitOf(min(rib, lantern), MAT_MARBLE));
  return res;
}

// Polished plaza, stairs (a ghat) down into the river, parapets and a pair
// of twisted (Solomonic) columns mirrored across the axis.
fn sdPlaza(p: vec3f) -> Hit {
  var res = hitOf(sdBox(p - vec3f(0.0, PLAT_Y - 0.5, 21.4), vec3f(15.0, 0.5, 8.6)), MAT_FLOOR);

  // Stairs by limited repetition: step i spans z ∈ [29.5, 30 + 0.55 (i+1)] and
  // tops out at PLAT - 0.28 (i+1). The taller neighbour can be nearer than the
  // step whose cell we are in, so the two neighbours are evaluated too.
  let si = clamp(floor((p.z - 30.0) / 0.55), 0.0, 7.0);
  var ds = 1e9;
  for (var j = -1; j <= 1; j++) {
    let i = clamp(si + f32(j), 0.0, 7.0);
    let zf = 30.0 + 0.55 * (i + 1.0);
    let top = PLAT_Y - 0.28 * (i + 1.0);
    ds = min(ds, sdBox(p - vec3f(0.0, (top - 3.5) * 0.5, (29.5 + zf) * 0.5),
                       vec3f(7.0, (top + 3.5) * 0.5, (zf - 29.5) * 0.5)));
  }
  let parapet = sdRoundBox(vec3f(abs(p.x) - 7.45, p.y + 0.825, p.z - 32.4), vec3f(0.45, 2.675, 2.6), 0.05);
  res = opU(res, hitOf(min(ds, parapet), MAT_SANDSTONE));

  // Twisted columns: a rounded square shaft twisted 1.25 rad/m. The cross
  // section reaches r = 0.42 m, so k·r = 0.53 and distances are scaled by 0.85.
  let tq = vec3f(abs(p.x) - 9.5, p.y - PLAT_Y, p.z - 28.6);
  var dt = sdRoundBox(tq - vec3f(0.0, 0.3, 0.0), vec3f(0.6, 0.3, 0.6), 0.04);
  dt = min(dt, 0.85 * sdRoundBox(opTwistY(tq - vec3f(0.0, 3.3, 0.0), 1.25), vec3f(0.3, 2.7, 0.3), 0.1));
  dt = min(dt, sdRoundBox(tq - vec3f(0.0, 6.15, 0.0), vec3f(0.5, 0.15, 0.5), 0.03));
  res = opU(res, hitOf(dt, MAT_MARBLE));
  res = opU(res, hitOf(sdSphere(tq - vec3f(0.0, 6.85, 0.0), 0.55), MAT_GOLD));
  return res;
}

// Two-tier aqueduct along x, infinitely repeated (it recedes into the haze in
// both directions). Arches are elongated circles subtracted from the walls.
fn sdAqueduct(p: vec3f) -> Hit {
  let q = vec3f(p.x, p.y, p.z - AQ_Z);
  var d = max(abs(q.z) - 1.3, max(q.y - 12.0, -10.0 - q.y));
  d = max(d, -sdArch(vec2f(opRep(q.x, 9.0), q.y + 6.4), 3.4, 13.6));        // big arches, crown at 10.6 m
  d = min(d, max(abs(q.z) - 1.45, abs(q.y - 12.2) - 0.2));                  // cornice
  var u = max(abs(q.z) - 1.0, abs(q.y - 14.7) - 2.3);
  u = max(u, -sdArch(vec2f(opRep(q.x, 4.5), q.y - 13.3), 1.5, 1.3));        // upper tier, crown at 16.1 m
  d = min(d, u);
  var c = max(abs(q.z) - 1.1, abs(q.y - 17.6) - 0.6);                        // channel with a U groove
  c = max(c, -max(abs(q.z) - 0.55, abs(q.y - 18.1) - 0.5));
  return hitOf(min(d, c), MAT_SANDSTONE);
}

// Footbridge: a straight deck and railings bent into an arch (cheap bend,
// ~10% metric distortion over the span, hence the 0.9 scale), balusters by
// limited repetition in the bent space so they follow the curve.
fn sdBridge(p: vec3f) -> Hit {
  let q = opBendZY(p - BR_C, -0.016);
  var d = sdRoundBox(q, vec3f(2.0, 0.3, 13.0), 0.05);
  let mx = vec3f(abs(q.x) - 1.85, q.y, q.z);
  d = min(d, sdRoundBox(mx - vec3f(0.0, 1.0, 0.0), vec3f(0.09, 0.06, 13.0), 0.03));
  let bal = vec3f(mx.x, mx.y - 0.62, opRepLim(q.z, 0.5, -26.0, 26.0));
  d = min(d, sdRoundBox(bal, vec3f(0.05, 0.36, 0.05), 0.02));
  return hitOf(d * 0.9, MAT_SANDSTONE);
}


// -----------------------------------------------------------------------------
// 7. Character: a stylised walker (capsules, ellipsoids, a round cone) with a
// procedural gait. The skeleton only depends on time, so it is solved once
// per pixel into private variables before any marching starts.
// -----------------------------------------------------------------------------

var<private> chO: vec3f;                 // world position (ground under the pelvis)
var<private> chX: vec3f;                 // character right axis
var<private> chZ: vec3f;                 // character forward axis
var<private> chPelvis: vec3f;
var<private> chChest: vec3f;
var<private> chHead: vec3f;
var<private> chHip: array<vec3f, 2>;
var<private> chKnee: array<vec3f, 2>;
var<private> chAnkle: array<vec3f, 2>;
var<private> chShoulder: array<vec3f, 2>;
var<private> chElbow: array<vec3f, 2>;
var<private> chHand: array<vec3f, 2>;
var<private> chBreath: f32;

fn setupCharacter() {
  let dist = F.time * F.anim.x;
  let radius = 4.8;
  let th = dist / radius + 1.0;
  // Walks counter-clockwise around the plaza, facing along the path.
  chO = vec3f(PLAZA_C.x + radius * cos(th), PLAT_Y, PLAZA_C.z + radius * sin(th));
  chX = vec3f(cos(th), 0.0, sin(th));
  chZ = vec3f(-sin(th), 0.0, cos(th));
  let walk = clamp(F.anim.x / 1.1, 0.0, 1.3);
  let phase = dist / 1.3 * TAU;             // one gait cycle per 1.3 m
  var lowest = 1e9;
  for (var s = 0u; s < 2u; s++) {
    let side = select(-1.0, 1.0, s == 0u);
    let ph = phase + f32(s) * PI;
    // Legs: thigh swings about the hip; the knee flexes during the swing phase.
    let swing = 0.42 * walk * sin(ph);
    let flex = 0.08 + 0.85 * min(walk, 1.0) * pow(max(cos(ph), 0.0), 1.5);
    let hip = vec3f(side * 0.1, -0.04, 0.0);
    let knee = hip + 0.44 * vec3f(0.0, -cos(swing), sin(swing));
    let ankle = knee + 0.43 * vec3f(0.0, -cos(swing - flex), sin(swing - flex));
    chHip[s] = hip;
    chKnee[s] = knee;
    chAnkle[s] = ankle;
    lowest = min(lowest, ankle.y);
    // Arms swing against the leg on the same side, elbows slightly bent.
    let armSwing = -0.38 * walk * sin(ph);
    let shoulder = vec3f(side * 0.21, 0.5, 0.0);
    let elbow = shoulder + 0.27 * vec3f(side * 0.14, -cos(armSwing), sin(armSwing));
    let fore = armSwing + 0.45 + 0.1 * walk;
    chShoulder[s] = shoulder;
    chElbow[s] = elbow;
    chHand[s] = elbow + 0.25 * vec3f(side * 0.04, -cos(fore), sin(fore));
  }
  // Plant the lower foot on the ground; this also produces the vertical bob.
  let lift = vec3f(0.015 * walk * sin(phase), 0.095 - lowest, 0.0);
  chPelvis = lift;
  for (var s = 0u; s < 2u; s++) {
    chHip[s] += lift;
    chKnee[s] += lift;
    chAnkle[s] += lift;
    chShoulder[s] += lift;
    chElbow[s] += lift;
    chHand[s] += lift;
  }
  chChest = lift + vec3f(0.0, 0.33, 0.015);
  chHead = lift + vec3f(0.0, 0.79, 0.03 + 0.01 * sin(2.0 * phase));
  chBreath = 1.0 + 0.035 * F.anim.y * sin(F.time * 1.7);
}

fn sdCharacter(pw: vec3f) -> Hit {
  let d0 = pw - chO;
  let pl = vec3f(dot(d0, chX), d0.y, dot(d0, chZ));   // character space
  let br = chBreath;
  // Torso (breathing ellipsoid) smoothly merged with the tunic; the smooth
  // union also blends skin into cloth.
  var res = hitOf(sdEllipsoid(pl - chChest, vec3f(0.19 * br, 0.27, 0.13 * br)), MAT_SKIN);
  let tunic = sdRoundCone(pl - (chPelvis + vec3f(0.0, -0.3, 0.0)), 0.25, 0.16, 0.6);
  res = opSmoothU(res, hitOf(tunic, MAT_CLOTH), 0.07);
  // Neck, head and nose.
  var dh = sdCapsule(pl, chChest + vec3f(0.0, 0.2, 0.0), chHead, 0.055);
  dh = smin(dh, sdSphere(pl - chHead, 0.13), 0.05);
  dh = smin(dh, sdEllipsoid(pl - chHead - vec3f(0.0, -0.015, 0.12), vec3f(0.028, 0.032, 0.05)), 0.02);
  res = opSmoothU(res, hitOf(dh, MAT_SKIN), 0.05);
  // Limbs.
  for (var s = 0u; s < 2u; s++) {
    let leg = smin(sdCapsule(pl, chHip[s], chKnee[s], 0.075), sdCapsule(pl, chKnee[s], chAnkle[s], 0.06), 0.03);
    res = opSmoothU(res, hitOf(leg, MAT_SKIN), 0.06);
    let foot = sdEllipsoid(pl - chAnkle[s] - vec3f(0.0, -0.035, 0.06), vec3f(0.065, 0.055, 0.12));
    res = opSmoothU(res, hitOf(foot, MAT_CLOTH), 0.03);
    var arm = smin(sdCapsule(pl, chShoulder[s], chElbow[s], 0.055), sdCapsule(pl, chElbow[s], chHand[s], 0.045), 0.03);
    arm = smin(arm, sdSphere(pl - chHand[s], 0.052), 0.03);
    res = opSmoothU(res, hitOf(arm, MAT_SKIN), 0.05);
  }
  // Eyes and a conical straw hat (sharp unions).
  let eyes = sdSphere(vec3f(abs(pl.x - chHead.x) - 0.045, pl.y - chHead.y - 0.025, pl.z - chHead.z - 0.108), 0.022);
  res = opU(res, hitOf(eyes, MAT_EYE));
  let hat = sdCappedCone(pl - chHead - vec3f(0.0, 0.14, 0.0), 0.075, 0.28, 0.012) - 0.008;
  res = opU(res, hitOf(hat, MAT_STRAW));
  return res;
}


// -----------------------------------------------------------------------------
// 8. Abstract sculpture floating over the river
// -----------------------------------------------------------------------------

var<private> scC: vec3f;
var<private> scCS: vec4f;   // cos/sin of the spin (y) and tilt (x) angles

fn setupSculpture() {
  scC = vec3f(-24.0, 7.8 + 0.25 * sin(F.time * 0.6), 40.0);
  let a = F.time * 0.25 * F.anim.z;
  let b = 0.2;
  scCS = vec4f(cos(a), sin(a), cos(b), sin(b));
}

fn sdSculpture(pw: vec3f) -> Hit {
  var q = pw - scC;
  q = vec3f(scCS.x * q.x - scCS.y * q.z, q.y, scCS.y * q.x + scCS.x * q.z);
  q = vec3f(q.x, scCS.z * q.y - scCS.w * q.z, scCS.w * q.y + scCS.z * q.z);
  // Domain warp: w(q) = A sin(B q + phase) has |dw/dq| <= A·B·sqrt(3), so the
  // warped field is divided by (1 + A·B·sqrt(3)) to stay a distance bound.
  let A = 0.2 * F.anim.w;
  let B = 1.7;
  let qw = q + A * sin(q.yzx * B + vec3f(0.9, 1.3, 0.7) * F.time);
  let lip = 1.0 / (1.0 + A * B * 1.7321);

  // Core: a chrome sphere with gold metaballs orbiting through it.
  var res = hitOf(sdSphere(qw, 0.75), MAT_CHROME);
  for (var i = 0u; i < 5u; i++) {
    let fi = f32(i);
    let ang = F.time * (0.5 + 0.13 * fi) + fi * 1.2566;
    let c = vec3f(cos(ang), 0.8 * sin(ang * 1.3 + fi), sin(ang)) * (0.95 + 0.25 * sin(F.time * 0.7 + fi * 2.0));
    res = opSmoothU(res, hitOf(sdSphere(qw - c, 0.28), MAT_GOLD), 0.35);
  }
  // Lattice: an onion shell smoothly intersected with a gyroid sheet, then
  // opened up by a smooth subtraction so the core shows.
  let shell = opOnion(length(qw) - 1.85, 0.13);
  let k = 3.3;
  let gyr = (abs(dot(sin(qw * k), cos(qw.zxy * k))) - 0.3) / (k * 1.8);
  var lat = smax(shell, gyr, 0.04);
  lat = smax(lat, -(length(qw - vec3f(0.0, 0.0, 1.9)) - 1.0), 0.2);
  res = opU(res, hitOf(lat, MAT_GOLD));
  // Twisted ring: a rounded square swept around a circle and rotated by 1.5×
  // the sweep angle (six quarter turns: the square maps onto itself, so the
  // ring closes seamlessly). The twist rate is 1.5 rad per 3.1 m of arc and
  // the section reaches 0.23 m, hence the 0.96 scale.
  let ang = atan2(qw.z, qw.x);
  let rq = rot2(vec2f(length(qw.xz) - 3.1, qw.y), 1.5 * ang + F.time * 0.4);
  res = opU(res, hitOf((sdBox2(rq, vec2f(0.13, 0.13)) - 0.04) * 0.96, MAT_CHROME));
  res.d *= lip;
  return res;
}


// -----------------------------------------------------------------------------
// 9. Scene map. Groups are only evaluated when their bounding volume is closer
// than what has been found so far: the bound is a lower bound on the group's
// distance, so skipping it is exact, and it saves most of the work for most
// samples (the terrain's own early outs do the same inside the fBm).
// -----------------------------------------------------------------------------

fn map(p: vec3f) -> Hit {
  gEvals += 1u;
  var res = hitOf(sdTerrain(p), MAT_TERRAIN);
  let all = F.toggles.w < 0.5;

  if (all || sdBox(p - TEMPLE_BC, TEMPLE_BH) < res.d) {
    gGroups += 1u;
    res = opU(res, sdTemple(p));
  }
  if (all || sdBox(p - PLAZA_BC, PLAZA_BH) < res.d) {
    gGroups += 1u;
    res = opU(res, sdPlaza(p));
  }
  let rq = p - ROT_C;
  if (all || max(length(rq.xz) - 9.6, abs(rq.y - 5.5) - 7.2) < res.d) {
    gGroups += 1u;
    res = opU(res, sdRotunda(p));
  }
  if (all || max(abs(p.z - AQ_Z) - 1.6, max(p.y - 18.4, -10.5 - p.y)) < res.d) {
    gGroups += 1u;
    res = opU(res, sdAqueduct(p));
  }
  if (all || sdBox(p - BRIDGE_BC, BRIDGE_BH) < res.d) {
    gGroups += 1u;
    res = opU(res, sdBridge(p));
  }
  if (all || length(p - chO - vec3f(0.0, 0.9, 0.0)) - 1.2 < res.d) {
    gGroups += 1u;
    res = opU(res, sdCharacter(p));
  }
  if (all || length(p - scC) - 3.9 < res.d) {
    gGroups += 1u;
    res = opU(res, sdSculpture(p));
  }
  return res;
}


// -----------------------------------------------------------------------------
// 10. Ray marching
//
// Over-relaxed sphere tracing (Keinert, Schäfer, Korndörfer, Ganse, Stamminger:
// "Enhanced Sphere Tracing", 2014). Steps are ω·d with ω > 1. If the unbounding
// spheres of two consecutive samples do not overlap (r_prev + r < step), the
// relaxed step may have jumped over a surface: step back and continue with
// ω = 1. The hit test is cone based: stop when d < ε·t, with ε a fraction of the
// pixel's angular size, so the precision tracks the pixel footprint (a distant
// hill needs centimetres less precision than a nearby column). If the step
// budget runs out inside the scene, the sample with the smallest d/t is used
// rather than returning a hole.
// -----------------------------------------------------------------------------

struct March {
  t: f32,
  steps: u32,
  hit: bool,
  maxed: bool,
};

fn march(ro: vec3f, rd: vec3f, tminIn: f32, tmaxIn: f32, maxSteps: u32, tOffset: f32) -> March {
  var out = March(1e9, 0u, false, false);
  // Clip the ray to the slab below SCENE_TOP (the whole world's bounding volume).
  var tmin = tminIn;
  var tmax = tmaxIn;
  if (ro.y > SCENE_TOP) {
    if (rd.y >= 0.0) { return out; }
    tmin = max(tmin, (ro.y - SCENE_TOP) / -rd.y);
  } else if (rd.y > 0.0) {
    tmax = min(tmax, (SCENE_TOP - ro.y) / rd.y);
  }
  if (tmin >= tmax) { return out; }

  let eps = F.pixelAngle * F.march.z;
  var omega = F.march.y;
  var t = tmin;
  var candT = tmin;
  var candErr = 1e30;
  var prevR = 0.0;
  var stepLen = 0.0;
  var r = 0.0;
  var i = 0u;
  loop {
    if (i >= maxSteps) {
      // Out of steps while still inside the scene: take the best candidate.
      out.hit = true;
      out.maxed = true;
      t = candT;
      break;
    }
    i++;
    r = map(ro + rd * t).d;
    let radius = abs(r);
    let sorFail = omega > 1.0 && (radius + prevR) < stepLen;
    if (sorFail) {
      stepLen -= omega * stepLen;   // back to (almost) the last safe step
      omega = 1.0;
    } else {
      stepLen = r * omega;
      let err = radius / (t + tOffset);
      if (err < candErr) {
        candT = t;
        candErr = err;
      }
      if (r < eps * (t + tOffset)) {
        out.hit = true;
        break;
      }
    }
    prevR = radius;
    t += stepLen;
    if (t > tmax) { break; }
  }
  // A slightly negative distance means a small overstep (e.g. the terrain's
  // first-order Lipschitz estimate on a convex bump): pull back onto the surface.
  if (out.hit && r < 0.0 && !out.maxed) {
    t += r;
  }
  out.t = t;
  out.steps = i;
  return out;
}


// -----------------------------------------------------------------------------
// 11. Normals, soft shadows, ambient occlusion, thickness
// -----------------------------------------------------------------------------

// Tetrahedron technique: four samples instead of six central differences, and
// written as a loop so the map is inlined once, not four times.
fn calcNormal(p: vec3f, eps: f32) -> vec3f {
  var n = vec3f(0.0);
  for (var i = 0u; i < 4u; i++) {
    let e = 0.5773 * (2.0 * vec3f(f32(((i + 3u) >> 1u) & 1u), f32((i >> 1u) & 1u), f32(i & 1u)) - 1.0);
    n += e * map(p + e * eps).d;
  }
  let l = length(n);
  return select(vec3f(0.0, 1.0, 0.0), n / l, l > 1e-8);
}

// Soft shadows with Quilez's improved penumbra estimate: the closest approach
// of the shadow ray to the occluder is triangulated from two consecutive
// unbounding spheres, which removes the banding the plain min(k·h/t) shows
// behind sharp corners.
fn softShadow(ro: vec3f, rd: vec3f, k: f32, steps: u32) -> f32 {
  var tmax = 160.0;
  if (rd.y > 0.0) { tmax = min(tmax, (SCENE_TOP - ro.y) / rd.y); }
  var res = 1.0;
  var t = 0.02;
  var ph = 1e10;
  for (var i = 0u; i < steps; i++) {
    let h = map(ro + rd * t).d;
    gShadowSteps += 1u;
    if (h < 0.0005) {
      res = 0.0;
      break;
    }
    // The triangulation assumes the two spheres overlap (h < 2·ph). After a
    // clamped minimum step, or where a bounding volume or the terrain's
    // Lipschitz blend makes h jump, it does not hold (y > h would give d = 0,
    // i.e. black speckles): fall back to the classic k·h/t estimate there.
    var y = 0.0;
    var d = h;
    if (h < 1.9 * ph) {
      y = h * h / (2.0 * ph);
      d = sqrt(h * h - y * y);
    }
    res = min(res, k * d / max(t - y, 1e-4));
    ph = h;
    t += clamp(h, 0.02 + 0.003 * t, 6.0);
    if (res < 0.002 || t > tmax) { break; }
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

// Multi-sample AO along the normal: compare the distance field with the
// distance an unoccluded open half-space would give, at growing radii
// (3 cm to 0.9 m, quadratic spacing), with decreasing weights.
fn ambientOcclusion(p: vec3f, n: vec3f, samples: u32) -> f32 {
  var occ = 0.0;
  var sca = 1.0;
  var wsum = 0.0;
  let span = 1.0 / f32(max(samples - 1u, 1u));
  for (var i = 0u; i < samples; i++) {
    let x = f32(i) * span;
    let h = 0.03 + 0.87 * x * x;
    let d = map(p + n * h).d;
    occ += (h - d) * sca;
    wsum += sca;
    sca *= 0.7;
  }
  // Normalised so 2 samples (reflections) and 5 (primary) agree on average.
  return clamp(1.0 - 2.5 * F.light.z * occ / wsum, 0.0, 1.0);
}

// How thin the object is behind the surface (1 = thin, light shines through):
// one sample of the field 10 cm inside.
fn thinness(p: vec3f, n: vec3f) -> f32 {
  return clamp(1.0 + map(p - n * 0.1).d / 0.1, 0.0, 1.0);
}


// -----------------------------------------------------------------------------
// 12. Materials and BRDF
// -----------------------------------------------------------------------------

struct Mat {
  albedo: vec3f,
  rough: f32,
  metal: f32,
  sss: f32,
};

fn marbleColor(p: vec3f) -> vec3f {
  let v = 2.0 * noise3(p * 1.3) + noise3(p * 3.1);
  let vein = pow(1.0 - abs(sin(p.x * 1.1 + p.y * 1.7 + p.z * 0.8 + 4.0 * v)), 10.0);
  return mix(vec3f(0.80, 0.78, 0.73), vec3f(0.5, 0.48, 0.45), vein * 0.6) * (0.93 + 0.07 * noise3(p * 9.0));
}

fn matTerrain(p: vec3f, n: vec3f) -> Mat {
  let nz = noise3(p * 0.35);
  let nf = noise3(p * 2.7);
  let slope = n.y;
  var grass = mix(vec3f(0.12, 0.18, 0.05), vec3f(0.25, 0.27, 0.08), nz) * (0.8 + 0.4 * nf);
  grass = mix(grass, vec3f(0.32, 0.28, 0.13), 0.6 * smoothstep(0.55, 0.75, noise3(p * 0.05 + 3.0)));
  let strata = 0.88 + 0.12 * sin(p.y * 1.7 + 6.0 * nz) * noise3(p * vec3f(0.1, 0.6, 0.1));
  let stain = mix(1.0, 0.7, smoothstep(0.55, 0.8, noise3(p * vec3f(0.6, 0.05, 0.6))));
  let tone = noise3(p * 0.08 + 11.0);
  let rock = mix(mix(vec3f(0.3, 0.26, 0.22), vec3f(0.5, 0.43, 0.34), nf), vec3f(0.42, 0.33, 0.25), tone) * strata * stain;
  let sand = vec3f(0.5, 0.44, 0.33) * (0.85 + 0.3 * nf);
  var a = mix(rock, grass, smoothstep(0.62, 0.8, slope + 0.12 * (nz - 0.5)));
  let bank = 1.0 - smoothstep(0.15, 0.65, p.y + 0.3 * (nz - 0.5));
  a = mix(a, sand, bank * smoothstep(0.35, 0.7, slope));
  let wet = 1.0 - smoothstep(-0.05, 0.15, p.y);
  a *= 1.0 - 0.45 * wet;
  let snow = 0.8 * smoothstep(94.0, 104.0, p.y + 10.0 * noise3(p * 0.04)) * smoothstep(0.7, 0.9, slope);
  a = mix(a, vec3f(0.85, 0.87, 0.9), snow);
  return Mat(a, mix(mix(0.9, 0.35, wet), 0.5, snow), 0.0, 0.0);
}

fn matFloor(p: vec3f) -> Mat {
  let uv = p.xz / 1.2;
  let cell = floor(uv);
  let f = uv - cell;
  let edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y)) * 1.2;
  let grout = 1.0 - smoothstep(0.008, 0.02, edge);
  let checker = abs(cell.x + cell.y) % 2.0;
  let tint = hash2i(vec2i(cell));
  var a = mix(marbleColor(p * 1.3), vec3f(0.24, 0.29, 0.27) * (0.8 + 0.4 * noise3(p * 4.0)), checker);
  a *= 0.92 + 0.12 * tint;
  a = mix(a, vec3f(0.3, 0.28, 0.25), grout);
  return Mat(a, mix(0.06 + 0.06 * tint, 0.7, grout), 0.0, 0.0);
}

fn matSandstone(p: vec3f, n: vec3f) -> Mat {
  let u = select(p.x, p.z, abs(n.x) > abs(n.z));
  let row = floor(p.y / 0.55);
  let fy = fract(p.y / 0.55);
  let fx = fract((u + 0.6 * (row % 2.0)) / 1.2);
  let mortar = 1.0 - smoothstep(0.02, 0.05, min(min(fy, 1.0 - fy) * 0.55, min(fx, 1.0 - fx) * 1.2));
  let block = hash2i(vec2i(i32(floor((u + 0.6 * (row % 2.0)) / 1.2)), i32(row)));
  var a = vec3f(0.62, 0.5, 0.36) * (0.8 + 0.25 * block) * (0.85 + 0.3 * fbm3(p * 3.0));
  a = mix(a, vec3f(0.35, 0.31, 0.26), mortar);
  return Mat(a, 0.85, 0.0, 0.0);
}

fn materialOf(id: f32, p: vec3f, n: vec3f) -> Mat {
  let i = i32(id + 0.5);
  switch i {
    case 1: { return matTerrain(p, n); }
    case 2: {
      // Grime creeps up from the ground.
      let grime = exp(-max(p.y - PLAT_Y, 0.0) * 1.4) * (0.6 + 0.4 * noise3(p * 2.0));
      return Mat(mix(marbleColor(p), vec3f(0.42, 0.38, 0.3), grime * 0.6), 0.45, 0.0, 0.0);
    }
    case 3: { return matFloor(p); }
    case 4: { return matSandstone(p, n); }
    case 5: { return Mat(vec3f(1.0, 0.76, 0.34), 0.18 + 0.12 * noise3(p * 3.0), 1.0, 0.0); }
    case 6: { return Mat(vec3f(0.92, 0.92, 0.93), 0.04, 1.0, 0.0); }
    case 7: { return Mat(vec3f(0.78, 0.52, 0.4), 0.45, 0.0, 1.0); }
    case 8: {
      let weave = 0.9 + 0.1 * sin(p.x * 180.0) * sin(p.y * 180.0);
      return Mat(vec3f(0.52, 0.1, 0.07) * weave, 0.95, 0.0, 0.3);
    }
    case 10: { return Mat(vec3f(0.02), 0.05, 0.0, 0.0); }
    case 11: {
      // Roof: terracotta tiles on the slopes, marble on the gable faces.
      if (abs(n.z) > 0.6) { return Mat(marbleColor(p), 0.45, 0.0, 0.0); }
      let rowsT = 0.75 + 0.25 * smoothstep(0.0, 0.3, abs(fract(p.x * 2.5) - 0.5));
      return Mat(vec3f(0.5, 0.22, 0.13) * rowsT * (0.85 + 0.3 * noise3(p * 1.7)), 0.7, 0.0, 0.0);
    }
    case 12: { return Mat(vec3f(0.72, 0.58, 0.3) * (0.85 + 0.15 * sin(atan2(p.z - chO.z, p.x - chO.x) * 60.0)), 0.9, 0.0, 0.2); }
    default: { return Mat(vec3f(0.5), 0.5, 0.0, 0.0); }
  }
}

fn blendedMaterial(h: Hit, p: vec3f, n: vec3f) -> Mat {
  let a = materialOf(h.m, p, n);
  if (h.w < 0.002 || h.m == h.m2) { return a; }
  let b = materialOf(h.m2, p, n);
  return Mat(mix(a.albedo, b.albedo, h.w), mix(a.rough, b.rough, h.w), mix(a.metal, b.metal, h.w), mix(a.sss, b.sss, h.w));
}

fn dGGX(NoH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Height-correlated Smith visibility (Heitz 2014), includes the 1/(4 NoV NoL).
fn vSmith(NoV: f32, NoL: f32, a: f32) -> f32 {
  let a2 = a * a;
  let gv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let gl = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn fSchlick(F0: vec3f, VoH: f32) -> vec3f {
  return F0 + (1.0 - F0) * pow(1.0 - VoH, 5.0);
}

// Split-sum environment BRDF, analytic fit (Karis 2014, mobile).
fn envBRDF(F0: vec3f, rough: f32, NoV: f32) -> vec3f {
  let r = rough * vec4f(-1.0, -0.0275, -0.572, 0.022) + vec4f(1.0, 0.0425, 1.04, -0.04);
  let a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
  let AB = vec2f(-1.04, 1.04) * a004 + r.zw;
  return F0 * AB.x + AB.y;
}

struct Shade {
  L: vec3f,        // radiance leaving the surface towards the viewer (without traced reflection)
  refl: vec3f,     // weight of the traced mirror reflection
  ao: f32,
  sh: f32,
};

fn shadeSurface(p: vec3f, v: vec3f, n: vec3f, h: Hit, primary: bool, footprint: f32) -> Shade {
  let mt = blendedMaterial(h, p, n);
  let l = F.sunDir;
  let NoL = dot(n, l);
  let lift = p + n * (0.01 + 2.0 * footprint);

  var sh = 1.0;
  if (F.toggles.x > 0.5 && (NoL > 0.0 || mt.sss > 0.0)) {
    let steps = select(max(u32(F.light.x) / 2u, 12u), u32(F.light.x), primary);
    sh = softShadow(lift, l, F.light.y, steps);
  }
  var ao = 1.0;
  if (F.toggles.y > 0.5) {
    ao = ambientOcclusion(p, n, select(2u, 5u, primary));
  }

  let a = max(mt.rough * mt.rough, 0.002);
  let F0 = mix(vec3f(0.04), mt.albedo, mt.metal);
  let diff = mt.albedo * (1.0 - mt.metal);
  let NoV = max(dot(n, v), 1e-4);
  let NoLc = max(NoL, 0.0);
  let H = normalize(l + v);
  let NoH = max(dot(n, H), 0.0);
  let Fs = fSchlick(F0, max(dot(v, H), 0.0));
  let spec = dGGX(NoH, a) * vSmith(NoV, NoLc, a) * Fs;
  var L = (diff / PI * (1.0 - Fs) + spec) * F.sunColor * NoLc * sh;

  // Subsurface-ish: wrapped diffuse bleeding red into the terminator, plus
  // light transmitted through thin parts (ears, fingers, the hat brim) when
  // the sun is behind them.
  if (mt.sss > 0.0 && F.fog.z > 0.0) {
    let s = mt.sss * F.fog.z;
    let wrap = max((NoL + 0.5) / 1.5, 0.0);
    L += diff / PI * vec3f(1.0, 0.35, 0.2) * max(wrap - NoLc, 0.0) * F.sunColor * mix(1.0, sh, 0.5) * s;
    if (primary) {
      let back = pow(max(dot(v, -l), 0.0), 3.0) * thinness(p, n);
      L += diff * vec3f(1.0, 0.3, 0.15) * back * F.sunColor * 0.25 * mix(1.0, sh, 0.7) * s;
    }
  }

  // Ambient: sky dome (brighter for up-facing normals) and warm ground bounce.
  let sky = F.ambient * (0.6 + 0.4 * n.y);
  let bounce = vec3f(0.3, 0.26, 0.2) * F.sunColor * (0.04 * (0.5 - 0.5 * n.y));
  L += diff * (sky + bounce) * ao;

  // Specular environment. Smooth surfaces get a traced reflection (the path
  // loop continues along the mirror direction); rough ones fall back to the
  // sky seen in the reflected direction, dimmed by AO.
  let env = envBRDF(F0, mt.rough, NoV);
  let mirror = 1.0 - smoothstep(0.12, 0.4, mt.rough);
  let R = reflect(-v, n);
  let skyR = mix(skyRadiance(R, false), F.ambient, smoothstep(0.2, 0.8, mt.rough));
  L += env * (1.0 - mirror) * skyR * ao;
  return Shade(L, env * mirror * ao, ao, sh);
}


// -----------------------------------------------------------------------------
// 13. Water: an analytic plane (a plane is the one SDF that is cheaper to
// intersect than to march), with procedural waves, Fresnel reflection traced
// by the path loop, and a short refracted march for what lies underneath.
// -----------------------------------------------------------------------------

fn wave(xz: vec2f, dir: vec2f, k: f32, amp: f32) -> vec2f {
  let ph = dot(dir, xz) * k - F.time * sqrt(9.81 * k);
  return dir * (amp * k * cos(ph));
}

fn waterNormal(xz: vec2f, dist: f32) -> vec3f {
  let t = F.time;
  var g = vec2f(0.0);
  g += wave(xz, vec2f(0.98, 0.2), 0.9, 0.02);
  g += wave(xz, vec2f(0.8, -0.6), 1.7, 0.012);
  g += wave(xz, vec2f(0.29, 0.96), 2.9, 0.007);
  g += wave(xz, vec2f(-0.5, 0.87), 4.3, 0.004);
  // Ripples drifting downstream (+x).
  let r1 = noised(xz * 1.3 - vec2f(t * 0.6, 0.0));
  let r2 = noised(xz * 3.1 + vec2f(-t * 0.9, t * 0.3));
  g += 0.035 * 1.3 * r1.yz + 0.012 * 3.1 * r2.yz;
  let fade = 1.0 / (1.0 + dist * 0.015);
  return normalize(vec3f(-g.x * fade, 1.0, -g.y * fade));
}

fn waterAbsorption() -> vec3f {
  return vec3f(0.45, 0.13, 0.09) * F.fog.w;
}

// Radiance arriving at the water surface from below along the refracted ray.
fn underwater(pw: vec3f, rr: vec3f) -> vec3f {
  let sigma = waterAbsorption();
  var t = 0.03;
  var hit = false;
  for (var i = 0u; i < 28u; i++) {
    let d = map(pw + rr * t).d;
    if (d < 0.004 + 0.004 * t) {
      hit = true;
      break;
    }
    t += d;
    if (t > 12.0) { break; }
  }
  // In-scattering by the water body (a dim teal glow lit by sun and sky).
  let inscatter = vec3f(0.02, 0.07, 0.075) * (F.sunColor * 0.08 + F.ambient) * F.fog.w;
  let Tv = exp(-sigma * min(t, 12.0));
  if (!hit) { return inscatter * (1.0 - Tv); }

  let pb = pw + rr * t;
  let nb = calcNormal(pb, 0.01);
  let mt = blendedMaterial(map(pb), pb, nb);
  let depth = max(WATER_Y - pb.y, 0.0);
  let sunDown = refract(-F.sunDir, vec3f(0.0, 1.0, 0.0), 0.75);
  let Tsun = exp(-sigma * depth / max(-sunDown.y, 0.2));
  // Cheap caustics: bright where two drifting noise layers cross.
  let c1 = noised(pb.xz * 1.6 + vec2f(F.time * 0.35, 0.0)).x;
  let c2 = noised(pb.xz * 2.1 - vec2f(F.time * 0.25, -F.time * 0.2) + 5.0).x;
  let caust = pow(clamp(1.0 - abs(c1 - c2), 0.0, 1.0), 8.0) * 1.6 * exp(-depth * 0.3);
  let Lb = mt.albedo / PI * F.sunColor * max(dot(nb, F.sunDir), 0.0) * Tsun * (0.6 + caust)
         + mt.albedo * F.ambient * 0.6 * exp(-sigma * depth);
  return Lb * Tv + inscatter * (1.0 - Tv);
}

struct WaterShade {
  L: vec3f,
  refl: vec3f,
  n: vec3f,
  sh: f32,
};

fn shadeWater(p: vec3f, rd: vec3f, dist: f32, primary: bool) -> WaterShade {
  let n = waterNormal(p.xz, dist);
  let v = -rd;
  let NoV = max(dot(n, v), 1e-3);
  let fres = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
  var L = underwater(p, refract(rd, n, 0.75)) * (1.0 - fres);
  // Sun glint (GGX, smooth water).
  var sh = 1.0;
  if (F.toggles.x > 0.5) {
    sh = softShadow(p + vec3f(0.0, 0.02, 0.0), F.sunDir, F.light.y, select(max(u32(F.light.x) / 2u, 12u), u32(F.light.x), primary));
  }
  let l = F.sunDir;
  let H = normalize(l + v);
  let NoL = max(dot(n, l), 0.0);
  let a = 0.012;
  L += dGGX(max(dot(n, H), 0.0), a) * vSmith(NoV, NoL, a) * fSchlick(vec3f(0.02), max(dot(v, H), 0.0)) * F.sunColor * NoL * sh;
  return WaterShade(L, vec3f(fres), n, sh);
}


// -----------------------------------------------------------------------------
// 14. Fog and the path loop
// -----------------------------------------------------------------------------

// Exponential height fog, integrated analytically along the segment.
fn fogTransmittance(ro: vec3f, rd: vec3f, t: f32) -> f32 {
  let a = F.fog.x;
  let b = F.fog.y;
  let kt = b * rd.y * t;
  var od = a * exp(-b * max(ro.y, -2.0)) * t;
  if (abs(kt) > 1e-4) { od *= (1.0 - exp(-kt)) / kt; }
  return exp(-od);
}

// Fog takes the colour of the sky in the same direction (without the sun
// disk), so geometry fades into exactly the sky behind it: no horizon seam at
// the far clip, and distant ridges pick up the sky's sunward glow.
fn fogColor(rd: vec3f) -> vec3f {
  return skyRadiance(vec3f(rd.x, max(rd.y, 0.0), rd.z), false);
}

struct Render {
  color: vec3f,
  hit: bool,
  water: bool,
  t: f32,
  n: vec3f,
  ao: f32,
  sh: f32,
  mat: vec3f,     // (m, m2, w) of the primary hit
  steps: u32,
  maxed: bool,
};

fn render(ro0: vec3f, rd0: vec3f) -> Render {
  var out = Render(vec3f(0.0), false, false, 1e9, vec3f(0.0), 1.0, 1.0, vec3f(0.0), 0u, false);
  var ro = ro0;
  var rd = rd0;
  var thr = vec3f(1.0);
  var col = vec3f(0.0);
  var pathLen = 0.0;
  let maxD = F.march.w;
  let bounces = u32(F.light.w);
  for (var b = 0u; b <= bounces; b++) {
    let primary = b == 0u;
    var tWater = 1e9;
    if (rd.y < 0.0 && ro.y > WATER_Y) { tWater = (WATER_Y - ro.y) / rd.y; }
    let budget = maxD - pathLen;
    let tmax = min(budget, tWater);
    var steps = u32(F.march.x);
    if (!primary) { steps = max(steps / 3u, 24u); }
    let m = march(ro, rd, select(0.02, 0.05, primary), tmax, steps, pathLen);
    if (primary) {
      out.steps = m.steps;
      out.maxed = m.maxed;
    } else {
      gReflSteps += m.steps;
    }

    var tHit = 0.0;
    var L = vec3f(0.0);
    var next = vec3f(0.0);
    var occ = 1.0;
    let roIn = ro;
    let rdIn = rd;
    if (m.hit) {
      tHit = m.t;
      let p = ro + rd * tHit;
      let fp = (pathLen + tHit) * F.pixelAngle;
      let n = calcNormal(p, max(0.0015, 0.5 * fp));
      let h = map(p);
      let s = shadeSurface(p, -rd, n, h, primary, fp);
      L = s.L;
      next = s.refl;
      occ = s.ao;
      if (primary) {
        out.hit = true;
        out.t = tHit;
        out.n = n;
        out.ao = s.ao;
        out.sh = s.sh;
        out.mat = vec3f(h.m, h.m2, h.w);
      }
      ro = p + n * (0.01 + 2.0 * fp);
      rd = reflect(rd, n);
    } else if (tWater <= budget) {
      tHit = tWater;
      let p = ro + rd * tHit;
      let w = shadeWater(p, rd, pathLen + tHit, primary);
      L = w.L;
      next = w.refl;
      if (primary) {
        out.hit = true;
        out.water = true;
        out.t = tHit;
        out.n = w.n;
        out.sh = w.sh;
        out.mat = vec3f(MAT_WATER, MAT_WATER, 0.0);
      }
      ro = p + vec3f(0.0, 0.01, 0.0);
      let r = reflect(rd, w.n);
      rd = normalize(vec3f(r.x, max(r.y, 0.01), r.z));
    } else {
      // Escaped: the sky (the sun disk only on the primary ray; reflections of
      // the sun are already in the GGX term).
      col += thr * skyRadiance(rd, primary);
      break;
    }

    // Height fog, plus a fade to fog colour over the last quarter of the
    // march distance so the far clip is invisible.
    let T = fogTransmittance(roIn, rdIn, tHit) * (1.0 - smoothstep(0.75 * maxD, maxD, pathLen + tHit));
    col += thr * (L * T + fogColor(rdIn) * (1.0 - T));
    thr *= next * T;
    pathLen += tHit;
    if (b == bounces || max(thr.x, max(thr.y, thr.z)) < 0.01) {
      // Out of bounces: close the path with the sky in the mirror direction.
      col += thr * skyRadiance(rd, false) * occ;
      break;
    }
  }
  out.color = col;
  return out;
}

// -----------------------------------------------------------------------------
// 15. Debug views, tone mapping, entry points
// -----------------------------------------------------------------------------

// Turbo colour map (polynomial fit by Anton Mikhailov).
fn turbo(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  let r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  let b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}

fn matColor(id: f32) -> vec3f {
  switch i32(id + 0.5) {
    case 1: { return vec3f(0.35, 0.6, 0.25); }
    case 2: { return vec3f(0.9, 0.9, 0.85); }
    case 3: { return vec3f(0.55, 0.75, 0.95); }
    case 4: { return vec3f(0.85, 0.6, 0.35); }
    case 5: { return vec3f(1.0, 0.8, 0.2); }
    case 6: { return vec3f(0.7, 0.3, 0.9); }
    case 7: { return vec3f(1.0, 0.55, 0.55); }
    case 8: { return vec3f(0.8, 0.1, 0.15); }
    case 9: { return vec3f(0.1, 0.35, 0.8); }
    case 10: { return vec3f(0.1, 0.1, 0.1); }
    case 11: { return vec3f(0.75, 0.3, 0.15); }
    case 12: { return vec3f(0.95, 0.9, 0.5); }
    default: { return vec3f(1.0, 0.0, 1.0); }
  }
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn primaryRay(px: vec2f) -> vec3f {
  let uv = px / F.resolution;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let p = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(p.xyz / p.w - F.camPos);
}

// Distance field slice: a plane coloured by the value of map() (orange
// outside, blue inside), with contour lines every `spacing` metres and a white
// zero iso-line. Note that it shows what the tracer sees: the Lipschitz-scaled
// terrain distance and the lower bounds that bounding volumes return.
fn sliceColor(ro: vec3f, rd: vec3f, tScene: f32, base: vec3f) -> vec3f {
  let axis = i32(F.slice.x + 0.5);
  var nP = vec3f(0.0, 1.0, 0.0);
  if (axis == 1) { nP = vec3f(0.0, 0.0, 1.0); }
  if (axis == 2) { nP = vec3f(1.0, 0.0, 0.0); }
  let denom = dot(rd, nP);
  if (abs(denom) < 1e-5) { return base; }
  let tp = (F.slice.y - dot(ro, nP)) / denom;
  if (tp <= 0.0 || tp > tScene || tp > F.march.w) { return base; }
  let ps = ro + rd * tp;
  let d = map(ps).d;
  let spacing = F.slice.z;
  let w = tp * F.pixelAngle / max(abs(denom), 0.15);   // world size of a pixel on the plane
  var c = select(vec3f(0.3, 0.6, 1.0), vec3f(0.95, 0.6, 0.3), d > 0.0);
  c *= 1.0 - 0.85 * exp(-3.0 * abs(d) / spacing);
  c *= 0.8 + 0.2 * cos(TAU * d / spacing);
  let fl = abs(fract(d / spacing + 0.5) - 0.5) * spacing;
  let lineFade = 1.0 - smoothstep(0.1 * spacing, 0.4 * spacing, w);
  c = mix(c, vec3f(0.08), (1.0 - smoothstep(0.0, 1.5 * w, fl)) * lineFade * 0.8);
  c = mix(c, vec3f(1.0), 1.0 - smoothstep(0.0, 2.0 * w, abs(d)));
  return mix(base, c, 0.9);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  gEvals = 0u;
  gGroups = 0u;
  gOct = 0u;
  gShadowSteps = 0u;
  gReflSteps = 0u;
  setupCharacter();
  setupSculpture();

  let ro = F.camPos;
  let mode = i32(F.debugView + 0.5);
  // Optional in-shader supersampling (rotated 2x2 grid): still one pass.
  let ss = select(1u, 4u, F.slice.w > 1.5);
  var color = vec3f(0.0);
  var first: Render;
  var primarySteps = 0u;
  var maxed = 0u;
  for (var s = 0u; s < ss; s++) {
    var off = vec2f(0.0);
    if (ss > 1u) {
      off = vec2f(select(select(0.125, 0.375, s == 1u), select(-0.125, -0.375, s == 3u), s >= 2u),
                  select(select(0.375, -0.125, s == 1u), select(-0.375, 0.125, s == 3u), s >= 2u));
    }
    let r = render(ro, primaryRay(fragPos.xy + off));
    color += r.color;
    primarySteps += r.steps;
    maxed += select(0u, 1u, r.maxed);
    if (s == 0u) { first = r; }
  }
  color /= f32(ss);
  let rd = primaryRay(fragPos.xy);

  var out = vec3f(0.0);
  let lit = aces(color * F.exposure);
  if (mode == 0 || mode == 7) {
    out = linearToSrgb(lit);
    if (mode == 7) { out = sliceColor(ro, rd, first.t, out); }
  } else if (mode == 1) {
    out = turbo(f32(primarySteps) / (f32(ss) * 100.0));      // legend: 0 → 100 steps
  } else if (mode == 2) {
    out = turbo(f32(gEvals) / (f32(ss) * 400.0));            // legend: 0 → 400 map calls
  } else if (mode == 3) {
    out = select(vec3f(0.0), first.n * 0.5 + 0.5, first.hit);
  } else if (mode == 4) {
    out = select(vec3f(0.0), vec3f(first.ao), first.hit && !first.water);
  } else if (mode == 5) {
    out = select(vec3f(0.0), vec3f(first.sh), first.hit);
  } else if (mode == 6) {
    if (first.hit) {
      let c = mix(matColor(first.mat.x), matColor(first.mat.y), first.mat.z);
      out = c * (0.55 + 0.45 * max(dot(first.n, F.sunDir), 0.0));
    }
  }
  // Colour bar legend along the bottom edge for the heatmaps.
  if ((mode == 1 || mode == 2) && fragPos.y > F.resolution.y - 8.0) {
    out = turbo(fragPos.x / F.resolution.x);
  }

  if (F.statsOn > 0.5 && ((u32(fragPos.x) | u32(fragPos.y)) & 3u) == 0u) {
    atomicAdd(&stats[0], 1u);
    atomicAdd(&stats[1], primarySteps);
    atomicAdd(&stats[2], gEvals);
    atomicAdd(&stats[3], gGroups);
    atomicAdd(&stats[4], gOct);
    atomicAdd(&stats[5], gShadowSteps);
    atomicAdd(&stats[6], gReflSteps);
    atomicAdd(&stats[7], maxed);
  }

  // Dither to break up banding in the sky gradients (8-bit target).
  let dither = (hash2i(vec2i(fragPos.xy)) - 0.5) / 255.0;
  return vec4f(out + dither, 1.0);
}
