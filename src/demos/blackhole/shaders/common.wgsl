// Shared constants, uniforms, hashing and noise. Keep the layout in sync with
// index.ts (writeUniforms).
//
// Units: G = c = 1 and lengths in Schwarzschild radii, r_s = 2M = 1 (so
// M = 0.5). Times are in r_s/c.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
// u = 1/r on the photon sphere (r = 1.5 r_s).
const U_PHOTON: f32 = 0.666666667;

struct Frame {
  camPos: vec3f,      // observer position (the hole is at the origin)
  rObs: f32,          // |camPos|
  camRight: vec3f,    // right axis scaled by tan(fov/2)·aspect
  time: f32,
  camUp: vec3f,       // up axis scaled by tan(fov/2)
  frameIndex: f32,
  camFwd: vec3f,
  split: f32,         // split-screen divider as a fraction of the width
  diskN: vec3f,       // disk normal (rotation axis, disk turns counter-clockwise about it)
  diskIn: f32,        // inner radius (ISCO = 3 r_s by default)
  diskA: vec3f,       // in-plane reference axis for the disk azimuth
  diskOut: f32,
  obsVel: vec3f,      // observer velocity relative to the static frame (units of c)
  obsGamma: f32,
  beaconDir: vec3f,   // direction of the bright alignment star on the celestial sphere
  beaconFlux: f32,
  galaxyN: vec3f,     // galactic plane normal
  pixelAngle: f32,    // angular size of one pixel at the image centre (rad)
  galaxyC: vec3f,     // direction of the galactic centre
  obsRedshift: f32,   // 1: the observer sits at r_obs (gravitational blueshift), 0: as seen from infinity
  resolution: vec2f,
  jitter: vec2f,      // sub-pixel offset for accumulation
  integ: vec4f,       // integrator (0 plane RK4, 1 plane Dormand–Prince, 2 Cartesian RK4), dφ, tolerance, max steps
  cart: vec4f,        // Cartesian step (fraction of r), escape radius, view window shift xy (tan units, off-axis zoom)
  art: vec4f,         // artistic pull strength, step (fraction of r), max steps, escape radius
  disk: vec4f,        // peak temperature (K), optical depth, turbulence, enabled
  diskFx: vec4f,      // Doppler on, gravitational redshift on, intensity mode (0 spectral, 1 g⁴, 2 g³), brightness
  anim: vec4f,        // disk time (r_s/c), crossfade period, light travel time on, 1 / blackbody luminance at the peak temperature
  env: vec4f,         // star brightness, galaxy brightness, footprint mode (0 none, 1 analytic, 2 finite differences), filter width (px)
  sky: vec4f,         // star angular size (rad), beacon size (rad), stars on, galaxy on
  view: vec4f,        // render mode (0 artistic, 1 physical, 2 split), debug view, stats on, unused
  post: vec4f,        // exposure, bloom strength, unused, unused
};

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> bb: array<vec4f, 256>;

fn finite(x: f32) -> bool {
  // Bit test: the compiler may assume NaN/Inf never occur and fold x != x.
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn finite3(v: vec3f) -> bool {
  return finite(v.x) && finite(v.y) && finite(v.z);
}

// PCG3D (Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT 2020).
fn pcg3d(p: vec3u) -> vec3u {
  var v = p * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn hash33(p: vec3u) -> vec3f {
  return vec3f(pcg3d(p)) * (1.0 / 4294967296.0);
}

fn hashCell(c: vec3f, seed: u32) -> vec3f {
  return hash33(bitcast<vec3u>(vec3i(c)) + vec3u(seed, seed * 7u, seed * 13u));
}

// 3D gradient noise with a quintic fade, roughly in [-1, 1].
fn gnoise(p: vec3f, seed: u32) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var n: array<f32, 8>;
  for (var k = 0u; k < 8u; k++) {
    let o = vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
    let g = hashCell(i + o, seed) * 2.0 - 1.0;
    n[k] = dot(g, f - o);
  }
  let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
  let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
  return mix(x0, x1, u.z) * 1.6;
}

// Prefiltered fBm. `footprint` is the pixel footprint in units of p. An octave
// whose features are smaller than about two footprints is faded out, like
// picking a coarser mip level: its mean (zero) is what a pixel would average.
fn fbmFiltered(p: vec3f, octaves: u32, footprint: f32, seed: u32) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var freq = 1.0;
  var q = p;
  for (var i = 0u; i < octaves; i++) {
    let w = 1.0 - smoothstep(0.3, 0.6, footprint * freq);
    if (w <= 0.0) { break; }
    sum += w * amp * gnoise(q, seed + i);
    amp *= 0.5;
    freq *= 2.03;
    q = q * 2.03 + vec3f(1.7, 9.2, 3.1);
  }
  return sum;
}

// Blackbody table (see blackbody.ts): rgb with luminance 1, a = log2 luminance
// relative to 6500 K, log-spaced in temperature.
const BB_T_MIN: f32 = 800.0;
const BB_T_MAX: f32 = 60000.0;

fn blackbody(T: f32) -> vec3f {
  if (T <= BB_T_MIN) { return vec3f(0.0); }
  let x = log2(T / BB_T_MIN) / log2(BB_T_MAX / BB_T_MIN) * 255.0;
  if (x >= 255.0) {
    // Beyond the table the visible band is in the Rayleigh–Jeans tail:
    // the colour stops changing and radiance grows linearly with T.
    let e = bb[255];
    return e.rgb * exp2(e.a) * (T / BB_T_MAX);
  }
  let i = u32(x);
  let t = x - f32(i);
  let e = mix(bb[i], bb[i + 1u], t);
  return e.rgb * exp2(e.a);
}

// Turbo colour map, polynomial fit by Mikhailov (2019).
fn turbo(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  let r = vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  let g = vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  let b = vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  let r2 = vec2f(-152.94239396, 59.28637943);
  let g2 = vec2f(4.27729857, 2.82956604);
  let b2 = vec2f(-89.90310912, 27.34824973);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return vec3f(dot(v4, r) + dot(v2, r2), dot(v4, g) + dot(v2, g2), dot(v4, b) + dot(v2, b2));
}
