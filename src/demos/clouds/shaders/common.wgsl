// Shared constants and uniform layouts. Keep in sync with index.ts.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

struct Frame {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec3f,
  time: f32,
  sunDir: vec3f,
  dt: f32,
  sunColor: vec3f,       // sun irradiance at the ground
  exposure: f32,
  ambient: vec3f,        // cosine-weighted average sky radiance
  debugMode: f32,
  resolution: vec2f,     // full canvas resolution
  near: f32,
  sunIntensity: f32,     // sun irradiance at the top of the atmosphere
  sky: vec4f,            // turbidity, rayleigh multiplier, mie g, multiple-scattering boost
};

struct Clouds {
  prevViewProj: mat4x4f,
  layer: vec4f,      // bottom altitude, top altitude, planet radius, max march distance (m)
  shape: vec4f,      // coverage, cloud type bias, density multiplier, base noise frequency (1/m)
  detail: vec4f,     // detail noise frequency (1/m), erosion strength, curl frequency (1/m), curl amplitude (m)
  wind: vec4f,       // base noise offset xz, detail noise offset xz (m)
  weather: vec4f,    // weather map size (m), weather offset xz (m), height skew (m)
  light: vec4f,      // extinction (1/m), scattering albedo, powder strength, ambient strength
  phase: vec4f,      // forward g, backward g, forward weight, light march distance (m)
  ms: vec4f,         // octaves, extinction falloff a, contribution falloff b, eccentricity falloff c
  march: vec4f,      // primary steps, light steps, jitter amount, horizon step multiplier
  temporal: vec4f,   // frame index, update pattern (1 or 4), history blend, enabled
  shadow: vec4f,     // map centre x, centre z, size (m), strength
  view: vec4f,       // camera altitude (m), fog density (1/m), unused, unused
  windDir: vec4f,    // wind direction xz
  extra: vec4f,      // sun-march absorption multiplier, unused x3
};

fn remap(v: f32, lo: f32, hi: f32, newLo: f32, newHi: f32) -> f32 {
  return newLo + (v - lo) * (newHi - newLo) / (hi - lo);
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2f, octaves: i32) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < octaves; i++) {
    sum += amp * valueNoise(q);
    q = mat2x2f(1.6, 1.2, -1.2, 1.6) * q;
    amp *= 0.5;
  }
  return sum;
}

// Interleaved gradient noise (Jimenez 2014): cheap, well-distributed per-pixel jitter.
fn interleavedGradientNoise(p: vec2f) -> f32 {
  return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

// Order in which the four pixels of each 2x2 block are refreshed.
fn bayerOffset(frame: u32) -> vec2u {
  let i = frame % 4u;
  return select(select(select(vec2u(0u, 1u), vec2u(1u, 0u), i == 2u), vec2u(1u, 1u), i == 1u), vec2u(0u, 0u), i == 0u);
}
