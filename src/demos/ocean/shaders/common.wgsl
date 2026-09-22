// Shared constants and uniform layouts. Keep in sync with uniforms.ts.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const CASCADES: u32 = 3u;
const FFT_SIZE: f32 = 256.0;
const WATER_IOR: f32 = 1.333;

struct Frame {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec3f,
  time: f32,
  sunDir: vec3f,
  dt: f32,
  sunColor: vec3f,       // sun irradiance at sea level (already attenuated by the atmosphere)
  exposure: f32,
  ambient: vec3f,        // cosine-weighted average sky radiance
  debugMode: f32,
  resolution: vec2f,
  near: f32,
  sunIntensity: f32,     // sun irradiance at the top of the atmosphere
  sky: vec4f,            // turbidity, rayleigh multiplier, mie g, multiple-scattering boost
};

struct Ocean {
  lengths: vec4f,        // patch length of cascades 0..2, choppiness
  sim: vec4f,            // base grid spacing, displacement scale, normal strength, shallow attenuation depth
  absorption: vec3f,     // Beer-Lambert absorption per metre
  refractStrength: f32,
  scatterColor: vec3f,   // albedo of the water body (in-scattering)
  sssStrength: f32,
  foamColor: vec3f,
  foamIntensity: f32,
  shore: vec4f,          // amplitude, wavelength, period, range
  shading: vec4f,        // roughness, glitter, shoreline foam, intersection foam width
  ripple: vec4f,         // origin x, origin z, domain size, height scale
  terrain: vec4f,        // terrain half extent, caustic intensity, caustic depth, fog density
  foam: vec4f,           // per-cascade foam weights, foam noise scale
};

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn hash22(p: vec2f) -> vec2f {
  let n = hash21(p);
  return vec2f(n, hash21(p + n + 17.17));
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

fn fbm(p: vec2f, octaves: i32) -> f32 {
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

fn fullscreenPosition(vi: u32) -> vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
