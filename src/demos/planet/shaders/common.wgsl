// Frame uniforms and small helpers of the planet study. Keep Frame in sync
// with writeFrame() in index.ts.

const PI: f32 = 3.14159265359;
const PLANET_R: f32 = 6360000.0;
const GRID_N: u32 = 32u;
const GRID: u32 = 33u;
const VERTS_PER_CHUNK: u32 = 1221u; // 33 x 33 grid + 4 x 33 skirt

struct Frame {
  viewProj: mat4x4f,     // camera-relative (the view has no translation)
  invViewProj: mat4x4f,
  camPos: vec3f,         // m, planet centred (float: only for directions)
  time: f32,
  sunDir: vec3f,
  exposure: f32,
  resolution: vec2f,
  pixelAngle: f32,       // radians per pixel
  debugMode: f32,
  cloud: vec4f,          // coverage, altitude (m), max optical depth, shadow strength
  cloud2: vec4f,         // swirl, enabled, time-lapse, unused
  city: vec4f,           // intensity, density, unused, unused
  ocean: vec4f,          // wave slope, base roughness, unused, unused
  misc: vec4f,           // camera altitude (m, double precise), stars, snow-line offset (m), moisture offset
  shape: vec4f,          // TerrainShape: sea bias, mountain height, detail height
  shadowMat0: mat4x4f,   // camera-relative -> sun cascade clip space
  shadowMat1: mat4x4f,
  shadow: vec4f,         // texel size of cascade 0 and 1 (m), enabled, unused
};

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}
