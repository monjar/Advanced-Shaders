// Shared constants, uniform layouts and small helpers. Keep in sync with index.ts.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// A camera. The main view and the six environment-cube faces each have one.
struct View {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  camPos: vec4f,          // xyz; w = 0 for the main camera, the face size in texels for cube faces
};

struct Frame {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  model: mat4x4f,         // object → world (rotation, uniform scale, translation)
  invModel: mat4x4f,
  lightViewProj: mat4x4f, // moonlight shadow map
  camPos: vec3f,
  time: f32,
  resolution: vec2f,
  frameIndex: f32,
  dt: f32,
  moonDir: vec3f,
  exposure: f32,
  moonColor: vec3f,
  bloom: f32,
  objCenter: vec3f,
  objScale: f32,
  brazierA: vec4f,        // position, flickering intensity
  brazierB: vec4f,
  debug: f32,
  fieldBox: f32,          // half extent of the baked field volume (object units)
  shapeBox: f32,          // half extent of the baked SDF volume
  fog: f32,
  fireColor: vec3f,
  fieldTime: f32,         // time as seen by the field (scaled by the time-scale slider)
  extra: vec4f,           // focal length in pixels, unused ×3
};

// How the material reads the field. Every slot is documented in params.ts.
struct Material {
  surface: vec4f,       // kind (0 dielectric, 1 metal), ior, dispersion, roughness
  surface2: vec4f,      // frost, anisotropy, runes, groove depth
  tint: vec4f,          // tint / metal F0, max march steps
  absorption: vec4f,    // Beer–Lambert rgb per object unit, energy extinction
  energyLow: vec4f,     // rgb, intensity
  energyHigh: vec4f,    // rgb, filament sharpness
  flow: vec4f,          // flow speed, energy scale, swirl, GRIN strength
  charge: vec4f,        // charge scale, charge speed, surge, crack cell scale
  crack: vec4f,         // width, open threshold, intensity, depth into volume
  crackColor: vec4f,
  crackHot: vec4f,
  rim: vec4f,           // rgb, power
  scatter: vec4f,       // rgb, amount
  particles: vec4f,     // spawn rate, size, lifetime, buoyancy
  particles2: vec4f,    // curl follow, eject speed, intensity, streak
  particleColor: vec4f, // rgb, particle count
  particleHot: vec4f,
  haze: vec4f,          // heat haze, lens, radius, rim intensity
  light: vec4f,         // object light, moon, energy skin depth, unused
};

// Written each frame by probe.wgsl: what the field emits as a whole.
struct Probe {
  light: vec4f,         // rgb radiant intensity of the object seen as a point light, mean charge
  stats: vec4f,         // mean energy, peak charge, crack openness, unused
};

fn pcg3d(v0: vec3u) -> vec3u {
  var v = v0 * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}

fn hash33(p: vec3i) -> vec3f {
  return vec3f(pcg3d(bitcast<vec3u>(p))) * (1.0 / 4294967296.0);
}

fn hash13(p: vec3f) -> f32 {
  return hash33(vec3i(floor(p))).x;
}

fn hashU(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) * (1.0 / 4294967296.0);
}

// Interleaved gradient noise (Jimenez 2014), rotated per frame.
fn ign(px: vec2f, frame: f32) -> f32 {
  let q = px + 5.588238 * (frame % 64.0);
  return fract(52.9829189 * fract(dot(q, vec2f(0.06711056, 0.00583715))));
}

fn fullscreenPosition(vi: u32) -> vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Object ↔ world. The model matrix is a rotation times a uniform scale.
fn toWorld(p: vec3f) -> vec3f { return (F.model * vec4f(p, 1.0)).xyz; }
fn toWorldDir(d: vec3f) -> vec3f { return normalize((F.model * vec4f(d, 0.0)).xyz); }
fn toObject(p: vec3f) -> vec3f { return (F.invModel * vec4f(p, 1.0)).xyz; }
fn toObjectDir(d: vec3f) -> vec3f { return normalize((F.invModel * vec4f(d, 0.0)).xyz); }

fn projectUv(world: vec3f) -> vec3f {
  let c = F.viewProj * vec4f(world, 1.0);
  let ndc = c.xyz / max(c.w, 1e-5);
  return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, c.w);
}
