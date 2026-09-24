// Shared declarations for the portal study. Keep in sync with index.ts.

const PI: f32 = 3.14159265359;
const NUM_PORTALS: u32 = 8u;
const NUM_LIGHTS: u32 = 6u;
const MAX_OCCLUDERS: u32 = 12u;
// Elliptical opening half axes (portal-local metres).
const PORTAL_AXES: vec2f = vec2f(0.95, 1.55);
// The recursion fades towards the rim colour at this brightness.
const FADE_BRIGHTNESS: f32 = 0.45;
// The glow target is rgb10a2unorm and holds glow / GLOW_SCALE.
const GLOW_SCALE: f32 = 8.0;

// One per rendered view (the camera and every virtual camera), bound with a dynamic offset.
struct View {
  viewProj: mat4x4f,     // possibly oblique: only the depth row differs from the plain projection
  camWorld: mat4x4f,     // camera to world (columns: right, up, back, position)
  camPos: vec3f,
  level: f32,            // recursion level; also the stencil reference of this view
  clipPlane: vec4f,      // world-space plane the view looks out of (n·p + d ≥ 0 visible); (0,0,0,1) for the root
  fadeColour: vec3f,
  fade: f32,             // blend towards the rim colour near the recursion limit
  tanHalf: vec2f,
  viewId: f32,
  location: f32,
  resolution: vec2f,
  oblique: f32,
  pad: f32,
};

// One per draw (static location, object part, or portal), bound with a dynamic offset.
struct Draw {
  model: mat4x4f,
  clip: vec4f,           // world-space clip plane for objects crossing a portal; (0,0,0,1) keeps everything
  flags: vec4f,          // x: carve portal holes (static geometry), y: crossing duplicate, z: portal index, w: object
};

struct PortalData {
  worldToLocal: mat4x4f,
  plane: vec4f,          // n, d; n points into the portal's room
  centre: vec4f,         // xyz, w = location
  colour: vec4f,         // rim colour, w = carve depth behind the plane
  spill: vec4f,          // radiance arriving through the portal from its destination
};

struct Globals {
  portals: array<PortalData, NUM_PORTALS>,
  lights: array<vec4f, NUM_LIGHTS>,          // hangar point lights: xyz, intensity
  occluders: array<vec4f, MAX_OCCLUDERS>,    // moving objects as spheres (ambient occlusion)
  shadowMats: array<mat4x4f, 2>,             // courtyard sun, forest sun
  params: vec4f,                             // time, light spill, fog multiplier, rim distortion
  params2: vec4f,                            // rim width, glow, occluder count, debug view
};

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn hash31(p: vec3f) -> f32 {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn noise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn noise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = hash31(i);
  let n100 = hash31(i + vec3f(1.0, 0.0, 0.0));
  let n010 = hash31(i + vec3f(0.0, 1.0, 0.0));
  let n110 = hash31(i + vec3f(1.0, 1.0, 0.0));
  let n001 = hash31(i + vec3f(0.0, 0.0, 1.0));
  let n101 = hash31(i + vec3f(1.0, 0.0, 1.0));
  let n011 = hash31(i + vec3f(0.0, 1.0, 1.0));
  let n111 = hash31(i + vec3f(1.0, 1.0, 1.0));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

fn fbm2(p: vec2f) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 4; i++) {
    s += a * noise2(q);
    q = q * 2.03 + vec2f(17.1, -5.3);
    a *= 0.5;
  }
  return s;
}

fn fbm3(p: vec3f) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 3; i++) {
    s += a * noise3(q);
    q = q * 2.07 + vec3f(3.1, -7.3, 1.7);
    a *= 0.5;
  }
  return s;
}

// Distance along the ray from the camera to the plane the view looks out of.
// Fog and the recursion fade only apply beyond it: the stretch up to the
// portal is fogged by the parent view (see the restore pass), so the light
// path through several portals is fogged piecewise by each location's air.
fn entryDistance(v: View, dir: vec3f) -> f32 {
  let denom = dot(v.clipPlane.xyz, dir);
  let num = -(dot(v.clipPlane.xyz, v.camPos) + v.clipPlane.w);
  return select(0.0, max(num / denom, 0.0), abs(denom) > 1e-5 && num > 0.0);
}

// Per-location air: rgb colour, density (1/m).
fn fogParams(loc: u32) -> vec4f {
  switch loc {
    case 0u: { return vec4f(0.9, 0.66, 0.45, 0.005); }
    case 1u: { return vec4f(0.06, 0.09, 0.13, 0.022); }
    default: { return vec4f(0.42, 0.54, 0.52, 0.022); }
  }
}

// Keep in sync with SUN_DIRECTIONS in scene.ts.
fn locationSunDir(loc: u32) -> vec3f {
  if (loc == 0u) { return normalize(vec3f(-0.55, 0.6, 0.58)); }
  return normalize(vec3f(0.45, 0.78, 0.43));
}

fn skyRadiance(loc: u32, d: vec3f) -> vec3f {
  let y = d.y;
  switch loc {
    case 0u: {
      // Golden hour over the courtyard: blue overhead, a warm band at the
      // horizon that is strongest towards the sun, and the sun disc.
      let s = locationSunDir(0u);
      let mu = max(dot(d, s), 0.0);
      let up = max(y, 0.0);
      var c = mix(vec3f(0.5, 0.66, 0.92), vec3f(0.16, 0.33, 0.78), sqrt(up));
      let band = exp(-up / (0.07 + 0.12 * mu));
      c = mix(c, vec3f(1.25, 0.78, 0.46) * (0.7 + 0.6 * mu), band);
      c += vec3f(1.4, 0.75, 0.35) * pow(mu, 16.0) + vec3f(0.35, 0.2, 0.08) * pow(mu, 4.0);
      c += vec3f(60.0, 42.0, 24.0) * smoothstep(0.99955, 0.9998, mu);
      return select(c, vec3f(0.35, 0.25, 0.18), y < -0.02);
    }
    case 1u: { return vec3f(0.015, 0.02, 0.025); }
    default: {
      // Misty, overcast morning in the forest.
      let s = locationSunDir(2u);
      let mu = max(dot(d, s), 0.0);
      var c = mix(vec3f(0.62, 0.74, 0.7), vec3f(0.42, 0.55, 0.62), clamp(y, 0.0, 1.0));
      c += vec3f(0.5, 0.55, 0.45) * pow(mu, 6.0);
      return c * 1.15;
    }
  }
}

// Screen position (pixels) → world ray direction for view `v`.
fn viewRay(v: View, px: vec2f) -> vec3f {
  let ndc = vec2f(px.x / v.resolution.x * 2.0 - 1.0, 1.0 - px.y / v.resolution.y * 2.0);
  let d = vec3f(ndc * v.tanHalf, -1.0);
  return normalize((v.camWorld * vec4f(d, 0.0)).xyz);
}

fn fullscreen(vi: u32) -> vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  // z = 0: the far end of the reversed-Z range.
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Rim offsets (pixels) in rgba8unorm: 128 is exactly zero, quarter-pixel steps, ±32 px.
fn encodeRimOffset(o: vec2f) -> vec2f {
  return clamp(round(o * 4.0) + 128.0, vec2f(0.0), vec2f(255.0)) / 255.0;
}

fn infoOut(v: View, flags: f32) -> vec4f {
  return vec4f(v.level / 8.0, fract(v.viewId / 64.0), flags, 1.0);
}
