// Shared uniforms and the noise used by every pass. Keep in sync with index.ts.

const PI: f32 = 3.14159265359;
const SKY_ID: f32 = 0.0;
const FAR: f32 = 1.0e6;
// Features per tile of the 3D noise texture (it is generated with this period).
const NOISE_CELLS: f32 = 16.0;

struct Frame {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  prevViewProj: mat4x4f,
  lightViewProj: mat4x4f,
  camPos: vec3f,
  time: f32,
  lightDir: vec3f,
  pixelAngle: f32,      // world size of one pixel at distance 1
  resolution: vec2f,
  pxScale: f32,         // resolution height / 1080: keeps pixel-sized effects resolution independent
  frameIndex: f32,
  paper0: vec4f,        // dynamic canvas octave 0: scale, offset xy, blend weight to octave 1
  paper1: vec4f,        // octave 1: scale, offset xy, unused
};

struct Paint {
  bands: vec4f,       // light threshold, shadow threshold, band softness, band jitter
  density: vec4f,     // light glaze, mid glaze, shadow glaze, pigment turbulence
  shadowTint: vec4f,  // rgb multiplier for shadow pigment, amount
  strokes: vec4f,     // streak strength, streak size (px), stroke smear length (px), direction jitter
  bleed: vec4f,       // bleed radius (px), wet-area coverage, cast-shadow strength, aerial perspective
  edges: vec4f,       // edge darkening, ink strength, ink width (px), ink breakup
  paper: vec4f,       // granulation, dry brush, wobble (px), paper relief
  paperColor: vec4f,  // rgb (sRGB), paper grain size (px)
  inkColor: vec4f,    // rgb (sRGB), crease sensitivity
  mode: vec4f,        // noise space (0 fractal world, 1 fixed world, 2 screen), paper (0 dynamic canvas, 1 screen), debug view, unused
};

fn srgbToLinear(c: vec3f) -> vec3f {
  return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

fn hash31(n: f32) -> vec3f {
  return fract(sin(vec3f(n, n + 1.0, n + 2.0) * vec3f(43758.5453, 22578.1459, 19642.3490)));
}

// Screen-space position (pixels) to world ray.
fn viewRay(px: vec2f) -> vec3f {
  let uv = px / F.resolution;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let p = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(p.xyz / p.w - F.camPos);
}

fn project(world: vec3f) -> vec2f {
  let c = F.viewProj * vec4f(world, 1.0);
  let ndc = c.xy / c.w;
  return vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * F.resolution;
}
