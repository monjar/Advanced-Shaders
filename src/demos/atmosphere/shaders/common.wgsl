// Uniforms and noise shared by every pass of the atmosphere study.
// Keep Frame in sync with writeUniforms() in index.ts.

const PI: f32 = 3.14159265359;

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

struct Frame {
  resolution: vec2f,
  time: f32,
  exposure: f32,
  debugMode: f32,
  refScale: f32,          // reference buffer is resolution / refScale
  refViewSteps: f32,
  refSunSteps: f32,
  refMultiScattering: f32,
  terrainScale: f32,      // mountain height multiplier (0 = smooth sphere)
  snowLine: f32,          // km
  stars: f32,
  siteUp: vec3f,          // unit vector of the terrain patch centre
  patchHalfSize: f32,     // km
  siteEast: vec3f,
  shadows: f32,
  siteNorth: vec3f,
  splitX: f32,            // split view divider (0..1)
  planet: f32,            // 0 Earth (oceans), 1 Mars
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

// ---- Hashing and noise ---------------------------------------------------

// PCG-style 2D/3D integer hashes (Jarzynski & Olano, "Hash Functions for GPU
// Rendering", JCGT 2020).
fn pcg2d(vin: vec2u) -> vec2u {
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v = v ^ (v >> vec2u(16u));
  v.x += v.y * 1664525u;
  v.y += v.x * 1664525u;
  v = v ^ (v >> vec2u(16u));
  return v;
}

fn pcg3d(vin: vec3u) -> vec3u {
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn unorm24(h: u32) -> f32 {
  return f32(h >> 8u) * (1.0 / 16777215.0);
}

fn grad2(i: vec2f) -> vec2f {
  let h = pcg2d(bitcast<vec2u>(vec2i(i)));
  return vec2f(unorm24(h.x), unorm24(h.y)) * 2.0 - 1.0;
}

// 2D gradient noise with analytic derivatives (Quilez): returns (n, dn/dx, dn/dy).
fn noised2(p: vec2f) -> vec3f {
  let i = floor(p);
  let f = p - i;
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  let ga = grad2(i);
  let gb = grad2(i + vec2f(1.0, 0.0));
  let gc = grad2(i + vec2f(0.0, 1.0));
  let gd = grad2(i + vec2f(1.0, 1.0));
  let va = dot(ga, f);
  let vb = dot(gb, f - vec2f(1.0, 0.0));
  let vc = dot(gc, f - vec2f(0.0, 1.0));
  let vd = dot(gd, f - vec2f(1.0, 1.0));
  let k = va - vb - vc + vd;
  let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * k;
  let deriv = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
    du * (u.yx * k + vec2f(vb - va, vc - va));
  return vec3f(value, deriv);
}

// Cheap 3D value noise for large-scale albedo variation on the sphere.
fn valueNoise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let b = bitcast<vec3u>(vec3i(i));
  var n: array<f32, 8>;
  for (var k = 0u; k < 8u; k++) {
    let o = vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
    n[k] = unorm24(pcg3d(b + o).x);
  }
  let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
  let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
  return mix(x0, x1, u.z);
}

fn fbm3(p: vec3f, octaves: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < octaves; i++) {
    s += a * valueNoise3(q);
    q = q * 2.03 + vec3f(1.7, 9.2, 4.1);
    a *= 0.5;
  }
  return s;
}

