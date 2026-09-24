// Bakes the mountain patch around the starting site (once): pass 1 writes
// heights (km) on a 1024^2 grid over the gnomonic patch, pass 2 adds central
// difference gradients for the normals.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var heightRaw: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var heightIn: texture_2d<f32>;
@group(0) @binding(3) var heightOut: texture_storage_2d<rgba16float, write>;

const ROT: mat2x2f = mat2x2f(0.8, 0.6, -0.6, 0.8);

fn fbm2(p: vec2f, octaves: i32) -> f32 {
  var s = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < octaves; i++) {
    s += a * noised2(q).x;
    q = ROT * q * 2.03;
    a *= 0.5;
  }
  return s;
}

// Patch coordinates in km -> height in km.
fn heightKm(p0: vec2f) -> f32 {
  // Domain warp (Quilez) bends the ranges off the noise lattice.
  let w = vec2f(fbm2(p0 * 0.008 + 3.1, 3), fbm2(p0 * 0.008 + 8.7, 3)) * 40.0;
  let p = p0 + w;
  // Mountain ranges where a low-frequency field is high.
  let range = smoothstep(-0.08, 0.35, fbm2(p * 0.0055 + 1.3, 3));
  // Ridged multifractal (Musgrave): sharp crests; each octave is weighted
  // by the previous one so detail concentrates on the ridges.
  var ridge = 0.0;
  var amp = 0.55;
  var weight = 1.0;
  var q = p * 0.02;
  for (var i = 0; i < 7; i++) {
    var n = 1.0 - abs(noised2(q).x * 1.3);
    n = n * n;
    ridge += n * amp * weight;
    weight = clamp(n * 1.3, 0.0, 1.0);
    amp *= 0.48;
    q = ROT * q * 2.05;
  }
  // Eroded foothills: fBm whose octaves are damped where the accumulated
  // slope is steep (Quilez), which reads as erosion channels.
  var d = vec2f(0.0);
  var e = 0.0;
  var b = 0.5;
  var r = p * 0.035;
  for (var i = 0; i < 6; i++) {
    let n = noised2(r);
    d += n.yz;
    e += b * n.x / (1.0 + dot(d, d));
    b *= 0.5;
    r = ROT * r * 2.0;
  }
  var h = range * 8.5 * pow(max(ridge - 0.22, 0.0), 1.6) + (0.3 + 0.6 * range) * (e + 0.35);
  let edge = smoothstep(F.patchHalfSize, F.patchHalfSize * 0.75, max(abs(p0.x), abs(p0.y)));
  return max(h, 0.0) * edge;
}

@compute @workgroup_size(8, 8)
fn heights(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(heightRaw);
  if (any(id.xy >= size)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let p = vec2f(uv.x - 0.5, 0.5 - uv.y) * 2.0 * F.patchHalfSize;
  textureStore(heightRaw, id.xy, vec4f(heightKm(p), 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn gradients(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(heightIn);
  if (any(id.xy >= size)) { return; }
  let c = vec2i(id.xy);
  let m = vec2i(size) - 1;
  let hx0 = textureLoad(heightIn, clamp(c - vec2i(1, 0), vec2i(0), m), 0).r;
  let hx1 = textureLoad(heightIn, clamp(c + vec2i(1, 0), vec2i(0), m), 0).r;
  let hy0 = textureLoad(heightIn, clamp(c + vec2i(0, 1), vec2i(0), m), 0).r; // row +1 is further south
  let hy1 = textureLoad(heightIn, clamp(c - vec2i(0, 1), vec2i(0), m), 0).r;
  let texel = 2.0 * F.patchHalfSize / f32(size.x);
  let h = textureLoad(heightIn, c, 0).r;
  textureStore(heightOut, id.xy, vec4f(h, (hx1 - hx0) / (2.0 * texel), (hy1 - hy0) / (2.0 * texel), 1.0));
}
