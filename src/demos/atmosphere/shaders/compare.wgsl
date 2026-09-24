// LUT path vs brute-force reference, per reference pixel. Accumulates
// (fixed point, atomics) for sky and ground pixels separately:
//   [0/1]  sum of relative luminance error |Y_lut - Y_ref| / (Y_ref + eps), x 1e4
//   [2/3]  pixel count
//   [4/5]  sum of mean absolute 8-bit difference after tone mapping, x 100
//   [6/7]  max relative luminance error, x 1e4
//   [8/9]  pixels with relative error > 1 %
//   [10/11] pixels with relative error > 5 %
// eps is 1 % of display white (after exposure), so near-black pixels such
// as the night sky do not dominate the relative error.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var lutTex: texture_2d<f32>;
@group(0) @binding(2) var refTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>, 12>;

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn display(c: vec3f) -> vec3f {
  return linearToSrgb(aces(c * F.exposure)) * 255.0;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(refTex);
  if (any(id.xy >= size)) { return; }
  let s = u32(F.refScale);
  let refc = textureLoad(refTex, id.xy, 0);
  let lp = id.xy * s + s / 2u;
  if (any(lp >= textureDimensions(lutTex))) { return; }
  let lut = textureLoad(lutTex, lp, 0);
  let eps = 0.01 / F.exposure;
  let yr = luminance(refc.rgb);
  let rel = abs(luminance(lut.rgb) - yr) / (yr + eps);
  let d8 = abs(display(lut.rgb) - display(refc.rgb));
  let k = select(0u, 1u, refc.a >= 0.0);
  atomicAdd(&stats[k], u32(min(rel, 4.0) * 1e4));
  atomicAdd(&stats[2u + k], 1u);
  atomicAdd(&stats[4u + k], u32((d8.r + d8.g + d8.b) / 3.0 * 100.0));
  atomicMax(&stats[6u + k], u32(min(rel, 40.0) * 1e4));
  if (rel > 0.01) { atomicAdd(&stats[8u + k], 1u); }
  if (rel > 0.05) { atomicAdd(&stats[10u + k], 1u); }
}
