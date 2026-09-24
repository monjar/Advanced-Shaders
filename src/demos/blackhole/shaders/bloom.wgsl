// Bloom (compute).
// Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare"
// (SIGGRAPH 2014): a 13-tap downsample chain (Karis average on the first
// level against fireflies from single hot pixels) and a 3×3 tent upsample
// that sums the levels back up.

struct Level {
  texel: vec2f,   // 1 / source size
  karis: f32,
  unused: f32,
};

@group(1) @binding(0) var bloomSrc: texture_2d<f32>;
@group(1) @binding(1) var bloomSampler: sampler;
@group(1) @binding(2) var bloomDst: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var<uniform> level: Level;
@group(1) @binding(4) var bloomAdd: texture_2d<f32>;

fn tap(uv: vec2f) -> vec3f {
  return textureSampleLevel(bloomSrc, bloomSampler, uv, 0.0).rgb;
}

fn karisWeight(c: vec3f) -> f32 {
  return 1.0 / (1.0 + dot(c, vec3f(0.2126, 0.7152, 0.0722)));
}

@compute @workgroup_size(8, 8)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(bloomDst);
  if (any(id.xy >= dims)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
  let t = level.texel;
  let a = tap(uv + t * vec2f(-2.0, -2.0));
  let b = tap(uv + t * vec2f(0.0, -2.0));
  let c = tap(uv + t * vec2f(2.0, -2.0));
  let d = tap(uv + t * vec2f(-2.0, 0.0));
  let e = tap(uv);
  let f = tap(uv + t * vec2f(2.0, 0.0));
  let g = tap(uv + t * vec2f(-2.0, 2.0));
  let h = tap(uv + t * vec2f(0.0, 2.0));
  let i = tap(uv + t * vec2f(2.0, 2.0));
  let j = tap(uv + t * vec2f(-1.0, -1.0));
  let k = tap(uv + t * vec2f(1.0, -1.0));
  let l = tap(uv + t * vec2f(-1.0, 1.0));
  let m = tap(uv + t * vec2f(1.0, 1.0));
  var o: vec3f;
  if (level.karis > 0.5) {
    // Weight each of the five 2×2 boxes by 1/(1 + luma) before averaging.
    let g0 = (j + k + l + m) * 0.25;
    let g1 = (a + b + d + e) * 0.25;
    let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25;
    let g4 = (e + f + h + i) * 0.25;
    let w0 = karisWeight(g0) * 0.5;
    let w1 = karisWeight(g1) * 0.125;
    let w2 = karisWeight(g2) * 0.125;
    let w3 = karisWeight(g3) * 0.125;
    let w4 = karisWeight(g4) * 0.125;
    o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
  } else {
    o = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  }
  textureStore(bloomDst, id.xy, vec4f(o, 1.0));
}

@compute @workgroup_size(8, 8)
fn upsample(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(bloomDst);
  if (any(id.xy >= dims)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
  let t = level.texel;
  var s = tap(uv) * 4.0;
  s += (tap(uv + vec2f(-t.x, 0.0)) + tap(uv + vec2f(t.x, 0.0)) + tap(uv + vec2f(0.0, -t.y)) + tap(uv + vec2f(0.0, t.y))) * 2.0;
  s += tap(uv - t) + tap(uv + t) + tap(uv + vec2f(t.x, -t.y)) + tap(uv + vec2f(-t.x, t.y));
  let here = textureLoad(bloomAdd, id.xy, 0).rgb;
  textureStore(bloomDst, id.xy, vec4f(here + s / 16.0, 1.0));
}
