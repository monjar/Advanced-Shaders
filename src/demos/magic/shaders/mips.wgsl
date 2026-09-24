// Downsample / upsample chain, used both for the scene's mip chain (the
// frosted refraction lookup) and for bloom. Downsampling is the 13-tap
// filter from Jimenez, "Next Generation Post Processing in Call of Duty:
// Advanced Warfare" (SIGGRAPH 2014), with the Karis average on the first
// bloom level so single bright pixels (particles, crack cores) do not
// flicker. Upsampling is a 3×3 tent added to the next level up.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var base: texture_2d<f32>;

override KARIS: f32 = 0.0;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn tap(uv: vec2f) -> vec4f {
  return textureSampleLevel(src, linearSampler, uv, 0.0);
}

fn karis(c: vec4f) -> f32 {
  return 1.0 / (1.0 + dot(c.rgb, vec3f(0.2126, 0.7152, 0.0722)));
}

@fragment
fn fsDown(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let srcSize = vec2f(textureDimensions(src));
  let dstSize = max(floor(srcSize / 2.0), vec2f(1.0));
  let uv = pos.xy / dstSize;
  let t = 1.0 / srcSize;
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
  if (KARIS > 0.5) {
    // Weight each of the five 2×2 groups by 1 / (1 + luma).
    let g0 = (j + k + l + m) * 0.25;
    let g1 = (a + b + d + e) * 0.25;
    let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25;
    let g4 = (e + f + h + i) * 0.25;
    let w0 = karis(g0) * 0.5;
    let w1 = karis(g1) * 0.125;
    let w2 = karis(g2) * 0.125;
    let w3 = karis(g3) * 0.125;
    let w4 = karis(g4) * 0.125;
    let sum = g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4;
    return vec4f(sum.rgb / (w0 + w1 + w2 + w3 + w4), e.a);
  }
  let sum = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
  return sum;
}

@fragment
fn fsUp(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let dstSize = vec2f(textureDimensions(base));
  let uv = pos.xy / dstSize;
  let t = 1.0 / vec2f(textureDimensions(src));
  var s = tap(uv) * 4.0;
  s += (tap(uv + vec2f(-t.x, 0.0)) + tap(uv + vec2f(t.x, 0.0)) + tap(uv + vec2f(0.0, -t.y)) + tap(uv + vec2f(0.0, t.y))) * 2.0;
  s += tap(uv + vec2f(-t.x, -t.y)) + tap(uv + vec2f(t.x, -t.y)) + tap(uv + vec2f(-t.x, t.y)) + tap(uv + vec2f(t.x, t.y));
  return textureLoad(base, vec2i(pos.xy), 0) + s / 16.0;
}
