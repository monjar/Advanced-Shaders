// 256-point inverse FFT along rows (HORIZONTAL) or columns, entirely in
// workgroup shared memory: one workgroup of 256 invocations per line, one
// invocation per output sample. Each texel holds two complex numbers, so both
// are transformed together. Radix-2 decimation in time after a bit-reversed load.

override HORIZONTAL: bool = true;

const N: u32 = 256u;
const LOG_N: u32 = 8u;

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba32float, write>;

var<workgroup> bufA: array<vec4f, 256>;
var<workgroup> bufB: array<vec4f, 256>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn texel(line: u32, i: u32) -> vec2u {
  if (HORIZONTAL) { return vec2u(i, line); }
  return vec2u(line, i);
}

@compute @workgroup_size(256, 1, 1)
fn fft(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u) {
  let i = lid.x;
  let line = wid.y;
  let layer = wid.z;
  let rev = reverseBits(i) >> (32u - LOG_N);
  bufA[i] = textureLoad(src, texel(line, rev), layer, 0);
  workgroupBarrier();

  var readA = true;
  for (var s = 0u; s < LOG_N; s++) {
    let halfSize = 1u << s;
    let size = halfSize << 1u;
    let k = i & (size - 1u);
    let j = k & (halfSize - 1u);
    let base = i - k;
    let angle = TAU * f32(j) / f32(size); // positive exponent: inverse transform
    let w = vec2f(cos(angle), sin(angle));

    var even: vec4f;
    var odd: vec4f;
    if (readA) {
      even = bufA[base + j];
      odd = bufA[base + j + halfSize];
    } else {
      even = bufB[base + j];
      odd = bufB[base + j + halfSize];
    }
    let wo = vec4f(cmul(w, odd.xy), cmul(w, odd.zw));
    let result = select(even - wo, even + wo, k < halfSize);
    if (readA) {
      bufB[i] = result;
    } else {
      bufA[i] = result;
    }
    workgroupBarrier();
    readA = !readA;
  }

  var outValue: vec4f;
  if (readA) { outValue = bufA[i]; } else { outValue = bufB[i]; }
  textureStore(dst, texel(line, i), layer, outValue);
}
