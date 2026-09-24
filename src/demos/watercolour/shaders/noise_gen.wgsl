// Tileable 3D gradient noise, 4 decorrelated channels, period NOISE_CELLS.

@group(0) @binding(0) var noiseOut: texture_storage_3d<rgba8unorm, write>;

fn pcg3d(input: vec3u) -> vec3u {
  var v = input * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> vec3u(16u);
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn gradient(cell: vec3i, period: i32, channel: u32) -> vec3f {
  let c = vec3u(((cell % period) + period) % period) + vec3u(channel * 101u, channel * 37u, channel * 59u);
  return normalize(vec3f(pcg3d(c)) / 4294967295.0 * 2.0 - 1.0);
}

fn perlin(q: vec3f, period: i32, channel: u32) -> f32 {
  let i = vec3i(floor(q));
  let f = fract(q);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  var n: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    n[k] = dot(gradient(i + o, period, channel), f - vec3f(o));
  }
  let x0 = mix(n[0], n[1], u.x);
  let x1 = mix(n[2], n[3], u.x);
  let x2 = mix(n[4], n[5], u.x);
  let x3 = mix(n[6], n[7], u.x);
  return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}

// Two octaves so a single lookup already has some texture.
fn channel(q: vec3f, c: u32) -> f32 {
  let n = perlin(q, 16, c) * 0.75 + perlin(q * 2.0, 32, c + 4u) * 0.25;
  return clamp(n * 0.9 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn generate(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(noiseOut);
  if (any(id >= size)) { return; }
  let q = (vec3f(id) + 0.5) / vec3f(size) * 16.0;
  textureStore(noiseOut, id, vec4f(channel(q, 0u), channel(q, 1u), channel(q, 2u), channel(q, 3u)));
}
