// Tileable noise functions used to build the cloud textures on the GPU.
//   base   128^3 rgba8: R Perlin-Worley, GBA Worley fBm at 4, 8, 16 cells
//   detail  32^3 rgba8: RGB Worley fBm at 2, 4, 8 cells
//   curl   128^2 rgba8snorm: 2D curl of tileable Perlin noise (xy)
//   weather 512^2 rgba8: R coverage, G cloud type, B density
// After Schneider, "The real-time volumetric cloudscapes of Horizon Zero Dawn"
// (SIGGRAPH 2015) and Hillaire's TileableVolumeNoise.

struct GenParams {
  sliceOffset: u32,
  seed: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> G: GenParams;

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

fn random3(cell: vec3i, period: i32) -> vec3f {
  let wrapped = ((cell % period) + period) % period;
  return vec3f(pcg3d(vec3u(wrapped) + vec3u(G.seed * 7919u))) / 4294967295.0;
}

// Inverted Worley (cellular) noise, periodic over `period` cells.
fn worley(p: vec3f, period: f32) -> f32 {
  let q = p * period;
  let cell = floor(q);
  let f = q - cell;
  var d = 1e9;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        let o = vec3f(f32(x), f32(y), f32(z));
        let feature = o + random3(vec3i(cell + o), i32(period));
        let v = feature - f;
        d = min(d, dot(v, v));
      }
    }
  }
  return 1.0 - clamp(sqrt(d), 0.0, 1.0);
}

fn worleyFbm(p: vec3f, period: f32) -> f32 {
  return worley(p, period) * 0.625 + worley(p, period * 2.0) * 0.25 + worley(p, period * 4.0) * 0.125;
}

fn fade(t: vec3f) -> vec3f {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn gradient(cell: vec3i, period: i32) -> vec3f {
  return normalize(random3(cell, period) * 2.0 - 1.0);
}

// Periodic gradient noise in [-1, 1].
fn perlin(p: vec3f, period: f32) -> f32 {
  let q = p * period;
  let i = vec3i(floor(q));
  let f = fract(q);
  let u = fade(f);
  let P = i32(period);
  var n: array<f32, 8>;
  for (var k = 0; k < 8; k++) {
    let o = vec3i(k & 1, (k >> 1) & 1, (k >> 2) & 1);
    n[k] = dot(gradient(i + o, P), f - vec3f(o));
  }
  let x0 = mix(n[0], n[1], u.x);
  let x1 = mix(n[2], n[3], u.x);
  let x2 = mix(n[4], n[5], u.x);
  let x3 = mix(n[6], n[7], u.x);
  return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z) * 1.15;
}

fn perlinFbm(p: vec3f, period: f32, octaves: i32) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var norm = 0.0;
  var freq = period;
  for (var o = 0; o < octaves; o++) {
    sum += perlin(p, freq) * amp;
    norm += amp;
    amp *= 0.5;
    freq *= 2.0;
  }
  return sum / norm;
}
