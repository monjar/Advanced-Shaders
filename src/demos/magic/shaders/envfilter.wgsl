// Prefilters the reflection cube: each mip of each face gathers the previous
// mip over a cone (Vogel disc of 16 taps) whose width grows with the mip, a
// cheap stand-in for GGX prefiltering. Sampling by direction through a cube
// view keeps face seams invisible.

@group(0) @binding(0) var src: texture_cube<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> P: vec4f; // face, target size, cone radius (radians), unused

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Direction of texel `uv` (v down) on cube `face` (WebGPU / D3D convention).
fn faceDir(face: u32, uv: vec2f) -> vec3f {
  let sc = uv.x * 2.0 - 1.0;
  let tc = uv.y * 2.0 - 1.0;
  switch face {
    case 0u: { return normalize(vec3f(1.0, -tc, -sc)); }
    case 1u: { return normalize(vec3f(-1.0, -tc, sc)); }
    case 2u: { return normalize(vec3f(sc, 1.0, tc)); }
    case 3u: { return normalize(vec3f(sc, -1.0, -tc)); }
    case 4u: { return normalize(vec3f(sc, -tc, 1.0)); }
    default: { return normalize(vec3f(-sc, -tc, -1.0)); }
  }
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let d = faceDir(u32(P.x + 0.5), pos.xy / P.y);
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(d.y) > 0.9);
  let t = normalize(cross(up, d));
  let b = cross(d, t);
  var sum = vec4f(0.0);
  var wsum = 0.0;
  for (var i = 0; i < 16; i++) {
    let r = sqrt((f32(i) + 0.5) / 16.0);
    let a = f32(i) * 2.39996323;
    let o = vec2f(cos(a), sin(a)) * r * P.z;
    let w = exp(-r * r * 2.0);
    sum += textureSampleLevel(src, linearSampler, normalize(d + t * o.x + b * o.y), 0.0) * w;
    wsum += w;
  }
  return sum / wsum;
}
