// Caustics by photon-area ratio (after Evan Wallace, "Rendering realtime
// caustics in WebGL"). A grid covering one tile of cascade 1 is refracted
// towards a plane at `causticDepth`; each triangle is rasterised where its
// light lands, with intensity = original area / projected area, and the
// results are summed additively. Instances offset by +/-L keep it tileable.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> O: Ocean;
@group(1) @binding(0) var derivTex: texture_2d_array<f32>;
@group(1) @binding(1) var linSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) original: vec2f,
  @location(1) projected: vec2f,
};

@vertex
fn vs(@location(0) ij: vec2f, @builtin(instance_index) instance: u32) -> VSOut {
  let n = 256.0;
  let L = O.lengths[1];
  let p = ij / n * L;
  let d = textureSampleLevel(derivTex, linSampler, ij / n, 1u, 0.0) * O.sim.y;
  let slope = vec2f(d.x / max(1.0 + d.z, 0.2), d.y / max(1.0 + d.w, 0.2));
  let N = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let r = refract(-F.sunDir, N, 1.0 / WATER_IOR);
  let flatDir = refract(-F.sunDir, vec3f(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
  let depth = O.terrain.z;
  // Landing point relative to where flat water would send the same light.
  let hit = p + r.xz * (depth / max(-r.y, 0.05)) - flatDir.xz * (depth / max(-flatDir.y, 0.05));

  let offset = vec2f(f32(instance % 3u) - 1.0, f32(instance / 3u) - 1.0) * L;
  let q = (hit + offset) / L;
  var out: VSOut;
  out.position = vec4f(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0, 0.0, 1.0);
  out.original = p;
  out.projected = hit;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let a0 = abs(determinant(mat2x2f(dpdx(in.original), dpdy(in.original))));
  let a1 = abs(determinant(mat2x2f(dpdx(in.projected), dpdy(in.projected))));
  let intensity = min(a0 / max(a1, 1e-8), 12.0);
  return vec4f(intensity, 0.0, 0.0, 1.0);
}
