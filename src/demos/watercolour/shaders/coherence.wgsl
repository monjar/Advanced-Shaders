// Temporal coherence meter. Every 4th pixel (in x and y) is reprojected into the previous
// frame using its depth and last frame's camera; the difference between the
// stylised images there is what a viewer perceives as flicker or sliding
// texture. Summed with atomics and read back a few times per second.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Paint;
@group(1) @binding(0) var current: texture_2d<f32>;
@group(1) @binding(1) var previous: texture_2d<f32>;
@group(1) @binding(2) var depthTex: texture_2d<f32>;
@group(1) @binding(3) var<storage, read_write> totals: array<atomic<u32>, 4>;
@group(1) @binding(4) var linearClamp: sampler;

@compute @workgroup_size(8, 8, 1)
fn measure(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(current));
  let p = vec2i(id.xy) * 4 + 2;
  if (any(p >= size)) { return; }
  let depth = textureLoad(depthTex, p, 0).xy;
  if (depth.y == SKY_ID) { return; }
  let px = vec2f(p) + 0.5;
  let world = F.camPos + viewRay(px) * depth.x;
  let c = F.prevViewProj * vec4f(world, 1.0);
  let ndc = c.xy / c.w;
  let prevUv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let q = prevUv * vec2f(size);
  if (c.w <= 0.0 || any(q < vec2f(2.0)) || any(q >= vec2f(size - 2))) { return; }
  // Bilinear lookup so sub-pixel motion alone isn't counted as incoherence.
  let a = textureLoad(current, p, 0).rgb;
  let b = textureSampleLevel(previous, linearClamp, prevUv, 0.0).rgb;
  let diff = dot(abs(a - b), vec3f(1.0 / 3.0));
  atomicAdd(&totals[0], u32(diff * 10000.0));
  atomicAdd(&totals[1], 1u);
}
