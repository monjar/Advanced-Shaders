// Temporal reconstruction at cloud-buffer resolution.
// Each pixel is reprojected into last frame's history using its cloud depth,
// the history is clipped to the variance of this frame's fresh samples around
// it (rejects ghosting when clouds or the camera move), and blended.
// With the 2x2 pattern, a pixel receives a new ray every 4th frame and relies
// on reprojected history in between.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> C: Clouds;
@group(1) @binding(0) var traceTex: texture_2d<f32>;
@group(1) @binding(1) var depthTex: texture_2d<f32>;
@group(1) @binding(2) var historyIn: texture_2d<f32>;
@group(1) @binding(3) var linearClamp: sampler;
@group(1) @binding(4) var historyOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(historyOut);
  if (any(id.xy >= size)) { return; }
  let p = id.xy;
  let quarter = u32(C.temporal.y) == 4u;
  let frame = u32(C.temporal.x);
  let traceSize = vec2i(textureDimensions(traceTex));
  let tp = vec2i(select(p, p / 2u, quarter));
  let fresh = !quarter || all(p % 2u == bayerOffset(frame));

  // Neighbourhood statistics from this frame's traced samples.
  var m1 = vec4f(0.0);
  var m2 = vec4f(0.0);
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let s = textureLoad(traceTex, clamp(tp + vec2i(x, y), vec2i(0), traceSize - 1), 0);
      m1 += s;
      m2 += s * s;
    }
  }
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec4f(0.0)));
  let current = textureLoad(traceTex, tp, 0);
  let depth = textureLoad(depthTex, tp, 0).r;

  // Reproject: where was this cloud point on screen last frame?
  let uv = (vec2f(p) + 0.5) / vec2f(size);
  let world = F.camPos + viewRay(uv) * depth;
  let prevClip = C.prevViewProj * vec4f(world, 1.0);
  let prevNdc = prevClip.xy / prevClip.w;
  let prevUv = vec2f(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
  let onScreen = prevClip.w > 0.0 && all(prevUv > vec2f(0.0)) && all(prevUv < vec2f(1.0));
  // temporal.w is 0 when disabled or right after a reset (resize, mode change).
  let useHistory = C.temporal.w > 0.5 && onScreen;

  var result = current;
  if (useHistory) {
    let history = textureSampleLevel(historyIn, linearClamp, prevUv, 0.0);
    // Variance clipping; wider for the sparse 2x2 pattern.
    let gamma = select(1.25, 2.0, quarter);
    let clipped = clamp(history, mean - gamma * sigma, mean + gamma * sigma);
    if (fresh) {
      result = mix(clipped, current, C.temporal.z);
    } else {
      result = clipped;
    }
  }
  textureStore(historyOut, p, result);
}
