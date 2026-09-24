// Temporal accumulation (compute).
// Progressive mean of jittered frames while nothing changes (F.post.z is the
// weight of the new frame, 1/(n+1)); weight 1 just copies.

@group(1) @binding(0) var accCurrent: texture_2d<f32>;
@group(1) @binding(1) var accHistory: texture_2d<f32>;
@group(1) @binding(2) var accOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= textureDimensions(accOut))) { return; }
  let c = textureLoad(accCurrent, id.xy, 0);
  let h = textureLoad(accHistory, id.xy, 0);
  textureStore(accOut, id.xy, mix(h, c, F.post.z));
}
