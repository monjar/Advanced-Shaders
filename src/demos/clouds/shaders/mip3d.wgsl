// 2x2x2 box downsample for one mip level of a 3D texture.

@group(0) @binding(0) var src: texture_3d<f32>;
@group(0) @binding(1) var dst: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (any(id >= size)) { return; }
  let p = id * 2u;
  var sum = vec4f(0.0);
  for (var k = 0u; k < 8u; k++) {
    sum += textureLoad(src, p + vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u), 0);
  }
  textureStore(dst, id, sum * 0.125);
}
