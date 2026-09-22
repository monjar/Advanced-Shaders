// 2x2 box downsample for one mip level of a 2D array texture.

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = id.xy * 2u;
  let c = textureLoad(src, p, id.z, 0) + textureLoad(src, p + vec2u(1u, 0u), id.z, 0)
        + textureLoad(src, p + vec2u(0u, 1u), id.z, 0) + textureLoad(src, p + vec2u(1u, 1u), id.z, 0);
  textureStore(dst, id.xy, id.z, c * 0.25);
}
