// Base shape noise, generated in slabs of slices across several frames.

@group(1) @binding(0) var baseOut: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn genBase(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid + vec3u(0u, 0u, G.sliceOffset);
  let size = textureDimensions(baseOut);
  if (any(id >= size)) { return; }
  let p = (vec3f(id) + 0.5) / vec3f(size);
  let perlin01 = clamp(perlinFbm(p, 4.0, 4) * 0.5 + 0.5, 0.0, 1.0);
  let w4 = worleyFbm(p, 4.0);
  let perlinWorley = remap(perlin01, 0.0, 1.0, w4, 1.0);
  textureStore(baseOut, id, vec4f(perlinWorley, w4, worleyFbm(p, 8.0), worleyFbm(p, 16.0)));
}
