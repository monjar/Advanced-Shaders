// Detail, curl and weather textures.

@group(1) @binding(0) var detailOut: texture_storage_3d<rgba8unorm, write>;
@group(1) @binding(1) var curlOut: texture_storage_2d<rgba8snorm, write>;
@group(1) @binding(2) var weatherOut: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn genDetail(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(detailOut);
  if (any(id >= size)) { return; }
  let p = (vec3f(id) + 0.5) / vec3f(size);
  textureStore(detailOut, id, vec4f(worleyFbm(p, 2.0), worleyFbm(p, 4.0), worleyFbm(p, 8.0), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn genCurl(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(curlOut);
  if (any(id.xy >= size)) { return; }
  let p = (vec2f(id.xy) + 0.5) / vec2f(size);
  let e = 1.0 / f32(size.x);
  let nx0 = perlinFbm(vec3f(p - vec2f(e, 0.0), 0.5), 4.0, 3);
  let nx1 = perlinFbm(vec3f(p + vec2f(e, 0.0), 0.5), 4.0, 3);
  let ny0 = perlinFbm(vec3f(p - vec2f(0.0, e), 0.5), 4.0, 3);
  let ny1 = perlinFbm(vec3f(p + vec2f(0.0, e), 0.5), 4.0, 3);
  let curl = vec2f(ny1 - ny0, -(nx1 - nx0)) / (2.0 * e);
  textureStore(curlOut, id.xy, vec4f(clamp(curl * 0.08, vec2f(-1.0), vec2f(1.0)), 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn genWeather(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(weatherOut);
  if (any(id.xy >= size)) { return; }
  let p = vec3f((vec2f(id.xy) + 0.5) / vec2f(size), 0.25);
  // Coverage: Perlin fBm shaped by Worley so clouds gather into clusters.
  let perlin01 = clamp(perlinFbm(p, 3.0, 5) * 0.5 + 0.5, 0.0, 1.0);
  let cells = worleyFbm(p, 4.0);
  let coverage = clamp((perlin01 * 0.65 + cells * 0.35 - 0.3) / 0.45, 0.0, 1.0);
  let kind = clamp(perlinFbm(p + vec3f(0.37, 0.61, 0.0), 2.0, 3) * 0.7 + 0.5, 0.0, 1.0);
  let density = clamp(perlinFbm(p + vec3f(0.71, 0.13, 0.5), 6.0, 2) * 0.5 + 0.5, 0.0, 1.0);
  textureStore(weatherOut, id.xy, vec4f(coverage, kind, density, 1.0));
}
