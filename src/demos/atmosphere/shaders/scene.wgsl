// LUT path, one thread per pixel: trace the scene, light the ground through
// the transmittance LUT and apply aerial perspective from the froxel volume;
// sky pixels read the sky-view LUT (or ray march it from orbit).
// Output: HDR radiance, alpha = hit distance in km (-1 for sky).

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdrOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= F.resolution)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / F.resolution;
  let dir = cameraRay(uv);
  let hit = traceScene(dir);
  var color: vec3f;
  if (hit.t >= 0.0) {
    color = atmoApply(shadeGround(hit, atmoSunTransmittanceAt(hit.pos), dir), uv, dir, hit.t);
  } else {
    color = atmoSkyRadiance(dir) + atmoSunDisk(dir) + stars(dir) * atmoCameraTransmittance(dir);
  }
  textureStore(hdrOut, id.xy, vec4f(color, hit.t));
}
