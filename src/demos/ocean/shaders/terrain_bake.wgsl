// Bakes the island / seabed height field once, together with a signed
// distance to the shoreline and the direction towards the shore, which drive
// shallow-water attenuation and the shoreline Gerstner waves.
// Output: (height m, shore distance m, shore direction x, shore direction z)

struct BakeParams {
  halfExtent: f32,
  islandX: f32,
  islandZ: f32,
  shoreRadius: f32,
};

@group(0) @binding(0) var<uniform> B: BakeParams;
@group(0) @binding(1) var terrainOut: texture_storage_2d<rgba16float, write>;

// Warped elliptical distance from the island centre.
fn islandDistance(p: vec2f) -> f32 {
  let q = (p - vec2f(B.islandX, B.islandZ)) * vec2f(1.0, 1.25);
  let warp = (fbm(p * 0.006 + vec2f(3.1, 1.7), 4) - 0.5) * 70.0;
  return length(q) + warp;
}

fn profile(t: f32) -> f32 {
  let R = B.shoreRadius;
  let beach = (R - t) * 0.055;
  let hill = 17.0 * pow(smoothstep(R - 5.0, 0.0, t), 1.6);
  let dropOff = -32.0 * smoothstep(R + 150.0, R + 260.0, t);
  return max(beach + hill + dropOff, -48.0);
}

@compute @workgroup_size(8, 8, 1)
fn bake(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(terrainOut);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let p = (uv - 0.5) * 2.0 * B.halfExtent;

  let t = islandDistance(p);
  let e = 1.0;
  let grad = vec2f(islandDistance(p + vec2f(e, 0.0)) - islandDistance(p - vec2f(e, 0.0)),
                   islandDistance(p + vec2f(0.0, e)) - islandDistance(p - vec2f(0.0, e)));
  let toShore = -normalize(grad + vec2f(1e-6));

  let R = B.shoreRadius;
  var h = profile(t);
  let land = smoothstep(R - 8.0, R - 40.0, t);
  h += (fbm(p * 0.035, 5) - 0.5) * mix(0.5, 3.5, land);
  h += (1.0 - abs(fbm(p * 0.012 + 7.0, 4) * 2.0 - 1.0)) * 6.0 * smoothstep(R - 30.0, R - 80.0, t);
  // Gentle sand bars on the shelf.
  h += sin(t * 0.18 + fbm(p * 0.02, 3) * 4.0) * 0.35 * smoothstep(R + 5.0, R + 30.0, t) * smoothstep(R + 160.0, R + 60.0, t);

  textureStore(terrainOut, id.xy, vec4f(h, t - R, toShore));
}
