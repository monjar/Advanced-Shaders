// Top-down cloud shadow map around the camera: transmittance of sunlight
// reaching the ground, marched through the layer with cheap density.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> C: Clouds;
@group(2) @binding(0) var shadowOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn shadow(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(shadowOut);
  if (any(id.xy >= size)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let world = C.shadow.xy + (uv - 0.5) * C.shadow.z;
  let d = world - F.camPos.xz;
  let R = C.layer.z;
  // Ground point in the planet-centred frame (curvature to second order).
  let ground = vec3f(d.x, R - dot(d, d) / (2.0 * R), d.y);

  let sunY = max(F.sunDir.y, 0.03);
  let t0 = C.layer.x / sunY;
  let t1 = min(C.layer.y / sunY, t0 + 30000.0);
  let steps = 16;
  let dt = (t1 - t0) / f32(steps);
  var od = 0.0;
  for (var i = 0; i < steps; i++) {
    let p = ground + F.sunDir * (t0 + (f32(i) + 0.5) * dt);
    od += cloudDensity(p, length(p) - R, true, 0.0) * dt;
  }
  let T = exp(-od * C.light.x * C.shadow.w);
  textureStore(shadowOut, id.xy, vec4f(T, 0.0, 0.0, 1.0));
}
