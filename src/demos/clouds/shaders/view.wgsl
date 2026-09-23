// View-ray and planet-sphere helpers. Require uniforms F: Frame and C: Clouds.

fn viewRay(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let p = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(p.xyz / p.w - F.camPos);
}

// Ray/sphere intersection around the planet centre, in a frame where the
// camera sits at (0, R + altitude, 0). `c` is |o|^2 - r^2 computed from
// altitudes to avoid cancellation with 6,360 km radii. Returns (t0, t1),
// or t0 > t1 on a miss.
fn raySphere(dir: vec3f, sphereAltitude: f32) -> vec2f {
  let R = C.layer.z;
  let h = C.view.x;
  let b = (R + h) * dir.y;
  let c = (h - sphereAltitude) * (2.0 * R + h + sphereAltitude);
  let disc = b * b - c;
  if (disc < 0.0) { return vec2f(1.0, -1.0); }
  let s = sqrt(disc);
  let q = -b - select(-s, s, b >= 0.0);
  let t0 = q;
  let t1 = c / q;
  return vec2f(min(t0, t1), max(t0, t1));
}

fn cameraLocal() -> vec3f {
  return vec3f(0.0, C.layer.z + C.view.x, 0.0);
}

// Distance range the view ray spends inside the cloud layer (first segment
// only), clipped by the ground and the maximum march distance.
// Returns start > end when the ray misses the layer.
fn layerSegment(dir: vec3f) -> vec2f {
  let top = raySphere(dir, C.layer.y);
  if (top.x > top.y || top.y <= 0.0) { return vec2f(1.0, -1.0); }
  var start = max(top.x, 0.0);
  var end = top.y;
  let bottom = raySphere(dir, C.layer.x);
  if (bottom.x <= bottom.y) {
    if (bottom.x > 0.0) {
      end = min(end, bottom.x);          // from above: stop where the ray drops below the layer
    } else if (bottom.y > 0.0) {
      start = max(start, bottom.y);      // from below: start where the ray rises into the layer
    }
  }
  let ground = raySphere(dir, 0.0);
  if (ground.x <= ground.y && ground.x > 0.0) { end = min(end, ground.x); }
  end = min(end, C.layer.w);
  return vec2f(start, end);
}
