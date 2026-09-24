// Analytic signed distance fields of the object shapes, baked into a 3D
// texture (distance, outward normal) whenever the shape changes. The meshes
// in geometry.ts are built from the same parameters (the Shape uniform).

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> S: Shape;
@group(0) @binding(2) var shapeOut: texture_storage_3d<rgba16float, write>;

struct Crystal {
  c: vec4f,   // centre, hexagon inradius
  a: vec4f,   // axis, half length of the body
  b: vec4f,   // hexagon frame axis, tip height
};

struct Shape {
  info: vec4f,   // kind, crystal count, sphere radius, bake half extent
  knot: vec4f,   // R, r, tube radius, unused
  crystals: array<Crystal, 8>,
};

// (p, q) of the torus knot; must match KNOT in geometry.ts.
const KNOT_P: f32 = 2.0;
const KNOT_Q: f32 = 3.0;

// Convex hexagonal bipyramid-prism as the max of its 18 face planes (exact
// inside, a slight underestimate outside near edges, which is harmless here).
fn crystalSdf(p: vec3f, cr: Crystal) -> vec4f {
  let a = cr.a.xyz;
  let b = cr.b.xyz;
  let c = cross(a, b);
  let q = p - cr.c.xyz;
  let l = vec3f(dot(q, b), dot(q, a), dot(q, c));
  let r = cr.c.w;
  let h = cr.a.w;
  let tip = cr.b.w;
  let denom = sqrt(tip * tip + r * r);
  var best = -1e9;
  var nl = vec3f(0.0, 1.0, 0.0);
  for (var k = 0; k < 6; k++) {
    let ang = (f32(k) + 0.5) * PI / 3.0;
    let nk = vec2f(cos(ang), sin(ang));
    let radial = dot(l.xz, nk);
    let side = radial - r;
    if (side > best) { best = side; nl = vec3f(nk.x, 0.0, nk.y); }
    let top = (radial * tip + (l.y - h) * r - r * tip) / denom;
    if (top > best) { best = top; nl = vec3f(nk.x * tip, r, nk.y * tip) / denom; }
    let bottom = (radial * tip + (-l.y - h) * r - r * tip) / denom;
    if (bottom > best) { best = bottom; nl = vec3f(nk.x * tip, -r, nk.y * tip) / denom; }
  }
  return vec4f(best, nl.x * b + nl.y * a + nl.z * c);
}

fn knotPoint(t: f32) -> vec3f {
  let s = S.knot.x + S.knot.y * cos(KNOT_Q * t);
  return vec3f(s * cos(KNOT_P * t), s * sin(KNOT_P * t), S.knot.y * sin(KNOT_Q * t));
}

// Distance to the knot's centre curve: coarse search, then Newton on t for
// the closest point, minus the tube radius.
fn knotSdf(p: vec3f) -> vec4f {
  let N = 128;
  var bestT = 0.0;
  var bestD = 1e9;
  for (var i = 0; i < N; i++) {
    let t = f32(i) / f32(N) * TAU;
    let d = distance(knotPoint(t), p);
    if (d < bestD) { bestD = d; bestT = t; }
  }
  var t = bestT;
  let R = S.knot.x;
  let r = S.knot.y;
  for (var it = 0; it < 4; it++) {
    let s = R + r * cos(KNOT_Q * t);
    let ds = -r * KNOT_Q * sin(KNOT_Q * t);
    let dds = -r * KNOT_Q * KNOT_Q * cos(KNOT_Q * t);
    let cp = cos(KNOT_P * t);
    let sp = sin(KNOT_P * t);
    let c = vec3f(s * cp, s * sp, r * sin(KNOT_Q * t));
    let d1 = vec3f(ds * cp - s * KNOT_P * sp, ds * sp + s * KNOT_P * cp, r * KNOT_Q * cos(KNOT_Q * t));
    let d2 = vec3f(
      dds * cp - 2.0 * ds * KNOT_P * sp - s * KNOT_P * KNOT_P * cp,
      dds * sp + 2.0 * ds * KNOT_P * cp - s * KNOT_P * KNOT_P * sp,
      -r * KNOT_Q * KNOT_Q * sin(KNOT_Q * t));
    let e = c - p;
    let g = dot(e, d1);
    let hh = dot(d1, d1) + dot(e, d2);
    t -= g / max(hh, 1e-3);
  }
  let c = knotPoint(t);
  let e = p - c;
  let len = length(e);
  return vec4f(len - S.knot.z, e / max(len, 1e-5));
}

fn shapeSdf(p: vec3f) -> vec4f {
  let kind = u32(S.info.x + 0.5);
  if (kind == 1u) {
    var best = vec4f(1e9, 0.0, 1.0, 0.0);
    for (var i = 0u; i < u32(S.info.y); i++) {
      let d = crystalSdf(p, S.crystals[i]);
      if (d.x < best.x) { best = d; }
    }
    return best;
  }
  if (kind == 2u) {
    return knotSdf(p);
  }
  let len = length(p);
  return vec4f(len - S.info.z, p / max(len, 1e-5));
}

@compute @workgroup_size(4, 4, 4)
fn bakeShape(@builtin(global_invocation_id) id: vec3u) {
  let n = textureDimensions(shapeOut);
  if (any(id >= n)) { return; }
  let p = ((vec3f(id) + 0.5) / vec3f(n) * 2.0 - 1.0) * S.info.w;
  textureStore(shapeOut, id, shapeSdf(p));
}
