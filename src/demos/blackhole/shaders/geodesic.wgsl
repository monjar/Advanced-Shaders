// Light paths around the hole. Three ways, all selectable for comparison:
//
//  1. Artistic (traceArtistic): march in flat 3D space and bend the direction
//     with an ad-hoc inverse-square pull, the usual shadertoy trick.
//  2. Orbital plane (tracePlane): the exact Schwarzschild null geodesic. A
//     photon moves in the plane through the hole spanned by its position and
//     direction, and u = 1/r obeys the Binet-form orbit equation
//          d²u/dφ² + u = 3Mu² = (3/2) r_s u²        (G = c = 1)
//     (Misner, Thorne and Wheeler §25.6; Chandrasekhar, "The Mathematical
//     Theory of Black Holes" §20). This is integrated with RK4 at a fixed dφ,
//     or with an adaptive Dormand–Prince 5(4) step, together with the
//     variational equation for ∂u/∂(du/dφ₀), which gives exact ray
//     differentials for texture filtering.
//  3. Cartesian (traceCartesian): the same geodesics written as flat-space
//     motion under a pseudo-force, a = -(3/2) r_s h² r̂ / r⁴ with
//     h = |r × v| conserved. For a central force f(r) Binet's equation reads
//     u'' + u = f / (h² u²), so f = (3/2) r_s h² u⁴ reproduces the orbit
//     equation above exactly. Integrated with RK4 at a step ∝ r. It exists
//     to cross-check the plane integrator and to compare costs.

const ESCAPED: u32 = 0u;
const CAPTURED: u32 = 1u;
const STEP_LIMIT: u32 = 2u;
const ABSORBED: u32 = 3u;

struct CamRay {
  n: vec3f,      // traced (outgoing) direction in the local static frame
  dnx: vec3f,    // ∂n/∂(pixel x), ∂n/∂(pixel y): the camera's own ray differentials
  dny: vec3f,
  gObs: f32,     // frequency factor at the observer for light from infinity along n
};

struct Ray {
  radiance: vec3f,   // disk emission picked up along the way, already attenuated
  trans: f32,        // transmittance left for the background
  dir: vec3f,        // escape direction on the celestial sphere
  status: u32,
  dDx: vec3f,        // ∂dir/∂(pixel x), ∂dir/∂(pixel y), when analytic is true
  steps: u32,
  dDy: vec3f,
  crossings: u32,    // crossings of the disk annulus that were shaded
  nodes: u32,        // crossings of the whole disk plane, m: 1 direct image, 2 lensing ring, ≥ 3 photon ring
  gFirst: f32,       // redshift factor at the first crossing
  analytic: bool,
};

fn newRay() -> Ray {
  var r: Ray;
  r.radiance = vec3f(0.0);
  r.trans = 1.0;
  r.status = STEP_LIMIT;
  r.steps = 0u;
  r.crossings = 0u;
  r.nodes = 0u;
  r.gFirst = 0.0;
  r.analytic = false;
  return r;
}

// One crossing of the disk annulus at X (radius r). k: unit static-frame
// direction the photon travels in (towards the camera). fp: pixel footprint on
// the disk in noise units. delay: coordinate time the light took from X to
// the camera, so each image shows the disk when that light left it (the
// higher-order images lag behind the direct one).
fn addCrossing(ray: ptr<function, Ray>, X: vec3f, r: f32, k: vec3f, fp: f32, gObs: f32, artistic: bool, delay: f32) {
  var e: vec4f;
  var g = 1.0;
  let time = F.anim.x - select(0.0, delay, F.anim.z > 0.5);
  if (artistic) {
    e = diskShadeArtistic(X, r, fp, time);
  } else {
    g = diskRedshift(X, r, k, gObs);
    e = diskShade(X, r, fp, g, time);
  }
  (*ray).radiance += (*ray).trans * e.rgb;
  (*ray).trans *= 1.0 - e.a;
  if ((*ray).crossings == 0u) { (*ray).gFirst = g; }
  (*ray).crossings += 1u;
}

// Without ray differentials: the footprint a straight ray from the camera
// would have, a pixel-wide cone (w = dist · pixel angle) cut by the disk
// plane: stretched by 1/cos i along the ray's in-plane direction m. Lensing
// magnification is ignored, which is exactly what goes wrong near the hole.
fn unlensedFootprintAt(X: vec3f, r: f32, dir: vec3f, dist: f32) -> f32 {
  let w = dist * F.pixelAngle;
  let d = normalize(dir);
  let cosI = max(abs(dot(d, F.diskN)), 0.02);
  var m = d - F.diskN * dot(d, F.diskN);
  m = select(F.diskA, normalize(m), dot(m, m) > 1e-10);
  let q = cross(F.diskN, m);
  return diskFootprint(X, r, m * (w / cosI), q * w);
}

fn unlensedDiskFootprint(X: vec3f, r: f32, nloc: vec3f) -> f32 {
  return unlensedFootprintAt(X, r, nloc, length(X - F.camPos));
}

// ---- 1. Artistic -------------------------------------------------------------

fn traceArtistic(cr: CamRay, disk: bool) -> Ray {
  var ray = newRay();
  var x = F.camPos;
  var v = cr.n;
  let pull = F.art.x;
  let maxSteps = u32(F.art.z);
  var travelled = 0.0;
  loop {
    if (ray.steps >= maxSteps) { break; }
    let r2 = dot(x, x);
    let r = sqrt(r2);
    if (r < 1.0) { ray.status = CAPTURED; break; }
    if (r > F.art.w && dot(x, v) > 0.0) { ray.status = ESCAPED; break; }
    let ds = F.art.y * r;
    // Pull the direction towards the hole with an inverse-square "gravity" and
    // keep |v| = 1. Not a geodesic: no photon sphere, the wrong shadow size.
    v = normalize(v - x * (pull * ds / (r2 * r)));
    let xn = x + v * ds;
    ray.steps += 1u;
    let d0 = dot(x, F.diskN);
    let d1 = dot(xn, F.diskN);
    if (d0 * d1 < 0.0) { ray.nodes += 1u; }
    if (disk && F.disk.w > 0.5) {
      if (d0 * d1 < 0.0) {
        let t = d0 / (d0 - d1);
        let X = mix(x, xn, t);
        let rr = length(X);
        if (rr >= F.diskIn && rr <= F.diskOut) {
          addCrossing(&ray, X, rr, -v, unlensedDiskFootprint(X, rr, v), 1.0, true, travelled + t * ds);
          if (ray.trans < 0.004) { ray.status = ABSORBED; break; }
        }
      }
    }
    travelled += ds;
    x = xn;
  }
  ray.dir = v;
  return ray;
}

// ---- 2. Orbital plane --------------------------------------------------------

// State s = (u, du/dφ, w, dw/dφ) with w = ∂u/∂p, p = du/dφ at the camera.
// Variational equation: w'' = (-1 + 3u) w  (r_s = 1).
fn derivU(s: vec4f) -> vec4f {
  return vec4f(s.y, s.x * (1.5 * s.x - 1.0), s.w, s.z * (3.0 * s.x - 1.0));
}

// Coordinate time per radian of orbit, dt/dφ = r² / (b (1 - r_s/r)), from
// dt/dλ = E/(1 - r_s/r) and dφ/dλ = L/r² (b = L/E). It is carried along
// with the same stages as u; clamped where a stage overshoots the ends.
fn timeRate(u: f32, invB: f32) -> f32 {
  let uc = clamp(u, 1e-4, 0.999);
  return invB / (uc * uc * (1.0 - uc));
}

struct PlaneStep {
  y: vec4f,
  k7: vec4f,   // f(y): first stage of the next step (FSAL, Dormand–Prince only)
  err: f32,
  dt: f32,     // coordinate time elapsed over the step
};

fn rk4Plane(s: vec4f, h: f32, invB: f32) -> PlaneStep {
  let k1 = derivU(s);
  let s2 = s + (0.5 * h) * k1;
  let k2 = derivU(s2);
  let s3 = s + (0.5 * h) * k2;
  let k3 = derivU(s3);
  let s4 = s + h * k3;
  let k4 = derivU(s4);
  var o: PlaneStep;
  o.y = s + (h / 6.0) * (k1 + 2.0 * (k2 + k3) + k4);
  o.dt = (h / 6.0) * (timeRate(s.x, invB) + 2.0 * (timeRate(s2.x, invB) + timeRate(s3.x, invB)) + timeRate(s4.x, invB));
  return o;
}

// Dormand–Prince 5(4) (Dormand and Prince 1980, the ode45 pair), local
// extrapolation with the 5th-order solution. The error norm is |δ(u, u')| /
// |(u, u')|: for a straight line u = sin(φ∞ - φ)/b this is exactly the error
// in the escape angle, independent of the impact parameter.
fn dp45Plane(s: vec4f, k1: vec4f, h: f32, invB: f32) -> PlaneStep {
  let s2 = s + h * (0.2 * k1);
  let k2 = derivU(s2);
  let s3 = s + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2);
  let k3 = derivU(s3);
  let s4 = s + h * ((44.0 / 45.0) * k1 - (56.0 / 15.0) * k2 + (32.0 / 9.0) * k3);
  let k4 = derivU(s4);
  let s5 = s + h * ((19372.0 / 6561.0) * k1 - (25360.0 / 2187.0) * k2 + (64448.0 / 6561.0) * k3 - (212.0 / 729.0) * k4);
  let k5 = derivU(s5);
  let s6 = s + h * ((9017.0 / 3168.0) * k1 - (355.0 / 33.0) * k2 + (46732.0 / 5247.0) * k3 + (49.0 / 176.0) * k4 - (5103.0 / 18656.0) * k5);
  let k6 = derivU(s6);
  var o: PlaneStep;
  o.y = s + h * ((35.0 / 384.0) * k1 + (500.0 / 1113.0) * k3 + (125.0 / 192.0) * k4 - (2187.0 / 6784.0) * k5 + (11.0 / 84.0) * k6);
  o.k7 = derivU(o.y);
  let e = h * ((71.0 / 57600.0) * k1 - (71.0 / 16695.0) * k3 + (71.0 / 1920.0) * k4 - (17253.0 / 339200.0) * k5 + (22.0 / 525.0) * k6 - (1.0 / 40.0) * o.k7);
  o.err = length(e.xy) / max(length(o.y.xy), 1e-6);
  // Same 5th-order weights for the time integral (the error norm ignores it).
  o.dt = h * ((35.0 / 384.0) * timeRate(s.x, invB) + (500.0 / 1113.0) * timeRate(s3.x, invB) + (125.0 / 192.0) * timeRate(s4.x, invB)
    - (2187.0 / 6784.0) * timeRate(s5.x, invB) + (11.0 / 84.0) * timeRate(s6.x, invB));
  return o;
}

// Cubic Hermite on [0, 1] with end values p and φ-derivatives m over a step h.
fn hermite(p0: f32, m0: f32, p1: f32, m1: f32, h: f32, t: f32) -> f32 {
  let t2 = t * t;
  let t3 = t2 * t;
  return (2.0 * t3 - 3.0 * t2 + 1.0) * p0 + (t3 - 2.0 * t2 + t) * h * m0 + (3.0 * t2 - 2.0 * t3) * p1 + (t3 - t2) * h * m1;
}

// d/dφ of the Hermite interpolant.
fn hermiteD(p0: f32, m0: f32, p1: f32, m1: f32, h: f32, t: f32) -> f32 {
  let t2 = t * t;
  return ((6.0 * t2 - 6.0 * t) * (p0 - p1) + (3.0 * t2 - 4.0 * t + 1.0) * h * m0 + (3.0 * t2 - 2.0 * t) * h * m1) / h;
}

fn tracePlane(cr: CamRay, disk: bool) -> Ray {
  var ray = newRay();
  // Orbital-plane basis: e1 towards the camera, e2 the in-plane direction the
  // ray turns towards (φ increases along the ray), e3 = e1 × e2 the normal.
  let u0 = 1.0 / F.rObs;
  let e1 = F.camPos * u0;
  let nr = dot(cr.n, e1);
  var c = cross(e1, cr.n);
  var sinT = length(c);
  if (sinT < 1e-7) {
    // Looking straight at (or away from) the hole: any plane will do.
    let helper = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(e1.y) > 0.9);
    c = normalize(cross(e1, helper)) * 1e-7;
    sinT = 1e-7;
  }
  let e3 = c / sinT;
  let e2 = cross(e3, e1);

  // Initial slope. A static observer measures the angle θ between the ray and
  // the outward radial in an orthonormal frame, where the radial component is
  // stretched by (1 - r_s/r)^(-1/2); hence du/dφ = -u √(1 - u) cot θ.
  // (Check: it satisfies the first integral u'² = 1/b² - u² + u³ with the
  // impact parameter b = r sin θ / √(1 - r_s/r).)
  let K = u0 * sqrt(1.0 - u0);
  let p0 = -K * nr / sinT;
  // 1/b from the first integral: 1/b² = u'² + u² - u³.
  let invB = sqrt(max(p0 * p0 + u0 * u0 * (1.0 - u0), 1e-12));

  // Ray differentials. A pixel step dn changes the slope p through θ and
  // rotates the plane about e1 at rate κ (de2 = κ e3).
  let s3 = sinT * sinT * sinT;
  let dpx = -K * dot(cr.dnx, e1) / s3;
  let dpy = -K * dot(cr.dny, e1) / s3;
  let kx = dot(e3, cr.dnx) / sinT;
  let ky = dot(e3, cr.dny) / sinT;

  // The photon's plane meets the disk plane along the line of nodes ℓ, so
  // disk crossings happen at φ = φ_ℓ + kπ. Steps are clipped to land on them
  // when there is a disk to shade; otherwise they are only counted.
  let shade = disk && F.disk.w > 0.5;
  var phiC = 1e9;
  {
    let l = cross(e3, F.diskN);
    if (dot(l, l) > 1e-12) {
      let phiN = atan2(dot(l, e2), dot(l, e1));
      phiC = phiN - PI * floor(phiN / PI);
      if (phiC < 1e-4) { phiC += PI; }
    }
  }

  var s = vec4f(u0, p0, 0.0, 1.0);
  var phi = 0.0;
  let adaptive = F.integ.x > 0.5;
  var h = F.integ.y;
  let tol = F.integ.z;
  let maxSteps = u32(F.integ.w);
  var k1 = derivU(s);
  var phiInf = 0.0;
  var dphidp = 0.0;
  var time = 0.0;
  loop {
    if (ray.steps >= maxSteps) { break; }
    // Inside the photon sphere and falling: u'' > 0 there, so it cannot turn
    // round (and no disk lies inside r_in ≥ 1.5 r_s). Saves the steps to r_s.
    if (s.y > 0.0 && s.x > U_PHOTON) { ray.status = CAPTURED; break; }
    let toCross = phiC - phi;
    let land = shade && h >= toCross;
    let hs = select(h, toCross, land);
    ray.steps += 1u;
    var next: vec4f;
    var dt: f32;
    if (adaptive) {
      let st = dp45Plane(s, k1, hs, invB);
      let fac = clamp(0.9 * pow(tol / max(st.err, 1e-12), 0.2), 0.2, 5.0);
      if (st.err > tol && hs > 1e-5) {
        h = hs * fac;
        continue;
      }
      next = st.y;
      dt = st.dt;
      k1 = st.k7;
      // A step clipped to land on a crossing says nothing against the longer
      // proposal, so don't let it shrink h.
      h = min(select(hs * fac, max(h, hs * fac), land), 0.8);
    } else {
      let st = rk4Plane(s, hs, invB);
      next = st.y;
      dt = st.dt;
    }
    if (next.x <= 0.0) {
      // Escaped: u = 0 on the step's Hermite interpolant, a few Newton steps.
      var t = s.x / (s.x - next.x);
      for (var i = 0; i < 3; i++) {
        let f = hermite(s.x, s.y, next.x, next.y, hs, t);
        let df = hermiteD(s.x, s.y, next.x, next.y, hs, t) * hs;
        t = clamp(t - f / min(df, -1e-9), 0.0, 1.0);
      }
      phiInf = phi + t * hs;
      let w = hermite(s.z, s.w, next.z, next.w, hs, t);
      let v = hermiteD(s.x, s.y, next.x, next.y, hs, t);
      // u(φ∞(p); p) = 0  ⇒  ∂φ∞/∂p = -w / u'
      dphidp = -w / min(v, -1e-9);
      ray.status = ESCAPED;
      break;
    }
    // Through the horizon (or, with a huge step, past the finite-φ blow-up of a
    // plunging orbit).
    if (next.x >= 1.0 || !finite(next.x)) { ray.status = CAPTURED; break; }
    s = next;
    phi += hs;
    time += dt;
    if (!land && phi >= phiC) {
      ray.nodes += 1u;
      phiC += PI;
    }
    if (land) {
      phi = phiC;
      phiC += PI;
      ray.nodes += 1u;
      let u = s.x;
      let r = 1.0 / u;
      if (r >= F.diskIn && r <= F.diskOut) {
        let cph = cos(phi);
        let sph = sin(phi);
        let rh = cph * e1 + sph * e2;
        let T = -sph * e1 + cph * e2;
        let X = rh * r;
        // Local direction of the ray in the static frame at X: radial part
        // (dr/dφ)/(r √(1 - r_s/r)) = -u' / (u √(1 - u)) per unit tangential.
        let nloc = normalize(rh * (-s.y / (u * sqrt(1.0 - u))) + T);
        var fp: f32;
        if (F.env.z > 0.5) {
          // Hit-point differential: the crossing angle moves with the plane
          // (dot(N, e(φc)) = 0), and u there moves with p and φc.
          let nT = dot(F.diskN, T);
          let sNT = select(nT, select(-1e-4, 1e-4, nT >= 0.0), abs(nT) < 1e-4);
          let nE3 = dot(F.diskN, e3);
          let dphx = -sph * nE3 * kx / sNT;
          let dphy = -sph * nE3 * ky / sNT;
          let dux = s.z * dpx + s.y * dphx;
          let duy = s.z * dpy + s.y * dphy;
          let dXx = rh * (-dux * r * r) + T * (r * dphx) + e3 * (r * sph * kx);
          let dXy = rh * (-duy * r * r) + T * (r * dphy) + e3 * (r * sph * ky);
          fp = diskFootprint(X, r, dXx, dXy);
        } else {
          fp = unlensedDiskFootprint(X, r, nloc);
        }
        addCrossing(&ray, X, r, -nloc, fp, cr.gObs, false, time);
        if (ray.trans < 0.004) { ray.status = ABSORBED; break; }
      }
    }
  }
  if (ray.status == ESCAPED) {
    let c1 = cos(phiInf);
    let s1 = sin(phiInf);
    ray.dir = c1 * e1 + s1 * e2;
    let tInf = -s1 * e1 + c1 * e2;
    ray.dDx = tInf * (dphidp * dpx) + e3 * (s1 * kx);
    ray.dDy = tInf * (dphidp * dpy) + e3 * (s1 * ky);
    ray.analytic = true;
  }
  return ray;
}

// ---- 3. Cartesian pseudo-force ---------------------------------------------------

fn accelCart(x: vec3f, h2: f32) -> vec3f {
  let r2 = dot(x, x);
  return x * (-1.5 * h2 / (r2 * r2 * sqrt(r2)));
}

fn traceCartesian(cr: CamRay, disk: bool) -> Ray {
  var ray = newRay();
  let u0 = 1.0 / F.rObs;
  let e1 = F.camPos * u0;
  let nr = dot(cr.n, e1);
  var x = F.camPos;
  // Same metric correction as in the plane: the radial part of the local
  // direction is compressed by √(1 - r_s/r) in these flat coordinates.
  var v = cr.n + e1 * (nr * (sqrt(1.0 - u0) - 1.0));
  let L = cross(x, v);
  let h2 = dot(L, L);
  let maxSteps = u32(F.integ.w);
  // Coordinate time per unit of the pseudo-time: dt/dλ = (h/b) / (1 - r_s/r),
  // and h/b = √(1 - r_s/r_obs) for this start velocity.
  let hOverB = sqrt(1.0 - u0);
  var time = 0.0;
  loop {
    if (ray.steps >= maxSteps) { break; }
    let r = length(x);
    if (r < 1.0 || !finite(r)) { ray.status = CAPTURED; break; }
    if (dot(x, v) > 0.0 && r > F.cart.y) { ray.status = ESCAPED; break; }
    let vl = length(v);
    let dt = F.cart.x * r / vl;
    let a1 = accelCart(x, h2);
    let v2 = v + (0.5 * dt) * a1;
    let a2 = accelCart(x + (0.5 * dt) * v, h2);
    let v3 = v + (0.5 * dt) * a2;
    let a3 = accelCart(x + (0.5 * dt) * v2, h2);
    let v4 = v + dt * a3;
    let a4 = accelCart(x + dt * v3, h2);
    let dTime = (dt / 6.0) * hOverB * (1.0 / (1.0 - 1.0 / r) + 2.0 / (1.0 - 1.0 / length(x + (0.5 * dt) * v))
      + 2.0 / (1.0 - 1.0 / length(x + (0.5 * dt) * v2)) + 1.0 / (1.0 - 1.0 / length(x + dt * v3)));
    let xn = x + (dt / 6.0) * (v + 2.0 * (v2 + v3) + v4);
    let vn = v + (dt / 6.0) * (a1 + 2.0 * (a2 + a3) + a4);
    ray.steps += 1u;
    let d0 = dot(x, F.diskN);
    let d1 = dot(xn, F.diskN);
    if (d0 * d1 < 0.0) { ray.nodes += 1u; }
    if (disk && F.disk.w > 0.5) {
      if (d0 * d1 < 0.0) {
        // Plane crossing on the step's cubic Hermite curve (Newton).
        let m0 = dot(v, F.diskN) * dt;
        let m1 = dot(vn, F.diskN) * dt;
        var t = d0 / (d0 - d1);
        for (var i = 0; i < 3; i++) {
          let f = hermite(d0, m0, d1, m1, 1.0, t);
          let df = hermiteD(d0, m0, d1, m1, 1.0, t);
          t = clamp(t - f / select(df, 1e-9, abs(df) < 1e-9), 0.0, 1.0);
        }
        let t2 = t * t;
        let t3 = t2 * t;
        let X = (2.0 * t3 - 3.0 * t2 + 1.0) * x + (t3 - 2.0 * t2 + t) * dt * v + (3.0 * t2 - 2.0 * t3) * xn + (t3 - t2) * dt * vn;
        let V = ((6.0 * t2 - 6.0 * t) * (x - xn) + (3.0 * t2 - 4.0 * t + 1.0) * dt * v + (3.0 * t2 - 2.0 * t) * dt * vn) / dt;
        let rr = length(X);
        if (rr >= F.diskIn && rr <= F.diskOut) {
          let rh = X / rr;
          let Vr = dot(V, rh);
          let nloc = normalize(rh * (Vr / sqrt(1.0 - 1.0 / rr)) + (V - rh * Vr));
          addCrossing(&ray, X, rr, -nloc, unlensedDiskFootprint(X, rr, nloc), cr.gObs, false, time + t * dTime);
          if (ray.trans < 0.004) { ray.status = ABSORBED; break; }
        }
      }
    }
    time += dTime;
    x = xn;
    v = vn;
  }
  var d = normalize(v);
  if (ray.status == ESCAPED) {
    // The pseudo-force falls off as r⁻⁴, so the bending still to come is
    // small but not zero: add it to first order along the straight line
    // (Born approximation), ∫ a⊥ dt = (3/2) b³ ∫ ds / (b² + s²)^(5/2) from s0.
    let s0 = dot(x, d);
    let xp = x - d * s0;
    let b = max(length(xp), 1e-6);
    let q = b * b / (s0 * s0);
    var I: f32;
    if (q < 0.01) {
      I = (1.0 - q * (5.0 / 3.0)) / (4.0 * s0 * s0 * s0 * s0);
    } else {
      let r0 = length(x);
      I = (2.0 - s0 * (2.0 * s0 * s0 + 3.0 * b * b) / (r0 * r0 * r0)) / (3.0 * b * b * b * b);
    }
    let dTheta = 1.5 * b * b * b * I;
    d = normalize(d * cos(dTheta) - (xp / b) * sin(dTheta));
  }
  ray.dir = d;
  return ray;
}
