// THE procedural field. Everything the magical object shows (interior energy,
// cracks, rim, scattering, heat haze, particles, the light it casts) is read
// from this one object-space field, so all effects stay coherent:
//
//   v(p, t)  flow velocity. Curl noise (Bridson et al., "Curl-Noise for
//            Procedural Fluid Flow", SIGGRAPH 2007) plus a divergence-free
//            swirl about the object's axis. The potential is multiplied by a
//            smooth ramp of the shape's distance, so the flow is tangent to
//            the surface: energy inside circulates *inside*, air outside
//            streams *around* the object.
//   E(p, t)  energy density. Filaments (the product of two ridged noises is
//            large only near the intersection curve of their zero sets) that
//            are advected by v with two-phase "flow noise" (Neyret, "Advected
//            Textures", SCA 2003): each phase is displaced along v for a
//            limited time and the two are cross-faded, so the texture flows
//            without stretching without bound.
//   C(p, t)  charge potential: slow low-frequency noise plus travelling
//            surges. It opens cracks, tints energy, powers the rim, spawns
//            particles and bends the interior rays (its gradient is stored).
//   cracks   a domain-warped Voronoi edge network at two scales (static in
//            object space, like real fractures); the field decides how far
//            each crack is open.
//
// fieldA = (v, E), fieldB = (C, ∇C) are baked every frame into 3D textures
// by bake.wgsl; the crack network is baked once into crackTex. The surface
// shader evaluates the crack network analytically (same function) for
// pixel-exact lines.

// Gradient noise with analytic derivatives (Quilez): (value, ∂/∂x, ∂/∂y, ∂/∂z).
fn noised(x: vec3f) -> vec4f {
  let i = vec3i(floor(x));
  let w = fract(x);
  let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
  let du = 30.0 * w * w * (w * (w - 2.0) + 1.0);
  let ga = hash33(i) * 2.0 - 1.0;
  let gb = hash33(i + vec3i(1, 0, 0)) * 2.0 - 1.0;
  let gc = hash33(i + vec3i(0, 1, 0)) * 2.0 - 1.0;
  let gd = hash33(i + vec3i(1, 1, 0)) * 2.0 - 1.0;
  let ge = hash33(i + vec3i(0, 0, 1)) * 2.0 - 1.0;
  let gf = hash33(i + vec3i(1, 0, 1)) * 2.0 - 1.0;
  let gg = hash33(i + vec3i(0, 1, 1)) * 2.0 - 1.0;
  let gh = hash33(i + vec3i(1, 1, 1)) * 2.0 - 1.0;
  let va = dot(ga, w);
  let vb = dot(gb, w - vec3f(1.0, 0.0, 0.0));
  let vc = dot(gc, w - vec3f(0.0, 1.0, 0.0));
  let vd = dot(gd, w - vec3f(1.0, 1.0, 0.0));
  let ve = dot(ge, w - vec3f(0.0, 0.0, 1.0));
  let vf = dot(gf, w - vec3f(1.0, 0.0, 1.0));
  let vg = dot(gg, w - vec3f(0.0, 1.0, 1.0));
  let vh = dot(gh, w - vec3f(1.0, 1.0, 1.0));
  let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
  let value = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
    + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
    + u.z * u.x * (va - vb - ve + vf) + k7 * u.x * u.y * u.z;
  let grad = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
    + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
    + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
    + du * (vec3f(vb, vc, ve) - va
      + u.yzx * vec3f(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
      + u.zxy * vec3f(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
      + u.yzx * u.zxy * k7);
  return vec4f(value, grad);
}

// Same gradient noise, value only (no derivative work).
fn noise(x: vec3f) -> f32 {
  let i = vec3i(floor(x));
  let w = fract(x);
  let u = w * w * w * (w * (w * 6.0 - 15.0) + 10.0);
  let va = dot(hash33(i) * 2.0 - 1.0, w);
  let vb = dot(hash33(i + vec3i(1, 0, 0)) * 2.0 - 1.0, w - vec3f(1.0, 0.0, 0.0));
  let vc = dot(hash33(i + vec3i(0, 1, 0)) * 2.0 - 1.0, w - vec3f(0.0, 1.0, 0.0));
  let vd = dot(hash33(i + vec3i(1, 1, 0)) * 2.0 - 1.0, w - vec3f(1.0, 1.0, 0.0));
  let ve = dot(hash33(i + vec3i(0, 0, 1)) * 2.0 - 1.0, w - vec3f(0.0, 0.0, 1.0));
  let vf = dot(hash33(i + vec3i(1, 0, 1)) * 2.0 - 1.0, w - vec3f(1.0, 0.0, 1.0));
  let vg = dot(hash33(i + vec3i(0, 1, 1)) * 2.0 - 1.0, w - vec3f(0.0, 1.0, 1.0));
  let vh = dot(hash33(i + vec3i(1, 1, 1)) * 2.0 - 1.0, w - vec3f(1.0, 1.0, 1.0));
  return mix(mix(mix(va, vb, u.x), mix(vc, vd, u.x), u.y), mix(mix(ve, vf, u.x), mix(vg, vh, u.x), u.y), u.z);
}

// Bridson's ramp: 1 away from the boundary, 0 on it, C2 continuous.
fn ramp(r: f32) -> vec2f {
  if (r >= 1.0) { return vec2f(1.0, 0.0); }
  let r2 = r * r;
  return vec2f(r * (15.0 - 10.0 * r2 + 3.0 * r2 * r2) / 8.0, (15.0 - 30.0 * r2 + 15.0 * r2 * r2) / 8.0);
}

// Ridge of a noise value: 1 on its zero set, falling to 0 at |n| = 1/k.
fn ridge(n: f32, k: f32) -> f32 {
  let r = clamp(1.0 - abs(n) * k, 0.0, 1.0);
  return r * r;
}

struct Field {
  v: vec3f,
  n: vec3f,      // advected noise values (smooth); E is formed from them per sample
  E: f32,
  C: f32,
  gradC: vec3f,
};

// Energy from the advected noise values. Two ridges multiplied are large only
// near the intersection curve of both zero sets: thin filaments. The
// nonlinearity is applied per sample rather than before baking, so the
// filaments stay sub-voxel sharp even though the smooth noise values are
// stored at a modest resolution (the same reason SDFs interpolate well).
fn energyFromNoise(n: vec3f) -> f32 {
  let k = M.energyHigh.w;
  return ridge(n.x, k) * ridge(n.y, k) * 1.4 + ridge(n.z, k * 0.8) * ridge(n.x, k * 0.35) * 0.45;
}

// Curl-noise velocity. `shape` is (signed distance, outward normal) of the object.
fn flowVelocity(p: vec3f, t: f32, shape: vec4f) -> vec3f {
  let ks = 1.15;
  let q = p * ks + vec3f(0.0, -0.11, 0.07) * t;
  let a = noised(q);
  let b = noised(q + vec3f(31.4, 17.2, 5.9));
  let c = noised(q + vec3f(-12.7, 47.1, 23.3));
  let amp = 0.3;
  // Potential ψ and its curl. The swirl ψ = (0, s(x²+z²)/2, 0) has curl s(−z, 0, x).
  let s = M.flow.z;
  let psi = vec3f(a.x, b.x, c.x) * amp + vec3f(0.0, 0.5 * s * (p.x * p.x + p.z * p.z), 0.0);
  let ga = a.yzw * ks * amp;
  let gb = b.yzw * ks * amp;
  let gc = c.yzw * ks * amp;
  let curl = vec3f(gc.y - gb.z, ga.z - gc.x, gb.x - ga.y) + s * vec3f(-p.z, 0.0, p.x);
  // Boundary: ψ' = ramp(|d|/d0) ψ, so curl ψ' = ramp · curl ψ + ∇ramp × ψ.
  let d0 = 0.22;
  let rr = ramp(abs(shape.x) / d0);
  let gradR = rr.y / d0 * sign(shape.x) * shape.yzw;
  return rr.x * curl + cross(gradR, psi);
}

fn noisePhase(p: vec3f, v: vec3f, displace: f32, offset: vec3f) -> vec3f {
  let q = (p - v * displace) * M.flow.y + offset;
  return vec3f(noise(q), noise(q * 1.03 + vec3f(19.1, 7.3, 3.7)), noise(q * 2.3 + vec3f(5.2, 1.3, 9.9)));
}

fn evalField(p: vec3f, t: f32, shape: vec4f) -> Field {
  var f: Field;
  f.v = flowVelocity(p, t, shape);

  // Two-phase flow noise: each phase is advected for PERIOD seconds along v
  // (displacement centred on zero to halve the worst stretch), then fades out
  // while the other, re-seeded phase fades in. Blending the noise *values*
  // with variance-preserving weights keeps their amplitude, so the zero sets
  // (the filaments) morph and reconnect instead of cross-fading.
  let PERIOD = 2.4;
  let cyc = t * M.flow.x / PERIOD;
  var nsum = vec3f(0.0);
  var w2 = 0.0;
  for (var k = 0; k < 2; k++) {
    let shifted = cyc + 0.5 * f32(k);
    let ph = fract(shifted);
    let w = 1.0 - abs(2.0 * ph - 1.0);
    let seed = hash33(vec3i(i32(floor(shifted)), k, 7)) * 37.0;
    nsum += w * noisePhase(p, f.v, (ph - 0.5) * PERIOD * 0.6, seed);
    w2 += w * w;
  }
  f.n = nsum / sqrt(max(w2, 1e-3));
  f.E = energyFromNoise(f.n);

  // Charge: slow noise drifting upward, with surges that sweep through the
  // object along a slowly turning axis.
  let cs = M.charge.x;
  let ct = t * M.charge.y;
  let c1 = noised(p * cs + vec3f(0.0, -ct, 0.3 * ct));
  let c2 = noised(p * cs * 2.07 + vec3f(ct * 0.7, 3.1, -ct * 0.4));
  var C = 0.5 + 0.8 * (c1.x + 0.45 * c2.x);
  var gC = 0.8 * (c1.yzw * cs + 0.45 * c2.yzw * cs * 2.07);
  let axis = normalize(vec3f(sin(t * 0.13), 1.0, cos(t * 0.13) * 0.6));
  let phase = dot(p, axis) * 2.6 - t * 1.1;
  let sw = sin(phase);
  let band = pow(max(sw, 0.0), 6.0);
  C += M.charge.z * band * 0.7;
  gC += M.charge.z * 0.7 * 6.0 * pow(max(sw, 0.0), 5.0) * cos(phase) * 2.6 * axis;
  f.C = max(C, 0.0);
  f.gradC = gC;
  return f;
}

// ---------------------------------------------------------------------------
// Crack network: exact distance to Voronoi cell borders (Quilez, "Voronoi
// edges", two passes) so lines have constant width. Returns distance in
// cell units, its gradient, the nearest feature point and a per-cell id.

struct Vor {
  d: f32,
  grad: vec3f,
  feature: vec3f,
  id: f32,
};

fn voronoiEdge(x: vec3f) -> Vor {
  let n = floor(x);
  let f = x - n;
  var mg = vec3f(0.0);
  var mr = vec3f(0.0);
  var md = 8.0;
  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let g = vec3f(f32(i), f32(j), f32(k));
        let o = hash33(vec3i(n + g)) * 0.8 + 0.1;
        let r = g + o - f;
        let d = dot(r, r);
        if (d < md) { md = d; mr = r; mg = g; }
      }
    }
  }
  var out: Vor;
  out.feature = x + mr;
  out.id = hash33(vec3i(n + mg) + vec3i(91, 17, 3)).x;
  md = 8.0;
  var nrm = vec3f(0.0, 1.0, 0.0);
  for (var k = -1; k <= 1; k++) {
    for (var j = -1; j <= 1; j++) {
      for (var i = -1; i <= 1; i++) {
        let g = mg + vec3f(f32(i), f32(j), f32(k));
        let o = hash33(vec3i(n + g)) * 0.8 + 0.1;
        let r = g + o - f;
        let dr = r - mr;
        if (dot(dr, dr) > 1e-6) {
          let dir = normalize(dr);
          let dist = dot(0.5 * (mr + r), dir);
          if (dist < md) { md = dist; nrm = dir; }
        }
      }
    }
  }
  out.d = md;
  out.grad = -nrm;
  return out;
}

struct Crack {
  d1: f32,          // distance to a primary crack (object units)
  d2: f32,          // distance to a secondary crack
  grad1: vec3f,     // ∇d1
  feature: vec3f,   // nearest primary feature point (object units)
  id: f32,          // primary cell id in [0, 1)
};

fn crackField(p: vec3f) -> Crack {
  let s = M.charge.w;
  // Low-frequency domain warp: fractures meander instead of running straight.
  let wq = p * 1.4;
  let warp = vec3f(noise(wq), noise(wq + vec3f(11.3, 2.9, 7.1)), noise(wq + vec3f(3.7, 23.1, 13.3))) * 0.16;
  let x = (p + warp) * s;
  let a = voronoiEdge(x);
  let b = voronoiEdge(x * 2.3 + vec3f(17.0, 5.0, 11.0));
  var c: Crack;
  c.d1 = a.d / s;
  c.d2 = b.d / (s * 2.3);
  c.grad1 = a.grad;
  c.feature = a.feature / s - warp;
  c.id = a.id;
  return c;
}

// How open the cracks are where the field has charge C and energy E.
fn crackOpen(C: f32, E: f32) -> f32 {
  return smoothstep(M.crack.y - 0.14, M.crack.y + 0.14, C + 0.3 * (E - 0.2));
}

// Energy emission colour and strength from the field (shared by the interior
// march, the surface, the probe and the particles).
fn energyColor(C: f32) -> vec3f {
  return mix(M.energyLow.rgb, M.energyHigh.rgb, smoothstep(0.35, 1.1, C));
}

// Filaments plus a faint diffuse glow where the charge is high, so the
// volume reads as luminous and not only as lines.
fn energyEmission(E: f32, C: f32) -> vec3f {
  return energyColor(C) * M.energyLow.w * (E * (0.2 + 1.1 * C * C) + 0.03 * C * C);
}
