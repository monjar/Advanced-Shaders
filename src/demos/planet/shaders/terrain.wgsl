// Procedural terrain on the GPU: 3D gradient noise with analytic
// derivatives on per-slot lattices, and the height function. Mirrors
// terrain.ts (see the precision notes there). The including shader defines
//   fn tableSlot(s: u32) -> NoiseSlot   the reference split for slot s
// and binds SLOTS (static per-slot rotation, seed and frequency).

struct NoiseSlot {
  cell: vec4i,   // integer lattice cell of the rotated, scaled reference point
  frac: vec4f,   // its fractional part
};

struct SlotStatic {
  r0: vec4f,     // rotation rows
  r1: vec4f,
  r2: vec4f,
  seed: vec3u,
  freq: f32,     // lattice cells per metre, 2^(level - 22)
};

struct TerrainShape {
  seaBias: f32,
  mountainHeight: f32,
  detailHeight: f32,
  pad: f32,
};

const WARP_AMP: f32 = 700000.0;
const RIDGE_GAIN: f32 = 1.9;
const RIDGE_SHAPE: f32 = 1.6;

fn pcg3d(vin: vec3u) -> vec3u {
  var v = vin * 1664525u + 1013904223u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v = v ^ (v >> vec3u(16u));
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

fn hashGrad(c: vec3i, seed: vec3u) -> vec3f {
  let h = pcg3d(bitcast<vec3u>(c) + seed);
  return vec3f(h >> vec3u(8u)) * (2.0 / 16777215.0) - 1.0;
}

// Gradient noise with analytic derivatives (Quilez): (n, dn/dq).
fn perlinD(cell: vec3i, f: vec3f, seed: vec3u) -> vec4f {
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  let ga = hashGrad(cell, seed);
  let gb = hashGrad(cell + vec3i(1, 0, 0), seed);
  let gc = hashGrad(cell + vec3i(0, 1, 0), seed);
  let gd = hashGrad(cell + vec3i(1, 1, 0), seed);
  let ge = hashGrad(cell + vec3i(0, 0, 1), seed);
  let gf = hashGrad(cell + vec3i(1, 0, 1), seed);
  let gg = hashGrad(cell + vec3i(0, 1, 1), seed);
  let gh = hashGrad(cell + vec3i(1, 1, 1), seed);
  let va = dot(ga, f);
  let vb = dot(gb, f - vec3f(1.0, 0.0, 0.0));
  let vc = dot(gc, f - vec3f(0.0, 1.0, 0.0));
  let vd = dot(gd, f - vec3f(1.0, 1.0, 0.0));
  let ve = dot(ge, f - vec3f(0.0, 0.0, 1.0));
  let vf = dot(gf, f - vec3f(1.0, 0.0, 1.0));
  let vg = dot(gg, f - vec3f(0.0, 1.0, 1.0));
  let vh = dot(gh, f - vec3f(1.0, 1.0, 1.0));
  let k1 = vb - va;
  let k2 = vc - va;
  let k3 = ve - va;
  let k4 = va - vb - vc + vd;
  let k5 = va - vc - ve + vg;
  let k6 = va - vb - ve + vf;
  let k7 = -va + vb + vc - vd + ve - vf - vg + vh;
  let value = va + u.x * k1 + u.y * k2 + u.z * k3 + u.x * u.y * k4 + u.y * u.z * k5 + u.z * u.x * k6 + u.x * u.y * u.z * k7;
  var d = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga) + u.x * u.y * (ga - gb - gc + gd) +
    u.y * u.z * (ga - gc - ge + gg) + u.z * u.x * (ga - gb - ge + gf) + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh);
  d += du * vec3f(
    k1 + u.y * k4 + u.z * k6 + u.y * u.z * k7,
    k2 + u.z * k5 + u.x * k4 + u.z * u.x * k7,
    k3 + u.x * k6 + u.y * k5 + u.x * u.y * k7);
  return vec4f(value, d);
}

// One octave at a point `x` given relative to the table's reference point
// (metres). Returns (n, world-space gradient per metre).
fn slotNoise(s: u32, x: vec3f) -> vec4f {
  let st = SLOTS[s];
  let t = tableSlot(s);
  let q = t.frac.xyz + vec3f(dot(st.r0.xyz, x), dot(st.r1.xyz, x), dot(st.r2.xyz, x)) * st.freq;
  let fl = floor(q);
  let n = perlinD(t.cell.xyz + vec3i(fl), q - fl, st.seed);
  return vec4f(n.x, (st.r0.xyz * n.y + st.r1.xyz * n.z + st.r2.xyz * n.w) * st.freq);
}

fn slotLevel(s: u32) -> f32 {
  return log2(SLOTS[s].freq) + 22.0;
}

// Smoothstep and its derivative.
fn ssd(e0: f32, e1: f32, x: f32) -> vec2f {
  let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  let d = select(0.0, 6.0 * t * (1.0 - t) / (e1 - e0), t > 0.0 && t < 1.0);
  return vec2f(t * t * (3.0 - 2.0 * t), d);
}

// Octave weights of the form clamp(A - level, 0, 1): w.x = A, w.y = levels
// at or below it always count fully, w.z = highest level allowed, w.w = the
// pixel footprint in metres (0 for mesh vertices), which softens creases.
fn octaveWeight(level: f32, w: vec4f) -> f32 {
  if (level > w.z) { return 0.0; }
  if (level <= w.y) { return 1.0; }
  return clamp(w.x - level, 0.0, 1.0);
}

struct Terrain {
  h: f32,          // metres above sea level (weights w)
  h2: f32,         // the same with weights w2 (the parent LOD, for morphing)
  grad: vec3f,     // dh/dx in world space (weights w)
  land: f32,       // continent value minus sea bias: < 0 ocean
  mask: f32,       // mountain-range mask
  ridge: f32,      // ridged multifractal value
};

// Terrain at a point on the sphere given relative to the table's
// reference. `w` weights the octaves of h and grad, `w2` those of h2.
fn terrainEval(x: vec3f, shape: TerrainShape, w: vec4f, w2: vec4f) -> Terrain {
  var t: Terrain;
  // Domain warp at continent scale, with its Jacobian for the chain rule.
  var warp = vec3f(0.0);
  var gwx = vec3f(0.0);
  var gwy = vec3f(0.0);
  var gwz = vec3f(0.0);
  let wc = 3u;
  for (var i = 0u; i < wc; i++) {
    let a = pow(0.5, f32(i + 1u)) * WARP_AMP;
    let nx = slotNoise(SLOT_WARP + i, x);
    let ny = slotNoise(SLOT_WARP + wc + i, x);
    let nz = slotNoise(SLOT_WARP + 2u * wc + i, x);
    warp += a * vec3f(nx.x, ny.x, nz.x);
    gwx += a * nx.yzw;
    gwy += a * ny.yzw;
    gwz += a * nz.yzw;
  }
  let xw = x + warp;
  var c = 0.0;
  var gcw = vec3f(0.0);
  for (var i = 0u; i < 9u; i++) {
    let n = slotNoise(SLOT_CONT + i, xw);
    let a = pow(0.5, f32(i));
    c += a * n.x;
    gcw += a * n.yzw;
  }
  let gc = gcw + gcw.x * gwx + gcw.y * gwy + gcw.z * gwz;
  let land = c - shape.seaBias;
  t.land = land;
  // Abyssal plain, continental slope, a shelf that ramps linearly to the
  // coast, and land rising with the same slope so the coast has no flat kink.
  let s1 = ssd(-0.22, -0.05, land);
  let shelf = clamp((land + 0.05) / 0.05, 0.0, 1.0);
  let dShelf = select(0.0, 20.0, land > -0.05 && land < 0.0);
  let inland = clamp(land / 0.5, 0.0, 1.0);
  let dInland = select(0.0, 2.0 * (1.0 - inland) / 0.5, land > 0.0 && land < 0.5);
  var h = -4200.0 + 4070.0 * s1.x + 130.0 * shelf + 700.0 * (1.0 - (1.0 - inland) * (1.0 - inland));
  var g = (4070.0 * s1.y + 130.0 * dShelf + 700.0 * dInland) * gc;

  var mraw = 0.0;
  var gm = vec3f(0.0);
  for (var i = 0u; i < 3u; i++) {
    let n = slotNoise(SLOT_MASK + i, x);
    let a = pow(0.5, f32(i));
    mraw += a * n.x;
    gm += a * n.yzw;
  }
  // Ranges follow the zero lines of the mask noise, so they form chains.
  let em = max(0.01, w.w * SLOTS[SLOT_MASK].freq);
  let am = sqrt(mraw * mraw + em * em);
  let chain = 1.0 - am * 6.0;
  let gchain = -6.0 * (mraw / am) * gm;
  let m1 = ssd(0.25, 0.75, chain);
  let m2 = ssd(0.0, 0.15, land);
  // Full ranges along the chains, low ridged hills elsewhere on land.
  let mask = (0.07 + 0.93 * m1.x) * m2.x;
  let gmask = 0.93 * m1.y * gchain * m2.x + (0.07 + 0.93 * m1.x) * m2.y * gc;
  t.mask = mask;

  var ridge = 0.0;
  var ridge2 = 0.0;
  var gr = vec3f(0.0);
  var fb = 1.0;
  var gfb = vec3f(0.0);
  for (var i = 0u; i < 13u; i++) {
    let s = SLOT_RIDGE + i;
    let level = slotLevel(s);
    let wo = octaveWeight(level, w);
    let wo2 = octaveWeight(level, w2);
    if (max(wo, wo2) <= 0.0) { break; }
    let n = slotNoise(s, x);
    // |n| with its crease rounded over ~1.5 pixel footprints (and 0.01 for
    // the mesh, matching the CPU): a hard crease would flip normals at
    // ridge-top pixels between frames.
    let e = max(0.01, 1.5 * w.w * SLOTS[s].freq);
    let an = sqrt(n.x * n.x + e * e);
    let r = 1.0 - an;
    let gr1 = -(n.x / an) * n.yzw;
    let r2 = r * r;
    let gr2 = 2.0 * r * gr1;
    let a = pow(0.5, f32(i + 1u));
    ridge += wo * a * r2 * fb;
    ridge2 += wo2 * a * r2 * fb;
    gr += wo * a * (gr2 * fb + r2 * gfb);
    let nf = r2 * RIDGE_GAIN;
    gfb = select(vec3f(0.0), gr2 * RIDGE_GAIN, nf > 0.0 && nf < 1.0);
    fb = clamp(nf, 0.0, 1.0);
  }
  t.ridge = ridge;
  // Squaring the ridges deepens the valleys between them.
  let H = shape.mountainHeight * RIDGE_SHAPE;
  h += mask * H * ridge * ridge;
  g += H * (ridge * ridge * gmask + mask * 2.0 * ridge * gr);
  let h2base = h - mask * H * ridge * ridge + mask * H * ridge2 * ridge2;

  // Eroded detail: derivative-damped fBm (Quilez). The damping uses the
  // accumulated slope in each octave's lattice units, unweighted, so every
  // LOD damps identically.
  let s4 = ssd(0.0, 0.3, land);
  let damp = 60.0 + shape.detailHeight * mask + 140.0 * s4.x;
  let gdamp = shape.detailHeight * gmask + 140.0 * s4.y * gc;
  var dsum = vec3f(0.0);
  var e = 0.0;
  var e2 = 0.0;
  var ge = vec3f(0.0);
  for (var i = 0u; i < 14u; i++) {
    let s = SLOT_DETAIL + i;
    let level = slotLevel(s);
    let wo = octaveWeight(level, w);
    let wo2 = octaveWeight(level, w2);
    if (max(wo, wo2) <= 0.0) { break; }
    let n = slotNoise(s, x);
    let a = pow(0.5, f32(i));
    dsum += n.yzw / SLOTS[s].freq;
    let k = 1.0 / (1.0 + dot(dsum, dsum));
    e += wo * a * n.x * k;
    e2 += wo2 * a * n.x * k;
    ge += wo * a * n.yzw * k;
  }
  h += damp * e;
  g += e * gdamp + damp * ge;
  t.h = h;
  t.h2 = h2base + damp * e2;
  t.grad = g;
  return t;
}
