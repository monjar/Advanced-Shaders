// Procedural terrain shared by the CPU (camera ground clamp, LOD bounds) and
// the GPU (terrain.wgsl), plus the precision scheme that keeps its noise
// exact at every scale.
//
// Precision: every octave is 3D gradient noise on a cubic lattice whose
// spacing is 2^(22 - level) m. A point is split, per octave, into an
// integer lattice cell of a reference point (computed here in double
// precision) and a float offset from it: noise(ref + frac + R x f), where x
// is the (small) float position relative to the reference. On the GPU the
// reference is the chunk origin while baking vertices and the camera while
// shading, so a 6,360 km planet keeps millimetre-precise noise inputs at
// walking height. Each slot (one octave of one channel) also has its own
// rotation and hash seed; rotating the reference in double keeps the split
// exact, which is why octaves can be decorrelated by rotation at all.
//
// The same function is written twice (here and in terrain.wgsl); keep them
// in sync.

import type { Vec3 } from '../../core/math';

export const PLANET_RADIUS = 6_360_000;
/** log2 of the lattice spacing (m) at level 0. */
export const LEVEL0_LOG2 = 22;

export interface Channel {
  name: string;
  first: number; // first lattice level
  count: number;
}

// Order matters: slots are numbered channel by channel. The terrain
// channels come first (they are the only ones the chunk bake needs).
export const CHANNELS = {
  warp: { first: 1, count: 3, components: 3 },
  cont: { first: 0, count: 9, components: 1 },
  mask: { first: 2, count: 3, components: 1 },
  ridge: { first: 5, count: 13, components: 1 },
  detail: { first: 10, count: 14, components: 1 },
  moist: { first: 2, count: 5, components: 1 },
  temp: { first: 3, count: 3, components: 1 },
  cloud: { first: 1, count: 13, components: 1 },
  city: { first: 5, count: 12, components: 1 },
  wave: { first: 17, count: 6, components: 1 },
  albedo: { first: 9, count: 13, components: 1 },
} as const;
export type ChannelName = keyof typeof CHANNELS;

export const SLOT_BASE: Record<ChannelName, number> = {} as Record<ChannelName, number>;
export const SLOT_LEVEL: number[] = [];
let nSlots = 0;
for (const [name, c] of Object.entries(CHANNELS) as [ChannelName, (typeof CHANNELS)[ChannelName]][]) {
  SLOT_BASE[name] = nSlots;
  for (let comp = 0; comp < c.components; comp++) {
    for (let i = 0; i < c.count; i++) SLOT_LEVEL.push(c.first + i);
  }
  nSlots += c.count * c.components;
}
export const SLOT_COUNT = nSlots;
/** Slots the terrain height needs (warp, continents, mask, ridges, detail). */
export const TERRAIN_SLOTS = SLOT_BASE.moist;

// Deterministic per-slot rotation (random unit quaternion) and hash seed.
// Rotation entries are rounded to float32 so CPU and GPU rotate identically.
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export const SLOT_ROT: number[][] = []; // row-major 3x3 (rotation, or rotation times a stretch)
export const SLOT_SEED: [number, number, number][] = [];
{
  const cloudFirst = SLOT_BASE.cloud;
  const rnd = mulberry32(0x9e3779b9);
  for (let s = 0; s < SLOT_COUNT; s++) {
    // Shoemake's uniform random quaternion.
    const u1 = rnd(), u2 = rnd() * 2 * Math.PI, u3 = rnd() * 2 * Math.PI;
    const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
    const x = a * Math.sin(u2), y = a * Math.cos(u2), z = b * Math.sin(u3), w = b * Math.cos(u3);
    const m = [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ];
    // Cloud octaves are squashed north-south (zonal winds stretch cloud
    // bands east-west); any linear map works with the lattice split.
    if (s >= cloudFirst && s < cloudFirst + CHANNELS.cloud.count) for (const k of [1, 4, 7]) m[k] *= 2.2;
    SLOT_ROT.push(m.map(Math.fround));
    SLOT_SEED.push([(rnd() * 4294967296) >>> 0, (rnd() * 4294967296) >>> 0, (rnd() * 4294967296) >>> 0]);
  }
}

export const slotFreq = (s: number) => 2 ** (SLOT_LEVEL[s] - LEVEL0_LOG2);

/**
 * The per-slot lattice split of a reference point P (double precision):
 * rotated, scaled P = ref (integer cell) + frac. Written as
 * [ref.xyz (i32), 0, frac.xyz (f32), 0] per slot.
 */
export function writeSlotTable(p: Vec3, slots: number, i32: Int32Array, f32: Float32Array, offset: number, extra?: (s: number) => Vec3 | null) {
  for (let s = 0; s < slots; s++) {
    const m = SLOT_ROT[s];
    const f = slotFreq(s);
    let px = p[0], py = p[1], pz = p[2];
    const e = extra?.(s);
    if (e) { px += e[0]; py += e[1]; pz += e[2]; }
    const q = [
      (m[0] * px + m[1] * py + m[2] * pz) * f,
      (m[3] * px + m[4] * py + m[5] * pz) * f,
      (m[6] * px + m[7] * py + m[8] * pz) * f,
    ];
    const o = offset + s * 8;
    for (let k = 0; k < 3; k++) {
      const c = Math.floor(q[k]);
      i32[o + k] = c;
      f32[o + 4 + k] = q[k] - c;
    }
    i32[o + 3] = 0;
    f32[o + 7] = 0;
  }
}

/** WGSL constants mirroring the tables above. */
export function terrainConstantsWgsl(): string {
  const lines = [`const SLOT_COUNT: u32 = ${SLOT_COUNT}u;`, `const TERRAIN_SLOTS: u32 = ${TERRAIN_SLOTS}u;`];
  for (const [name, base] of Object.entries(SLOT_BASE)) lines.push(`const SLOT_${name.toUpperCase()}: u32 = ${base}u;`);
  return lines.join('\n');
}

/** Static slot data for the GPU: rotation (3 x vec4 rows), seed.xyz + level. */
export function slotStaticData(): ArrayBuffer {
  const buf = new ArrayBuffer(SLOT_COUNT * 64);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  for (let s = 0; s < SLOT_COUNT; s++) {
    const m = SLOT_ROT[s];
    for (let r = 0; r < 3; r++) f.set([m[r * 3], m[r * 3 + 1], m[r * 3 + 2], 0], s * 16 + r * 4);
    u.set([...SLOT_SEED[s], 0], s * 16 + 12);
    f[s * 16 + 15] = slotFreq(s);
  }
  return buf;
}

// ---- CPU evaluation ---------------------------------------------------------

function pcg3d(x: number, y: number, z: number): [number, number, number] {
  x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
  y = (Math.imul(y, 1664525) + 1013904223) >>> 0;
  z = (Math.imul(z, 1664525) + 1013904223) >>> 0;
  x = (x + Math.imul(y, z)) >>> 0;
  y = (y + Math.imul(z, x)) >>> 0;
  z = (z + Math.imul(x, y)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  y = (y ^ (y >>> 16)) >>> 0;
  z = (z ^ (z >>> 16)) >>> 0;
  x = (x + Math.imul(y, z)) >>> 0;
  y = (y + Math.imul(z, x)) >>> 0;
  z = (z + Math.imul(x, y)) >>> 0;
  return [x, y, z];
}

const G = new Float64Array(3);
function grad(cx: number, cy: number, cz: number, seed: [number, number, number]) {
  const h = pcg3d((cx + seed[0]) >>> 0, (cy + seed[1]) >>> 0, (cz + seed[2]) >>> 0);
  const k = 2 / 16777215;
  G[0] = (h[0] >>> 8) * k - 1;
  G[1] = (h[1] >>> 8) * k - 1;
  G[2] = (h[2] >>> 8) * k - 1;
  return G;
}

/** Gradient noise with derivatives (Quilez) at lattice coordinates q; returns [n, dn/dq xyz]. */
const N4 = new Float64Array(4);
function perlinD(qx: number, qy: number, qz: number, seed: [number, number, number]) {
  const ix = Math.floor(qx), iy = Math.floor(qy), iz = Math.floor(qz);
  const fx = qx - ix, fy = qy - iy, fz = qz - iz;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
  const uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
  const dux = 30 * fx * fx * (fx * (fx - 2) + 1);
  const duy = 30 * fy * fy * (fy * (fy - 2) + 1);
  const duz = 30 * fz * fz * (fz * (fz - 2) + 1);
  const g = (a: number, b: number, c: number) => {
    const v = grad(ix + a, iy + b, iz + c, seed);
    return [v[0], v[1], v[2], v[0] * (fx - a) + v[1] * (fy - b) + v[2] * (fz - c)];
  };
  const A = g(0, 0, 0), B = g(1, 0, 0), C = g(0, 1, 0), D = g(1, 1, 0);
  const E = g(0, 0, 1), F = g(1, 0, 1), Gg = g(0, 1, 1), H = g(1, 1, 1);
  const va = A[3], vb = B[3], vc = C[3], vd = D[3], ve = E[3], vf = F[3], vg = Gg[3], vh = H[3];
  const k1 = vb - va, k2 = vc - va, k3 = ve - va, k4 = va - vb - vc + vd;
  const k5 = va - vc - ve + vg, k6 = va - vb - ve + vf, k7 = -va + vb + vc - vd + ve - vf - vg + vh;
  N4[0] = va + ux * k1 + uy * k2 + uz * k3 + ux * uy * k4 + uy * uz * k5 + uz * ux * k6 + ux * uy * uz * k7;
  for (let i = 0; i < 3; i++) {
    const ga = A[i], gb = B[i], gc = C[i], gd = D[i], ge = E[i], gf = F[i], gg = Gg[i], gh = H[i];
    N4[1 + i] = ga + ux * (gb - ga) + uy * (gc - ga) + uz * (ge - ga) + ux * uy * (ga - gb - gc + gd) +
      uy * uz * (ga - gc - ge + gg) + uz * ux * (ga - gb - ge + gf) + ux * uy * uz * (-ga + gb + gc - gd + ge - gf - gg + gh);
  }
  N4[1] += dux * (k1 + uy * k4 + uz * k6 + uy * uz * k7);
  N4[2] += duy * (k2 + uz * k5 + ux * k4 + uz * ux * k7);
  N4[3] += duz * (k3 + ux * k6 + uy * k5 + ux * uy * k7);
  return N4;
}

/** One slot at an absolute position (m): [n, dn/dx world xyz]. */
const S4 = new Float64Array(4);
function slot(s: number, x: number, y: number, z: number) {
  const m = SLOT_ROT[s];
  const f = slotFreq(s);
  const n = perlinD((m[0] * x + m[1] * y + m[2] * z) * f, (m[3] * x + m[4] * y + m[5] * z) * f, (m[6] * x + m[7] * y + m[8] * z) * f, SLOT_SEED[s]);
  S4[0] = n[0];
  // World gradient = R^T grad * f.
  S4[1] = (m[0] * n[1] + m[3] * n[2] + m[6] * n[3]) * f;
  S4[2] = (m[1] * n[1] + m[4] * n[2] + m[7] * n[3]) * f;
  S4[3] = (m[2] * n[1] + m[5] * n[2] + m[8] * n[3]) * f;
  return S4;
}

export interface TerrainShape {
  /** Continent value offset: larger means more ocean. */
  seaBias: number;
  /** Ridged mountain height (m). */
  mountainHeight: number;
  /** Detail amplitude on mountains (m). */
  detailHeight: number;
}

export const WARP_AMP = 700_000;
export const RIDGE_GAIN = 1.9;
/** Mountains are mask * height * RIDGE_SHAPE * ridge^2 (squaring deepens the valleys). */
export const RIDGE_SHAPE = 1.6;

const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** Octave weight for vertex heights baked at LOD level L (and for the CPU at the finest level). */
export function geometryWeight(level: number, lod: number, maxLevel: number): number {
  if (level <= 8) return 1;
  if (level > maxLevel) return 0;
  return Math.min(1, Math.max(0, lod + 3.16 - level));
}

/**
 * Terrain height (m above sea level) at an absolute planet-centred point on
 * the sphere (m), with octave weights from `weight(level)`. Mirrors
 * terrainHeight() in terrain.wgsl (value only).
 */
export function terrainHeight(p: Vec3, shape: TerrainShape, weight: (level: number) => number): number {
  const [x, y, z] = p;
  // Domain warp (continent scale).
  let wx = 0, wy = 0, wz = 0;
  const wb = SLOT_BASE.warp;
  const wc = CHANNELS.warp.count;
  for (let i = 0; i < wc; i++) {
    const a = 0.5 ** (i + 1) * WARP_AMP;
    wx += a * slot(wb + i, x, y, z)[0];
    wy += a * slot(wb + wc + i, x, y, z)[0];
    wz += a * slot(wb + 2 * wc + i, x, y, z)[0];
  }
  const xw = x + wx, yw = y + wy, zw = z + wz;
  // Continents: fBm on the warped point.
  let c = 0;
  for (let i = 0; i < CHANNELS.cont.count; i++) c += 0.5 ** i * slot(SLOT_BASE.cont + i, xw, yw, zw)[0];
  const land = c - shape.seaBias;
  // Abyssal plain, continental slope, a shelf that ramps linearly to the
  // coast, and land rising with the same slope so the coast has no flat kink.
  const shelf = Math.min(Math.max((land + 0.05) / 0.05, 0), 1);
  const inland = Math.min(Math.max(land / 0.5, 0), 1);
  let h = -4200 + 4070 * smooth(-0.22, -0.05, land) + 130 * shelf + 700 * (1 - (1 - inland) * (1 - inland));
  // Mountain ranges: a low-frequency mask, only on land.
  let mraw = 0;
  for (let i = 0; i < CHANNELS.mask.count; i++) mraw += 0.5 ** i * slot(SLOT_BASE.mask + i, x, y, z)[0];
  // Ranges follow the zero lines of the mask noise, so they form chains.
  const chain = 1 - Math.sqrt(mraw * mraw + 1e-4) * 6;
  const onLand = smooth(0.0, 0.15, land);
  // Full ranges along the chains, low ridged hills elsewhere on land.
  const mask = (0.07 + 0.93 * smooth(0.25, 0.75, chain)) * onLand;
  // Ridged multifractal (Musgrave): each octave weighted by the previous.
  let ridge = 0;
  let fb = 1;
  for (let i = 0; i < CHANNELS.ridge.count; i++) {
    const s = SLOT_BASE.ridge + i;
    const w = weight(SLOT_LEVEL[s]);
    if (w <= 0) break;
    const n = slot(s, x, y, z)[0];
    const r = 1 - Math.sqrt(n * n + 1e-4);
    const r2 = r * r;
    ridge += w * 0.5 ** (i + 1) * r2 * fb;
    fb = Math.min(1, Math.max(0, r2 * RIDGE_GAIN));
  }
  h += mask * shape.mountainHeight * RIDGE_SHAPE * ridge * ridge;
  // Eroded detail: derivative-damped fBm (Quilez).
  const damp = 60 + shape.detailHeight * mask + 140 * smooth(0, 0.3, land);
  let dx = 0, dy = 0, dz = 0, e = 0;
  for (let i = 0; i < CHANNELS.detail.count; i++) {
    const s = SLOT_BASE.detail + i;
    const w = weight(SLOT_LEVEL[s]);
    if (w <= 0) break;
    const n = slot(s, x, y, z);
    // Slopes in lattice units of the first detail octave.
    const k = 2 ** (LEVEL0_LOG2 - CHANNELS.detail.first);
    dx += n[1] * k * 0.5 ** i; dy += n[2] * k * 0.5 ** i; dz += n[3] * k * 0.5 ** i;
    e += w * 0.5 ** i * n[0] / (1 + dx * dx + dy * dy + dz * dz);
  }
  h += damp * e;
  return h;
}
