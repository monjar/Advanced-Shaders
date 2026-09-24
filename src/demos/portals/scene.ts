// Three procedural locations joined only by portals. They sit far apart in
// one world space (courtyard at x ≈ 0, hangar at x ≈ 100, forest at x ≈ −100)
// and are enclosed, so without portals none can see another. Each location's
// geometry is a separate index range, so a view only draws the location its
// camera looks into.
//
// Vertex: position f32x3 | normal f32x3 | colour unorm8x4 (rgb albedo, a = material)

import type { Vec3 } from '../../core/math';
import { linkPortals, makePortal, type Portal } from './portal-math';

export const VERTEX_STRIDE = 28;

export const enum Loc { Courtyard = 0, Hangar = 1, Forest = 2 }
export const LOCATION_NAMES = ['courtyard', 'hangar', 'forest'];
export const LOCATION_CENTRES: Vec3[] = [[0, 0, 0], [100, 0, 0], [-100, 0, 0]];

/** Which location a world position belongs to (they are 100 m apart). */
/** Towards the sun, per location (the hangar has none). Keep in sync with locationSunDir() in common.wgsl. */
export const SUN_DIRECTIONS: Vec3[] = [[-0.55, 0.6, 0.58], [0, 1, 0], [0.45, 0.78, 0.43]].map((d) => {
  const l = Math.hypot(d[0], d[1], d[2]);
  return [d[0] / l, d[1] / l, d[2] / l] as Vec3;
});

export const locationOf = (p: Vec3): Loc => (p[0] > 50 ? Loc.Hangar : p[0] < -50 ? Loc.Forest : Loc.Courtyard);

// Keep in sync with scene.wgsl.
export const enum Mat {
  Plain = 0, Brick = 1, Flagstone = 2, MetalPanel = 3, HangarFloor = 4, Emissive = 5, Grass = 6,
  Bark = 7, Foliage = 8, Stone = 9, Water = 10, Wood = 11, Painted = 12, Ball = 13, Cube = 14,
}

type V3 = [number, number, number];

interface Part {
  positions: number[];
  normals: number[];
  indices: number[];
}

interface Placement {
  t?: V3;
  ry?: number;
  rx?: number;
  s?: V3;
}

// --- Primitives (y up, origin at the base centre unless stated) ------------

function box(sx: number, sy: number, sz: number, centred = false): Part {
  const part: Part = { positions: [], normals: [], indices: [] };
  const y0 = centred ? -sy / 2 : 0;
  const [hx, hz] = [sx / 2, sz / 2];
  const faces: [V3, V3, V3][] = [
    // normal, u axis, v axis
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
  ];
  const half: V3 = [hx, sy / 2, hz];
  const centre: V3 = [0, y0 + sy / 2, 0];
  for (const [n, u, v] of faces) {
    const base = part.positions.length / 3;
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      for (let k = 0; k < 3; k++) {
        part.positions.push(centre[k] + (n[k] + u[k] * a + v[k] * b) * half[k]);
        part.normals.push(n[k]);
      }
    }
    part.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return part;
}

function cylinder(r: number, h: number, seg = 24, rTop = r): Part {
  const part: Part = { positions: [], normals: [], indices: [] };
  const slope = (r - rTop) / h;
  for (let i = 0; i <= seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    const c = Math.cos(a), s = Math.sin(a);
    const nl = Math.hypot(1, slope);
    part.positions.push(c * r, 0, s * r, c * rTop, h, s * rTop);
    part.normals.push(c / nl, slope / nl, s / nl, c / nl, slope / nl, s / nl);
  }
  for (let i = 0; i < seg; i++) {
    const a = i * 2;
    part.indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
  // Caps.
  for (const [y, ny, rr] of [[h, 1, rTop], [0, -1, r]] as const) {
    if (rr <= 0) continue;
    const base = part.positions.length / 3;
    part.positions.push(0, y, 0);
    part.normals.push(0, ny, 0);
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      part.positions.push(Math.cos(a) * rr, y, Math.sin(a) * rr);
      part.normals.push(0, ny, 0);
    }
    for (let i = 0; i < seg; i++) part.indices.push(base, base + 1 + i, base + 2 + i);
  }
  return part;
}

/** Annulus extrusion: a basin wall. */
function tube(rOut: number, rIn: number, h: number, seg = 32): Part {
  const part: Part = { positions: [], normals: [], indices: [] };
  const ring = (r: number, y: number, n: (c: number, s: number) => V3) => {
    const base = part.positions.length / 3;
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      const c = Math.cos(a), s = Math.sin(a);
      part.positions.push(c * r, y, s * r);
      part.normals.push(...n(c, s));
    }
    return base;
  };
  const strip = (a: number, b: number) => {
    for (let i = 0; i < seg; i++) part.indices.push(a + i, a + i + 1, b + i, b + i, a + i + 1, b + i + 1);
  };
  strip(ring(rOut, 0, (c, s) => [c, 0, s]), ring(rOut, h, (c, s) => [c, 0, s]));
  strip(ring(rIn, h, (c, s) => [-c, 0, -s]), ring(rIn, 0, (c, s) => [-c, 0, -s]));
  strip(ring(rOut, h, () => [0, 1, 0]), ring(rIn, h, () => [0, 1, 0]));
  return part;
}

function sphere(r: number, seg = 24, rings = 14, jitter?: (x: number, y: number, z: number) => number): Part {
  const part: Part = { positions: [], normals: [], indices: [] };
  for (let j = 0; j <= rings; j++) {
    const v = (j / rings) * Math.PI;
    for (let i = 0; i <= seg; i++) {
      const u = (i / seg) * Math.PI * 2;
      const n: V3 = [Math.sin(v) * Math.cos(u), Math.cos(v), Math.sin(v) * Math.sin(u)];
      // Keep the seam closed: jitter from the direction, not the index.
      const rr = r * (jitter ? jitter(n[0], n[1], n[2]) : 1);
      part.positions.push(n[0] * rr, n[1] * rr, n[2] * rr);
      part.normals.push(...n);
    }
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < seg; i++) {
      const a = j * (seg + 1) + i;
      const b = a + seg + 1;
      part.indices.push(a, a + 1, b, b, a + 1, b + 1);
    }
  }
  if (jitter) recomputeNormals(part);
  return part;
}

function recomputeNormals(part: Part) {
  const n = new Float64Array(part.positions.length);
  const p = part.positions;
  for (let i = 0; i < part.indices.length; i += 3) {
    const [a, b, c] = [part.indices[i] * 3, part.indices[i + 1] * 3, part.indices[i + 2] * 3];
    const e1 = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]];
    const e2 = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
    const f = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    for (const v of [a, b, c]) for (let k = 0; k < 3; k++) n[v + k] += f[k];
  }
  // Average across duplicated seam vertices by position.
  const key = (i: number) => `${p[i].toFixed(4)},${p[i + 1].toFixed(4)},${p[i + 2].toFixed(4)}`;
  const acc = new Map<string, number[]>();
  for (let i = 0; i < p.length; i += 3) {
    const k = key(i);
    const s = acc.get(k) ?? [0, 0, 0];
    s[0] += n[i]; s[1] += n[i + 1]; s[2] += n[i + 2];
    acc.set(k, s);
  }
  for (let i = 0; i < p.length; i += 3) {
    const s = acc.get(key(i))!;
    const l = Math.hypot(s[0], s[1], s[2]) || 1;
    part.normals[i] = s[0] / l; part.normals[i + 1] = s[1] / l; part.normals[i + 2] = s[2] / l;
  }
}

function grid(size: number, n: number, height: (x: number, z: number) => number): Part {
  const part: Part = { positions: [], normals: [], indices: [] };
  const step = size / n;
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const x = -size / 2 + i * step;
      const z = -size / 2 + j * step;
      const e = 0.05;
      const hx = height(x + e, z) - height(x - e, z);
      const hz = height(x, z + e) - height(x, z - e);
      const nl = Math.hypot(hx, 2 * e, hz);
      part.positions.push(x, height(x, z), z);
      part.normals.push(-hx / nl, (2 * e) / nl, -hz / nl);
    }
  }
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      const c = a + n + 1;
      part.indices.push(a, c, a + 1, a + 1, c, c + 1);
    }
  }
  return part;
}

// --- Builder ------------------------------------------------------------------

function hash(n: number): number {
  const s = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return s - Math.floor(s);
}

export interface Range { first: number; count: number }

class Builder {
  data: number[] = [];
  colours: number[] = [];
  indices: number[] = [];

  /** Adds a part; triangles are re-wound so they are counter-clockwise seen from their normal side. */
  add(part: Part, colour: V3, material: Mat, place: Placement = {}) {
    const [tx, ty, tz] = place.t ?? [0, 0, 0];
    const [sx, sy, sz] = place.s ?? [1, 1, 1];
    const cy = Math.cos(place.ry ?? 0), syy = Math.sin(place.ry ?? 0);
    const cx = Math.cos(place.rx ?? 0), sxx = Math.sin(place.rx ?? 0);
    const base = this.data.length / 6;
    const xf = (x: number, y: number, z: number): V3 => {
      // rx then ry (column vectors: ry · rx · v).
      const y1 = y * cx - z * sxx;
      const z1 = y * sxx + z * cx;
      return [x * cy + z1 * syy, y1, -x * syy + z1 * cy];
    };
    for (let i = 0; i < part.positions.length; i += 3) {
      const p = xf(part.positions[i] * sx, part.positions[i + 1] * sy, part.positions[i + 2] * sz);
      let n = xf(part.normals[i] / sx, part.normals[i + 1] / sy, part.normals[i + 2] / sz);
      const nl = Math.hypot(n[0], n[1], n[2]) || 1;
      n = [n[0] / nl, n[1] / nl, n[2] / nl];
      this.data.push(p[0] + tx, p[1] + ty, p[2] + tz, n[0], n[1], n[2]);
      this.colours.push(colour[0], colour[1], colour[2], material);
    }
    const d = this.data;
    for (let i = 0; i < part.indices.length; i += 3) {
      let [a, b, c] = [part.indices[i] + base, part.indices[i + 1] + base, part.indices[i + 2] + base];
      const e1 = [d[b * 6] - d[a * 6], d[b * 6 + 1] - d[a * 6 + 1], d[b * 6 + 2] - d[a * 6 + 2]];
      const e2 = [d[c * 6] - d[a * 6], d[c * 6 + 1] - d[a * 6 + 1], d[c * 6 + 2] - d[a * 6 + 2]];
      const f = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const nx = d[a * 6 + 3] + d[b * 6 + 3] + d[c * 6 + 3];
      const ny = d[a * 6 + 4] + d[b * 6 + 4] + d[c * 6 + 4];
      const nz = d[a * 6 + 5] + d[b * 6 + 5] + d[c * 6 + 5];
      if (f[0] * nx + f[1] * ny + f[2] * nz < 0) [b, c] = [c, b];
      this.indices.push(a, b, c);
    }
  }

  begin(): number {
    return this.indices.length;
  }

  end(first: number): Range {
    return { first, count: this.indices.length - first };
  }

  finish() {
    const n = this.data.length / 6;
    const buf = new ArrayBuffer(n * VERTEX_STRIDE);
    const f = new Float32Array(buf);
    const u8 = new Uint8Array(buf);
    for (let i = 0; i < n; i++) {
      f.set(this.data.slice(i * 6, i * 6 + 6), i * 7);
      const o = i * VERTEX_STRIDE + 24;
      u8[o] = Math.round(this.colours[i * 4] * 255);
      u8[o + 1] = Math.round(this.colours[i * 4 + 1] * 255);
      u8[o + 2] = Math.round(this.colours[i * 4 + 2] * 255);
      u8[o + 3] = this.colours[i * 4 + 3];
    }
    return { vertices: new Float32Array(buf), indices: new Uint32Array(this.indices) };
  }
}

// --- Colliders (xz axis-aligned boxes; the player is a 0.3 m circle) ------

export interface Collider { x0: number; x1: number; z0: number; z1: number }

export interface Layout {
  portals: Portal[];
  colliders: Collider[][];
  /** Forest: the walkable clearing is a disc. */
  clearing: { centre: [number, number]; radius: number };
  hangarLights: { pos: Vec3; colour: Vec3 }[];
}

// Portal centres sit 1 cm in front of their wall; the ellipse (0.95 × 1.55 m
// half axes) is centred 1.35 m up, so its bottom dips 0.2 m below the floor and
// the opening is 0.93 m wide at floor level: things can walk or roll through.
const PY = 1.35;
const WALL_T = 0.8;

function buildLayout(): Layout {
  const colliders: Collider[][] = [[], [], []];
  const wallBox = (loc: number, x0: number, x1: number, z0: number, z1: number) => colliders[loc].push({ x0, x1, z0, z1 }) - 1;

  // Courtyard walls: interior x, z ∈ [−10, 10].
  const cN = wallBox(0, -10.8, 10.8, -10.8, -10);
  wallBox(0, -10.8, 10.8, 10, 10.8);
  const cW = wallBox(0, -10.8, -10, -10.8, 10.8);
  wallBox(0, 10, 10.8, -10.8, 10.8);
  // Hangar walls: interior x ∈ [88, 112], z ∈ [−9, 9].
  const hN = wallBox(1, 87.2, 112.8, -9.8, -9);
  const hS = wallBox(1, 87.2, 112.8, 9, 9.8);
  wallBox(1, 87.2, 88, -9.8, 9.8);
  const hE = wallBox(1, 112, 112.8, -9.8, 9.8);
  const bulkhead = wallBox(1, 104.4, 107.6, -1.51, -1.01);
  // Forest standing stones.
  const stoneB = wallBox(2, -104.4, -101.6, -6.87, -6.65);
  const stoneD = wallBox(2, -94.35, -93.95, 0.6, 3.4);
  // Props (boxes around their footprints).
  const prop = (loc: number, x: number, z: number, hx: number, hz = hx) => wallBox(loc, x - hx, x + hx, z - hz, z + hz);
  prop(0, 2.5, 0.5, 2.3);
  for (let x = -8; x <= 8; x += 4) prop(0, x, 7.2, 0.4);
  for (const [x, z] of [[7, -7.5], [8, -3.5], [-7.5, -6.5]]) prop(0, x, z, 0.7);
  prop(0, 8.05, 4.65, 0.9, 1.2);
  prop(0, -6, 5.4, 1.2, 0.3);
  for (const [x, z] of [[91, -4.5], [91, 4.5], [100, -5.5], [100, 5.5]]) prop(1, x, z, 0.5);
  prop(1, 89.9, -6.6, 0.8, 1.6);
  prop(1, 109.6, 7.7, 1.2, 0.9);
  prop(1, 89.5, 1, 1.5, 2.5);
  prop(2, -108, 3, 0.9);
  prop(2, -99, 9, 0.7);

  const defs = [
    // Pair A: courtyard north wall ↔ hangar south wall (classic orange / blue).
    { name: 'A courtyard', location: 0, center: [-3, PY, -9.99], yaw: 0, link: 1, colour: [1.0, 0.42, 0.08], carve: 0.5, host: cN },
    { name: 'A hangar', location: 1, center: [96, PY, 8.99], yaw: Math.PI, link: 0, colour: [0.12, 0.45, 1.0], carve: 0.5, host: hS },
    // Pair B: hangar east wall ↔ forest standing stone.
    { name: 'B hangar', location: 1, center: [111.99, PY, 3], yaw: -Math.PI / 2, link: 3, colour: [0.15, 1.0, 0.45], carve: 0.5, host: hE },
    { name: 'B forest', location: 2, center: [-103, PY, -6.64], yaw: 0, link: 2, colour: [0.15, 1.0, 0.45], carve: 0.17, host: stoneB },
    // Pair C: facing portals in the hangar bay: an infinite corridor.
    { name: 'C bay north', location: 1, center: [106, PY, -8.99], yaw: 0, link: 5, colour: [0.35, 0.9, 1.0], carve: 0.5, host: hN },
    { name: 'C bay bulkhead', location: 1, center: [106, PY, -1.52], yaw: Math.PI, link: 4, colour: [1.0, 0.7, 0.2], carve: 0.4, host: bulkhead },
    // Pair D: courtyard west wall ↔ forest standing stone.
    { name: 'D courtyard', location: 0, center: [-9.99, PY, 3], yaw: Math.PI / 2, link: 7, colour: [0.7, 0.3, 1.0], carve: 0.5, host: cW },
    { name: 'D forest', location: 2, center: [-94.36, PY, 2], yaw: -Math.PI / 2, link: 6, colour: [0.7, 0.3, 1.0], carve: 0.32, host: stoneD },
  ] as const;
  const portals = defs.map((d, i) => makePortal({ ...d, center: [...d.center] as Vec3, colour: [...d.colour] as Vec3 }, i));
  linkPortals(portals);

  const hangarLights: Layout['hangarLights'] = [];
  for (const x of [92, 100, 108]) {
    for (const z of [-4.5, 4.5]) hangarLights.push({ pos: [x, 8.0, z], colour: [0.78, 0.9, 1.0] });
  }
  return { portals, colliders, clearing: { centre: [-100, 0], radius: 12 }, hangarLights };
}

export const LAYOUT = buildLayout();

// --- Locations ------------------------------------------------------------------

const avoid = (x: number, lo: number, hi: number) => x < lo || x > hi;

function buildCourtyard(b: Builder) {
  const sand: V3 = [0.78, 0.6, 0.42];
  // Floor slab and walls (walls extend under the floor edge).
  b.add(box(21.6, 0.5, 21.6), [0.62, 0.49, 0.35], Mat.Flagstone, { t: [0, -0.5, 0] });
  b.add(box(21.6, 7, WALL_T), sand, Mat.Brick, { t: [0, -0.2, -10.4] });
  b.add(box(21.6, 7, WALL_T), sand, Mat.Brick, { t: [0, -0.2, 10.4] });
  b.add(box(WALL_T, 7, 20), sand, Mat.Brick, { t: [-10.4, -0.2, 0] });
  b.add(box(WALL_T, 7, 20), sand, Mat.Brick, { t: [10.4, -0.2, 0] });
  // Cornice.
  const cornice: V3 = [0.84, 0.7, 0.52];
  b.add(box(22.4, 0.35, 1.3), cornice, Mat.Plain, { t: [0, 6.8, -10.4] });
  b.add(box(22.4, 0.35, 1.3), cornice, Mat.Plain, { t: [0, 6.8, 10.4] });
  b.add(box(1.3, 0.35, 20.2), cornice, Mat.Plain, { t: [-10.4, 6.8, 0] });
  b.add(box(1.3, 0.35, 20.2), cornice, Mat.Plain, { t: [10.4, 6.8, 0] });
  // Pilasters, kept clear of the portals.
  for (let x = -8; x <= 8; x += 4) {
    if (avoid(x, -5, -1)) b.add(box(0.6, 6.8, 0.25), cornice, Mat.Brick, { t: [x, 0, -9.875] });
  }
  for (let z = -8; z <= 8; z += 4) {
    if (avoid(z, 1, 5)) b.add(box(0.25, 6.8, 0.6), cornice, Mat.Brick, { t: [-9.875, 0, z] });
    b.add(box(0.25, 6.8, 0.6), cornice, Mat.Brick, { t: [9.875, 0, z] });
  }
  // Portico along the south wall.
  for (let x = -8; x <= 8; x += 4) {
    b.add(box(0.8, 0.3, 0.8), cornice, Mat.Plain, { t: [x, 0, 7.2] });
    b.add(cylinder(0.26, 4.0, 20, 0.22), [0.86, 0.76, 0.6], Mat.Plain, { t: [x, 0.3, 7.2] });
    b.add(box(0.75, 0.3, 0.75), cornice, Mat.Plain, { t: [x, 4.3, 7.2] });
  }
  b.add(box(17.6, 0.5, 0.8), cornice, Mat.Brick, { t: [0, 4.6, 7.2] });
  b.add(box(17.6, 0.25, 3.4), [0.62, 0.36, 0.26], Mat.Plain, { t: [0, 5.1, 8.5] });
  // Fountain.
  const f: V3 = [2.5, 0, 0.5];
  b.add(tube(2.3, 2.0, 0.6, 40), [0.8, 0.68, 0.52], Mat.Stone, { t: f });
  b.add(cylinder(2.02, 0.42, 40), [0.1, 0.22, 0.26], Mat.Water, { t: f });
  b.add(cylinder(0.28, 1.5, 16, 0.2), [0.82, 0.7, 0.54], Mat.Stone, { t: f });
  b.add(cylinder(0.3, 0.2, 24, 0.75), [0.82, 0.7, 0.54], Mat.Stone, { t: [f[0], 1.35, f[2]] });
  // Planters with shrubs.
  for (const [x, z] of [[7, -7.5], [8, -3.5], [-7.5, -6.5]]) {
    b.add(cylinder(0.55, 0.7, 20, 0.65), [0.66, 0.3, 0.17], Mat.Plain, { t: [x, 0, z] });
    b.add(sphere(0.75, 14, 8, (x2, y2, z2) => 1 + 0.12 * Math.sin(x2 * 7 + z2 * 5 + y2 * 3)), [0.22, 0.36, 0.12], Mat.Foliage, { t: [x, 1.2, z], s: [1, 0.85, 1] });
  }
  // Crates and a bench.
  b.add(box(1, 1, 1), [0.55, 0.38, 0.22], Mat.Wood, { t: [7.8, 0, 4], ry: 0.2 });
  b.add(box(1, 1, 1), [0.55, 0.38, 0.22], Mat.Wood, { t: [8.3, 0, 5.3], ry: -0.15 });
  b.add(box(0.8, 0.8, 0.8), [0.58, 0.4, 0.24], Mat.Wood, { t: [7.9, 1, 4.3], ry: 0.5 });
  b.add(box(2.4, 0.12, 0.6), [0.8, 0.68, 0.52], Mat.Stone, { t: [-6, 0.45, 5.4] });
  b.add(box(0.3, 0.45, 0.5), [0.8, 0.68, 0.52], Mat.Stone, { t: [-7, 0, 5.4] });
  b.add(box(0.3, 0.45, 0.5), [0.8, 0.68, 0.52], Mat.Stone, { t: [-5, 0, 5.4] });
}

function buildHangar(b: Builder) {
  const metal: V3 = [0.34, 0.38, 0.42];
  const cx = 100;
  b.add(box(25.6, 0.5, 19.6), [0.13, 0.14, 0.16], Mat.HangarFloor, { t: [cx, -0.5, 0] });
  b.add(box(25.6, 9.4, WALL_T), metal, Mat.MetalPanel, { t: [cx, -0.2, -9.4] });
  b.add(box(25.6, 9.4, WALL_T), metal, Mat.MetalPanel, { t: [cx, -0.2, 9.4] });
  b.add(box(WALL_T, 9.4, 18), metal, Mat.MetalPanel, { t: [87.6, -0.2, 0] });
  b.add(box(WALL_T, 9.4, 18), metal, Mat.MetalPanel, { t: [112.4, -0.2, 0] });
  b.add(box(25.6, 0.5, 19.6), [0.2, 0.22, 0.25], Mat.MetalPanel, { t: [cx, 9, 0] });
  // Roof trusses and light panels.
  for (let x = 90; x <= 110; x += 4) b.add(box(0.35, 0.6, 18), [0.25, 0.27, 0.3], Mat.Painted, { t: [x, 8.4, 0] });
  for (const x of [92, 100, 108]) {
    for (const z of [-4.5, 4.5]) {
      b.add(box(2.6, 0.12, 1.2), [0.3, 0.3, 0.33], Mat.Painted, { t: [x, 8.3, z] });
      b.add(box(2.4, 0.04, 1.0), [0.85, 0.95, 1.0], Mat.Emissive, { t: [x, 8.27, z] });
    }
  }
  // Cyan light strips at 3.4 m, above every portal.
  const strip: V3 = [0.1, 0.85, 1.0];
  b.add(box(24, 0.08, 0.05), strip, Mat.Emissive, { t: [cx, 3.4, -8.97] });
  b.add(box(24, 0.08, 0.05), strip, Mat.Emissive, { t: [cx, 3.4, 8.97] });
  b.add(box(0.05, 0.08, 18), strip, Mat.Emissive, { t: [88.03, 3.4, 0] });
  b.add(box(0.05, 0.08, 18), strip, Mat.Emissive, { t: [111.97, 3.4, 0] });
  // Floor edge strips (orange), broken at the portals.
  const orange: V3 = [1.0, 0.45, 0.1];
  for (const [x0, x1] of [[88, 94.8], [97.2, 112]]) b.add(box(x1 - x0, 0.03, 0.06), orange, Mat.Emissive, { t: [(x0 + x1) / 2, 0, 8.96] });
  for (const [x0, x1] of [[88, 104.8], [107.2, 112]]) b.add(box(x1 - x0, 0.03, 0.06), orange, Mat.Emissive, { t: [(x0 + x1) / 2, 0, -8.96] });
  // Pillars.
  for (const [x, z] of [[91, -4.5], [91, 4.5], [100, -5.5], [100, 5.5]]) {
    b.add(box(0.8, 8.4, 0.8), [0.3, 0.33, 0.37], Mat.MetalPanel, { t: [x, 0, z] });
    b.add(box(1.0, 0.25, 1.0), [0.85, 0.6, 0.1], Mat.Painted, { t: [x, 0, z] });
  }
  // The bulkhead for the bay portal, with a frame.
  b.add(box(3.2, 4.5, 0.5), [0.3, 0.33, 0.36], Mat.MetalPanel, { t: [106, 0, -1.26] });
  b.add(box(3.5, 0.3, 0.7), [0.85, 0.6, 0.1], Mat.Painted, { t: [106, 4.5, -1.26] });
  // Crates.
  const crate: V3 = [0.22, 0.3, 0.36];
  b.add(box(1.4, 1.4, 1.4), crate, Mat.MetalPanel, { t: [89.8, 0, -7.4] });
  b.add(box(1.4, 1.4, 1.4), crate, Mat.MetalPanel, { t: [89.9, 0, -5.8], ry: 0.1 });
  b.add(box(1.2, 1.2, 1.2), [0.6, 0.35, 0.1], Mat.Painted, { t: [89.9, 1.4, -6.8], ry: -0.2 });
  b.add(box(1.4, 1.4, 1.4), crate, Mat.MetalPanel, { t: [110.2, 0, 7.6], ry: 0.3 });
  b.add(box(1.0, 1.0, 1.0), [0.6, 0.35, 0.1], Mat.Painted, { t: [108.6, 0, 7.8], ry: -0.1 });
  // A low platform against the west wall.
  b.add(box(3, 0.6, 5), [0.24, 0.26, 0.3], Mat.MetalPanel, { t: [89.5, 0, 1] });
}

// Forest ground: flat clearing, gentle rise into the trees.
export function forestHeight(x: number, z: number): number {
  const r = Math.hypot(x + 100, z);
  const rise = Math.max(0, r - 13);
  return rise * rise * 0.012 + Math.sin(x * 0.21) * Math.cos(z * 0.17) * 0.4 * Math.min(1, rise / 4);
}

function buildForest(b: Builder) {
  b.add(grid(90, 90, (x, z) => forestHeight(x - 100, z)), [0.2, 0.33, 0.12], Mat.Grass, { t: [-100, 0, 0] });
  // Pines in a ring around the clearing.
  let seed = 1;
  for (let i = 0; i < 70; i++) {
    const a = hash(seed++) * Math.PI * 2;
    const r = 14 + hash(seed++) * 26;
    const x = -100 + Math.cos(a) * r;
    const z = Math.sin(a) * r;
    const y = forestHeight(x, z) - 0.2;
    const h = 7 + hash(seed++) * 6;
    const tr = 0.22 + hash(seed++) * 0.2;
    b.add(cylinder(tr, h * 0.9, 10, tr * 0.5), [0.3, 0.2, 0.13], Mat.Bark, { t: [x, y, z] });
    const layers = 3 + Math.floor(hash(seed++) * 2);
    const tint = 0.8 + hash(seed++) * 0.4;
    for (let k = 0; k < layers; k++) {
      const f = k / layers;
      const cr = (2.4 - f * 1.5) * (0.8 + tr);
      b.add(cylinder(cr, h * 0.42, 14, 0), [0.05 * tint, 0.15 * tint, 0.07 * tint], Mat.Foliage, { t: [x, y + h * (0.28 + f * 0.5), z], ry: k });
    }
  }
  // Rocks.
  for (let i = 0; i < 16; i++) {
    const a = hash(seed++) * Math.PI * 2;
    const r = 12.5 + hash(seed++) * 14;
    const x = -100 + Math.cos(a) * r;
    const z = Math.sin(a) * r;
    const s = 0.4 + hash(seed++) * 0.9;
    const k = seed++;
    b.add(sphere(1, 12, 8, (x2, y2, z2) => 1 + 0.18 * Math.sin(x2 * 4 + k) * Math.cos(z2 * 3 + y2 * 5 - k)), [0.42, 0.43, 0.4], Mat.Stone,
      { t: [x, forestHeight(x, z) + s * 0.25, z], s: [s * 1.3, s * 0.7, s], ry: hash(seed++) * 6 });
  }
  // A fallen log.
  b.add(cylinder(0.35, 5, 14), [0.32, 0.22, 0.14], Mat.Bark, { t: [-92, 0.35, 10.5], rx: Math.PI / 2, ry: 1.1 });
  // Standing stones carrying portals B (a thin 0.22 m slab: without the
  // per-object clip plane the robot would poke out of its back) and D, plus two plain ones.
  const stone: V3 = [0.4, 0.4, 0.38];
  b.add(box(2.8, 3.9, 0.22), stone, Mat.Stone, { t: [-103, -0.3, -6.76] });
  b.add(box(0.4, 3.9, 2.8), stone, Mat.Stone, { t: [-94.15, -0.3, 2] });
  b.add(box(1.4, 2.6, 0.9), stone, Mat.Stone, { t: [-108, -0.2, 3], ry: 0.5 });
  b.add(box(1.1, 1.8, 0.8), stone, Mat.Stone, { t: [-99, -0.2, 9], ry: -0.3 });
}

// --- Moving objects ---------------------------------------------------------------

export interface ObjectMeshes {
  ball: Range;
  cube: Range;
  robot: { body: Range; head: Range; visor: Range; legL: Range; legR: Range; armL: Range; armR: Range };
}

function buildObjects(b: Builder): ObjectMeshes {
  let s = b.begin();
  b.add(sphere(0.3, 28, 16), [1, 1, 1], Mat.Ball);
  const ball = b.end(s);
  s = b.begin();
  b.add(box(0.5, 0.5, 0.5, true), [0.6, 0.62, 0.66], Mat.Cube);
  const cube = b.end(s);
  const white: V3 = [0.86, 0.87, 0.9];
  const grey: V3 = [0.25, 0.27, 0.3];
  s = b.begin();
  b.add(cylinder(0.24, 0.5, 20, 0.2), white, Mat.Painted, { t: [0, 0.55, 0] });
  b.add(sphere(0.24, 20, 10), white, Mat.Painted, { t: [0, 0.6, 0], s: [1, 0.5, 1] });
  b.add(cylinder(0.08, 0.12, 12), grey, Mat.Painted, { t: [0, 1.03, 0] });
  const body = b.end(s);
  s = b.begin();
  b.add(sphere(0.21, 20, 12), white, Mat.Painted, { t: [0, 1.3, 0] });
  const head = b.end(s);
  s = b.begin();
  b.add(sphere(0.2, 20, 10), [0.2, 0.9, 1.0], Mat.Emissive, { t: [0, 1.32, 0.08], s: [0.8, 0.35, 0.8] });
  const visor = b.end(s);
  // Limbs hang from their pivots (hip at y 0.58, shoulder at y 0.98).
  const limb = (x: number, top: number, len: number, w: number) => {
    const st = b.begin();
    b.add(box(w, len, w), grey, Mat.Painted, { t: [x, top - len, 0] });
    b.add(box(w * 1.3, 0.08, w * 1.8), white, Mat.Painted, { t: [x, top - len, w * 0.3] });
    return b.end(st);
  };
  return {
    ball, cube,
    robot: { body, head, visor, legL: limb(-0.11, 0.58, 0.56, 0.1), legR: limb(0.11, 0.58, 0.56, 0.1), armL: limb(-0.3, 0.98, 0.45, 0.07), armR: limb(0.3, 0.98, 0.45, 0.07) },
  };
}

// --- Portal meshes (portal-local; positions only) -----------------------------------

const SEG = 96;

/**
 * The opening as a thin cup: a front ellipse at z = 0, a side wall and a
 * back cap at z = −depth. From far away only the front cap is visible; when
 * the eye is closer than the near plane, the near plane cuts the cap and the
 * sides and back (inside the carved hole) still cover every ray through the
 * opening, so the portal never flickers off during a crossing.
 * The front cap comes first: the restore pass keeps the first fragment.
 */
export function portalCupMesh(depth: number) {
  const pos: number[] = [0, 0, 0];
  const idx: number[] = [];
  for (let i = 0; i < SEG; i++) {
    const a = (i / SEG) * Math.PI * 2;
    pos.push(Math.cos(a), Math.sin(a), 0);
  }
  for (let i = 0; i < SEG; i++) idx.push(0, 1 + i, 1 + ((i + 1) % SEG));
  const back = pos.length / 3;
  for (let i = 0; i < SEG; i++) {
    const a = (i / SEG) * Math.PI * 2;
    pos.push(Math.cos(a), Math.sin(a), -depth);
  }
  for (let i = 0; i < SEG; i++) {
    const a = 1 + i, b = 1 + ((i + 1) % SEG);
    const c = back + i, d = back + ((i + 1) % SEG);
    idx.push(a, c, b, b, c, d);
  }
  const centre = pos.length / 3;
  pos.push(0, 0, -depth);
  for (let i = 0; i < SEG; i++) idx.push(centre, back + ((i + 1) % SEG), back + i);
  // x, y are in units of the half axes (scaled in the vertex shader); z in metres.
  return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
}

/** A flat ring from ρ = inner to ρ = outer (ellipse units) for the rim glow. */
export function portalRingMesh(inner: number, outer: number) {
  const pos: number[] = [];
  const idx: number[] = [];
  for (let i = 0; i <= SEG; i++) {
    const a = (i / SEG) * Math.PI * 2;
    pos.push(Math.cos(a) * inner, Math.sin(a) * inner, 0, Math.cos(a) * outer, Math.sin(a) * outer, 0);
  }
  for (let i = 0; i < SEG; i++) {
    const a = i * 2;
    idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3);
  }
  return { positions: new Float32Array(pos), indices: new Uint16Array(idx) };
}

export function buildScene() {
  const b = new Builder();
  const ranges: Range[] = [];
  for (const build of [buildCourtyard, buildHangar, buildForest]) {
    const s = b.begin();
    build(b);
    ranges.push(b.end(s));
  }
  const objects = buildObjects(b);
  return { ...b.finish(), locations: ranges, objects };
}
