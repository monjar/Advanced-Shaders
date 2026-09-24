// Procedural countryside for the watercolour study: rolling terrain with a
// path and a pond, a cottage and a barn, trees, rocks, a fence and distant
// hills. Everything is baked into one vertex buffer:
//   position f32x3 | normal f32x3 | colour unorm8x4 (rgb albedo, a = material) | object id f32

export const enum Material {
  Grass = 0,
  Foliage = 1,
  Wall = 2,
  Roof = 3,
  Wood = 4,
  Rock = 5,
  Water = 6,
  Dark = 7,
  Path = 8,
  Hills = 9,
}

export const VERTEX_STRIDE = 32;

type V3 = [number, number, number];

interface Part {
  positions: number[];
  normals: number[];
  indices: number[];
}

interface Placement {
  t?: V3;
  s?: V3;
  ry?: number;
}

// --- Small deterministic noise for geometry ---------------------------------

function hash2(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

function valueNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

function fbm(x: number, y: number, octaves = 4): number {
  let sum = 0;
  let amp = 0.5;
  for (let i = 0; i < octaves; i++) {
    sum += amp * valueNoise(x, y);
    x = x * 2.03 + 17.1;
    y = y * 2.03 - 5.3;
    amp *= 0.5;
  }
  return sum;
}

const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

// --- Layout --------------------------------------------------------------------

export const COTTAGE: V3 = [0, 0, 0];
const BARN: V3 = [21, 0, -13];
const POND: [number, number, number] = [-24, 15, 11]; // x, z, radius
const PATH: [number, number][] = [[0, 4], [3, 14], [1, 26], [-4, 40], [-2, 60], [4, 90]];

function distanceToPath(x: number, z: number): number {
  let best = Infinity;
  for (let i = 0; i < PATH.length - 1; i++) {
    const [ax, az] = PATH[i];
    const [bx, bz] = PATH[i + 1];
    const dx = bx - ax;
    const dz = bz - az;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz)));
    best = Math.min(best, Math.hypot(x - ax - dx * t, z - az - dz * t));
  }
  return best;
}

function rawHeight(x: number, z: number): number {
  let h = (fbm(x * 0.018, z * 0.018) - 0.5) * 7;
  h += 9 * Math.exp(-((x - 15) ** 2 + (z + 75) ** 2) / 1800);
  h += 6 * Math.exp(-((x + 60) ** 2 + (z + 30) ** 2) / 1200);
  return h;
}

const FLAT_COTTAGE = rawHeight(COTTAGE[0], COTTAGE[2]);
const FLAT_BARN = rawHeight(BARN[0], BARN[2]);
export const WATER_LEVEL = rawHeight(POND[0], POND[1]) - 0.4;

export function terrainHeight(x: number, z: number): number {
  let h = rawHeight(x, z);
  h += (FLAT_COTTAGE - h) * smooth(16, 9, Math.hypot(x - COTTAGE[0], z - COTTAGE[2]));
  h += (FLAT_BARN - h) * smooth(15, 9, Math.hypot(x - BARN[0], z - BARN[2]));
  const pond = Math.hypot(x - POND[0], z - POND[1]) / POND[2];
  h -= 2.6 * smooth(1.25, 0.2, pond);
  return h;
}

// --- Builder ---------------------------------------------------------------------

class SceneBuilder {
  private data: number[] = [];
  private colours: number[] = [];
  private ids: number[] = [];
  private indices: number[] = [];
  private nextId = 1;

  newId(): number {
    return this.nextId++;
  }

  add(part: Part, colour: V3, material: Material, place: Placement = {}, id = this.newId(), jitter = 0) {
    const [tx, ty, tz] = place.t ?? [0, 0, 0];
    const [sx, sy, sz] = place.s ?? [1, 1, 1];
    const c = Math.cos(place.ry ?? 0);
    const s = Math.sin(place.ry ?? 0);
    const base = this.data.length / 6;
    for (let i = 0; i < part.positions.length; i += 3) {
      const px = part.positions[i] * sx;
      const py = part.positions[i + 1] * sy;
      const pz = part.positions[i + 2] * sz;
      let nx = part.normals[i] / sx;
      let ny = part.normals[i + 1] / sy;
      let nz = part.normals[i + 2] / sz;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      this.data.push(px * c + pz * s + tx, py + ty, -px * s + pz * c + tz, nx * c + nz * s, ny, -nx * s + nz * c);
      // Per-vertex albedo variation so washes aren't flat.
      const v = jitter ? 1 + (hash2(px + tx * 3.1, pz + tz * 1.7 + py) - 0.5) * jitter : 1;
      this.colours.push(colour[0] * v, colour[1] * v, colour[2] * v, material);
      this.ids.push(id);
    }
    for (const index of part.indices) this.indices.push(base + index);
    return id;
  }

  /** Adds a vertex list with explicit per-vertex colours (terrain). */
  addColoured(part: Part, colours: number[], material: (i: number) => Material, id = this.newId()) {
    const base = this.data.length / 6;
    for (let i = 0; i < part.positions.length / 3; i++) {
      this.data.push(...part.positions.slice(i * 3, i * 3 + 3), ...part.normals.slice(i * 3, i * 3 + 3));
      this.colours.push(colours[i * 3], colours[i * 3 + 1], colours[i * 3 + 2], material(i));
      this.ids.push(id);
    }
    for (const index of part.indices) this.indices.push(base + index);
    return id;
  }

  build(): { vertices: ArrayBuffer; indices: Uint32Array<ArrayBuffer>; count: number } {
    const count = this.data.length / 6;
    const buffer = new ArrayBuffer(count * VERTEX_STRIDE);
    const f = new Float32Array(buffer);
    const u8 = new Uint8Array(buffer);
    for (let i = 0; i < count; i++) {
      const o = (i * VERTEX_STRIDE) / 4;
      for (let k = 0; k < 6; k++) f[o + k] = this.data[i * 6 + k];
      const b = i * VERTEX_STRIDE + 24;
      for (let k = 0; k < 3; k++) u8[b + k] = Math.round(Math.min(1, Math.max(0, this.colours[i * 4 + k])) * 255);
      u8[b + 3] = this.colours[i * 4 + 3];
      f[o + 7] = this.ids[i];
    }
    return { vertices: buffer, indices: new Uint32Array(this.indices), count: this.indices.length };
  }
}

// --- Primitives ------------------------------------------------------------------

function quad(p: Part, a: V3, b: V3, c: V3, d: V3) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
  let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const l = Math.hypot(n[0], n[1], n[2]) || 1;
  n = n.map((x) => x / l);
  const base = p.positions.length / 3;
  for (const q of [a, b, c, d]) {
    p.positions.push(...q);
    p.normals.push(...n);
  }
  p.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

function tri(p: Part, a: V3, b: V3, c: V3) {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const l = Math.hypot(n[0], n[1], n[2]) || 1;
  n = n.map((x) => x / l);
  const base = p.positions.length / 3;
  for (const q of [a, b, c]) {
    p.positions.push(...q);
    p.normals.push(...n);
  }
  p.indices.push(base, base + 1, base + 2);
}

const part = (): Part => ({ positions: [], normals: [], indices: [] });

/** Unit box: x, z in [-0.5, 0.5], y in [0, 1]. */
function box(): Part {
  const p = part();
  const x0 = -0.5, x1 = 0.5, z0 = -0.5, z1 = 0.5, y0 = 0, y1 = 1;
  quad(p, [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]);
  quad(p, [x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]);
  quad(p, [x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]);
  quad(p, [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]);
  quad(p, [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]);
  quad(p, [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]);
  return p;
}

/** Gable roof: x in [-0.5, 0.5] along the ridge, z in [-0.5, 0.5], y in [0, 1]. */
function gable(): Part {
  const p = part();
  quad(p, [-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 1, 0], [-0.5, 1, 0]);
  quad(p, [0.5, 0, -0.5], [-0.5, 0, -0.5], [-0.5, 1, 0], [0.5, 1, 0]);
  tri(p, [0.5, 0, 0.5], [0.5, 0, -0.5], [0.5, 1, 0]);
  tri(p, [-0.5, 0, -0.5], [-0.5, 0, 0.5], [-0.5, 1, 0]);
  return p;
}

/** Cylinder or cone (topRadius 0) of height 1 around the y axis. */
function cylinder(segments: number, topRadius = 1): Part {
  const p = part();
  const slope = 1 - topRadius;
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    const base = p.positions.length / 3;
    for (const [a, y, r] of [[a0, 0, 1], [a1, 0, 1], [a1, 1, topRadius], [a0, 1, topRadius]] as const) {
      p.positions.push(Math.cos(a) * r, y, Math.sin(a) * r);
      const n = [Math.cos(a), slope, Math.sin(a)];
      const l = Math.hypot(n[0], n[1], n[2]);
      p.normals.push(n[0] / l, n[1] / l, n[2] / l);
    }
    p.indices.push(base, base + 2, base + 1, base, base + 3, base + 2);
  }
  return p;
}

/** Icosphere with smooth normals, radially displaced by noise for organic blobs. */
function blob(subdivisions: number, roughness: number, seed: number): Part {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts: V3[] = [
    [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
    [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
  ].map((v) => {
    const l = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / l, v[1] / l, v[2] / l] as V3;
  });
  let faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  for (let s = 0; s < subdivisions; s++) {
    const cache = new Map<string, number>();
    const mid = (a: number, b: number) => {
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      let i = cache.get(key);
      if (i === undefined) {
        const m: V3 = [(verts[a][0] + verts[b][0]) / 2, (verts[a][1] + verts[b][1]) / 2, (verts[a][2] + verts[b][2]) / 2];
        const l = Math.hypot(m[0], m[1], m[2]);
        verts.push([m[0] / l, m[1] / l, m[2] / l]);
        i = verts.length - 1;
        cache.set(key, i);
      }
      return i;
    };
    faces = faces.flatMap(([a, b, c]) => {
      const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
      return [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]];
    });
  }
  const radius = (v: V3) => 1 + (fbm(v[0] * 1.7 + seed, v[2] * 1.7 + v[1] * 1.3 - seed, 3) - 0.5) * 2 * roughness;
  verts = verts.map((v) => {
    const r = radius(v);
    return [v[0] * r, v[1] * r, v[2] * r];
  });
  const normals = verts.map(() => [0, 0, 0]);
  for (const [a, b, c] of faces) {
    const u = verts[b].map((x, k) => x - verts[a][k]);
    const w = verts[c].map((x, k) => x - verts[a][k]);
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    for (const i of [a, b, c]) for (let k = 0; k < 3; k++) normals[i][k] += n[k];
  }
  const p = part();
  verts.forEach((v, i) => {
    const n = normals[i];
    const l = Math.hypot(n[0], n[1], n[2]) || 1;
    p.positions.push(...v);
    p.normals.push(n[0] / l, n[1] / l, n[2] / l);
  });
  for (const [a, b, c] of faces) p.indices.push(a, b, c);
  return p;
}

/** Height field over a rectangle with smooth normals. */
function heightfield(x0: number, z0: number, size: number, n: number, h: (x: number, z: number) => number): Part {
  const p = part();
  const step = size / n;
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const x = x0 + i * step;
      const z = z0 + j * step;
      p.positions.push(x, h(x, z), z);
      const e = step * 0.5;
      const nx = h(x - e, z) - h(x + e, z);
      const nz = h(x, z - e) - h(x, z + e);
      const l = Math.hypot(nx, 2 * e, nz);
      p.normals.push(nx / l, (2 * e) / l, nz / l);
    }
  }
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      p.indices.push(a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2);
    }
  }
  return p;
}

// --- Scene -----------------------------------------------------------------------

function house(b: SceneBuilder, at: V3, ry: number, size: V3, wall: V3, roof: V3, chimney: boolean) {
  const [w, h, d] = size;
  const place = (x: number, y: number, z: number): V3 => {
    const c = Math.cos(ry), s = Math.sin(ry);
    return [at[0] + x * c + z * s, at[1] + y, at[2] - x * s + z * c];
  };
  b.add(box(), wall, Material.Wall, { t: place(0, -0.5, 0), s: [w, h + 0.5, d], ry }, undefined, 0.08);
  b.add(gable(), roof, Material.Roof, { t: place(0, h, 0), s: [w + 0.8, h * 0.7, d + 1.2], ry }, undefined, 0.1);
  if (chimney) b.add(box(), [0.62, 0.32, 0.24], Material.Wall, { t: place(w * 0.28, h, d * 0.18), s: [0.9, h * 0.85, 0.9], ry });
  const dark: V3 = [0.24, 0.27, 0.33];
  b.add(box(), [0.36, 0.22, 0.14], Material.Dark, { t: place(-w * 0.18, 0, d / 2 + 0.02), s: [1.2, 2.2, 0.12], ry });
  for (const x of [-w * 0.38, w * 0.2, w * 0.38]) {
    b.add(box(), dark, Material.Dark, { t: place(x, h * 0.45, d / 2 + 0.02), s: [1.0, 1.0, 0.1], ry });
  }
  for (const x of [-w * 0.25, w * 0.25]) {
    b.add(box(), dark, Material.Dark, { t: place(x, h * 0.45, -d / 2 - 0.02), s: [1.0, 1.0, 0.1], ry });
  }
}

function deciduous(b: SceneBuilder, x: number, z: number, scale: number, leaf: V3, seed: number) {
  const y = terrainHeight(x, z);
  const trunkId = b.add(cylinder(8, 0.6), [0.38, 0.27, 0.18], Material.Wood, { t: [x, y - 0.2, z], s: [0.35 * scale, 3.4 * scale, 0.35 * scale] });
  void trunkId;
  const id = b.newId();
  const blobs: [number, number, number, number][] = [[0, 4.6, 0, 2.6], [1.3, 4.0, 0.6, 1.8], [-1.1, 4.2, -0.7, 1.9], [0.2, 5.8, -0.3, 1.7]];
  blobs.forEach(([bx, by, bz, r], i) => {
    b.add(blob(2, 0.22, seed + i * 3.7), leaf, Material.Foliage, { t: [x + bx * scale, y + by * scale, z + bz * scale], s: [r * scale, r * scale * 0.85, r * scale] }, id, 0.12);
  });
}

function conifer(b: SceneBuilder, x: number, z: number, scale: number, leaf: V3) {
  const y = terrainHeight(x, z);
  b.add(cylinder(6, 0.8), [0.35, 0.24, 0.16], Material.Wood, { t: [x, y - 0.2, z], s: [0.3 * scale, 2.2 * scale, 0.3 * scale] });
  const id = b.newId();
  for (let i = 0; i < 3; i++) {
    const r = (2.2 - i * 0.55) * scale;
    b.add(cylinder(9, 0), leaf, Material.Foliage, { t: [x, y + (1.4 + i * 1.7) * scale, z], s: [r, 3.0 * scale, r], ry: i }, id, 0.1);
  }
}

export function buildScene() {
  const b = new SceneBuilder();

  // Terrain with painted-in colour regions.
  const terrain = heightfield(-110, -110, 220, 176, terrainHeight);
  const colours: number[] = [];
  const materials: Material[] = [];
  for (let i = 0; i < terrain.positions.length; i += 3) {
    const x = terrain.positions[i];
    const y = terrain.positions[i + 1];
    const z = terrain.positions[i + 2];
    const n = fbm(x * 0.05, z * 0.05);
    let c: V3 = n > 0.5 ? [0.46, 0.6, 0.26] : [0.58, 0.64, 0.3];
    const meadow = fbm(x * 0.012 + 40, z * 0.012, 3);
    if (meadow > 0.58) c = [0.7, 0.66, 0.34];
    let m = Material.Grass;
    const path = distanceToPath(x, z);
    if (path < 1.8) {
      c = [0.78, 0.66, 0.46];
      m = Material.Path;
    }
    if (y < WATER_LEVEL + 0.3) c = [0.42, 0.44, 0.3];
    colours.push(...c);
    materials.push(m);
  }
  b.addColoured(terrain, colours, (i) => materials[i]);

  // Distant hills: a ring of low-poly relief, painted in blue-grey.
  const hills = part();
  const rings = 10;
  const segments = 96;
  for (let r = 0; r <= rings; r++) {
    for (let s = 0; s <= segments; s++) {
      const a = (s / segments) * Math.PI * 2;
      const rad = 105 + (r / rings) * 320;
      const x = Math.cos(a) * rad;
      const z = Math.sin(a) * rad;
      const rise = smooth(0, 0.35, r / rings) * smooth(1, 0.7, r / rings);
      const y = rise * (18 + 40 * fbm(Math.cos(a) * 3 + 7, Math.sin(a) * 3, 4)) - 2;
      hills.positions.push(x, y, z);
      hills.normals.push(0, 1, 0);
    }
  }
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r * (segments + 1) + s;
      hills.indices.push(a, a + 1, a + segments + 1, a + 1, a + segments + 2, a + segments + 1);
    }
  }
  // Face normals from the geometry (flat-ish shading reads as distant planes).
  const hp = hills.positions;
  const hn = hills.normals.map(() => 0);
  for (let i = 0; i < hills.indices.length; i += 3) {
    const [ia, ib, ic] = [hills.indices[i] * 3, hills.indices[i + 1] * 3, hills.indices[i + 2] * 3];
    const u = [hp[ib] - hp[ia], hp[ib + 1] - hp[ia + 1], hp[ib + 2] - hp[ia + 2]];
    const v = [hp[ic] - hp[ia], hp[ic + 1] - hp[ia + 1], hp[ic + 2] - hp[ia + 2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    for (const idx of [ia, ib, ic]) for (let k = 0; k < 3; k++) hn[idx + k] += n[k];
  }
  for (let i = 0; i < hn.length; i += 3) {
    const l = Math.hypot(hn[i], hn[i + 1], hn[i + 2]) || 1;
    const sign = hn[i + 1] < 0 ? -1 : 1;
    hills.normals[i] = (sign * hn[i]) / l;
    hills.normals[i + 1] = (sign * hn[i + 1]) / l;
    hills.normals[i + 2] = (sign * hn[i + 2]) / l;
  }
  b.add(hills, [0.4, 0.5, 0.7], Material.Hills, {}, undefined, 0.1);

  // Pond.
  const disc = part();
  for (let i = 0; i < 48; i++) {
    const a0 = (i / 48) * Math.PI * 2;
    const a1 = ((i + 1) / 48) * Math.PI * 2;
    tri(disc, [0, 0, 0], [Math.cos(a1), 0, Math.sin(a1)], [Math.cos(a0), 0, Math.sin(a0)]);
  }
  b.add(disc, [0.42, 0.6, 0.72], Material.Water, { t: [POND[0], WATER_LEVEL, POND[1]], s: [POND[2] * 1.15, 1, POND[2] * 1.15] });

  // Buildings.
  house(b, [COTTAGE[0], FLAT_COTTAGE, COTTAGE[2]], 0.15, [9, 4.2, 6], [0.9, 0.84, 0.7], [0.74, 0.34, 0.22], true);
  house(b, [BARN[0], FLAT_BARN, BARN[2]], -0.45, [10, 5, 7], [0.66, 0.24, 0.18], [0.46, 0.46, 0.5], false);

  // Trees.
  const leaves: V3[] = [[0.32, 0.5, 0.2], [0.44, 0.56, 0.2], [0.26, 0.42, 0.22], [0.82, 0.52, 0.16]];
  const trees: [number, number, number, number, 'd' | 'c'][] = [
    [-12, -8, 1.1, 0, 'd'], [-18, -2, 0.9, 1, 'd'], [12, 9, 1.0, 3, 'd'], [-9, 22, 1.2, 2, 'd'],
    [-36, 4, 1.3, 0, 'd'], [30, 12, 1.1, 1, 'd'], [8, -28, 1.0, 2, 'c'], [2, -32, 1.2, 2, 'c'],
    [-4, -27, 0.9, 2, 'c'], [36, -30, 1.3, 2, 'c'], [-30, -25, 1.1, 3, 'd'], [-45, 30, 1.2, 0, 'd'],
    [16, 34, 1.0, 1, 'd'], [40, 2, 1.1, 2, 'c'], [-20, 40, 1.3, 1, 'd'], [26, -45, 1.4, 2, 'c'],
  ];
  trees.forEach(([x, z, s, leaf, kind], i) => {
    if (kind === 'd') deciduous(b, x, z, s, leaves[leaf], i * 5.3);
    else conifer(b, x, z, s, leaves[leaf]);
  });

  // Rocks around the pond.
  for (let i = 0; i < 7; i++) {
    const a = i * 0.9 + 0.4;
    const x = POND[0] + Math.cos(a) * POND[2] * 1.2;
    const z = POND[1] + Math.sin(a) * POND[2] * 1.15;
    const r = 0.7 + hash2(i, 3) * 1.1;
    b.add(blob(1, 0.3, i * 2.1), [0.6, 0.58, 0.54], Material.Rock, { t: [x, terrainHeight(x, z), z], s: [r * 1.3, r * 0.8, r], ry: i }, undefined, 0.15);
  }

  // Fence along the path.
  const fenceId = b.newId();
  const wood: V3 = [0.58, 0.45, 0.32];
  for (let i = 0; i <= 12; i++) {
    const z = 6 + i * 2.6;
    const x = 5.5 + Math.sin(z * 0.12) * 1.5;
    b.add(box(), wood, Material.Wood, { t: [x, terrainHeight(x, z) - 0.2, z], s: [0.22, 1.4, 0.22] }, fenceId);
    if (i < 12) {
      const z2 = z + 2.6;
      const x2 = 5.5 + Math.sin(z2 * 0.12) * 1.5;
      const len = Math.hypot(x2 - x, z2 - z);
      const ry = Math.atan2(x2 - x, z2 - z) + Math.PI / 2;
      for (const hgt of [0.55, 1.0]) {
        const mx = (x + x2) / 2;
        const mz = (z + z2) / 2;
        b.add(box(), wood, Material.Wood, { t: [mx, terrainHeight(mx, mz) + hgt, mz], s: [len, 0.1, 0.08], ry }, fenceId);
      }
    }
  }

  return b.build();
}
