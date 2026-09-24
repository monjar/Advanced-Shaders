// Procedural geometry: the ruined altar environment and the three magical
// object shapes. Every object shape exists twice: as a triangle mesh (for
// rasterising the surface, so it depth-tests against the scene and occludes
// particles) and as an analytic signed distance field in shape.wgsl (for the
// interior march, thickness and particle spawning). Both are built from the
// same parameters, which are uploaded in the `Shape` uniform, so they cannot
// drift apart.

import { vec3, type Vec3 } from '../../core/math';

/** Environment vertex: position, normal, material id (float32 each). */
export const ENV_STRIDE = 7 * 4;
/** Object vertex: object-space position, normal, tangent (float32 each). */
export const OBJ_STRIDE = 9 * 4;

// Environment materials (see env.wgsl).
export const MAT = { GROUND: 0, FLOOR: 1, STONE: 2, METAL: 3, COALS: 4, RUNE: 5 } as const;

/** Object shapes. Order matches the GUI and shape.wgsl. */
export const SHAPES = { Sphere: 0, 'Crystal cluster': 1, 'Torus knot': 2 } as const;

// ---------------------------------------------------------------------------
// Shape parameters shared with the SDF.

export const SPHERE_RADIUS = 0.92;

/** A hexagonal prism with pyramidal tips at both ends (a quartz crystal). */
export interface Crystal {
  center: Vec3;
  axis: Vec3;
  /** Inradius of the hexagon. */
  radius: number;
  /** Half length of the prism body. */
  half: number;
  /** Height of each pyramidal tip. */
  tip: number;
  /** Rotation of the hexagon about the axis. */
  twist: number;
}

function tilted(yawDeg: number, tiltDeg: number): Vec3 {
  const y = (yawDeg * Math.PI) / 180;
  const t = (tiltDeg * Math.PI) / 180;
  return [Math.sin(t) * Math.cos(y), Math.cos(t), Math.sin(t) * Math.sin(y)];
}

/** A floating cluster: one tall crystal with smaller ones fused to its waist. */
export const CRYSTALS: Crystal[] = (() => {
  const list: Crystal[] = [{ center: [0, 0.04, 0], axis: [0, 1, 0], radius: 0.3, half: 0.36, tip: 0.5, twist: 0 }];
  const sats: [number, number, number, number, number][] = [
    // yaw, tilt, radius, half, tip
    [20, 52, 0.16, 0.2, 0.26],
    [110, 60, 0.13, 0.16, 0.22],
    [200, 48, 0.17, 0.22, 0.28],
    [290, 64, 0.12, 0.14, 0.2],
    [160, 140, 0.13, 0.16, 0.22],
    [330, 128, 0.15, 0.18, 0.24],
  ];
  sats.forEach(([yaw, tilt, radius, half, tip], i) => {
    const axis = tilted(yaw, tilt);
    const offset = 0.2 + half + (i % 2) * 0.03;
    list.push({ center: vec3.scale(axis, offset), axis, radius, half, tip, twist: i * 0.37 });
  });
  return list;
})();

/** (p, q) torus knot around the z axis, swept by a circular tube. */
export const KNOT = { p: 2, q: 3, R: 0.6, r: 0.27, tube: 0.18 };

export function knotPoint(t: number): Vec3 {
  const { p, q, R, r } = KNOT;
  const s = R + r * Math.cos(q * t);
  return [s * Math.cos(p * t), s * Math.sin(p * t), r * Math.sin(q * t)];
}

function knotTangent(t: number): Vec3 {
  const { p, q, R, r } = KNOT;
  const s = R + r * Math.cos(q * t);
  const ds = -r * q * Math.sin(q * t);
  return vec3.normalize([
    ds * Math.cos(p * t) - s * p * Math.sin(p * t),
    ds * Math.sin(p * t) + s * p * Math.cos(p * t),
    r * q * Math.cos(q * t),
  ]);
}

/** Crystal frame: axis `a`, and two perpendicular unit vectors `b`, `c`. */
export function crystalFrame(c: Crystal): { a: Vec3; b: Vec3; c: Vec3 } {
  const a = vec3.normalize(c.axis);
  const ref: Vec3 = Math.abs(a[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  let b = vec3.normalize(vec3.cross(ref, a));
  let cc = vec3.cross(a, b);
  const cs = Math.cos(c.twist);
  const sn = Math.sin(c.twist);
  const b2 = vec3.add(vec3.scale(b, cs), vec3.scale(cc, sn));
  cc = vec3.cross(a, b2);
  b = b2;
  return { a, b, c: cc };
}

/** Packs the Shape uniform (see shape.wgsl). */
export function shapeUniform(kind: number, box: number): Float32Array<ArrayBuffer> {
  const out = new Float32Array(8 + 12 * 8);
  out.set([kind, CRYSTALS.length, SPHERE_RADIUS, box], 0);
  out.set([KNOT.R, KNOT.r, KNOT.tube, 0], 4);
  CRYSTALS.forEach((c, i) => {
    const f = crystalFrame(c);
    const o = 8 + i * 12;
    out.set([...c.center, c.radius], o);
    out.set([...f.a, c.half], o + 4);
    out.set([...f.b, c.tip], o + 8);
  });
  return out;
}

// ---------------------------------------------------------------------------
// Mesh building.

class MeshBuilder {
  data: number[] = [];
  indices: number[] = [];
  constructor(private stride: number) {}

  get count() {
    return this.data.length / this.stride;
  }

  vertex(values: number[]): number {
    this.data.push(...values);
    return this.count - 1;
  }

  /** Adds a triangle, flipping it if needed so it winds CCW around its vertex normals. */
  tri(a: number, b: number, c: number) {
    const s = this.stride;
    const d = this.data;
    const P = (i: number): Vec3 => [d[i * s], d[i * s + 1], d[i * s + 2]];
    const N = (i: number): Vec3 => [d[i * s + 3], d[i * s + 4], d[i * s + 5]];
    const n = vec3.add(vec3.add(N(a), N(b)), N(c));
    const cr = vec3.cross(vec3.sub(P(b), P(a)), vec3.sub(P(c), P(a)));
    if (vec3.dot(cr, n) < 0) this.indices.push(a, c, b);
    else this.indices.push(a, b, c);
  }

  quad(a: number, b: number, c: number, d: number) {
    this.tri(a, b, c);
    this.tri(a, c, d);
  }

  build() {
    return { vertices: new Float32Array(this.data), indices: new Uint32Array(this.indices), count: this.indices.length };
  }
}

// ---------------------------------------------------------------------------
// Object meshes (object space, tangents for anisotropic brushing).

function sphereMesh() {
  const m = new MeshBuilder(9);
  const nu = 160;
  const nv = 96;
  for (let j = 0; j <= nv; j++) {
    const th = (j / nv) * Math.PI;
    for (let i = 0; i <= nu; i++) {
      const ph = (i / nu) * Math.PI * 2;
      const n: Vec3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
      const t: Vec3 = [-Math.sin(ph), 0, Math.cos(ph)];
      m.vertex([...vec3.scale(n, SPHERE_RADIUS), ...n, ...t]);
    }
  }
  for (let j = 0; j < nv; j++) {
    for (let i = 0; i < nu; i++) {
      const a = j * (nu + 1) + i;
      m.quad(a, a + 1, a + nu + 2, a + nu + 1);
    }
  }
  return m.build();
}

function crystalMesh() {
  const m = new MeshBuilder(9);
  for (const c of CRYSTALS) {
    const f = crystalFrame(c);
    const R = c.radius / Math.cos(Math.PI / 6); // circumradius
    const W = (x: number, y: number, z: number): Vec3 =>
      vec3.add(c.center, vec3.add(vec3.scale(f.b, x), vec3.add(vec3.scale(f.a, y), vec3.scale(f.c, z))));
    const corner = (k: number, y: number) => {
      const ang = (k * Math.PI) / 3;
      return W(R * Math.cos(ang), y, R * Math.sin(ang));
    };
    const top = W(0, c.half + c.tip, 0);
    const bottom = W(0, -c.half - c.tip, 0);
    const flat = (pts: Vec3[]) => {
      const n = vec3.normalize(vec3.cross(vec3.sub(pts[1], pts[0]), vec3.sub(pts[2], pts[0])));
      const out = vec3.dot(n, vec3.sub(pts[0], c.center)) < 0 ? vec3.scale(n, -1) : n;
      return pts.map((p) => m.vertex([...p, ...out, ...f.a]));
    };
    for (let k = 0; k < 6; k++) {
      const a0 = corner(k, -c.half);
      const a1 = corner(k + 1, -c.half);
      const b0 = corner(k, c.half);
      const b1 = corner(k + 1, c.half);
      const [i0, i1, i2, i3] = flat([a0, a1, b1, b0]);
      m.quad(i0, i1, i2, i3);
      const t = flat([b0, b1, top]);
      m.tri(t[0], t[1], t[2]);
      const u = flat([a0, a1, bottom]);
      m.tri(u[0], u[1], u[2]);
    }
  }
  return m.build();
}

function knotMesh() {
  const m = new MeshBuilder(9);
  const nu = 520;
  const nv = 40;
  for (let i = 0; i <= nu; i++) {
    const t = (i / nu) * Math.PI * 2;
    const c = knotPoint(t);
    const T = knotTangent(t);
    // The radial direction from the knot's centre is never parallel to its
    // tangent, so it gives a seamless frame without parallel transport.
    const radial = vec3.normalize(c);
    const Nn = vec3.normalize(vec3.sub(radial, vec3.scale(T, vec3.dot(radial, T))));
    const B = vec3.cross(T, Nn);
    for (let j = 0; j <= nv; j++) {
      const a = (j / nv) * Math.PI * 2;
      const n = vec3.add(vec3.scale(Nn, Math.cos(a)), vec3.scale(B, Math.sin(a)));
      m.vertex([...vec3.add(c, vec3.scale(n, KNOT.tube)), ...n, ...T]);
    }
  }
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const a = i * (nv + 1) + j;
      m.quad(a, a + nv + 1, a + nv + 2, a + 1);
    }
  }
  return m.build();
}

export function objectMesh(shape: number) {
  if (shape === 1) return crystalMesh();
  if (shape === 2) return knotMesh();
  return sphereMesh();
}

// ---------------------------------------------------------------------------
// Environment (world space): a ruined open-air temple at night.

type Rot = [Vec3, Vec3, Vec3]; // columns: local x, y, z axes in world space

const rotY = (a: number): Rot => [[Math.cos(a), 0, -Math.sin(a)], [0, 1, 0], [Math.sin(a), 0, Math.cos(a)]];
const mul = (r: Rot, v: Vec3): Vec3 =>
  vec3.add(vec3.add(vec3.scale(r[0], v[0]), vec3.scale(r[1], v[1])), vec3.scale(r[2], v[2]));
function rotAxis(axis: Vec3, a: number): Rot {
  const [x, y, z] = vec3.normalize(axis);
  const c = Math.cos(a);
  const s = Math.sin(a);
  const t = 1 - c;
  return [
    [t * x * x + c, t * x * y + s * z, t * x * z - s * y],
    [t * x * y - s * z, t * y * y + c, t * y * z + s * x],
    [t * x * z + s * y, t * y * z - s * x, t * z * z + c],
  ];
}

class EnvBuilder extends MeshBuilder {
  constructor() {
    super(7);
  }

  v(p: Vec3, n: Vec3, mat: number) {
    return this.vertex([...p, ...n, mat]);
  }

  /** Box centred at `c` with full size `s`, rotated by `r`. */
  box(c: Vec3, s: Vec3, mat: number, r: Rot = rotY(0)) {
    const h = vec3.scale(s, 0.5);
    for (let axis = 0; axis < 3; axis++) {
      for (const sign of [-1, 1]) {
        const n: Vec3 = [0, 0, 0];
        n[axis] = sign;
        const u = (axis + 1) % 3;
        const w = (axis + 2) % 3;
        const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => {
          const p: Vec3 = [0, 0, 0];
          p[axis] = sign * h[axis];
          p[u] = a * h[u];
          p[w] = b * h[w];
          return this.v(vec3.add(c, mul(r, p)), mul(r, n), mat);
        });
        this.quad(corners[0], corners[1], corners[2], corners[3]);
      }
    }
  }

  /**
   * Cylinder along local y from 0 to `height`, based at `c`. `flat` gives
   * faceted sides (octagonal pedestal), `sideMat`/`capMat` pick materials.
   */
  cylinder(
    c: Vec3, radius: number, height: number, segments: number,
    opt: { side?: number; top?: number; bottom?: number; flat?: boolean; r?: Rot; topRadius?: number },
  ) {
    const r = opt.r ?? rotY(0);
    const r1 = opt.topRadius ?? radius;
    const P = (x: number, y: number, z: number) => vec3.add(c, mul(r, [x, y, z]));
    const ring = (k: number, rad: number, y: number): Vec3 => {
      const a = (k / segments) * Math.PI * 2;
      return [rad * Math.cos(a), y, rad * Math.sin(a)];
    };
    if (opt.side !== undefined) {
      const slope = (radius - r1) / height;
      for (let k = 0; k < segments; k++) {
        const a0 = (k / segments) * Math.PI * 2;
        const a1 = ((k + 1) / segments) * Math.PI * 2;
        const nrm = (a: number): Vec3 => mul(r, vec3.normalize([Math.cos(a), slope, Math.sin(a)]));
        const n0 = opt.flat ? nrm((a0 + a1) / 2) : nrm(a0);
        const n1 = opt.flat ? n0 : nrm(a1);
        const b0 = ring(k, radius, 0);
        const b1 = ring(k + 1, radius, 0);
        const t0 = ring(k, r1, height);
        const t1 = ring(k + 1, r1, height);
        this.quad(
          this.v(P(...b0), n0, opt.side), this.v(P(...b1), n1, opt.side),
          this.v(P(...t1), n1, opt.side), this.v(P(...t0), n0, opt.side),
        );
      }
    }
    for (const [mat, y, rad, ny] of [[opt.top, height, r1, 1], [opt.bottom, 0, radius, -1]] as const) {
      if (mat === undefined) continue;
      const n = mul(r, [0, ny, 0]);
      const centre = this.v(P(0, y, 0), n, mat);
      for (let k = 0; k < segments; k++) {
        this.tri(centre, this.v(P(...ring(k, rad, y)), n, mat), this.v(P(...ring(k + 1, rad, y)), n, mat));
      }
    }
  }

  /** Flat annulus (the top of a step) between two radii at height y. */
  annulus(y: number, inner: number, outer: number, segments: number, mat: number) {
    const n: Vec3 = [0, 1, 0];
    for (let k = 0; k < segments; k++) {
      const a0 = (k / segments) * Math.PI * 2;
      const a1 = ((k + 1) / segments) * Math.PI * 2;
      const q = (rad: number, a: number): Vec3 => [rad * Math.cos(a), y, rad * Math.sin(a)];
      this.quad(this.v(q(inner, a0), n, mat), this.v(q(outer, a0), n, mat), this.v(q(outer, a1), n, mat), this.v(q(inner, a1), n, mat));
    }
  }
}

/** Height of the pedestal top and resting height of the object's centre. */
export const PEDESTAL_TOP = 1.39;
export const OBJECT_HEIGHT = 2.45;
export const BRAZIERS: Vec3[] = [[-2.9, 1.18, -1.9], [3.0, 1.18, -1.6]];

export function buildEnvironment() {
  const b = new EnvBuilder();
  const S = MAT.STONE;

  // Ground and the round platform.
  b.cylinder([0, -0.62, 0], 90, 0.01, 72, { top: MAT.GROUND });
  b.cylinder([0, -0.6, 0], 9.4, 0.6, 96, { side: S });
  b.annulus(0, 2.7, 9.4, 96, MAT.FLOOR);
  b.cylinder([0, -0.95, 0], 10.2, 0.35, 96, { side: S, top: S });

  // Altar steps and pedestal.
  b.cylinder([0, 0, 0], 2.7, 0.2, 96, { side: S });
  b.annulus(0.2, 2.15, 2.7, 96, S);
  b.cylinder([0, 0.2, 0], 2.15, 0.2, 96, { side: S, top: MAT.RUNE });
  b.cylinder([0, 0.4, 0], 0.66, 0.2, 8, { side: S, top: S, flat: true });
  b.cylinder([0, 0.6, 0], 0.44, 0.66, 8, { side: S, flat: true });
  b.cylinder([0, 1.26, 0], 0.56, 0.13, 8, { side: S, top: S, bottom: S, flat: true, topRadius: 0.68 });

  // Ring of pillars, some broken, some joined by lintels.
  const ringR = 6.8;
  const count = 10;
  const heights = [5.2, 5.2, 5.2, 2.1, 5.2, 5.2, 5.2, 3.3, 5.2, 5.2];
  const pillarTop = (i: number): Vec3 => {
    const a = ((i + 0.5) / count) * Math.PI * 2;
    return [ringR * Math.cos(a), heights[i], ringR * Math.sin(a)];
  };
  for (let i = 0; i < count; i++) {
    const [x, h, z] = pillarTop(i);
    const a = Math.atan2(z, x);
    b.box([x, 0.18, z], [1.15, 0.36, 1.15], S, rotY(-a));
    b.cylinder([x, 0.36, z], 0.44, h - 0.36 - (h > 5 ? 0.3 : 0), 24, { side: S, top: h > 5 ? undefined : S, topRadius: 0.4 });
    if (h > 5) b.box([x, h - 0.15, z], [1.1, 0.3, 1.1], S, rotY(-a));
  }
  for (const [i, j] of [[0, 1], [1, 2], [4, 5], [5, 6], [8, 9]]) {
    const p = pillarTop(i);
    const q = pillarTop(j);
    const mid = vec3.scale(vec3.add(p, q), 0.5);
    const d = vec3.sub(q, p);
    const len = Math.hypot(d[0], d[2]);
    b.box([mid[0], p[1] + 0.25, mid[2]], [len + 1.0, 0.5, 0.85], S, rotY(-Math.atan2(d[2], d[0])));
  }
  // A fallen drum of pillar 3 and scattered blocks.
  {
    const [x, , z] = pillarTop(3);
    const along = vec3.normalize([-z, 0, x]);
    b.cylinder([x + along[0] * 0.8 + 0.6, 0.42, z + along[2] * 0.8], 0.42, 2.4, 24, {
      side: S, top: S, bottom: S, r: rotAxis([along[2], 0, -along[0]], Math.PI / 2 - 0.08),
    });
  }
  const rng = mulberry(7);
  for (let k = 0; k < 14; k++) {
    const a = rng() * Math.PI * 2;
    const rad = 3.3 + rng() * 5.2;
    const s = 0.2 + rng() * 0.45;
    b.box([rad * Math.cos(a), s * 0.35, rad * Math.sin(a)], [s * 1.6, s * 0.8, s], S, rotAxis([rng() - 0.5, 1, rng() - 0.5], rng() * 3));
  }

  // Two braziers: tripod column, bowl, glowing coals.
  for (const [x, y, z] of BRAZIERS) {
    b.cylinder([x, 0, z], 0.2, 0.1, 12, { side: MAT.METAL, top: MAT.METAL });
    b.cylinder([x, 0.1, z], 0.07, y - 0.34, 10, { side: MAT.METAL });
    b.cylinder([x, y - 0.26, z], 0.12, 0.26, 16, { side: MAT.METAL, bottom: MAT.METAL, topRadius: 0.42 });
    b.cylinder([x, y - 0.06, z], 0.4, 0.02, 16, { top: MAT.COALS });
  }
  return b.build();
}

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
