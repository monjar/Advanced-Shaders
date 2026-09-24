// Portal transforms, oblique near-plane clipping for the repo's reversed-Z
// infinite projection, and screen-space bounds of a portal opening.
// Pure functions (no GPU): they are shared by the renderer, the camera
// teleport and the objects that cross portals.

import { mat4, vec3, type Mat4, type Vec3 } from '../../core/math';

export type Vec4 = [number, number, number, number];

export const m4 = {
  translation(t: Vec3): Mat4 {
    const m = mat4.identity();
    m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
    return m;
  },
  /** Rotation about +Y: x' = x cos a + z sin a, z' = −x sin a + z cos a. */
  rotationY(a: number): Mat4 {
    const c = Math.cos(a), s = Math.sin(a);
    const m = mat4.identity();
    m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
    return m;
  },
  rotationX(a: number): Mat4 {
    const c = Math.cos(a), s = Math.sin(a);
    const m = mat4.identity();
    m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
    return m;
  },
  /** Rotation about an arbitrary unit axis (Rodrigues). */
  rotationAxis(axis: Vec3, a: number): Mat4 {
    const [x, y, z] = vec3.normalize(axis);
    const c = Math.cos(a), s = Math.sin(a), t = 1 - c;
    const m = mat4.identity();
    m[0] = t * x * x + c; m[1] = t * x * y + s * z; m[2] = t * x * z - s * y;
    m[4] = t * x * y - s * z; m[5] = t * y * y + c; m[6] = t * y * z + s * x;
    m[8] = t * x * z + s * y; m[9] = t * y * z - s * x; m[10] = t * z * z + c;
    return m;
  },
  mul(...ms: Mat4[]): Mat4 {
    return ms.reduce((a, b) => mat4.multiply(a, b));
  },
  point(m: Mat4, p: Vec3): Vec3 {
    const r = mat4.transformPoint(m, [p[0], p[1], p[2], 1]);
    return [r[0], r[1], r[2]];
  },
  dir(m: Mat4, d: Vec3): Vec3 {
    const r = mat4.transformPoint(m, [d[0], d[1], d[2], 0]);
    return [r[0], r[1], r[2]];
  },
};

/** Elliptical opening, in metres (portal-local x = right, y = up, z = normal into the room). */
export const PORTAL_HALF_WIDTH = 0.95;
export const PORTAL_HALF_HEIGHT = 1.55;

export interface Portal {
  index: number;
  name: string;
  location: number;
  center: Vec3;
  yaw: number;
  link: number;
  colour: Vec3;
  /** Depth of the hole carved into the host wall behind the opening (< host thickness). */
  carve: number;
  /** Collider (in its location) the portal is mounted on; ignored while walking through. */
  host: number;
  // Derived:
  world: Mat4;
  inv: Mat4;
  normal: Vec3;
  /** World plane (n, d) with n·p + d > 0 in front of the portal (inside its room). */
  plane: Vec4;
  /** Maps world space in front of this portal to world space in front of the linked one. */
  toLinked: Mat4;
  /** Yaw change for anything passing through (camera, objects). */
  yawDelta: number;
  outline: Vec3[];
}

export function makePortal(p: Pick<Portal, 'name' | 'location' | 'center' | 'yaw' | 'link' | 'colour' | 'carve' | 'host'>, index: number): Portal {
  const world = m4.mul(m4.translation(p.center), m4.rotationY(p.yaw));
  const normal: Vec3 = [Math.sin(p.yaw), 0, Math.cos(p.yaw)];
  const outline: Vec3[] = [];
  for (let i = 0; i < 48; i++) {
    const a = (i / 48) * Math.PI * 2;
    outline.push(m4.point(world, [Math.cos(a) * PORTAL_HALF_WIDTH, Math.sin(a) * PORTAL_HALF_HEIGHT, 0]));
  }
  return {
    ...p, index, world, inv: mat4.invert(world), normal,
    plane: [normal[0], normal[1], normal[2], -vec3.dot(normal, p.center)],
    toLinked: mat4.identity(), yawDelta: 0, outline,
  };
}

/**
 * Pair transform: destination · rotateY(180°) · source⁻¹. A point in front of
 * the source (local z > 0) lands behind the destination (local z < 0), which
 * is where the virtual camera must stand to look out of the destination.
 */
export function linkPortals(portals: Portal[]) {
  for (const p of portals) {
    const d = portals[p.link];
    p.toLinked = m4.mul(d.world, m4.rotationY(Math.PI), p.inv);
    const dy = d.yaw + Math.PI - p.yaw;
    p.yawDelta = Math.atan2(Math.sin(dy), Math.cos(dy));
  }
}

export const planeDist = (pl: Vec4, p: Vec3) => pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3];

/** Is the portal-local point inside the (optionally shrunk) elliptical opening? */
export function insideOpening(local: Vec3, margin = 0): boolean {
  const x = local[0] / (PORTAL_HALF_WIDTH - margin);
  const y = local[1] / (PORTAL_HALF_HEIGHT - margin);
  return x * x + y * y < 1;
}

/**
 * Oblique near-plane clipping (Lengyel 2005), derived for WebGPU clip space
 * (0 ≤ z ≤ w) and the reversed-Z infinite projection of core/math.ts:
 *
 *   rows of P:  r0 = (f/a,0,0,0)  r1 = (0,f,0,0)  r2 = (0,0,0,n)  r3 = (0,0,−1,0)
 *   near plane: z ≤ w  ⇔ (r3 − r2)·v ≥ 0        far plane: z ≥ 0 ⇔ r2·v ≥ 0
 *
 * Keep r0, r1, r3 (so x, y and w, i.e. the image, are unchanged) and replace
 * r2 with r3 − a·C, where C is the clip plane in view space with C·v ≥ 0 on
 * the visible side and a > 0. The new near plane is (r3 − r2')·v = a·C·v ≥ 0:
 * exactly the portal plane, mapped to depth 1. Depth becomes
 *
 *   depth(v) = 1 − a·(C·v)/(−v_z)
 *
 * and the new far plane r2'·v ≥ 0 is tilted. Along a view ray v = t·d
 * (d_z = −1) with the camera behind the plane (C_w < 0), (C·v)/t rises
 * monotonically towards C·d, so nothing visible is lost if
 * a ≤ 1 / max_d(C·d) over the frustum. The maximum of a linear function over
 * the frustum section is at a corner d = (±tanX, ±tanY, −1):
 *
 *   a = 1 / (|Cx|·tanX + |Cy|·tanY − Cz)
 *
 * With that a the far plane touches the frustum only at infinity along one
 * corner ray, so depth stays in [0, 1] for everything beyond the portal and
 * the reversed-Z convention (clear 0, compare 'greater') is kept.
 *
 * Returns null when no view direction reaches the visible side at all.
 */
export function obliqueProjection(proj: Mat4, clipView: Vec4): Mat4 | null {
  const tanX = 1 / proj[0];
  const tanY = 1 / proj[5];
  const [cx, cy, cz, cw] = clipView;
  const maxDot = Math.abs(cx) * tanX + Math.abs(cy) * tanY - cz;
  if (maxDot <= 1e-6) return null;
  const a = 1 / maxDot;
  const m = new Float32Array(proj);
  // Column-major: row 2 lives at indices 2, 6, 10, 14. r3 = (0, 0, −1, 0).
  m[2] = -a * cx;
  m[6] = -a * cy;
  m[10] = -1 - a * cz;
  m[14] = -a * cw;
  return m;
}

/** World-space plane → view-space plane for a rigid view matrix. */
export function planeToView(view: Mat4, pl: Vec4): Vec4 {
  const n = m4.dir(view, [pl[0], pl[1], pl[2]]);
  // A point on the plane, moved into view space.
  const p = m4.point(view, [-pl[0] * pl[3], -pl[1] * pl[3], -pl[2] * pl[3]]);
  return [n[0], n[1], n[2], -vec3.dot(n, p)];
}

export type Rect = [number, number, number, number]; // x0, y0, x1, y1 in pixels

/**
 * Pixel bounds of the portal opening seen from `view`. The outline is
 * clipped against a plane just in front of the eye (not the near plane):
 * when the eye is closer than the near plane the portal is drawn as a thin
 * box, and its visible part is still the cone of rays through the opening.
 */
export function portalRect(view: Mat4, proj: Mat4, outline: Vec3[], width: number, height: number): Rect | null {
  const eps = 1e-3;
  const pts = outline.map((p) => m4.point(view, p));
  const clipped: Vec3[] = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    const ina = a[2] < -eps;
    const inb = b[2] < -eps;
    if (ina) clipped.push(a);
    if (ina !== inb) {
      const t = (-eps - a[2]) / (b[2] - a[2]);
      clipped.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, -eps]);
    }
  }
  if (clipped.length < 3) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of clipped) {
    const w = -p[2];
    const nx = (proj[0] * p[0]) / w;
    const ny = (proj[5] * p[1]) / w;
    const px = (nx * 0.5 + 0.5) * width;
    const py = (0.5 - ny * 0.5) * height;
    x0 = Math.min(x0, px); x1 = Math.max(x1, px);
    y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  // One pixel of slack for rasterisation of the polygon edge and MSAA.
  x0 = Math.max(0, Math.floor(x0) - 1);
  y0 = Math.max(0, Math.floor(y0) - 1);
  x1 = Math.min(width, Math.ceil(x1) + 1);
  y1 = Math.min(height, Math.ceil(y1) + 1);
  if (x1 <= x0 || y1 <= y0) return null;
  return [x0, y0, x1, y1];
}

export function intersectRect(a: Rect, b: Rect): Rect | null {
  const r: Rect = [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])];
  return r[2] > r[0] && r[3] > r[1] ? r : null;
}
