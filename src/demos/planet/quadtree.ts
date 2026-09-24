// Cube-sphere quadtree: CPU-side LOD selection by screen-space error, a
// fixed pool of GPU chunk slots with LRU eviction, and the generation queue.
//
// Each cube face is mapped to the sphere with the equal-angle ("tangent")
// warp u' = tan(u pi/4), which keeps cells within ~1.4x of each other in
// area instead of ~5x for a plain normalised cube. A node is split when one
// of its quads would cover more than `maxErrorPx` pixels at its nearest
// distance; it is only split once all four children have been baked, so
// the drawn set always covers the sphere without overlaps. Distances,
// origins and bounds are doubles; only camera-relative offsets reach the GPU.

import { vec3, type Vec3 } from '../../core/math';
import { PLANET_RADIUS } from './terrain';

export const GRID_N = 32;
export const GRID = GRID_N + 1;
export const VERTS_PER_CHUNK = GRID * GRID + 4 * GRID;

// Face frames: normal n and tangent axes a, b with a x b = n.
export const FACES: { n: Vec3; a: Vec3; b: Vec3 }[] = [
  { n: [1, 0, 0], a: [0, 0, -1], b: [0, 1, 0] },
  { n: [-1, 0, 0], a: [0, 0, 1], b: [0, 1, 0] },
  { n: [0, 1, 0], a: [1, 0, 0], b: [0, 0, -1] },
  { n: [0, -1, 0], a: [1, 0, 0], b: [0, 0, 1] },
  { n: [0, 0, 1], a: [1, 0, 0], b: [0, 1, 0] },
  { n: [0, 0, -1], a: [-1, 0, 0], b: [0, 1, 0] },
];

/** Arc length of a node's side at `level` (m), used as the per-level LOD scale. */
export const nodeSize = (level: number) => (PLANET_RADIUS * Math.PI) / 2 / 2 ** level;

export function cubePoint(face: number, u: number, v: number): Vec3 {
  const f = FACES[face];
  const tu = Math.tan((u * Math.PI) / 4);
  const tv = Math.tan((v * Math.PI) / 4);
  return [f.n[0] + tu * f.a[0] + tv * f.b[0], f.n[1] + tu * f.a[1] + tv * f.b[1], f.n[2] + tu * f.a[2] + tv * f.b[2]];
}

export interface QNode {
  key: string;
  face: number;
  level: number;
  i: number;
  j: number;
  /** Centre of the node in face coordinates. */
  uc: number;
  vc: number;
  /** Unit direction of the centre. */
  dir: Vec3;
  /** Bounding sphere (double, m, planet centred). */
  center: Vec3;
  radius: number;
  hMin: number;
  hMax: number;
  slot: number;
  ready: boolean;
  lastUsed: number;
  lastSplit: number;
  children: QNode[] | null;
}

export interface Draw {
  node: QNode;
  /** Distance from the camera at which this level splits (m). */
  splitDistance: number;
}

export interface SelectParams {
  camera: Vec3;
  /** Camera-relative frustum planes (a, b, c, d), inside where a x + b y + c z + d >= 0. */
  planes: [number, number, number, number][];
  pxPerRadian: number;
  maxErrorPx: number;
  maxLevel: number;
  /** Radius below which nothing can be (for horizon culling). */
  occluderRadius: number;
}

export class ChunkTree {
  readonly roots: QNode[];
  readonly poolSize: number;
  private free: number[] = [];
  private bySlot: (QNode | null)[];
  private frame = 0;
  requests: QNode[] = [];
  private requested = new Set<QNode>();
  draws: Draw[] = [];
  stats = { visited: 0, culled: 0, drawn: 0, maxLevel: 0, resident: 0 };

  constructor(poolSize: number, private heightRange: (node: QNode) => [number, number]) {
    this.poolSize = poolSize;
    this.bySlot = new Array(poolSize).fill(null);
    for (let s = poolSize - 1; s >= 0; s--) this.free.push(s);
    this.roots = FACES.map((_, f) => this.makeNode(f, 0, 0, 0));
  }

  private makeNode(face: number, level: number, i: number, j: number): QNode {
    const size = 2 / 2 ** level;
    const uc = -1 + (i + 0.5) * size;
    const vc = -1 + (j + 0.5) * size;
    const dir = vec3.normalize(cubePoint(face, uc, vc));
    const node: QNode = {
      key: `${face}/${level}/${i}/${j}`, face, level, i, j, uc, vc, dir,
      center: [0, 0, 0], radius: 0, hMin: 0, hMax: 0, slot: -1, ready: false, lastUsed: -1, lastSplit: -1, children: null,
    };
    // Horizontal extent: farthest corner or edge midpoint from the centre.
    const c0 = vec3.scale(dir, PLANET_RADIUS);
    let rXY = 0;
    for (const [du, dv] of [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1], [0, 1], [-1, 0], [1, 0]]) {
      const p = vec3.scale(vec3.normalize(cubePoint(face, uc + (du * size) / 2, vc + (dv * size) / 2)), PLANET_RADIUS);
      rXY = Math.max(rXY, vec3.length(vec3.sub(p, c0)));
    }
    const [hMin, hMax] = this.heightRange(node);
    node.hMin = hMin;
    node.hMax = hMax;
    // The sphere bulges above the chord between corners: include the sagitta.
    const sag = (rXY * rXY) / (2 * PLANET_RADIUS);
    node.center = vec3.scale(dir, PLANET_RADIUS + (hMin + hMax) / 2 - sag / 2);
    node.radius = Math.hypot(rXY, (hMax - hMin) / 2 + sag / 2) * 1.02 + 1;
    return node;
  }

  private childrenOf(n: QNode): QNode[] {
    if (!n.children) {
      n.children = [];
      for (let dj = 0; dj < 2; dj++) {
        for (let di = 0; di < 2; di++) n.children.push(this.makeNode(n.face, n.level + 1, n.i * 2 + di, n.j * 2 + dj));
      }
    }
    return n.children;
  }

  private visible(n: QNode, p: SelectParams): boolean {
    const rel = vec3.sub(n.center, p.camera);
    for (const pl of p.planes) {
      if (pl[0] * rel[0] + pl[1] * rel[1] + pl[2] * rel[2] + pl[3] < -n.radius) return false;
    }
    // Horizon culling against the occluder sphere (nothing lies below it):
    // a point is hidden if it is further than the sum of both tangent lengths.
    const dc = vec3.length(p.camera);
    const ro = p.occluderRadius;
    if (ro > 0 && dc > ro) {
      const tCam = Math.sqrt(dc * dc - ro * ro);
      const top = vec3.length(n.center) + n.radius;
      const tNode = Math.sqrt(Math.max(0, top * top - ro * ro));
      if (vec3.length(rel) - n.radius > tCam + tNode) return false;
    }
    return true;
  }

  /** Starts a frame: clears the requests that the selections below collect. */
  beginFrame() {
    this.frame++;
    this.requests = [];
    this.requested.clear();
    this.stats = { visited: 0, culled: 0, drawn: 0, maxLevel: 0, resident: this.poolSize - this.free.length };
  }

  /**
   * Walks the tree for one view (the camera, or a shadow cascade: same LOD
   * metric, different culling volume); returns what to draw and adds what
   * is missing to `requests`. `main` selections count towards `stats`.
   */
  select(p: SelectParams, main = true): Draw[] {
    const draws: Draw[] = [];
    const splitDistance = (level: number) => (nodeSize(level) / GRID_N) * p.pxPerRadian / p.maxErrorPx;
    const request = (n: QNode) => {
      if (n.ready || this.requested.has(n)) return;
      this.requested.add(n);
      this.requests.push(n);
    };
    const draw = (n: QNode) => {
      // Only drawn chunks count as used: ancestors that are merely split
      // may be evicted, since baked descendants can stand in for them.
      n.lastUsed = this.frame;
      draws.push({ node: n, splitDistance: splitDistance(n.level) });
      if (!main) return;
      this.stats.drawn++;
      this.stats.maxLevel = Math.max(this.stats.maxLevel, n.level);
    };
    // Draws n, or when it is not baked yet its baked descendants.
    const cover = (n: QNode): boolean => {
      if (n.ready) {
        draw(n);
        return true;
      }
      request(n);
      if (n.children && n.children.every((c) => !this.visible(c, p) || coverable(c, 0))) {
        n.children.forEach((c) => {
          if (this.visible(c, p)) cover(c);
        });
        return true;
      }
      return false;
    };
    // A node can be drawn if it is baked, or if its visible children can.
    const coverable = (n: QNode, depth: number): boolean => {
      if (n.ready) return true;
      if (!n.children || depth > 3) return false;
      return n.children.every((c) => !this.visible(c, p) || coverable(c, depth + 1));
    };
    const visit = (n: QNode) => {
      if (main) this.stats.visited++;
      if (!this.visible(n, p)) {
        if (main) this.stats.culled++;
        return;
      }
      const dist = Math.max(vec3.length(vec3.sub(n.center, p.camera)) - n.radius, 0);
      const wantSplit = n.level < p.maxLevel && dist < splitDistance(n.level);
      // Forget unbaked children of nodes that stopped splitting (their
      // bounds cost CPU height samples to rebuild, so keep them a while).
      if (!wantSplit && n.children && this.frame - n.lastSplit > 600 && n.children.every((c) => !c.ready && !c.children)) {
        n.children = null;
      }
      if (wantSplit) {
        n.lastSplit = this.frame;
        // Only the children in view have to be baked before splitting;
        // the others are simply not drawn.
        const kids = this.childrenOf(n).filter((c) => this.visible(c, p));
        if (kids.every((c) => coverable(c, 0))) {
          kids.forEach(visit);
          return;
        }
        kids.forEach(request);
      }
      cover(n);
    };
    this.roots.forEach(visit);
    if (main) this.draws = draws;
    return draws;
  }

  /**
   * Assigns pool slots to up to `budget` requested nodes (coarsest and
   * nearest first), evicting the least recently used chunks. Returns the
   * nodes to bake this frame; they are marked ready (the bake runs before
   * the draw in the same submission).
   */
  allocate(budget: number, camera: Vec3): QNode[] {
    const reqs = this.requests
      .map((n) => ({ n, d: vec3.length(vec3.sub(n.center, camera)) }))
      .sort((a, b) => a.n.level - b.n.level || a.d - b.d)
      .slice(0, budget);
    const out: QNode[] = [];
    let evictable: QNode[] | null = null;
    for (const { n } of reqs) {
      let slot = this.free.pop();
      if (slot === undefined) {
        if (!evictable) {
          evictable = this.bySlot
            .filter((c): c is QNode => !!c && c.lastUsed < this.frame && c.level > 0)
            .sort((a, b) => a.lastUsed - b.lastUsed || b.level - a.level);
        }
        const victim = evictable.shift();
        if (!victim) break;
        slot = victim.slot;
        victim.slot = -1;
        victim.ready = false;
      }
      n.slot = slot;
      n.ready = true;
      n.lastUsed = this.frame;
      this.bySlot[slot] = n;
      out.push(n);
    }
    return out;
  }
}
