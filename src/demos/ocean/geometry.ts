/** Interleaved position/normal meshes for the floating objects. */
export interface Mesh {
  vertices: Float32Array<ArrayBuffer>;
  indices: Uint16Array<ArrayBuffer>;
}

class MeshBuilder {
  private v: number[] = [];
  private i: number[] = [];

  vertex(p: number[], n: number[]): number {
    this.v.push(p[0], p[1], p[2], n[0], n[1], n[2]);
    return this.v.length / 6 - 1;
  }

  tri(a: number, b: number, c: number) {
    this.i.push(a, b, c);
  }

  build(): Mesh {
    return { vertices: new Float32Array(this.v), indices: new Uint16Array(this.i) };
  }
}

/** Unit cube centred at the origin. */
export function box(): Mesh {
  const m = new MeshBuilder();
  const faces: [number[], number[], number[]][] = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
  ];
  for (const [n, u, w] of faces) {
    const corner = (su: number, sw: number) =>
      m.vertex([0.5 * (n[0] + su * u[0] + sw * w[0]), 0.5 * (n[1] + su * u[1] + sw * w[1]), 0.5 * (n[2] + su * u[2] + sw * w[2])], n);
    const a = corner(-1, -1);
    const b = corner(1, -1);
    const c = corner(1, 1);
    const d = corner(-1, 1);
    m.tri(a, c, b);
    m.tri(a, d, c);
  }
  return m.build();
}

/** Navigation buoy: cylinder hull with a conical top, spanning y in [-0.6, 0.6]. */
export function buoy(segments = 32): Mesh {
  const m = new MeshBuilder();
  const r = 0.32;
  const y0 = -0.6;
  const y1 = 0.25;
  const y2 = 0.6;
  const coneSlope = r / (y2 - y1);
  for (let s = 0; s < segments; s++) {
    const a0 = (s / segments) * Math.PI * 2;
    const a1 = ((s + 1) / segments) * Math.PI * 2;
    const c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
    // Side.
    const p = m.vertex([r * c0, y0, r * s0], [c0, 0, s0]);
    const q = m.vertex([r * c1, y0, r * s1], [c1, 0, s1]);
    const t = m.vertex([r * c1, y1, r * s1], [c1, 0, s1]);
    const u = m.vertex([r * c0, y1, r * s0], [c0, 0, s0]);
    m.tri(p, u, t);
    m.tri(p, t, q);
    // Cone.
    const ny = coneSlope / Math.hypot(1, coneSlope);
    const nr = 1 / Math.hypot(1, coneSlope);
    const cm = Math.cos((a0 + a1) / 2), sm = Math.sin((a0 + a1) / 2);
    const k0 = m.vertex([r * c0, y1, r * s0], [nr * c0, ny, nr * s0]);
    const k1 = m.vertex([r * c1, y1, r * s1], [nr * c1, ny, nr * s1]);
    const apex = m.vertex([0, y2, 0], [nr * cm, ny, nr * sm]);
    m.tri(k0, apex, k1);
    // Bottom cap.
    const b0 = m.vertex([r * c0, y0, r * s0], [0, -1, 0]);
    const b1 = m.vertex([r * c1, y0, r * s1], [0, -1, 0]);
    const bc = m.vertex([0, y0, 0], [0, -1, 0]);
    m.tri(b0, b1, bc);
  }
  return m.build();
}

/** UV sphere of radius 0.5. */
export function sphere(rings = 16, segments = 32): Mesh {
  const m = new MeshBuilder();
  for (let y = 0; y <= rings; y++) {
    const v = y / rings;
    const theta = v * Math.PI;
    for (let x = 0; x <= segments; x++) {
      const phi = (x / segments) * Math.PI * 2;
      const n = [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];
      m.vertex([n[0] * 0.5, n[1] * 0.5, n[2] * 0.5], n);
    }
  }
  for (let y = 0; y < rings; y++) {
    for (let x = 0; x < segments; x++) {
      const a = y * (segments + 1) + x;
      const b = a + segments + 1;
      m.tri(a, a + 1, b);
      m.tri(b, a + 1, b + 1);
    }
  }
  return m.build();
}
