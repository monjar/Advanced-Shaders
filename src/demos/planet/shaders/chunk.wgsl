// Chunk bake: positions of one quadtree node's 33 x 33 vertices plus skirts,
// relative to the node's origin O = R * dir(centre) (O itself stays in
// double precision on the CPU). One workgroup row per chunk of the batch.
//
//   heights: grid vertex -> fine position (this LOD's octaves) and the
//            parent-LOD position at the same point
//   morph:   odd vertices' morph target = the parent's surface there
//            (average of the neighbouring even vertices, following the
//            parent's triangle diagonal); skirt vertices
//
// Relative positions are formed without cancellation: with c = c0 + dc on
// the cube, dir - dir0 = dc / |c| + c0 (1/|c| - 1/|c0|) and
// 1/|c| - 1/|c0| = -(2 c0.dc + dc.dc) / (|c| |c0| (|c| + |c0|)), where
// tan(u pi/4) - tan(uc pi/4) = sin(du pi/4) / (cos(u pi/4) cos(uc pi/4)).
// Every term is proportional to the small offset, so float32 keeps
// ~1e-7 of the chunk size: 2 um for a 20 m chunk, 1 m for the 6,000 km
// root faces.

struct ChunkGen {
  faceN: vec4f,        // face normal, w = LOD level
  faceA: vec4f,        // face axis a, w = centre u
  faceB: vec4f,        // face axis b, w = centre v
  c0: vec4f,           // cube point of the centre, w = node size in u
  params: vec4f,       // pool slot, skirt depth (m), highest geometry octave level, unused
  table: array<NoiseSlot, TERRAIN_SLOTS>,
};

@group(0) @binding(0) var<uniform> SLOTS: array<SlotStatic, SLOT_COUNT>;
@group(0) @binding(1) var<storage, read> GEN: array<ChunkGen>;
@group(0) @binding(2) var<storage, read_write> VERTS: array<vec4f>;
@group(0) @binding(3) var<uniform> SHAPE: TerrainShape;

var<private> genIndex: u32;

fn tableSlot(s: u32) -> NoiseSlot {
  return GEN[genIndex].table[s];
}

struct Offset {
  dir: vec3f,
  rel: vec3f,   // R (dir - dir0), m
};

fn gridOffset(ia: u32, ib: u32) -> Offset {
  let g = GEN[genIndex];
  let q = PI * 0.25;
  let size = g.c0.w;
  let du = size * (f32(ia) / f32(GRID_N) - 0.5);
  let dv = size * (f32(ib) / f32(GRID_N) - 0.5);
  let uc = g.faceA.w;
  let vc = g.faceB.w;
  let dtu = sin(du * q) / (cos((uc + du) * q) * cos(uc * q));
  let dtv = sin(dv * q) / (cos((vc + dv) * q) * cos(vc * q));
  let dc = dtu * g.faceA.xyz + dtv * g.faceB.xyz;
  let c0 = g.c0.xyz;
  let c = c0 + dc;
  let lc = length(c);
  let lc0 = length(c0);
  let inv = -(2.0 * dot(c0, dc) + dot(dc, dc)) / (lc * lc0 * (lc + lc0));
  var o: Offset;
  o.dir = c / lc;
  o.rel = PLANET_R * (dc / lc + c0 * inv);
  return o;
}

fn vertexBase() -> u32 {
  return u32(GEN[genIndex].params.x) * VERTS_PER_CHUNK;
}

@compute @workgroup_size(64)
fn heights(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) wg: vec3u) {
  genIndex = wg.y;
  let vi = id.x;
  if (vi >= GRID * GRID) { return; }
  let g = GEN[genIndex];
  let o = gridOffset(vi % GRID, vi / GRID);
  let lod = g.faceN.w;
  let top = g.params.z;
  // This LOD's octaves, and the parent's for the morph target.
  let t = terrainEval(o.rel, SHAPE, vec4f(lod + 3.16, 8.0, top, 0.0), vec4f(lod + 2.16, 8.0, top, 0.0));
  let hf = max(t.h, 0.0);
  let hc = max(t.h2, 0.0);
  let k = (vertexBase() + vi) * 2u;
  VERTS[k] = vec4f(o.rel + o.dir * hf, hf);
  VERTS[k + 1u] = vec4f(o.rel + o.dir * hc, hc);
}

// The parent-LOD position at a grid vertex: its own for even-even vertices,
// otherwise interpolated from even neighbours along the parent's edges or
// its quad diagonal (the index buffer uses the main diagonal where
// (i + j) is even, at every level).
fn morphTarget(ia: u32, ib: u32) -> vec4f {
  let base = vertexBase();
  let at = (vertexBase() + ib * GRID + ia) * 2u + 1u;
  let oddA = (ia & 1u) == 1u;
  let oddB = (ib & 1u) == 1u;
  if (!oddA && !oddB) { return VERTS[at]; }
  var p = vec2u(0u);
  var q = vec2u(0u);
  if (oddA && !oddB) {
    p = vec2u(ia - 1u, ib);
    q = vec2u(ia + 1u, ib);
  } else if (!oddA && oddB) {
    p = vec2u(ia, ib - 1u);
    q = vec2u(ia, ib + 1u);
  } else if ((((ia >> 1u) + (ib >> 1u)) & 1u) == 0u) {
    p = vec2u(ia - 1u, ib - 1u);
    q = vec2u(ia + 1u, ib + 1u);
  } else {
    p = vec2u(ia + 1u, ib - 1u);
    q = vec2u(ia - 1u, ib + 1u);
  }
  let a = VERTS[(base + p.y * GRID + p.x) * 2u + 1u];
  let b = VERTS[(base + q.y * GRID + q.x) * 2u + 1u];
  return 0.5 * (a + b);
}

// Grid vertex on edge e (0: v = 0, 1: u = 1, 2: v = 1, 3: u = 0) at step s.
fn edgeVertex(e: u32, s: u32) -> vec2u {
  if (e == 0u) { return vec2u(s, 0u); }
  if (e == 1u) { return vec2u(GRID_N, s); }
  if (e == 2u) { return vec2u(s, GRID_N); }
  return vec2u(0u, s);
}

@compute @workgroup_size(64)
fn morph(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) wg: vec3u) {
  genIndex = wg.y;
  let vi = id.x;
  if (vi >= VERTS_PER_CHUNK) { return; }
  let base = vertexBase();
  if (vi < GRID * GRID) {
    let ia = vi % GRID;
    let ib = vi / GRID;
    // Only odd vertices change here, and they only read even ones.
    if (((ia | ib) & 1u) == 1u) {
      VERTS[(base + vi) * 2u + 1u] = morphTarget(ia, ib);
    }
    return;
  }
  // Skirt: the edge vertex dropped along its own vertical.
  let k = vi - GRID * GRID;
  let e = edgeVertex(k / GRID, k % GRID);
  let depth = GEN[genIndex].params.y;
  let dir = gridOffset(e.x, e.y).dir;
  let fine = VERTS[(base + e.y * GRID + e.x) * 2u];
  let coarse = morphTarget(e.x, e.y);
  VERTS[(base + vi) * 2u] = vec4f(fine.xyz - dir * depth, fine.w - depth);
  VERTS[(base + vi) * 2u + 1u] = vec4f(coarse.xyz - dir * depth, coarse.w - depth);
}
