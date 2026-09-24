// Terrain G-buffer: every visible chunk in one instanced draw. The vertex
// shader fetches the baked vertex, morphs it towards the parent LOD's
// surface with distance (CDLOD-style: fully morphed where the next coarser
// level takes over, so neighbouring levels meet exactly) and places it
// camera-relative: O - camera is computed in double on the CPU, so float32
// only ever holds small numbers near the camera.
// Targets: rgba32float (camera-relative position m, interpolated height)
// and rgba8unorm (LOD level, morph, 1 = geometry).

struct Draw {
  offset: vec3f,       // chunk origin minus camera (m)
  level: f32,
  morph: vec2f,        // morph start / end distance (m)
  slot: u32,
  flags: u32,
};

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> DRAWS: array<Draw>;
@group(0) @binding(2) var<storage, read> VERTS: array<vec4f>;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) height: f32,
  @location(2) @interpolate(flat) info: vec2f,
  @location(3) morph: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
  let d = DRAWS[ii];
  let k = (d.slot * VERTS_PER_CHUNK + vi) * 2u;
  let fine = VERTS[k];
  let coarse = VERTS[k + 1u];
  let dist = length(d.offset + fine.xyz);
  let m = smoothstep(d.morph.x, d.morph.y, dist);
  let p = d.offset + mix(fine.xyz, coarse.xyz, m);
  var o: VsOut;
  o.pos = F.viewProj * vec4f(p, 1.0);
  o.rel = p;
  o.height = mix(fine.w, coarse.w, m);
  o.info = vec2f(d.level, f32(d.slot));
  o.morph = m;
  return o;
}

struct GOut {
  @location(0) g0: vec4f,
  @location(1) g1: vec4f,
};

@fragment
fn fs(i: VsOut) -> GOut {
  var o: GOut;
  o.g0 = vec4f(i.rel, i.height);
  o.g1 = vec4f(i.info.x / 32.0, i.morph, fract(i.info.y * 0.618034), 1.0);
  return o;
}
