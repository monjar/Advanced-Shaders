// Sun shadow cascades: the chunks selected for each cascade's box, with the
// same vertex fetch and camera-distance morph as the G-buffer pass, into a
// depth-only orthographic map. Everything stays camera-relative; the
// cascade centre is snapped to its texel grid in double precision on the
// CPU so shadows do not shimmer as the camera moves.

struct Draw {
  offset: vec3f,
  level: f32,
  morph: vec2f,
  slot: u32,
  flags: u32,
};

@group(0) @binding(0) var<uniform> lightViewProj: mat4x4f;
@group(0) @binding(1) var<storage, read> DRAWS: array<Draw>;
@group(0) @binding(2) var<storage, read> VERTS: array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let d = DRAWS[ii];
  let k = (d.slot * VERTS_PER_CHUNK + vi) * 2u;
  let fine = VERTS[k];
  let coarse = VERTS[k + 1u];
  let m = smoothstep(d.morph.x, d.morph.y, length(d.offset + fine.xyz));
  return lightViewProj * vec4f(d.offset + mix(fine.xyz, coarse.xyz, m), 1.0);
}
