// Edge extraction from the G-buffer (not from colour, so it is noise-free and
// moves exactly with the geometry). Output:
//   r: ink edges — silhouettes (depth), object boundaries (id), creases (normal)
//   g: wash boundaries — where one wash ends and another begins (id, depth)

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Paint;
@group(1) @binding(0) var surfaceTex: texture_2d<f32>;
@group(1) @binding(1) var depthTex: texture_2d<f32>;
@group(1) @binding(2) var edgesOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn edges(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(depthTex));
  let p = vec2i(id.xy);
  if (any(p >= size)) { return; }
  let c = textureLoad(depthTex, p, 0).xy;
  let n = textureLoad(surfaceTex, p, 0).xyz;
  var depthEdge = 0.0;
  var idEdge = 0.0;
  var crease = 0.0;
  let offsets = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
  for (var i = 0; i < 4; i++) {
    let q = clamp(p + offsets[i], vec2i(0), size - 1);
    let d = textureLoad(depthTex, q, 0).xy;
    let nq = textureLoad(surfaceTex, q, 0).xyz;
    // Only the nearer side of a depth jump draws the line, so it hugs the object.
    let rel = (d.x - c.x) / max(c.x, 1e-3);
    depthEdge = max(depthEdge, smoothstep(0.03, 0.12, rel));
    idEdge = max(idEdge, select(0.0, 1.0, d.y != c.y && d.x >= c.x));
    if (c.y != SKY_ID && d.y == c.y) {
      crease = max(crease, smoothstep(P.inkColor.a, P.inkColor.a + 0.25, 1.0 - dot(n, nq)));
    }
  }
  let ink = max(max(depthEdge, idEdge), crease);
  let wash = max(depthEdge, select(0.0, 1.0, idEdge > 0.0));
  textureStore(edgesOut, p, vec4f(ink, wash, 0.0, 0.0));
}
