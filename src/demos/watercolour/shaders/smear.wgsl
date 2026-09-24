// Brush smear: a short line-integral blur of the pigment along the screen
// projection of each pixel's world-space brush direction. Taps stop at object
// boundaries so strokes don't cross silhouettes.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Paint;
@group(1) @binding(0) var pigmentIn: texture_2d<f32>;
@group(1) @binding(1) var strokeTex: texture_2d<f32>;
@group(1) @binding(2) var depthTex: texture_2d<f32>;
@group(1) @binding(3) var pigmentOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn smear(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(pigmentIn));
  let p = vec2i(id.xy);
  if (any(p >= size)) { return; }
  let centre = textureLoad(pigmentIn, p, 0);
  let dir = textureLoad(strokeTex, p, 0).xy;
  let objectId = textureLoad(depthTex, p, 0).y;
  let len = P.strokes.z * F.pxScale;
  var sum = centre.rgb;
  var weight = 1.0;
  for (var i = 1; i <= 4; i++) {
    let w = 1.0 - f32(i) / 5.0;
    for (var s = -1; s <= 1; s += 2) {
      let q = clamp(vec2i(round(vec2f(p) + dir * (f32(s * i) * len / 4.0))), vec2i(0), size - 1);
      if (textureLoad(depthTex, q, 0).y == objectId) {
        sum += textureLoad(pigmentIn, q, 0).rgb * w;
        weight += w;
      }
    }
  }
  textureStore(pigmentOut, p, vec4f(sum / weight, centre.a));
}
