// Colour bleeding (wet-in-wet). A separable blur of the pigment absorbance
// whose weights require BOTH the centre and the sample to be wet, so pigment
// only runs across boundaries between wet washes; dry washes keep hard edges.
// The radius follows the (world-space) wetness, which makes blooms irregular
// but stable. Run once horizontally and once vertically.

override HORIZONTAL: bool = true;
const TAPS: i32 = 12;

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Paint;
@group(1) @binding(0) var pigmentIn: texture_2d<f32>;
@group(1) @binding(1) var surfaceTex: texture_2d<f32>;
@group(1) @binding(2) var pigmentOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn bleed(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(pigmentIn));
  let p = vec2i(id.xy);
  if (any(p >= size)) { return; }
  let centre = textureLoad(pigmentIn, p, 0);
  let wetC = textureLoad(surfaceTex, p, 0).a;
  let radius = P.bleed.x * F.pxScale * wetC;
  if (radius < 0.75) {
    textureStore(pigmentOut, p, centre);
    return;
  }
  let axis = select(vec2i(0, 1), vec2i(1, 0), HORIZONTAL);
  let step = radius / f32(TAPS);
  var sum = centre.rgb;
  var weight = 1.0;
  for (var i = -TAPS; i <= TAPS; i++) {
    if (i == 0) { continue; }
    let q = clamp(p + axis * i32(round(f32(i) * step)), vec2i(0), size - 1);
    let wetQ = textureLoad(surfaceTex, q, 0).a;
    let x = f32(i) / f32(TAPS);
    let w = exp(-x * x * 3.0) * sqrt(wetC * wetQ);
    sum += textureLoad(pigmentIn, q, 0).rgb * w;
    weight += w;
  }
  textureStore(pigmentOut, p, vec4f(sum / weight, centre.a));
}
