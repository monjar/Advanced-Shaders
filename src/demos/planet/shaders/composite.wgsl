// Exposure, ACES and sRGB for the final image (and the clouds-only view);
// the other debug views are already display colours.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let mode = u32(F.debugMode + 0.5);
  var c = textureLoad(hdrTex, vec2i(pos.xy), 0).rgb;
  if (mode == 0u || mode == 6u) { c = aces(c * F.exposure); }
  let dither = (hash21(pos.xy + fract(F.time)) - 0.5) / 255.0;
  return vec4f(linearToSrgb(clamp(c, vec3f(0.0), vec3f(1.0))) + dither, 1.0);
}
