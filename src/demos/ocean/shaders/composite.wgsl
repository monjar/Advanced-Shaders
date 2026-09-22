// HDR resolve: exposure, ACES filmic tone mapping, sRGB encoding, dithering.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdr: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return fullscreenPosition(vi);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = textureLoad(hdr, vec2i(pos.xy), 0).rgb;
  let mode = u32(F.debugMode + 0.5);
  var mapped: vec3f;
  if ((mode >= 1u && mode <= 5u) || mode == 7u) {
    mapped = clamp(c, vec3f(0.0), vec3f(1.0));
  } else {
    mapped = aces(c * F.exposure);
  }
  let dither = (hash21(pos.xy + fract(F.time)) - 0.5) / 255.0;
  return vec4f(linearToSrgb(mapped) + dither, 1.0);
}
