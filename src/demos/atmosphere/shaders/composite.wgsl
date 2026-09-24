// Final pass: exposure, ACES, sRGB, plus the comparison and LUT debug views.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var refTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn referenceAt(px: vec2f) -> vec4f {
  let size = vec2i(textureDimensions(refTex));
  return textureLoad(refTex, clamp(vec2i(px / F.refScale), vec2i(0), size - 1), 0);
}

// A box of the given aspect centred on screen; returns (u, v, inside).
fn lutBox(uv: vec2f, aspect: f32) -> vec3f {
  let screen = F.resolution.x / F.resolution.y;
  var size = vec2f(0.92, 0.92 * screen / aspect);
  if (size.y > 0.92) { size = vec2f(0.92 * aspect / screen, 0.92); }
  let q = (uv - 0.5) / size + 0.5;
  return vec3f(q, select(0.0, 1.0, all(q >= vec2f(0.0)) && all(q <= vec2f(1.0))));
}

// Turbo-like false colour for error heat maps.
fn heat(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  return clamp(vec3f(
    1.5 - abs(4.0 * t - 3.0),
    1.5 - abs(4.0 * t - 2.0),
    1.5 - abs(4.0 * t - 1.0)), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = pos.xy;
  let uv = px / F.resolution;
  let mode = u32(F.debugMode + 0.5);
  let hdr = textureLoad(hdrTex, vec2i(px), 0);
  var color = hdr.rgb * F.exposure;
  var tonemap = true;

  if (mode == 1u) {
    color = referenceAt(px).rgb * F.exposure;
  } else if (mode == 2u) {
    if (uv.x > F.splitX) { color = referenceAt(px).rgb * F.exposure; }
    if (abs(px.x - F.splitX * F.resolution.x) < 1.0) { color = vec3f(4.0); }
  } else if (mode == 3u) {
    // Relative luminance error LUT vs reference: full scale = 10 %.
    let s = F.refScale;
    let refc = referenceAt(px);
    let lut = textureLoad(hdrTex, vec2i(floor(px / s) * s + floor(s * 0.5)), 0);
    let w = vec3f(0.2126, 0.7152, 0.0722);
    let yr = dot(refc.rgb, w);
    let rel = abs(dot(lut.rgb, w) - yr) / (yr + 0.01 / F.exposure);
    color = heat(rel / 0.1);
    tonemap = false;
  } else if (mode >= 4u && mode <= 9u) {
    tonemap = false;
    color = vec3f(0.02);
    if (mode == 4u) {
      let b = lutBox(uv, 4.0);
      if (b.z > 0.5) { color = textureSampleLevel(atmoTransmittanceLut, atmoSampler, b.xy, 0.0).rgb; }
    } else if (mode == 5u) {
      let b = lutBox(uv, 1.0);
      if (b.z > 0.5) { color = aces(textureSampleLevel(atmoMultiScatLut, atmoSampler, b.xy, 0.0).rgb * 50.0); }
    } else if (mode == 6u) {
      let b = lutBox(uv, ATMO.skyViewSize.x / ATMO.skyViewSize.y);
      if (b.z > 0.5) { color = aces(textureSampleLevel(atmoSkyViewLut, atmoSampler, b.xy, 0.0).rgb * F.exposure); }
    } else if (mode == 7u || mode == 8u) {
      // 32 slices as an 8 x 4 mosaic, nearest slice first.
      let b = lutBox(uv, 2.0);
      if (b.z > 0.5) {
        let cells = vec2f(8.0, 4.0);
        let c = floor(b.xy * cells);
        let local = fract(b.xy * cells);
        let slices = ATMO.apSlices;
        let z = (c.y * cells.x + c.x + 0.5) / slices;
        let coord = vec3f(local, z);
        if (mode == 7u) {
          color = aces(textureSampleLevel(atmoApInscatter, atmoSampler, coord, 0.0).rgb * F.exposure);
        } else {
          color = textureSampleLevel(atmoApTransmittance, atmoSampler, coord, 0.0).rgb;
        }
        if (any(local < vec2f(0.01)) || any(local > vec2f(0.99))) { color = vec3f(0.3); }
      }
    } else {
      let b = lutBox(uv, 4.0);
      if (b.z > 0.5) { color = aces(textureSampleLevel(atmoIrradianceLut, atmoSampler, b.xy, 0.0).rgb * 3.0); }
    }
  } else if (mode == 10u) {
    tonemap = false;
    color = select(vec3f(0.02, 0.02, 0.06), vec3f(fract(log2(max(hdr.a, 1e-3)))), hdr.a >= 0.0);
  }

  if (tonemap) { color = aces(color); }
  let dither = (hash21(px + fract(F.time)) - 0.5) / 255.0;
  return vec4f(linearToSrgb(color) + dither, 1.0);
}
