// HDR resolve: bloom, exposure, ACES filmic tone mapping (Narkowicz fit),
// vignette, sRGB encoding, dithering, and the debug views.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> M: Material;
@group(0) @binding(2) var hdr: texture_2d<f32>;
@group(0) @binding(3) var bloomTex: texture_2d<f32>;
@group(0) @binding(4) var linearSampler: sampler;
@group(0) @binding(5) var envCube: texture_cube<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return fullscreenPosition(vi);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

fn heatmap(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  return clamp(vec3f(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)), vec3f(0.0), vec3f(1.0));
}

fn tonemap(c: vec3f) -> vec3f {
  return aces(c * F.exposure);
}

// Raw field on the plane through the object's centre facing the camera.
fn fieldSlice(uv: vec2f, under: vec3f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let h = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  let d = normalize(h.xyz / h.w - F.camPos);
  let nrm = normalize(F.camPos - F.objCenter);
  let t = dot(F.objCenter - F.camPos, nrm) / dot(d, nrm);
  let q = toObject(F.camPos + d * t);
  if (t < 0.0 || any(abs(q) > vec3f(F.fieldBox))) { return under * 0.25; }
  let f = sampleField(q);
  var c = vec3f(pow(f.E, 1.5) * 1.5, f.C * 0.6, length(f.v) * 0.5);
  // Object outline (SDF zero set) and open cracks.
  let s = sampleShape(q).x;
  c = mix(c, vec3f(1.0), (1.0 - smoothstep(0.0, 0.012, abs(s))) * 0.8);
  let cr = sampleCrack(q);
  c = mix(c, vec3f(1.0, 0.9, 0.3), (1.0 - smoothstep(0.0, 0.01, cr.x)) * crackOpen(f.C, f.E) * 0.7);
  return c;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = vec2i(pos.xy);
  let uv = pos.xy / F.resolution;
  let src = textureLoad(hdr, px, 0);
  let mode = u32(F.debug + 0.5);
  var mapped: vec3f;
  if (mode == 0u) {
    let bloom = textureSampleLevel(bloomTex, linearSampler, uv, 0.0).rgb / 6.0;
    let c = mix(src.rgb, bloom, F.bloom);
    mapped = tonemap(c);
    let v = uv - 0.5;
    mapped *= 1.0 - 0.35 * smoothstep(0.25, 0.85, dot(v, v) * 1.6);
  } else if (mode == 1u) {
    mapped = fieldSlice(uv, tonemap(src.rgb));
  } else if (mode == 5u) {
    mapped = heatmap(luminance(src.rgb) * 1.2) * step(0.001, luminance(src.rgb));
  } else if (mode == 8u) {
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let h = F.invViewProj * vec4f(ndc, 1.0, 1.0);
    let d = normalize(h.xyz / h.w - F.camPos);
    mapped = tonemap(textureSampleLevel(envCube, linearSampler, d, 0.0).rgb);
  } else {
    // Object / halo debug colours are marked with negative alpha.
    mapped = select(tonemap(src.rgb) * 0.2, clamp(src.rgb, vec3f(0.0), vec3f(1.0)), src.a < 0.0);
  }
  let dither = (hash33(vec3i(px, i32(F.frameIndex) & 63)).x - 0.5) / 255.0;
  return vec4f(linearToSrgb(mapped) + dither, 1.0);
}
