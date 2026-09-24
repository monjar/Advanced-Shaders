// Final pass: bloom, exposure, ACES tone mapping and sRGB, or a debug view.

@group(1) @binding(0) var hdrTex: texture_2d<f32>;
@group(1) @binding(1) var bloomTex: texture_2d<f32>;
@group(1) @binding(2) var auxTex: texture_2d<f32>;
@group(1) @binding(3) var linearClamp: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

// Red below 0 (redshift), white at 0, blue above (blueshift), over [lo, hi].
fn diverging(x: f32, lo: f32, hi: f32) -> vec3f {
  let t = clamp(x / select(-lo, hi, x > 0.0), -1.0, 1.0);
  let blue = vec3f(0.1, 0.3, 1.0);
  let red = vec3f(1.0, 0.12, 0.05);
  return select(mix(vec3f(0.95), red, -t), mix(vec3f(0.95), blue, t), t > 0.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2i(pos.xy);
  let uv = pos.xy / F.resolution;
  let hdr = textureLoad(hdrTex, p, 0).rgb;
  let aux = textureLoad(auxTex, p, 0);
  let bloom = textureSampleLevel(bloomTex, linearClamp, uv, 0.0).rgb;
  // The bloom chain sums its levels: F.post.w = 1 / level count.
  var colour = aces(mix(hdr, bloom * F.post.w, F.post.y) * F.post.x);

  let view = u32(F.view.y + 0.5);
  let flags = u32(aux.y + 0.5);
  let crossings = flags & 7u;
  let status = (flags >> 3u) & 3u;
  let bad = (flags & 32u) != 0u;
  let context = vec3f(dot(colour, vec3f(0.3, 0.5, 0.2))) * 0.35;
  if (view == 1u) {
    // Integration steps per pixel (accepted + rejected), log scale: 1 → dark
    // blue, 256 or more → dark red.
    colour = turbo(log2(1.0 + aux.x) / 8.0);
  } else if (view == 2u) {
    // Image order: m = crossings of the disk plane before escape or capture
    // (1 direct, 2 lensing ring, ≥ 3 photon ring, as in Gralla, Holz and
    // Wald 2019). Full colour where the ray actually hit the disk annulus.
    let m = (flags >> 6u) & 7u;
    var c = vec3f(0.0);
    if (m == 1u) { c = vec3f(1.0, 0.55, 0.1); }
    else if (m == 2u) { c = vec3f(0.1, 0.85, 0.55); }
    else if (m == 3u) { c = vec3f(0.85, 0.2, 0.95); }
    else if (m >= 4u) { c = vec3f(1.0); }
    colour = select(c * 0.1 + context * 0.6, c * 0.9, crossings > 0u);
    // Rays that end in the hole (the shadow) stay dark whatever their m.
    if (status == 1u && crossings == 0u) { colour = c * 0.12; }
  } else if (view == 3u) {
    // Escaped (blue), captured (black), step limit (red), absorbed by the disk (orange).
    var c = vec3f(0.15, 0.3, 0.8) + context;
    if (status == 1u) { c = vec3f(0.0); }
    else if (status == 2u) { c = vec3f(1.0, 0.0, 0.0); }
    else if (status == 3u) { c = vec3f(0.9, 0.5, 0.1); }
    colour = c;
  } else if (view == 4u) {
    // Redshift factor g at the first disk crossing, log2 from 1/2 to 2.
    colour = select(context * 0.5, diverging(log2(max(aux.z, 1e-6)), -1.0, 1.0), crossings > 0u);
  } else if (view == 5u) {
    // Sky footprint relative to an unlensed pixel (log2 of the linear size).
    colour = select(vec3f(0.0), turbo(0.5 + aux.w / 12.0), status == 0u);
  }
  if (bad) { colour = vec3f(1.0, 0.0, 1.0); }

  // Split-screen divider.
  if (u32(F.view.x + 0.5) == 2u && abs(pos.x - F.split * F.resolution.x) < 0.75) {
    colour = vec3f(0.8);
  }
  let dither = (fract(sin(dot(pos.xy + fract(F.time), vec2f(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
  return vec4f(linearToSrgb(colour) + dither, 1.0);
}
