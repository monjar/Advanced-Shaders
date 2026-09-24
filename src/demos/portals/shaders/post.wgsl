// Final pass: rim refraction (screen-space resample using the offsets the
// restore passes wrote), debug views, scissor-rectangle overlay, ACES and sRGB.

const MAX_RECTS: u32 = 64u;

struct Post {
  resolution: vec2f,
  time: f32,
  debugView: f32,
  exposure: f32,
  showRects: f32,
  rectCount: f32,
  pad: f32,
  rects: array<vec4f, MAX_RECTS>,        // x0, y0, x1, y1 (pixels)
  rectLevels: array<vec4f, 16>,          // level of each rect, 4 per vec4
};

@group(0) @binding(0) var<uniform> U: Post;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var rimTex: texture_2d<f32>;
@group(0) @binding(3) var infoTex: texture_2d<f32>;
@group(0) @binding(4) var glowTex: texture_2d<f32>;
@group(0) @binding(5) var linearSampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

fn hue(h: f32) -> vec3f {
  return clamp(abs(fract(h + vec3f(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
}

fn levelColour(level: f32) -> vec3f {
  // 1 red, 2 orange, 3 yellow, 4 green, ... 8 violet.
  return hue((level - 1.0) * 0.1);
}

fn hash11(x: f32) -> f32 {
  return fract(sin(x * 91.3458) * 47453.5453);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let px = vec2i(pos.xy);
  let rim = textureLoad(rimTex, px, 0);
  // Inverse of encodeRimOffset (common.wgsl); 128/255 decodes to exactly 0.
  let offset = (round(rim.rg * 255.0) - 128.0) * 0.25;
  let uv = (pos.xy + offset) / U.resolution;
  // The glow is added after the resample, so the rim stays sharp.
  let c = textureSampleLevel(hdr, linearSampler, uv, 0.0).rgb + textureLoad(glowTex, px, 0).rgb * 8.0; // GLOW_SCALE
  let info = textureLoad(infoTex, px, 0);
  let level = round(info.r * 8.0);

  var mapped = aces(c * U.exposure);
  let view = u32(U.debugView);
  if (view == 1u) {
    // Recursion depth: every level tinted by its own colour.
    // Level 0 stays grey so the nested levels stand out.
    let luma = dot(mapped, vec3f(0.3, 0.55, 0.15));
    mapped = select(mix(vec3f(luma), levelColour(level), 0.65) * (0.45 + 0.75 * luma), vec3f(luma * 0.8), level < 0.5);
  } else if (view == 2u) {
    // Stencil masks: a flat colour per view (mask), outlined where masks meet.
    let id = info.g;
    let right = textureLoad(infoTex, min(px + vec2i(1, 0), vec2i(U.resolution) - 1), 0).g;
    let down = textureLoad(infoTex, min(px + vec2i(0, 1), vec2i(U.resolution) - 1), 0).g;
    let edge = select(0.0, 1.0, abs(right - id) > 0.001 || abs(down - id) > 0.001);
    mapped = mix(hue(hash11(id * 64.0 + 1.0)) * (0.35 + 0.65 * (1.0 - level / 9.0)), vec3f(1.0), edge);
    // Hatched: the recursion ended here (rim-colour fill instead of a view).
    if (info.b > 0.15 && info.b < 0.35) { mapped *= 0.5 + 0.5 * step(0.5, fract((pos.x + pos.y) / 12.0)); }
  } else if (view == 3u) {
    // Objects crossing a portal: the original half cyan, the duplicate half magenta.
    if (info.b > 0.9) { mapped = mix(mapped, vec3f(1.0, 0.1, 0.8), 0.6); }
    else if (info.b > 0.4 && info.b < 0.6) { mapped = mix(mapped, vec3f(0.1, 0.9, 1.0), 0.6); }
  }

  if (U.showRects > 0.5) {
    for (var i = 0u; i < u32(U.rectCount); i++) {
      let r = U.rects[i];
      let inside = pos.x >= r.x && pos.x <= r.z && pos.y >= r.y && pos.y <= r.w;
      let d = min(min(pos.x - r.x, r.z - pos.x), min(pos.y - r.y, r.w - pos.y));
      if (inside && d < 2.0) {
        let lv = U.rectLevels[i / 4u][i % 4u];
        mapped = levelColour(lv);
      }
    }
  }

  let dither = (fract(sin(dot(pos.xy, vec2f(12.9898, 78.233))) * 43758.5453) - 0.5) / 255.0;
  return vec4f(linearToSrgb(mapped) + dither, 1.0);
}
