// Final painting: pigment laid onto paper.
//  - paper: a procedural cold-press sheet on a dynamic canvas (see index.ts)
//  - wobble: the image is displaced by the paper relief, like pigment
//    following the tooth of the paper
//  - edge darkening: pigment collects at the rim of each wash
//  - granulation: pigment settles into the paper's valleys
//  - dry brush: paper peaks stay white where the wash is thin
//  - ink: optional pen lines with world-space pressure and breaks
//  - Beer–Lambert compositing: paper * exp(-absorbance)

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Paint;
@group(1) @binding(0) var pigmentTex: texture_2d<f32>;
@group(1) @binding(1) var edgesTex: texture_2d<f32>;
@group(1) @binding(2) var depthTex: texture_2d<f32>;
@group(1) @binding(3) var surfaceTex: texture_2d<f32>;
@group(1) @binding(4) var strokeTex: texture_2d<f32>;
@group(1) @binding(5) var rawPigment: texture_2d<f32>;
@group(1) @binding(6) var linearClamp: sampler;
@group(1) @binding(7) var noiseTex: texture_3d<f32>;
@group(1) @binding(8) var noiseSampler: sampler;
@group(1) @binding(9) var finalOut: texture_storage_2d<rgba8unorm, write>;

// Paper height at canvas coordinates: round bumps of cold-press paper plus
// faint fibres, from two decorrelated channels of the noise texture.
fn paperOctave(q: vec2f) -> f32 {
  let bumps = noiseAt(vec3f(q, 0.21)).x;
  let fibres = noiseAt(vec3f(q.x * 0.35 + q.y * 0.9, q.y * 0.35 - q.x * 0.9, 0.73)).y;
  return bumps * 0.8 + fibres * 0.2;
}

fn canvasCoords(px: vec2f, o: vec4f) -> vec2f {
  if (u32(P.mode.y + 0.5) == 1u) {
    // Paper fixed to the screen (for comparison).
    return px / (P.paperColor.a * F.pxScale * NOISE_CELLS);
  }
  return (px - F.resolution * 0.5) * o.x + o.yz;
}

fn paperHeight(px: vec2f) -> f32 {
  let a = paperOctave(canvasCoords(px, F.paper0));
  if (u32(P.mode.y + 0.5) == 1u) { return a; }
  let b = paperOctave(canvasCoords(px, F.paper1));
  // Octave blend keeps the grain size constant while the canvas zooms.
  let w = F.paper0.w;
  return ((a - 0.5) * (1.0 - w) + (b - 0.5) * w) / sqrt((1.0 - w) * (1.0 - w) + w * w) + 0.5;
}

// Grain attached to the painted surface: depth-adaptive world noise at the
// paper's grain size, or view-direction noise for the sky. The pixel offset
// lets callers take finite differences in screen space.
fn surfaceGrain(px: vec2f, depth: vec2f) -> f32 {
  let grainPx = P.paperColor.a;
  if (u32(P.mode.y + 0.5) == 1u) {
    return paperOctave(px / (grainPx * F.pxScale * NOISE_CELLS));
  }
  if (depth.y == SKY_ID) {
    return skyNoise(viewRay(px), px, grainPx).x * 0.8 + skyNoise(viewRay(px) * 1.7, px, grainPx * 0.6).y * 0.2;
  }
  let dir = viewRay(px);
  let p = F.camPos + dir * depth.x;
  let n = coherentNoise(p, px, depth.x, grainPx);
  return n.x * 0.8 + n.y * 0.2;
}

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.3, 0.55, 0.15));
}

@compute @workgroup_size(8, 8, 1)
fn composite(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(finalOut));
  let p = vec2i(id.xy);
  if (any(p >= size)) { return; }
  let px = vec2f(p) + 0.5;
  let res = vec2f(size);

  // Two grains. The sheet's relief (lit below) lives on the dynamic canvas.
  // How pigment sat on that sheet (granulation, dry brush, wobble) is part of
  // the painted surface, so it uses surface-attached grain and moves with the
  // scene instead of sliding across it.
  let depth = textureLoad(depthTex, p, 0).xy;
  let e = 1.0;
  let sheet = paperHeight(px);
  let sx = paperHeight(px + vec2f(e, 0.0)) - paperHeight(px - vec2f(e, 0.0));
  let sy = paperHeight(px + vec2f(0.0, e)) - paperHeight(px - vec2f(0.0, e));
  let sheetGrad = vec2f(sx, sy) / (2.0 * e);

  let world = F.camPos + viewRay(px) * min(depth.x, 2000.0);
  let h = surfaceGrain(px, depth);
  let gx = surfaceGrain(px + vec2f(1.0, 0.0), depth) - surfaceGrain(px - vec2f(1.0, 0.0), depth);
  let gy = surfaceGrain(px + vec2f(0.0, 1.0), depth) - surfaceGrain(px - vec2f(0.0, 1.0), depth);
  let grad = vec2f(gx, gy) * 0.5;

  // Wobble: sample the painting through the grain, like pigment following the tooth of the paper.
  let wobble = grad * P.paper.z * F.pxScale * 12.0;
  let uv = (px + wobble) / res;
  let pig = textureSampleLevel(pigmentTex, linearClamp, uv, 0.0);
  var A = pig.rgb;
  let granulation = pig.a;

  // Edge darkening: pigment accumulates where the wash density changes.
  let d = 1.5 * F.pxScale;
  let ax = luminance(textureSampleLevel(pigmentTex, linearClamp, uv + vec2f(d, 0.0) / res, 0.0).rgb)
         - luminance(textureSampleLevel(pigmentTex, linearClamp, uv - vec2f(d, 0.0) / res, 0.0).rgb);
  let ay = luminance(textureSampleLevel(pigmentTex, linearClamp, uv + vec2f(0.0, d) / res, 0.0).rgb)
         - luminance(textureSampleLevel(pigmentTex, linearClamp, uv - vec2f(0.0, d) / res, 0.0).rgb);
  let wash = textureSampleLevel(edgesTex, linearClamp, uv, 0.0).g;
  let rim = clamp(length(vec2f(ax, ay)) * 2.0 + wash * 0.25, 0.0, 1.0);
  A *= 1.0 + rim * P.edges.x;

  // Granulation: more pigment in the valleys of the paper.
  A *= 1.0 + P.paper.x * granulation * (0.5 - h) * 2.0;
  // Dry brush: thin washes skip over the paper's peaks.
  let thin = 1.0 - smoothstep(0.15, 0.9, luminance(A));
  A *= 1.0 - P.paper.y * smoothstep(0.58, 0.72, h) * thin;

  let paper = srgbToLinear(P.paperColor.rgb);
  var colour = paper * exp(-max(A, vec3f(0.0)));

  // Ink lines: thickness and breaks vary with world-space noise, so a line
  // keeps its character as the camera moves.
  if (P.edges.y > 0.0) {
    let n = coherentNoise(world, px, depth.x, 24.0);
    let width = P.edges.z * F.pxScale * (0.6 + 0.8 * n.x);
    var ink = textureSampleLevel(edgesTex, linearClamp, uv, 0.0).r;
    for (var i = 0; i < 8; i++) {
      let a = f32(i) * PI * 0.25 + n.y * 2.0;
      let o = vec2f(cos(a), sin(a)) * width;
      ink = max(ink, textureSampleLevel(edgesTex, linearClamp, uv + o / res, 0.0).r * 0.85);
    }
    let breaks = smoothstep(P.edges.w - 0.1, P.edges.w + 0.1, n.z);
    ink *= mix(1.0, breaks, 0.9);
    let inkColour = srgbToLinear(P.inkColor.rgb);
    colour = mix(colour, colour * inkColour, clamp(ink * P.edges.y, 0.0, 1.0));
  }

  // Light raking across the paper relief.
  // Paper mode 2 attaches the sheet relief to surfaces as well.
  let reliefGrad = select(sheetGrad, grad, u32(P.mode.y + 0.5) == 2u);
  let relief = dot(normalize(vec3f(-reliefGrad * 3.0, 1.0)), normalize(vec3f(-0.5, -0.6, 0.65)));
  colour *= 1.0 + P.paper.w * (relief - 0.65);

  // Debug views.
  let mode = u32(P.mode.z + 0.5);
  if (mode == 1u) {
    colour = srgbToLinear(exp(-textureLoad(rawPigment, p, 0).rgb));
  } else if (mode == 2u) {
    colour = vec3f(textureLoad(strokeTex, p, 0).z);
  } else if (mode == 3u) {
    let s = textureLoad(strokeTex, p, 0).xy;
    colour = srgbToLinear(vec3f(s * 0.5 + 0.5, 0.5));
  } else if (mode == 4u) {
    let ed = textureLoad(edgesTex, p, 0).rg;
    colour = vec3f(1.0 - ed.r, 1.0 - ed.g * 0.6, 1.0);
  } else if (mode == 5u) {
    colour = vec3f(textureLoad(surfaceTex, p, 0).a);
  } else if (mode == 6u) {
    colour = vec3f(h);
  } else if (mode == 7u) {
    colour = srgbToLinear(coherentNoise(world, px, depth.x, 14.0).xyz);
  }

  textureStore(finalOut, p, vec4f(linearToSrgb(clamp(colour, vec3f(0.0), vec3f(1.0))), 1.0));
}
