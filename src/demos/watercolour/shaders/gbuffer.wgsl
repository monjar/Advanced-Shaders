// Scene pass. Instead of shading colour, each pixel decides how a painter
// would treat it and writes that into four targets:
//   pigment:  absorbance (rgb) of the wash at this pixel, granulation amount
//   surface:  world normal, wetness (how much this wash bleeds)
//   depth:    view distance, object id
//   stroke:   screen-space brush direction (xy), light value, material
// Pigment is stored as absorbance (optical density) so later blurs and
// mixing are subtractive, like real pigment, rather than averaging colours.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Paint;
@group(1) @binding(0) var noiseTex: texture_3d<f32>;
@group(1) @binding(1) var noiseSampler: sampler;
@group(1) @binding(2) var shadowMap: texture_depth_2d;
@group(1) @binding(3) var shadowSampler: sampler_comparison;

// Materials (see scene.ts).
const GRASS: u32 = 0u;
const FOLIAGE: u32 = 1u;
const WALL: u32 = 2u;
const ROOF: u32 = 3u;
const WOOD: u32 = 4u;
const ROCK: u32 = 5u;
const WATER: u32 = 6u;
const DARK: u32 = 7u;
const PATH: u32 = 8u;
const HILLS: u32 = 9u;

struct Targets {
  @location(0) pigment: vec4f,
  @location(1) surface: vec4f,
  @location(2) depth: vec2f,
  @location(3) stroke: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) @interpolate(flat) material: u32,
  @location(4) @interpolate(flat) objectId: f32,
};

@vertex
fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) colour: vec4f, @location(3) objectId: f32) -> VSOut {
  var out: VSOut;
  out.position = F.viewProj * vec4f(position, 1.0);
  out.world = position;
  out.normal = normal;
  out.albedo = colour.rgb;
  out.material = u32(colour.a * 255.0 + 0.5);
  out.objectId = objectId;
  return out;
}

// Painters follow form: strokes run along walls, down roofs, up trunks,
// across slopes along the contour, and around foliage.
fn brushDirection(material: u32, N: vec3f) -> vec3f {
  let up = vec3f(0.0, 1.0, 0.0);
  let downhill = up - N * dot(N, up);
  var contour = cross(N, up);
  // Flat ground has no contour: fall back to a fixed world direction.
  let flatness = 1.0 - smoothstep(0.05, 0.3, length(contour));
  contour = normalize(mix(contour, vec3f(0.94, 0.0, 0.34), flatness) + vec3f(1e-5));
  switch material {
    case ROOF, WOOD: { return normalize(downhill + contour * 1e-3); }
    case WATER: { return vec3f(1.0, 0.0, 0.0); }
    default: { return contour; }
  }
}

fn materialWetness(material: u32) -> f32 {
  switch material {
    case FOLIAGE: { return 1.0; }
    case WATER, HILLS: { return 0.9; }
    case GRASS, PATH: { return 0.7; }
    case ROCK: { return 0.5; }
    case ROOF: { return 0.35; }
    case WALL: { return 0.25; }
    default: { return 0.1; }
  }
}

fn materialGranulation(material: u32) -> f32 {
  switch material {
    case ROCK, HILLS, PATH: { return 1.0; }
    case WALL, ROOF: { return 0.7; }
    case WATER: { return 0.5; }
    default: { return 0.35; }
  }
}

fn castShadow(world: vec3f, N: vec3f, jitter: vec2f) -> f32 {
  let p = F.lightViewProj * vec4f(world + N * 0.15, 1.0);
  let ndc = p.xyz / p.w;
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
  // Wobbly, hand-drawn shadow edge: offset the lookup with surface noise.
  let texel = 1.0 / f32(textureDimensions(shadowMap).x);
  let base = uv + (jitter - 0.5) * texel * 6.0;
  var s = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      s += textureSampleCompareLevel(shadowMap, shadowSampler, base + vec2f(f32(x), f32(y)) * texel * 1.5, ndc.z - 0.0015);
    }
  }
  return s / 9.0;
}

// Colour of a wash of pigment with transmittance `pigment` and `density`,
// as absorbance: paper * exp(-absorbance) reproduces it over white paper.
fn absorbance(pigmentSrgb: vec3f, density: f32) -> vec3f {
  let t = clamp(srgbToLinear(pigmentSrgb), vec3f(0.02), vec3f(0.999));
  return -log(t) * density;
}

@fragment
fn fs(in: VSOut, @builtin(front_facing) front: bool) -> Targets {
  let V = normalize(F.camPos - in.world);
  var N = normalize(in.normal);
  if (!front && dot(N, V) < 0.0) { N = -N; }
  let dist = length(F.camPos - in.world);
  let px = in.position.xy;

  // Large, slow variations: pigment turbulence, glaze edge jitter, wet areas, hue.
  let broad = coherentNoise(in.world, px, dist, 70.0);
  let fine = coherentNoise(in.world * 1.37 + 11.0, px, dist, 14.0);

  // --- Lighting abstraction ---------------------------------------------------
  // Diffuse light and a cast shadow are reduced to three glazes with soft,
  // irregular boundaries: paper left white for lit planes, a mid wash, and a
  // darker, cooler shadow wash.
  let lambert = dot(N, F.lightDir);
  let shadow = mix(1.0, castShadow(in.world, N, fine.xy), P.bleed.z);
  var light = clamp(lambert * 0.5 + 0.5, 0.0, 1.0) * mix(0.35, 1.0, shadow);
  light += (broad.y - 0.5) * P.bands.w;
  let soft = P.bands.z;
  let lit = smoothstep(P.bands.x - soft, P.bands.x + soft, light);
  let notShadow = smoothstep(P.bands.y - soft, P.bands.y + soft, light);
  var density = mix(P.density.z, mix(P.density.y, P.density.x, lit), notShadow);
  density *= 1.0 + (broad.x - 0.5) * 2.0 * P.density.w;

  // Shadows lean cool, as painters mix a blue or violet into them.
  var pigment = in.albedo * (1.0 + (broad.w - 0.5) * 0.25);
  pigment = mix(pigment, pigment * P.shadowTint.rgb, P.shadowTint.a * (1.0 - notShadow));

  // Aerial perspective: distant washes are paler and bluer.
  let haze = 1.0 - exp(-dist * P.bleed.w);
  pigment = mix(pigment, vec3f(0.55, 0.64, 0.82), haze * 0.7);
  density *= mix(1.0, 0.6, haze);
  if (in.material == HILLS) { density = max(density, 0.6); }

  // --- World-space brush strokes ---------------------------------------------
  var brush = brushDirection(in.material, N);
  let angle = (broad.z - 0.5) * P.strokes.w;
  brush = normalize(brush * cos(angle) + cross(N, brush) * sin(angle));
  // Stretch the noise along the brush so its streaks follow the stroke.
  let along = dot(in.world, brush);
  let streakP = in.world - brush * along * 0.85;
  let streak = coherentNoise(streakP * 1.9, px, dist, P.strokes.y).x;
  density *= 1.0 + (streak - 0.5) * 2.0 * P.strokes.x;

  var A = absorbance(pigment, max(density, 0.0));
  // Dark details (windows, doors) are painted as small solid touches.
  if (in.material == DARK) { A = absorbance(in.albedo, 1.3); }

  let wet = materialWetness(in.material) * smoothstep(0.62 - P.bleed.y, 0.72 - P.bleed.y * 0.6, broad.z);

  // Brush direction in screen space for the smear pass.
  let s0 = project(in.world);
  let s1 = project(in.world + brush * dist * 0.02);
  var dir = s1 - s0;
  dir = select(vec2f(1.0, 0.0), normalize(dir), length(dir) > 1e-3);

  var out: Targets;
  out.pigment = vec4f(A, materialGranulation(in.material));
  out.surface = vec4f(N, wet);
  out.depth = vec2f(dist, in.objectId);
  out.stroke = vec4f(dir, light, f32(in.material));
  return out;
}

// --- Sky wash -----------------------------------------------------------------
// Drawn first as a fullscreen triangle behind everything: a graded wash
// from warm at the horizon to blue overhead, with clouds lifted out as
// paler shapes. It is the wettest wash in the painting.

struct SkyOut {
  @builtin(position) position: vec4f,
};

@vertex
fn skyVs(@builtin(vertex_index) vi: u32) -> SkyOut {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: SkyOut;
  out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
  return out;
}

@fragment
fn skyFs(in: SkyOut) -> Targets {
  let px = in.position.xy;
  let dir = viewRay(px);
  let n = skyNoise(dir, px, 160.0);
  let n2 = skyNoise(dir * 2.3 + 5.0, px, 60.0);
  let h = clamp(dir.y, 0.0, 1.0);
  let blue = vec3f(0.36, 0.58, 0.88);
  let warm = vec3f(0.97, 0.84, 0.62);
  var pigment = mix(warm, blue, smoothstep(0.0, 0.18, h + (n.y - 0.5) * 0.1));
  var density = mix(0.45, 1.25, smoothstep(0.0, 0.45, h));
  // Clouds: lifted-out (paler) patches with a slightly grey belly.
  let cloud = smoothstep(0.55, 0.7, n.x * 0.75 + n2.x * 0.25) * smoothstep(0.03, 0.15, h);
  density *= 1.0 - 0.85 * cloud;
  pigment = mix(pigment, vec3f(0.55, 0.58, 0.72), cloud * smoothstep(0.66, 0.85, n.x) * 0.35);
  density *= 1.0 + (n2.z - 0.5) * 0.4;

  var out: Targets;
  out.pigment = vec4f(absorbance(pigment, density), 0.3);
  out.surface = vec4f(0.0, 0.0, 0.0, 1.0);
  out.depth = vec2f(FAR, SKY_ID);
  out.stroke = vec4f(1.0, 0.0, 1.0, 255.0);
  return out;
}
