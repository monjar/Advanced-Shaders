// Ocean surface.
// Vertex: camera-centred geometry clipmap (10 nested levels, each twice as
// coarse), CDLOD-style morphing at level borders, displaced by the FFT
// cascades, shoreline Gerstner waves and the interactive ripple field.
// Fragment: multi-scale normals, Fresnel sky reflection, GGX sun specular with
// glitter, screen-space refraction with Beer-Lambert depth colour,
// subsurface scattering, whitecap / shoreline / contact foam, aerial fog.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> O: Ocean;

@group(1) @binding(0) var dispTex: texture_2d_array<f32>;
@group(1) @binding(1) var derivTex: texture_2d_array<f32>;
@group(1) @binding(2) var linSampler: sampler;
@group(1) @binding(3) var clampSampler: sampler;
@group(1) @binding(4) var terrainTex: texture_2d<f32>;
@group(1) @binding(5) var rippleTex: texture_2d<f32>;
@group(1) @binding(6) var sceneColor: texture_2d<f32>;
@group(1) @binding(7) var sceneDepth: texture_2d<f32>;

const GRID_N: f32 = 128.0;
const LEVELS: u32 = 10u;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) gridXZ: vec2f,
  @location(2) @interpolate(flat) level: u32,
};

fn levelSpacing(level: u32) -> f32 {
  return O.sim.x * exp2(f32(level));
}

// Level centre in units of its own spacing, snapped to even vertices so the
// morph targets line up with the next coarser level.
fn levelCenter(level: u32) -> vec2f {
  let s = levelSpacing(level);
  return floor(F.camPos.xz / (2.0 * s)) * 2.0;
}

@vertex
fn vs(@location(0) ij: vec2f, @builtin(instance_index) level: u32) -> VSOut {
  let s = levelSpacing(level);
  let gi = levelCenter(level) + ij - vec2f(GRID_N * 0.5);
  let local = abs(ij - vec2f(GRID_N * 0.5)) / (GRID_N * 0.5);
  var morph = clamp((max(local.x, local.y) - 0.7) / 0.25, 0.0, 1.0);
  if (level == LEVELS - 1u) { morph = 0.0; }
  // Odd vertices slide onto their even neighbours towards the outer edge.
  let odd = gi - 2.0 * floor(gi * 0.5);
  let g = (gi - odd * morph) * s;

  let d = surfaceDisplacement(g, s * (1.0 + morph));
  let world = vec3f(g.x, 0.0, g.y) + d;

  var out: VSOut;
  out.position = F.viewProj * vec4f(world, 1.0);
  out.worldPos = world;
  out.gridXZ = g;
  out.level = level;
  return out;
}

// --- BRDF ---------------------------------------------------------------

fn fresnelSchlick(cosTheta: f32, f0: f32) -> f32 {
  return f0 + (1.0 - f0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
}

fn ggxD(nDotH: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn smithGGXCorrelated(nDotV: f32, nDotL: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let gv = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
  let gl = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
  return 0.5 / max(gv + gl, 1e-5);
}

fn foamPattern(p: vec2f) -> f32 {
  let t = F.time;
  let a = valueNoise(p * 0.9 + vec2f(t * 0.07, -t * 0.05));
  let b = valueNoise(p * 2.3 - vec2f(t * 0.05, t * 0.09));
  let c = valueNoise(p * 6.1 + vec2f(-t * 0.11, t * 0.13));
  return a * 0.5 + b * 0.32 + c * 0.18;
}

fn levelColor(level: u32) -> vec3f {
  let h = f32(level) * 0.13;
  return 0.5 + 0.5 * cos(TAU * (vec3f(h) + vec3f(0.0, 0.33, 0.67)));
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let g = in.gridXZ;

  // --- Multi-scale normals: sum slopes of all cascades --------------------
  var deriv = vec4f(0.0);
  var foamSim = 0.0;
  for (var c = 0u; c < CASCADES; c++) {
    let uv = g / cascadeLength(c);
    deriv += textureSample(derivTex, linSampler, uv, c);
    foamSim += textureSample(dispTex, linSampler, uv, c).w * O.foam[c];
  }

  let terrain = terrainSample(g);
  let atten = depthAttenuation(max(-terrain.x, 0.0));
  deriv *= atten * O.sim.y;
  var slope = vec2f(deriv.x / max(1.0 + deriv.z, 0.2), deriv.y / max(1.0 + deriv.w, 0.2)) * O.sim.z;
  let shore = shoreWave(g, terrain);
  slope += shore.slope + rippleSlope(g);

  var N = normalize(vec3f(-slope.x, 1.0, -slope.y));
  let toCam = F.camPos - in.worldPos;
  let dist = length(toCam);
  let V = toCam / dist;
  let L = F.sunDir;
  // Keep normals facing the viewer to avoid black silhouettes on steep crests.
  let nv = dot(N, V);
  if (nv < 0.05) { N = normalize(N + V * (0.05 - nv)); }
  let nDotV = max(dot(N, V), 1e-3);
  let nDotL = max(dot(N, L), 0.0);
  let H = normalize(V + L);
  let nDotH = max(dot(N, H), 0.0);

  // --- Reflection --------------------------------------------------------
  let fresnel = fresnelSchlick(nDotV, 0.02);
  var R = reflect(-V, N);
  R.y = abs(R.y);
  let reflection = skyRadiance(R, false);

  // --- Sun specular (GGX) and glitter ------------------------------------
  // Roughness grows with distance to account for sub-pixel waves.
  let rough = clamp(O.shading.x + dist * 0.00012, 0.02, 0.6);
  let alpha = rough * rough;
  let specF = fresnelSchlick(max(dot(V, H), 0.0), 0.02);
  var specular = ggxD(nDotH, alpha) * smithGGXCorrelated(nDotV, nDotL, alpha) * specF * nDotL * F.sunColor;
  // Glitter: sparse, very sharp micro-facets that twinkle as they rotate.
  let cell = floor(g * 5.0);
  let jitter = hash22(cell + floor(F.time * 6.0 + hash21(cell) * 6.0));
  let microN = normalize(N + vec3f(jitter.x - 0.5, 0.0, jitter.y - 0.5) * 0.12);
  let glint = pow(max(dot(microN, H), 0.0), 4000.0) * step(0.93, hash21(cell * 1.37 + 3.1));
  specular += glint * O.shading.y * 60.0 * F.sunColor * specF * smoothstep(400.0, 20.0, dist);

  // --- Refraction and depth colour ----------------------------------------
  let res = F.resolution;
  let uv0 = in.position.xy / res;
  let px0 = vec2i(in.position.xy);
  let depth0 = textureLoad(sceneDepth, px0, 0).r;
  let thickness0 = max(depth0 - dist, 0.0);

  let distortion = N.xz * O.refractStrength * clamp(thickness0 * 0.5, 0.0, 1.0) / (1.0 + dist * 0.05);
  var uv = clamp(uv0 + distortion, vec2f(0.001), vec2f(0.999));
  var sceneDist = textureLoad(sceneDepth, vec2i(uv * res), 0).r;
  if (sceneDist < dist) {
    // The offset landed on something in front of the water; don't refract it.
    uv = uv0;
    sceneDist = depth0;
  }
  let refracted = textureSampleLevel(sceneColor, clampSampler, uv, 0.0).rgb;
  let thickness = min(max(sceneDist - dist, 0.0), 500.0);
  let transmittance = exp(-O.absorption * thickness);
  // Light scattered back out of the water column, lit by sun and sky.
  let inScatter = O.scatterColor * (F.ambient + F.sunColor * max(F.sunDir.y, 0.0) * 0.25);
  var water = refracted * transmittance + inScatter * (1.0 - transmittance);

  // --- Subsurface scattering (Atlas, GDC 2019) ----------------------------
  let crest = max(in.worldPos.y + 0.4, 0.0);
  let sss1 = O.sssStrength * crest * pow(max(dot(L, -V), 0.0), 4.0) * pow(0.5 - 0.5 * dot(L, N), 3.0);
  let sss2 = 0.35 * pow(nDotV, 2.0);
  water += (sss1 + sss2 * O.sssStrength * 0.2) * O.scatterColor * F.sunColor;

  var color = mix(water, reflection, fresnel) + specular;

  // --- Foam --------------------------------------------------------------
  let pattern = foamPattern(g * O.foam.w);
  let contact = 1.0 - smoothstep(0.0, O.shading.w, thickness0);
  let rippleFoam = smoothstep(0.08, 0.35, abs(rippleHeight(g)));
  let coverage = clamp(foamSim + shore.foam * O.shading.z + contact * 0.9 + rippleFoam * 0.5, 0.0, 1.2);
  let foamMask = smoothstep(1.0 - coverage, 1.0 - coverage + 0.3, pattern) * min(coverage * 2.0, 1.0);
  // Foam is a volumetric scatterer: wrapped diffuse keeps it bright at low sun.
  let foamLit = O.foamColor * (F.sunColor * clamp((dot(N, L) + 0.6) / 1.6, 0.0, 1.0) / PI + F.ambient);
  color = mix(color, foamLit, clamp(foamMask * O.foamIntensity, 0.0, 1.0));

  // Very thin water at the contact line is almost fully transparent.
  let edge = smoothstep(0.0, 0.12, thickness0);
  color = mix(textureSampleLevel(sceneColor, clampSampler, uv0, 0.0).rgb, color, edge);

  color = applyFog(color, in.worldPos, O.terrain.w);

  // --- Debug views -------------------------------------------------------
  let mode = u32(F.debugMode + 0.5);
  if (mode == 1u) { color = N * 0.5 + 0.5; }
  else if (mode == 2u) { color = vec3f(foamMask); }
  else if (mode == 3u) { color = vec3f(1.0 - exp(-thickness * 0.15)); }
  else if (mode == 4u) { color = levelColor(in.level) * (0.6 + 0.4 * nDotL); }
  else if (mode == 5u) { color = vec3f(fresnel); }

  if (mode == 7u) { discard; }

  // Clipmap hole: this region is covered by the next finer level.
  if (in.level > 0u) {
    let fineSpacing = levelSpacing(in.level - 1u);
    let fineCenter = levelCenter(in.level - 1u) * fineSpacing;
    let halfWidth = GRID_N * 0.5 * fineSpacing;
    let q = abs(g - fineCenter);
    if (max(q.x, q.y) < halfWidth - fineSpacing * 0.01) { discard; }
  }
  return vec4f(color, 1.0);
}
