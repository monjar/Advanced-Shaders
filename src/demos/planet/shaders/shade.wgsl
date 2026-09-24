// Deferred shading, one thread per pixel. For terrain pixels the full
// height function is evaluated again, per pixel and in the camera's noise
// frame, with analytic derivatives: the normal, height, biome and snow come
// from all octaves the pixel footprint resolves, independent of the mesh
// LOD, so LOD changes only move silhouettes and never the lighting. Then
// ocean, clouds (a 2D-on-sphere layer with optical depth), city lights, sun
// shadow cascades and the atmosphere (sky, sun transmittance, aerial
// perspective).

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> SLOTS: array<SlotStatic, SLOT_COUNT>;
@group(0) @binding(2) var<uniform> CAM: array<NoiseSlot, SLOT_COUNT>;
@group(0) @binding(3) var gbuf0: texture_2d<f32>;
@group(0) @binding(4) var gbuf1: texture_2d<f32>;
@group(0) @binding(5) var hdrOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var shadowMap: texture_depth_2d_array;
@group(0) @binding(7) var shadowSampler: sampler_comparison;

fn tableSlot(s: u32) -> NoiseSlot {
  return CAM[s];
}

fn shape() -> TerrainShape {
  return TerrainShape(F.shape.x, F.shape.y, F.shape.z, 0.0);
}

// An octave counts fully while its wavelength spans >= 4 pixel footprints
// and fades out by 2 (wavelength of level k: 2^(22 - k) m).
fn footprintWeights(fp: f32) -> vec4f {
  return vec4f(21.0 - log2(max(fp, 1e-4)), -1.0, 23.0, fp);
}

const NO_WEIGHTS: vec4f = vec4f(0.0, -1.0, -1.0, 0.0);

// fBm over `count` slots of a channel, footprint-weighted.
fn channelFbmGain(base: u32, count: u32, x: vec3f, w: vec4f, gain: f32) -> vec4f {
  var sum = vec4f(0.0);
  var a = 0.5;
  for (var i = 0u; i < count; i++) {
    let s = base + i;
    let wo = octaveWeight(slotLevel(s), w);
    if (wo <= 0.0) { break; }
    sum += (wo * a) * slotNoise(s, x);
    a *= gain;
  }
  return sum;
}

fn channelFbm(base: u32, count: u32, x: vec3f, w: vec4f) -> vec4f {
  return channelFbmGain(base, count, x, w, 0.5);
}

// fBm value plus the variance of the octaves the footprint removed
// (gradient noise has a variance of ~0.06), so thresholds of it can be
// filtered: E[step(x + noise)] ~ a step widened by the missing sigma.
fn channelFbmVar(base: u32, count: u32, x: vec3f, w: vec4f) -> vec2f {
  var sum = 0.0;
  var lost = 0.0;
  var a = 0.5;
  for (var i = 0u; i < count; i++) {
    let s = base + i;
    let wo = octaveWeight(slotLevel(s), w);
    lost += (1.0 - wo) * a * a * 0.06;
    if (wo > 0.0) { sum += wo * a * slotNoise(s, x).x; }
    a *= 0.5;
  }
  return vec2f(sum, lost);
}

fn filteredStep(e0: f32, e1: f32, x: f32, variance: f32) -> f32 {
  let s = 1.5 * sqrt(variance);
  return smoothstep(e0 - s, e1 + s, x);
}

fn cameraRay(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let q = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(q.xyz / q.w);
}

fn toKm(rel: vec3f) -> vec3f {
  return (F.camPos + rel) * 1e-3;
}

// ---- Biomes -----------------------------------------------------------------

struct Surface {
  albedo: vec3f,
  snow: f32,
  habit: f32,       // habitability for city lights
  rock: f32,
  moisture: f32,
  temperature: f32,
};

// Temperature from latitude (and a lapse rate of 6.5 K/km), moisture from
// the circulation cells (wet equator and 60°, dry 30° and poles), distance
// to the coast and noise; biomes are blended from both, then rock on steep
// slopes, beaches, and snow above a latitude-dependent snow line that
// steep slopes shed.
fn biome(t: Terrain, n: vec3f, up: vec3f, xs: vec3f, fp: f32) -> Surface {
  var s: Surface;
  let w = footprintWeights(fp);
  let alat = abs(asin(clamp(up.y, -1.0, 1.0)));
  let slope = 1.0 - dot(n, up);
  let h = t.h;
  let moistN = channelFbm(SLOT_MOIST, 5u, xs, w).x;
  let tempN = channelFbm(SLOT_TEMP, 3u, xs, w).x;
  let T = 29.0 - 50.0 * pow(sin(alat), 1.4) - 0.0065 * max(h, 0.0) + 9.0 * tempN;
  let band = 0.5 + 0.32 * cos(alat * 6.0);
  let coast = 1.0 - smoothstep(0.02, 0.3, t.land);
  let M = clamp(0.75 * band + 0.3 * coast + 1.1 * moistN - 0.14 + F.misc.w, 0.0, 1.0);
  s.moisture = M;
  s.temperature = T;

  let desert = mix(vec3f(0.34, 0.22, 0.13), vec3f(0.45, 0.37, 0.26), smoothstep(-0.25, 0.25, tempN * 2.0 + moistN));
  let savanna = vec3f(0.27, 0.23, 0.11);
  let grass = vec3f(0.1, 0.16, 0.055);
  let forest = vec3f(0.04, 0.08, 0.03);
  let jungle = vec3f(0.022, 0.065, 0.02);
  let taiga = vec3f(0.03, 0.055, 0.035);
  let tundra = vec3f(0.16, 0.15, 0.11);
  let rock = vec3f(0.19, 0.17, 0.15) * (0.8 + 0.5 * smoothstep(-0.3, 0.3, t.ridge - 0.4));
  let sand = vec3f(0.42, 0.37, 0.28);

  let hot = smoothstep(14.0, 22.0, T);
  let cold = 1.0 - smoothstep(-3.0, 5.0, T);
  let wet = smoothstep(0.4, 0.7, M);
  let dry = 1.0 - smoothstep(0.12, 0.3, M);
  var a = mix(grass, forest, wet);
  a = mix(a, mix(savanna, jungle, wet), hot);
  a = mix(a, desert, dry * (1.0 - cold));
  let boreal = mix(tundra, taiga, smoothstep(0.3, 0.5, M) * smoothstep(-6.0, 2.0, T));
  a = mix(a, boreal, cold);

  // Surface variation at every scale down to the footprint: patches of
  // darker and lighter vegetation, soil and scree.
  // A flatter spectrum (gain 0.78) than terrain, so patches stay visible
  // at every distance instead of fading like fractal height does.
  // (Normalised by the total amplitude of 13 octaves, ~2.1.)
  let av = channelFbmGain(SLOT_ALBEDO, 13u, xs, w, 0.78).x / 2.1;
  a = mix(a, a * vec3f(0.5, 0.68, 0.45), smoothstep(-0.02, 0.08, av) * (1.0 - dry) * (1.0 - cold));
  a *= 1.0 + 2.2 * clamp(av, -0.2, 0.2);

  let rockW = max(smoothstep(0.28, 0.5, slope + av), smoothstep(2200.0, 3200.0, h) * smoothstep(0.08, 0.2, slope));
  a = mix(a, rock * (1.0 + 2.0 * clamp(av, -0.2, 0.2)), rockW);
  s.rock = rockW;
  let beach = (1.0 - smoothstep(0.6, 2.0, h + 8.0 * av)) * (1.0 - smoothstep(0.05, 0.15, slope)) * (1.0 - cold);
  a = mix(a, sand, beach);

  // Patchy snow line (noise of every scale), shed from slopes steeper than
  // ~20-25 degrees so rock ribs show through high snowfields.
  let snowLine = 5900.0 * pow(cos(alat), 2.0) - 500.0 + F.misc.z + 500.0 * tempN + 2500.0 * av;
  let snowAlt = smoothstep(snowLine - 150.0, snowLine + 150.0, h);
  let ice = 1.0 - smoothstep(-15.0, -9.0, T);
  let snow = max(snowAlt, ice) * (1.0 - smoothstep(0.2, 0.4, slope + 1.5 * av));
  s.snow = snow;
  s.albedo = mix(a, vec3f(0.8, 0.82, 0.86), snow);
  s.habit = smoothstep(-2.0, 8.0, T) * (1.0 - 0.6 * smoothstep(27.0, 33.0, T)) * smoothstep(0.1, 0.3, M) *
    (1.0 - smoothstep(700.0, 1800.0, h)) * (1.0 - snow) * (1.0 - rockW) * (0.3 + 0.7 * coast);
  return s;
}

// ---- Clouds -----------------------------------------------------------------

// Vertical optical depth of the cloud layer at a point on its shell (given
// camera-relative). Coverage: domain-swirled fBm (a displacement along the
// curl of the lowest octave, rotating features into cyclone-like spirals)
// plus latitude bands (ITCZ, subtropical highs, storm tracks); the slots are
// advected over time on the CPU (per-octave wind offsets in the table).
fn cloudDepth(x: vec3f, up: vec3f, fp: f32) -> f32 {
  if (F.cloud2.y < 0.5) { return 0.0; }
  let w = footprintWeights(fp);
  let n0 = slotNoise(SLOT_CLOUD, x);
  let gT = (n0.yzw - up * dot(n0.yzw, up)) / SLOTS[SLOT_CLOUD].freq;
  let xw = x + cross(up, gT) * F.cloud2.x;
  // Large systems from the three lowest octaves; the rest only shapes
  // their edges and the density inside.
  let big = 0.55 * n0.x + 0.35 * slotNoise(SLOT_CLOUD + 1u, xw).x + 0.2 * slotNoise(SLOT_CLOUD + 2u, xw).x;
  let sm = channelFbmVar(SLOT_CLOUD + 3u, 10u, xw, w);
  let small = sm.x;
  let lat = asin(clamp(up.y, -1.0, 1.0));
  let band = 0.14 * cos(lat * 6.0) - 0.04;
  let cov = (big + 0.6 * small) * 1.7 + band + (F.cloud.x - 0.5) * 1.1;
  let d = filteredStep(-0.05, 0.55, cov, sm.y * 1.04);
  return F.cloud.z * d * d * (0.45 + 0.55 * filteredStep(-0.1, 0.14, small, sm.y));
}

// Distances where the ray (origin at altitude h0, zenith cosine mu) crosses
// the cloud shell; t1 < t0 when it does not.
fn cloudShell(h0: f32, mu: f32) -> vec2f {
  let hc = F.cloud.y;
  let R = PLANET_R;
  return atmoRaySphereC(R + h0, mu, (h0 - hc) * (2.0 * R + h0 + hc));
}

// Direct-sun transmittance through the clouds at a surface point.
fn cloudShadow(rel: vec3f, up: vec3f, h: f32) -> f32 {
  let muS = dot(up, F.sunDir);
  if (F.cloud2.y < 0.5 || h >= F.cloud.y || muS <= 0.0) { return 1.0; }
  let s = cloudShell(h, muS);
  if (s.y < 0.0) { return 1.0; }
  let x = rel + F.sunDir * s.y;
  let tau = cloudDepth(x, normalize(F.camPos + x), 2000.0);
  return mix(1.0, exp(-tau / max(muS, 0.05)), F.cloud.w);
}

fn hgPhase(mu: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

// Composite the cloud layer where the view ray crosses it before `tMax`.
// The slab is lit with a two-stream estimate for a conservative scatterer
// (g ~ 0.85): reflectance R = t'/(2 + t') with t' = (1 - g) tau, diffuse
// transmittance 1 - R - exp(-tau/mu_s); the viewer sees R on the sun's
// side and the diffuse transmission from the other. A forward Henyey-
// Greenstein term brightens thin edges towards the sun.
fn compositeClouds(color: vec3f, dir: vec3f, uv: vec2f, tMax: f32) -> vec3f {
  if (F.cloud2.y < 0.5) { return color; }
  let camAlt = F.misc.x;
  let upCam = normalize(F.camPos);
  let s = cloudShell(camAlt, dot(upCam, dir));
  if (s.y < s.x) { return color; }
  var c = color;
  // Far crossing first, then the near one.
  for (var k = 0; k < 2; k++) {
    let t = select(s.x, s.y, k == 0);
    if (t <= 0.0 || t >= tMax) { continue; }
    let x = dir * t;
    let up = normalize(F.camPos + x);
    let muV = abs(dot(dir, up));
    let fp = t * F.pixelAngle / max(muV, 0.1);
    let tau = cloudDepth(x, up, fp);
    if (tau < 0.02) { continue; }
    let pos = toKm(x);
    let muS = dot(up, F.sunDir);
    let Tv = exp(-tau / max(muV, 0.03));
    let sunE = atmoSunIlluminance() * atmoSunTransmittanceAt(pos);
    // Two-stream-style reflectance, rising towards grazing sun.
    let ts = 0.15 * tau;
    let R = ts / (ts + 0.5 + 1.5 * abs(muS));
    let viewerAbove = camAlt > F.cloud.y;
    let sunAbove = muS > 0.0;
    var frac = R;
    if (viewerAbove != sunAbove) { frac = max(1.0 - R - exp(-tau / max(abs(muS), 0.05)), 0.0); }
    var L = sunE * abs(muS) / PI * frac;
    // Forward single scattering through thin parts, lit through the cloud's own depth.
    L += 0.5 * sunE * hgPhase(dot(dir, F.sunDir), 0.6) * (1.0 - Tv) * exp(-0.5 * tau / max(abs(muS), 0.05));
    L += atmoSkyIrradiance(pos, up) / PI * (0.35 + 0.65 * R) * (1.0 - Tv);
    let ap = atmoAerialPerspective(uv, dir, t * 1e-3);
    c = Tv * c + (1.0 - Tv) * ap.inscatter + ap.transmittance * L;
  }
  return c;
}

// ---- City lights -------------------------------------------------------------

// Cities as 3D Gaussian balls, one candidate per cell of a lattice level,
// that the surface slices through. The pixel footprint blurs each ball with
// its mass conserved ((r/r_e)^3), so a city is a sharp spot close up and a
// dimmer, wider one further away; once the footprint spans a sizeable part
// of a cell the eight-cell sum is replaced by its mean density, the correct
// average glow from orbit. Cells come from the camera's lattice split, so
// cities are exact and stable at every altitude.
fn cityScale(s: u32, xs: vec3f, fp: f32, density: f32, radiusCells: f32) -> f32 {
  let st = SLOTS[s];
  let t = tableSlot(s);
  let q = t.frac.xyz + vec3f(dot(st.r0.xyz, xs), dot(st.r1.xyz, xs), dot(st.r2.xyz, xs)) * st.freq;
  let fpc = fp * st.freq;
  // Mean over sizes: E[(0.3 + 2 s^2)(0.3 + 0.7 s^2)^3] = 0.364; pi^1.5 = 5.568.
  let mean = density * 0.364 * 5.568 * radiusCells * radiusCells * radiusCells;
  let blend = smoothstep(0.25, 0.6, fpc);
  if (blend >= 1.0) { return mean; }
  let base = floor(q - 0.5);
  var sum = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let cf = base + vec3f(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
    let h = pcg3d(bitcast<vec3u>(t.cell.xyz + vec3i(cf)) + st.seed);
    let r = vec3f(h >> vec3u(8u)) / 16777215.0;
    if (fract(r.x * 91.7 + r.y * 13.3) > density) { continue; }
    let size = r.x;
    let centre = cf + 0.5 + (vec3f(r.y, r.z, fract(r.x * 37.1)) - 0.5) * 0.3;
    let rad = radiusCells * (0.3 + 0.7 * size * size);
    let re2 = rad * rad + fpc * fpc;
    let dq = q - centre;
    sum += (0.3 + 2.0 * size * size) * exp(-dot(dq, dq) / re2) * pow(rad * rad / re2, 1.5);
  }
  return mix(sum, mean, blend);
}

// ---- Terrain shadows ---------------------------------------------------------

// Sun visibility from the cascades: cascade 0 where the point projects
// inside it (blending into cascade 1 near its border), 3x3 PCF, with a
// normal offset of ~1.5 texels against acne.
fn sampleCascade(k: i32, rel: vec3f) -> vec4f {
  var m = F.shadowMat0;
  if (k == 1) { m = F.shadowMat1; }
  let c = m * vec4f(rel, 1.0);
  return vec4f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5, c.z, 0.0);
}

fn pcf(k: i32, c: vec4f) -> f32 {
  let texel = 1.0 / f32(textureDimensions(shadowMap).x);
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum += textureSampleCompareLevel(shadowMap, shadowSampler, c.xy + vec2f(f32(x), f32(y)) * texel, k, c.z);
    }
  }
  return sum / 9.0;
}

fn terrainShadow(rel: vec3f, n: vec3f) -> f32 {
  if (F.shadow.z < 0.5) { return 1.0; }
  let c0 = sampleCascade(0, rel + n * (1.5 * F.shadow.x));
  let edge0 = max(abs(c0.x - 0.5), abs(c0.y - 0.5));
  let c1 = sampleCascade(1, rel + n * (1.5 * F.shadow.y));
  let edge1 = max(abs(c1.x - 0.5), abs(c1.y - 0.5));
  var s1 = 1.0;
  if (edge1 < 0.49 && c1.z > 0.0 && c1.z < 1.0) { s1 = pcf(1, c1); }
  if (edge0 < 0.49 && c0.z > 0.0 && c0.z < 1.0) {
    return mix(pcf(0, c0), s1, smoothstep(0.4, 0.49, edge0));
  }
  return s1;
}

// ---- Ocean ------------------------------------------------------------------

// Wind waves: animated fBm slopes over the wave slots; the slope variance
// the pixel cannot resolve widens the GGX lobe instead (a LEAN-like
// roughness), so the glint broadens smoothly from orbit.
fn waterNormal(xs: vec3f, up: vec3f, fp: f32) -> vec4f {
  let w = footprintWeights(fp);
  var g = vec3f(0.0);
  var unresolved = 0.0;
  var a = 1.0;
  for (var i = 0u; i < 6u; i++) {
    let s = SLOT_WAVE + i;
    let wo = octaveWeight(slotLevel(s), w);
    let slopeAmp = F.ocean.x * a;
    unresolved += (1.0 - wo) * slopeAmp * slopeAmp;
    if (wo > 0.0) {
      let n = slotNoise(s, xs);
      g += wo * slopeAmp * n.yzw / SLOTS[s].freq;
    }
    a *= 0.7;
  }
  let gT = g - up * dot(g, up);
  let rough = sqrt(F.ocean.y * F.ocean.y + unresolved * 0.5);
  return vec4f(normalize(up - gT), clamp(rough, 0.02, 0.6));
}

fn ggx(n: vec3f, v: vec3f, l: vec3f, rough: f32) -> f32 {
  let h = normalize(v + l);
  let ndl = max(dot(n, l), 0.0);
  let ndv = max(dot(n, v), 1e-3);
  let ndh = max(dot(n, h), 0.0);
  let a2 = rough * rough * rough * rough;
  let d = ndh * ndh * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * d * d);
  let k = rough * rough * 0.5;
  let G = ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
  let Fr = 0.02 + 0.98 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
  return D * G * Fr / (4.0 * ndv);
}

// ---- Stars ------------------------------------------------------------------

fn stars(dir: vec3f) -> vec3f {
  if (F.misc.y <= 0.0) { return vec3f(0.0); }
  let p = dir * 120.0;
  let cell = floor(p);
  let h = pcg3d(bitcast<vec3u>(vec3i(cell)) + vec3u(7u, 11u, 13u));
  let r = vec3f(h >> vec3u(8u)) / 16777215.0;
  if (r.x < 0.99) { return vec3f(0.0); }
  let starPos = normalize(cell + 0.2 + 0.6 * r);
  let d = length(dir - starPos) / (1.2 * F.pixelAngle);
  let mag = pow((r.x - 0.99) / 0.01, 8.0);
  let tint = mix(vec3f(1.0, 0.8, 0.6), vec3f(0.7, 0.8, 1.0), r.y);
  return tint * mag * exp(-d * d) * 5e-4 * F.misc.y;
}

// ---- Main -------------------------------------------------------------------

fn heightColour(h: f32) -> vec3f {
  if (h < 0.0) { return mix(vec3f(0.3, 0.6, 0.9), vec3f(0.0, 0.03, 0.2), smoothstep(0.0, 5000.0, -h)); }
  let k = h / 6000.0;
  return mix(mix(vec3f(0.1, 0.5, 0.1), vec3f(0.6, 0.45, 0.2), smoothstep(0.0, 0.4, k)), vec3f(1.0), smoothstep(0.4, 1.0, k));
}

fn levelColour(level: f32) -> vec3f {
  let h = fract(level * 0.137 + 0.1) * 6.0;
  return clamp(vec3f(abs(h - 3.0) - 1.0, 2.0 - abs(h - 2.0), 2.0 - abs(h - 4.0)), vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(vec2f(id.xy) >= F.resolution)) { return; }
  let px = vec2i(id.xy);
  let uv = (vec2f(id.xy) + 0.5) / F.resolution;
  let mode = u32(F.debugMode + 0.5);
  let g0 = textureLoad(gbuf0, px, 0);
  let g1 = textureLoad(gbuf1, px, 0);

  if (g1.a < 0.5) {
    // Sky: atmosphere (LUT or ray marched), sun disk, stars, clouds.
    let dir = cameraRay(uv);
    var sky = atmoSkyRadiance(dir) + atmoSunDisk(dir) + stars(dir) * atmoCameraTransmittance(dir);
    if (mode == 6u) { sky = vec3f(0.0); }
    sky = compositeClouds(sky, dir, uv, 1e12);
    if (mode != 0u && mode != 6u) { sky = vec3f(0.01, 0.01, 0.02); }
    textureStore(hdrOut, px, vec4f(sky, -1.0));
    return;
  }

  let rel = g0.xyz;
  let dist = length(rel);
  let dir = rel / dist;
  let up = normalize(F.camPos + rel);
  let xs = rel - g0.w * up;
  // Pixel footprint on the surface: the distance to the neighbouring
  // pixels' surface points (catches slopes seen edge-on, where the octaves
  // would otherwise alias into sparkles), bounded by the view-ray estimate.
  var fp = dist * F.pixelAngle / max(abs(dot(dir, up)), 0.2);
  let size = vec2i(F.resolution) - 1;
  let nx = textureLoad(gbuf0, min(px + vec2i(1, 0), size), 0);
  let ny = textureLoad(gbuf0, min(px + vec2i(0, 1), size), 0);
  if (dot(nx.xyz, nx.xyz) > 0.0 && dot(ny.xyz, ny.xyz) > 0.0) {
    let d = max(length(nx.xyz - rel), length(ny.xyz - rel));
    fp = clamp(d, dist * F.pixelAngle, dist * F.pixelAngle * 20.0);
  }
  let t = terrainEval(xs, shape(), footprintWeights(fp), NO_WEIGHTS);
  let gT = t.grad - up * dot(t.grad, up);
  let n = normalize(up - gT);
  let pos = toKm(rel);
  let v = -dir;
  let muS = dot(up, F.sunDir);

  let sunE = atmoSunIlluminance() * atmoSunTransmittanceAt(pos);
  var cs = cloudShadow(rel, up, max(t.h, 0.0));
  let ts = select(1.0, terrainShadow(rel, up), muS > -0.05);
  let cloudOnly = cs;
  cs *= ts;
  var color: vec3f;
  var albedo: vec3f;
  var s: Surface;
  let night = smoothstep(0.05, -0.12, muS);

  if (t.h < 0.0) {
    // Ocean: Beer-Lambert over a sandy shelf near coasts, deep blue offshore.
    let depth = -t.h;
    let wn = waterNormal(xs, up, fp);
    let wnorm = wn.xyz;
    let ndv = max(dot(wnorm, v), 1e-3);
    let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let absorb = vec3f(0.45, 0.09, 0.05);
    let path = depth * (1.0 + 1.0 / max(dot(up, v), 0.2));
    let seabed = vec3f(0.5, 0.45, 0.33) * exp(-absorb * path);
    let deep = vec3f(0.004, 0.018, 0.035);
    albedo = seabed + deep * (1.0 - exp(-absorb * path * 0.5));
    let skyE = atmoSkyIrradiance(pos, up);
    let ndl = max(dot(wnorm, F.sunDir), 0.0);
    color = albedo / PI * (sunE * cs * max(muS, 0.0) + skyE) * (1.0 - fres);
    color += sunE * cs * ggx(wnorm, v, F.sunDir, wn.w) * ndl;
    // Sky reflection: the camera's sky view when low, a uniform sky above.
    // Keep reflections above the horizon: steep wave facets seen at grazing
    // angles would otherwise reflect the ground below it (dark streaks).
    var refl = reflect(dir, wnorm);
    let ru = dot(refl, up);
    if (ru < 0.02) { refl = normalize(refl + up * (0.02 - ru)); }
    var skyL = skyE / PI;
    if (F.misc.x < 20000.0 && dist < 60000.0) { skyL = atmoSkyRadiance(refl); }
    color += fres * skyL;
    s.snow = 0.0;
    s.habit = 0.0;
    // Sea ice near the poles.
    let alat = abs(asin(clamp(up.y, -1.0, 1.0)));
    let seaIce = smoothstep(1.2, 1.3, alat + 0.1 * channelFbm(SLOT_TEMP, 3u, xs, footprintWeights(fp)).x);
    if (seaIce > 0.0) {
      let iceC = vec3f(0.75, 0.78, 0.82) / PI * (sunE * cs * max(muS, 0.0) + skyE);
      color = mix(color, iceC, seaIce);
      albedo = mix(albedo, vec3f(0.75, 0.78, 0.82), seaIce);
    }
  } else {
    s = biome(t, n, up, xs, fp);
    albedo = s.albedo;
    let ndl = max(dot(n, F.sunDir), 0.0);
    color = albedo / PI * (sunE * cs * ndl + atmoSkyIrradiance(pos, n) * (0.6 + 0.4 * cs));
    // City lights on the night side: cities and towns scattered over
    // habitable, low-lying land (biased to coasts), clustered by a regional
    // population field.
    if (s.habit > 0.01 && night > 0.0) {
      let w = footprintWeights(fp);
      let region = channelFbmVar(SLOT_CITY, 3u, xs, w);
      let n2 = channelFbmVar(SLOT_CITY + 7u, 5u, xs, w);
      let populated = filteredStep(-0.2, 0.35, region.x + 0.2 * s.habit - 0.1 + (F.city.y - 0.5) * 0.3, region.y);
      let dens = populated * s.habit;
      // Metropolises (33 km cells), cities (8 km) and towns (2 km): a few
      // large ones stay distinct points from orbit, the rest add up.
      let metro = cityScale(SLOT_CITY + 2u, xs, fp, dens * 0.3, 0.08);
      let big = cityScale(SLOT_CITY + 4u, xs, fp, dens * 0.45, 0.13);
      let towns = cityScale(SLOT_CITY + 6u, xs, fp, dens * 0.4, 0.1);
      let streets = 0.25 + 0.75 * filteredStep(-0.05, 0.2, n2.x, n2.y);
      let glow = (10.0 * metro + 1.4 * big + 0.45 * towns) * streets + 0.002 * dens;
      color += vec3f(1.0, 0.72, 0.42) * (glow * night * F.city.x);
    }
  }

  color = atmoApply(color, uv, dir, dist * 1e-3);
  if (mode == 6u) { color = vec3f(0.0); }
  color = compositeClouds(color, dir, uv, dist);

  if (mode == 1u) {
    color = heightColour(t.h);
  } else if (mode == 2u) {
    color = albedo * 2.0;
  } else if (mode == 3u) {
    color = n * 0.5 + 0.5;
  } else if (mode == 4u) {
    let edge = select(1.0, 0.6, g1.y > 0.5);
    color = levelColour(g1.x * 32.0) * (0.55 + 0.45 * g1.z) * edge;
  } else if (mode == 5u) {
    color = vec3f(cloudOnly, cs, ts);
  } else if (mode == 7u) {
    color = vec3f(s.habit, 0.0, 0.0) + select(vec3f(0.0), vec3f(0.0, 0.0, 0.4), t.h < 0.0);
  }
  if (mode != 0u && mode != 6u) {
    color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(2.2));
  }
  textureStore(hdrOut, px, vec4f(color, dist));
}
