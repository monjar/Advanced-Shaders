// Cloud density field. Requires F, C and the noise bindings below.
// Positions are in the planet-centred local frame (see view.wgsl).

@group(1) @binding(0) var baseNoise: texture_3d<f32>;
@group(1) @binding(1) var detailNoise: texture_3d<f32>;
@group(1) @binding(2) var curlNoise: texture_2d<f32>;
@group(1) @binding(3) var weatherTex: texture_2d<f32>;
@group(1) @binding(4) var noiseSampler: sampler;

struct Weather {
  coverage: f32,
  kind: f32,     // 0 stratus, 0.5 stratocumulus, 1 cumulus
  density: f32,
};

fn sampleWeather(xz: vec2f) -> Weather {
  let w = textureSampleLevel(weatherTex, noiseSampler, (xz + C.weather.yz) / C.weather.x, 0.0);
  var out: Weather;
  // Global coverage slides a threshold over the weather map: 0 clear, 1 overcast.
  out.coverage = smoothstep(0.8 - C.shape.x, 1.3 - C.shape.x, w.r);
  out.kind = clamp(C.shape.y + (w.g - 0.5) * 0.8, 0.0, 1.0);
  out.density = mix(0.6, 1.2, w.b);
  return out;
}

// Vertical density profile for each cloud type, blended by `kind`.
fn heightProfile(hf: f32, kind: f32) -> f32 {
  let stratus = smoothstep(0.0, 0.07, hf) * smoothstep(0.28, 0.14, hf);
  let strato = smoothstep(0.0, 0.12, hf) * smoothstep(0.6, 0.35, hf);
  let cumulus = smoothstep(0.0, 0.1, hf) * smoothstep(1.0, 0.65, hf);
  let t = kind * 2.0;
  return select(mix(strato, cumulus, t - 1.0), mix(stratus, strato, t), t < 1.0);
}

fn heightFraction(altitude: f32) -> f32 {
  return (altitude - C.layer.x) / (C.layer.y - C.layer.x);
}

// Density at `pos`. `cheap` skips detail erosion (used for empty-space
// probing, distant light samples and the shadow map). `lod` selects the
// detail-noise mip to fight aliasing far away.
fn cloudDensity(pos: vec3f, altitude: f32, cheap: bool, lod: f32) -> f32 {
  let hf = heightFraction(altitude);
  if (hf <= 0.0 || hf >= 1.0) { return 0.0; }
  // Cloud tops lean downwind.
  let xz = F.camPos.xz + pos.xz - C.windDir.xy * (hf * C.weather.w);
  let w = sampleWeather(xz);
  if (w.coverage <= 0.001) { return 0.0; }

  // Base shape: Perlin-Worley eroded by low-frequency Worley fBm.
  let p = vec3f(xz.x + C.wind.x, altitude, xz.y + C.wind.y) * C.shape.w;
  let low = textureSampleLevel(baseNoise, noiseSampler, p, 0.0);
  let lowFbm = low.g * 0.625 + low.b * 0.25 + low.a * 0.125;
  var base = remap(low.r, lowFbm - 1.0, 1.0, 0.0, 1.0);
  base *= heightProfile(hf, w.kind);
  // Coverage: only the densest parts survive where the weather map is sparse.
  base = clamp(remap(base, 1.0 - w.coverage, 1.0, 0.0, 1.0), 0.0, 1.0) * w.coverage;
  if (base <= 0.0) { return 0.0; }
  if (cheap) { return base * w.density * C.shape.z; }

  // Detail erosion: curl-distorted Worley, wispy at the base and billowy on top.
  let curl = textureSampleLevel(curlNoise, noiseSampler, xz * C.detail.z, 0.0).xy * C.detail.w * (1.0 - hf);
  let pd = vec3f(xz.x + C.wind.z + curl.x, altitude, xz.y + C.wind.w + curl.y) * C.detail.x;
  let hi = textureSampleLevel(detailNoise, noiseSampler, pd, lod).rgb;
  let hiFbm = hi.r * 0.625 + hi.g * 0.25 + hi.b * 0.125;
  let hiMod = mix(hiFbm, 1.0 - hiFbm, clamp(hf * 8.0, 0.0, 1.0));
  let d = clamp(remap(base, hiMod * C.detail.y, 1.0, 0.0, 1.0), 0.0, 1.0);
  return d * w.density * C.shape.z;
}
