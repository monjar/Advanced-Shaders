// Primary ray march through the cloud layer.
// One invocation per traced pixel. With the 2x2 update pattern only one pixel
// of each block is traced per frame; resolve.wgsl fills in the rest.
// Output: (in-scattered radiance, transmittance) and a transmittance-weighted
// depth used for reprojection and aerial perspective.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> C: Clouds;
@group(2) @binding(0) var traceOut: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var depthOut: texture_storage_2d<r32float, write>;
@group(2) @binding(2) var history: texture_2d<f32>;

// Fixed cone of offsets for the light march (unit vectors).
const CONE = array<vec3f, 6>(
  vec3f(0.38051305, 0.92453449, -0.02111345),
  vec3f(-0.50625799, -0.03590792, -0.86163418),
  vec3f(-0.32509218, -0.94557439, 0.01428793),
  vec3f(0.09026238, -0.27376545, 0.95755165),
  vec3f(0.28128598, 0.42443639, -0.86065785),
  vec3f(-0.16852403, 0.14748697, 0.97460106),
);

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4), 1.5));
}

// Blend of a strong forward lobe (silver lining) and a weak backward lobe.
fn phase(cosTheta: f32, scale: f32) -> f32 {
  return mix(henyeyGreenstein(cosTheta, -C.phase.y * scale), henyeyGreenstein(cosTheta, C.phase.x * scale), C.phase.z);
}

// Optical depth towards the sun: a short cone of samples with quadratically
// growing spacing; the far samples skip detail noise.
fn sunOpticalDepth(pos: vec3f) -> f32 {
  let steps = i32(C.march.y);
  let dist = C.phase.w;
  var od = 0.0;
  var prev = 0.0;
  for (var i = 0; i < steps; i++) {
    let s = (f32(i) + 1.0) / f32(steps);
    let t = dist * s * s;
    let dt = t - prev;
    let mid = (t + prev) * 0.5;
    prev = t;
    let p = pos + F.sunDir * mid + CONE[i % 6] * (mid * 0.12);
    let alt = length(p) - C.layer.z;
    od += cloudDensity(p, alt, i >= steps / 2, 1.0) * dt;
  }
  // Multiple scattering lets light penetrate further than single-scattering
  // extinction suggests; the absorption multiplier approximates that.
  return od * C.light.x * C.extra.x;
}

// Multiple scattering approximation (Wrenninge et al. 2013, as used by
// Hillaire 2016): successive octaves with less extinction, less energy and
// a more isotropic phase function.
fn sunScattering(opticalDepth: f32, cosTheta: f32) -> f32 {
  var sum = 0.0;
  var a = 1.0;
  var b = 1.0;
  var c = 1.0;
  let octaves = i32(C.ms.x);
  for (var o = 0; o < octaves; o++) {
    sum += b * phase(cosTheta, c) * exp(-opticalDepth * a);
    a *= C.ms.y;
    b *= C.ms.z;
    c *= C.ms.w;
  }
  return sum;
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) gid: vec3u) {
  let traceSize = textureDimensions(traceOut);
  if (any(gid.xy >= traceSize)) { return; }
  let cloudSize = textureDimensions(history);
  let frame = u32(C.temporal.x);
  var px = gid.xy;
  if (u32(C.temporal.y) == 4u) {
    px = min(gid.xy * 2u + bayerOffset(frame), cloudSize - 1u);
  }

  let uv = (vec2f(px) + 0.5) / vec2f(cloudSize);
  let dir = viewRay(uv);
  let segment = layerSegment(dir);
  if (segment.x >= segment.y) {
    textureStore(traceOut, gid.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(depthOut, gid.xy, vec4f(C.layer.w));
    return;
  }

  // More steps along long, grazing paths.
  let horizon = 1.0 - abs(dir.y);
  let steps = i32(C.march.x * mix(1.0, C.march.w, horizon * horizon));
  let dt = (segment.y - segment.x) / f32(steps);
  // Per-pixel, per-frame jitter of the start offset turns banding into noise
  // that the temporal pass averages out.
  let jitter = fract(interleavedGradientNoise(vec2f(px)) + f32(frame % 64u) * 0.61803398875) * C.march.z;
  var t = segment.x + dt * jitter;

  let origin = cameraLocal();
  let cosTheta = dot(dir, F.sunDir);
  var transmittance = 1.0;
  var scattered = vec3f(0.0);
  var depthSum = 0.0;
  var weightSum = 0.0;
  let sigmaT0 = C.light.x;
  let albedo = C.light.y;

  for (var i = 0; i < steps; i++) {
    if (transmittance < 0.01 || t > segment.y) { break; }
    let pos = origin + dir * t;
    let alt = length(pos) - C.layer.z;
    // Probe with the cheap density first; skip the expensive work in empty space.
    if (cloudDensity(pos, alt, true, 0.0) > 0.0) {
      let lod = log2(1.0 + t / 4000.0);
      let density = cloudDensity(pos, alt, false, lod);
      if (density > 0.0) {
        let sigmaT = density * sigmaT0;
        let sigmaS = sigmaT * albedo;
        let lightOD = sunOpticalDepth(pos);
        // Powder: in-scattering builds up with depth, so edges facing away
        // from the viewer's sun direction appear darker.
        let powder = mix(1.0, 1.0 - exp(-2.0 * lightOD), C.light.z * (0.5 - 0.5 * cosTheta));
        let hf = heightFraction(alt);
        let ambient = F.ambient * C.light.w * mix(0.3, 1.0, hf);
        let S = sigmaS * (F.sunColor * sunScattering(lightOD, cosTheta) * powder + ambient);
        // Energy-conserving integration over the step (Hillaire 2015).
        let stepT = exp(-sigmaT * dt);
        let Sint = (S - S * stepT) / sigmaT;
        scattered += transmittance * Sint;
        let absorbed = transmittance * (1.0 - stepT);
        depthSum += t * absorbed;
        weightSum += absorbed;
        transmittance *= stepT;
      }
    }
    t += dt;
  }

  let depth = select(C.layer.w, depthSum / max(weightSum, 1e-6), weightSum > 1e-4);
  // Aerial perspective: distant clouds fade into the horizon sky.
  let fog = 1.0 - exp(-depth * C.view.y);
  let fogColor = skyRadiance(vec3f(dir.x, max(dir.y, 0.0) * 0.5 + 0.02, dir.z), false);
  scattered = mix(scattered, fogColor * (1.0 - transmittance), fog);

  textureStore(traceOut, gid.xy, vec4f(scattered, transmittance));
  textureStore(depthOut, gid.xy, vec4f(depth));
}
