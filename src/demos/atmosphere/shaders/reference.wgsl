// Brute-force reference, at 1/refScale resolution: the same scene and the
// same scattering integral as the LUT path, but with every transmittance
// integrated numerically instead of looked up. Per view sample the sun's
// optical depth is marched to the top of the atmosphere (refSunSteps), the
// view ray uses refViewSteps samples, and nothing is cached across pixels.
// The multiple-scattering term (Hillaire's Psi_ms) still comes from its LUT
// when refMultiScattering is on: it is part of the model being compared,
// not of the integration. Ground skylight uses the irradiance LUT on both
// sides. Rays go through the centre of full-resolution pixel (x, y) * refScale
// + refScale / 2, so compare.wgsl can match pixels exactly.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var refOut: texture_storage_2d<rgba16float, write>;

fn refOpticalDepth(p: vec3f, d: vec3f, steps: i32) -> vec3f {
  let r = length(p);
  let tTop = max(atmoRaySphere(r, dot(p, d) / r, ATMO.topRadius).y, 0.0);
  let dt = tTop / f32(steps);
  var tau = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let q = p + d * ((f32(i) + 0.5) * dt);
    tau += atmoMedium(length(q) - ATMO.bottomRadius).extinction * dt;
  }
  return tau;
}

fn refSunTransmittance(p: vec3f) -> vec3f {
  let r = length(p);
  let vis = atmoSunVisibility(r, dot(p, ATMO.sunDir) / r);
  if (vis <= 0.0) { return vec3f(0.0); }
  return exp(-refOpticalDepth(p, ATMO.sunDir, i32(F.refSunSteps))) * vis;
}

fn refIntegrate(origin: vec3f, dir: vec3f, t0: f32, t1: f32, quadratic: bool) -> AtmoScatter {
  var res: AtmoScatter;
  res.inscatter = vec3f(0.0);
  res.transmittance = vec3f(1.0);
  if (t1 <= t0) { return res; }
  let nu = dot(dir, ATMO.sunDir);
  let phaseR = atmoRayleighPhase(nu);
  let phaseM = atmoMiePhase(nu);
  let steps = i32(F.refViewSteps);
  let n = f32(steps);
  for (var i = 0; i < steps; i++) {
    var a = f32(i) / n;
    var b = f32(i + 1) / n;
    if (quadratic) {
      a = a * a;
      b = b * b;
    }
    let ta = mix(t0, t1, a);
    let tb = mix(t0, t1, b);
    let dt = tb - ta;
    let p = origin + dir * (0.5 * (ta + tb));
    let r = length(p);
    let med = atmoMedium(r - ATMO.bottomRadius);
    let stepT = exp(-med.extinction * dt);
    var S = refSunTransmittance(p) * (med.rayleigh * phaseR + med.mie * phaseM);
    if (F.refMultiScattering > 0.5) { S += atmoMultiScattering(r, dot(p, ATMO.sunDir) / r) * med.scattering; }
    res.inscatter += res.transmittance * (S - S * stepT) / max(med.extinction, vec3f(1e-9));
    res.transmittance *= stepT;
  }
  return res;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(refOut);
  if (any(id.xy >= size)) { return; }
  let s = F.refScale;
  let px = vec2f(id.xy) * s + floor(s * 0.5) + 0.5;
  let uv = px / F.resolution;
  let dir = cameraRay(uv);
  let hit = traceScene(dir);
  let seg = atmoCameraSegment(dir);
  let quadratic = seg.x == 0.0;
  var color = vec3f(0.0);
  if (hit.t >= 0.0) {
    let ground = shadeGround(hit, refSunTransmittance(hit.pos), dir);
    if (seg.y > 0.5) {
      let sc = refIntegrate(ATMO.cameraPos, dir, seg.x, min(hit.t, seg.w), quadratic);
      color = ground * sc.transmittance + sc.inscatter * ATMO.sunIlluminance;
    } else {
      color = ground;
    }
  } else {
    var camT = vec3f(1.0);
    if (seg.y > 0.5) {
      let sc = refIntegrate(ATMO.cameraPos, dir, seg.x, seg.w, quadratic);
      color = sc.inscatter * ATMO.sunIlluminance;
      camT = exp(-refOpticalDepth(ATMO.cameraPos + dir * seg.x, dir, i32(F.refViewSteps)));
      if (seg.z >= 0.0) { camT = vec3f(0.0); }
    }
    color += (atmoSunDiskRadiance(dir) + stars(dir)) * camT;
  }
  textureStore(refOut, id.xy, vec4f(color, hit.t));
}
