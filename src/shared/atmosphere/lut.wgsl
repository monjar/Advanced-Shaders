// LUT lookups and the ray-march integrator shared by the LUT passes and the
// consumer lookups. Requires `atmoTransmittanceLut`, `atmoMultiScatLut` and
// `atmoSampler` (a linear clamp sampler) besides ATMO.

fn atmoTransmittance(r: f32, mu: f32) -> vec3f {
  return textureSampleLevel(atmoTransmittanceLut, atmoSampler, atmoTransmittanceUv(r, mu), 0.0).rgb;
}

// Fraction of the sun disk above the geometric horizon at radius r. The
// Earth's shadow on the atmosphere comes from this term, softened over the
// disk's angular size so the terminator does not alias.
fn atmoSunVisibility(r: f32, muS: f32) -> f32 {
  let sinH = min(ATMO.bottomRadius / r, 1.0);
  let cosH = -sqrt(max(0.0, 1.0 - sinH * sinH));
  return smoothstep(-1.0, 1.0, (muS - cosH) / ATMO.sunAngularRadius);
}

fn atmoTransmittanceToSun(r: f32, muS: f32) -> vec3f {
  return atmoTransmittance(r, muS) * atmoSunVisibility(r, muS);
}

// Hillaire's Psi_ms: radiance scattered towards any direction by all orders
// >= 2, per unit sun illuminance and unit scattering coefficient.
fn atmoMultiScattering(r: f32, muS: f32) -> vec3f {
  return textureSampleLevel(atmoMultiScatLut, atmoSampler, atmoMsUv(r, muS, vec2f(ATMO_MS_SIZE)), 0.0).rgb;
}

struct AtmoScatter {
  inscatter: vec3f,       // per unit sun illuminance
  transmittance: vec3f,
};

// Integrates in-scattered radiance along origin + t dir for t in [t0, t1]:
// single scattering with the Rayleigh and Cornette-Shanks phase functions,
// sun transmittance from the LUT and the Earth's shadow, plus the
// multiple-scattering term. Each segment is integrated analytically for
// constant medium (Hillaire 2015, "Physically Based and Unified Volumetric
// Rendering in Frostbite", slide 28). `quadratic` places samples densely
// near t0 (the camera for views inside the atmosphere). Each segment is
// sampled at its midpoint: Hillaire's 0.3 offset measured a 2.5 % brighter
// sky than the converged reference at the sky-view LUT's 24-40 steps, the
// midpoint 0.1 % (see the README section).
fn atmoIntegrate(origin: vec3f, dir: vec3f, sunDir: vec3f, t0: f32, t1: f32, steps: i32, quadratic: bool) -> AtmoScatter {
  var result: AtmoScatter;
  result.inscatter = vec3f(0.0);
  result.transmittance = vec3f(1.0);
  if (t1 <= t0) { return result; }
  let nu = dot(dir, sunDir);
  let phaseR = atmoRayleighPhase(nu);
  let phaseM = atmoMiePhase(nu);
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
    let p = origin + dir * mix(ta, tb, 0.5);
    let r = length(p);
    let muS = dot(p, sunDir) / r;
    let med = atmoMedium(r - ATMO.bottomRadius);
    let stepT = exp(-med.extinction * dt);
    let sunT = atmoTransmittanceToSun(r, muS);
    let ms = atmoMultiScattering(r, muS);
    let S = sunT * (med.rayleigh * phaseR + med.mie * phaseM) + ms * med.scattering;
    let Sint = (S - S * stepT) / max(med.extinction, vec3f(1e-9));
    result.inscatter += result.transmittance * Sint;
    result.transmittance *= stepT;
  }
  return result;
}

// Sample count that grows with the path length (Hillaire's variable count).
fn atmoStepCount(length: f32, minSteps: f32, maxSteps: f32) -> i32 {
  return i32(clamp(length / 12.0, minSteps, maxSteps));
}
