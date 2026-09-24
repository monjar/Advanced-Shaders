// Thin accretion disk in the plane ⟂ F.diskN, from the ISCO outward.
//
// Temperature: the Shakura–Sunyaev / Page–Thorne-like profile for a
// zero-torque inner edge, T ∝ r^(-3/4) (1 - √(r_in/r))^(1/4), normalised to
// its maximum at r = 49/36 r_in. Emission is a blackbody at that temperature.
//
// Turbulence: gradient-noise fBm in (azimuth, ln r), advected with the
// Keplerian angular velocity Ω = √(M/r³). Differential rotation winds any
// pattern up without bound, so two copies of the noise are advected over a
// finite cycle, restarted with a fresh seed, and cross-faded with a variance
// preserving weight (texture advection with periodic reset, as in
// Max and Becker 1995 / Neyret 2003 for flow noise).

const DISK_RADIAL_FREQ: f32 = 11.0;  // noise units per unit of ln r
const DISK_AZIMUTH_R: f32 = 2.6;     // radius of the azimuth circle in noise space (streaks ~4× longer than wide)

fn ssProfile(x: f32) -> f32 {
  return pow(x, -0.75) * pow(max(1.0 - inverseSqrt(x), 0.0), 0.25);
}

fn diskTemperature(r: f32) -> f32 {
  return F.disk.x * ssProfile(r / F.diskIn) / ssProfile(49.0 / 36.0);
}

fn diskAzimuth(X: vec3f) -> f32 {
  let B = cross(F.diskN, F.diskA);
  return atan2(dot(X, B), dot(X, F.diskA));
}

fn diskLayer(r: f32, psi: f32, fpNoise: f32, seed: u32) -> f32 {
  let off = hash33(vec3u(seed, 17u, 5u)) * 200.0;
  let p = vec3f(DISK_AZIMUTH_R * cos(psi), DISK_AZIMUTH_R * sin(psi), DISK_RADIAL_FREQ * log(r)) + off;
  let base = fbmFiltered(p, 5u, fpNoise, 3u);
  // Sharper clumps on top: ridged detail, prefiltered the same way.
  let fine = fbmFiltered(p * vec3f(2.1, 2.1, 1.3) + 7.0, 3u, fpNoise * 2.1, 9u);
  return base + 0.45 * (0.3 - abs(fine));
}

// Zero-mean-ish turbulence in about [-1, 1]. fpNoise: pixel footprint in noise units.
fn diskTurbulence(r: f32, psi: f32, fpNoise: f32) -> f32 {
  let period = F.anim.y;
  let omega = sqrt(0.5 / (r * r * r));
  var acc = 0.0;
  var w2 = 0.0;
  for (var j = 0u; j < 2u; j++) {
    let tau = F.anim.x / period + 0.5 * f32(j);
    let cycle = floor(tau);
    let phase = tau - cycle;
    let w = 1.0 - abs(2.0 * phase - 1.0);
    let seed = u32(i32(cycle) + 65536) * 2u + j;
    acc += w * diskLayer(r, psi - omega * phase * period, fpNoise, seed);
    w2 += w * w;
  }
  return acc / sqrt(max(w2, 1e-4));
}

// Pixel footprint in noise space. The noise coordinates stretch a radial step
// dr by F_r/r and an azimuthal step by R_c/r, so the two pixel differentials
// on the disk (dX, in r_s) are measured in that metric; the larger one sets
// the filter width, as for isotropic mip selection.
fn diskFootprint(X: vec3f, r: f32, dXx: vec3f, dXy: vec3f) -> f32 {
  let rh = X / r;
  let ph = cross(F.diskN, rh);
  let fx = vec2f(dot(dXx, rh) * DISK_RADIAL_FREQ, dot(dXx, ph) * DISK_AZIMUTH_R);
  let fy = vec2f(dot(dXy, rh) * DISK_RADIAL_FREQ, dot(dXy, ph) * DISK_AZIMUTH_R);
  return max(length(fx), length(fy)) / r;
}

// Density in [0, ~3]: radial envelope times turbulence.
fn diskDensity(r: f32, psi: f32, fpNoise: f32) -> f32 {
  let inner = smoothstep(F.diskIn, F.diskIn * 1.12, r);
  let outer = 1.0 - smoothstep(0.62 * F.diskOut, F.diskOut, r);
  let turb = diskTurbulence(r, psi, fpNoise);
  return inner * outer * clamp(0.6 + F.disk.z * 2.2 * turb, 0.02, 3.0);
}

// Redshift factor g = ν_obs / ν_emit for light leaving the disk at X
// (radius r) along the unit static-frame photon direction k.
//  - Doppler: the gas orbits on circular geodesics with local speed
//    β = √(M/(r - 2M)) along φ̂ = N × r̂ (as measured by a static observer),
//    δ = 1 / (γ (1 - β φ̂·k)). This includes the transverse (time dilation) part.
//  - Gravitational: static emitter at r to static observer at r_obs,
//    √(1 - r_s/r) / √(1 - r_s/r_obs); the observer's factor is in gObs.
// Together this equals the textbook √(1 - 3M/r) / (1 - Ω b_z) with b_z the
// photon's angular momentum about the disk axis (see README).
fn diskRedshift(X: vec3f, r: f32, k: vec3f, gObs: f32) -> f32 {
  var g = gObs;
  if (F.diskFx.x > 0.5) {
    let phiHat = normalize(cross(F.diskN, X));
    let beta = sqrt(0.5 / (r - 1.0));
    let gamma = inverseSqrt(1.0 - beta * beta);
    g /= gamma * (1.0 - beta * dot(phiHat, k));
  }
  if (F.diskFx.y > 0.5) {
    g *= sqrt(1.0 - 1.0 / r);
  }
  return g;
}

// Emitted radiance times coverage, and coverage α, for one disk crossing.
// fp: footprint in noise units (diskFootprint). Dense clumps are both more
// opaque and hotter-looking (emission ∝ ρ^0.5 on top of the coverage).
fn diskShade(X: vec3f, r: f32, fp: f32, g: f32) -> vec4f {
  let psi = diskAzimuth(X);
  let dens = diskDensity(r, psi, fp);
  let alpha = 1.0 - exp(-F.disk.y * dens);
  let T = diskTemperature(r);
  var c: vec3f;
  let mode = u32(F.diskFx.z + 0.5);
  // F.anim.w = 1 / luminance at the peak temperature (see index.ts).
  if (mode == 0u) {
    // Blackbody seen through redshift g is a blackbody at g·T (exact).
    c = blackbody(g * T);
  } else if (mode == 1u) {
    c = blackbody(T) * (g * g * g * g);
  } else {
    c = blackbody(T) * (g * g * g);
  }
  return vec4f(c * (F.diskFx.w * F.anim.w * alpha * sqrt(dens)), alpha);
}

// The artistic stage: same turbulence, a fixed orange-to-white ramp, no redshift.
fn diskShadeArtistic(X: vec3f, r: f32, fp: f32) -> vec4f {
  let psi = diskAzimuth(X);
  let dens = diskDensity(r, psi, fp);
  let alpha = 1.0 - exp(-F.disk.y * dens);
  let heat = smoothstep(F.diskOut, F.diskIn, r);
  let c = mix(vec3f(1.0, 0.32, 0.08), vec3f(1.0, 0.85, 0.65), heat * heat) * (0.1 + 1.2 * heat * heat);
  return vec4f(c * (F.diskFx.w * alpha * sqrt(dens)), alpha);
}
