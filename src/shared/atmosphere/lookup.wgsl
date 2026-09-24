// Consumer-side API. Requires the bindings declared by atmosphereWgsl().
// Directions are unit world-space vectors; distances in km; positions in km
// relative to the planet centre. Radiances are absolute (sun illuminance
// applied), except where noted.

// Sky radiance seen from the camera, without the sun disk. Inside the
// atmosphere this is one sky-view LUT fetch; above ATMO.rayMarchAltitude (in
// orbit, where the sky is a thin shell seen edge-on and the LUT's angular
// resolution would smear the limb) it is ray marched per pixel.
fn atmoSkyRadiance(dir: vec3f) -> vec3f {
  if (ATMO.cameraAltitude < ATMO.rayMarchAltitude) {
    return textureSampleLevel(atmoSkyViewLut, atmoSampler, atmoSkyViewLookupUv(dir), 0.0).rgb;
  }
  let seg = atmoCameraSegment(dir);
  if (seg.y < 0.5) { return vec3f(0.0); }
  let steps = atmoStepCount(seg.w - seg.x, 16, 48);
  return atmoIntegrate(ATMO.cameraPos, dir, ATMO.sunDir, seg.x, seg.w, steps, false).inscatter * ATMO.sunIlluminance;
}

fn atmoSkyViewLookupUv(dir: vec3f) -> vec2f {
  let up = normalize(ATMO.cameraPos);
  let h = atmoSkyViewAltitude();
  let cosZ = dot(dir, up);
  let dirT = dir - up * cosZ;
  let sunT = ATMO.sunDir - up * dot(ATMO.sunDir, up);
  let ld = length(dirT);
  let ls = length(sunT);
  var cosAz = 1.0;
  if (ld > 1e-5 && ls > 1e-5) { cosAz = clamp(dot(dirT, sunT) / (ld * ls), -1.0, 1.0); }
  return atmoSkyViewUv(ATMO.bottomRadius + h, h, cosZ, cosAz);
}

// Transmittance from the camera to space along dir (1 when the ray misses
// the atmosphere, 0 when it hits the ground).
fn atmoCameraTransmittance(dir: vec3f) -> vec3f {
  let seg = atmoCameraSegment(dir);
  if (seg.y < 0.5) { return vec3f(1.0); }
  if (seg.z >= 0.0) { return vec3f(0.0); }
  let p = ATMO.cameraPos + dir * seg.x;
  let r = select(length(p), ATMO.bottomRadius + ATMO.cameraAltitude, seg.x == 0.0);
  return atmoTransmittance(r, dot(p, dir) / length(p));
}

// The sun disk: radiance normalised so the disk integrates to the sun
// illuminance, with limb darkening I(mu)/I(1) = mu^alpha (Hestroffer &
// Magnan 1998, alpha ~ 0.40 / 0.50 / 0.65 for red / green / blue), times the
// camera's transmittance. 0 outside the disk.
fn atmoSunDisk(dir: vec3f) -> vec3f {
  let L = atmoSunDiskRadiance(dir);
  if (all(L == vec3f(0.0))) { return L; }
  return L * atmoCameraTransmittance(dir);
}

// The disk's radiance above the atmosphere (no transmittance applied).
fn atmoSunDiskRadiance(dir: vec3f) -> vec3f {
  let chord = length(dir - ATMO.sunDir);
  let chordR = 2.0 * sin(ATMO.sunAngularRadius * 0.5);
  if (chord >= chordR) { return vec3f(0.0); }
  let x = chord / chordR;
  let muDisk = sqrt(max(0.0, 1.0 - x * x));
  let alpha = vec3f(0.397, 0.503, 0.652);
  let limb = mix(vec3f(1.0), pow(vec3f(max(muDisk, 1e-4)), alpha), ATMO.limbDarkening);
  let norm = mix(vec3f(1.0), (alpha + 2.0) * 0.5, ATMO.limbDarkening);
  let solidAngle = 2.0 * ATMO_PI * (1.0 - cos(ATMO.sunAngularRadius));
  // Soft 1-texel-ish edge in angle space avoids a jagged disk.
  let edge = smoothstep(1.0, 0.97, x);
  return ATMO.sunIlluminance / solidAngle * limb * norm * edge;
}

// In-scattering and transmittance between the camera and a surface at
// distance `dist` (km) along `dir`, seen at screen position `uv` (0..1, top
// left). Uses the froxel volume inside the atmosphere and a per-pixel ray
// march above ATMO.rayMarchAltitude. Past the end of the volume the
// remainder is ray marched and composited.
fn atmoAerialPerspective(uv: vec2f, dir: vec3f, dist: f32) -> AtmoScatter {
  var res: AtmoScatter;
  res.inscatter = vec3f(0.0);
  res.transmittance = vec3f(1.0);
  let seg = atmoCameraSegment(dir);
  if (seg.y < 0.5) { return res; }
  let tEnd = min(dist, seg.w);
  if (tEnd <= seg.x) { return res; }

  if (ATMO.cameraAltitude >= ATMO.rayMarchAltitude) {
    let steps = atmoStepCount(tEnd - seg.x, 12, 40);
    res = atmoIntegrate(ATMO.cameraPos, dir, ATMO.sunDir, seg.x, tEnd, steps, false);
    res.inscatter *= ATMO.sunIlluminance;
    return res;
  }

  let d = tEnd - seg.x;
  let slices = ATMO.apSlices;
  let dc = min(d, ATMO.apMaxDistance);
  let s = sqrt(dc / ATMO.apMaxDistance) * slices;
  var w = 1.0;
  var z = 0.5 / slices;
  if (s < 1.0) {
    // Between the camera and the first slice: fade linearly in distance.
    w = s * s;
  } else {
    // Slice k ends at D_k = maxDistance ((k + 1) / slices)^2. Interpolate
    // between neighbouring slices linearly in distance, not in the squared
    // slice coordinate: in-scattering grows ~linearly with distance, and
    // interpolating in sqrt(distance) measured 1.5-4x the mean error
    // against the reference (sunset: 0.74 % vs 0.28 %).
    let k = min(floor(s) - 1.0, slices - 2.0);
    let d0 = ATMO.apMaxDistance * pow((k + 1.0) / slices, 2.0);
    let d1 = ATMO.apMaxDistance * pow((k + 2.0) / slices, 2.0);
    z = (k + 0.5 + clamp((dc - d0) / (d1 - d0), 0.0, 1.0)) / slices;
  }
  let coord = vec3f(uv, z);
  res.inscatter = textureSampleLevel(atmoApInscatter, atmoSampler, coord, 0.0).rgb * w;
  res.transmittance = mix(vec3f(1.0), textureSampleLevel(atmoApTransmittance, atmoSampler, coord, 0.0).rgb, w);
  if (d > ATMO.apMaxDistance) {
    let rest = atmoIntegrate(ATMO.cameraPos, dir, ATMO.sunDir, seg.x + ATMO.apMaxDistance, tEnd, 8, false);
    res.inscatter += res.transmittance * rest.inscatter * ATMO.sunIlluminance;
    res.transmittance *= rest.transmittance;
  }
  return res;
}

fn atmoApply(radiance: vec3f, uv: vec2f, dir: vec3f, dist: f32) -> vec3f {
  let ap = atmoAerialPerspective(uv, dir, dist);
  return radiance * ap.transmittance + ap.inscatter;
}

// Transmittance from a point to the sun, including the planet's shadow
// (multiply by atmoSunIlluminance() for irradiance).
fn atmoSunTransmittanceAt(pos: vec3f) -> vec3f {
  let r = max(length(pos), ATMO.bottomRadius);
  return atmoTransmittanceToSun(r, dot(pos, ATMO.sunDir) / length(pos));
}

// Sky irradiance on a surface with the given normal: the horizontal
// irradiance LUT scaled by (1 + n.up) / 2, the view factor of a uniform sky.
fn atmoSkyIrradiance(pos: vec3f, normal: vec3f) -> vec3f {
  let r = max(length(pos), ATMO.bottomRadius);
  let up = pos / length(pos);
  let uv = atmoMsUv(r, dot(up, ATMO.sunDir), ATMO_IRRADIANCE_SIZE);
  let E = textureSampleLevel(atmoIrradianceLut, atmoSampler, uv, 0.0).rgb;
  return E * ATMO.sunIlluminance * (0.5 + 0.5 * dot(normal, up));
}

fn atmoSunIlluminance() -> vec3f {
  return ATMO.sunIlluminance;
}

// Sky radiance seen from an arbitrary point (e.g. for water reflections far
// from the camera), ray marched with a few samples against the LUTs.
fn atmoSkyRadianceFrom(pos: vec3f, dir: vec3f, steps: i32) -> vec3f {
  let r = length(pos);
  let mu = dot(pos, dir) / r;
  let top = atmoRaySphere(r, mu, ATMO.topRadius);
  if (top.y <= 0.0 || top.y < top.x) { return vec3f(0.0); }
  let t0 = max(top.x, 0.0);
  var t1 = top.y;
  let g = atmoRaySphere(r, mu, ATMO.bottomRadius);
  if (g.x <= g.y && g.x > 0.0) { t1 = g.x; }
  return atmoIntegrate(pos, dir, ATMO.sunDir, t0, t1, steps, r < ATMO.topRadius).inscatter * ATMO.sunIlluminance;
}
