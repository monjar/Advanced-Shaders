// Physically based atmosphere shared by the atmosphere and planet studies.
//
// After Hillaire, "A Scalable and Production Ready Sky and Atmosphere
// Rendering Technique" (EGSR 2020), with the transmittance parameterisation of
// Bruneton & Neyret, "Precomputed Atmospheric Scattering" (EGSR 2008).
//
// Units: kilometres, positions relative to the planet centre. Every file of
// the module expects a uniform `ATMO: AtmosphereUniforms` (declared by the
// including shader, see atmosphereWgsl() in index.ts).

const ATMO_PI: f32 = 3.14159265359;
const ATMO_TRANSMITTANCE_SIZE: vec2f = vec2f(256.0, 64.0);
const ATMO_MS_SIZE: f32 = 32.0;
const ATMO_IRRADIANCE_SIZE: vec2f = vec2f(64.0, 16.0);

// Keep in sync with Atmosphere.writeUniforms() in index.ts.
struct AtmosphereUniforms {
  bottomRadius: f32,
  topRadius: f32,
  rayleighExpScale: f32,      // -1 / scale height (1/km)
  mieExpScale: f32,
  rayleighScattering: vec3f,  // 1/km at the ground
  ozoneCenter: f32,           // km
  mieScattering: vec3f,
  ozoneHalfWidth: f32,        // km
  mieExtinction: vec3f,
  multiScatteringFactor: f32,
  ozoneAbsorption: vec3f,     // 1/km at the peak of the tent profile
  sunAngularRadius: f32,      // radians
  mieG: vec3f,                // Cornette-Shanks asymmetry per channel
  apMaxDistance: f32,         // km covered by the aerial-perspective volume
  groundAlbedo: vec3f,
  rayMarchAltitude: f32,      // km: above it, sky and aerial perspective are ray marched per pixel
  sunIlluminance: vec3f,
  apSlices: f32,
  cameraPos: vec3f,           // km, planet centred
  cameraAltitude: f32,        // km above bottomRadius, computed in double precision
  sunDir: vec3f,
  limbDarkening: f32,         // 0: flat disk, 1: Hestroffer & Magnan power law
  invViewProj: mat4x4f,       // camera-relative (no translation), for froxel ray directions
  skyViewSize: vec2f,
  pad0: vec2f,
};

struct AtmoMedium {
  scattering: vec3f,
  extinction: vec3f,
  rayleigh: vec3f,
  mie: vec3f,
};

// Rayleigh and Mie decay exponentially; ozone is a tent centred on
// ozoneCenter (Bruneton's two linear layers, 10-40 km for Earth).
fn atmoMedium(altitude: f32) -> AtmoMedium {
  let h = max(altitude, 0.0);
  let dR = exp(ATMO.rayleighExpScale * h);
  let dM = exp(ATMO.mieExpScale * h);
  let dO = max(0.0, 1.0 - abs(h - ATMO.ozoneCenter) / ATMO.ozoneHalfWidth);
  var m: AtmoMedium;
  m.rayleigh = ATMO.rayleighScattering * dR;
  m.mie = ATMO.mieScattering * dM;
  m.scattering = m.rayleigh + m.mie;
  m.extinction = m.rayleigh + ATMO.mieExtinction * dM + ATMO.ozoneAbsorption * dO;
  return m;
}

fn atmoRayleighPhase(mu: f32) -> f32 {
  return 3.0 / (16.0 * ATMO_PI) * (1.0 + mu * mu);
}

// Cornette-Shanks (1992): Henyey-Greenstein with a Rayleigh-like (1 + mu^2)
// term, as used by Hillaire. Per channel, so dusty atmospheres can scatter
// blue further forward than red (the blue Martian sunset).
fn atmoMiePhase(mu: f32) -> vec3f {
  let g = ATMO.mieG;
  let g2 = g * g;
  let denom = (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * mu, vec3f(1e-5)), vec3f(1.5));
  return 3.0 / (8.0 * ATMO_PI) * (1.0 - g2) * (1.0 + mu * mu) / denom;
}

// Maps [0, 1] to texel centres so the first and last texels sit exactly on
// the domain boundary (Bruneton's getTextureCoordFromUnitRange).
fn atmoUnitToUv(x: f32, size: f32) -> f32 {
  return 0.5 / size + x * (1.0 - 1.0 / size);
}
fn atmoUvToUnit(u: f32, size: f32) -> f32 {
  return (u - 0.5 / size) / (1.0 - 1.0 / size);
}

// Intersections of the ray (radius r, zenith cosine mu) with a centred sphere,
// where c = r^2 - radius^2 is passed in so callers that know the altitude
// can form it as h (2R + h) without float cancellation. Stable quadratic.
// Returns (t0, t1); t1 < t0 on a miss.
fn atmoRaySphereC(r: f32, mu: f32, c: f32) -> vec2f {
  let b = r * mu;
  let disc = b * b - c;
  if (disc < 0.0) { return vec2f(1.0, -1.0); }
  let s = sqrt(disc);
  let q = -b - select(-s, s, b >= 0.0);
  if (abs(q) < 1e-12) { return vec2f(0.0, 0.0); }
  let t0 = q;
  let t1 = c / q;
  return vec2f(min(t0, t1), max(t0, t1));
}

fn atmoRaySphere(r: f32, mu: f32, radius: f32) -> vec2f {
  return atmoRaySphereC(r, mu, (r - radius) * (r + radius));
}

// Transmittance LUT (Bruneton 2008, section 4): x_r is the distance to the
// horizon rho / H, x_mu interpolates between the shortest and longest
// distance to the top boundary. Only directions above the horizon are
// stored; the planet's shadow is applied separately.
fn atmoTransmittanceUv(r: f32, mu: f32) -> vec2f {
  let R = ATMO.bottomRadius;
  let T = ATMO.topRadius;
  let H = sqrt(max(0.0, T * T - R * R));
  let rho = sqrt(max(0.0, (r - R) * (r + R)));
  let disc = r * r * (mu * mu - 1.0) + T * T;
  let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
  let dMin = T - r;
  let dMax = rho + H;
  let xMu = clamp((d - dMin) / max(dMax - dMin, 1e-6), 0.0, 1.0);
  let xR = clamp(rho / H, 0.0, 1.0);
  return vec2f(atmoUnitToUv(xMu, ATMO_TRANSMITTANCE_SIZE.x), atmoUnitToUv(xR, ATMO_TRANSMITTANCE_SIZE.y));
}

// Inverse of atmoTransmittanceUv: returns (r, mu).
fn atmoTransmittanceParams(uv: vec2f) -> vec2f {
  let R = ATMO.bottomRadius;
  let T = ATMO.topRadius;
  let xMu = atmoUvToUnit(uv.x, ATMO_TRANSMITTANCE_SIZE.x);
  let xR = atmoUvToUnit(uv.y, ATMO_TRANSMITTANCE_SIZE.y);
  let H = sqrt(max(0.0, T * T - R * R));
  let rho = H * xR;
  let r = sqrt(rho * rho + R * R);
  let dMin = T - r;
  let dMax = rho + H;
  let d = dMin + xMu * (dMax - dMin);
  var mu = 1.0;
  if (d > 0.0) { mu = (H * H - rho * rho - d * d) / (2.0 * r * d); }
  return vec2f(r, clamp(mu, -1.0, 1.0));
}

// Multiple-scattering and irradiance LUTs: sun zenith cosine by altitude.
fn atmoMsUv(r: f32, muS: f32, size: vec2f) -> vec2f {
  let x = clamp(muS * 0.5 + 0.5, 0.0, 1.0);
  let y = clamp((r - ATMO.bottomRadius) / (ATMO.topRadius - ATMO.bottomRadius), 0.0, 1.0);
  return vec2f(atmoUnitToUv(x, size.x), atmoUnitToUv(y, size.y));
}

// Local frame used by the LUTs: z up, sun in the x-z plane.
fn atmoLocalSun(muS: f32) -> vec3f {
  return vec3f(sqrt(max(0.0, 1.0 - muS * muS)), 0.0, muS);
}

// Radius at distance t along a ray from radius r with zenith cosine mu.
fn atmoRadiusAt(r: f32, mu: f32, t: f32) -> f32 {
  return sqrt(max(0.0, r * r + t * t + 2.0 * r * mu * t));
}

// ---- Camera-relative helpers -------------------------------------------

// Altitude used by the sky-view LUT (kept strictly inside the atmosphere).
fn atmoSkyViewAltitude() -> f32 {
  return clamp(ATMO.cameraAltitude, 0.0, ATMO.topRadius - ATMO.bottomRadius - 0.01);
}

// A camera ray against the atmosphere, using the double-precision camera
// altitude for the ground test so it stays exact at walking height:
// x = entry distance (0 when inside), y = 1 if the ray crosses the
// atmosphere, z = ground hit distance (-1 if none), w = distance at which the
// ray leaves the atmosphere or hits the ground.
fn atmoCameraSegment(dir: vec3f) -> vec4f {
  let R = ATMO.bottomRadius;
  let h = max(ATMO.cameraAltitude, 0.0);
  let r = R + h;
  let mu = dot(ATMO.cameraPos, dir) / length(ATMO.cameraPos);
  let top = atmoRaySphere(r, mu, ATMO.topRadius);
  if (top.y < top.x || top.y <= 0.0) { return vec4f(0.0, 0.0, -1.0, 0.0); }
  let entry = max(top.x, 0.0);
  let g = atmoRaySphereC(r, mu, h * (2.0 * R + h));
  var ground = -1.0;
  if (g.x <= g.y && g.y > 0.0) { ground = max(g.x, 0.0); }
  let exit = select(top.y, ground, ground >= 0.0);
  return vec4f(entry, 1.0, ground, exit);
}

// Hillaire's sky-view parameterisation around the camera's zenith: v in
// [0, 0.5) above the horizon and [0.5, 1] below it, each half compressed
// towards the horizon (sqrt), where the sky changes fastest; u is the
// azimuth from the sun, sqrt-compressed towards the sun. The sky is
// symmetric about the sun's vertical plane, so half the azimuth suffices.
fn atmoSkyViewUv(r: f32, h: f32, cosZ: f32, cosAz: f32) -> vec2f {
  let R = ATMO.bottomRadius;
  let vHorizon = sqrt(max(0.0, h * (2.0 * R + h)));
  let beta = acos(clamp(vHorizon / r, -1.0, 1.0));
  let zenithHorizon = ATMO_PI - beta;
  let zenith = acos(clamp(cosZ, -1.0, 1.0));
  var v: f32;
  if (zenith < zenithHorizon) {
    let c = zenith / zenithHorizon;
    v = (1.0 - sqrt(max(0.0, 1.0 - c))) * 0.5;
  } else {
    let c = (zenith - zenithHorizon) / max(beta, 1e-6);
    v = sqrt(clamp(c, 0.0, 1.0)) * 0.5 + 0.5;
  }
  let u = sqrt(clamp(-cosAz * 0.5 + 0.5, 0.0, 1.0));
  return vec2f(atmoUnitToUv(u, ATMO.skyViewSize.x), atmoUnitToUv(v, ATMO.skyViewSize.y));
}

// Inverse: (cos zenith, cos azimuth from the sun).
fn atmoSkyViewParams(uv: vec2f, r: f32, h: f32) -> vec2f {
  let R = ATMO.bottomRadius;
  let u = clamp(atmoUvToUnit(uv.x, ATMO.skyViewSize.x), 0.0, 1.0);
  let v = clamp(atmoUvToUnit(uv.y, ATMO.skyViewSize.y), 0.0, 1.0);
  let vHorizon = sqrt(max(0.0, h * (2.0 * R + h)));
  let beta = acos(clamp(vHorizon / r, -1.0, 1.0));
  let zenithHorizon = ATMO_PI - beta;
  var cosZ: f32;
  if (v < 0.5) {
    let c = 1.0 - 2.0 * v;
    cosZ = cos(zenithHorizon * (1.0 - c * c));
  } else {
    let c = v * 2.0 - 1.0;
    cosZ = cos(zenithHorizon + beta * c * c);
  }
  let cosAz = 1.0 - 2.0 * u * u;
  return vec2f(cosZ, cosAz);
}
