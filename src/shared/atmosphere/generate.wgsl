// LUT passes. One module, one entry point per LUT; each pipeline's auto
// layout keeps only the bindings its entry point uses.
//
//   transmittance     256x64    once per parameter change
//   multiScattering   32x32     once per parameter change (64 threads per texel)
//   irradiance        64x16     once per parameter change (64 threads per texel)
//   skyView           192x108   every frame (camera altitude and sun)
//   aerial            32x32x32  every frame (camera frustum)

@group(0) @binding(0) var<uniform> ATMO: AtmosphereUniforms;
@group(0) @binding(1) var atmoSampler: sampler;
@group(0) @binding(2) var atmoTransmittanceLut: texture_2d<f32>;
@group(0) @binding(3) var atmoMultiScatLut: texture_2d<f32>;
@group(0) @binding(4) var out2d: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var apInscatterOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var apTransmittanceOut: texture_storage_3d<rgba16float, write>;

// Hillaire keeps LUT sample points 10 m off the ground so rays starting
// there do not immediately intersect the planet.
const PLANET_OFFSET: f32 = 0.01;

@compute @workgroup_size(8, 8)
fn transmittance(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= vec2u(ATMO_TRANSMITTANCE_SIZE))) { return; }
  let uv = (vec2f(id.xy) + 0.5) / ATMO_TRANSMITTANCE_SIZE;
  let rm = atmoTransmittanceParams(uv);
  let r = rm.x;
  let mu = rm.y;
  let tTop = max(atmoRaySphere(r, mu, ATMO.topRadius).y, 0.0);
  let steps = 40;
  let dt = tTop / f32(steps);
  var depth = vec3f(0.0);
  for (var i = 0; i < steps; i++) {
    let ri = atmoRadiusAt(r, mu, (f32(i) + 0.5) * dt);
    depth += atmoMedium(ri - ATMO.bottomRadius).extinction * dt;
  }
  textureStore(out2d, id.xy, vec4f(exp(-depth), 1.0));
}

var<workgroup> sharedA: array<vec3f, 64>;
var<workgroup> sharedB: array<vec3f, 64>;

fn reduceShared(li: u32) {
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li < s) {
      sharedA[li] += sharedA[li + s];
      sharedB[li] += sharedB[li + s];
    }
    workgroupBarrier();
  }
}

// Hillaire 2020, section 5.5. For each (sun zenith, altitude) the 64
// threads of the workgroup integrate one direction each of a uniform
// sphere of directions:
//   L2  = mean over directions of single scattering with an isotropic phase,
//         plus sunlight bounced off the ground (Lambertian, groundAlbedo),
//   fms = mean over directions of the transfer ∫ T sigma_s dt,
// and store Psi_ms = L2 / (1 - fms): the geometric series of all higher
// orders, assuming each order is isotropic and spatially uniform around x.
@compute @workgroup_size(64)
fn multiScattering(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let uv = (vec2f(wg.xy) + 0.5) / ATMO_MS_SIZE;
  let muS = atmoUvToUnit(uv.x, ATMO_MS_SIZE) * 2.0 - 1.0;
  let hFrac = clamp(atmoUvToUnit(uv.y, ATMO_MS_SIZE), 0.0, 1.0);
  let R = ATMO.bottomRadius;
  let r = R + PLANET_OFFSET + hFrac * (ATMO.topRadius - R - 2.0 * PLANET_OFFSET);
  let sunDir = atmoLocalSun(muS);

  let i = f32(li / 8u) + 0.5;
  let j = f32(li % 8u) + 0.5;
  let theta = 2.0 * ATMO_PI * i / 8.0;
  let cosPhi = 1.0 - 2.0 * j / 8.0;
  let sinPhi = sqrt(max(0.0, 1.0 - cosPhi * cosPhi));
  let dir = vec3f(cos(theta) * sinPhi, sin(theta) * sinPhi, cosPhi);
  let origin = vec3f(0.0, 0.0, r);

  let top = atmoRaySphere(r, dir.z, ATMO.topRadius);
  let ground = atmoRaySphere(r, dir.z, R);
  let hitsGround = ground.x <= ground.y && ground.x > 0.0;
  let tMax = select(top.y, ground.x, hitsGround);

  let steps = 20;
  let dt = tMax / f32(steps);
  let isotropic = 1.0 / (4.0 * ATMO_PI);
  var L = vec3f(0.0);
  var fms = vec3f(0.0);
  var T = vec3f(1.0);
  for (var s = 0; s < steps; s++) {
    let p = origin + dir * ((f32(s) + 0.5) * dt);
    let pr = length(p);
    let pMuS = dot(p, sunDir) / pr;
    let med = atmoMedium(pr - R);
    let stepT = exp(-med.extinction * dt);
    let ext = max(med.extinction, vec3f(1e-9));
    let S = atmoTransmittanceToSun(pr, pMuS) * med.scattering * isotropic;
    L += T * (S - S * stepT) / ext;
    fms += T * (med.scattering - med.scattering * stepT) / ext;
    T *= stepT;
  }
  if (hitsGround) {
    let p = origin + dir * tMax;
    let pr = length(p);
    let gMuS = dot(p, sunDir) / pr;
    L += atmoTransmittanceToSun(pr, gMuS) * T * max(gMuS, 0.0) * ATMO.groundAlbedo / ATMO_PI;
  }

  sharedA[li] = L / 64.0;
  sharedB[li] = fms / 64.0;
  workgroupBarrier();
  reduceShared(li);
  if (li == 0u) {
    let psi = sharedA[0] / max(vec3f(1.0) - sharedB[0], vec3f(1e-3));
    textureStore(out2d, wg.xy, vec4f(psi * ATMO.multiScatteringFactor, 1.0));
  }
}

// Sky irradiance on a horizontal surface (Bruneton's irradiance texture,
// here integrated from the LUT-based sky): cosine-weighted hemisphere of 64
// stratified directions, E = pi * mean(L). Used for skylight on the ground
// and on clouds; per unit sun illuminance.
@compute @workgroup_size(64)
fn irradiance(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let uv = (vec2f(wg.xy) + 0.5) / ATMO_IRRADIANCE_SIZE;
  let muS = atmoUvToUnit(uv.x, ATMO_IRRADIANCE_SIZE.x) * 2.0 - 1.0;
  let hFrac = clamp(atmoUvToUnit(uv.y, ATMO_IRRADIANCE_SIZE.y), 0.0, 1.0);
  let R = ATMO.bottomRadius;
  let r = R + PLANET_OFFSET + hFrac * (ATMO.topRadius - R - 2.0 * PLANET_OFFSET);
  let sunDir = atmoLocalSun(muS);

  let u1 = (f32(li / 8u) + 0.5) / 8.0;
  let u2 = (f32(li % 8u) + 0.5) / 8.0;
  let sinT = sqrt(u1);
  let phi = 2.0 * ATMO_PI * u2;
  let dir = vec3f(sinT * cos(phi), sinT * sin(phi), sqrt(1.0 - u1));
  let top = atmoRaySphere(r, dir.z, ATMO.topRadius);
  let res = atmoIntegrate(vec3f(0.0, 0.0, r), dir, sunDir, 0.0, top.y, 24, true);

  sharedA[li] = res.inscatter * (ATMO_PI / 64.0);
  sharedB[li] = vec3f(0.0);
  workgroupBarrier();
  reduceShared(li);
  if (li == 0u) {
    textureStore(out2d, wg.xy, vec4f(sharedA[0], 1.0));
  }
}

@compute @workgroup_size(8, 8)
fn skyView(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2u(ATMO.skyViewSize);
  if (any(id.xy >= size)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / ATMO.skyViewSize;
  let h = atmoSkyViewAltitude();
  let r = ATMO.bottomRadius + h;
  let params = atmoSkyViewParams(uv, r, h);
  let cosZ = params.x;
  let cosAz = params.y;
  let sinZ = sqrt(max(0.0, 1.0 - cosZ * cosZ));
  let dir = vec3f(sinZ * cosAz, sinZ * sqrt(max(0.0, 1.0 - cosAz * cosAz)), cosZ);
  let muS = dot(normalize(ATMO.cameraPos), ATMO.sunDir);

  let R = ATMO.bottomRadius;
  let g = atmoRaySphereC(r, cosZ, h * (2.0 * R + h));
  let hitsGround = atmoUvToUnit(uv.y, ATMO.skyViewSize.y) >= 0.5 && g.x <= g.y && g.y > 0.0;
  let tMax = select(atmoRaySphere(r, cosZ, ATMO.topRadius).y, max(g.x, 0.0), hitsGround);
  let steps = atmoStepCount(tMax, 24, 40);
  let res = atmoIntegrate(vec3f(0.0, 0.0, r), dir, atmoLocalSun(muS), 0.0, tMax, steps, true);
  textureStore(out2d, id.xy, vec4f(res.inscatter * ATMO.sunIlluminance, 1.0));
}

// Aerial perspective froxels over the camera frustum: one thread per
// column integrates slice by slice (2 samples per slice). Slice z holds the
// in-scattering and transmittance from the atmosphere entry point (the
// camera, when inside) to maxDistance * ((z + 1) / slices)^2: the squared
// distribution gives nearby slices more resolution. Distances are measured
// from the entry point so the same volume also works from orbit.
@compute @workgroup_size(8, 8)
fn aerial(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(apInscatterOut);
  if (any(id.xy >= size.xy)) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size.xy);
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let q = ATMO.invViewProj * vec4f(ndc, 1.0, 1.0);
  let dir = normalize(q.xyz / q.w);
  let seg = atmoCameraSegment(dir);
  // Froxel rays are not stopped at the ground: a froxel column spans a range
  // of ground distances, and clamping at its central ray's hit made the
  // pixels just beyond it read too little in-scattering (bands one froxel
  // row high, 6-7 % mean error from 30 km up). Past the ground the medium
  // continues at its sea-level density, a smooth extension that is only
  // ever read through interpolation.
  let rc = ATMO.bottomRadius + max(ATMO.cameraAltitude, 0.0);
  let topExit = atmoRaySphere(rc, dot(ATMO.cameraPos, dir) / length(ATMO.cameraPos), ATMO.topRadius).y;
  let slices = size.z;
  var acc: AtmoScatter;
  acc.inscatter = vec3f(0.0);
  acc.transmittance = vec3f(1.0);
  var prev = 0.0;
  for (var z = 0u; z < slices; z++) {
    let w = f32(z + 1u) / f32(slices);
    let d = ATMO.apMaxDistance * w * w;
    let a = seg.x + prev;
    let b = min(seg.x + d, topExit);
    if (seg.y > 0.5 && b > a) {
      let s = atmoIntegrate(ATMO.cameraPos, dir, ATMO.sunDir, a, b, 2, false);
      acc.inscatter += acc.transmittance * s.inscatter;
      acc.transmittance *= s.transmittance;
    }
    prev = d;
    textureStore(apInscatterOut, vec3u(id.xy, z), vec4f(acc.inscatter * ATMO.sunIlluminance, 1.0));
    textureStore(apTransmittanceOut, vec3u(id.xy, z), vec4f(acc.transmittance, 1.0));
  }
}
