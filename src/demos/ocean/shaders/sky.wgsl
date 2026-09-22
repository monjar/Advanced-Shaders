// Analytic single-scattering sky (Rayleigh + Mie) with a flat-earth airmass
// approximation. Requires a uniform `F: Frame`. Mirrored on the CPU in sky.ts
// to derive the sun colour and ambient light.

const BETA_R: vec3f = vec3f(5.8e-3, 13.5e-3, 33.1e-3); // Rayleigh scattering per km

fn airmass(cosZenith: f32) -> f32 {
  let c = clamp(cosZenith, 0.0, 1.0);
  let zenithDeg = acos(c) * 57.29578;
  return 1.0 / (c + 0.15 * pow(max(93.885 - zenithDeg, 1e-3), -1.253));
}

fn skyExtinction() -> vec3f {
  return BETA_R * F.sky.y * 8.4 + vec3f(0.004 * F.sky.x) * 1.25;
}

fn skyRadiance(dirIn: vec3f, withSun: bool) -> vec3f {
  let d = normalize(vec3f(dirIn.x, max(dirIn.y, 0.0), dirIn.z));
  let bR = BETA_R * F.sky.y;
  let bM = vec3f(0.004 * F.sky.x);
  let ext = skyExtinction();
  let viewT = exp(-ext * airmass(d.y));
  let sunT = exp(-ext * airmass(F.sunDir.y));

  let mu = dot(d, F.sunDir);
  let g = F.sky.z;
  let phaseR = 0.0596831 * (1.0 + mu * mu);
  let phaseM = 0.0795775 * (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * mu, 1e-4), 1.5);

  // Light reaching high-altitude scatterers is less reddened than at the horizon.
  let scatterT = pow(sunT, vec3f(0.2 + 0.45 * (1.0 - d.y)));
  var L = F.sunIntensity * scatterT * (bR * phaseR + bM * phaseM) / (bR + bM) * (1.0 - viewT) * F.sky.w;
  L += vec3f(0.002, 0.004, 0.008);

  if (withSun) {
    let disk = smoothstep(0.99994, 0.99997, dot(normalize(dirIn), F.sunDir));
    L += F.sunColor * disk * 6000.0;
  }
  return L;
}

// Aerial perspective: blend towards the horizon sky with distance.
fn applyFog(color: vec3f, worldPos: vec3f, density: f32) -> vec3f {
  let toP = worldPos - F.camPos;
  let dist = length(toP);
  let dir = toP / max(dist, 1e-3);
  let fogColor = skyRadiance(vec3f(dir.x, max(dir.y, 0.0) * 0.5 + 0.01, dir.z), false);
  let amount = 1.0 - exp(-dist * density);
  return mix(color, fogColor, amount);
}
