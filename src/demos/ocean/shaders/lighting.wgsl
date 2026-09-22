// Lighting for opaque geometry that may sit under the water surface
// (seabed, submerged parts of floating objects). Requires F, O,
// causticTex and linSampler.

// Direction of sunlight after refracting through a flat water surface.
fn refractedSun() -> vec3f {
  return refract(-F.sunDir, vec3f(0.0, 1.0, 0.0), 1.0 / WATER_IOR);
}

fn causticsAt(worldPos: vec3f, depth: f32) -> vec3f {
  let r = refractedSun();
  let t = depth / max(-r.y, 0.2);
  // Trace back to where this light crossed the surface.
  let surfaceXZ = worldPos.xz - r.xz * t;
  let L = O.lengths[1];
  let uv = surfaceXZ / L;
  // Slight chromatic dispersion grows with depth.
  let spread = r.xz * depth * 0.004 / L;
  let cr = textureSampleLevel(causticTex, linSampler, uv - spread, 0.0).r;
  let cg = textureSampleLevel(causticTex, linSampler, uv, 0.0).r;
  let cb = textureSampleLevel(causticTex, linSampler, uv + spread, 0.0).r;
  let focus = exp(-abs(depth - O.terrain.z) * 0.12);
  let strength = O.terrain.y * smoothstep(0.0, 0.6, depth) * mix(0.35, 1.0, focus);
  return mix(vec3f(1.0), vec3f(cr, cg, cb), strength);
}

// Direct + ambient light for a diffuse surface. Below sea level the light is
// attenuated along the refracted path and modulated by caustics.
fn shadeDiffuse(albedo: vec3f, N: vec3f, worldPos: vec3f) -> vec3f {
  let nDotL = max(dot(N, F.sunDir), 0.0);
  let depth = max(-worldPos.y, 0.0);
  var direct = F.sunColor * nDotL / PI;
  var ambient = F.ambient * (0.6 + 0.4 * N.y);
  if (depth > 0.0) {
    // Irradiance through a horizontal surface is E cos(theta_i) times Fresnel
    // transmission; refraction spreads it over the same area, so a surface
    // facing the refracted beam gets (N . -r) / cos(theta_t) of that.
    let r = refractedSun();
    let cosI = max(F.sunDir.y, 0.0);
    let transmission = 1.0 - (0.02 + 0.98 * pow(1.0 - cosI, 5.0));
    let path = depth / max(-r.y, 0.2);
    let spread = max(dot(N, -r), 0.0) / max(-r.y, 0.2);
    direct = F.sunColor * cosI * transmission * spread / PI * exp(-O.absorption * path) * causticsAt(worldPos, depth);
    ambient *= exp(-O.absorption * depth * 1.5);
  }
  return albedo * (direct + ambient);
}
