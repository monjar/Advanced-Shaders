// Sampling helpers shared by the ocean surface shader and the buoyancy solver.
// Expects these module-scope declarations in the including shader:
//   F: Frame, O: Ocean, dispTex, linSampler, clampSampler, terrainTex, rippleTex

// Terrain texture: (height, signed distance to shoreline, direction to shore xz).
fn terrainSample(xz: vec2f) -> vec4f {
  let uv = xz / (2.0 * O.terrain.x) + 0.5;
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) {
    return vec4f(-48.0, 1000.0, 0.0, 0.0);
  }
  return textureSampleLevel(terrainTex, clampSampler, uv, 0.0);
}

// Open-ocean waves lose energy as they reach shallow water.
fn depthAttenuation(depth: f32) -> f32 {
  return mix(0.08, 1.0, smoothstep(0.0, O.sim.w, depth));
}

fn cascadeLength(c: u32) -> f32 {
  return O.lengths[c];
}

// Sum of all FFT cascades at an undisplaced position. `spacing` is the sample
// footprint in metres and selects a mip so distant vertices don't alias.
fn fftDisplacement(xz: vec2f, spacing: f32) -> vec4f {
  var d = vec4f(0.0);
  for (var c = 0u; c < CASCADES; c++) {
    let L = cascadeLength(c);
    let lod = max(log2(spacing * FFT_SIZE / L), 0.0);
    d += textureSampleLevel(dispTex, linSampler, xz / L, c, lod);
  }
  return d;
}

struct ShoreWave {
  displacement: vec3f,
  slope: vec2f,
  foam: f32,
};

// Gerstner waves that travel along the shore-distance field towards the beach,
// grow as the water shoals, steepen and break into foam.
fn shoreWave(xz: vec2f, terrain: vec4f) -> ShoreWave {
  var r: ShoreWave;
  r.displacement = vec3f(0.0);
  r.slope = vec2f(0.0);
  r.foam = 0.0;

  let amp0 = O.shore.x;
  let range = O.shore.w;
  let s = terrain.y;
  if (amp0 <= 0.0 || s > range || s < -4.0) { return r; }

  let dir = normalize(terrain.zw + vec2f(1e-5, 0.0));
  let k = TAU / O.shore.y;
  let omega = TAU / O.shore.z;
  let phase = k * s + omega * F.time;

  // Sets of waves arrive in groups and vary along the coast.
  let groups = 0.55 + 0.45 * sin(dot(xz, vec2f(0.023, -0.017)) + F.time * 0.21 + 0.6 * sin(phase * 0.21));
  let envelope = smoothstep(range, range * 0.35, s) * smoothstep(-3.0, 3.0, s);
  let amplitude = amp0 * envelope * groups;
  let rel = amplitude / amp0;
  let steep = mix(0.3, 0.92, smoothstep(range * 0.6, 3.0, s)) * rel;
  let horiz = steep / k;

  let c = cos(phase);
  let sn = sin(phase);
  r.displacement = vec3f(dir.x * horiz * sn, amplitude * c, dir.y * horiz * sn);

  // Slope of the displaced surface along the travel direction.
  let jacobian = 1.0 - steep * c;
  let slopeU = amplitude * k * sn / max(jacobian, 0.15);
  r.slope = dir * slopeU;
  r.foam = smoothstep(0.55, 0.95, c) * smoothstep(0.35, 0.8, steep) + smoothstep(4.0, 0.0, s) * 0.6 * rel;
  return r;
}

fn rippleUV(xz: vec2f) -> vec2f {
  return (xz - O.ripple.xy) / O.ripple.z + 0.5;
}

fn rippleHeight(xz: vec2f) -> f32 {
  let uv = rippleUV(xz);
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return 0.0; }
  return textureSampleLevel(rippleTex, clampSampler, uv, 0.0).x * O.ripple.w;
}

fn rippleSlope(xz: vec2f) -> vec2f {
  let e = O.ripple.z / f32(textureDimensions(rippleTex).x);
  let hx = rippleHeight(xz + vec2f(e, 0.0)) - rippleHeight(xz - vec2f(e, 0.0));
  let hz = rippleHeight(xz + vec2f(0.0, e)) - rippleHeight(xz - vec2f(0.0, e));
  return vec2f(hx, hz) / (2.0 * e);
}

// Complete surface displacement at an undisplaced grid position.
fn surfaceDisplacement(xz: vec2f, spacing: f32) -> vec3f {
  let terrain = terrainSample(xz);
  let depth = max(-terrain.x, 0.0);
  var d = fftDisplacement(xz, spacing).xyz * depthAttenuation(depth) * O.sim.y;
  d += shoreWave(xz, terrain).displacement;
  // Troughs cannot dig below the seabed: soft-limit them to a fraction of the
  // local depth, otherwise big swells expose the bottom over shallow shelves.
  let limit = max(depth * 0.75, 0.05);
  if (d.y < 0.0) { d.y = -limit * tanh(-d.y / limit); }
  d.y += rippleHeight(xz);
  return d;
}
