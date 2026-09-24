// GPU particles born from the same field. A dead particle proposes a random
// point on the surface (projected onto the SDF) and is born there with
// probability ∝ how open the crack is × the local energy, so emission
// follows the cracks as they open and close. Alive particles are advected by
// the field's curl velocity (in world space via the model matrix), feel
// buoyancy, cannot enter the object, and fade over their lifetime. They are
// drawn as additive soft sprites stretched along their screen-space motion,
// depth-tested against scene + object.

struct Particle {
  pos: vec3f,
  age: f32,
  vel: vec3f,
  life: f32,
};

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> M: Material;
@group(0) @binding(2) var<storage, read_write> parts: array<Particle>;
@group(0) @binding(3) var<storage, read> partsRead: array<Particle>;

fn objToWorldVec(v: vec3f) -> vec3f {
  return (F.model * vec4f(v, 0.0)).xyz;
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(M.particleColor.w)) { return; }
  var P = parts[i];
  let dt = F.dt;
  if (P.age >= P.life) {
    let r = hash33(vec3i(i32(i), i32(F.frameIndex), 91));
    let r2 = hash33(vec3i(i32(i), i32(F.frameIndex), 17));
    // Uniform direction, projected onto the surface along the SDF gradient.
    let z = r.x * 2.0 - 1.0;
    let phi = r.y * TAU;
    let dir = vec3f(sqrt(1.0 - z * z) * cos(phi), z, sqrt(1.0 - z * z) * sin(phi));
    var q = dir * 0.9;
    for (var k = 0; k < 4; k++) {
      let s = sampleShape(q);
      q -= s.yzw * s.x;
    }
    let s = sampleShape(q);
    let f = sampleField(q);
    let c = sampleCrack(q);
    let open = crackOpen(f.C, f.E);
    let w = M.crack.x * open;
    let onCrack = (1.0 - smoothstep(w * 0.6, w * 1.6 + 0.004, c.x)) * open;
    // Spawn attempts scale with the pool, so normalise: the pool size is a
    // capacity, the spawn rate sets the density.
    let pool = 32768.0 / max(M.particleColor.w, 1.0);
    let prob = onCrack * (0.25 + 1.5 * f.E) * M.particles.x * dt * 1.5 * pool * select(0.0, 1.0, abs(s.x) < 0.02);
    if (r.z < prob) {
      let nrm = s.yzw;
      P.pos = toWorld(q + nrm * 0.012);
      P.vel = objToWorldVec(nrm * M.particles2.y * (0.4 + r2.x) + f.v * M.particles2.x * 0.5);
      P.life = M.particles.z * (0.45 + r2.y);
      P.age = 0.0;
    } else {
      P.age = P.life + 1.0;
    }
    parts[i] = P;
    return;
  }
  let q = toObject(P.pos);
  let f = sampleField(q);
  let s = sampleShape(q);
  let flowVel = objToWorldVec(f.v * M.particles2.x) + vec3f(0.0, M.particles.w, 0.0);
  // Relax toward the flow; the eject impulse survives for the first ~0.4 s.
  P.vel = mix(P.vel, flowVel, 1.0 - exp(-dt * 2.5));
  var np = P.pos + P.vel * dt;
  // Keep out of the object: project back onto its surface and drop the inward velocity.
  let nq = toObject(np);
  let ns = sampleShape(nq);
  if (ns.x < 0.004) {
    np = toWorld(nq - ns.yzw * (ns.x - 0.004));
    let nw = toWorldDir(ns.yzw);
    P.vel -= nw * min(dot(P.vel, nw), 0.0);
  }
  P.pos = np;
  P.age += dt;
  parts[i] = P;
}

struct PVOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) color: vec3f,
  @location(2) stretch: f32,
};

@vertex
fn vsParticle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PVOut {
  var out: PVOut;
  let P = partsRead[ii];
  // Hidden in the debug views that inspect the object alone.
  let dbg = u32(F.debug + 0.5);
  if (P.age >= P.life || (dbg >= 2u && dbg != 5u)) {
    out.position = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  let u = P.age / max(P.life, 1e-3);
  var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let corner = corners[vi];
  let clip = F.viewProj * vec4f(P.pos, 1.0);
  if (clip.w < 0.05) {
    out.position = vec4f(0.0, 0.0, 2.0, 1.0);
    return out;
  }
  // Size in pixels from the world size at this depth.
  let focal = F.extra.x;
  let envelope = smoothstep(0.0, 0.08, u) * (1.0 - smoothstep(0.55, 1.0, u));
  var sizePx = M.particles.y * F.objScale * focal / clip.w * (0.6 + 0.6 * envelope);
  // Motion over ~1/30 s on screen gives the streak.
  let clip2 = F.viewProj * vec4f(P.pos + P.vel * (0.033 * M.particles2.w), 1.0);
  let s0 = clip.xy / clip.w * 0.5 * F.resolution;
  let s1 = clip2.xy / max(clip2.w, 0.05) * 0.5 * F.resolution;
  let mv = s1 - s0;
  let len = length(mv);
  let axis = select(vec2f(1.0, 0.0), mv / max(len, 1e-4), len > 1e-3);
  let perp = vec2f(-axis.y, axis.x);
  // Sub-pixel sprites become 1 px wide and dimmer (same energy), which
  // avoids aliasing sparkle.
  var energy = 1.0;
  if (sizePx < 1.0) {
    energy = sizePx * sizePx;
    sizePx = 1.0;
  }
  let halfLen = sizePx + len * 0.5;
  let offPx = axis * corner.x * halfLen + perp * corner.y * sizePx + mv * 0.5;
  out.position = clip + vec4f(offPx / (0.5 * F.resolution) * clip.w, 0.0, 0.0);
  out.uv = corner;
  out.stretch = sizePx / halfLen;
  let flicker = 0.75 + 0.25 * sin(P.age * (9.0 + hashU(ii) * 14.0) + hashU(ii + 7u) * 40.0);
  let col = mix(M.particleHot.rgb, M.particleColor.rgb, smoothstep(0.0, 0.5, u));
  // Longer streaks spread the same light over more pixels.
  out.color = col * M.particles2.z * envelope * flicker * energy * out.stretch;
  if (F.debug == 5.0) { out.color = vec3f(0.02); }
  return out;
}

@fragment
fn fsParticle(in: PVOut) -> @location(0) vec4f {
  // Round ends: along the streak the falloff is squeezed to the head size.
  let along = max(abs(in.uv.x) - (1.0 - in.stretch), 0.0) / max(in.stretch, 1e-3);
  let r2 = along * along + in.uv.y * in.uv.y;
  let g = exp(-r2 * 4.0) - exp(-4.0);
  if (F.debug == 5.0) { return vec4f(in.color * step(r2, 1.0), 0.0); }
  return vec4f(in.color * max(g, 0.0), 0.0);
}
