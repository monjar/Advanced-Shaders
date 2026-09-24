// Heat haze / space warp around the object, driven by the field. Each pixel
// whose ray passes through the halo sphere around the object integrates the
// field along that chord (outside the object only): the flow velocity,
// weighted by energy and charge and by closeness to the surface, is
// projected to the screen and offsets the scene lookup. The haze therefore
// streams in the same direction as the particles and the interior energy.
// `lens` adds a gravitational-lens-like pull toward the object's centre
// (the Void stone preset). Only what is behind the halo is warped.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> M: Material;
@group(0) @binding(2) var scene: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return fullscreenPosition(vi);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / F.resolution;
  let base = textureSampleLevel(scene, linearSampler, uv, 0.0);
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let h = F.invViewProj * vec4f(ndc, 1.0, 1.0);
  let dirW = normalize(h.xyz / h.w - F.camPos);
  let ro = toObject(F.camPos);
  let rd = toObjectDir(dirW);
  let Rh = M.haze.z;
  let b = dot(ro, rd);
  let c = dot(ro, ro) - Rh * Rh;
  let disc = b * b - c;
  if (disc <= 0.0) {
    if (F.debug == 7.0) { return vec4f(0.5, 0.5, 0.5, -1.0); }
    return base;
  }
  let sq = sqrt(disc);
  let t0 = max(-b - sq, 0.0);
  // Background distance in object units; nothing in front of the halo is warped.
  let sceneT = base.a / F.objScale;
  let t1 = min(-b + sq, sceneT);
  var offset = vec2f(0.0);
  var amount = 0.0;
  if (t1 > t0) {
    let K = 6;
    let dt = (t1 - t0) / f32(K);
    let j = ign(pos.xy, F.frameIndex);
    var acc = vec3f(0.0);
    for (var i = 0; i < K; i++) {
      let q = ro + rd * (t0 + (f32(i) + j) * dt);
      let sh = sampleShape(q);
      if (sh.x > 0.0) {
        let f = sampleField(q);
        let w = exp(-sh.x / 0.28) * (0.25 + 1.6 * f.E + 0.6 * f.C);
        acc += f.v * w * dt;
        amount += w * dt;
      }
    }
    // Project the accumulated object-space flow onto the screen.
    let mid = ro + rd * 0.5 * (t0 + t1);
    let pw = toWorld(mid);
    let a0 = projectUv(pw);
    let a1 = projectUv(pw + (F.model * vec4f(acc, 0.0)).xyz * 0.05);
    offset = (a1.xy - a0.xy) * M.haze.x;
    // Space warp: deflection ∝ 1 / impact parameter, pulling the background
    // around the object like a lens.
    let impact = sqrt(max(dot(ro, ro) - b * b, 1e-4));
    let centre = projectUv(F.objCenter).xy;
    let toC = centre - uv;
    offset += normalize(toC + vec2f(1e-6)) * M.haze.y * 0.025 / max(impact * impact, 0.35)
      * (1.0 - smoothstep(Rh * 0.6, Rh, impact));
  }
  // Debug: grey = no offset, red/green = screen offset (×40), blue = haze amount.
  if (F.debug == 7.0) { return vec4f(offset * 40.0 + 0.5, 0.5 + amount * 0.25, -1.0); }
  // Slight chromatic split of the warped lookup.
  let r = textureSampleLevel(scene, linearSampler, uv + offset * 1.06, 0.0).r;
  let g = textureSampleLevel(scene, linearSampler, uv + offset, 0.0).g;
  let bl = textureSampleLevel(scene, linearSampler, uv + offset * 0.94, 0.0).b;
  return vec4f(r, g, bl, base.a);
}
