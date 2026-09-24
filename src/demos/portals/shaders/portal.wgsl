// Portal passes. For every portal visible in a view at level L:
//   mask     cup mesh, stencil == L → L+1 where it passes the depth test (no colour, no depth)
//   (the destination view is rendered inside stencil == L+1, or `fill` if the recursion ends)
//   restore  cup mesh, stencil == L+1 → L; writes the portal surface depth so later
//            geometry at level L is occluded correctly, fogs the view by the air in
//            front of the portal, and writes the rim distortion offsets
//   halo     flat ring, stencil == L, additive energy rim and glow

@group(0) @binding(0) var<uniform> V: View;
@group(1) @binding(0) var<uniform> D: Draw;
@group(2) @binding(0) var<uniform> G: Globals;

struct PortalVSOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
};

@vertex
fn portalVs(@location(0) p: vec3f) -> PortalVSOut {
  // x, y in units of the half axes; z ∈ [−1, 0] scaled to just inside the carved hole.
  let depth = G.portals[u32(D.flags.z)].colour.w - 0.03;
  let world = D.model * vec4f(p.xy * PORTAL_AXES, p.z * depth, 1.0);
  return PortalVSOut(V.viewProj * world, world.xyz);
}

@fragment
fn maskFs() {}

// Where the eye ray through this fragment crosses the portal plane, in
// portal-local coordinates. The cup's sides and back (seen only when the near
// plane cuts the front) map back onto the opening this way.
fn openingPoint(P: PortalData, world: vec3f) -> vec3f {
  let dir = world - V.camPos;
  let t = -(dot(P.plane.xyz, V.camPos) + P.plane.w) / dot(P.plane.xyz, dir);
  let hit = V.camPos + dir * clamp(t, 0.0, 1.0);
  return (P.worldToLocal * vec4f(hit, 1.0)).xyz;
}

fn rimNoise(theta: f32, t: f32) -> f32 {
  // Integer angular frequencies: seamless around the ellipse.
  return 0.5 + 0.22 * sin(7.0 * theta + 2.3 * t) + 0.16 * sin(13.0 * theta - 3.1 * t + 1.7) + 0.12 * sin(23.0 * theta + 4.7 * t + 0.4);
}

fn toPixels(world: vec3f) -> vec2f {
  let c = V.viewProj * vec4f(world, 1.0);
  return (vec2f(c.x, -c.y) / c.w * 0.5 + 0.5) * V.resolution;
}

struct RestoreOut {
  @location(0) fog: vec4f,
  @location(1) rim: vec4f,
  @location(3) glow: vec4f,
};

@fragment
fn restoreFs(in: PortalVSOut) -> RestoreOut {
  let P = G.portals[u32(D.flags.z)];
  let l = openingPoint(P, in.world);
  let e = l.xy / PORTAL_AXES;
  let rho = length(e);
  let theta = atan2(e.y, e.x);
  let t = G.params.x;
  let width = G.params2.x;

  // Refraction only inside a thin band at the edge: the sample point is
  // pulled towards the centre (radially) and swirled (tangentially). At the
  // band's inner edge the displacement is zero, so the interior is untouched.
  let band = smoothstep(1.0 - width, 1.0, rho);
  let n = rimNoise(theta, t);
  let pull = G.params.w * 1.4 * width * band * (0.5 + n);
  let swirl = G.params.w * 0.5 * width * band * sin(5.0 * theta - 1.7 * t) * n;
  let rho2 = max(rho - pull, 0.0);
  let theta2 = theta + swirl;
  let src = vec3f(vec2f(cos(theta2), sin(theta2)) * rho2 * PORTAL_AXES, 0.0);
  let localToWorld = transpose(mat3x3f(P.worldToLocal[0].xyz, P.worldToLocal[1].xyz, P.worldToLocal[2].xyz));
  let centre = P.centre.xyz;
  let w0 = centre + localToWorld * l;
  let w1 = centre + localToWorld * src;
  let offset = toPixels(w1) - toPixels(w0);

  // Air between this view's camera (or its entry plane) and the portal.
  let fog = fogParams(u32(V.location));
  let dir = w0 - V.camPos;
  let seg = max(length(dir) - entryDistance(V, normalize(dir)), 0.0);
  let tr = exp(-fog.w * G.params.z * seg);

  var out: RestoreOut;
  out.fog = vec4f(mix(fog.rgb, V.fadeColour, V.fade), 1.0 - tr);
  out.rim = vec4f(encodeRimOffset(offset), 0.0, select(0.0, 1.0, rho > 1.0 - width));
  // Glow from deeper levels is dimmed by the same air.
  out.glow = vec4f(0.0, 0.0, 0.0, 1.0 - tr);
  return out;
}

// The energy rim: a thin bright edge with flowing noise, a softer halo on
// the wall around the opening, and a faint inner glow.
struct HaloVSOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) e: vec2f,
};

@vertex
fn haloVs(@location(0) p: vec3f) -> HaloVSOut {
  // 3 mm in front of the opening.
  let world = D.model * vec4f(p.xy * PORTAL_AXES, 0.003, 1.0);
  return HaloVSOut(V.viewProj * world, world.xyz, p.xy);
}

@fragment
fn haloFs(in: HaloVSOut) -> @location(3) vec4f {
  let P = G.portals[u32(D.flags.z)];
  let rho = length(in.e);
  let theta = atan2(in.e.y, in.e.x);
  let t = G.params.x;
  let n = rimNoise(theta, t * 1.3);
  let edge = exp(-pow((rho - 1.0) / (0.012 + 0.01 * n), 2.0));
  let outer = select(0.0, exp(-(rho - 1.0) / 0.07) * 0.35, rho > 1.0);
  let inner = select(0.0, exp(-(1.0 - rho) / 0.035) * 0.5 * n, rho <= 1.0);
  let flicker = 0.85 + 0.15 * sin(t * 11.0 + theta * 3.0);
  var c = P.colour.rgb * (edge * (2.5 + 3.0 * n) * flicker + outer + inner) * G.params2.y;
  // The glow is seen through this view's air as well.
  let fog = fogParams(u32(V.location));
  let dir = in.world - V.camPos;
  let seg = max(length(dir) - entryDistance(V, normalize(dir)), 0.0);
  c *= exp(-fog.w * G.params.z * seg) * (1.0 - V.fade);
  return vec4f(c / GLOW_SCALE, 0.0);
}

// End of the recursion: the innermost opening shows the rim colour instead of
// another view (the fade over the last levels makes this nearly invisible).
struct FillOut {
  @location(0) colour: vec4f,
  @location(2) info: vec4f,
  @location(3) glow: vec4f,
};

@vertex
fn fillVs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return fullscreen(vi);
}

@fragment
fn fillFs() -> FillOut {
  var out: FillOut;
  out.colour = vec4f(G.portals[u32(D.flags.z)].colour.rgb * FADE_BRIGHTNESS, 1.0);
  out.info = vec4f((V.level + 1.0) / 8.0, fract(V.viewId / 64.0 + 0.5), 0.25, 1.0);
  out.glow = vec4f(0.0);
  return out;
}
