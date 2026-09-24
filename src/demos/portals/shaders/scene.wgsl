// Scene pass, run once per view inside that view's stencil mask. Lighting
// depends only on the world position (which location a surface is in), so a
// surface looks the same seen directly or through any chain of portals; only
// the camera position used for view-dependent terms and fog is virtual.

@group(0) @binding(0) var<uniform> V: View;
@group(1) @binding(0) var<uniform> D: Draw;
@group(2) @binding(0) var<uniform> G: Globals;
@group(2) @binding(1) var shadowMap: texture_depth_2d_array;
@group(2) @binding(2) var shadowSampler: sampler_comparison;

// Materials (see scene.ts).
const PLAIN: u32 = 0u;
const BRICK: u32 = 1u;
const FLAGSTONE: u32 = 2u;
const METAL_PANEL: u32 = 3u;
const HANGAR_FLOOR: u32 = 4u;
const EMISSIVE: u32 = 5u;
const GRASS: u32 = 6u;
const BARK: u32 = 7u;
const FOLIAGE: u32 = 8u;
const STONE: u32 = 9u;
const WATER: u32 = 10u;
const WOOD: u32 = 11u;
const PAINTED: u32 = 12u;
const BALL: u32 = 13u;
const CUBE: u32 = 14u;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) albedo: vec3f,
  @location(3) @interpolate(flat) material: u32,
  @location(4) local: vec3f,
};

struct Targets {
  @location(0) colour: vec4f,
  @location(2) info: vec4f,
};

struct SkyTargets {
  @location(0) colour: vec4f,
  @location(2) info: vec4f,
  @location(3) glow: vec4f,
};

@vertex
fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) colour: vec4f) -> VSOut {
  var out: VSOut;
  let world = D.model * vec4f(position, 1.0);
  out.position = V.viewProj * world;
  out.world = world.xyz;
  out.normal = (D.model * vec4f(normal, 0.0)).xyz;
  out.albedo = colour.rgb;
  out.material = u32(colour.a * 255.0 + 0.5);
  out.local = position;
  return out;
}

fn locationOf(p: vec3f) -> u32 {
  if (p.x > 50.0) { return 1u; }
  if (p.x < -50.0) { return 2u; }
  return 0u;
}

struct Surface {
  albedo: vec3f,
  rough: f32,
  metal: f32,
  emission: vec3f,
  n: vec3f,
  cavity: f32,
};

// Coordinates on the dominant plane of a surface (for wall and floor patterns).
fn planeUV(p: vec3f, n: vec3f) -> vec2f {
  let a = abs(n);
  if (a.y > max(a.x, a.z)) { return p.xz; }
  if (a.x > a.z) { return vec2f(p.z, p.y); }
  return vec2f(p.x, p.y);
}

// Running-bond blocks: cell id, and distance (m) to the nearest joint.
fn blocks(uv: vec2f, size: vec2f) -> vec3f {
  let row = floor(uv.y / size.y);
  let u = uv.x / size.x + 0.5 * (row - 2.0 * floor(row * 0.5));
  let cell = vec2f(floor(u), row);
  let f = vec2f(fract(u), fract(uv.y / size.y));
  let edge = min(min(f.x, 1.0 - f.x) * size.x, min(f.y, 1.0 - f.y) * size.y);
  return vec3f(cell, edge);
}

fn surface(in: VSOut, n: vec3f) -> Surface {
  var s: Surface;
  s.albedo = in.albedo;
  s.rough = 0.7;
  s.metal = 0.0;
  s.emission = vec3f(0.0);
  s.n = n;
  s.cavity = 1.0;
  let p = in.world;
  switch in.material {
    case BRICK: {
      let b = blocks(planeUV(p, n), vec2f(0.56, 0.26));
      let v = hash21(b.xy);
      let grime = fbm3(p * 1.3);
      s.albedo *= (0.78 + 0.34 * v) * vec3f(1.0, 0.97 - 0.06 * v, 0.92 - 0.1 * v) * (0.8 + 0.35 * grime);
      let joint = 1.0 - smoothstep(0.006, 0.014, b.z);
      s.albedo = mix(s.albedo, vec3f(0.6, 0.53, 0.44), joint);
      s.cavity = 1.0 - 0.45 * joint;
      s.rough = 0.85;
    }
    case FLAGSTONE: {
      let b = blocks(p.xz + vec2f(0.3, 0.1), vec2f(1.1, 0.75));
      let v = hash21(b.xy + 7.0);
      s.albedo *= (0.8 + 0.3 * v) * (0.85 + 0.3 * fbm2(p.xz * 3.0));
      let joint = 1.0 - smoothstep(0.008, 0.02, b.z);
      s.albedo = mix(s.albedo, vec3f(0.32, 0.27, 0.22), joint);
      s.cavity = 1.0 - 0.5 * joint;
      s.rough = mix(0.75, 0.5, v);
    }
    case METAL_PANEL: {
      let uv = planeUV(p, n);
      let cell = floor(uv / vec2f(2.0, 1.5));
      let f = fract(uv / vec2f(2.0, 1.5)) * vec2f(2.0, 1.5);
      let edge = min(min(f.x, 2.0 - f.x), min(f.y, 1.5 - f.y));
      let seam = 1.0 - smoothstep(0.008, 0.016, edge);
      let rivet = 1.0 - smoothstep(0.012, 0.02, length(abs(f - vec2f(1.0, 0.75)) - vec2f(0.92, 0.67)));
      s.albedo *= (0.85 + 0.25 * hash21(cell)) * (0.9 + 0.2 * fbm2(uv * 4.0));
      s.albedo = mix(s.albedo, s.albedo * 0.35, seam) + rivet * 0.08;
      s.cavity = 1.0 - 0.6 * seam;
      s.rough = 0.42 + 0.2 * fbm2(uv * 9.0);
      s.metal = 0.55;
    }
    case HANGAR_FLOOR: {
      let f = fract(p.xz / 1.5) * 1.5;
      let edge = min(min(f.x, 1.5 - f.x), min(f.y, 1.5 - f.y));
      let seam = 1.0 - smoothstep(0.01, 0.02, edge);
      s.albedo *= (0.85 + 0.3 * hash21(floor(p.xz / 1.5))) * (0.8 + 0.4 * fbm2(p.xz * 2.0));
      s.albedo = mix(s.albedo, vec3f(0.02), seam);
      s.cavity = 1.0 - 0.5 * seam;
      s.rough = 0.35 + 0.35 * fbm2(p.xz * 5.0);
      s.metal = 0.3;
      // Hazard frames on the floor in front of every hangar portal.
      for (var i = 0u; i < NUM_PORTALS; i++) {
        let P = G.portals[i];
        if (u32(P.centre.w) != 1u) { continue; }
        let l = (P.worldToLocal * vec4f(p, 1.0)).xyz;
        let q = vec2f(abs(l.x), l.z - 0.95);
        let box = max(q.x - 1.35, abs(q.y) - 0.95);
        if (box < 0.0 && box > -0.16) {
          let stripe = step(0.5, fract((l.x + l.z) * 2.2));
          s.albedo = mix(vec3f(0.02), vec3f(0.85, 0.6, 0.05), stripe);
          s.rough = 0.5;
          s.metal = 0.0;
        }
      }
    }
    case EMISSIVE: {
      s.emission = in.albedo * 7.0;
      s.albedo = vec3f(0.05);
    }
    case GRASS: {
      let g = fbm2(p.xz * 0.35);
      let fine = noise2(p.xz * 7.0);
      s.albedo = mix(vec3f(0.12, 0.24, 0.07), vec3f(0.3, 0.38, 0.12), g) * (0.75 + 0.45 * fine);
      s.rough = 0.95;
    }
    case BARK: {
      let streak = noise2(vec2f((p.x + p.z) * 9.0, p.y * 1.2));
      s.albedo *= 0.65 + 0.6 * streak;
      s.rough = 0.95;
    }
    case FOLIAGE: {
      s.albedo *= 0.7 + 0.6 * noise3(p * 2.5);
      s.rough = 0.9;
    }
    case STONE: {
      let v = fbm3(p * 2.2);
      s.albedo *= 0.75 + 0.5 * v;
      // Moss on upward faces in the forest.
      if (locationOf(p) == 2u) {
        let moss = smoothstep(0.35, 0.8, n.y + 0.3 * (v - 0.5)) + smoothstep(0.6, 0.0, p.y + 0.4 * v) * 0.6;
        s.albedo = mix(s.albedo, vec3f(0.12, 0.25, 0.08), clamp(moss, 0.0, 1.0));
      }
      s.rough = 0.8;
    }
    case WATER: {
      let t = G.params.x;
      let w = vec2f(noise2(p.xz * 2.0 + t * 0.4), noise2(p.xz * 2.0 - t * 0.35 + 9.0)) - 0.5;
      s.n = normalize(n + vec3f(w.x, 0.0, w.y) * 0.12);
      s.rough = 0.06;
    }
    case WOOD: {
      let uv = planeUV(in.local, n);
      let plank = fract(uv.y * 5.0);
      let gap = 1.0 - smoothstep(0.02, 0.06, min(plank, 1.0 - plank));
      s.albedo *= (0.8 + 0.3 * noise2(vec2f(uv.x * 30.0, floor(uv.y * 5.0) * 7.0))) * (1.0 - 0.6 * gap);
      s.cavity = 1.0 - 0.5 * gap;
      s.rough = 0.8;
    }
    case PAINTED: {
      s.rough = 0.35;
      s.metal = 0.1;
    }
    case BALL: {
      // Beach-ball segments around the object's local y axis.
      let seg = floor((atan2(in.local.z, in.local.x) / PI * 0.5 + 0.5) * 6.0);
      let k = u32(seg) % 3u;
      s.albedo = select(select(vec3f(0.08, 0.3, 0.9), vec3f(0.95, 0.35, 0.05), k == 1u), vec3f(0.9), k == 0u);
      if (abs(in.local.y) > 0.27) { s.albedo = vec3f(0.9); }
      s.rough = 0.3;
    }
    case CUBE: {
      // A crate with a glowing emblem on every face.
      let a = abs(in.local);
      let face = select(select(in.local.xy, in.local.xz, a.y > max(a.x, a.z)), in.local.yz, a.x > max(a.y, a.z));
      let r = length(face);
      let ring = smoothstep(0.012, 0.0, abs(r - 0.13));
      let bevel = smoothstep(0.2, 0.235, max(abs(face.x), abs(face.y)));
      s.albedo = mix(vec3f(0.62, 0.64, 0.68), vec3f(0.35, 0.36, 0.4), bevel);
      s.emission = vec3f(1.0, 0.45, 0.1) * ring * 5.0;
      s.rough = 0.4;
      s.metal = 0.5;
    }
    default: {}
  }
  return s;
}

// Sun shadows in the courtyard (layer 0) and the forest (layer 1): 3×3 PCF.
fn sunShadow(loc: u32, p: vec3f, n: vec3f) -> f32 {
  if (loc == 1u) { return 1.0; }
  let layer = select(0u, 1u, loc == 2u);
  let sp = G.shadowMats[layer] * vec4f(p + n * 0.04, 1.0);
  let uv = sp.xy * vec2f(0.5, -0.5) + 0.5;
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
  let texel = 1.0 / vec2f(textureDimensions(shadowMap));
  var sum = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      sum += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2f(f32(x), f32(y)) * texel * 1.5, layer, sp.z - 0.0008);
    }
  }
  return sum / 9.0;
}

// GGX specular with Schlick Fresnel and Smith-Schlick visibility, times N·L.
fn brdf(s: Surface, n: vec3f, v: vec3f, l: vec3f) -> vec3f {
  let h = normalize(v + l);
  let nl = max(dot(n, l), 0.0);
  let nv = max(dot(n, v), 1e-3);
  let nh = max(dot(n, h), 0.0);
  let a = max(s.rough * s.rough, 0.002);
  let a2 = a * a;
  let dd = nh * nh * (a2 - 1.0) + 1.0;
  let ndf = a2 / (PI * dd * dd);
  let k = a * 0.5;
  let vis = 0.25 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
  let f0 = mix(vec3f(0.04), s.albedo, s.metal);
  let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
  let diffuse = s.albedo * (1.0 - s.metal) * (1.0 - fres) / PI;
  return (diffuse + ndf * vis * fres) * nl;
}

// Cheap geometric occlusion: corners where floors meet walls, plus the
// moving objects as spheres (so the ball darkens the floor it bounces on).
fn ambientOcclusion(loc: u32, p: vec3f, n: vec3f) -> f32 {
  var ao = 1.0;
  if (loc != 2u) {
    let half = select(vec2f(10.0), vec2f(12.0, 9.0), loc == 1u);
    let c = select(vec2f(0.0), vec2f(100.0, 0.0), loc == 1u);
    let q = abs(p.xz - c) - half;
    let wall = -max(q.x, q.y);
    if (n.y > 0.5 && p.y < 0.05) { ao *= mix(0.55, 1.0, smoothstep(0.0, 1.8, wall)); }
  }
  if (abs(n.y) < 0.6) { ao *= mix(0.62, 1.0, smoothstep(0.0, 1.0, p.y)); }
  if (D.flags.w < 0.5) {
    for (var i = 0u; i < u32(G.params2.z); i++) {
      let o = G.occluders[i];
      let d = o.xyz - p;
      let l = length(d);
      if (l > o.w) {
        ao *= 1.0 - clamp(o.w * o.w / (l * l) * max(dot(n, d / l), 0.0), 0.0, 1.0) * 0.85;
      }
    }
  }
  return ao;
}

fn shade(in: VSOut, frontFacing: bool) -> vec3f {
  let p = in.world;
  var n0 = normalize(in.normal);
  if (!frontFacing) { n0 = -n0; }
  let s = surface(in, n0);
  let n = s.n;
  let v = normalize(V.camPos - p);
  let loc = locationOf(p);
  let ao = ambientOcclusion(loc, p, n0) * s.cavity;
  var c = s.emission;

  switch loc {
    case 0u: {
      let l = locationSunDir(0u);
      c += brdf(s, n, v, l) * vec3f(3.0, 2.05, 1.3) * sunShadow(0u, p, n0);
      let sky = mix(vec3f(0.3, 0.22, 0.15), vec3f(0.3, 0.33, 0.42), n.y * 0.5 + 0.5);
      // Warm bounce from the sunlit stone.
      let bounce = vec3f(0.34, 0.22, 0.12) * (1.0 - 0.5 * max(n.y, 0.0));
      c += s.albedo * (1.0 - s.metal * 0.5) * (sky + bounce) * ao;
    }
    case 1u: {
      for (var i = 0u; i < NUM_LIGHTS; i++) {
        let L = G.lights[i];
        let d = L.xyz - p;
        let d2 = dot(d, d);
        let fall = L.w / (d2 + 1.0) * pow(clamp(1.0 - pow(d2 / 900.0, 2.0), 0.0, 1.0), 2.0);
        c += brdf(s, n, v, d * inverseSqrt(d2)) * vec3f(0.78, 0.9, 1.0) * fall;
      }
      let amb = mix(vec3f(0.02, 0.03, 0.035), vec3f(0.05, 0.1, 0.13), n.y * 0.5 + 0.5);
      c += s.albedo * amb * ao;
      // Specular ambient from the lit ceiling for the metal.
      let r = reflect(-v, n);
      c += mix(vec3f(0.04), s.albedo, s.metal) * vec3f(0.12, 0.16, 0.2) * max(r.y, 0.0) * (1.0 - s.rough) * ao;
    }
    default: {
      let l = locationSunDir(2u);
      c += brdf(s, n, v, l) * vec3f(1.5, 1.6, 1.35) * sunShadow(2u, p, n0);
      let sky = mix(vec3f(0.1, 0.13, 0.08), vec3f(0.42, 0.55, 0.52), n.y * 0.5 + 0.5);
      c += s.albedo * sky * ao;
    }
  }

  if (in.material == WATER) {
    let r = reflect(-v, n);
    let fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    c = mix(c, skyRadiance(loc, r), fres);
  }

  // Light spilling through nearby portals from their destinations, plus the
  // rim's glow: each opening is treated as a small area light facing into
  // the room (radiance × area × cosines / distance², softened near it).
  let spill = G.params.y;
  if (spill > 0.0) {
    for (var i = 0u; i < NUM_PORTALS; i++) {
      let P = G.portals[i];
      if (u32(P.centre.w) != loc) { continue; }
      let d = P.centre.xyz - p;
      let l2 = dot(d, d);
      let dir = d * inverseSqrt(l2);
      let cp = -dot(P.plane.xyz, dir);
      let cs = dot(n, dir);
      if (cp > 0.0 && cs > 0.0) {
        let e = (P.spill.rgb + P.colour.rgb * 0.35) * (4.6 * cp * cs / (l2 + 1.5));
        c += s.albedo * (1.0 - s.metal * 0.5) * e * spill / PI;
      }
    }
  }
  return c;
}

fn applyFog(c: vec3f, p: vec3f) -> vec3f {
  let loc = u32(V.location);
  let fog = fogParams(loc);
  let dir = normalize(p - V.camPos);
  let seg = max(length(p - V.camPos) - entryDistance(V, dir), 0.0);
  let tr = exp(-fog.w * G.params.z * seg);
  return c * tr + fog.rgb * (1.0 - tr) * select(1.0, 0.35 + 0.65 * smoothstep(-0.2, 0.3, dir.y), loc == 1u);
}

@fragment
fn fs(in: VSOut, @builtin(front_facing) frontFacing: bool) -> Targets {
  // Per-draw clip plane: the part of an object that has already gone through
  // a portal (fragment discard: clip distances are an optional feature).
  if (dot(D.clip.xyz, in.world) + D.clip.w < 0.0) { discard; }
  // Static geometry has a hole behind every opening, so the portal's thin
  // box stays visible when the eye is closer than the near plane.
  if (D.flags.x > 0.5) {
    for (var i = 0u; i < NUM_PORTALS; i++) {
      let P = G.portals[i];
      let l = (P.worldToLocal * vec4f(in.world, 1.0)).xyz;
      let e = l.xy / PORTAL_AXES;
      if (l.z < 0.004 && l.z > -P.colour.w && dot(e, e) < 0.994) { discard; }
    }
  }
  var c = applyFog(shade(in, frontFacing), in.world);
  c = mix(c, V.fadeColour, V.fade);
  var out: Targets;
  out.colour = vec4f(c, 1.0);
  out.info = infoOut(V, D.flags.y);
  return out;
}

// Sky and depth reset for a view: a fullscreen triangle inside the view's
// stencil mask writes the far depth (0 with reversed Z) and the sky.
struct SkyOut {
  @builtin(position) position: vec4f,
};

@vertex
fn skyVs(@builtin(vertex_index) vi: u32) -> SkyOut {
  return SkyOut(fullscreen(vi));
}

@fragment
fn skyFs(@builtin(position) pos: vec4f) -> SkyTargets {
  let d = viewRay(V, pos.xy);
  var c = skyRadiance(u32(V.location), d);
  c = mix(c, V.fadeColour, V.fade);
  var out: SkyTargets;
  out.colour = vec4f(c, 1.0);
  out.info = infoOut(V, 0.0);
  out.glow = vec4f(0.0); // clears glow a neighbouring portal's halo left in this mask
  return out;
}

// Shadow maps: depth only, with the same per-draw clip plane.
struct ShadowOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
};

@vertex
fn shadowVs(@location(0) position: vec3f) -> ShadowOut {
  let world = D.model * vec4f(position, 1.0);
  return ShadowOut(V.viewProj * world, world.xyz);
}

@fragment
fn shadowFs(in: ShadowOut) {
  if (dot(D.clip.xyz, in.world) + D.clip.w < 0.0) { discard; }
}
