// The altar environment: a ruined open-air temple under a moonlit sky. It is
// rendered for the main camera and for the six faces of the reflection cube
// (from the object's centre). Output alpha is the view distance, which the
// refraction and distortion passes use to find what lies behind the object.
//
// Light sources: moonlight (shadow-mapped), two flickering braziers, and the
// magical object itself as a point light whose colour and power come from
// the field (probe.wgsl). All of them also glow in the height fog through a
// closed-form single-scattering integral.

@group(0) @binding(0) var<uniform> Vw: View;
@group(0) @binding(1) var<uniform> F: Frame;
@group(0) @binding(2) var<uniform> M: Material;
@group(0) @binding(3) var<storage, read> probe: Probe;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;

const GROUND: u32 = 0u;
const FLOOR: u32 = 1u;
const STONE: u32 = 2u;
const METAL: u32 = 3u;
const COALS: u32 = 4u;
const RUNE: u32 = 5u;

const SKY_DIST: f32 = 20000.0;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) material: u32,
};

@vertex
fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) material: f32) -> VSOut {
  var out: VSOut;
  out.position = Vw.viewProj * vec4f(position, 1.0);
  out.world = position;
  out.normal = normal;
  out.material = u32(material + 0.5);
  return out;
}

@vertex
fn shadowVs(@location(0) position: vec3f) -> @builtin(position) vec4f {
  return F.lightViewProj * vec4f(position, 1.0);
}

fn fbm3(p: vec3f) -> f32 {
  return noise(p) * 0.55 + noise(p * 2.03 + 7.1) * 0.28 + noise(p * 4.1 + 3.3) * 0.14;
}

// --- lighting ---------------------------------------------------------------

fn moonShadow(P: vec3f, N: vec3f) -> f32 {
  let c = F.lightViewProj * vec4f(P + N * 0.04, 1.0);
  let uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { return 1.0; }
  let texel = 1.0 / 2048.0;
  var s = 0.0;
  for (var j = -1; j <= 1; j++) {
    for (var i = -1; i <= 1; i++) {
      s += textureSampleCompareLevel(shadowMap, shadowSampler, uv + vec2f(f32(i), f32(j)) * texel * 1.5, c.z - 0.0015);
    }
  }
  return s / 9.0;
}

// The object as a soft spherical occluder of the moon, as opaque as its material.
fn objectShadow(P: vec3f) -> f32 {
  let L = F.moonDir;
  let toC = F.objCenter - P;
  let b = dot(toC, L);
  if (b < 0.0) { return 1.0; }
  let h = length(toC - L * b);
  let R = 0.75 * F.objScale;
  let occ = smoothstep(R * 0.35, R * (1.1 + 0.06 * b), h);
  let opacity = select(1.0 - exp(-dot(M.absorption.rgb, vec3f(0.33)) * 1.2) * 0.8, 1.0, M.surface.x > 0.5);
  return mix(1.0, occ, opacity);
}

fn pointLight(P: vec3f, N: vec3f, V: vec3f, pos: vec3f, I: vec3f, radius: f32, rough: f32, spec: f32) -> vec3f {
  let Lv = pos - P;
  let d2 = dot(Lv, Lv);
  let L = Lv * inverseSqrt(d2);
  // Sphere-light wrap: a light of this radius still reaches slightly past the terminator.
  let w = clamp(radius * radius / d2, 0.0, 0.6);
  let diff = max((dot(N, L) + w) / (1.0 + w), 0.0);
  let H = normalize(L + V);
  let shin = 2.0 / max(rough * rough * rough * rough, 1e-3);
  let sp = spec * pow(max(dot(N, H), 0.0), shin) * (shin + 8.0) / 25.0;
  return I * (diff / PI + sp * max(dot(N, L), 0.0)) / (d2 + radius * radius);
}

fn brazier(i: u32) -> vec4f {
  return select(F.brazierA, F.brazierB, i == 1u);
}

fn lightSurface(P: vec3f, N: vec3f, V: vec3f, albedo: vec3f, rough: f32, spec: f32) -> vec3f {
  let L = F.moonDir;
  let ndl = max(dot(N, L), 0.0);
  let sh = moonShadow(P, N) * objectShadow(P);
  let H = normalize(L + V);
  let shin = 2.0 / max(rough * rough * rough * rough, 1e-3);
  let moonSpec = spec * pow(max(dot(N, H), 0.0), shin) * (shin + 8.0) / 25.0;
  var c = F.moonColor * sh * ndl * (albedo / PI + moonSpec);
  // Sky ambient: moonlit blue from above, faint bounce from below.
  let skyAmb = mix(vec3f(0.006, 0.007, 0.01), vec3f(0.018, 0.024, 0.045), N.y * 0.5 + 0.5);
  c += skyAmb * albedo;
  for (var i = 0u; i < 2u; i++) {
    let b = brazier(i);
    c += pointLight(P, N, V, b.xyz + vec3f(0.0, 0.25, 0.0), F.fireColor * b.w, 0.25, rough, spec) * albedo;
  }
  c += pointLight(P, N, V, F.objCenter, probe.light.rgb, 0.7 * F.objScale, rough, spec) * albedo;
  return c;
}

// Single scattering from a point light in homogeneous fog along a ray segment:
// ∫ I / |o + s d − c|² ds = I / h · [atan((t − b) / h) + atan(b / h)].
fn pointGlow(ro: vec3f, rd: vec3f, tmax: f32, c: vec3f, I: vec3f) -> vec3f {
  let oc = c - ro;
  let b = dot(oc, rd);
  let h = sqrt(max(dot(oc, oc) - b * b, 0.02));
  return I * (atan((tmax - b) / h) + atan(b / h)) / h;
}

fn applyFog(col: vec3f, ro: vec3f, rd: vec3f, dist: f32) -> vec3f {
  // The reflection cube is rendered from inside the object: its own glow is
  // not in front of anything there.
  let objGlow = select(1.2, 0.0, Vw.camPos.w > 0.5);
  // Height fog, densest near the ground (analytic optical depth).
  let k = 0.35;
  let dens = F.fog;
  let y0 = ro.y;
  let dy = rd.y * dist;
  let od = select(dens * exp(-k * y0) * dist,
                  dens * exp(-k * y0) * (1.0 - exp(-k * dy)) / (k * rd.y), abs(rd.y) > 1e-3);
  let tr = exp(-min(od, 50.0));
  let fogCol = vec3f(0.012, 0.016, 0.03) + F.moonColor * 0.012;
  let tmax = min(dist, 60.0);
  let scatter = dens * 0.9 / (4.0 * PI);
  var glow = pointGlow(ro, rd, tmax, F.objCenter, probe.light.rgb) * objGlow;
  glow += pointGlow(ro, rd, tmax, F.brazierA.xyz + vec3f(0.0, 0.3, 0.0), F.fireColor * F.brazierA.w);
  glow += pointGlow(ro, rd, tmax, F.brazierB.xyz + vec3f(0.0, 0.3, 0.0), F.fireColor * F.brazierB.w);
  return col * tr + fogCol * (1.0 - tr) + glow * scatter * mix(1.0, tr, 0.5);
}

// --- materials --------------------------------------------------------------

// Concentric flagstones around the altar: rings of tiles, each ring split
// into roughly square stones, with bevelled joints.
fn flagstones(P: vec3f) -> vec4f {
  let r = length(P.xz);
  let ringW = 0.85;
  let ring = floor((r - 2.7) / ringW);
  let fr = fract((r - 2.7) / ringW);
  let circumference = TAU * (2.7 + (ring + 0.5) * ringW);
  let n = max(floor(circumference / 0.95), 6.0);
  let a = (atan2(P.z, P.x) / TAU + 0.5) * n + hash33(vec3i(i32(ring), 3, 9)).x * n;
  let seg = floor(a);
  let fa = fract(a);
  let id = hash33(vec3i(i32(ring), i32(seg), 1));
  let edge = min(min(fr, 1.0 - fr) * ringW, min(fa, 1.0 - fa) * circumference / n);
  let joint = smoothstep(0.0, 0.035, edge);
  return vec4f(id, joint);
}

// The engraved rune circle on the altar step: two rings, a hexagram, and a
// band of procedural glyphs. Returns the glow mask.
fn runeCircle(P: vec3f) -> f32 {
  let q = P.xz;
  let r = length(q);
  let w = 0.012;
  var m = 1.0 - smoothstep(w, w * 2.0, abs(r - 1.95));
  m = max(m, 1.0 - smoothstep(w, w * 2.0, abs(r - 1.52)));
  m = max(m, 1.0 - smoothstep(w, w * 2.0, abs(r - 1.0)));
  // Hexagram inscribed in r = 1.52: distance to six lines.
  var hex = 1e3;
  for (var k = 0; k < 6; k++) {
    let ang = f32(k) * PI / 3.0 + PI / 6.0;
    let nrm = vec2f(cos(ang), sin(ang));
    hex = min(hex, abs(dot(q, nrm) - 0.76));
  }
  if (r < 1.52 && r > 0.7) { m = max(m, 1.0 - smoothstep(w, w * 2.0, hex)); }
  // Glyph band between r = 1.55 and 1.92.
  if (r > 1.56 && r < 1.91) {
    let cells = 36.0;
    let a = (atan2(q.y, q.x) / TAU + 0.5) * cells;
    let cell = floor(a);
    let u = fract(a) * 2.0 - 1.0;
    let v = (r - 1.735) / 0.17;
    let h = hash33(vec3i(i32(cell), 5, 11));
    var g = 1e3;
    g = min(g, select(1e3, abs(u * 0.5), h.x > 0.3));                          // vertical stem
    g = min(g, select(1e3, abs(u * 0.5 - v * 0.35 * sign(h.y - 0.5)), h.y > 0.2)); // diagonal
    g = min(g, select(1e3, abs(length(vec2f(u * 0.5, v - 0.35 * (h.z - 0.5))) - 0.22), h.z > 0.55)); // ring
    g = min(g, select(1e3, abs(v + 0.6 * (h.x - 0.5)), h.z < 0.45));           // bar
    g = max(g, abs(v) - 0.85);
    m = max(m, 1.0 - smoothstep(0.03, 0.07, g * 0.17));
  }
  return m;
}

struct Out {
  @location(0) color: vec4f,
};

@fragment
fn fs(in: VSOut) -> Out {
  let P = in.world;
  var N = normalize(in.normal);
  let ro = Vw.camPos.xyz;
  let dist = distance(P, ro);
  let V = (ro - P) / dist;
  var albedo = vec3f(0.3);
  var rough = 0.8;
  var spec = 0.04;
  var emit = vec3f(0.0);
  let grain = fbm3(P * 3.1);

  switch in.material {
    case GROUND: {
      let n = fbm3(P * 0.6) * 0.5 + 0.5;
      albedo = mix(vec3f(0.035, 0.04, 0.03), vec3f(0.09, 0.08, 0.06), n) * (0.8 + 0.4 * grain);
      rough = 0.95;
    }
    case FLOOR: {
      let t = flagstones(P);
      let tone = 0.75 + 0.5 * t.x;
      albedo = vec3f(0.24, 0.23, 0.22) * tone * (0.85 + 0.3 * grain) * mix(0.35, 1.0, t.w);
      // Moss creeping in the joints.
      albedo = mix(albedo, vec3f(0.05, 0.08, 0.04), (1.0 - t.w) * 0.5 * smoothstep(0.0, 0.4, fbm3(P * 1.7)));
      rough = mix(0.9, 0.55, t.w * t.y);
      spec = 0.03;
      N = normalize(N + vec3f(noise(P * 9.0), 0.0, noise(P * 9.0 + 3.0)) * 0.08 * t.w);
    }
    case STONE, RUNE: {
      albedo = vec3f(0.3, 0.285, 0.27) * (0.8 + 0.35 * grain);
      // Masonry courses on vertical faces.
      let vertical = 1.0 - abs(N.y);
      let course = abs(fract(P.y / 0.52) - 0.5) * 0.52;
      albedo *= mix(1.0, mix(0.45, 1.0, smoothstep(0.0, 0.02, 0.26 - course)), vertical);
      // Weathering: darker streaks running down.
      albedo *= 1.0 - 0.35 * smoothstep(0.1, 0.6, noise(vec3f(P.x * 4.0, P.y * 0.5, P.z * 4.0))) * vertical;
      rough = 0.85;
      N = normalize(N + vec3f(noise(P * 11.0), noise(P * 11.0 + 5.0), noise(P * 11.0 + 9.0)) * 0.06);
      if (in.material == RUNE) {
        let m = runeCircle(P);
        albedo *= 1.0 - 0.6 * m;
        // The engraving glows with the object's light, a little late at the rim.
        let pulse = probe.light.rgb * 0.12 + energyColor(probe.light.w) * 0.25 * probe.stats.x;
        emit = m * pulse * (0.4 + 0.6 * smoothstep(0.8, 2.0, length(P.xz)));
      }
    }
    case METAL: {
      albedo = vec3f(0.035, 0.03, 0.028);
      rough = 0.45;
      spec = 0.5;
    }
    case COALS: {
      let n = fbm3(vec3f(P.xz * 9.0, F.time * 0.7).xzy);
      let heat = smoothstep(-0.25, 0.5, n);
      albedo = vec3f(0.02);
      emit = F.fireColor * (0.35 + 3.0 * heat * heat) * (F.brazierA.w + F.brazierB.w) * 0.08;
    }
    default: {}
  }

  var c = lightSurface(P, N, V, albedo, rough, spec) + emit;
  c = applyFog(c, ro, -V, dist);
  if (F.debug == 5.0) { c = vec3f(0.0); }
  var out: Out;
  out.color = vec4f(c, dist);
  return out;
}

// --- brazier flames ---------------------------------------------------------
// Additive billboards turning about the vertical axis toward the camera,
// drawn in the scene pass so the flames are refracted by the object and
// appear in its reflection cube.

struct FlameOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) which: u32,
};

@vertex
fn flameVs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FlameOut {
  var corners = array<vec2f, 6>(vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi];
  let b = brazier(ii);
  let base = b.xyz + vec3f(0.0, -0.08, 0.0);
  let toCam = Vw.camPos.xyz - base;
  let side = normalize(vec3f(toCam.z, 0.0, -toCam.x) + vec3f(1e-5, 0.0, 0.0));
  let world = base + side * c.x * 0.42 + vec3f(0.0, c.y * 1.25, 0.0);
  var out: FlameOut;
  out.position = Vw.viewProj * vec4f(world, 1.0);
  out.uv = c;
  out.which = ii;
  return out;
}

@fragment
fn flameFs(in: FlameOut) -> @location(0) vec4f {
  let t = F.time;
  let seed = f32(in.which) * 17.0;
  let uv = in.uv;
  // Rising turbulence, faster and finer toward the tips.
  let q = vec3f(uv.x * 2.2, uv.y * 2.0 - t * 1.9, seed + t * 0.35);
  let n = noise(q) * 0.6 + noise(q * 2.1 + 3.0) * 0.3 + noise(q * 4.3 + 7.0) * 0.15;
  let x = uv.x + n * 0.35 * uv.y;
  let width = mix(0.85, 0.08, pow(uv.y, 0.8));
  let body = 1.0 - smoothstep(width * 0.4, width, abs(x));
  let heightFade = 1.0 - smoothstep(0.25, 1.0, uv.y + n * 0.35);
  let dens = clamp(body * heightFade * (1.0 + n * 0.8), 0.0, 1.0);
  let tempK = dens * dens;
  let col = mix(vec3f(0.9, 0.12, 0.02), mix(vec3f(1.0, 0.45, 0.08), vec3f(1.0, 0.85, 0.5), smoothstep(0.55, 1.0, tempK)), smoothstep(0.1, 0.5, tempK));
  let b = brazier(in.which);
  var c = col * dens * 2.2 * b.w;
  if (F.debug == 5.0) { c = vec3f(0.0); }
  return vec4f(c, 0.0);
}

// --- sky --------------------------------------------------------------------

@vertex
fn skyVs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return fullscreenPosition(vi);
}

fn skyDir(px: vec2f, size: vec2f) -> vec3f {
  let ndc = vec2f(px.x / size.x * 2.0 - 1.0, 1.0 - px.y / size.y * 2.0);
  let p = Vw.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(p.xyz / p.w - Vw.camPos.xyz);
}

fn skyRadiance(d: vec3f) -> vec3f {
  let up = max(d.y, 0.0);
  var c = mix(vec3f(0.022, 0.028, 0.055), vec3f(0.003, 0.005, 0.014), sqrt(up));
  // Faint aurora-like veil: the same colours as the object, very dim.
  let veil = smoothstep(0.1, 0.7, fbm3(vec3f(d.x * 2.0, d.y * 5.0 - F.time * 0.01, d.z * 2.0)) + 0.35);
  c += mix(vec3f(0.0, 0.01, 0.012), vec3f(0.012, 0.004, 0.018), smoothstep(-0.5, 0.5, d.x)) * veil * smoothstep(0.05, 0.4, up);
  // Stars.
  let sp = d * 220.0;
  let cell = floor(sp);
  let h = hash33(vec3i(cell));
  if (h.x > 0.985) {
    let starPos = cell + 0.5 + (h - 0.5) * 0.6;
    let dd = length(sp - starPos);
    let tw = 0.7 + 0.3 * sin(F.time * (1.5 + 3.0 * h.y) + h.z * 40.0);
    c += vec3f(0.8, 0.85, 1.0) * mix(vec3f(1.0, 0.8, 0.6), vec3f(0.7, 0.8, 1.0), h.z) * exp(-dd * dd * 18.0)
      * (h.x - 0.985) * 120.0 * tw * smoothstep(0.0, 0.15, up);
  }
  // Moon: disc with limb darkening and a halo.
  let md = dot(d, F.moonDir);
  let ang = acos(clamp(md, -1.0, 1.0));
  let disc = 1.0 - smoothstep(0.028, 0.031, ang);
  let mare = 0.8 + 0.2 * fbm3(d * 90.0);
  c += vec3f(2.2, 2.3, 2.5) * disc * mare * (0.75 + 0.25 * sqrt(max(1.0 - ang / 0.03, 0.0)));
  c += F.moonColor * (0.05 * exp(-ang * 14.0) + 0.012 * exp(-ang * 3.0));
  // Mountain silhouettes on the horizon, moonlit on their edges.
  let az = atan2(d.z, d.x);
  let ridgeH = 0.035 + 0.06 * (noise(vec3f(az * 2.2, 1.0, 0.0)) * 0.6 + noise(vec3f(az * 7.0, 3.0, 0.0)) * 0.25 + 0.4);
  if (d.y < ridgeH) {
    let rim = exp(-(ridgeH - d.y) * 120.0) * max(dot(normalize(vec3f(d.x, 0.0, d.z)), F.moonDir), 0.0);
    c = vec3f(0.006, 0.007, 0.011) + F.moonColor * rim * 0.03;
  }
  return c;
}

@fragment
fn skyFs(@builtin(position) pos: vec4f) -> Out {
  // Cube faces render at their own size (camPos.w), the main camera at the screen's.
  let size = select(vec2f(Vw.camPos.w), F.resolution, Vw.camPos.w < 0.5);
  let d = skyDir(pos.xy, size);
  var c = skyRadiance(d);
  c = applyFog(c, Vw.camPos.xyz, d, 90.0);
  if (F.debug == 5.0) { c = vec3f(0.0); }
  var out: Out;
  out.color = vec4f(c, SKY_DIST);
  return out;
}
