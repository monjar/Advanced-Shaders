// The magical object. Rasterised from its mesh (so it depth-tests and
// occludes particles), shaded per pixel from the baked field:
//
//  surface   analytic crack network (pixel-exact, anti-aliased with fwidth)
//            whose opening is decided by the field's charge; grooves tilt the
//            normal along ∇(crack distance); runes/sigils per crack cell for
//            the metal preset; field-driven rim.
//  interior  (dielectrics) the view ray is refracted in, then marched through
//            the volume up to the exit found by sphere tracing the SDF:
//            Beer–Lambert absorption, energy emission, glowing fracture
//            sheets near the surface, and in-scattered moonlight. The march
//            direction is bent by ∇C (a gradient-index medium whose index
//            follows the charge), so the energy itself refracts light.
//  exit      the ray is refracted out per colour channel (dispersion) and
//            the scene behind is looked up in screen space at the point the
//            refracted ray would reach, blurred by the frost level.
//  metal     anisotropic GGX (brushed along the mesh tangent) with glowing
//            channels that are windows into the same interior energy.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> M: Material;
@group(0) @binding(2) var sceneTex: texture_2d<f32>;   // scene behind the object (warped), mips, alpha = distance
@group(0) @binding(3) var envCube: texture_cube<f32>;  // reflection cube rendered from the object centre
@group(0) @binding(4) var linearSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) world: vec3f,
  @location(1) local: vec3f,
  @location(2) normal: vec3f,
  @location(3) tangent: vec3f,
};

@vertex
fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) tangent: vec3f) -> VSOut {
  var out: VSOut;
  let world = toWorld(position);
  out.position = F.viewProj * vec4f(world, 1.0);
  out.world = world;
  out.local = position;
  out.normal = normal;
  out.tangent = tangent;
  return out;
}

fn ggxD(NoH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn smithV(NoV: f32, NoL: f32, a: f32) -> f32 {
  return 0.5 / max(NoL * (NoV * (1.0 - a) + a) + NoV * (NoL * (1.0 - a) + a), 1e-5);
}

// Anisotropic GGX with height-correlated Smith visibility (Heitz 2014).
fn anisoSpec(n: vec3f, t: vec3f, b: vec3f, V: vec3f, L: vec3f, at: f32, ab: f32) -> f32 {
  let H = normalize(V + L);
  let NoL = max(dot(n, L), 0.0);
  let NoV = max(dot(n, V), 1e-4);
  let NoH = max(dot(n, H), 0.0);
  let ToH = dot(t, H);
  let BoH = dot(b, H);
  let dd = ToH * ToH / (at * at) + BoH * BoH / (ab * ab) + NoH * NoH;
  let D = 1.0 / (PI * at * ab * dd * dd);
  let lv = NoL * length(vec3f(at * dot(t, V), ab * dot(b, V), NoV));
  let ll = NoV * length(vec3f(at * dot(t, L), ab * dot(b, L), NoL));
  return D * 0.5 / max(lv + ll, 1e-5) * NoL;
}

fn isoSpec(n: vec3f, V: vec3f, L: vec3f, a: f32) -> f32 {
  let H = normalize(V + L);
  let NoL = max(dot(n, L), 0.0);
  return ggxD(max(dot(n, H), 0.0), a) * smithV(max(dot(n, V), 1e-4), NoL, a) * NoL;
}

fn schlick(F0: vec3f, c: f32) -> vec3f {
  return F0 + (1.0 - F0) * pow(1.0 - c, 5.0);
}

// Pixel-exact band coverage: 1 where d < w, anti-aliased over `aa`, and
// fading out entirely as the band closes (w → 0) so closed cracks vanish.
fn band(d: f32, w: f32, aa: f32) -> f32 {
  return clamp((w - d) / (2.0 * aa) + 0.5, 0.0, 1.0) * clamp(w / aa, 0.0, 1.0);
}

struct March {
  L: vec3f,
  Tr: vec3f,
  exitP: vec3f,
  dir: vec3f,
  thick: f32,
  steps: f32,
};

// Interior march (object space). Step count adapts to the chord length:
// a full diameter uses the maximum, thin parts only a few.
fn marchInterior(p0: vec3f, dir0: vec3f, jitter: f32, moonO: vec3f) -> March {
  var m: March;
  // 1. Exit distance along the straight ray: sphere trace the interior
  //    distance (−SDF). The first step leaves the entry surface.
  var t = 0.012;
  for (var i = 0; i < 40; i++) {
    let s = -sampleShape(p0 + dir0 * t).x;
    if (s < 0.004 && t > 0.03) { break; }
    t += max(s, 0.012);
    if (t > 2.6) { break; }
  }
  let chord = min(t, 2.6);
  let maxSteps = M.tint.w;
  let n = clamp(ceil(chord / 2.0 * maxSteps), 4.0, maxSteps);
  let dt = chord / n;

  var q = p0 + dir0 * dt * jitter;
  var dir = dir0;
  var Tr = vec3f(1.0);
  var L = vec3f(0.0);
  var steps = 0.0;
  var travelled = 0.0;
  let sigmaAvg = dot(M.absorption.rgb, vec3f(0.333));
  for (var i = 0; i < 96; i++) {
    if (f32(i) >= n) { break; }
    let sh = sampleShape(q);
    // Bent rays can leave before the straight-line exit.
    if (sh.x > 0.01 && f32(i) > 1.0) { break; }
    let depth = max(-sh.x, 0.0);
    let f = sampleField(q);
    let c = sampleCrack(q);
    let open = crackOpen(f.C, f.E);
    // Fracture sheets: the crack network continues inside as planes that
    // glow where open and fade with depth below the surface.
    let w = M.crack.x;
    let sheet = open * (exp(-c.x / (w * 1.5 + 0.004)) + 0.6 * smoothstep(0.5, 1.0, open) * exp(-c.y / (w + 0.003)))
      * exp(-depth / max(M.crack.w, 1e-3));
    let core = smoothstep(M.light.z * 0.3, M.light.z * 1.4 + 1e-3, depth);
    var emit = energyEmission(f.E, f.C) * core + M.crackColor.rgb * M.crack.z * 0.35 * sheet * (0.5 + 1.5 * f.E);
    // In-scattering (the subsurface look): moonlight reaching this depth,
    // estimated from the distance to the surface, plus a faint self-glow.
    let toLight = depth / max(dot(sh.yzw, moonO) * 0.5 + 0.5, 0.25);
    emit += M.scatter.rgb * M.scatter.w * (F.moonColor * 0.25 * exp(-toLight * sigmaAvg * 1.5) + 0.01);
    let sigma = max(M.absorption.rgb + f.E * f.E * M.absorption.w, vec3f(1e-4));
    let tr = exp(-sigma * dt);
    // Exact integral of constant emission over the step (energy conserving).
    L += Tr * emit * (1.0 - tr) / sigma;
    Tr *= tr;
    // Gradient-index medium: the ray bends toward higher charge.
    dir = normalize(dir + sampleGradC(q) * M.flow.w * dt);
    q += dir * dt;
    travelled += dt;
    steps += 1.0;
    if (max(max(Tr.x, Tr.y), Tr.z) < 0.01) { break; }
  }
  // Snap the exit onto the surface.
  let sh = sampleShape(q);
  m.exitP = q - sh.yzw * sh.x;
  m.L = L;
  m.Tr = Tr;
  m.dir = dir;
  m.thick = travelled;
  m.steps = steps;
  return m;
}

// Background seen through the exit point, per colour channel.
fn refractedBackground(exitP: vec3f, dir: vec3f, uv0: vec2f, frost: f32) -> array<vec3f, 2> {
  let sh = sampleShape(exitP);
  let nOut = sh.yzw;
  let Pw = toWorld(exitP);
  let bgDist = textureSampleLevel(sceneTex, linearSampler, uv0, 0.0).a;
  let dBg = clamp(bgDist - distance(F.camPos, Pw), 0.25, 40.0);
  var col = vec3f(0.0);
  var offG = vec2f(0.0);
  for (var ch = 0; ch < 3; ch++) {
    let eta = M.surface.y + (f32(ch) - 1.0) * M.surface.z;
    var T = refract(dir, -nOut, eta);
    var c = vec3f(0.0);
    if (dot(T, T) < 1e-4) {
      // Total internal reflection: light that bounced inside comes, on
      // average, from the environment around the object.
      T = reflect(dir, -nOut);
      c = textureSampleLevel(envCube, linearSampler, toWorldDir(T), 3.0).rgb * 0.6;
    } else {
      let Tw = toWorldDir(T);
      let pr = projectUv(Pw + Tw * dBg);
      if (pr.z > 0.0 && all(pr.xy > vec2f(0.0)) && all(pr.xy < vec2f(1.0))) {
        c = textureSampleLevel(sceneTex, linearSampler, pr.xy, frost).rgb;
        if (ch == 1) { offG = pr.xy - uv0; }
      } else {
        c = textureSampleLevel(envCube, linearSampler, Tw, frost).rgb;
      }
    }
    col[ch] = c[ch];
  }
  return array<vec3f, 2>(col, vec3f(offG, 0.0));
}

// A sigil in one crack cell: rings, a k-gon with spokes, ticks, drawn in
// the cell's tangent plane. Returns the line mask. `reveal` in [0, 1]
// draws it progressively around the circle as the cell charges up.
fn sigil(lp: vec2f, id: f32, pw: f32, reveal: f32) -> f32 {
  let h = hash33(vec3i(i32(id * 65536.0), 7, 3));
  let r = length(lp);
  let a = atan2(lp.y, lp.x);
  let lw = 0.035;
  let k = 3.0 + floor(h.y * 4.0);
  let rot = h.z * TAU;
  let sa = TAU / k;
  let aa = (fract((a + rot) / sa) - 0.5) * sa;
  var d = abs(r - 0.62);
  d = min(d, select(1e3, abs(r - 0.48), h.x > 0.35));
  let apothem = 0.44 * cos(PI / k);
  if (r < 0.48) { d = min(d, abs(r * cos(aa) - apothem)); }
  // Spokes to the polygon's vertices.
  let av = (fract((a + rot) / sa + 0.5) - 0.5) * sa;
  if (r < 0.44 && h.x < 0.75) { d = min(d, abs(r * sin(av))); }
  // Ticks around the outer ring.
  let ticks = 8.0 + floor(h.z * 3.0) * 4.0;
  let at = (fract(a / TAU * ticks) - 0.5) / ticks * TAU;
  if (r > 0.62 && r < 0.74) { d = min(d, abs(r * sin(at))); }
  d = min(d, abs(r - 0.08));
  let m = 1.0 - smoothstep(lw - pw, lw + pw, d);
  let t = fract(a / TAU + h.x);
  return m * (1.0 - smoothstep(reveal - 0.05, reveal, t)) * step(r, 0.76);
}

fn heat(x: f32) -> vec3f {
  let t = clamp(x, 0.0, 1.0);
  return clamp(vec3f(
    1.5 - abs(4.0 * t - 3.0),
    1.5 - abs(4.0 * t - 2.0),
    1.5 - abs(4.0 * t - 1.0)), vec3f(0.0), vec3f(1.0));
}

struct Out {
  @location(0) color: vec4f,
};

@fragment
fn fs(in: VSOut) -> Out {
  // Everything needing derivatives first, in uniform control flow.
  let p = in.local;
  let cr = crackField(p);
  let aa1 = max(fwidth(cr.d1), 1e-5);
  let aa2 = max(fwidth(cr.d2), 1e-5);
  let pw = max(length(fwidth(p)), 1e-5);

  let uv0 = in.position.xy / F.resolution;
  let jitter = ign(in.position.xy, F.frameIndex);
  var n = normalize(in.normal);
  let Vw = normalize(F.camPos - in.world);
  let rd = toObjectDir(-Vw);
  let V = -rd;
  let moonO = toObjectDir(F.moonDir);

  // The field at the surface, and just outside it (for the rim corona).
  let f = sampleField(p);
  let fo = sampleField(p + n * 0.12);
  let open = crackOpen(f.C, f.E);
  let metal = M.surface.x > 0.5;

  // Cracks. Metal channels are engraved: always present, lit by the field.
  let w1 = M.crack.x * select(open, 0.7 + 0.3 * open, metal);
  let w2 = M.crack.x * 0.55 * smoothstep(0.45, 1.0, open);
  let m1 = band(cr.d1, w1, aa1);
  let core = band(cr.d1, w1 * 0.35, aa1);
  let m2 = band(cr.d2, w2, aa2) * (1.0 - m1);
  let hairline = 1.0 - smoothstep(0.0, aa1 * 1.5, cr.d1);

  // Groove: height rises away from the crack, so the normal tilts toward it.
  let gt = cr.grad1 - n * dot(cr.grad1, n);
  let prof = 1.0 - smoothstep(0.0, max(w1 * 2.2, aa1 * 2.0), cr.d1);
  let ns = normalize(n - gt * prof * M.surface2.w * max(open, select(0.0, 1.0, metal)));
  let NoV = clamp(dot(ns, V), 1e-3, 1.0);

  // Energy runs along open cracks with the field's filaments.
  let along = 0.3 + 2.0 * pow(f.E, 1.4) * (0.5 + f.C);
  var crackEmit = (M.crackColor.rgb * m1 + M.crackHot.rgb * core * 1.5 + M.crackColor.rgb * m2 * 0.7) * M.crack.z * along;
  // Light leaking into the material around each crack.
  crackEmit += M.crackColor.rgb * M.crack.z * 0.07 * open * exp(-cr.d1 / (w1 * 1.4 + 0.006));
  if (metal) { crackEmit *= open * 0.85 + 0.15; }

  // Rim corona: grazing angles see the energy flowing just outside the surface.
  let rimF = pow(1.0 - NoV, M.rim.w);
  let rim = M.rim.rgb * M.haze.w * rimF * (0.2 + 2.0 * fo.E * (0.3 + fo.C)) * (0.5 + 0.7 * fo.C);

  let rough = M.surface.w;
  let a = max(rough * rough, 0.002);
  var col = vec3f(0.0);
  var thick = 0.0;
  var steps = 0.0;
  var offset = vec2f(0.0);

  if (!metal) {
    // ---- Dielectric -------------------------------------------------------
    let ior = M.surface.y;
    let f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let Fr = f0 + (1.0 - f0) * pow(1.0 - NoV, 5.0);
    let R = reflect(rd, ns);
    var refl = textureSampleLevel(envCube, linearSampler, toWorldDir(R), sqrt(rough) * 5.0).rgb;
    var spec = F.moonColor * 3.0 * isoSpec(ns, V, moonO, max(a, 0.004));
    for (var i = 0; i < 2; i++) {
      let b = select(F.brazierA, F.brazierB, i == 1);
      let Lb = toObject(b.xyz + vec3f(0.0, 0.25, 0.0)) - p;
      let dw = distance(b.xyz, in.world);
      spec += F.fireColor * b.w * isoSpec(ns, V, normalize(Lb), max(a, 0.01)) / (dw * dw + 0.1);
    }

    let T = refract(rd, ns, 1.0 / ior);
    let mr = marchInterior(p, T, jitter, moonO);
    let bg = refractedBackground(mr.exitP, mr.dir, uv0, M.surface2.x);
    offset = bg[1].xy;
    thick = mr.thick;
    steps = mr.steps;
    let transmitted = bg[0] * mr.Tr * M.tint.rgb + mr.L;

    // Translucency (Barré-Brisebois & Bouchard, GDC 2011): moonlight through
    // thin parts, attenuated by the thickness the march just measured.
    let sigmaAvg = dot(M.absorption.rgb, vec3f(0.333));
    let tl = pow(clamp(dot(rd, normalize(moonO + ns * 0.35)), 0.0, 1.0), 6.0);
    let thin = exp(-thick * sigmaAvg * 0.7);
    let wrap = max((dot(ns, moonO) + 0.6) / 1.6, 0.0);
    let sss = M.scatter.rgb * M.scatter.w * F.moonColor * (tl * thin * 2.5 + wrap * 0.06 * (1.0 - thin * 0.5));

    // Broken surface inside open cracks does not reflect.
    let surfMask = 1.0 - m1;
    col = (1.0 - Fr) * transmitted + (Fr * refl + spec) * surfMask + sss + crackEmit + rim;
    // Closed fractures still catch the light as faint lines.
    col *= 1.0 - 0.25 * hairline * (1.0 - open);
  } else {
    // ---- Anisotropic metal with rune channels -----------------------------
    let tng = normalize(in.tangent - n * dot(in.tangent, n) + vec3f(1e-5, 0.0, 0.0));
    let t0 = normalize(tng - ns * dot(tng, ns));
    let b0 = cross(ns, t0);
    let an = M.surface2.y;
    let at = max(a * (1.0 + an), 0.002);
    let ab = max(a * (1.0 - an), 0.002);
    // Bent reflection normal for the environment (Filament's approximation).
    let anT = cross(b0, V);
    let anN = cross(anT, b0);
    let bent = normalize(mix(ns, anN, an * 0.8));
    let R = reflect(rd, bent);
    let F0 = M.tint.rgb;
    let Fm = schlick(F0, NoV);
    let refl = textureSampleLevel(envCube, linearSampler, toWorldDir(R), sqrt(rough) * 5.0).rgb;
    var spec = F.moonColor * 2.0 * anisoSpec(ns, t0, b0, V, moonO, at, ab);
    for (var i = 0; i < 2; i++) {
      let b = select(F.brazierA, F.brazierB, i == 1);
      let Lb = normalize(toObject(b.xyz + vec3f(0.0, 0.25, 0.0)) - p);
      let dw = distance(b.xyz, in.world);
      spec += F.fireColor * b.w * anisoSpec(ns, t0, b0, V, Lb, at, ab) / (dw * dw + 0.1);
    }

    // Sigils: one per crack cell, in the cell's tangent plane, activated by
    // the charge at the cell's feature point.
    var runeMask = 0.0;
    var runeAct = 0.0;
    if (M.surface2.z > 0.0) {
      let fc = normalize(cr.feature + vec3f(1e-4));
      let ax = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(fc.x) > 0.6);
      let e1 = normalize(cross(fc, ax));
      let e2 = cross(fc, e1);
      let rel = p - cr.feature;
      let relT = rel - n * dot(rel, n);
      let scale = M.charge.w * 2.2;
      let lp = vec2f(dot(relT, e1), dot(relT, e2)) * scale;
      let cellC = sampleField(cr.feature).C;
      runeAct = smoothstep(M.crack.y - 0.15, M.crack.y + 0.35, cellC + 0.2 * f.E);
      runeMask = sigil(lp, cr.id, pw * scale, runeAct) * M.surface2.z * step(0.3, fract(cr.id * 7.31));
    }

    // Channels and sigils are windows into the interior energy: march a
    // short way beneath them, so the glow has depth and parallax.
    let window = max(m1, runeMask);
    var inner = vec3f(0.0);
    if (window > 0.001) {
      var q = p;
      let dtw = 0.018;
      var tr = 1.0;
      for (var i = 0; i < 10; i++) {
        q += rd * dtw;
        let fi = sampleField(q);
        let e = energyEmission(fi.E, fi.C) * 1.4 + M.crackColor.rgb * 0.4 * crackOpen(fi.C, fi.E);
        inner += tr * e * dtw * 9.0;
        tr *= exp(-dtw * 7.0);
        steps += 1.0;
      }
    }
    // Glow reflecting off the brushed metal next to lit channels.
    let spill = M.crackColor.rgb * M.crack.z * 0.12 * open * exp(-cr.d1 / 0.035) * F0;
    let base = Fm * refl + spec * F0;
    col = base * (1.0 - window * 0.9) * (1.0 - 0.6 * prof)
      + window * inner * (0.35 + 0.65 * max(open, runeAct)) * M.crack.z * 0.35
      + crackEmit * m1 * 0.5 + runeMask * runeAct * (M.crackColor.rgb * 1.2 + M.crackHot.rgb * 0.6) * M.crack.z * (0.4 + f.E)
      + spill + rim;
  }

  var out: Out;
  out.color = vec4f(col, distance(F.camPos, in.world));

  // Debug views (alpha < 0 marks object pixels for the composite).
  let dbg = u32(F.debug + 0.5);
  if (dbg == 2u) { out.color = vec4f(m1, m2 + core * 0.5, open * 0.35, -1.0); }
  if (dbg == 3u) { out.color = vec4f(heat(thick / 2.0), -1.0); }
  if (dbg == 4u) { out.color = vec4f(heat(steps / M.tint.w), -1.0); }
  if (dbg == 5u) { out.color = vec4f(0.0, 0.0, 0.0, distance(F.camPos, in.world)); }
  if (dbg == 6u) { out.color = vec4f(offset * 4.0 + 0.5, 0.5, -1.0); }
  return out;
}
