// The scene traced by both the LUT path (scene.wgsl) and the brute-force
// reference (reference.wgsl): a spherical planet whose surface around the
// starting site carries a 400 km height-field patch of mountains (baked
// once by terrain.wgsl). Requires F: Frame and the atmosphere bindings.

@group(0) @binding(2) var heightTex: texture_2d<f32>;
@group(0) @binding(3) var linearClamp: sampler;

struct Hit {
  t: f32,          // km along the ray, < 0 for sky
  pos: vec3f,      // km, planet centred
  normal: vec3f,
  albedo: vec3f,
  water: f32,      // 1 on the ocean
};

// Gnomonic patch coordinates (km) of a planet-centred point, and whether
// the point is on the patch side of the planet.
fn patchCoords(p: vec3f) -> vec3f {
  let up = dot(p, F.siteUp);
  let s = ATMO.bottomRadius / max(up, 1e-3);
  return vec3f(dot(p, F.siteEast) * s, dot(p, F.siteNorth) * s, select(0.0, 1.0, up > 0.0));
}

// (height km, d/dx, d/dy) of the terrain at a planet-centred point.
fn terrainAt(p: vec3f) -> vec3f {
  let c = patchCoords(p);
  let uv = c.xy / (2.0 * F.patchHalfSize) + 0.5;
  if (c.z < 0.5 || any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return vec3f(0.0); }
  // Flip v: texel row 0 is the north edge.
  let s = textureSampleLevel(heightTex, linearClamp, vec2f(uv.x, 1.0 - uv.y), 0.0).xyz;
  return s * F.terrainScale;
}

const TERRAIN_TOP: f32 = 6.5; // km, above the highest peak at terrainScale 1

// Surface colour: low-frequency continents and oceans on the sphere (Earth
// only; the mountain patch always sits on land), then grass, rock and snow
// on the terrain by height and slope. Returns (albedo, water).
fn sceneAlbedo(p: vec3f, n: vec3f, height: f32) -> vec4f {
  let up = normalize(p);
  let slope = 1.0 - dot(n, up);
  let large = fbm3(up * 60.0, 4);
  let fine = fbm3(up * 2400.0, 3);
  if (F.planet < 0.5) {
    let site = acos(clamp(dot(up, F.siteUp), -1.0, 1.0)) * ATMO.bottomRadius;
    let continent = fbm3(up * 2.2 + vec3f(0.3, 5.1, 2.7), 5) - 0.5 + 0.35 * smoothstep(900.0, 250.0, site);
    if (continent < 0.0 && height <= 0.0) {
      // Deep water: dark, slightly blue; shallower towards the coast.
      let shelf = smoothstep(-0.06, 0.0, continent);
      return vec4f(mix(vec3f(0.012, 0.025, 0.045), vec3f(0.03, 0.07, 0.08), shelf), 1.0);
    }
  }
  // Plains: the ground albedo of the atmosphere model (so the
  // multiple-scattering bounce matches what is drawn), tinted green or ochre
  // at unit luminance by large-scale "fields and forests" noise.
  let fields = smoothstep(0.4, 0.6, fbm3(up * 25.0 + 3.0, 3));
  let tint = mix(vec3f(0.4, 0.75, 0.3), vec3f(1.35, 1.1, 0.62), fields);
  var albedo = ATMO.groundAlbedo * select(tint, vec3f(1.0), F.planet > 0.5) * (0.55 + 0.9 * large) * (0.75 + 0.5 * fine);
  let grass = select(vec3f(0.07, 0.1, 0.04), ATMO.groundAlbedo * 0.8, F.planet > 0.5) * (0.8 + 0.6 * fine);
  let rock = select(vec3f(0.2, 0.18, 0.16), ATMO.groundAlbedo * vec3f(0.9, 0.8, 0.75), F.planet > 0.5) * (0.8 + 0.4 * fine);
  let t = smoothstep(0.05, 0.4, height);
  albedo = mix(albedo, grass, t * 0.8);
  albedo = mix(albedo, rock, smoothstep(0.15, 0.35, slope) * smoothstep(0.1, 0.8, height));
  let snowAt = F.snowLine - 0.6 * (fine - 0.5) - 1.2 * slope;
  let snow = smoothstep(snowAt - 0.1, snowAt + 0.1, height) * (1.0 - smoothstep(0.4, 0.55, slope));
  return vec4f(mix(albedo, vec3f(0.8, 0.82, 0.86), snow), 0.0);
}

fn cameraRay(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let q = ATMO.invViewProj * vec4f(ndc, 1.0, 1.0);
  return normalize(q.xyz / q.w);
}

// March the height field inside the shell [R, R + TERRAIN_TOP], then fall
// back to the sphere. Steps scale with the clearance above the terrain and
// with distance; the hit is refined by bisection.
fn traceScene(dir: vec3f) -> Hit {
  var hit: Hit;
  hit.t = -1.0;
  let R = ATMO.bottomRadius;
  let h = max(ATMO.cameraAltitude, 0.0);
  let r = R + h;
  let mu = dot(ATMO.cameraPos, dir) / length(ATMO.cameraPos);
  let cam = ATMO.cameraPos;
  let sphere = atmoRaySphereC(r, mu, h * (2.0 * R + h));
  let sphereHit = sphere.x <= sphere.y && sphere.y > 0.0;
  var tEnd = select(1e9, max(sphere.x, 0.0), sphereHit);

  let top = TERRAIN_TOP * F.terrainScale;
  if (top > 0.0) {
    let shell = atmoRaySphereC(r, mu, (h - top) * (2.0 * R + h + top));
    if (shell.x <= shell.y && shell.y > 0.0) {
      var t = max(shell.x, 0.0);
      let tMax = min(select(shell.y, tEnd, sphereHit), t + 600.0);
      var tPrev = t;
      var found = false;
      for (var i = 0; i < 320; i++) {
        let p = cam + dir * t;
        let alt = length(p) - R;
        let d = alt - terrainAt(p).x;
        if (d < 0.0) { found = true; break; }
        tPrev = t;
        t += max(d * 0.55, 0.0008 * t + 0.004);
        if (t > tMax) {
          // Test the end point itself so a long step cannot skip low terrain.
          if (tPrev >= tMax) { break; }
          t = tMax;
        }
      }
      if (found) {
        var a = tPrev;
        var b = t;
        for (var k = 0; k < 6; k++) {
          let m = 0.5 * (a + b);
          let p = cam + dir * m;
          if (length(p) - R - terrainAt(p).x < 0.0) { b = m; } else { a = m; }
        }
        tEnd = b;
      } else if (!sphereHit) {
        return hit;
      }
    } else if (!sphereHit) {
      return hit;
    }
  } else if (!sphereHit) {
    return hit;
  }

  hit.t = tEnd;
  hit.pos = cam + dir * tEnd;
  let up = normalize(hit.pos);
  let terr = terrainAt(hit.pos);
  hit.normal = normalize(up - terr.y * F.siteEast - terr.z * F.siteNorth);
  let a = sceneAlbedo(hit.pos, hit.normal, terr.x);
  hit.albedo = a.rgb;
  hit.water = a.w;
  return hit;
}

// Soft height-field shadow towards the sun (terrain only).
fn terrainShadow(p: vec3f, n: vec3f) -> f32 {
  if (F.shadows < 0.5 || F.terrainScale <= 0.0) { return 1.0; }
  let R = ATMO.bottomRadius;
  var t = 0.03;
  var s = 1.0;
  let start = p + n * 0.005;
  for (var i = 0; i < 28; i++) {
    let q = start + ATMO.sunDir * t;
    let alt = length(q) - R;
    if (alt > TERRAIN_TOP * F.terrainScale) { break; }
    let d = alt - terrainAt(q).x;
    s = min(s, clamp(12.0 * d / t, 0.0, 1.0));
    if (s <= 0.0) { break; }
    t *= 1.3;
  }
  return s;
}

// Procedural stars: one candidate per cell of a direction grid, drawn as a
// pixel-sized Gaussian so they neither alias nor vanish.
fn stars(dir: vec3f) -> vec3f {
  if (F.stars <= 0.0) { return vec3f(0.0); }
  let scale = 120.0;
  let p = dir * scale;
  let cell = floor(p);
  let h = pcg3d(bitcast<vec3u>(vec3i(cell)) + vec3u(7u, 11u, 13u));
  let r = vec3f(unorm24(h.x), unorm24(h.y), unorm24(h.z));
  if (r.x < 0.99) { return vec3f(0.0); }
  let starPos = normalize(cell + 0.2 + 0.6 * r);
  let pixelAngle = 1.2 / F.resolution.y;
  let d = length(dir - starPos) / pixelAngle;
  let mag = pow((r.x - 0.99) / 0.01, 8.0);
  let tint = mix(vec3f(1.0, 0.8, 0.6), vec3f(0.7, 0.8, 1.0), r.y);
  return tint * mag * exp(-d * d) * 5e-4 * F.stars;
}

// Lambertian ground: direct sun (atmospheric transmittance `sunT`, terrain
// shadow) plus sky irradiance from the atmosphere's irradiance LUT. Water
// adds a GGX sun glint (roughness 0.2, Schlick Fresnel with F0 = 0.02) and
// a Fresnel-weighted reflection of a uniform sky of the same irradiance.
fn shadeGround(hit: Hit, sunT: vec3f, view: vec3f) -> vec3f {
  let n = hit.normal;
  let ndl = max(dot(n, ATMO.sunDir), 0.0);
  let sky = atmoSkyIrradiance(hit.pos, n);
  var direct = vec3f(0.0);
  if (ndl > 0.0) { direct = atmoSunIlluminance() * sunT * ndl * terrainShadow(hit.pos, n); }
  var color = hit.albedo / PI * (direct + sky);
  if (hit.water > 0.5) {
    let v = -view;
    let h = normalize(v + ATMO.sunDir);
    let ndv = max(dot(n, v), 1e-3);
    let ndh = max(dot(n, h), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let a2 = 0.2 * 0.2;
    let dd = ndh * ndh * (a2 - 1.0) + 1.0;
    let D = a2 / (PI * dd * dd);
    let k = 0.02;
    let G = ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
    let specular = D * G * (0.02 + 0.98 * pow(1.0 - max(dot(h, v), 0.0), 5.0)) / (4.0 * ndv);
    color = color * (1.0 - fresnel) + direct / max(ndl, 1e-4) * specular + fresnel * sky / PI;
  }
  return color;
}
