// Full-resolution composite: analytic sky or shaded ground (with cloud
// shadows and aerial perspective), clouds from the resolved buffer on top,
// debug views and map insets, then ACES tone mapping and sRGB output.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> C: Clouds;
@group(1) @binding(0) var cloudTex: texture_2d<f32>;
@group(1) @binding(1) var traceTex: texture_2d<f32>;
@group(1) @binding(2) var depthTex: texture_2d<f32>;
@group(1) @binding(3) var shadowTex: texture_2d<f32>;
@group(1) @binding(4) var weatherTex: texture_2d<f32>;
@group(1) @binding(5) var linearClamp: sampler;
@group(1) @binding(6) var linearRepeat: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3f) -> vec3f {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

fn linearToSrgb(c: vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

fn groundColor(worldXZ: vec2f, dist: f32) -> vec3f {
  let fields = fbm2(worldXZ * 0.0021, 4);
  let forest = smoothstep(0.52, 0.6, fbm2(worldXZ * 0.0007 + 11.0, 4));
  var albedo = mix(vec3f(0.16, 0.18, 0.07), vec3f(0.26, 0.24, 0.11), fields);
  albedo = mix(albedo, vec3f(0.05, 0.09, 0.04), forest);
  let fine = fbm2(worldXZ * 0.05, 3);
  return albedo * mix(0.8 + 0.4 * fine, 1.0, smoothstep(200.0, 2000.0, dist));
}

fn inset(uv: vec2f, origin: vec2f, size: f32) -> vec3f {
  // Returns (u, v, inside) for a square inset in the bottom-left.
  let aspect = F.resolution.x / F.resolution.y;
  let q = (uv - origin) / vec2f(size / aspect, size);
  let inside = all(q >= vec2f(0.0)) && all(q <= vec2f(1.0));
  return vec3f(q, select(0.0, 1.0, inside));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / F.resolution;
  let dir = viewRay(uv);
  let mode = u32(F.debugMode + 0.5);

  var background: vec3f;
  let ground = raySphere(dir, 0.0);
  if (ground.x <= ground.y && ground.x > 0.0) {
    let hit = cameraLocal() + dir * ground.x;
    let worldXZ = F.camPos.xz + hit.xz;
    let N = normalize(hit);
    let suv = (worldXZ - C.shadow.xy) / C.shadow.z + 0.5;
    var shadow = textureSampleLevel(shadowTex, linearClamp, suv, 0.0).r;
    shadow = mix(1.0, shadow, smoothstep(0.5, 0.45, max(abs(suv.x - 0.5), abs(suv.y - 0.5))));
    let albedo = groundColor(worldXZ, ground.x);
    let direct = F.sunColor * max(dot(N, F.sunDir), 0.0) * shadow / PI;
    background = albedo * (direct + F.ambient * 0.9);
    let fog = 1.0 - exp(-ground.x * C.view.y * 2.0);
    background = mix(background, skyRadiance(vec3f(dir.x, 0.02, dir.z), false), fog);
  } else {
    background = skyRadiance(dir, true);
  }

  let clouds = textureSampleLevel(cloudTex, linearClamp, uv, 0.0);
  var color = background * clouds.a + clouds.rgb;
  var tonemap = true;

  if (mode == 1u) {
    color = clouds.rgb;
  } else if (mode == 2u) {
    color = vec3f(clouds.a);
    tonemap = false;
  } else if (mode == 3u) {
    let tsize = vec2f(textureDimensions(traceTex));
    let d = textureLoad(depthTex, vec2i(uv * tsize), 0).r;
    color = vec3f(fract(d / 5000.0), d / C.layer.w, 0.0) * step(clouds.a, 0.99);
    tonemap = false;
  } else if (mode == 4u) {
    // Raw traced samples, nearest-neighbour: what the temporal pass works with.
    let tsize = vec2f(textureDimensions(traceTex));
    let raw = textureLoad(traceTex, vec2i(uv * tsize), 0);
    color = background * raw.a + raw.rgb;
  }

  if (tonemap) { color = aces(color * F.exposure); }

  // Insets: weather map (coverage red, type green) and shadow map.
  if (C.view.z > 0.5) {
    let w = inset(uv, vec2f(0.015, 0.63), 0.35);
    if (w.z > 0.5) {
      let wx = textureSampleLevel(weatherTex, linearRepeat, w.xy + C.weather.yz / C.weather.x, 0.0);
      color = vec3f(wx.r, wx.g * 0.6, wx.b * 0.3);
    }
  }
  if (C.view.w > 0.5) {
    let offset = select(0.0, 0.35 * F.resolution.y / F.resolution.x + 0.01, C.view.z > 0.5);
    let s = inset(uv, vec2f(0.015 + offset, 0.63), 0.35);
    if (s.z > 0.5) {
      color = vec3f(textureSampleLevel(shadowTex, linearClamp, s.xy, 0.0).r);
    }
  }

  let dither = (hash21(pos.xy + fract(F.time)) - 0.5) / 255.0;
  return vec4f(linearToSrgb(color) + dither, 1.0);
}
