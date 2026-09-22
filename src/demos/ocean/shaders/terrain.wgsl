// Island and seabed. Height from the baked terrain texture; sand, grass and
// rock by height and slope; wet sand near the water line; caustics and
// light absorption underwater. Also writes linear view distance for the
// water pass (refraction and depth colour).

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> O: Ocean;
@group(1) @binding(0) var terrainTex: texture_2d<f32>;
@group(1) @binding(1) var clampSampler: sampler;
@group(1) @binding(2) var causticTex: texture_2d<f32>;
@group(1) @binding(3) var linSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) uv: vec2f,
};

struct FSOut {
  @location(0) color: vec4f,
  @location(1) distance: f32,
};

@vertex
fn vs(@location(0) ij: vec2f) -> VSOut {
  let n = f32(textureDimensions(terrainTex).x) * 0.5;
  let uv = ij / n;
  let h = textureSampleLevel(terrainTex, clampSampler, uv, 0.0).x;
  let xz = (uv - 0.5) * 2.0 * O.terrain.x;
  let world = vec3f(xz.x, h, xz.y);
  var out: VSOut;
  out.position = F.viewProj * vec4f(world, 1.0);
  out.worldPos = world;
  out.uv = uv;
  return out;
}

@fragment
fn fs(in: VSOut) -> FSOut {
  let texel = 1.0 / vec2f(textureDimensions(terrainTex));
  let world = 2.0 * O.terrain.x * texel.x;
  let hl = textureSampleLevel(terrainTex, clampSampler, in.uv - vec2f(texel.x, 0.0), 0.0).x;
  let hr = textureSampleLevel(terrainTex, clampSampler, in.uv + vec2f(texel.x, 0.0), 0.0).x;
  let hd = textureSampleLevel(terrainTex, clampSampler, in.uv - vec2f(0.0, texel.y), 0.0).x;
  let hu = textureSampleLevel(terrainTex, clampSampler, in.uv + vec2f(0.0, texel.y), 0.0).x;
  var N = normalize(vec3f(hl - hr, 2.0 * world, hd - hu));

  let p = in.worldPos;
  // Small-scale sand ripples / grain in the normal.
  let rip = sin(p.x * 1.3 + fbm(p.xz * 0.3, 2) * 6.0) * 0.08;
  N = normalize(N + vec3f(rip, 0.0, rip * 0.5) * smoothstep(3.0, -1.0, p.y));

  let slope = 1.0 - N.y;
  let grain = fbm(p.xz * 0.7, 3);
  let drySand = vec3f(0.78, 0.69, 0.52) * (0.9 + 0.2 * grain);
  let wetSand = vec3f(0.42, 0.36, 0.27) * (0.9 + 0.2 * grain);
  let seabed = vec3f(0.62, 0.58, 0.46) * (0.85 + 0.3 * fbm(p.xz * 0.1, 3));
  let grass = mix(vec3f(0.16, 0.26, 0.08), vec3f(0.30, 0.34, 0.12), fbm(p.xz * 0.08, 3));
  let rock = vec3f(0.36, 0.33, 0.30) * (0.8 + 0.4 * grain);

  var albedo = mix(seabed, wetSand, smoothstep(-2.0, 0.0, p.y));
  albedo = mix(albedo, drySand, smoothstep(0.4, 1.4, p.y + grain * 0.6));
  albedo = mix(albedo, grass, smoothstep(2.5, 4.0, p.y + grain * 2.0) * smoothstep(0.35, 0.15, slope));
  albedo = mix(albedo, rock, smoothstep(0.25, 0.45, slope + grain * 0.1) * smoothstep(-1.0, 1.0, p.y));

  var color = shadeDiffuse(albedo, N, p);
  if (p.y > 0.0) { color = applyFog(color, p, O.terrain.w); }

  if (u32(F.debugMode + 0.5) == 6u) {
    color = causticsAt(p, max(-p.y, 0.0)) * 0.5 * select(0.0, 1.0, p.y < 0.0);
  } else if (u32(F.debugMode + 0.5) == 7u) {
    // Height bands: 1 m (red), 10 m (green), land in blue.
    color = vec3f(fract(p.y), fract(p.y / 10.0), step(0.0, p.y)) * 0.8;
  }

  var out: FSOut;
  out.color = vec4f(color, 1.0);
  out.distance = length(p - F.camPos);
  return out;
}
