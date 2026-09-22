// Floating bodies. The model matrix comes straight from the buoyancy compute
// pass through a storage buffer (instance index = body index).

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> O: Ocean;
@group(1) @binding(0) var<storage, read> bodies: array<Body>;
@group(1) @binding(1) var causticTex: texture_2d<f32>;
@group(1) @binding(2) var linSampler: sampler;

struct Body {
  model: mat4x4f,
  pos: vec4f,
  vel: vec4f,
  rot: vec4f,
  info: vec4f,
};

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) local: vec3f,
  @location(3) @interpolate(flat) kind: u32,
};

struct FSOut {
  @location(0) color: vec4f,
  @location(1) distance: f32,
};

@vertex
fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @builtin(instance_index) index: u32) -> VSOut {
  let b = bodies[index];
  let world = (b.model * vec4f(position, 1.0)).xyz;
  var out: VSOut;
  out.position = F.viewProj * vec4f(world, 1.0);
  out.worldPos = world;
  out.normal = normalize((b.model * vec4f(normal, 0.0)).xyz);
  out.local = position;
  out.kind = u32(b.info.x + 0.5);
  return out;
}

@fragment
fn fs(in: VSOut) -> FSOut {
  let N = normalize(in.normal);
  let p = in.local;
  var albedo: vec3f;
  var specular = 0.04;
  if (in.kind == 0u) {
    // Wooden crate: planks with grain and a darker frame.
    let plank = fract(p.y * 4.0 + 0.5);
    let grain = fbm(vec2f(p.x * 3.0 + p.z * 3.0, p.y * 40.0), 3);
    albedo = vec3f(0.55, 0.36, 0.18) * (0.75 + 0.35 * grain) * (0.8 + 0.2 * smoothstep(0.0, 0.08, plank) * smoothstep(1.0, 0.92, plank));
    // Darker frame where at least two coordinates are near a face.
    let a = abs(p);
    let nearFaces = step(0.43, a.x) + step(0.43, a.y) + step(0.43, a.z);
    albedo = mix(albedo, vec3f(0.3, 0.2, 0.11), step(1.5, nearFaces));
  } else if (in.kind == 1u) {
    // Navigation buoy: red and white bands.
    let band = step(0.5, fract(p.y * 2.5));
    albedo = mix(vec3f(0.75, 0.08, 0.05), vec3f(0.9, 0.9, 0.88), band);
    specular = 0.08;
  } else {
    // Beach ball: coloured segments.
    let a = atan2(p.z, p.x) / TAU + 0.5;
    let seg = u32(floor(a * 6.0)) % 3u;
    albedo = select(select(vec3f(0.95, 0.8, 0.1), vec3f(0.1, 0.35, 0.85), seg == 1u), vec3f(0.9, 0.15, 0.1), seg == 0u);
    albedo = mix(albedo, vec3f(0.95), step(0.45, abs(p.y)));
    specular = 0.1;
  }

  var color = shadeDiffuse(albedo, N, in.worldPos);
  let V = normalize(F.camPos - in.worldPos);
  let H = normalize(V + F.sunDir);
  let above = step(0.0, in.worldPos.y);
  color += F.sunColor * specular * pow(max(dot(N, H), 0.0), 64.0) * max(dot(N, F.sunDir), 0.0) * above;
  if (in.worldPos.y > 0.0) { color = applyFog(color, in.worldPos, O.terrain.w); }

  var out: FSOut;
  out.color = vec4f(color, 1.0);
  out.distance = length(in.worldPos - F.camPos);
  return out;
}
