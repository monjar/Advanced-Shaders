// Interactive ripples: a damped 2D wave equation on a local height field.
// Floating bodies inject disturbances from their vertical motion (splashes,
// bobbing rings) and horizontal motion (bow waves); land cells reflect.
// State texel: (height, previous height).

struct RippleParams {
  origin: vec2f,
  size: f32,
  alpha: f32,        // (c dt / dx)^2, kept below 0.5 for stability
  damping: f32,
  strength: f32,
  dt: f32,
  terrainHalf: f32,
};

struct Body {
  model: mat4x4f,
  pos: vec4f,   // xyz, horizontal radius
  vel: vec4f,   // xyz velocity, submerged fraction
  rot: vec4f,   // orientation quaternion
  info: vec4f,  // kind, half height, drop generation, unused
};

@group(0) @binding(0) var<uniform> R: RippleParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> bodies: array<Body>;
@group(0) @binding(4) var terrainTex: texture_2d<f32>;

fn load(p: vec2i, size: i32) -> f32 {
  return textureLoad(src, clamp(p, vec2i(0), vec2i(size - 1)), 0).x;
}

@compute @workgroup_size(8, 8, 1)
fn simulateRipples(@builtin(global_invocation_id) id: vec3u) {
  let size = i32(textureDimensions(src).x);
  let p = vec2i(id.xy);
  if (p.x >= size || p.y >= size) { return; }

  let state = textureLoad(src, p, 0);
  let h = state.x;
  let prev = state.y;
  let lap = load(p + vec2i(1, 0), size) + load(p - vec2i(1, 0), size)
          + load(p + vec2i(0, 1), size) + load(p - vec2i(0, 1), size) - 4.0 * h;
  var next = (2.0 * h - prev + R.alpha * lap) * R.damping;

  let world = R.origin + ((vec2f(p) + 0.5) / f32(size) - 0.5) * R.size;
  for (var i = 0u; i < arrayLength(&bodies); i++) {
    let b = bodies[i];
    // Keep the source at least a couple of cells wide so it stays resolvable.
    let r = max(b.pos.w, 1.5 * R.size / f32(size));
    let offset = world - b.pos.xz;
    let d = length(offset);
    if (d > r * 2.5) { continue; }
    // Only bodies crossing the water line disturb it.
    let wet = smoothstep(0.0, 0.1, b.vel.w) * smoothstep(1.0, 0.9, b.vel.w);
    // Water displaced by the hull goes to a ring around the water line:
    // sinking raises it, rising draws it down. The ring is ~5x the hull area.
    let ring = smoothstep(r * 0.8, r * 1.2, d) * smoothstep(r * 2.5, r * 1.6, d);
    let displaced = -b.vel.y * 0.2;
    // Horizontal motion piles water up ahead and leaves a trough behind.
    let push = dot(b.vel.xz, offset / max(d, 1e-3)) * 0.5;
    next += ring * (displaced + push) * R.dt * R.strength * wet;
  }

  // Land acts as a wall; a sponge layer absorbs waves at the domain border.
  let tuv = clamp(world / (2.0 * R.terrainHalf) + 0.5, vec2f(0.0), vec2f(0.999));
  let ground = textureLoad(terrainTex, vec2i(tuv * vec2f(textureDimensions(terrainTex))), 0).x;
  if (ground > -0.15) { next = 0.0; }
  let border = min(min(p.x, p.y), min(size - 1 - p.x, size - 1 - p.y));
  next *= smoothstep(0.0, 16.0, f32(border));

  textureStore(dst, p, vec4f(clamp(next, -1.5, 1.5), h, 0.0, 1.0));
}
