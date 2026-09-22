// GPU rigid-body buoyancy. Each floating body samples the same displaced ocean
// surface used for rendering (inverting the horizontal displacement with a few
// fixed-point iterations), then integrates a damped buoyancy spring and aligns
// with the local surface normal. The resulting model matrix is read directly by
// the object vertex shader and by the ripple solver. No CPU readback.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> O: Ocean;

@group(1) @binding(0) var dispTex: texture_2d_array<f32>;
@group(1) @binding(1) var derivTex: texture_2d_array<f32>;
@group(1) @binding(2) var linSampler: sampler;
@group(1) @binding(3) var clampSampler: sampler;
@group(1) @binding(4) var terrainTex: texture_2d<f32>;
@group(1) @binding(5) var rippleTex: texture_2d<f32>;
@group(1) @binding(6) var<storage, read_write> bodies: array<Body>;
@group(1) @binding(7) var<storage, read> anchors: array<Anchor>;

struct Body {
  model: mat4x4f,
  pos: vec4f,
  vel: vec4f,
  rot: vec4f,
  info: vec4f,
};

struct Anchor {
  a: vec4f, // goal x, goal z, scale, kind
  b: vec4f, // buoyancy ratio, half height, radius, drop generation
};

fn qmul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

fn qrotate(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn qfromTo(a: vec3f, b: vec3f) -> vec4f {
  let c = dot(a, b);
  if (c < -0.9999) {
    // Opposite vectors: rotate half a turn around any perpendicular axis.
    let axis = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(a.x) > 0.9);
    return vec4f(normalize(cross(a, axis)), 0.0);
  }
  return normalize(vec4f(cross(a, b), 1.0 + c));
}

fn qaxisAngle(axis: vec3f, angle: f32) -> vec4f {
  return vec4f(axis * sin(angle * 0.5), cos(angle * 0.5));
}

struct WaterSample {
  height: f32,
  normal: vec3f,
  drift: vec2f,
};

fn sampleWater(goal: vec2f) -> WaterSample {
  // Find the undisplaced point whose displaced position lands on `goal`.
  var p = goal;
  for (var i = 0; i < 4; i++) {
    let d = surfaceDisplacement(p, 0.1);
    p = goal - d.xz;
  }
  let d = surfaceDisplacement(p, 0.1);

  var deriv = vec4f(0.0);
  for (var c = 0u; c < CASCADES; c++) {
    let L = cascadeLength(c);
    // Bodies respond to waves larger than themselves: skip the finest detail.
    deriv += textureSampleLevel(derivTex, linSampler, p / L, c, 2.0);
  }
  let terrain = terrainSample(p);
  deriv *= depthAttenuation(max(-terrain.x, 0.0)) * O.sim.y;
  var slope = vec2f(deriv.x / max(1.0 + deriv.z, 0.2), deriv.y / max(1.0 + deriv.w, 0.2));
  slope += shoreWave(p, terrain).slope + rippleSlope(p) * 0.5;

  var w: WaterSample;
  w.height = d.y;
  w.normal = normalize(vec3f(-slope.x, 1.0, -slope.y));
  w.drift = d.xz;
  return w;
}

@compute @workgroup_size(16, 1, 1)
fn simulate(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&bodies)) { return; }
  var b = bodies[i];
  let a = anchors[i];
  let scale = a.a.z;
  let halfHeight = a.b.y * scale;
  let radius = a.b.z * scale;

  if (b.info.z != a.b.w) {
    // Dropped from the sky: reset position and give it a random tumble.
    b.pos = vec4f(a.a.x, 6.0 + f32(i) * 1.5, a.a.y, radius);
    b.vel = vec4f(0.0);
    b.rot = normalize(qaxisAngle(normalize(vec3f(0.3, 1.0, 0.7 + f32(i))), f32(i) * 1.7) + vec4f(0.2, 0.0, 0.1, 0.0));
    b.info.z = a.b.w;
  }

  let substeps = 4;
  let dt = min(F.dt, 1.0 / 20.0) / f32(substeps);
  for (var s = 0; s < substeps; s++) {
    let water = sampleWater(b.pos.xz);
    let bottom = b.pos.y - halfHeight;
    let submerged = clamp((water.height - bottom) / (2.0 * halfHeight), 0.0, 1.0);

    // Vertical: gravity + buoyancy proportional to submerged volume + drag.
    let buoyancy = 9.81 * a.b.x * submerged;
    let drag = -b.vel.y * (0.3 + 3.0 * submerged);
    b.vel.y += (-9.81 + buoyancy + drag) * dt;

    // Horizontal: follow the orbital motion of the waves around the anchor.
    let targetXZ = vec2f(a.a.x, a.a.y) + water.drift * 0.6;
    let toTarget = targetXZ - b.pos.xz;
    let accel = toTarget * 3.0 * (0.2 + submerged) - b.vel.xz * (1.5 + 2.0 * submerged);
    b.vel = vec4f(b.vel.x + accel.x * dt, b.vel.y, b.vel.z + accel.y * dt, submerged);
    b.pos = vec4f(b.pos.xyz + b.vel.xyz * dt, radius);

    // Orientation: ease the body's up vector towards the water normal.
    let up = qrotate(b.rot, vec3f(0.0, 1.0, 0.0));
    let targetUp = normalize(mix(vec3f(0.0, 1.0, 0.0), water.normal, 0.85 * submerged + 0.15));
    let align = qfromTo(up, targetUp);
    let rate = clamp((0.5 + 5.0 * submerged) * dt, 0.0, 1.0);
    let partial = normalize(mix(vec4f(0.0, 0.0, 0.0, 1.0), align, rate));
    let yaw = qaxisAngle(vec3f(0.0, 1.0, 0.0), (0.05 + 0.02 * f32(i)) * dt * submerged);
    b.rot = normalize(qmul(yaw, qmul(partial, b.rot)));
  }

  let q = b.rot;
  let x = qrotate(q, vec3f(1.0, 0.0, 0.0)) * scale;
  let y = qrotate(q, vec3f(0.0, 1.0, 0.0)) * scale;
  let z = qrotate(q, vec3f(0.0, 0.0, 1.0)) * scale;
  b.model = mat4x4f(vec4f(x, 0.0), vec4f(y, 0.0), vec4f(z, 0.0), vec4f(b.pos.xyz, 1.0));
  b.info = vec4f(a.a.w, halfHeight, b.info.z, 0.0);
  bodies[i] = b;
}
