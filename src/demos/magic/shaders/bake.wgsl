// Field bakes. `bakeField` runs every frame (the field is animated),
// `bakeCracks` whenever the crack cell scale changes, `probe` every frame
// after `bakeField`.

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> M: Material;
@group(0) @binding(2) var outA: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var outB: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var crackOut: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> probe: Probe;
@group(0) @binding(6) var outG: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn bakeField(@builtin(global_invocation_id) id: vec3u) {
  let n = textureDimensions(outA);
  if (any(id >= n)) { return; }
  let p = ((vec3f(id) + 0.5) / vec3f(n) * 2.0 - 1.0) * F.fieldBox;
  let f = evalField(p, F.fieldTime, sampleShape(p));
  textureStore(outA, id, vec4f(f.v, f.C));
  textureStore(outB, id, vec4f(f.n, 0.0));
  textureStore(outG, id, vec4f(f.gradC, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn bakeCracks(@builtin(global_invocation_id) id: vec3u) {
  let n = textureDimensions(crackOut);
  if (any(id >= n)) { return; }
  let p = ((vec3f(id) + 0.5) / vec3f(n) * 2.0 - 1.0) * F.shapeBox;
  let c = crackField(p);
  textureStore(crackOut, id, vec4f(c.d1 / CRACK_RANGE, c.d2 / CRACK_RANGE, c.id, 0.0));
}

// The object as a light source: average the field's emission over a 6³
// lattice inside the shape (one workgroup, shared-memory reduction). The
// altar, pillars, fog and the reflection cube are lit by this, so the light
// the object casts pulses exactly with its interior energy.
var<workgroup> acc: array<vec4f, 216>;
var<workgroup> acc2: array<vec4f, 216>;

@compute @workgroup_size(216)
fn probeField(@builtin(local_invocation_index) li: u32) {
  let c = vec3f(f32(li % 6u), f32((li / 6u) % 6u), f32(li / 36u));
  let p = ((c + 0.5) / 6.0 * 2.0 - 1.0) * 0.95;
  let s = sampleShape(p);
  let inside = select(0.0, 1.0, s.x < 0.0);
  let f = sampleField(p);
  let cr = sampleCrack(p);
  let open = crackOpen(f.C, f.E);
  let crackGlow = M.crackColor.rgb * M.crack.z * open * exp(-cr.x / 0.03) * 0.15;
  acc[li] = vec4f((energyEmission(f.E, f.C) + crackGlow) * inside, inside);
  acc2[li] = vec4f(f.C * inside, f.E * inside, open * inside, 0.0);
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride >>= 1u) {
    if (li < stride && li + stride < 216u) {
      acc[li] += acc[li + stride];
      acc2[li] += acc2[li + stride];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let count = max(acc[0].w, 1.0);
    // Radiant intensity ~ mean emission × volume: scale to the object's size.
    let volume = acc[0].w / 216.0 * 8.0 * 0.95 * 0.95 * 0.95;
    probe.light = vec4f(acc[0].rgb / count * volume * 0.35 * M.light.x, acc2[0].x / count);
    probe.stats = vec4f(acc2[0].y / count, 0.0, acc2[0].z / count, 0.0);
  }
}
