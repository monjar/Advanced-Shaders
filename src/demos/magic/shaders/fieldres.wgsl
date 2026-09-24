// Baked field resources (group 1 in every pass that reads the field) and
// their lookups. All coordinates are object space.

@group(1) @binding(0) var fieldA: texture_3d<f32>;   // v.xyz, C
@group(1) @binding(1) var fieldB: texture_3d<f32>;   // advected noise values n.xyz
@group(1) @binding(2) var shapeTex: texture_3d<f32>; // signed distance, outward normal
@group(1) @binding(3) var crackTex: texture_3d<f32>; // d1, d2 (÷ CRACK_RANGE), cell id
@group(1) @binding(4) var fieldSampler: sampler;
@group(1) @binding(5) var fieldG: texture_3d<f32>;   // ∇C

// Crack distances are stored in rgba8unorm; only the neighbourhood of a
// crack needs precision, so distances saturate at this range.
const CRACK_RANGE: f32 = 0.12;

fn sampleShape(p: vec3f) -> vec4f {
  let uv = p / (2.0 * F.shapeBox) + 0.5;
  let s = textureSampleLevel(shapeTex, fieldSampler, uv, 0.0);
  let outside = length(max(abs(p) - F.shapeBox, vec3f(0.0)));
  if (outside > 0.0) {
    return vec4f(max(s.x, 0.0) + outside, normalize(p));
  }
  return vec4f(s.x, normalize(s.yzw + vec3f(0.0, 1e-6, 0.0)));
}

fn sampleField(p: vec3f) -> Field {
  let uv = p / (2.0 * F.fieldBox) + 0.5;
  let a = textureSampleLevel(fieldA, fieldSampler, uv, 0.0);
  let b = textureSampleLevel(fieldB, fieldSampler, uv, 0.0);
  // Outside the baked volume the field fades to still air.
  let fade = 1.0 - smoothstep(0.9, 1.0, max(max(abs(p.x), abs(p.y)), abs(p.z)) / F.fieldBox);
  var f: Field;
  f.v = a.xyz * fade;
  f.C = a.w * fade;
  f.n = b.xyz;
  f.E = energyFromNoise(b.xyz) * fade;
  f.gradC = vec3f(0.0);
  return f;
}

// ∇C lives in its own texture; only the interior march needs it.
fn sampleGradC(p: vec3f) -> vec3f {
  return textureSampleLevel(fieldG, fieldSampler, p / (2.0 * F.fieldBox) + 0.5, 0.0).xyz;
}

fn sampleCrack(p: vec3f) -> vec3f {
  let uv = p / (2.0 * F.shapeBox) + 0.5;
  let c = textureSampleLevel(crackTex, fieldSampler, uv, 0.0);
  return vec3f(c.xy * CRACK_RANGE, c.z);
}
