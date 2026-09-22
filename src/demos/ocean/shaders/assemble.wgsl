// Turns the inverse-FFT output into render-ready textures and evolves foam.
//   displacement: (Dx, Dy, Dz, foam)          rgba16float, one layer per cascade
//   derivatives:  (dDy/dx, dDy/dz, dDx/dx, dDz/dz)
// Foam is injected where the Jacobian of the horizontal displacement drops
// (the surface folds over itself at breaking crests) and decays over time.

struct SimParams {
  time: f32,
  dt: f32,
  choppiness: f32,
  foamDecay: f32,
  foamThreshold: f32,
  foamSharpness: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> S: SimParams;
@group(0) @binding(1) var fields: texture_2d_array<f32>;
@group(0) @binding(2) var prevDisplacement: texture_2d_array<f32>;
@group(0) @binding(3) var displacementOut: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var derivativesOut: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn assemble(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= 256u || id.y >= 256u || id.z >= CASCADES) { return; }
  // The spectrum is centred at N/2, which multiplies the spatial signal by (-1)^(x+y).
  let parity = select(-1.0, 1.0, ((id.x + id.y) & 1u) == 0u);
  let a = textureLoad(fields, id.xy, id.z * 2u, 0) * parity;
  let b = textureLoad(fields, id.xy, id.z * 2u + 1u, 0) * parity;

  let lambda = S.choppiness;
  let dx = a.x;
  let dz = a.y;
  let dy = a.z;
  let dz_dx = a.w;
  let dy_dx = b.x;
  let dy_dz = b.y;
  let dx_dx = b.z;
  let dz_dz = b.w;

  let jacobian = (1.0 + lambda * dx_dx) * (1.0 + lambda * dz_dz) - lambda * lambda * dz_dx * dz_dx;
  let coverage = clamp((S.foamThreshold - jacobian) * S.foamSharpness, 0.0, 1.0);
  let previous = textureLoad(prevDisplacement, id.xy, id.z, 0).w;
  let foam = clamp(max(previous * exp(-S.foamDecay * S.dt), coverage), 0.0, 1.0);

  textureStore(displacementOut, id.xy, id.z, vec4f(lambda * dx, dy, lambda * dz, foam));
  textureStore(derivativesOut, id.xy, id.z, vec4f(dy_dx, dy_dz, lambda * dx_dx, lambda * dz_dz));
}
