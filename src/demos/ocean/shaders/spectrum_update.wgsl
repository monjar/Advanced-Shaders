// Advances the spectrum to time t and builds the eight complex fields that the
// inverse FFT turns into displacement and its spatial derivatives:
//   layer 2c   : (Dx + i Dz, Dy + i dDz/dx)
//   layer 2c+1 : (dDy/dx + i dDy/dz, dDx/dx + i dDz/dz)
// Packing two real fields into one complex signal halves the FFT work.

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
@group(0) @binding(1) var h0Tex: texture_2d_array<f32>;
@group(0) @binding(2) var waveData: texture_2d_array<f32>;
@group(0) @binding(3) var fieldsOut: texture_storage_2d_array<rgba32float, write>;

fn cmul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8, 1)
fn updateSpectrum(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= 256u || id.y >= 256u || id.z >= CASCADES) { return; }
  let wave = textureLoad(waveData, id.xy, id.z, 0);
  let h0 = textureLoad(h0Tex, id.xy, id.z, 0);
  let phase = wave.w * S.time;
  let e = vec2f(cos(phase), sin(phase));
  // Waves travel along +k: h(k,t) = h0(k) e^{-iwt} + conj(h0(-k)) e^{iwt}
  let h = cmul(h0.xy, vec2f(e.x, -e.y)) + cmul(h0.zw, e);
  let ih = vec2f(-h.y, h.x);

  let kx = wave.x;
  let kz = wave.z;
  let invK = wave.y;

  let dx = ih * kx * invK;
  let dy = h;
  let dz = ih * kz * invK;
  let dx_dx = -h * kx * kx * invK;
  let dy_dx = ih * kx;
  let dz_dx = -h * kx * kz * invK;
  let dy_dz = ih * kz;
  let dz_dz = -h * kz * kz * invK;

  let a = vec4f(dx.x - dz.y, dx.y + dz.x, dy.x - dz_dx.y, dy.y + dz_dx.x);
  let b = vec4f(dy_dx.x - dy_dz.y, dy_dx.y + dy_dz.x, dx_dx.x - dz_dz.y, dx_dx.y + dz_dz.x);
  textureStore(fieldsOut, id.xy, id.z * 2u, a);
  textureStore(fieldsOut, id.xy, id.z * 2u + 1u, b);
}
