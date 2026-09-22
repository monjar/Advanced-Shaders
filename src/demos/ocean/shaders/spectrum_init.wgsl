// Builds the initial wave spectrum h0(k) for each cascade.
// JONSWAP energy spectrum, finite-depth dispersion, Hasselmann/Donelan-style
// directional spreading with a swell term, and short-wave suppression.
// Reference: Horvath, "Empirical directional wave spectra for computer graphics" (2015).

struct SpectrumParams {
  cascades: array<vec4f, 3>, // patch length, k cutoff low, k cutoff high, unused
  wind: vec4f,               // speed (m/s), direction (rad), fetch (m), spread blend
  shape: vec4f,              // swell, depth (m), short waves fade (m), peak enhancement (gamma)
  misc: vec4f,               // amplitude scale, gravity, seed, unused
};

@group(0) @binding(0) var<uniform> P: SpectrumParams;
@group(0) @binding(1) var h0Out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(2) var waveDataOut: texture_storage_2d_array<rgba32float, write>;

const N: u32 = 256u;

fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(seed: u32) -> f32 {
  return (f32(pcg(seed)) + 0.5) / 4294967296.0;
}

// Two independent standard normal samples (Box-Muller).
fn gaussian2(id: vec3u) -> vec2f {
  let seed = id.x * 1973u + id.y * 9277u + id.z * 26699u + u32(P.misc.z) * 83492791u;
  let u1 = rand01(seed);
  let u2 = rand01(pcg(seed) ^ 0x9e3779b9u);
  let r = sqrt(-2.0 * log(u1));
  return vec2f(r * cos(TAU * u2), r * sin(TAU * u2));
}

fn dispersion(k: f32, g: f32, depth: f32) -> f32 {
  return sqrt(g * k * tanh(min(k * depth, 20.0)));
}

fn dispersionDerivative(k: f32, g: f32, depth: f32) -> f32 {
  let th = tanh(min(k * depth, 20.0));
  let ch = cosh(min(k * depth, 20.0));
  return g * (depth * k / ch / ch + th) / dispersion(k, g, depth) / 2.0;
}

fn jonswap(omega: f32, g: f32) -> f32 {
  let U = P.wind.x;
  let F = P.wind.z;
  let alpha = 0.076 * pow(g * F / (U * U), -0.22);
  let peakOmega = 22.0 * pow(U * F / (g * g), -0.33);
  let sigma = select(0.09, 0.07, omega <= peakOmega);
  let r = exp(-(omega - peakOmega) * (omega - peakOmega) / (2.0 * sigma * sigma * peakOmega * peakOmega));
  let invOmega = 1.0 / omega;
  let peakRatio = peakOmega / omega;
  return alpha * g * g * pow(invOmega, 5.0) * exp(-1.25 * pow(peakRatio, 4.0)) * pow(abs(P.shape.w), r);
}

fn spreadPower(omega: f32, peakOmega: f32) -> f32 {
  if (omega > peakOmega) {
    return 9.77 * pow(abs(omega / peakOmega), -2.5);
  }
  return 6.97 * pow(abs(omega / peakOmega), 5.0);
}

fn cosine2sNormalisation(s: f32) -> f32 {
  let s2 = s * s;
  let s3 = s2 * s;
  let s4 = s3 * s;
  if (s < 5.0) {
    return -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163;
  }
  return -4.80e-08 * s4 + 1.07e-05 * s3 - 9.53e-04 * s2 + 5.90e-02 * s + 3.93e-01;
}

fn directionalSpreading(theta: f32, omega: f32, g: f32) -> f32 {
  let U = P.wind.x;
  let F = P.wind.z;
  let peakOmega = 22.0 * pow(U * F / (g * g), -0.33);
  let swell = P.shape.x;
  let s = spreadPower(omega, peakOmega) + 16.0 * tanh(min(omega / peakOmega, 20.0)) * swell * swell;
  // Wrap the angle between the wave vector and the wind to [-pi, pi].
  var dTheta = theta - P.wind.y;
  dTheta = dTheta - TAU * floor((dTheta + PI) / TAU);
  let cos2s = cosine2sNormalisation(s) * pow(abs(cos(0.5 * dTheta)), 2.0 * s);
  let cosSq = 2.0 / PI * cos(dTheta) * cos(dTheta);
  return mix(cosSq, cos2s, P.wind.w);
}

@compute @workgroup_size(8, 8, 1)
fn initSpectrum(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= N || id.y >= N || id.z >= CASCADES) { return; }
  let cascade = P.cascades[id.z];
  let L = cascade.x;
  let g = P.misc.y;
  let depth = P.shape.y;
  let dk = TAU / L;
  let n = vec2f(id.xy) - vec2f(f32(N) * 0.5);
  let k = n * dk;
  let kLen = length(k);

  var h0 = vec2f(0.0);
  var omega = 0.0;
  if (kLen > 1e-6) {
    omega = dispersion(kLen, g, depth);
  }
  if (kLen >= cascade.y && kLen <= cascade.z && kLen > 1e-6) {
    let theta = atan2(k.y, k.x);
    let dOmegadk = dispersionDerivative(kLen, g, depth);
    let fade = exp(-P.shape.z * P.shape.z * kLen * kLen);
    let spectrum = jonswap(omega, g) * directionalSpreading(theta, omega, g) * fade * P.misc.x;
    h0 = gaussian2(id) * sqrt(2.0 * spectrum * abs(dOmegadk) / kLen * dk * dk);
  }
  textureStore(h0Out, id.xy, id.z, vec4f(h0, 0.0, 0.0));
  textureStore(waveDataOut, id.xy, id.z, vec4f(k.x, 1.0 / max(kLen, 1e-4), k.y, omega));
}
