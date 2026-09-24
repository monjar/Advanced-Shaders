// Blackbody colour table: Planck's law integrated against the CIE 1931
// colour-matching functions, converted to linear sRGB.
//
// The disk and the stars are treated as blackbodies, so a redshift factor g
// changes an emitter at temperature T into one at g·T exactly
// (I_ν/ν³ is invariant, and B_ν(T) transforms into B_ν(gT)). The shader looks
// up g·T in this table, which gives both the colour shift and the brightness
// change the eye actually sees (not the bolometric g⁴).

export const BB_SIZE = 256;
export const BB_T_MIN = 800;
export const BB_T_MAX = 60000;
/** Temperature rendered as neutral white with luminance 1. */
export const BB_T_WHITE = 6500;

// Wyman, Sloan and Shirley, "Simple Analytic Approximations to the CIE XYZ
// Color Matching Functions", JCGT 2013 (multi-lobe piecewise Gaussian fit).
function lobe(x: number, mu: number, s1: number, s2: number) {
  const t = (x - mu) / (x < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
}
function cie(lambda: number): [number, number, number] {
  const x = 1.056 * lobe(lambda, 599.8, 37.9, 31.0) + 0.362 * lobe(lambda, 442.0, 16.0, 26.7) - 0.065 * lobe(lambda, 501.1, 20.4, 26.2);
  const y = 0.821 * lobe(lambda, 568.8, 46.9, 40.5) + 0.286 * lobe(lambda, 530.9, 16.3, 31.1);
  const z = 1.217 * lobe(lambda, 437.0, 11.8, 36.0) + 0.681 * lobe(lambda, 459.0, 26.0, 13.8);
  return [x, y, z];
}

/** Spectral radiance B_λ(T) up to a constant (λ in nm). */
function planck(lambdaNm: number, T: number) {
  const l = lambdaNm * 1e-9;
  const c2 = 1.438777e-2; // hc/k in m·K
  return 1 / (l ** 5 * (Math.exp(c2 / (l * T)) - 1));
}

function linearSrgb(T: number): [number, number, number] {
  let X = 0, Y = 0, Z = 0;
  for (let l = 360; l <= 830; l += 2) {
    const b = planck(l, T);
    const [x, y, z] = cie(l);
    X += b * x;
    Y += b * y;
    Z += b * z;
  }
  return [
    3.2406 * X - 1.5372 * Y - 0.4986 * Z,
    -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
    0.0557 * X - 0.204 * Y + 1.057 * Z,
  ];
}

/**
 * 256 entries, log-spaced in temperature. rgb: colour with luminance 1
 * (white-balanced so BB_T_WHITE is neutral), a: log2 of the luminance
 * relative to BB_T_WHITE.
 */
export function blackbodyTable(): Float32Array<ArrayBuffer> {
  const white = linearSrgb(BB_T_WHITE);
  const lum = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  const out = new Float32Array(BB_SIZE * 4);
  for (let i = 0; i < BB_SIZE; i++) {
    const T = BB_T_MIN * Math.pow(BB_T_MAX / BB_T_MIN, i / (BB_SIZE - 1));
    const raw = linearSrgb(T);
    // von Kries-style channel scaling to the white point, clamp out-of-gamut reds.
    const c = raw.map((v, k) => Math.max(0, v / white[k]));
    const y = Math.max(lum(c), 1e-30);
    out.set([c[0] / y, c[1] / y, c[2] / y, Math.log2(y)], i * 4);
  }
  return out;
}

/** log2 luminance (relative to BB_T_WHITE) at temperature T, interpolated from the table. */
export function blackbodyLog2Luminance(table: Float32Array, T: number): number {
  const x = Math.min(Math.max((Math.log2(T / BB_T_MIN) / Math.log2(BB_T_MAX / BB_T_MIN)) * (BB_SIZE - 1), 0), BB_SIZE - 1.001);
  const i = Math.floor(x);
  return table[i * 4 + 3] * (i + 1 - x) + table[(i + 1) * 4 + 3] * (x - i);
}
