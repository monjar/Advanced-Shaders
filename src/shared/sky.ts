import { smoothstep, type Vec3 } from '../core/math';

/** CPU mirror of sky.wgsl, used to derive sun colour and ambient light once per frame. */
export interface SkySettings {
  sunDir: Vec3;
  sunIntensity: number;
  turbidity: number;
  rayleigh: number;
  mieG: number;
  boost: number;
}

const BETA_R: Vec3 = [5.8e-3, 13.5e-3, 33.1e-3];
const OZONE: Vec3 = [0.00975, 0.0282, 0.00128];

function ozoneTransmittance(cosZenith: number, i: number): number {
  const c = Math.max(cosZenith, 0);
  return Math.exp(-OZONE[i] / Math.sqrt(c * c + 0.0078));
}

function airmass(cosZenith: number): number {
  const c = Math.min(1, Math.max(0, cosZenith));
  const zenithDeg = (Math.acos(c) * 180) / Math.PI;
  return 1 / (c + 0.15 * Math.pow(Math.max(93.885 - zenithDeg, 1e-3), -1.253));
}

function extinction(s: SkySettings): Vec3 {
  const m = 0.004 * s.turbidity * 1.25;
  return [BETA_R[0] * s.rayleigh * 8.4 + m, BETA_R[1] * s.rayleigh * 8.4 + m, BETA_R[2] * s.rayleigh * 8.4 + m];
}

/** Top-of-atmosphere sun irradiance, fading out as the sun sets. */
export function sunTopIntensity(s: SkySettings): number {
  return s.sunIntensity * smoothstep(-0.06, 0.04, s.sunDir[1]);
}

export function sunColor(s: SkySettings): Vec3 {
  const ext = extinction(s);
  const m = airmass(s.sunDir[1]);
  const top = sunTopIntensity(s);
  const y = s.sunDir[1];
  return [0, 1, 2].map((i) => top * Math.exp(-ext[i] * m) * ozoneTransmittance(y, i)) as Vec3;
}

export function skyRadiance(dir: Vec3, s: SkySettings): Vec3 {
  const len = Math.hypot(dir[0], Math.max(dir[1], 0), dir[2]) || 1;
  const d: Vec3 = [dir[0] / len, Math.max(dir[1], 0) / len, dir[2] / len];
  const ext = extinction(s);
  const mv = airmass(d[1]);
  const ms = airmass(s.sunDir[1]);
  const mu = d[0] * s.sunDir[0] + d[1] * s.sunDir[1] + d[2] * s.sunDir[2];
  const g = s.mieG;
  const phaseR = 0.0596831 * (1 + mu * mu);
  const phaseM = (0.0795775 * (1 - g * g)) / Math.pow(Math.max(1 + g * g - 2 * g * mu, 1e-4), 1.5);
  const top = sunTopIntensity(s);
  const bM = 0.004 * s.turbidity;
  const out: Vec3 = [0, 0, 0];
  const floor: Vec3 = [0.002, 0.004, 0.008];
  for (let i = 0; i < 3; i++) {
    const bR = BETA_R[i] * s.rayleigh;
    const viewT = Math.exp(-ext[i] * mv);
    const sunT = Math.exp(-ext[i] * ms) * ozoneTransmittance(s.sunDir[1], i);
    const scatterT = Math.pow(sunT, 0.1 + 0.55 * (1 - d[1]) * (1 - d[1]));
    out[i] = (top * scatterT * (bR * phaseR + bM * phaseM)) / (bR + bM) * (1 - viewT) * s.boost + floor[i];
  }
  return out;
}

/** Cosine-weighted average of sky radiance over the upper hemisphere. */
export function ambientLight(s: SkySettings): Vec3 {
  const out: Vec3 = [0, 0, 0];
  let wsum = 0;
  const rings = 6;
  const segments = 12;
  for (let r = 0; r < rings; r++) {
    const theta = ((r + 0.5) / rings) * (Math.PI / 2); // from zenith
    const w = Math.cos(theta) * Math.sin(theta);
    for (let k = 0; k < segments; k++) {
      const phi = ((k + 0.5) / segments) * Math.PI * 2;
      const L = skyRadiance([Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)], s);
      out[0] += L[0] * w;
      out[1] += L[1] * w;
      out[2] += L[2] * w;
      wsum += w;
    }
  }
  return [out[0] / wsum, out[1] / wsum, out[2] / wsum];
}
