// Physically based atmosphere (Hillaire 2020) as a reusable module.
//
// Usage:
//   const atmo = new Atmosphere(device);
//   atmo.params = { ...EARTH_ATMOSPHERE };            // or edit fields, then atmo.invalidate()
//   // every frame, before the passes that use it:
//   atmo.update(encoder, { cameraKm, altitudeKm, sunDir, invViewProj });
//   // in WGSL: prepend atmosphereWgsl(group) and call atmoSkyRadiance(),
//   // atmoSunDisk(), atmoAerialPerspective(), atmoSunTransmittanceAt(), ...
//   // in TS:   atmo.bindGroup(pipeline, group)
//
// The transmittance, multiple-scattering and irradiance LUTs only depend on
// the medium and are rebuilt when `params` change; the sky-view LUT and the
// aerial-perspective froxels depend on the camera and are rebuilt every frame.

import { bindGroup, createShader, uniformBuffer } from '../../core/gpu';
import type { Mat4, Vec3 } from '../../core/math';
import commonWgsl from './common.wgsl?raw';
import generateWgsl from './generate.wgsl?raw';
import lookupWgsl from './lookup.wgsl?raw';
import lutWgsl from './lut.wgsl?raw';

export interface AtmosphereParams {
  /** Planet radius (km). */
  bottomRadius: number;
  /** Radius of the top of the atmosphere (km). */
  topRadius: number;
  /** Rayleigh scattering at the ground (1/km). */
  rayleighScattering: Vec3;
  rayleighScaleHeight: number;
  mieScattering: Vec3;
  mieAbsorption: Vec3;
  mieScaleHeight: number;
  /** Cornette-Shanks asymmetry, per channel. */
  mieG: Vec3;
  /** Absorption at the peak of the tent profile (1/km). */
  ozoneAbsorption: Vec3;
  ozoneCenter: number;
  ozoneWidth: number;
  groundAlbedo: Vec3;
  multiScatteringFactor: number;
  /** Radians. */
  sunAngularRadius: number;
  sunIlluminance: Vec3;
  limbDarkening: boolean;
}

/**
 * Earth, with the values of Hillaire 2020 / Bruneton 2017: Rayleigh
 * (5.802, 13.558, 33.1)e-6 /m with an 8 km scale height, Mie scattering
 * 3.996e-6 /m and extinction 4.44e-6 /m (absorption 0.444e-6) with 1.2 km,
 * ozone (0.650, 1.881, 0.085)e-6 /m in a 30 km tent centred at 25 km.
 */
export const EARTH_ATMOSPHERE: AtmosphereParams = {
  bottomRadius: 6360,
  topRadius: 6460,
  rayleighScattering: [5.802e-3, 13.558e-3, 33.1e-3],
  rayleighScaleHeight: 8,
  mieScattering: [3.996e-3, 3.996e-3, 3.996e-3],
  mieAbsorption: [0.444e-3, 0.444e-3, 0.444e-3],
  mieScaleHeight: 1.2,
  mieG: [0.8, 0.8, 0.8],
  ozoneAbsorption: [0.65e-3, 1.881e-3, 0.085e-3],
  ozoneCenter: 25,
  ozoneWidth: 30,
  groundAlbedo: [0.3, 0.3, 0.3],
  multiScatteringFactor: 1,
  sunAngularRadius: 0.004675,
  sunIlluminance: [1, 1, 1],
  limbDarkening: true,
};

/**
 * A Mars-like atmosphere: thin CO2 (Rayleigh ~3 % of Earth's) under a dust
 * haze of vertical optical depth ~0.5 with an 11 km scale height. Dust
 * extinction is nearly grey (large particles) but its single-scattering
 * albedo drops in blue (butterscotch sky away from the sun), and blue is
 * scattered further forward than red (larger g), which leaves a blue glow
 * around the setting sun.
 */
export const MARS_ATMOSPHERE: AtmosphereParams = {
  bottomRadius: 3389.5,
  topRadius: 3489.5,
  rayleighScattering: [0.19e-3, 0.45e-3, 1.1e-3],
  rayleighScaleHeight: 11,
  mieScattering: [44e-3, 38e-3, 29e-3],
  mieAbsorption: [2e-3, 6e-3, 13e-3],
  mieScaleHeight: 11,
  mieG: [0.63, 0.72, 0.84],
  ozoneAbsorption: [0, 0, 0],
  ozoneCenter: 25,
  ozoneWidth: 30,
  groundAlbedo: [0.35, 0.2, 0.12],
  multiScatteringFactor: 1,
  sunAngularRadius: 0.00305,
  sunIlluminance: [0.43, 0.43, 0.43],
  limbDarkening: true,
};

export interface AtmosphereView {
  /** Camera position relative to the planet centre (km). */
  cameraKm: Vec3;
  /** Camera altitude above bottomRadius (km), computed in double precision. */
  altitudeKm: number;
  /** Unit direction towards the sun. */
  sunDir: Vec3;
  /** Inverse of the camera-relative view-projection (view without translation). */
  invViewProj: Mat4;
  /** Depth of the aerial-perspective volume (km). Default: grows with altitude. */
  apMaxDistanceKm?: number;
  /** Above this altitude (km) sky and aerial perspective are ray marched per pixel. */
  rayMarchAltitudeKm?: number;
}

export type AtmosphereResource =
  | 'uniforms' | 'sampler' | 'transmittance' | 'multiScattering' | 'irradiance' | 'skyView' | 'aerialInscatter' | 'aerialTransmittance';

const RESOURCE_ORDER: AtmosphereResource[] = [
  'uniforms', 'sampler', 'transmittance', 'multiScattering', 'irradiance', 'skyView', 'aerialInscatter', 'aerialTransmittance',
];

/**
 * WGSL for a shader that consumes the atmosphere: bindings in `group`
 * (0 uniforms, 1 sampler, 2-7 LUTs, see RESOURCE_ORDER), the shared
 * functions and the lookup API.
 */
export function atmosphereWgsl(group: number): string {
  const g = `@group(${group})`;
  const bindings = `
${g} @binding(0) var<uniform> ATMO: AtmosphereUniforms;
${g} @binding(1) var atmoSampler: sampler;
${g} @binding(2) var atmoTransmittanceLut: texture_2d<f32>;
${g} @binding(3) var atmoMultiScatLut: texture_2d<f32>;
${g} @binding(4) var atmoIrradianceLut: texture_2d<f32>;
${g} @binding(5) var atmoSkyViewLut: texture_2d<f32>;
${g} @binding(6) var atmoApInscatter: texture_3d<f32>;
${g} @binding(7) var atmoApTransmittance: texture_3d<f32>;
`;
  return [commonWgsl, bindings, lutWgsl, lookupWgsl].join('\n');
}

export const TRANSMITTANCE_SIZE: [number, number] = [256, 64];
export const MULTISCAT_SIZE = 32;
export const IRRADIANCE_SIZE: [number, number] = [64, 16];

export interface AtmosphereOptions {
  skyViewSize?: [number, number];
  aerialSize?: [number, number, number];
}

export class Atmosphere {
  params: AtmosphereParams = structuredClone(EARTH_ATMOSPHERE);

  readonly skyViewSize: [number, number];
  readonly aerialSize: [number, number, number];
  readonly uniforms: GPUBuffer;
  readonly sampler: GPUSampler;
  readonly views: Record<Exclude<AtmosphereResource, 'uniforms' | 'sampler'>, GPUTextureView>;
  /** Number of times the parameter-dependent LUTs were rebuilt. */
  rebuilds = 0;
  /** apMaxDistance used by the last update (km). */
  apMaxDistance = 0;

  private device: GPUDevice;
  private textures: GPUTexture[] = [];
  private data = new Float32Array(60);
  private lastParams = '';
  private pipes: Record<'transmittance' | 'multiScattering' | 'irradiance' | 'skyView' | 'aerial', GPUComputePipeline>;
  private groups: Record<keyof Atmosphere['pipes'], GPUBindGroup>;

  constructor(device: GPUDevice, options: AtmosphereOptions = {}) {
    this.device = device;
    this.skyViewSize = options.skyViewSize ?? [192, 108];
    this.aerialSize = options.aerialSize ?? [32, 32, 32];
    this.uniforms = uniformBuffer(device, this.data.byteLength, 'atmosphere');
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });

    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, size: number[], dimension: GPUTextureDimension = '2d') => {
      const t = device.createTexture({ label, size, dimension, format: 'rgba16float', usage });
      this.textures.push(t);
      return t.createView();
    };
    this.views = {
      transmittance: tex('atmosphere transmittance', TRANSMITTANCE_SIZE),
      multiScattering: tex('atmosphere multiple scattering', [MULTISCAT_SIZE, MULTISCAT_SIZE]),
      irradiance: tex('atmosphere irradiance', IRRADIANCE_SIZE),
      skyView: tex('atmosphere sky view', this.skyViewSize),
      aerialInscatter: tex('aerial perspective in-scattering', this.aerialSize, '3d'),
      aerialTransmittance: tex('aerial perspective transmittance', this.aerialSize, '3d'),
    };

    const module = createShader(device, 'atmosphere LUTs', [commonWgsl, lutWgsl, generateWgsl].join('\n'));
    const pipe = (entryPoint: string) =>
      device.createComputePipeline({ label: `atmosphere ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
    this.pipes = {
      transmittance: pipe('transmittance'),
      multiScattering: pipe('multiScattering'),
      irradiance: pipe('irradiance'),
      skyView: pipe('skyView'),
      aerial: pipe('aerial'),
    };
    const v = this.views;
    const s = this.sampler;
    const u = this.uniforms;
    this.groups = {
      transmittance: bindGroup(device, this.pipes.transmittance, 0, [u, null, null, null, v.transmittance]),
      multiScattering: bindGroup(device, this.pipes.multiScattering, 0, [u, s, v.transmittance, null, v.multiScattering]),
      irradiance: bindGroup(device, this.pipes.irradiance, 0, [u, s, v.transmittance, v.multiScattering, v.irradiance]),
      skyView: bindGroup(device, this.pipes.skyView, 0, [u, s, v.transmittance, v.multiScattering, v.skyView]),
      aerial: bindGroup(device, this.pipes.aerial, 0, [u, s, v.transmittance, v.multiScattering, null, v.aerialInscatter, v.aerialTransmittance]),
    };
  }

  /** Forces the parameter-dependent LUTs to be rebuilt on the next update. */
  invalidate() {
    this.lastParams = '';
  }

  /**
   * A bind group for `pipeline` at `group` holding the atmosphere resources.
   * Auto layouts only keep the bindings the entry point statically uses, so
   * pass the subset when it does not use all of them.
   */
  bindGroup(pipeline: GPURenderPipeline | GPUComputePipeline, group: number, use: AtmosphereResource[] = RESOURCE_ORDER): GPUBindGroup {
    const res = (name: AtmosphereResource) => {
      if (!use.includes(name)) return null;
      if (name === 'uniforms') return this.uniforms;
      if (name === 'sampler') return this.sampler;
      return this.views[name];
    };
    return bindGroup(this.device, pipeline, group, RESOURCE_ORDER.map(res), 'atmosphere');
  }

  private writeUniforms(view: AtmosphereView) {
    const p = this.params;
    const d = this.data;
    const mieExt: Vec3 = [0, 1, 2].map((i) => p.mieScattering[i] + p.mieAbsorption[i]) as Vec3;
    const alt = view.altitudeKm;
    // Deep enough to reach the horizon and the far side of the atmosphere
    // along a grazing ray, but not wasted on empty space.
    const R = p.bottomRadius;
    const horizon = Math.sqrt(Math.max(0, alt) * (2 * R + Math.max(0, alt)));
    const thick = Math.sqrt(p.topRadius ** 2 - R ** 2);
    this.apMaxDistance = view.apMaxDistanceKm ?? Math.min(Math.max(80, horizon + 0.3 * thick), 2 * thick);
    d.set([p.bottomRadius, p.topRadius, -1 / p.rayleighScaleHeight, -1 / p.mieScaleHeight], 0);
    d.set([...p.rayleighScattering, p.ozoneCenter], 4);
    d.set([...p.mieScattering, p.ozoneWidth / 2], 8);
    d.set([...mieExt, p.multiScatteringFactor], 12);
    d.set([...p.ozoneAbsorption, p.sunAngularRadius], 16);
    d.set([...p.mieG, this.apMaxDistance], 20);
    d.set([...p.groundAlbedo, view.rayMarchAltitudeKm ?? 0.8 * (p.topRadius - p.bottomRadius)], 24);
    d.set([...p.sunIlluminance, this.aerialSize[2]], 28);
    d.set([...view.cameraKm, alt], 32);
    d.set([...view.sunDir, p.limbDarkening ? 1 : 0], 36);
    d.set(view.invViewProj, 40);
    d.set([...this.skyViewSize, 0, 0], 56);
    this.device.queue.writeBuffer(this.uniforms, 0, d);
  }

  /** Writes the uniforms and records the LUT passes. */
  update(encoder: GPUCommandEncoder, view: AtmosphereView) {
    this.writeUniforms(view);
    const key = JSON.stringify(this.params);
    const pass = encoder.beginComputePass({ label: 'atmosphere LUTs' });
    const run = (name: keyof Atmosphere['pipes'], x: number, y: number) => {
      pass.setPipeline(this.pipes[name]);
      pass.setBindGroup(0, this.groups[name]);
      pass.dispatchWorkgroups(x, y);
    };
    if (key !== this.lastParams) {
      run('transmittance', TRANSMITTANCE_SIZE[0] / 8, TRANSMITTANCE_SIZE[1] / 8);
      run('multiScattering', MULTISCAT_SIZE, MULTISCAT_SIZE);
      run('irradiance', IRRADIANCE_SIZE[0], IRRADIANCE_SIZE[1]);
      this.lastParams = key;
      this.rebuilds++;
    }
    run('skyView', Math.ceil(this.skyViewSize[0] / 8), Math.ceil(this.skyViewSize[1] / 8));
    run('aerial', Math.ceil(this.aerialSize[0] / 8), Math.ceil(this.aerialSize[1] / 8));
    pass.end();
  }

  destroy() {
    this.textures.forEach((t) => t.destroy());
    this.uniforms.destroy();
  }
}
