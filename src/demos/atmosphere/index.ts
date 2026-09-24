import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, uniformBuffer } from '../../core/gpu';
import { deg, vec3, type Vec3 } from '../../core/math';
import { Atmosphere, EARTH_ATMOSPHERE, MARS_ATMOSPHERE, atmosphereWgsl, type AtmosphereParams } from '../../shared/atmosphere';
import { PlanetCamera } from '../../shared/planet-camera';
import { DEFAULTS, PRESETS, buildGui, solarPosition, type AtmosphereDemoParams } from './params';

import commonWgsl from './shaders/common.wgsl?raw';
import compareWgsl from './shaders/compare.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import referenceWgsl from './shaders/reference.wgsl?raw';
import sceneLibWgsl from './shaders/scene_lib.wgsl?raw';
import sceneWgsl from './shaders/scene.wgsl?raw';
import terrainWgsl from './shaders/terrain.wgsl?raw';

const HEIGHT_SIZE = 1024;
const PATCH_HALF_KM = 200;
// The site (centre of the mountain patch) sits at latitude 0, longitude 0.
const SITE_UP: Vec3 = [0, 0, 1];
const SITE_NORTH: Vec3 = [0, 1, 0];
const SITE_EAST: Vec3 = [1, 0, 0];

const wgsl = (...parts: string[]) => parts.join('\n');

export interface CompareStats {
  sky: { meanRel: number; maxRel: number; mean8: number; over1: number; over5: number; count: number };
  ground: { meanRel: number; maxRel: number; mean8: number; over1: number; over5: number; count: number };
  all: { meanRel: number; mean8: number };
}

class AtmosphereDemo implements Demo {
  private device: GPUDevice;
  private orbit: DemoContext['camera'];
  params: AtmosphereDemoParams = structuredClone(DEFAULTS);
  cam: PlanetCamera;
  atmo: Atmosphere;

  private width = 1;
  private height = 1;
  private sizeDirty = true;
  private terrainReady = false;
  private heights: Float32Array | null = null;
  private refDirty = true;
  private refKey = '';
  private readPending = false;
  private statsCopyQueued = false;
  private heightCopyQueued = false;
  /** Latest LUT-vs-reference statistics (null until measured). */
  stats: CompareStats | null = null;
  /** Number of comparisons measured so far. */
  statsVersion = 0;

  private frameData = new Float32Array(32);
  private frameUBO: GPUBuffer;
  private statsBuffer: GPUBuffer;
  private statsRead: GPUBuffer;
  private heightRaw: GPUTexture;
  private heightTex: GPUTexture;
  private heightRead: GPUBuffer;
  private screen: GPUTexture[] = [];
  private linearClamp: GPUSampler;

  private terrainHeightsPipe: GPUComputePipeline;
  private terrainGradPipe: GPUComputePipeline;
  private scenePipe: GPUComputePipeline;
  private refPipe: GPUComputePipeline;
  private comparePipe: GPUComputePipeline;
  private compositePipe: GPURenderPipeline;
  private terrainBGs: GPUBindGroup[];
  private sceneBG!: GPUBindGroup;
  private refBG!: GPUBindGroup;
  private compareBG!: GPUBindGroup;
  private compositeBG!: GPUBindGroup;
  private atmoBGs: { scene: GPUBindGroup; ref: GPUBindGroup; composite: GPUBindGroup };

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.orbit = ctx.camera;
    this.atmo = new Atmosphere(device);
    this.cam = new PlanetCamera(EARTH_ATMOSPHERE.bottomRadius * 1000);
    this.cam.surfaceHeight = (dir) => this.surfaceHeight(dir);
    this.applyPreset('Morning, mountains');
    this.cam.bind(this.orbit);

    buildGui(ctx.gui, this.params, {
      preset: (name) => this.applyPreset(name),
      timeOfDay: () => this.applyTimeOfDay(),
      altitude: () => this.setAltitude(10 ** this.params.altitudeLog),
      reference: () => (this.refDirty = true),
    });

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'atmosphere frame');
    this.statsBuffer = device.createBuffer({ label: 'compare stats', size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.statsRead = device.createBuffer({ label: 'compare stats readback', size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.linearClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    const storage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    this.heightRaw = device.createTexture({ label: 'height raw', size: [HEIGHT_SIZE, HEIGHT_SIZE], format: 'r32float', usage: storage | GPUTextureUsage.COPY_SRC });
    this.heightTex = device.createTexture({ label: 'height + gradient', size: [HEIGHT_SIZE, HEIGHT_SIZE], format: 'rgba16float', usage: storage });
    this.heightRead = device.createBuffer({ label: 'height readback', size: HEIGHT_SIZE * HEIGHT_SIZE * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const compute = (label: string, code: string, entryPoint = 'main') =>
      device.createComputePipeline({ label, layout: 'auto', compute: { module: createShader(device, label, code), entryPoint } });
    const atmoLib = atmosphereWgsl(1);
    const terrainModule = createShader(device, 'terrain', wgsl(commonWgsl, terrainWgsl));
    this.terrainHeightsPipe = device.createComputePipeline({ label: 'terrain heights', layout: 'auto', compute: { module: terrainModule, entryPoint: 'heights' } });
    this.terrainGradPipe = device.createComputePipeline({ label: 'terrain gradients', layout: 'auto', compute: { module: terrainModule, entryPoint: 'gradients' } });
    this.scenePipe = compute('atmosphere scene', wgsl(commonWgsl, atmoLib, sceneLibWgsl, sceneWgsl));
    this.refPipe = compute('atmosphere reference', wgsl(commonWgsl, atmoLib, sceneLibWgsl, referenceWgsl));
    this.comparePipe = compute('atmosphere compare', wgsl(commonWgsl, compareWgsl));
    const compositeModule = createShader(device, 'atmosphere composite', wgsl(commonWgsl, atmoLib, compositeWgsl));
    this.compositePipe = device.createRenderPipeline({
      label: 'atmosphere composite',
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });

    this.terrainBGs = [
      bindGroup(device, this.terrainHeightsPipe, 0, [this.frameUBO, this.heightRaw.createView()]),
      bindGroup(device, this.terrainGradPipe, 0, [this.frameUBO, null, this.heightRaw.createView(), this.heightTex.createView()]),
    ];
    this.atmoBGs = {
      scene: this.atmo.bindGroup(this.scenePipe, 1),
      ref: this.atmo.bindGroup(this.refPipe, 1, ['uniforms', 'sampler', 'multiScattering', 'irradiance']),
      composite: this.atmo.bindGroup(this.compositePipe, 1),
    };
  }

  // ---- Camera and presets --------------------------------------------------

  private get planetRadiusKm() {
    return this.params.planet === 'Mars' ? MARS_ATMOSPHERE.bottomRadius : EARTH_ATMOSPHERE.bottomRadius;
  }

  /** Terrain height (m) under a unit direction, from the baked patch read back to the CPU. */
  private surfaceHeight(dir: Vec3): number {
    if (!this.heights) return 0;
    const up = vec3.dot(dir, SITE_UP);
    if (up <= 0) return 0;
    const R = this.planetRadiusKm;
    const x = (vec3.dot(dir, SITE_EAST) * R) / up;
    const y = (vec3.dot(dir, SITE_NORTH) * R) / up;
    const u = (x / (2 * PATCH_HALF_KM) + 0.5) * HEIGHT_SIZE - 0.5;
    const v = (0.5 - y / (2 * PATCH_HALF_KM)) * HEIGHT_SIZE - 0.5;
    if (u < 0 || v < 0 || u >= HEIGHT_SIZE - 1 || v >= HEIGHT_SIZE - 1) return 0;
    const i = Math.floor(u);
    const j = Math.floor(v);
    const fu = u - i;
    const fv = v - j;
    const h = this.heights;
    const at = (a: number, b: number) => h[b * HEIGHT_SIZE + a];
    const top = at(i, j) * (1 - fu) + at(i + 1, j) * fu;
    const bottom = at(i, j + 1) * (1 - fu) + at(i + 1, j + 1) * fu;
    return (top * (1 - fv) + bottom * fv) * this.params.terrainScale * 1000;
  }

  applyPreset(name: string) {
    const preset = PRESETS[name];
    if (!preset) return;
    Object.assign(this.params, structuredClone(DEFAULTS), structuredClone(preset.params));
    this.params.debugView = DEFAULTS.debugView;
    const c = preset.camera;
    this.cam.radius = this.planetRadiusKm * 1000;
    this.cam.setGeo(c.lat, c.lon, c.altitude, c.heading, c.pitch);
    this.params.altitudeLog = Math.log10(c.altitude);
    this.refDirty = true;
  }

  private applyTimeOfDay() {
    const s = solarPosition(this.params.timeOfDay, this.params.latitude);
    this.params.sunElevation = s.elevation;
    this.params.sunAzimuth = s.azimuth;
  }

  setAltitude(meters: number) {
    const up = vec3.normalize(this.cam.position);
    this.cam.position = vec3.scale(up, this.cam.radius + meters);
  }

  private sunDir(): Vec3 {
    const el = deg(this.params.sunElevation);
    const az = deg(this.params.sunAzimuth);
    const h = Math.cos(el);
    return vec3.normalize([
      SITE_UP[0] * Math.sin(el) + h * (SITE_NORTH[0] * Math.cos(az) + SITE_EAST[0] * Math.sin(az)),
      SITE_UP[1] * Math.sin(el) + h * (SITE_NORTH[1] * Math.cos(az) + SITE_EAST[1] * Math.sin(az)),
      SITE_UP[2] * Math.sin(el) + h * (SITE_NORTH[2] * Math.cos(az) + SITE_EAST[2] * Math.sin(az)),
    ]);
  }

  private atmosphereParams(): AtmosphereParams {
    const p = this.params;
    const base = p.planet === 'Mars' ? MARS_ATMOSPHERE : EARTH_ATMOSPHERE;
    const scale = (v: Vec3, s: number) => v.map((x) => x * s) as Vec3;
    // Mie g is set for green; the base's per-channel spread is kept.
    const gShift = p.mieG - base.mieG[1];
    return {
      ...base,
      rayleighScattering: scale(base.rayleighScattering, p.rayleighScale),
      rayleighScaleHeight: p.rayleighHeight,
      mieScattering: scale(base.mieScattering, p.mieScale),
      mieAbsorption: scale(base.mieAbsorption, p.mieAbsorptionScale),
      mieScaleHeight: p.mieHeight,
      mieG: base.mieG.map((g) => Math.min(0.99, Math.max(0, g + gShift))) as Vec3,
      ozoneAbsorption: scale(base.ozoneAbsorption, p.ozoneScale),
      // Mars keeps its rust tint; the slider scales it.
      groundAlbedo: scale(p.planet === 'Mars' ? [1.2, 0.68, 0.42] : [1, 1, 1], p.groundAlbedo),
      multiScatteringFactor: p.multiScattering,
      sunIlluminance: scale(base.sunIlluminance, p.sunIlluminance),
      sunAngularRadius: deg(p.sunDiskDeg),
      limbDarkening: p.limbDarkening,
    };
  }

  // ---- Resources ------------------------------------------------------------

  private rebuildTargets() {
    const d = this.device;
    this.screen.forEach((t) => t.destroy());
    const s = this.params.refScale;
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const hdr = d.createTexture({ label: 'hdr', size: [this.width, this.height], format: 'rgba16float', usage });
    const ref = d.createTexture({ label: 'reference', size: [Math.ceil(this.width / s), Math.ceil(this.height / s)], format: 'rgba16float', usage });
    this.screen = [hdr, ref];
    const heightView = this.heightTex.createView();
    this.sceneBG = bindGroup(d, this.scenePipe, 0, [this.frameUBO, hdr.createView(), heightView, this.linearClamp]);
    this.refBG = bindGroup(d, this.refPipe, 0, [this.frameUBO, ref.createView(), heightView, this.linearClamp]);
    this.compareBG = bindGroup(d, this.comparePipe, 0, [this.frameUBO, hdr.createView(), ref.createView(), this.statsBuffer]);
    this.compositeBG = bindGroup(d, this.compositePipe, 0, [this.frameUBO, hdr.createView(), ref.createView()]);
    this.sizeDirty = false;
    this.refDirty = true;
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.sizeDirty = true;
  }

  private writeUniforms(info: FrameInfo) {
    const p = this.params;
    const f = this.frameData;
    f.set([this.width, this.height, info.time, p.exposure], 0);
    f.set([p.debugView, p.refScale, p.refViewSteps, p.refSunSteps], 4);
    f.set([p.refMultiScattering ? 1 : 0, p.terrainScale, p.snowLine, p.stars], 8);
    f.set([...SITE_UP, PATCH_HALF_KM], 12);
    f.set([...SITE_EAST, p.shadows ? 1 : 0], 16);
    f.set([...SITE_NORTH, p.splitX], 20);
    f.set([p.planet === 'Mars' ? 1 : 0, 0, 0, 0], 24);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);
  }

  private readStats() {
    if (this.readPending) return;
    this.readPending = true;
    this.statsRead.mapAsync(GPUMapMode.READ).then(() => {
      const u = new Uint32Array(this.statsRead.getMappedRange().slice(0));
      this.statsRead.unmap();
      this.readPending = false;
      const part = (k: number) => {
        const n = Math.max(1, u[2 + k]);
        return { meanRel: u[k] / 1e4 / n, maxRel: u[6 + k] / 1e4, mean8: u[4 + k] / 100 / n, over1: u[8 + k] / n, over5: u[10 + k] / n, count: u[2 + k] };
      };
      const sky = part(0);
      const ground = part(1);
      const n = Math.max(1, sky.count + ground.count);
      this.stats = {
        sky, ground,
        all: { meanRel: (sky.meanRel * sky.count + ground.meanRel * ground.count) / n, mean8: (sky.mean8 * sky.count + ground.mean8 * ground.count) / n },
      };
      this.statsVersion++;
    }).catch(() => (this.readPending = false));
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    const R = this.planetRadiusKm * 1000;
    if (this.cam.radius !== R) {
      const alt = this.cam.altitude;
      this.cam.radius = R;
      this.setAltitude(alt);
    }
    if (this.sizeDirty || this.screen[1]?.width !== Math.ceil(this.width / p.refScale)) this.rebuildTargets();
    this.writeUniforms(info);
    // Readbacks are mapped one frame after the copy was recorded, once that
    // command buffer has been submitted.
    if (this.statsCopyQueued) {
      this.statsCopyQueued = false;
      this.readStats();
    }
    if (this.heightCopyQueued) {
      this.heightCopyQueued = false;
      this.heightRead.mapAsync(GPUMapMode.READ).then(() => {
        this.heights = new Float32Array(this.heightRead.getMappedRange().slice(0));
        this.heightRead.unmap();
      }).catch(() => {}); // destroyed before the map resolved
    }

    if (!this.terrainReady) {
      const tp = encoder.beginComputePass({ label: 'terrain bake' });
      tp.setPipeline(this.terrainHeightsPipe);
      tp.setBindGroup(0, this.terrainBGs[0]);
      tp.dispatchWorkgroups(HEIGHT_SIZE / 8, HEIGHT_SIZE / 8);
      tp.setPipeline(this.terrainGradPipe);
      tp.setBindGroup(0, this.terrainBGs[1]);
      tp.dispatchWorkgroups(HEIGHT_SIZE / 8, HEIGHT_SIZE / 8);
      tp.end();
      encoder.copyTextureToBuffer({ texture: this.heightRaw }, { buffer: this.heightRead, bytesPerRow: HEIGHT_SIZE * 4 }, [HEIGHT_SIZE, HEIGHT_SIZE]);
      this.terrainReady = true;
      // The CPU copy only serves the camera's ground clamp.
      this.heightCopyQueued = true;
    }

    this.cam.fovY = deg(p.fov);
    this.cam.update(this.orbit, this.width / this.height);
    if (Math.abs(Math.log10(Math.max(this.cam.altitude, 1)) - p.altitudeLog) > 1e-4) p.altitudeLog = Math.log10(Math.max(this.cam.altitude, 1));
    const sunDir = this.sunDir();
    this.atmo.params = this.atmosphereParams();
    this.atmo.update(encoder, {
      cameraKm: vec3.scale(this.cam.position, 1e-3),
      altitudeKm: this.cam.altitude * 1e-3,
      sunDir,
      invViewProj: this.cam.invViewProj,
      apMaxDistanceKm: p.apDistance > 0 ? p.apDistance : undefined,
      rayMarchAltitudeKm: p.rayMarchAltitude,
    });

    const groups = (n: number) => Math.ceil(n / 8);
    const cp = encoder.beginComputePass({ label: 'atmosphere scene' });
    cp.setPipeline(this.scenePipe);
    cp.setBindGroup(0, this.sceneBG);
    cp.setBindGroup(1, this.atmoBGs.scene);
    cp.dispatchWorkgroups(groups(this.width), groups(this.height));

    // The reference is expensive: recompute it only when the view changes.
    const refKey = JSON.stringify([this.cam.position, this.cam.heading, this.cam.pitch, sunDir, this.atmo.params, p.terrainScale, p.fov, p.refViewSteps, p.refSunSteps, p.refMultiScattering, p.shadows, p.snowLine, this.width, this.height]);
    const wantRef = p.reference || (p.debugView >= 1 && p.debugView <= 3);
    let compared = false;
    if (wantRef && (this.refDirty || refKey !== this.refKey) && !this.readPending && !this.statsCopyQueued) {
      const [rw, rh] = [this.screen[1].width, this.screen[1].height];
      cp.setPipeline(this.refPipe);
      cp.setBindGroup(0, this.refBG);
      cp.setBindGroup(1, this.atmoBGs.ref);
      cp.dispatchWorkgroups(groups(rw), groups(rh));
      this.refKey = refKey;
      this.refDirty = false;
      compared = true;
    }
    cp.end();
    if (compared) {
      const [rw, rh] = [this.screen[1].width, this.screen[1].height];
      encoder.clearBuffer(this.statsBuffer);
      const pass = encoder.beginComputePass({ label: 'compare' });
      pass.setPipeline(this.comparePipe);
      pass.setBindGroup(0, this.compareBG);
      pass.dispatchWorkgroups(groups(rw), groups(rh));
      pass.end();
      encoder.copyBufferToBuffer(this.statsBuffer, 0, this.statsRead, 0, 48);
      this.statsCopyQueued = true;
    }

    const rp = encoder.beginRenderPass({
      label: 'atmosphere composite',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.compositePipe);
    rp.setBindGroup(0, this.compositeBG);
    rp.setBindGroup(1, this.atmoBGs.composite);
    rp.draw(3);
    rp.end();
  }

  hud() {
    const c = this.cam;
    const alt = c.altitude;
    const altText = alt > 1e5 ? `${(alt / 1000).toFixed(0)} km` : alt > 2000 ? `${(alt / 1000).toFixed(2)} km` : `${alt.toFixed(0)} m`;
    let text = `altitude ${altText} · sun ${this.params.sunElevation.toFixed(1)}° · ${this.params.planet}` +
      `\nLUTs: transmittance 256×64, multi-scattering 32², sky-view ${this.atmo.skyViewSize.join('×')}, AP ${this.atmo.aerialSize.join('×')} over ${this.atmo.apMaxDistance.toFixed(0)} km` +
      ` · LUT rebuilds ${this.atmo.rebuilds}` +
      `\n${alt / 1000 >= this.params.rayMarchAltitude ? 'sky: per-pixel ray march' : 'sky: sky-view LUT, ground: froxels'} · drag: look · wheel: altitude · WASD/QE: fly`;
    const s = this.stats;
    if (s && (this.params.reference || (this.params.debugView >= 1 && this.params.debugView <= 3))) {
      const pc = (x: number) => `${(x * 100).toFixed(2)}%`;
      text += `\nLUT vs reference: sky ${pc(s.sky.meanRel)} mean / ${pc(s.sky.maxRel)} max (${s.sky.mean8.toFixed(2)}/255)` +
        ` · ground ${pc(s.ground.meanRel)} / ${pc(s.ground.maxRel)} (${s.ground.mean8.toFixed(2)}/255)`;
    }
    return text;
  }

  destroy() {
    this.atmo.destroy();
    [...this.screen, this.heightRaw, this.heightTex].forEach((t) => t.destroy());
    [this.frameUBO, this.statsBuffer, this.statsRead, this.heightRead].forEach((b) => b.destroy());
  }
}

export const atmosphereEntry: DemoEntry = {
  id: 'atmosphere',
  title: 'Atmospheric scattering',
  tags: ['Hillaire 2020', 'LUTs', 'multiple scattering', 'aerial perspective'],
  info: `
    <h2>Atmospheric scattering</h2>
    <ul>
      <li>Rayleigh, Mie (Cornette-Shanks) and ozone, planet-scale geometry</li>
      <li>Transmittance LUT (Bruneton parameterisation)</li>
      <li>Hillaire's multiple-scattering LUT, with ground bounce</li>
      <li>Sky-view LUT per frame, per-pixel ray march from orbit</li>
      <li>Aerial-perspective froxels over the camera frustum</li>
      <li>Sun disk with limb darkening, Earth's shadow, twilight</li>
      <li>Brute-force reference and a live error measurement</li>
    </ul>
    <p><kbd>drag</kbd> look · <kbd>wheel</kbd> altitude<br>
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up · <kbd>shift</kbd> faster</p>`,
  create: (ctx) => new AtmosphereDemo(ctx),
};
