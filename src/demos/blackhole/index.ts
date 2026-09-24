import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, uniformBuffer } from '../../core/gpu';
import { deg, vec3, type Vec3 } from '../../core/math';
import { blackbodyLog2Luminance, blackbodyTable } from './blackbody';
import { DEFAULTS, applyPreset, buildGui, type BlackHoleParams } from './params';

import accumulateWgsl from './shaders/accumulate.wgsl?raw';
import bloomWgsl from './shaders/bloom.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import diskWgsl from './shaders/disk.wgsl?raw';
import envWgsl from './shaders/env.wgsl?raw';
import geodesicWgsl from './shaders/geodesic.wgsl?raw';
import traceWgsl from './shaders/trace.wgsl?raw';

const wgsl = (...parts: string[]) => parts.join('\n');

/** Critical impact parameter (shadow radius seen from infinity): 3√3/2 r_s. */
export const B_CRIT = 1.5 * Math.sqrt(3);
const BLOOM_LEVELS = 6;
const R_MIN_OBS = 1.05;
// Galaxy: bulge off to one side, the band passing behind the hole in the
// default views so its lensed images wrap around the shadow.
const GALAXY_C = vec3.normalize([-0.8, 0.3, -0.5]);
const GALAXY_N = vec3.normalize(vec3.cross(GALAXY_C, vec3.normalize([0.25, -0.2, -0.95])));
const STATS = 16;
const ST = { escaped: 0, captured: 1, limit: 2, absorbed: 3, steps: 4, maxSteps: 5, nonFinite: 6, order1: 7, order2: 8, order3: 9, accCaptured: 10, accFrames: 11, extraSteps: 12, nodes2: 13, nodes3: 14 };

interface Stats {
  values: Uint32Array;
  pixels: number;
  /** Image-plane area of one pixel (tan units²) when the stats were taken. */
  pixelArea: number;
  tanHalfHeight: number;
  rObs: number;
  centred: boolean;
  physical: boolean;
  diskOn: boolean;
  staticObserver: boolean;
}

class BlackHoleDemo implements Demo {
  private device: GPUDevice;
  private gui: DemoContext['gui'];
  private camera: DemoContext['camera'];
  private params: BlackHoleParams = structuredClone(DEFAULTS);

  private width = 1;
  private height = 1;
  private sizeDirty = true;
  private frameIndex = 0;
  private diskTime = 0;
  private current = 0;
  private accumulated = 0;
  private signature = '';

  private frameData = new Float32Array(84);
  private bbTable: Float32Array;
  private frameUBO: GPUBuffer;
  private bbBuffer: GPUBuffer;
  private statsBuffer: GPUBuffer;
  private readback: GPUBuffer;
  private readPending = false;
  private stats: Stats | null = null;

  private tracePipe: GPUComputePipeline;
  private accumulatePipe: GPUComputePipeline;
  private downPipe: GPUComputePipeline;
  private upPipe: GPUComputePipeline;
  private compositePipe: GPURenderPipeline;
  private linearClamp: GPUSampler;

  private textures: GPUTexture[] = [];
  private levelUBOs: GPUBuffer[] = [];
  private bloomLevels = 0;
  private bloomSizes: [number, number][] = [];
  private traceBG0: GPUBindGroup;
  private traceBG1!: GPUBindGroup;
  private accumulateBG0: GPUBindGroup;
  private accumulateBG1!: GPUBindGroup[];
  private downBG0: GPUBindGroup;
  private upBG0: GPUBindGroup;
  private downBGs!: GPUBindGroup[][];
  private upBGs!: GPUBindGroup[];
  private compositeBG0: GPUBindGroup;
  private compositeBG1!: GPUBindGroup[];

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.camera = ctx.camera;
    this.gui = ctx.gui;
    Object.assign(this.camera, { minDistance: 1.6, minPitch: -1.55, maxPitch: 1.55, moveSpeed: 4 });
    this.camera.near = 0.01;
    applyPreset(this.params, this.camera, 'Inclined');
    buildGui(ctx.gui, this.params, this.camera);

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'blackhole frame');
    const table = (this.bbTable = blackbodyTable());
    this.bbBuffer = device.createBuffer({ label: 'blackbody table', size: table.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.bbBuffer, 0, table);
    this.statsBuffer = device.createBuffer({ label: 'blackhole stats', size: STATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.readback = device.createBuffer({ label: 'blackhole stats readback', size: STATS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.linearClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    const compute = (label: string, code: string, entryPoint: string) =>
      device.createComputePipeline({ label, layout: 'auto', compute: { module: createShader(device, label, code), entryPoint } });
    this.tracePipe = compute('geodesic trace', wgsl(commonWgsl, envWgsl, diskWgsl, geodesicWgsl, traceWgsl), 'main');
    this.accumulatePipe = compute('accumulate', wgsl(commonWgsl, accumulateWgsl), 'accumulate');
    const bloomModule = createShader(device, 'bloom', wgsl(commonWgsl, bloomWgsl));
    this.downPipe = device.createComputePipeline({ label: 'bloom down', layout: 'auto', compute: { module: bloomModule, entryPoint: 'downsample' } });
    this.upPipe = device.createComputePipeline({ label: 'bloom up', layout: 'auto', compute: { module: bloomModule, entryPoint: 'upsample' } });
    const compositeModule = createShader(device, 'blackhole composite', wgsl(commonWgsl, compositeWgsl));
    this.compositePipe = device.createRenderPipeline({
      label: 'blackhole composite',
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });

    this.traceBG0 = bindGroup(device, this.tracePipe, 0, [this.frameUBO, this.bbBuffer, this.statsBuffer]);
    this.accumulateBG0 = bindGroup(device, this.accumulatePipe, 0, [this.frameUBO]);
    this.downBG0 = bindGroup(device, this.downPipe, 0, []);
    this.upBG0 = bindGroup(device, this.upPipe, 0, []);
    this.compositeBG0 = bindGroup(device, this.compositePipe, 0, [this.frameUBO]);
  }

  private rebuildTargets() {
    const d = this.device;
    this.textures.forEach((t) => t.destroy());
    this.levelUBOs.forEach((b) => b.destroy());
    this.textures = [];
    this.levelUBOs = [];
    const w = this.width;
    const h = this.height;
    // COPY_SRC lets measurement scripts read the images back.
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    const tex = (label: string, width: number, height: number, mips = 1) => {
      const t = d.createTexture({ label, size: [width, height], format: 'rgba16float', usage, mipLevelCount: mips });
      this.textures.push(t);
      return t;
    };
    const hdr = tex('trace hdr', w, h);
    const aux = tex('trace aux', w, h);
    const history = [0, 1].map((i) => tex(`history ${i}`, w, h));
    this.traceBG1 = bindGroup(d, this.tracePipe, 1, [hdr.createView(), aux.createView()]);
    this.accumulateBG1 = [0, 1].map((i) => bindGroup(d, this.accumulatePipe, 1, [hdr.createView(), history[1 - i].createView(), history[i].createView()]));

    // Bloom chain from half resolution down.
    const bw = Math.max(1, w >> 1);
    const bh = Math.max(1, h >> 1);
    this.bloomLevels = Math.max(1, Math.min(BLOOM_LEVELS, Math.floor(Math.log2(Math.min(bw, bh))) - 2));
    const down = tex('bloom down', bw, bh, this.bloomLevels);
    const up = tex('bloom up', bw, bh, this.bloomLevels);
    const level = (t: GPUTexture, m: number) => t.createView({ baseMipLevel: m, mipLevelCount: 1 });
    this.bloomSizes = [];
    for (let m = 0; m < this.bloomLevels; m++) this.bloomSizes.push([Math.max(1, bw >> m), Math.max(1, bh >> m)]);
    const levelUBO = (srcW: number, srcH: number, karis: number) => {
      const b = uniformBuffer(d, 16, 'bloom level');
      d.queue.writeBuffer(b, 0, new Float32Array([1 / srcW, 1 / srcH, karis, 0]));
      this.levelUBOs.push(b);
      return b;
    };
    this.downBGs = [0, 1].map((i) =>
      this.bloomSizes.map((_, m) => {
        const src = m === 0 ? history[i].createView() : level(down, m - 1);
        const [sw, sh] = m === 0 ? [w, h] : this.bloomSizes[m - 1];
        return bindGroup(d, this.downPipe, 1, [src, this.linearClamp, level(down, m), levelUBO(sw, sh, m === 0 ? 1 : 0)]);
      }),
    );
    this.upBGs = [];
    for (let m = 0; m < this.bloomLevels - 1; m++) {
      const src = m === this.bloomLevels - 2 ? level(down, m + 1) : level(up, m + 1);
      const [sw, sh] = this.bloomSizes[m + 1];
      this.upBGs.push(bindGroup(d, this.upPipe, 1, [src, this.linearClamp, level(up, m), levelUBO(sw, sh, 0), level(down, m)]));
    }
    const bloomResult = this.bloomLevels > 1 ? level(up, 0) : level(down, 0);
    this.compositeBG1 = [0, 1].map((i) => bindGroup(d, this.compositePipe, 1, [history[i].createView(), bloomResult, aux.createView(), this.linearClamp]));
    this.sizeDirty = false;
    this.signature = '';
  }

  /** Applies a preset by name (also used by tools/shots-blackhole.mjs). */
  preset(name: string) {
    applyPreset(this.params, this.camera, name);
    this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.sizeDirty = true;
  }

  /** Observer velocity relative to the local static frame (units of c). */
  private observerVelocity(eye: Vec3, diskN: Vec3): Vec3 {
    const r = vec3.length(eye);
    const rhat = vec3.scale(eye, 1 / r);
    if (this.params.observer === 1) {
      // Free fall from rest at infinity: a static observer clocks β = √(r_s/r) inwards.
      return vec3.scale(rhat, -Math.min(Math.sqrt(1 / r), 0.999));
    }
    if (this.params.observer === 2) {
      // Circular geodesic speed β = √(M/(r - 2M)) about the disk axis (a
      // geodesic only in the disk plane; elsewhere just a velocity for aberration).
      const phi = vec3.cross(diskN, rhat);
      const len = vec3.length(phi);
      if (len < 1e-6 || r <= 1.5) return [0, 0, 0];
      return vec3.scale(phi, Math.min(Math.sqrt(0.5 / (r - 1)), 0.999) / len);
    }
    return [0, 0, 0];
  }

  private writeUniforms(info: FrameInfo, jitter: [number, number], weight: number) {
    const p = this.params;
    const cam = this.camera;
    const f = this.frameData;
    const m = cam.view;
    const tanY = Math.tan(deg(p.fovDeg) / 2);
    const aspect = this.width / this.height;
    const right: Vec3 = [m[0], m[4], m[8]];
    const up: Vec3 = [m[1], m[5], m[9]];
    const fwd: Vec3 = [-m[2], -m[6], -m[10]];
    // A static observer cannot exist at or inside the horizon: keep the
    // camera outside (panning can move it anywhere).
    const rEye = vec3.length(cam.eye);
    const eye = rEye < R_MIN_OBS ? vec3.scale(cam.eye, R_MIN_OBS / Math.max(rEye, 1e-6)) : cam.eye;
    const rObs = vec3.length(eye);
    const tilt = deg(p.diskTilt);
    const diskN: Vec3 = [0, Math.cos(tilt), Math.sin(tilt)];
    const beta = this.observerVelocity(eye, diskN);
    const b2 = vec3.dot(beta, beta);
    const periodIn = 2 * Math.PI / Math.sqrt(0.5 / p.diskIn ** 3);

    f.set([...eye, rObs], 0);
    f.set([...vec3.scale(right, tanY * aspect), info.time], 4);
    f.set([...vec3.scale(up, tanY), this.frameIndex], 8);
    f.set([...fwd, p.renderMode === 2 ? p.split : -1], 12);
    f.set([...diskN, Math.max(p.diskIn, 1.51)], 16);
    f.set([1, 0, 0, Math.max(p.diskOut, p.diskIn + 0.1)], 20);
    f.set([...beta, 1 / Math.sqrt(1 - b2)], 24);
    f.set([0, 0, -1, p.beacon ? p.beaconFlux : 0], 28);
    f.set([...GALAXY_N, (2 * tanY) / this.height], 32);
    f.set([...GALAXY_C, p.obsRedshift ? 1 : 0], 36);
    f.set([this.width, this.height, jitter[0], jitter[1]], 40);
    f.set([p.integrator, p.dphi, p.tolerance, p.maxSteps], 44);
    f.set([p.cartStep, Math.max(2 * rObs, 2 * p.diskOut, 30), Math.tan(deg(p.shiftX)), Math.tan(deg(p.shiftY))], 48);
    f.set([p.artPull, p.artStep, p.artMaxSteps, Math.max(2 * rObs, 2 * p.diskOut, 30)], 52);
    f.set([p.diskTemp, p.opticalDepth, p.turbulence, p.disk ? 1 : 0], 56);
    f.set([p.doppler ? 1 : 0, p.gravRedshift ? 1 : 0, p.intensityMode, p.diskBrightness], 60);
    // Normalise disk emission by the luminance at the peak temperature, so the
    // temperature slider changes colour (and how strongly g changes the
    // brightness) but not the overall exposure.
    f.set([this.diskTime, p.cycle * periodIn, 0, Math.pow(2, -blackbodyLog2Luminance(this.bbTable, p.diskTemp))], 64);
    f.set([p.starBrightness, p.galaxyBrightness, p.footprintMode, p.filterWidth], 68);
    f.set([p.starSizeMicro * 1e-6, p.beaconSizeMicro * 1e-6, p.stars ? 1 : 0, p.galaxy ? 1 : 0], 72);
    f.set([p.renderMode, p.debugView, p.stats ? 1 : 0, 0], 76);
    f.set([p.exposure, p.bloom, weight, 1 / this.bloomLevels], 80);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    if (this.sizeDirty) this.rebuildTargets();
    const p = this.params;
    const cam = this.camera;
    cam.fovY = deg(p.fovDeg);

    // Progressive accumulation restarts whenever anything that changes the
    // image changes (time is frozen while accumulating).
    const sig = JSON.stringify([p, cam.eye, cam.target, this.width, this.height]);
    const changed = sig !== this.signature;
    this.signature = sig;
    if (changed || !p.accumulate) this.accumulated = 0;
    if (!p.accumulate) this.diskTime += info.dt * p.timeScale;
    // R2 low-discrepancy jitter (Roberts 2018).
    const n = this.accumulated;
    const jitter: [number, number] = p.accumulate && n > 0
      ? [((0.5 + n * 0.7548776662) % 1) - 0.5, ((0.5 + n * 0.5698402910) % 1) - 0.5]
      : [0, 0];
    this.writeUniforms(info, jitter, 1 / (n + 1));

    encoder.clearBuffer(this.statsBuffer, 0, 40);
    encoder.clearBuffer(this.statsBuffer, 48, 16);
    if (changed) encoder.clearBuffer(this.statsBuffer, 40, 8);

    const groups = (x: number) => Math.ceil(x / 8);
    const cp = encoder.beginComputePass({ label: 'black hole' });
    cp.setPipeline(this.tracePipe);
    cp.setBindGroup(0, this.traceBG0);
    cp.setBindGroup(1, this.traceBG1);
    cp.dispatchWorkgroups(groups(this.width), groups(this.height));
    cp.setPipeline(this.accumulatePipe);
    cp.setBindGroup(0, this.accumulateBG0);
    cp.setBindGroup(1, this.accumulateBG1[this.current]);
    cp.dispatchWorkgroups(groups(this.width), groups(this.height));
    cp.setPipeline(this.downPipe);
    cp.setBindGroup(0, this.downBG0);
    this.bloomSizes.forEach(([w, h], m) => {
      cp.setBindGroup(1, this.downBGs[this.current][m]);
      cp.dispatchWorkgroups(groups(w), groups(h));
    });
    cp.setPipeline(this.upPipe);
    cp.setBindGroup(0, this.upBG0);
    for (let m = this.bloomLevels - 2; m >= 0; m--) {
      const [w, h] = this.bloomSizes[m];
      cp.setBindGroup(1, this.upBGs[m]);
      cp.dispatchWorkgroups(groups(w), groups(h));
    }
    cp.end();

    const rp = encoder.beginRenderPass({
      label: 'blackhole composite',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.compositePipe);
    rp.setBindGroup(0, this.compositeBG0);
    rp.setBindGroup(1, this.compositeBG1[this.current]);
    rp.draw(3);
    rp.end();

    if (p.stats && !this.readPending) {
      encoder.copyBufferToBuffer(this.statsBuffer, 0, this.readback, 0, STATS * 4);
      this.readPending = true;
      const tanY = Math.tan(cam.fovY / 2);
      const meta = {
        pixels: this.width * this.height,
        pixelArea: ((2 * tanY * this.width / this.height) / this.width) * ((2 * tanY) / this.height),
        tanHalfHeight: tanY,
        rObs: Math.max(vec3.length(cam.eye), R_MIN_OBS),
        centred: vec3.length(cam.target) < 1e-6 && p.shiftX === 0 && p.shiftY === 0,
        physical: p.renderMode === 1,
        diskOn: p.disk,
        staticObserver: p.observer === 0,
      };
      this.device.queue.onSubmittedWorkDone().then(() =>
        this.readback.mapAsync(GPUMapMode.READ).then(() => {
          this.stats = { values: new Uint32Array(this.readback.getMappedRange().slice(0)), ...meta };
          this.readback.unmap();
          this.readPending = false;
        }),
      );
    }

    this.current = 1 - this.current;
    this.accumulated++;
    this.frameIndex++;
  }

  /**
   * Shadow size from the captured-pixel count. A circle of angular radius α
   * centred on the view axis projects to a disc of radius tan α on the image
   * plane, so α = atan √(N·A_px/π). A static observer at r_obs sees the
   * shadow edge at sin α = (b_c / r_obs) √(1 - r_s/r_obs) (Synge 1966),
   * which we invert for b.
   */
  shadowMeasurement() {
    const s = this.stats;
    if (!s) return null;
    const frames = Math.max(1, s.values[ST.accFrames]);
    const captured = s.values[ST.accCaptured] / frames;
    const alpha = Math.atan(Math.sqrt((captured * s.pixelArea) / Math.PI));
    const b = (s.rObs * Math.sin(alpha)) / Math.sqrt(1 - 1 / s.rObs);
    const alphaTheory = Math.asin(Math.min(1, (B_CRIT / s.rObs) * Math.sqrt(1 - 1 / s.rObs)));
    return { alpha, alphaTheory, b, frames, captured };
  }

  hud() {
    const p = this.params;
    const mode = ['artistic', 'Schwarzschild', 'split artistic | Schwarzschild'][p.renderMode];
    const integ = ['plane RK4', 'plane DP5(4)', 'Cartesian RK4'][p.integrator];
    const rObs = Math.max(vec3.length(this.camera.eye), R_MIN_OBS);
    let text = `${mode} · ${integ} · r_obs ${rObs.toFixed(2)} r_s`;
    const s = this.stats;
    if (p.stats && s) {
      const v = s.values;
      const px = Math.max(1, s.pixels);
      const pct = (x: number) => `${((100 * x) / px).toFixed(1)}%`;
      text += `\nsteps/px ${(v[ST.steps] / px).toFixed(1)} (max ${v[ST.maxSteps]})`;
      if (v[ST.extraSteps] > 0) text += ` + ${(v[ST.extraSteps] / px).toFixed(1)} for finite differences`;
      text += `\nescaped ${pct(v[ST.escaped])} · captured ${pct(v[ST.captured])} · in disk ${pct(v[ST.absorbed])} · step limit ${v[ST.limit]} px · non-finite ${v[ST.nonFinite]}`;
      text += `\ndisk hits ≥1 ${pct(v[ST.order1])} · ≥2 ${pct(v[ST.order2])} · ≥3 ${v[ST.order3]} px · plane crossings m≥2 ${pct(v[ST.nodes2])} · m≥3 ${v[ST.nodes3]} px`;
      const m = this.shadowMeasurement();
      // Only meaningful while the whole shadow is in view.
      const inView = m !== null && Math.tan(m.alpha) < 0.95 * s.tanHalfHeight;
      if (m && inView && s.centred && s.staticObserver && m.captured > 0) {
        const note = s.diskOn ? ' (disk hides part: turn it off to measure)' : '';
        if (s.physical) {
          text += `\nshadow b = ${m.b.toFixed(4)} r_s vs 3√3/2 = ${B_CRIT.toFixed(4)} (${((100 * (m.b - B_CRIT)) / B_CRIT).toFixed(2)}%), ${m.frames} frame(s)${note}`;
        } else if (p.renderMode === 0) {
          text += `\nshadow ${(m.alpha * 180 / Math.PI).toFixed(3)}° vs Schwarzschild ${(m.alphaTheory * 180 / Math.PI).toFixed(3)}° (equiv. b = ${m.b.toFixed(3)} r_s)${note}`;
        }
      }
    }
    return text + '\ndrag: orbit · wheel: distance · right-drag: pan';
  }

  destroy() {
    this.textures.forEach((t) => t.destroy());
    this.levelUBOs.forEach((b) => b.destroy());
    [this.frameUBO, this.bbBuffer, this.statsBuffer, this.readback].forEach((b) => b.destroy());
  }
}

export const blackholeEntry: DemoEntry = {
  id: 'blackhole',
  title: 'Black hole lensing',
  tags: ['null geodesics', 'ray differentials', 'relativistic beaming', 'prefiltering'],
  info: `
    <h2>Schwarzschild black hole</h2>
    <ul>
      <li>Artistic inverse-square bending vs exact null geodesics (split screen)</li>
      <li>Orbit equation u'' + u = (3/2) r_s u² in each photon's plane: RK4 or adaptive Dormand–Prince; Cartesian pseudo-force cross-check</li>
      <li>Thin disk from the ISCO, Shakura–Sunyaev temperatures as blackbodies, Keplerian turbulence</li>
      <li>Doppler beaming and gravitational redshift (T → gT), observer blueshift and aberration</li>
      <li>Procedural starfield and galaxy, prefiltered by analytic ray differentials</li>
      <li>Live shadow-size measurement against 3√3/2 r_s</li>
    </ul>
    <p><kbd>drag</kbd> orbit · <kbd>wheel</kbd> distance · units of r_s</p>`,
  create: (ctx) => new BlackHoleDemo(ctx),
};
