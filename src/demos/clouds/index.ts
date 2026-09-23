import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, uniformBuffer } from '../../core/gpu';
import { deg, type Mat4, type Vec3 } from '../../core/math';
import { ambientLight, sunColor, sunTopIntensity, type SkySettings } from '../../shared/sky';
import skyWgsl from '../../shared/sky.wgsl?raw';
import { DEFAULTS, buildGui, type CloudParams } from './params';

import commonWgsl from './shaders/common.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import densityWgsl from './shaders/density.wgsl?raw';
import mip3dWgsl from './shaders/mip3d.wgsl?raw';
import noiseBaseWgsl from './shaders/noise_base.wgsl?raw';
import noiseLibWgsl from './shaders/noise_lib.wgsl?raw';
import noiseSmallWgsl from './shaders/noise_small.wgsl?raw';
import resolveWgsl from './shaders/resolve.wgsl?raw';
import shadowWgsl from './shaders/shadow.wgsl?raw';
import traceWgsl from './shaders/trace.wgsl?raw';
import viewWgsl from './shaders/view.wgsl?raw';

const PLANET_RADIUS = 6_360_000;
const BASE_SIZE = 128;
const BASE_SLAB = 16; // slices generated per frame, keeps each dispatch short
const DETAIL_SIZE = 32;
const CURL_SIZE = 128;
const WEATHER_SIZE = 512;
const SHADOW_SIZE = 256;
const SHADOW_EXTENT = 40_000;
const CURL_TILE = 9_000;

const wgsl = (...parts: string[]) => parts.join('\n');
const wrap = (x: number, m: number) => x - Math.floor(x / m) * m;

class CloudsDemo implements Demo {
  private device: GPUDevice;
  private camera: DemoContext['camera'];
  private params: CloudParams = structuredClone(DEFAULTS);

  private frameIndex = 0;
  private historyValid = false;
  private current = 0;
  private baseSlab = 0;
  private weatherDirty = true;
  private prevViewProj: Mat4 | null = null;
  private wind = { base: [0, 0], detail: [0, 0], weather: [0, 0] };
  private width = 1;
  private height = 1;
  private cloudSize: [number, number] = [1, 1];
  private traceSize: [number, number] = [1, 1];
  private sizeDirty = true;

  private frameData = new Float32Array(56);
  private cloudData = new Float32Array(72);
  private frameUBO: GPUBuffer;
  private cloudUBO: GPUBuffer;
  private genUBO: GPUBuffer;
  private weatherUBO: GPUBuffer;

  private textures: GPUTexture[] = [];
  private base: GPUTexture;
  private detail: GPUTexture;
  private weatherView: GPUTextureView;
  private shadowView: GPUTextureView;
  private screenTextures: GPUTexture[] = [];

  private noiseSampler: GPUSampler;
  private linearClamp: GPUSampler;
  private linearRepeat: GPUSampler;

  private genBasePipe: GPUComputePipeline;
  private genDetailPipe: GPUComputePipeline;
  private genCurlPipe: GPUComputePipeline;
  private genWeatherPipe: GPUComputePipeline;
  private mipPipe: GPUComputePipeline;
  private shadowPipe: GPUComputePipeline;
  private tracePipe: GPUComputePipeline;
  private resolvePipe: GPUComputePipeline;
  private compositePipe: GPURenderPipeline;

  private genBaseBG: GPUBindGroup[];
  private genSmallBG: { detail: GPUBindGroup[]; curl: GPUBindGroup[]; weather: GPUBindGroup[] };
  private mipBGs: { group: GPUBindGroup; size: number }[] = [];
  private shadowBG: GPUBindGroup[];
  private traceBG01!: GPUBindGroup[];
  private traceBG2!: GPUBindGroup;
  private resolveBG0!: GPUBindGroup;
  private resolveBG1!: GPUBindGroup[];
  private compositeBG0!: GPUBindGroup;
  private compositeBG1!: GPUBindGroup[];

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.camera = ctx.camera;
    Object.assign(this.camera, {
      target: [0, 250, 0] as Vec3,
      distance: 0.5,
      minDistance: 0.5,
      yaw: 0.6,
      pitch: -0.18,
      minPitch: -1.45,
      maxPitch: 1.45,
      moveSpeed: 250,
      minEyeHeight: 2,
    });
    this.camera.near = 0.5;

    buildGui(ctx.gui, this.params, {
      weatherChanged: () => (this.weatherDirty = true),
      resized: () => (this.sizeDirty = true),
      resetHistory: () => (this.historyValid = false),
    });

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'frame');
    this.cloudUBO = uniformBuffer(device, this.cloudData.byteLength, 'clouds');
    this.genUBO = uniformBuffer(device, 16, 'noise gen');
    this.weatherUBO = uniformBuffer(device, 16, 'weather gen');

    const storage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (d: GPUTextureDescriptor) => {
      const t = device.createTexture(d);
      this.textures.push(t);
      return t;
    };
    this.base = tex({ label: 'base noise', size: [BASE_SIZE, BASE_SIZE, BASE_SIZE], dimension: '3d', format: 'rgba8unorm', usage: storage, mipLevelCount: Math.log2(BASE_SIZE) + 1 });
    this.detail = tex({ label: 'detail noise', size: [DETAIL_SIZE, DETAIL_SIZE, DETAIL_SIZE], dimension: '3d', format: 'rgba8unorm', usage: storage, mipLevelCount: Math.log2(DETAIL_SIZE) + 1 });
    const curl = tex({ label: 'curl noise', size: [CURL_SIZE, CURL_SIZE], format: 'rgba8snorm', usage: storage });
    const weather = tex({ label: 'weather map', size: [WEATHER_SIZE, WEATHER_SIZE], format: 'rgba8unorm', usage: storage });
    const shadow = tex({ label: 'cloud shadow map', size: [SHADOW_SIZE, SHADOW_SIZE], format: 'rgba16float', usage: storage });
    this.weatherView = weather.createView();
    this.shadowView = shadow.createView();

    this.noiseSampler = device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
    });
    this.linearClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.linearRepeat = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear' });

    const compute = (label: string, code: string, entryPoint: string) =>
      device.createComputePipeline({ label, layout: 'auto', compute: { module: createShader(device, label, code), entryPoint } });
    this.genBasePipe = compute('base noise', wgsl(commonWgsl, noiseLibWgsl, noiseBaseWgsl), 'genBase');
    const smallModule = wgsl(commonWgsl, noiseLibWgsl, noiseSmallWgsl);
    this.genDetailPipe = compute('detail noise', smallModule, 'genDetail');
    this.genCurlPipe = compute('curl noise', smallModule, 'genCurl');
    this.genWeatherPipe = compute('weather map', smallModule, 'genWeather');
    this.mipPipe = compute('mip 3d', mip3dWgsl, 'downsample');
    this.shadowPipe = compute('cloud shadows', wgsl(commonWgsl, densityWgsl, shadowWgsl), 'shadow');
    this.tracePipe = compute('cloud trace', wgsl(commonWgsl, skyWgsl, viewWgsl, densityWgsl, traceWgsl), 'trace');
    this.resolvePipe = compute('temporal resolve', wgsl(commonWgsl, viewWgsl, resolveWgsl), 'resolve');
    const compositeModule = createShader(device, 'composite', wgsl(commonWgsl, skyWgsl, viewWgsl, compositeWgsl));
    this.compositePipe = device.createRenderPipeline({
      label: 'composite',
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });

    const bg = (p: GPUComputePipeline | GPURenderPipeline, g: number, r: Parameters<typeof bindGroup>[3]) => bindGroup(device, p, g, r);
    const level = (t: GPUTexture, mip: number) =>
      t.createView({ dimension: t.dimension === '3d' ? '3d' : '2d', baseMipLevel: mip, mipLevelCount: 1 });
    this.genBaseBG = [bg(this.genBasePipe, 0, [this.genUBO]), bg(this.genBasePipe, 1, [level(this.base, 0)])];
    this.genSmallBG = {
      detail: [bg(this.genDetailPipe, 0, [this.genUBO]), bg(this.genDetailPipe, 1, [level(this.detail, 0)])],
      curl: [bg(this.genCurlPipe, 0, [this.genUBO]), bg(this.genCurlPipe, 1, [null, curl.createView()])],
      weather: [bg(this.genWeatherPipe, 0, [this.weatherUBO]), bg(this.genWeatherPipe, 1, [null, null, this.weatherView])],
    };
    for (const t of [this.base, this.detail]) {
      for (let m = 1; m < t.mipLevelCount; m++) {
        this.mipBGs.push({ group: bg(this.mipPipe, 0, [level(t, m - 1), level(t, m)]), size: Math.max(1, t.width >> m) });
      }
    }
    const noise = (p: GPUComputePipeline) =>
      bg(p, 1, [this.base.createView(), this.detail.createView(), curl.createView(), this.weatherView, this.noiseSampler]);
    this.shadowBG = [bg(this.shadowPipe, 0, [this.frameUBO, this.cloudUBO]), noise(this.shadowPipe), bg(this.shadowPipe, 2, [this.shadowView])];
    this.traceBG01 = [bg(this.tracePipe, 0, [this.frameUBO, this.cloudUBO]), noise(this.tracePipe)];
  }

  /** (Re)creates the resolution-dependent cloud buffers. */
  private rebuildTargets() {
    const d = this.device;
    const p = this.params;
    this.screenTextures.forEach((t) => t.destroy());
    const cw = Math.max(1, Math.round(this.width * p.resolutionScale));
    const ch = Math.max(1, Math.round(this.height * p.resolutionScale));
    const quarter = p.updatePattern === 4;
    const tw = quarter ? Math.ceil(cw / 2) : cw;
    const th = quarter ? Math.ceil(ch / 2) : ch;
    this.cloudSize = [cw, ch];
    this.traceSize = [tw, th];
    const storage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const trace = d.createTexture({ label: 'cloud trace', size: [tw, th], format: 'rgba16float', usage: storage });
    const depth = d.createTexture({ label: 'cloud depth', size: [tw, th], format: 'r32float', usage: storage });
    const history = [0, 1].map((i) => d.createTexture({ label: `cloud history ${i}`, size: [cw, ch], format: 'rgba16float', usage: storage }));
    this.screenTextures = [trace, depth, ...history];

    const traceView = trace.createView();
    const depthView = depth.createView();
    const historyViews = history.map((t) => t.createView());
    this.traceBG2 = bindGroup(d, this.tracePipe, 2, [traceView, depthView, historyViews[0]]);
    this.resolveBG0 = bindGroup(d, this.resolvePipe, 0, [this.frameUBO, this.cloudUBO]);
    this.resolveBG1 = [0, 1].map((i) =>
      bindGroup(d, this.resolvePipe, 1, [traceView, depthView, historyViews[1 - i], this.linearClamp, historyViews[i]]),
    );
    this.compositeBG0 = bindGroup(d, this.compositePipe, 0, [this.frameUBO, this.cloudUBO]);
    this.compositeBG1 = [0, 1].map((i) =>
      bindGroup(d, this.compositePipe, 1, [
        historyViews[i], traceView, depthView, this.shadowView, this.weatherView, this.linearClamp, this.linearRepeat,
      ]),
    );
    this.historyValid = false;
    this.sizeDirty = false;
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.sizeDirty = true;
  }

  private get noiseReady() {
    return this.baseSlab >= BASE_SIZE / BASE_SLAB;
  }

  private skySettings(): SkySettings {
    const p = this.params;
    const el = deg(p.sunElevation);
    const az = deg(p.sunAzimuth);
    const sunDir: Vec3 = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
    return { sunDir, sunIntensity: p.sunIntensity, turbidity: p.turbidity, rayleigh: 1, mieG: 0.8, boost: 2.2 };
  }

  private writeUniforms(info: FrameInfo) {
    const p = this.params;
    const cam = this.camera;
    const sky = this.skySettings();

    const f = this.frameData;
    f.set(cam.viewProj, 0);
    f.set(cam.invViewProj, 16);
    f.set([...cam.eye, info.time], 32);
    f.set([...sky.sunDir, info.dt], 36);
    f.set([...sunColor(sky), p.exposure], 40);
    f.set([...ambientLight(sky), p.debugView], 44);
    f.set([this.width, this.height, cam.near, sunTopIntensity(sky)], 48);
    f.set([sky.turbidity, sky.rayleigh, sky.mieG, sky.boost], 52);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);

    // Wind: accumulate offsets (sampling at x - v t moves features along v)
    // and wrap them to their noise tiles to keep float precision.
    const wd = deg(p.windDirection);
    const dir = [Math.cos(wd), Math.sin(wd)];
    const step = p.windSpeed * info.dt;
    const baseTile = p.baseScaleKm * 1000;
    const detailTile = p.detailScaleKm * 1000;
    const weatherTile = p.weatherSizeKm * 1000;
    for (let i = 0; i < 2; i++) {
      this.wind.base[i] = wrap(this.wind.base[i] - dir[i] * step, baseTile);
      this.wind.detail[i] = wrap(this.wind.detail[i] - dir[i] * step * p.detailSpeed, detailTile);
      this.wind.weather[i] = wrap(this.wind.weather[i] - dir[i] * step * p.weatherSpeed, weatherTile);
    }

    const shadowTexel = SHADOW_EXTENT / SHADOW_SIZE;
    const quarter = p.updatePattern === 4;
    const c = this.cloudData;
    c.set(this.prevViewProj ?? cam.viewProj, 0);
    c.set([p.layerBottom, Math.max(p.layerTop, p.layerBottom + 100), PLANET_RADIUS, p.maxDistanceKm * 1000], 16);
    c.set([p.coverage, p.cloudType, this.noiseReady ? p.density : 0, 1 / baseTile], 20);
    c.set([1 / detailTile, p.erosion, 1 / CURL_TILE, p.curlAmplitude], 24);
    c.set([this.wind.base[0], this.wind.base[1], this.wind.detail[0], this.wind.detail[1]], 28);
    c.set([weatherTile, this.wind.weather[0], this.wind.weather[1], p.heightSkew], 32);
    c.set([p.extinction, p.albedo, p.powder, p.ambient], 36);
    c.set([p.forwardG, p.backwardG, p.forwardWeight, p.lightDistance], 40);
    c.set([p.octaves, p.msExtinction, p.msContribution, p.msPhase], 44);
    c.set([p.steps, p.lightSteps, p.jitter ? 1 : 0, p.horizonSteps], 48);
    // A pixel refreshed every 4th frame needs a larger weight to converge as fast.
    const blend = quarter ? Math.min(1, p.historyBlend * 2.5) : p.historyBlend;
    c.set([this.frameIndex, p.updatePattern, blend, p.temporal && this.historyValid ? 1 : 0], 52);
    c.set([
      Math.round(cam.eye[0] / shadowTexel) * shadowTexel,
      Math.round(cam.eye[2] / shadowTexel) * shadowTexel,
      SHADOW_EXTENT,
      p.shadowStrength,
    ], 56);
    c.set([cam.eye[1], p.fog, p.showWeather ? 1 : 0, p.showShadow ? 1 : 0], 60);
    c.set([dir[0], dir[1], 0, 0], 64);
    c.set([p.lightAbsorption, 0, 0, 0], 68);
    this.device.queue.writeBuffer(this.cloudUBO, 0, c);
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    if (this.sizeDirty) this.rebuildTargets();
    const p = this.params;
    const [tw, th] = this.traceSize;
    const [cw, ch] = this.cloudSize;
    const groups = (n: number, s = 8) => Math.ceil(n / s);

    const pass = encoder.beginComputePass({ label: 'clouds' });
    // Noise generation, spread over the first frames.
    if (!this.noiseReady) {
      if (this.baseSlab === 0) {
        this.device.queue.writeBuffer(this.genUBO, 0, new Uint32Array([0, 1, 0, 0]));
        pass.setPipeline(this.genDetailPipe);
        this.genSmallBG.detail.forEach((g, i) => pass.setBindGroup(i, g));
        pass.dispatchWorkgroups(groups(DETAIL_SIZE, 4), groups(DETAIL_SIZE, 4), groups(DETAIL_SIZE, 4));
        pass.setPipeline(this.genCurlPipe);
        this.genSmallBG.curl.forEach((g, i) => pass.setBindGroup(i, g));
        pass.dispatchWorkgroups(groups(CURL_SIZE), groups(CURL_SIZE));
      } else {
        this.device.queue.writeBuffer(this.genUBO, 0, new Uint32Array([this.baseSlab * BASE_SLAB, 1, 0, 0]));
      }
      pass.setPipeline(this.genBasePipe);
      this.genBaseBG.forEach((g, i) => pass.setBindGroup(i, g));
      pass.dispatchWorkgroups(groups(BASE_SIZE, 4), groups(BASE_SIZE, 4), BASE_SLAB / 4);
      this.baseSlab++;
      if (this.noiseReady) {
        pass.setPipeline(this.mipPipe);
        for (const m of this.mipBGs) {
          pass.setBindGroup(0, m.group);
          pass.dispatchWorkgroups(groups(m.size, 4), groups(m.size, 4), groups(m.size, 4));
        }
        this.historyValid = false;
      }
    }
    // The weather map has its own uniform: queue writes all land before the
    // submission, so sharing one with the slab offset would clobber it.
    if (this.weatherDirty) {
      this.device.queue.writeBuffer(this.weatherUBO, 0, new Uint32Array([0, p.seed, 0, 0]));
      pass.setPipeline(this.genWeatherPipe);
      this.genSmallBG.weather.forEach((g, i) => pass.setBindGroup(i, g));
      pass.dispatchWorkgroups(groups(WEATHER_SIZE), groups(WEATHER_SIZE));
      this.weatherDirty = false;
    }
    pass.end();

    this.writeUniforms(info);
    const cp = encoder.beginComputePass({ label: 'cloud render' });
    cp.setPipeline(this.shadowPipe);
    this.shadowBG.forEach((g, i) => cp.setBindGroup(i, g));
    cp.dispatchWorkgroups(groups(SHADOW_SIZE), groups(SHADOW_SIZE));
    cp.setPipeline(this.tracePipe);
    cp.setBindGroup(0, this.traceBG01[0]);
    cp.setBindGroup(1, this.traceBG01[1]);
    cp.setBindGroup(2, this.traceBG2);
    cp.dispatchWorkgroups(groups(tw), groups(th));
    cp.setPipeline(this.resolvePipe);
    cp.setBindGroup(0, this.resolveBG0);
    cp.setBindGroup(1, this.resolveBG1[this.current]);
    cp.dispatchWorkgroups(groups(cw), groups(ch));
    cp.end();

    const rp = encoder.beginRenderPass({
      label: 'composite',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.compositePipe);
    rp.setBindGroup(0, this.compositeBG0);
    rp.setBindGroup(1, this.compositeBG1[this.current]);
    rp.draw(3);
    rp.end();

    this.prevViewProj = new Float32Array(this.camera.viewProj);
    this.current = 1 - this.current;
    this.frameIndex++;
    this.historyValid = true;
  }

  hud() {
    const p = this.params;
    const [cw, ch] = this.cloudSize;
    const [tw, th] = this.traceSize;
    const status = this.noiseReady ? '' : `  generating noise ${this.baseSlab}/${BASE_SIZE / BASE_SLAB}`;
    const pattern = p.updatePattern === 4 ? '1/4 px per frame' : 'every px';
    return `clouds ${cw}x${ch}, traced ${tw}x${th} (${pattern}), ${p.steps} steps${status}\n` +
      `altitude ${Math.round(this.camera.eye[1])} m · drag: look · WASD/QE: fly (shift: faster)`;
  }

  destroy() {
    [...this.textures, ...this.screenTextures].forEach((t) => t.destroy());
    [this.frameUBO, this.cloudUBO, this.genUBO, this.weatherUBO].forEach((b) => b.destroy());
  }
}

export const cloudsEntry: DemoEntry = {
  id: 'clouds',
  title: 'Volumetric clouds',
  tags: ['ray marching', '3D noise', 'multiple scattering', 'temporal reprojection'],
  info: `
    <h2>Volumetric clouds</h2>
    <ul>
      <li>Tileable Perlin-Worley / Worley 3D noise built in compute shaders</li>
      <li>Weather map: coverage, cloud type, density</li>
      <li>Height profiles per cloud type, coverage remap, curl-distorted detail erosion</li>
      <li>Beer–Lambert light march in a cone, Beer–powder, dual-lobe HG phase</li>
      <li>Multiple-scattering octaves, energy-conserving integration</li>
      <li>Jittered march, 1-of-4 pixel updates, reprojection with variance clipping</li>
      <li>Cloud shadow map on the ground, aerial perspective</li>
    </ul>
    <p><kbd>drag</kbd> look around · <kbd>wheel</kbd> zoom<br>
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up · hold <kbd>shift</kbd> for speed</p>`,
  create: (ctx) => new CloudsDemo(ctx),
};
