import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, gridMesh, indexBuffer, uniformBuffer, vertexBuffer } from '../../core/gpu';
import { deg, type Vec3 } from '../../core/math';
import { box, buoy, sphere, type Mesh } from './geometry';
import { DEFAULTS, SPECTRUM_KEYS, buildGui, type OceanParams } from './params';
import { ambientLight, sunColor, sunTopIntensity, type SkySettings } from './sky';

import assembleWgsl from './shaders/assemble.wgsl?raw';
import buoyancyWgsl from './shaders/buoyancy.wgsl?raw';
import causticsWgsl from './shaders/caustics.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import fftWgsl from './shaders/fft.wgsl?raw';
import lightingWgsl from './shaders/lighting.wgsl?raw';
import mipgenWgsl from './shaders/mipgen.wgsl?raw';
import objectsWgsl from './shaders/objects.wgsl?raw';
import oceanWgsl from './shaders/ocean.wgsl?raw';
import ripplesWgsl from './shaders/ripples.wgsl?raw';
import skyWgsl from './shaders/sky.wgsl?raw';
import skyPassWgsl from './shaders/sky_pass.wgsl?raw';
import conjugateWgsl from './shaders/spectrum_conjugate.wgsl?raw';
import spectrumInitWgsl from './shaders/spectrum_init.wgsl?raw';
import spectrumUpdateWgsl from './shaders/spectrum_update.wgsl?raw';
import terrainWgsl from './shaders/terrain.wgsl?raw';
import terrainBakeWgsl from './shaders/terrain_bake.wgsl?raw';
import wavesWgsl from './shaders/waves.wgsl?raw';

// --- Configuration ----------------------------------------------------------

const FFT_N = 256;
const CASCADES = 3;
/** Patch sizes in metres. Non-integer ratios avoid visible alignment of tiles. */
const LENGTHS = [250, 34, 7.3];
const MIPS = Math.log2(FFT_N) + 1;
const CLIPMAP_GRID = 128;
const CLIPMAP_LEVELS = 10;
const BASE_SPACING = 0.2;
const TERRAIN_RES = 1024;
const TERRAIN_MESH = 512;
const TERRAIN_HALF = 700;
const ISLAND: [number, number] = [20, -300];
const SHORE_RADIUS = 100;
const RIPPLE_RES = 256;
const RIPPLE_SIZE = 128;
const RIPPLE_ORIGIN: [number, number] = [0, -10];
const CAUSTIC_RES = 512;
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT: GPUTextureFormat = 'depth32float';

type Kind = 0 | 1 | 2; // crate, buoy, beach ball
const KIND_PHYSICS: Record<Kind, { ratio: number; halfHeight: number; radius: number }> = {
  0: { ratio: 2.2, halfHeight: 0.5, radius: 0.62 },
  1: { ratio: 2.3, halfHeight: 0.6, radius: 0.34 },
  2: { ratio: 5.0, halfHeight: 0.5, radius: 0.5 },
};
/** Sorted by kind so each kind is one instanced draw. */
const BODIES: { kind: Kind; x: number; z: number; scale: number }[] = [
  { kind: 0, x: -7, z: -14, scale: 1.4 },
  { kind: 0, x: 10, z: -6, scale: 1.1 },
  { kind: 1, x: 6, z: -22, scale: 1.8 },
  { kind: 1, x: -14, z: -30, scale: 1.6 },
  { kind: 2, x: -1.5, z: -7, scale: 1.0 },
  { kind: 2, x: 3.5, z: -1, scale: 0.8 },
];
const BODY_FLOATS = 32;

const wgsl = (...parts: string[]) => parts.join('\n');

// --- Demo -------------------------------------------------------------------

class OceanDemo implements Demo {
  private device: GPUDevice;
  private params: OceanParams = structuredClone(DEFAULTS);
  private camera: DemoContext['camera'];
  private canvas: HTMLCanvasElement;

  private simTime = 0;
  private spectrumDirty = true;
  private current = 0;
  private dropGeneration = 0;
  private anchors = BODIES.map((b) => ({ ...b }));
  private dragging: { body: number; pointer: number } | null = null;
  private width = 1;
  private height = 1;
  private disposers: (() => void)[] = [];

  // Uniforms
  private frameData = new Float32Array(56);
  private oceanData = new Float32Array(40);
  private simData = new Float32Array(8);
  private rippleData = new Float32Array(8);
  private frameUBO: GPUBuffer;
  private oceanUBO: GPUBuffer;
  private spectrumUBO: GPUBuffer;
  private simUBO: GPUBuffer;
  private rippleUBO: GPUBuffer;
  private bodiesBuffer: GPUBuffer;
  private anchorsBuffer: GPUBuffer;

  // Textures owned for the lifetime of the demo
  private textures: GPUTexture[] = [];
  private dispViews: GPUTextureView[];
  private derivView: GPUTextureView;
  private rippleViews: GPUTextureView[];
  private terrainView: GPUTextureView;
  private causticView: GPUTextureView;

  // Screen-size targets
  private hdr!: GPUTexture;
  private sceneCopy!: GPUTexture;
  private distance!: GPUTexture;
  private depth!: GPUTexture;

  private linSampler: GPUSampler;
  private clampSampler: GPUSampler;

  // Pipelines
  private initPipe: GPUComputePipeline;
  private conjugatePipe: GPUComputePipeline;
  private updatePipe: GPUComputePipeline;
  private fftHPipe: GPUComputePipeline;
  private fftVPipe: GPUComputePipeline;
  private assemblePipe: GPUComputePipeline;
  private mipPipe: GPUComputePipeline;
  private ripplePipe: GPUComputePipeline;
  private buoyancyPipe: GPUComputePipeline;
  private skyPipe: GPURenderPipeline;
  private terrainPipe: GPURenderPipeline;
  private objectsPipe: GPURenderPipeline;
  private causticsPipe: GPURenderPipeline;
  private oceanPipe: GPURenderPipeline;
  private compositePipe: GPURenderPipeline;

  // Bind groups
  private initBG: GPUBindGroup;
  private conjugateBG: GPUBindGroup;
  private updateBG: GPUBindGroup;
  private fftHBG: GPUBindGroup;
  private fftVBG: GPUBindGroup;
  private assembleBG: GPUBindGroup[];
  private mipBGs: GPUBindGroup[][];
  private rippleBG: GPUBindGroup[];
  private buoyancyBG0: GPUBindGroup;
  private buoyancyBG1: GPUBindGroup[];
  private skyBG: GPUBindGroup;
  private terrainBG0: GPUBindGroup;
  private terrainBG1: GPUBindGroup;
  private objectsBG0: GPUBindGroup;
  private objectsBG1: GPUBindGroup;
  private causticsBG0: GPUBindGroup;
  private causticsBG1: GPUBindGroup;
  private oceanBG0: GPUBindGroup;
  private oceanBG1: GPUBindGroup[] = [];
  private compositeBG!: GPUBindGroup;

  // Geometry
  private clipmap: { vb: GPUBuffer; ib: GPUBuffer; count: number; format: GPUIndexFormat };
  private terrainGrid: { vb: GPUBuffer; ib: GPUBuffer; count: number; format: GPUIndexFormat };
  private causticGrid: { vb: GPUBuffer; ib: GPUBuffer; count: number; format: GPUIndexFormat };
  private meshes: { vb: GPUBuffer; ib: GPUBuffer; count: number; first: number; instances: number }[];

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.camera = ctx.camera;
    this.canvas = ctx.canvas;

    this.camera.target = [0, 0, -8];
    this.camera.distance = 34;
    this.camera.pitch = 0.17;
    this.camera.yaw = 0;
    this.camera.minEyeHeight = 0.8;

    buildGui(ctx.gui, this.params, {
      spectrumChanged: () => (this.spectrumDirty = true),
      drop: () => this.drop(),
      resetObjects: () => {
        this.anchors = BODIES.map((b) => ({ ...b }));
        this.drop();
      },
    });
    ctx.gui.onChange((e) => {
      if (SPECTRUM_KEYS.includes(e.property as keyof OceanParams)) this.spectrumDirty = true;
    });

    // --- Buffers ---
    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'frame');
    this.oceanUBO = uniformBuffer(device, this.oceanData.byteLength, 'ocean');
    this.spectrumUBO = uniformBuffer(device, 24 * 4, 'spectrum');
    this.simUBO = uniformBuffer(device, this.simData.byteLength, 'sim');
    this.rippleUBO = uniformBuffer(device, this.rippleData.byteLength, 'ripple');
    this.bodiesBuffer = device.createBuffer({
      label: 'bodies',
      size: BODIES.length * BODY_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.anchorsBuffer = device.createBuffer({
      label: 'anchors',
      size: BODIES.length * 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.initBodies();
    this.writeAnchors();

    // --- Textures ---
    const storage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, size: [number, number, number], format: GPUTextureFormat, usage: number, mips = 1) => {
      const t = device.createTexture({ label, size, format, usage, mipLevelCount: mips });
      this.textures.push(t);
      return t;
    };
    const h0Raw = tex('h0 raw', [FFT_N, FFT_N, CASCADES], 'rgba32float', storage);
    const h0 = tex('h0', [FFT_N, FFT_N, CASCADES], 'rgba32float', storage);
    const waveData = tex('wave data', [FFT_N, FFT_N, CASCADES], 'rgba32float', storage);
    const fftA = tex('fft A', [FFT_N, FFT_N, CASCADES * 2], 'rgba32float', storage);
    const fftB = tex('fft B', [FFT_N, FFT_N, CASCADES * 2], 'rgba32float', storage);
    const disp = [0, 1].map((i) => tex(`displacement ${i}`, [FFT_N, FFT_N, CASCADES], 'rgba16float', storage, MIPS));
    const deriv = tex('derivatives', [FFT_N, FFT_N, CASCADES], 'rgba16float', storage, MIPS);
    const terrain = tex('terrain', [TERRAIN_RES, TERRAIN_RES, 1], 'rgba16float', storage);
    const ripple = [0, 1].map((i) => tex(`ripple ${i}`, [RIPPLE_RES, RIPPLE_RES, 1], 'rgba16float', storage));
    const caustic = tex('caustics', [CAUSTIC_RES, CAUSTIC_RES, 1], 'rgba16float', GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING);

    const arr = (t: GPUTexture, base = 0, count?: number) =>
      t.createView({ dimension: '2d-array', baseMipLevel: base, mipLevelCount: count ?? t.mipLevelCount - base });
    this.dispViews = disp.map((t) => arr(t));
    this.derivView = arr(deriv);
    this.rippleViews = ripple.map((t) => t.createView());
    this.terrainView = terrain.createView();
    this.causticView = caustic.createView();

    this.linSampler = device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8,
    });
    this.clampSampler = device.createSampler({
      addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', magFilter: 'linear', minFilter: 'linear',
    });

    // --- Compute pipelines ---
    const compute = (label: string, code: string, entryPoint: string, constants?: Record<string, number>) =>
      device.createComputePipeline({
        label,
        layout: 'auto',
        compute: { module: createShader(device, label, code), entryPoint, constants },
      });
    this.initPipe = compute('spectrum init', wgsl(commonWgsl, spectrumInitWgsl), 'initSpectrum');
    this.conjugatePipe = compute('spectrum conjugate', wgsl(commonWgsl, conjugateWgsl), 'packConjugate');
    this.updatePipe = compute('spectrum update', wgsl(commonWgsl, spectrumUpdateWgsl), 'updateSpectrum');
    const fftModule = wgsl(commonWgsl, fftWgsl);
    this.fftHPipe = compute('fft horizontal', fftModule, 'fft', { HORIZONTAL: 1 });
    this.fftVPipe = compute('fft vertical', fftModule, 'fft', { HORIZONTAL: 0 });
    this.assemblePipe = compute('assemble', wgsl(commonWgsl, assembleWgsl), 'assemble');
    this.mipPipe = compute('mipgen', mipgenWgsl, 'downsample');
    this.ripplePipe = compute('ripples', wgsl(commonWgsl, ripplesWgsl), 'simulateRipples');
    this.buoyancyPipe = compute('buoyancy', wgsl(commonWgsl, wavesWgsl, buoyancyWgsl), 'simulate');
    const bakePipe = compute('terrain bake', wgsl(commonWgsl, terrainBakeWgsl), 'bake');

    // --- Render pipelines ---
    const gridLayout: GPUVertexBufferLayout = { arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] };
    const meshLayout: GPUVertexBufferLayout = {
      arrayStride: 24,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
      ],
    };
    const sceneTargets: GPUColorTargetState[] = [{ format: HDR_FORMAT }, { format: 'r32float' }];
    const depthWrite: GPUDepthStencilState = { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'greater' };
    const render = (
      label: string,
      code: string,
      buffers: GPUVertexBufferLayout[],
      targets: GPUColorTargetState[],
      depthStencil?: GPUDepthStencilState,
    ) => {
      const module = createShader(device, label, code);
      return device.createRenderPipeline({
        label,
        layout: 'auto',
        vertex: { module, entryPoint: 'vs', buffers },
        fragment: { module, entryPoint: 'fs', targets },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil,
      });
    };
    this.skyPipe = render('sky', wgsl(commonWgsl, skyWgsl, skyPassWgsl), [], sceneTargets, {
      format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always',
    });
    this.terrainPipe = render('terrain', wgsl(commonWgsl, skyWgsl, lightingWgsl, terrainWgsl), [gridLayout], sceneTargets, depthWrite);
    this.objectsPipe = render('objects', wgsl(commonWgsl, skyWgsl, lightingWgsl, objectsWgsl), [meshLayout], sceneTargets, depthWrite);
    this.causticsPipe = render('caustics', wgsl(commonWgsl, causticsWgsl), [gridLayout], [{
      format: 'rgba16float',
      blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } },
    }]);
    this.oceanPipe = render('ocean', wgsl(commonWgsl, skyWgsl, wavesWgsl, oceanWgsl), [gridLayout], [{ format: HDR_FORMAT }], depthWrite);
    this.compositePipe = render('composite', wgsl(commonWgsl, compositeWgsl), [], [{ format: ctx.format }]);

    // --- Bind groups ---
    const bg = (p: GPUComputePipeline | GPURenderPipeline, g: number, r: Parameters<typeof bindGroup>[3]) => bindGroup(device, p, g, r);
    this.initBG = bg(this.initPipe, 0, [this.spectrumUBO, arr(h0Raw), arr(waveData)]);
    this.conjugateBG = bg(this.conjugatePipe, 0, [arr(h0Raw), arr(h0)]);
    this.updateBG = bg(this.updatePipe, 0, [this.simUBO, arr(h0), arr(waveData), arr(fftA)]);
    this.fftHBG = bg(this.fftHPipe, 0, [arr(fftA), arr(fftB)]);
    this.fftVBG = bg(this.fftVPipe, 0, [arr(fftB), arr(fftA)]);
    this.assembleBG = [0, 1].map((i) =>
      bg(this.assemblePipe, 0, [this.simUBO, arr(fftA), arr(disp[1 - i], 0, 1), arr(disp[i], 0, 1), arr(deriv, 0, 1)]),
    );
    this.mipBGs = [0, 1].map((i) => {
      const groups: GPUBindGroup[] = [];
      for (const t of [disp[i], deriv]) {
        for (let level = 1; level < MIPS; level++) {
          groups.push(bg(this.mipPipe, 0, [arr(t, level - 1, 1), arr(t, level, 1)]));
        }
      }
      return groups;
    });
    this.rippleBG = [0, 1].map((i) =>
      bg(this.ripplePipe, 0, [this.rippleUBO, this.rippleViews[i], this.rippleViews[1 - i], { buffer: this.bodiesBuffer }, this.terrainView]),
    );
    this.buoyancyBG0 = bg(this.buoyancyPipe, 0, [this.frameUBO, this.oceanUBO]);
    this.buoyancyBG1 = [0, 1].map((i) =>
      bg(this.buoyancyPipe, 1, [
        this.dispViews[i], this.derivView, this.linSampler, this.clampSampler, this.terrainView, this.rippleViews[0],
        { buffer: this.bodiesBuffer }, { buffer: this.anchorsBuffer },
      ]),
    );
    this.skyBG = bg(this.skyPipe, 0, [this.frameUBO]);
    this.terrainBG0 = bg(this.terrainPipe, 0, [this.frameUBO, this.oceanUBO]);
    this.terrainBG1 = bg(this.terrainPipe, 1, [this.terrainView, this.clampSampler, this.causticView, this.linSampler]);
    this.objectsBG0 = bg(this.objectsPipe, 0, [this.frameUBO, this.oceanUBO]);
    this.objectsBG1 = bg(this.objectsPipe, 1, [{ buffer: this.bodiesBuffer }, this.causticView, this.linSampler]);
    this.causticsBG0 = bg(this.causticsPipe, 0, [this.frameUBO, this.oceanUBO]);
    this.causticsBG1 = bg(this.causticsPipe, 1, [this.derivView, this.linSampler]);
    this.oceanBG0 = bg(this.oceanPipe, 0, [this.frameUBO, this.oceanUBO]);

    // --- Geometry ---
    const grid = (n: number, label: string) => {
      const g = gridMesh(n);
      return {
        vb: vertexBuffer(device, g.vertices, label),
        ib: indexBuffer(device, g.indices, label),
        count: g.indices.length,
        format: (g.indices instanceof Uint32Array ? 'uint32' : 'uint16') as GPUIndexFormat,
      };
    };
    this.clipmap = grid(CLIPMAP_GRID, 'clipmap');
    this.terrainGrid = grid(TERRAIN_MESH, 'terrain grid');
    this.causticGrid = grid(FFT_N, 'caustic grid');
    const meshFor: Record<Kind, Mesh> = { 0: box(), 1: buoy(), 2: sphere() };
    this.meshes = ([0, 1, 2] as Kind[]).map((kind) => {
      const m = meshFor[kind];
      return {
        vb: vertexBuffer(device, m.vertices, `mesh ${kind}`),
        ib: indexBuffer(device, m.indices, `mesh ${kind}`),
        count: m.indices.length,
        first: BODIES.findIndex((b) => b.kind === kind),
        instances: BODIES.filter((b) => b.kind === kind).length,
      };
    });

    // --- One-off: bake the terrain ---
    const bakeUBO = uniformBuffer(device, 16, 'bake');
    device.queue.writeBuffer(bakeUBO, 0, new Float32Array([TERRAIN_HALF, ISLAND[0], ISLAND[1], SHORE_RADIUS]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(bakePipe);
    pass.setBindGroup(0, bg(bakePipe, 0, [bakeUBO, this.terrainView]));
    pass.dispatchWorkgroups(TERRAIN_RES / 8, TERRAIN_RES / 8);
    pass.end();
    device.queue.submit([encoder.finish()]);

    this.attachPointer();
  }

  // --- Objects ----------------------------------------------------------------

  private initBodies() {
    const data = new Float32Array(BODIES.length * BODY_FLOATS);
    BODIES.forEach((b, i) => {
      const o = i * BODY_FLOATS;
      const s = b.scale;
      data.set([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, b.x, 0, b.z, 1], o); // model
      data.set([b.x, 0, b.z, KIND_PHYSICS[b.kind].radius * s], o + 16); // pos
      data.set([0, 0, 0, 0.5], o + 20); // vel
      data.set([0, 0, 0, 1], o + 24); // rot
      data.set([b.kind, KIND_PHYSICS[b.kind].halfHeight * s, 0, 0], o + 28); // info
    });
    this.device.queue.writeBuffer(this.bodiesBuffer, 0, data);
  }

  private writeAnchors() {
    const data = new Float32Array(BODIES.length * 8);
    this.anchors.forEach((b, i) => {
      const k = KIND_PHYSICS[b.kind];
      data.set([b.x, b.z, b.scale, b.kind, k.ratio, k.halfHeight, k.radius, this.dropGeneration], i * 8);
    });
    this.device.queue.writeBuffer(this.anchorsBuffer, 0, data);
  }

  private drop() {
    this.dropGeneration++;
    this.writeAnchors();
  }

  /** Shift + drag on the water moves the nearest floating object. */
  private attachPointer() {
    const el = this.canvas;
    const waterHit = (e: PointerEvent): [number, number] | null => {
      const r = el.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * 2 - 1;
      const y = 1 - ((e.clientY - r.top) / r.height) * 2;
      const ray = this.camera.ray(x, y);
      if (ray.dir[1] >= -1e-3) return null;
      const t = -ray.origin[1] / ray.dir[1];
      return [ray.origin[0] + ray.dir[0] * t, ray.origin[2] + ray.dir[2] * t];
    };
    const down = (e: PointerEvent) => {
      if (!e.shiftKey || e.button !== 0) return;
      const hit = waterHit(e);
      if (!hit) return;
      let best = 0;
      let bestD = Infinity;
      this.anchors.forEach((a, i) => {
        const d = Math.hypot(a.x - hit[0], a.z - hit[1]);
        if (d < bestD) { bestD = d; best = i; }
      });
      this.dragging = { body: best, pointer: e.pointerId };
      el.setPointerCapture(e.pointerId);
      this.moveAnchor(best, hit);
    };
    const move = (e: PointerEvent) => {
      if (!this.dragging || this.dragging.pointer !== e.pointerId) return;
      const hit = waterHit(e);
      if (hit) this.moveAnchor(this.dragging.body, hit);
    };
    const up = (e: PointerEvent) => {
      if (this.dragging?.pointer === e.pointerId) this.dragging = null;
    };
    el.addEventListener('pointerdown', down);
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    this.disposers.push(() => {
      el.removeEventListener('pointerdown', down);
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
      el.removeEventListener('pointercancel', up);
    });
  }

  private moveAnchor(i: number, hit: [number, number]) {
    const half = RIPPLE_SIZE / 2 - 4;
    this.anchors[i].x = Math.min(RIPPLE_ORIGIN[0] + half, Math.max(RIPPLE_ORIGIN[0] - half, hit[0]));
    this.anchors[i].z = Math.min(RIPPLE_ORIGIN[1] + half, Math.max(RIPPLE_ORIGIN[1] - half, hit[1]));
    this.writeAnchors();
  }

  // --- Frame ------------------------------------------------------------------

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    for (const t of [this.hdr, this.sceneCopy, this.distance, this.depth]) t?.destroy();
    const d = this.device;
    const size = [width, height];
    this.hdr = d.createTexture({
      label: 'hdr', size, format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneCopy = d.createTexture({
      label: 'scene copy', size, format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.distance = d.createTexture({
      label: 'scene distance', size, format: 'r32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depth = d.createTexture({ label: 'depth', size, format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });

    this.oceanBG1 = [0, 1].map((i) =>
      bindGroup(d, this.oceanPipe, 1, [
        this.dispViews[i], this.derivView, this.linSampler, this.clampSampler, this.terrainView, this.rippleViews[0],
        this.sceneCopy.createView(), this.distance.createView(),
      ]),
    );
    this.compositeBG = bindGroup(d, this.compositePipe, 0, [this.frameUBO, this.hdr.createView()]);
  }

  private skySettings(): SkySettings {
    const p = this.params;
    const el = deg(p.sunElevation);
    const az = deg(p.sunAzimuth);
    const sunDir: Vec3 = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
    return { sunDir, sunIntensity: p.sunIntensity, turbidity: p.turbidity, rayleigh: p.rayleigh, mieG: p.mieG, boost: p.skyBoost };
  }

  private writeSpectrum() {
    const p = this.params;
    const b1 = ((2 * Math.PI) / LENGTHS[1]) * 6;
    const b2 = ((2 * Math.PI) / LENGTHS[2]) * 6;
    const data = new Float32Array([
      LENGTHS[0], 0.0001, b1, 0,
      LENGTHS[1], b1, b2, 0,
      LENGTHS[2], b2, 9999, 0,
      p.windSpeed, deg(p.windDirection), p.fetchKm * 1000, p.spreadBlend,
      p.swell, 500, p.shortWavesFade, p.peakEnhancement,
      p.spectrumScale, 9.81, p.seed, 0,
    ]);
    this.device.queue.writeBuffer(this.spectrumUBO, 0, data);
  }

  private writeUniforms(dt: number) {
    const p = this.params;
    const cam = this.camera;
    const sky = this.skySettings();
    const sun = sunColor(sky);
    const ambient = ambientLight(sky);

    const f = this.frameData;
    f.set(cam.viewProj, 0);
    f.set(cam.invViewProj, 16);
    f.set([...cam.eye, this.simTime], 32);
    f.set([...sky.sunDir, dt], 36);
    f.set([...sun, p.exposure], 40);
    f.set([...ambient, p.debugView], 44);
    f.set([this.width, this.height, cam.near, sunTopIntensity(sky)], 48);
    f.set([p.turbidity, p.rayleigh, p.mieG, p.skyBoost], 52);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);

    const o = this.oceanData;
    o.set([LENGTHS[0], LENGTHS[1], LENGTHS[2], p.choppiness], 0);
    o.set([BASE_SPACING, p.displacementScale, p.normalStrength, p.shallowDepth], 4);
    o.set([p.absorptionR, p.absorptionG, p.absorptionB, p.refraction], 8);
    o.set([...p.scatterColor, p.sss], 12);
    o.set([...p.foamColor, p.foamIntensity], 16);
    o.set([p.shoreAmplitude, p.shoreWavelength, p.shorePeriod, p.shoreRange], 20);
    o.set([p.roughness, p.glitter, p.shoreFoam, p.contactFoam], 24);
    o.set([RIPPLE_ORIGIN[0], RIPPLE_ORIGIN[1], RIPPLE_SIZE, p.rippleHeight], 28);
    o.set([TERRAIN_HALF, p.causticIntensity, p.causticDepth, p.fog], 32);
    o.set([1.0, 0.6, 0.15, p.foamScale], 36);
    this.device.queue.writeBuffer(this.oceanUBO, 0, o);

    this.simData.set([this.simTime, dt, p.choppiness, p.foamDecay, p.foamThreshold, p.foamSharpness, 0, 0]);
    this.device.queue.writeBuffer(this.simUBO, 0, this.simData);

    const step = dt / 2;
    const dx = RIPPLE_SIZE / RIPPLE_RES;
    const alpha = Math.min(0.45, ((p.rippleSpeed * step) / dx) ** 2);
    this.rippleData.set([RIPPLE_ORIGIN[0], RIPPLE_ORIGIN[1], RIPPLE_SIZE, alpha, Math.exp(-p.rippleDamping * step), p.rippleStrength, step, TERRAIN_HALF]);
    this.device.queue.writeBuffer(this.rippleUBO, 0, this.rippleData);
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    const dt = p.paused ? 0 : info.dt * p.timeScale;
    this.simTime += dt;
    const cur = this.current;
    this.writeUniforms(dt);

    // --- Compute: spectrum -> FFT -> textures -> ripples -> buoyancy ---
    const cp = encoder.beginComputePass({ label: 'simulation' });
    if (this.spectrumDirty) {
      this.writeSpectrum();
      cp.setPipeline(this.initPipe);
      cp.setBindGroup(0, this.initBG);
      cp.dispatchWorkgroups(FFT_N / 8, FFT_N / 8, CASCADES);
      cp.setPipeline(this.conjugatePipe);
      cp.setBindGroup(0, this.conjugateBG);
      cp.dispatchWorkgroups(FFT_N / 8, FFT_N / 8, CASCADES);
      this.spectrumDirty = false;
    }
    cp.setPipeline(this.updatePipe);
    cp.setBindGroup(0, this.updateBG);
    cp.dispatchWorkgroups(FFT_N / 8, FFT_N / 8, CASCADES);
    cp.setPipeline(this.fftHPipe);
    cp.setBindGroup(0, this.fftHBG);
    cp.dispatchWorkgroups(1, FFT_N, CASCADES * 2);
    cp.setPipeline(this.fftVPipe);
    cp.setBindGroup(0, this.fftVBG);
    cp.dispatchWorkgroups(1, FFT_N, CASCADES * 2);
    cp.setPipeline(this.assemblePipe);
    cp.setBindGroup(0, this.assembleBG[cur]);
    cp.dispatchWorkgroups(FFT_N / 8, FFT_N / 8, CASCADES);
    cp.setPipeline(this.mipPipe);
    this.mipBGs[cur].forEach((group, i) => {
      const level = (i % (MIPS - 1)) + 1;
      const size = Math.max(1, FFT_N >> level);
      cp.setBindGroup(0, group);
      cp.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8), CASCADES);
    });
    if (dt > 0) {
      cp.setPipeline(this.ripplePipe);
      for (const i of [0, 1]) {
        cp.setBindGroup(0, this.rippleBG[i]);
        cp.dispatchWorkgroups(RIPPLE_RES / 8, RIPPLE_RES / 8);
      }
    }
    cp.setPipeline(this.buoyancyPipe);
    cp.setBindGroup(0, this.buoyancyBG0);
    cp.setBindGroup(1, this.buoyancyBG1[cur]);
    cp.dispatchWorkgroups(1);
    cp.end();

    // --- Caustics ---
    const caustics = encoder.beginRenderPass({
      label: 'caustics',
      colorAttachments: [{ view: this.causticView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    caustics.setPipeline(this.causticsPipe);
    caustics.setBindGroup(0, this.causticsBG0);
    caustics.setBindGroup(1, this.causticsBG1);
    caustics.setVertexBuffer(0, this.causticGrid.vb);
    caustics.setIndexBuffer(this.causticGrid.ib, this.causticGrid.format);
    caustics.drawIndexed(this.causticGrid.count, 9);
    caustics.end();

    // --- Opaque scene: sky, terrain, floating objects ---
    const depthView = this.depth.createView();
    const hdrView = this.hdr.createView();
    const scene = encoder.beginRenderPass({
      label: 'scene',
      colorAttachments: [
        { view: hdrView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        { view: this.distance.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [1e6, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 0 },
    });
    scene.setPipeline(this.skyPipe);
    scene.setBindGroup(0, this.skyBG);
    scene.draw(3);
    scene.setPipeline(this.terrainPipe);
    scene.setBindGroup(0, this.terrainBG0);
    scene.setBindGroup(1, this.terrainBG1);
    scene.setVertexBuffer(0, this.terrainGrid.vb);
    scene.setIndexBuffer(this.terrainGrid.ib, this.terrainGrid.format);
    scene.drawIndexed(this.terrainGrid.count);
    scene.setPipeline(this.objectsPipe);
    scene.setBindGroup(0, this.objectsBG0);
    scene.setBindGroup(1, this.objectsBG1);
    for (const m of this.meshes) {
      scene.setVertexBuffer(0, m.vb);
      scene.setIndexBuffer(m.ib, 'uint16');
      scene.drawIndexed(m.count, m.instances, 0, 0, m.first);
    }
    scene.end();

    // The water pass samples the scene behind it for refraction.
    encoder.copyTextureToTexture({ texture: this.hdr }, { texture: this.sceneCopy }, [this.width, this.height]);

    // --- Water surface ---
    const water = encoder.beginRenderPass({
      label: 'ocean',
      colorAttachments: [{ view: hdrView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: depthView, depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    water.setPipeline(this.oceanPipe);
    water.setBindGroup(0, this.oceanBG0);
    water.setBindGroup(1, this.oceanBG1[cur]);
    water.setVertexBuffer(0, this.clipmap.vb);
    water.setIndexBuffer(this.clipmap.ib, this.clipmap.format);
    water.drawIndexed(this.clipmap.count, CLIPMAP_LEVELS);
    water.end();

    // --- Tone map to the canvas ---
    const post = encoder.beginRenderPass({
      label: 'composite',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    post.setPipeline(this.compositePipe);
    post.setBindGroup(0, this.compositeBG);
    post.draw(3);
    post.end();

    this.current = 1 - cur;
  }

  hud() {
    const p = this.params;
    return `wind ${p.windSpeed.toFixed(1)} m/s  t=${this.simTime.toFixed(1)}s\nshift+drag: move objects · drag: orbit · WASD/QE: fly`;
  }

  destroy() {
    this.disposers.forEach((d) => d());
    for (const t of [...this.textures, this.hdr, this.sceneCopy, this.distance, this.depth]) t?.destroy();
    for (const b of [this.frameUBO, this.oceanUBO, this.spectrumUBO, this.simUBO, this.rippleUBO, this.bodiesBuffer, this.anchorsBuffer]) b.destroy();
  }
}

export const oceanEntry: DemoEntry = {
  id: 'ocean',
  title: 'Deep-water ocean',
  tags: ['compute FFT', 'vertex displacement', 'BRDF', 'screen-space', 'caustics'],
  info: `
    <h2>Deep-water ocean</h2>
    <ul>
      <li>JONSWAP spectrum, 3 cascades, 256² GPU FFT in compute shaders</li>
      <li>Geometry clipmap with morphing, choppy displacement</li>
      <li>Fresnel sky reflection, GGX sun + glitter, SSS</li>
      <li>Screen-space refraction, Beer–Lambert depth colour</li>
      <li>Jacobian whitecaps, shoreline and contact foam</li>
      <li>Photon-area caustics on the seabed</li>
      <li>GPU buoyancy + wave-equation ripples, shoreline Gerstner waves</li>
    </ul>
    <p><kbd>drag</kbd> orbit · <kbd>right drag</kbd> pan · <kbd>wheel</kbd> zoom<br>
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move · <kbd>Q</kbd><kbd>E</kbd> down/up<br>
    <kbd>shift</kbd>+<kbd>drag</kbd> on water: move an object</p>`,
  create: (ctx) => new OceanDemo(ctx),
};
