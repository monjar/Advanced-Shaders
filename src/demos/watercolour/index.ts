import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, indexBuffer, uniformBuffer } from '../../core/gpu';
import { deg, mat4, vec3, type Mat4, type Vec3 } from '../../core/math';
import { DEFAULTS, buildGui, type PaintParams } from './params';
import { buildScene, terrainHeight, VERTEX_STRIDE } from './scene';

import bleedWgsl from './shaders/bleed.wgsl?raw';
import blitWgsl from './shaders/blit.wgsl?raw';
import coherenceWgsl from './shaders/coherence.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import edgesWgsl from './shaders/edges.wgsl?raw';
import gbufferWgsl from './shaders/gbuffer.wgsl?raw';
import noiseGenWgsl from './shaders/noise_gen.wgsl?raw';
import noiseWgsl from './shaders/noise.wgsl?raw';
import shadowWgsl from './shaders/shadow.wgsl?raw';
import smearWgsl from './shaders/smear.wgsl?raw';

const NOISE_SIZE = 64;
const NOISE_CELLS = 16;
const SHADOW_SIZE = 2048;
const wgsl = (...parts: string[]) => parts.join('\n');

/**
 * Dynamic canvas (after Cunzi et al., "Dynamic Canvas for Non-Photorealistic
 * Walkthroughs", 2003). The paper is not glued to the screen: every frame we
 * estimate how the visible scene moved on screen and move the paper with it.
 * A coarse grid of view rays is cast on the CPU against the terrain height
 * field (sky rays are points at infinity), the hit points are reprojected into
 * the previous frame, and a least-squares 2D similarity (uniform zoom +
 * shift) maps current to previous positions. Composing it into the paper
 * transform keeps the grain riding on the dominant surface instead of
 * sliding over it. Zoom is unbounded, so the grain is evaluated at two
 * octaves blended by the fractional zoom level (infinite zoom).
 */
class DynamicCanvas {
  scale = 1;
  offset: [number, number] = [0, 0];

  update(
    prevViewProj: Mat4, invViewProj: Mat4, eye: Vec3,
    width: number, height: number, ground: (x: number, z: number) => number,
  ) {
    const cx = width / 2;
    const cy = height / 2;
    const toScreen = (m: Mat4, p: Vec3): [number, number] | null => {
      const c = mat4.transformPoint(m, [p[0], p[1], p[2], 1]);
      if (c[3] <= 1e-4) return null;
      return [(c[0] / c[3] * 0.5 + 0.5) * width, (0.5 - c[1] / c[3] * 0.5) * height];
    };
    const now: [number, number][] = [];
    const prev: [number, number][] = [];
    const nx = 12;
    const ny = 7;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const sx = (i + 0.5) / nx;
        const sy = (j + 0.5) / ny;
        const q = mat4.transformPoint(invViewProj, [sx * 2 - 1, 1 - sy * 2, 1, 1]);
        const dir = vec3.normalize(vec3.sub([q[0] / q[3], q[1] / q[3], q[2] / q[3]], eye));
        const hit = raycastGround(eye, dir, ground) ?? vec3.add(eye, vec3.scale(dir, 1e5));
        const a = toScreen(prevViewProj, hit);
        if (!a) continue;
        now.push([sx * width - cx, sy * height - cy]);
        prev.push([a[0] - cx, a[1] - cy]);
      }
    }
    if (now.length < 8) return;
    // Fit prev ≈ alpha * now + beta (both relative to the screen centre).
    const n = now.length;
    const mr = [0, 0];
    const md = [0, 0];
    for (let k = 0; k < n; k++) {
      mr[0] += now[k][0] / n; mr[1] += now[k][1] / n;
      md[0] += prev[k][0] / n; md[1] += prev[k][1] / n;
    }
    let num = 0;
    let den = 0;
    for (let k = 0; k < n; k++) {
      const rx = now[k][0] - mr[0], ry = now[k][1] - mr[1];
      num += rx * (prev[k][0] - md[0]) + ry * (prev[k][1] - md[1]);
      den += rx * rx + ry * ry;
    }
    if (den < 1e-6) return;
    const alpha = Math.min(1.5, Math.max(0.67, num / den));
    const beta = [md[0] - alpha * mr[0], md[1] - alpha * mr[1]];
    // Paper coordinate u(x) = (x - c) / s + o must satisfy u_now(x) = u_prev(prev(x)).
    this.offset[0] += beta[0] / this.scale;
    this.offset[1] += beta[1] / this.scale;
    this.scale /= alpha;
  }

  /** Two octaves of paper lookup parameters: (scale, offset.xy, blend weight). */
  octaves(grainPx: number): [number[], number[]] {
    const level = Math.log2(this.scale);
    const l0 = Math.floor(level);
    const w = level - l0;
    const out = [0, 1].map((i) => {
      const m = 2 ** (l0 + i);
      const period = grainPx * NOISE_CELLS;
      const fx = (this.offset[0] * m) / period;
      const fy = (this.offset[1] * m) / period;
      return [m / (this.scale * period), fx - Math.floor(fx), fy - Math.floor(fy)];
    });
    return [[...out[0], w], [...out[1], 0]];
  }
}

/** Ray/height-field intersection by adaptive stepping and bisection. */
function raycastGround(eye: Vec3, dir: Vec3, ground: (x: number, z: number) => number): Vec3 | null {
  let t = 0.5;
  let prevT = 0;
  for (let i = 0; i < 96 && t < 600; i++) {
    const p = vec3.add(eye, vec3.scale(dir, t));
    const above = p[1] - ground(p[0], p[2]);
    if (above < 0) {
      let lo = prevT;
      let hi = t;
      for (let k = 0; k < 8; k++) {
        const mid = (lo + hi) / 2;
        const m = vec3.add(eye, vec3.scale(dir, mid));
        if (m[1] - ground(m[0], m[2]) < 0) hi = mid; else lo = mid;
      }
      return vec3.add(eye, vec3.scale(dir, hi));
    }
    prevT = t;
    t += Math.max(0.4, above * 0.6, t * 0.02);
  }
  return null;
}

class WatercolourDemo implements Demo {
  private device: GPUDevice;
  private camera: DemoContext['camera'];
  private params: PaintParams = structuredClone(DEFAULTS);
  private canvas = new DynamicCanvas();
  private prevViewProj: Mat4 | null = null;
  private width = 1;
  private height = 1;
  private frameIndex = 0;
  private error: number | null = null;
  private readPending = false;

  private frameData = new Float32Array(84);
  private paintData = new Float32Array(40);
  private frameUBO: GPUBuffer;
  private paintUBO: GPUBuffer;
  private totals: GPUBuffer;
  private readback: GPUBuffer;

  private vertexBuffer: GPUBuffer;
  private indexBuffer: GPUBuffer;
  private indexCount: number;

  private noise: GPUTexture;
  private shadowMap: GPUTexture;
  private targets: GPUTexture[] = [];
  private views!: Record<string, GPUTextureView>;
  private finalTex!: GPUTexture;
  private prevFinal!: GPUTexture;

  private noiseSampler: GPUSampler;
  private linearClamp: GPUSampler;
  private shadowSampler: GPUSampler;

  private shadowPipe: GPURenderPipeline;
  private skyPipe: GPURenderPipeline;
  private scenePipe: GPURenderPipeline;
  private edgesPipe: GPUComputePipeline;
  private bleedHPipe: GPUComputePipeline;
  private bleedVPipe: GPUComputePipeline;
  private smearPipe: GPUComputePipeline;
  private compositePipe: GPUComputePipeline;
  private coherencePipe: GPUComputePipeline;
  private blitPipe: GPURenderPipeline;
  private groups!: Record<string, GPUBindGroup[]>;

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.camera = ctx.camera;
    Object.assign(this.camera, { target: [2, 2.5, 4] as Vec3, distance: 38, yaw: 0.5, pitch: 0.3, minEyeHeight: 1, maxPitch: 1.3 });
    this.camera.fovY = deg(50);
    this.camera.near = 0.3;
    buildGui(ctx.gui, this.params);

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'frame');
    this.paintUBO = uniformBuffer(device, this.paintData.byteLength, 'paint');
    this.totals = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const scene = buildScene();
    this.vertexBuffer = device.createBuffer({ label: 'scene', size: scene.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.vertexBuffer, 0, scene.vertices);
    this.indexBuffer = indexBuffer(device, scene.indices, 'scene');
    this.indexCount = scene.count;

    this.noise = device.createTexture({
      label: 'noise', size: [NOISE_SIZE, NOISE_SIZE, NOISE_SIZE], dimension: '3d', format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowMap = device.createTexture({
      label: 'shadow map', size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.noiseSampler = device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat', magFilter: 'linear', minFilter: 'linear',
    });
    this.linearClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.shadowSampler = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: VERTEX_STRIDE,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'unorm8x4' },
        { shaderLocation: 3, offset: 28, format: 'float32' },
      ],
    };
    const gTargets: GPUColorTargetState[] = [
      { format: 'rgba16float' }, { format: 'rgba16float' }, { format: 'rg32float' }, { format: 'rgba16float' },
    ];
    const gModule = createShader(device, 'gbuffer', wgsl(commonWgsl, noiseWgsl, gbufferWgsl));
    this.scenePipe = device.createRenderPipeline({
      label: 'gbuffer', layout: 'auto',
      vertex: { module: gModule, entryPoint: 'vs', buffers: [vertexLayout] },
      fragment: { module: gModule, entryPoint: 'fs', targets: gTargets },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    this.skyPipe = device.createRenderPipeline({
      label: 'sky wash', layout: 'auto',
      vertex: { module: gModule, entryPoint: 'skyVs' },
      fragment: { module: gModule, entryPoint: 'skyFs', targets: gTargets },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' },
    });
    const shadowModule = createShader(device, 'shadow', wgsl(commonWgsl, shadowWgsl));
    this.shadowPipe = device.createRenderPipeline({
      label: 'shadow', layout: 'auto',
      vertex: { module: shadowModule, entryPoint: 'vs', buffers: [{ arrayStride: VERTEX_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
    });
    const compute = (label: string, code: string, entryPoint: string, constants?: Record<string, number>) =>
      device.createComputePipeline({ label, layout: 'auto', compute: { module: createShader(device, label, code), entryPoint, constants } });
    this.edgesPipe = compute('edges', wgsl(commonWgsl, edgesWgsl), 'edges');
    this.bleedHPipe = compute('bleed H', wgsl(commonWgsl, bleedWgsl), 'bleed', { HORIZONTAL: 1 });
    this.bleedVPipe = compute('bleed V', wgsl(commonWgsl, bleedWgsl), 'bleed', { HORIZONTAL: 0 });
    this.smearPipe = compute('smear', wgsl(commonWgsl, smearWgsl), 'smear');
    this.compositePipe = compute('composite', wgsl(commonWgsl, noiseWgsl, compositeWgsl), 'composite');
    this.coherencePipe = compute('coherence', wgsl(commonWgsl, coherenceWgsl), 'measure');
    const blitModule = createShader(device, 'blit', blitWgsl);
    this.blitPipe = device.createRenderPipeline({
      label: 'blit', layout: 'auto',
      vertex: { module: blitModule, entryPoint: 'vs' },
      fragment: { module: blitModule, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });

    // One-off: bake the noise texture.
    const gen = compute('noise', noiseGenWgsl, 'generate');
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gen);
    pass.setBindGroup(0, bindGroup(device, gen, 0, [this.noise.createView()]));
    pass.dispatchWorkgroups(NOISE_SIZE / 4, NOISE_SIZE / 4, NOISE_SIZE / 4);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    const d = this.device;
    this.targets.forEach((t) => t.destroy());
    const make = (label: string, format: GPUTextureFormat, usage: number) =>
      d.createTexture({ label, size: [width, height], format, usage });
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const ST = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    const t = {
      pigment: make('pigment', 'rgba16float', RT),
      surface: make('surface', 'rgba16float', RT),
      depthIds: make('depth + ids', 'rg32float', RT),
      stroke: make('stroke', 'rgba16float', RT),
      depth: make('depth', 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT),
      edges: make('edges', 'rgba16float', ST),
      bleedTmp: make('bleed tmp', 'rgba16float', ST),
      bled: make('bled', 'rgba16float', ST),
      painted: make('painted', 'rgba16float', ST),
    };
    this.finalTex = make('final', 'rgba8unorm', ST | GPUTextureUsage.COPY_SRC);
    this.prevFinal = make('previous final', 'rgba8unorm', GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST);
    this.targets = [...Object.values(t), this.finalTex, this.prevFinal];
    const v = Object.fromEntries(Object.entries(t).map(([k, tex]) => [k, tex.createView()])) as Record<keyof typeof t, GPUTextureView>;
    const finalView = this.finalTex.createView();
    this.views = { ...v, final: finalView };

    const bg = (p: GPUComputePipeline | GPURenderPipeline, g: number, r: Parameters<typeof bindGroup>[3]) => bindGroup(d, p, g, r);
    const uniforms = (p: GPUComputePipeline | GPURenderPipeline) => bg(p, 0, [this.frameUBO, this.paintUBO]);
    this.groups = {
      shadow: [bg(this.shadowPipe, 0, [this.frameUBO])],
      scene: [uniforms(this.scenePipe), bg(this.scenePipe, 1, [this.noise.createView(), this.noiseSampler, this.shadowMap.createView(), this.shadowSampler])],
      sky: [uniforms(this.skyPipe), bg(this.skyPipe, 1, [this.noise.createView(), this.noiseSampler])],
      edges: [bg(this.edgesPipe, 0, [null, this.paintUBO]), bg(this.edgesPipe, 1, [v.surface, v.depthIds, v.edges])],
      bleedH: [uniforms(this.bleedHPipe), bg(this.bleedHPipe, 1, [v.pigment, v.surface, v.bleedTmp])],
      bleedV: [uniforms(this.bleedVPipe), bg(this.bleedVPipe, 1, [v.bleedTmp, v.surface, v.bled])],
      smear: [uniforms(this.smearPipe), bg(this.smearPipe, 1, [v.bled, v.stroke, v.depthIds, v.painted])],
      composite: [uniforms(this.compositePipe), bg(this.compositePipe, 1, [
        v.painted, v.edges, v.depthIds, v.surface, v.stroke, v.pigment, this.linearClamp, this.noise.createView(), this.noiseSampler, finalView,
      ])],
      coherence: [bg(this.coherencePipe, 0, [this.frameUBO, null]), bg(this.coherencePipe, 1, [finalView, this.prevFinal.createView(), v.depthIds, { buffer: this.totals }, this.linearClamp])],
      blit: [bg(this.blitPipe, 0, [finalView])],
    };
    this.prevViewProj = null;
  }

  private sunDir(): Vec3 {
    const el = deg(this.params.sunElevation);
    const az = deg(this.params.sunAzimuth);
    return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
  }

  private writeUniforms(info: FrameInfo) {
    const p = this.params;
    const cam = this.camera;
    const sun = this.sunDir();

    // Dynamic canvas: move the paper with the visible scene.
    if (this.prevViewProj && p.paperMode !== 1) {
      this.canvas.update(this.prevViewProj, cam.invViewProj, cam.eye, this.width, this.height, terrainHeight);
    }
    const pxScale = this.height / 1080;
    const [paper0, paper1] = this.canvas.octaves(p.grainSize * pxScale);

    const lightView = mat4.lookAt(vec3.scale(sun, 200), [0, 0, 0], [0, 1, 0]);
    const lightViewProj = mat4.multiply(mat4.orthographic(-95, 95, -95, 95, 1, 420), lightView);

    const f = this.frameData;
    f.set(cam.viewProj, 0);
    f.set(cam.invViewProj, 16);
    f.set(this.prevViewProj ?? cam.viewProj, 32);
    f.set(lightViewProj, 48);
    f.set([...cam.eye, info.time], 64);
    f.set([...sun, (2 * Math.tan(cam.fovY / 2)) / this.height], 68);
    f.set([this.width, this.height, pxScale, this.frameIndex], 72);
    f.set(paper0, 76);
    f.set(paper1, 80);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);

    const q = this.paintData;
    q.set([p.lightThreshold, p.shadowThreshold, p.bandSoftness, p.bandJitter], 0);
    q.set([p.lightGlaze, p.midGlaze, p.shadowGlaze, p.turbulence], 4);
    q.set([...p.shadowTint, p.shadowTintAmount], 8);
    q.set([p.streaks, p.streakSize, p.smear, p.directionJitter], 12);
    q.set([p.bleedRadius, p.wetAreas, p.castShadows, p.aerial], 16);
    q.set([p.edgeDarkening, p.ink, p.inkWidth, p.inkBreakup], 20);
    q.set([p.granulation, p.dryBrush, p.wobble, p.relief], 24);
    q.set([...p.paperColor, p.grainSize], 28);
    q.set([...p.inkColor, p.creaseSensitivity], 32);
    q.set([p.noiseSpace, p.paperMode, p.debugView, 0], 36);
    this.device.queue.writeBuffer(this.paintUBO, 0, q);
    this.device.queue.writeBuffer(this.totals, 0, new Uint32Array(4));
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    if (p.autoOrbit) this.camera.yaw += p.orbitSpeed * info.dt;
    this.writeUniforms(info);
    const v = this.views;
    const g = this.groups;

    const shadow = encoder.beginRenderPass({
      label: 'shadow', colorAttachments: [],
      depthStencilAttachment: { view: this.shadowMap.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    shadow.setPipeline(this.shadowPipe);
    shadow.setBindGroup(0, g.shadow[0]);
    shadow.setVertexBuffer(0, this.vertexBuffer);
    shadow.setIndexBuffer(this.indexBuffer, 'uint32');
    shadow.drawIndexed(this.indexCount);
    shadow.end();

    const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] });
    const scene = encoder.beginRenderPass({
      label: 'gbuffer',
      colorAttachments: [clear(v.pigment), clear(v.surface), clear(v.depthIds), clear(v.stroke)],
      depthStencilAttachment: { view: v.depth, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 0 },
    });
    scene.setPipeline(this.skyPipe);
    g.sky.forEach((b, i) => scene.setBindGroup(i, b));
    scene.draw(3);
    scene.setPipeline(this.scenePipe);
    g.scene.forEach((b, i) => scene.setBindGroup(i, b));
    scene.setVertexBuffer(0, this.vertexBuffer);
    scene.setIndexBuffer(this.indexBuffer, 'uint32');
    scene.drawIndexed(this.indexCount);
    scene.end();

    const wx = Math.ceil(this.width / 8);
    const wy = Math.ceil(this.height / 8);
    const cp = encoder.beginComputePass({ label: 'paint' });
    for (const [pipe, groups] of [
      [this.edgesPipe, g.edges], [this.bleedHPipe, g.bleedH], [this.bleedVPipe, g.bleedV],
      [this.smearPipe, g.smear], [this.compositePipe, g.composite],
    ] as const) {
      cp.setPipeline(pipe);
      groups.forEach((b, i) => cp.setBindGroup(i, b));
      cp.dispatchWorkgroups(wx, wy);
    }
    const measuring = p.measure && this.frameIndex > 0;
    if (measuring) {
      cp.setPipeline(this.coherencePipe);
      g.coherence.forEach((b, i) => cp.setBindGroup(i, b));
      cp.dispatchWorkgroups(Math.ceil(this.width / 32), Math.ceil(this.height / 32));
    }
    cp.end();
    encoder.copyTextureToTexture({ texture: this.finalTex }, { texture: this.prevFinal }, [this.width, this.height]);

    const blit = encoder.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    blit.setPipeline(this.blitPipe);
    blit.setBindGroup(0, g.blit[0]);
    blit.draw(3);
    blit.end();

    // Read the coherence totals back a few times per second.
    if (measuring && !this.readPending && this.frameIndex % 10 === 0) {
      encoder.copyBufferToBuffer(this.totals, 0, this.readback, 0, 16);
      this.readPending = true;
      this.device.queue.onSubmittedWorkDone().then(() =>
        this.readback.mapAsync(GPUMapMode.READ).then(() => {
          const [sum, count] = new Uint32Array(this.readback.getMappedRange());
          this.error = count > 0 ? (sum / count / 10000) * 255 : null;
          this.readback.unmap();
          this.readPending = false;
        }),
      );
    }

    this.prevViewProj = new Float32Array(this.camera.viewProj);
    this.frameIndex++;
  }

  hud() {
    const err = this.error === null ? '–' : this.error.toFixed(2);
    return `reprojection error ${err} /255 (lower = more coherent)\ndrag: orbit · WASD/QE: move · wheel: zoom`;
  }

  destroy() {
    [...this.targets, this.noise, this.shadowMap].forEach((t) => t.destroy());
    [this.frameUBO, this.paintUBO, this.totals, this.readback, this.vertexBuffer, this.indexBuffer].forEach((b) => b.destroy());
  }
}

export const watercolourEntry: DemoEntry = {
  id: 'watercolour',
  title: 'Watercolour',
  tags: ['NPR', 'G-buffer stylisation', 'temporal coherence'],
  info: `
    <h2>Watercolour (and pen &amp; wash)</h2>
    <ul>
      <li>Lighting abstracted into three glazes with irregular edges and cool shadows</li>
      <li>World-space brush direction per surface, streaks and smear along it</li>
      <li>Wet-in-wet colour bleeding in absorbance space</li>
      <li>Edge darkening, granulation, dry brush, paper wobble</li>
      <li>Ink lines from depth, normal and object-id edges</li>
      <li>Coherence: depth-adaptive world-space noise and a dynamic canvas, with a live reprojection-error meter</li>
    </ul>
    <p><kbd>drag</kbd> orbit · <kbd>wheel</kbd> zoom · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move<br>
    Compare noise and paper modes under <em>Temporal coherence</em> with auto orbit on.</p>`,
  create: (ctx) => new WatercolourDemo(ctx),
};
