import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, uniformBuffer } from '../../core/gpu';
import { deg, type Vec3 } from '../../core/math';
import { ambientLight, sunColor, sunTopIntensity, type SkySettings } from '../../shared/sky';
import skyWgsl from '../../shared/sky.wgsl?raw';
import { DEFAULTS, VIEWS, buildGui, type SdfParams, type ViewPreset } from './params';

import worldWgsl from './shaders/world.wgsl?raw';

const FLY_SECONDS = 1.6;
const STAT_COUNT = 8;

/** Per-pixel averages from the shader's cost counters (every 16th pixel is sampled). */
export interface CostStats {
  pixels: number;
  primarySteps: number;
  mapCalls: number;
  groups: number;
  octaves: number;
  shadowSteps: number;
  reflectionSteps: number;
  maxedOut: number;
}

type CameraState = Omit<ViewPreset, 'follow'>;

const ease = (u: number) => u * u * (3 - 2 * u);

/**
 * A whole world from signed distance functions in one fullscreen fragment
 * shader. The CPU side is deliberately thin: it fills one uniform block
 * (camera, sun, time, GUI values) and issues a single draw of 3 vertices.
 * Everything else, including the character's skeleton, is derived in the
 * shader per pixel.
 *
 * Resolution scale is applied to the canvas backing store and the browser
 * upscales it: the constraint is one pass, so there is no upsampling pass.
 */
class SdfDemo implements Demo {
  private device: GPUDevice;
  private canvas: HTMLCanvasElement;
  private camera: DemoContext['camera'];
  private params: SdfParams = structuredClone(DEFAULTS);
  private pipeline: GPURenderPipeline;
  private group: GPUBindGroup;
  private frameData = new Float32Array(64);
  private frameUBO: GPUBuffer;
  private statsBuffer: GPUBuffer;
  private readback: GPUBuffer;
  private readPending = false;

  private fullWidth = 1;
  private fullHeight = 1;
  private width = 1;
  private height = 1;
  private simTime = 0;
  private follow = false;
  private fly: { from: CameraState; to: ViewPreset; u: number } | null = null;

  /** Latest cost measurement (per pixel), for the HUD and scripted measurements. */
  stats: CostStats | null = null;

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.canvas = ctx.canvas;
    this.camera = ctx.camera;
    Object.assign(this.camera, { minEyeHeight: 0.35, minPitch: -0.6, maxPitch: 1.45, minDistance: 0.5 });
    this.camera.fovY = deg(55);
    this.setView('Temple entrance', true);
    buildGui(ctx.gui, this.params, {
      view: (name) => this.setView(name),
      resized: () => this.applyScale(),
    });

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'sdf frame');
    this.statsBuffer = device.createBuffer({
      label: 'sdf cost counters', size: STAT_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.readback = device.createBuffer({ label: 'sdf cost readback', size: STAT_COUNT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const module = createShader(device, 'sdf world', `${worldWgsl}\n${skyWgsl}`);
    this.pipeline = device.createRenderPipeline({
      label: 'sdf world', layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });
    this.group = bindGroup(device, this.pipeline, 0, [this.frameUBO, this.statsBuffer], 'sdf world');
  }

  /** Jumps (or flies, over FLY_SECONDS) to a named viewpoint. */
  setView(name: string, instant = false) {
    const v = VIEWS[name];
    if (!v) return;
    this.follow = !!v.follow;
    const to: ViewPreset = { ...v, target: v.follow ? this.characterTarget() : [...v.target] };
    const c = this.camera;
    if (instant) {
      this.fly = null;
      Object.assign(c, { target: [...to.target], yaw: to.yaw, pitch: to.pitch, distance: to.distance });
      return;
    }
    // Take the short way round in yaw.
    let yaw = c.yaw;
    while (to.yaw - yaw > Math.PI) yaw += 2 * Math.PI;
    while (to.yaw - yaw < -Math.PI) yaw -= 2 * Math.PI;
    this.fly = { from: { target: [...c.target], yaw, pitch: c.pitch, distance: c.distance }, to, u: 0 };
  }

  /** CPU mirror of setupCharacter() in the shader: where the walker is now. */
  private characterTarget(): Vec3 {
    const radius = 4.8;
    const th = (this.simTime * this.params.walkSpeed) / radius + 1;
    return [radius * Math.cos(th), 2.5, 21.4 + radius * Math.sin(th)];
  }

  resize(width: number, height: number) {
    this.fullWidth = width;
    this.fullHeight = height;
    this.applyScale();
  }

  private applyScale() {
    const s = this.params.resolutionScale;
    this.width = Math.max(1, Math.round(this.fullWidth * s));
    this.height = Math.max(1, Math.round(this.fullHeight * s));
    this.canvas.width = this.width;
    this.canvas.height = this.height;
  }

  private skySettings(): SkySettings {
    const p = this.params;
    const el = deg(p.sunElevation);
    const az = deg(p.sunAzimuth);
    const sunDir: Vec3 = [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
    return { sunDir, sunIntensity: p.sunIntensity, turbidity: p.turbidity, rayleigh: 1, mieG: 0.8, boost: 2.2 };
  }

  private updateCamera(dt: number) {
    const c = this.camera;
    if (this.fly) {
      this.fly.u = Math.min(1, this.fly.u + dt / FLY_SECONDS);
      const u = ease(this.fly.u);
      const { from, to } = this.fly;
      const target = this.follow ? this.characterTarget() : to.target;
      c.target = [0, 1, 2].map((i) => from.target[i] + (target[i] - from.target[i]) * u) as Vec3;
      c.yaw = from.yaw + (to.yaw - from.yaw) * u;
      c.pitch = from.pitch + (to.pitch - from.pitch) * u;
      c.distance = from.distance + (to.distance - from.distance) * u;
      if (this.fly.u >= 1) this.fly = null;
    } else if (this.follow) {
      c.target = this.characterTarget();
    }
    // Recompute the matrices now that the target may have moved (dt = 0: no key input).
    c.update(0, this.width / this.height);
  }

  private writeUniforms(measure: boolean) {
    const p = this.params;
    const cam = this.camera;
    const sky = this.skySettings();
    const f = this.frameData;
    f.set(cam.invViewProj, 0);
    f.set([...cam.eye, this.simTime], 16);
    f.set([...sky.sunDir, p.exposure], 20);
    f.set([...sunColor(sky), sunTopIntensity(sky)], 24);
    f.set([...ambientLight(sky), p.debugView], 28);
    f.set([this.width, this.height, (2 * Math.tan(cam.fovY / 2)) / this.height, measure ? 1 : 0], 32);
    f.set([sky.turbidity, sky.rayleigh, sky.mieG, sky.boost], 36);
    f.set([p.maxSteps, p.omega, p.epsilon, p.maxDistance], 40);
    f.set([p.shadowSteps, p.penumbra, p.aoStrength, p.bounces], 44);
    f.set([+p.shadows, +p.ao, +p.lod, +p.bounds], 48);
    f.set([p.fog, p.fogFalloff, p.sss, p.waterMurk], 52);
    f.set([p.sliceAxis, p.sliceOffset, p.sliceSpacing, p.supersample ? 4 : 1], 56);
    f.set([p.walkSpeed, p.breathing, p.spin, p.warp], 60);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    if (!p.paused) this.simTime += info.dt * p.timeScale;
    this.updateCamera(info.dt);

    const measure = !this.readPending;
    if (measure) this.device.queue.writeBuffer(this.statsBuffer, 0, new Uint32Array(STAT_COUNT));
    this.writeUniforms(measure);

    const pass = encoder.beginRenderPass({
      label: 'sdf world',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.draw(3);
    pass.end();

    if (measure) {
      encoder.copyBufferToBuffer(this.statsBuffer, 0, this.readback, 0, STAT_COUNT * 4);
      this.readPending = true;
      this.device.queue.onSubmittedWorkDone().then(() =>
        this.readback.mapAsync(GPUMapMode.READ).then(() => {
          const s = new Uint32Array(this.readback.getMappedRange().slice(0));
          this.readback.unmap();
          this.readPending = false;
          const n = Math.max(1, s[0]);
          const samples = p.supersample ? 4 : 1;
          this.stats = {
            pixels: s[0],
            primarySteps: s[1] / n,
            mapCalls: s[2] / n,
            groups: s[3] / n,
            octaves: s[4] / n,
            shadowSteps: s[5] / n,
            reflectionSteps: s[6] / n,
            maxedOut: s[7] / (n * samples),
          };
        }).catch(() => { this.readPending = false; }),
      );
    }
  }

  hud() {
    const s = this.stats;
    const cost = s
      ? `steps/px ${s.primarySteps.toFixed(1)} primary + ${s.shadowSteps.toFixed(1)} shadow + ${s.reflectionSteps.toFixed(1)} reflection\n` +
        `map calls/px ${s.mapCalls.toFixed(0)} · fBm octaves/px ${s.octaves.toFixed(0)} · groups/px ${s.groups.toFixed(0)} · out of steps ${(s.maxedOut * 100).toFixed(2)}%`
      : 'measuring…';
    return `internal ${this.width}x${this.height} · ${cost}\ndrag: orbit · WASD/QE: move · wheel: zoom`;
  }

  destroy() {
    // Give the canvas back at full resolution for the next demo.
    this.canvas.width = this.fullWidth;
    this.canvas.height = this.fullHeight;
    [this.frameUBO, this.statsBuffer, this.readback].forEach((b) => b.destroy());
  }
}

export const sdfEntry: DemoEntry = {
  id: 'sdf',
  title: 'SDF world',
  tags: ['sphere tracing', 'SDF modelling', 'soft shadows', 'single pass'],
  info: `
    <h2>Signed-distance-field world</h2>
    <ul>
      <li>One fullscreen fragment shader: no meshes, textures, compute or extra passes</li>
      <li>Smooth booleans with material blending, rounding, onion, elongation</li>
      <li>Infinite / limited / polar repetition, mirroring, twist, bend, domain warping</li>
      <li>Warped fBm terrain with a Lipschitz-safe step, river, cliffs</li>
      <li>Temple, rotunda, aqueduct, bridge, a walking character, a floating gyroid sculpture</li>
      <li>Over-relaxed sphere tracing, cone epsilon, bounding volumes, fBm LOD</li>
      <li>Penumbra soft shadows, SDF AO, traced reflections, sky and height fog</li>
      <li>Debug: step and cost heatmaps, normals, AO, shadows, materials, distance slice</li>
    </ul>
    <p><kbd>drag</kbd> orbit · <kbd>wheel</kbd> zoom · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move<br>
    Pick a <em>Viewpoint</em> to fly between the landmarks.</p>`,
  create: (ctx) => new SdfDemo(ctx),
};
