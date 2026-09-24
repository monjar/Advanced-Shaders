import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, indexBuffer, uniformBuffer } from '../../core/gpu';
import { deg, vec3, type Vec3 } from '../../core/math';
import { Atmosphere, EARTH_ATMOSPHERE, atmosphereWgsl } from '../../shared/atmosphere';
import { PlanetCamera } from '../../shared/planet-camera';
import { DEFAULTS, PRESETS, buildGui, type PlanetParams } from './params';
import { ChunkTree, FACES, GRID, GRID_N, VERTS_PER_CHUNK, cubePoint, nodeSize, type Draw, type QNode } from './quadtree';
import {
  CHANNELS, PLANET_RADIUS, SLOT_BASE, SLOT_COUNT, SLOT_LEVEL, TERRAIN_SLOTS, geometryWeight, slotStaticData,
  terrainConstantsWgsl, terrainHeight, writeSlotTable, type TerrainShape,
} from './terrain';

import chunkWgsl from './shaders/chunk.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import gbufferWgsl from './shaders/gbuffer.wgsl?raw';
import shadeWgsl from './shaders/shade.wgsl?raw';
import shadowWgsl from './shaders/shadow.wgsl?raw';
import terrainWgsl from './shaders/terrain.wgsl?raw';

const POOL = 1536;
const MAX_BAKE = 256;
const GEN_STRIDE = 80 + TERRAIN_SLOTS * 32; // bytes, see ChunkGen in chunk.wgsl
const DRAW_STRIDE = 32;
const MAX_DRAWS = POOL * 3; // camera + two shadow cascades
const SHADOW_SIZE = 2048;

const wgsl = (...parts: string[]) => parts.join('\n');

/** Grid + skirt triangles. Quads use the main diagonal where (i + j) is even (see morphTarget in chunk.wgsl). */
function chunkIndices(): Uint16Array {
  const idx: number[] = [];
  const v = (i: number, j: number) => j * GRID + i;
  for (let j = 0; j < GRID_N; j++) {
    for (let i = 0; i < GRID_N; i++) {
      const a = v(i, j), b = v(i + 1, j), c = v(i, j + 1), d = v(i + 1, j + 1);
      if ((i + j) % 2 === 0) idx.push(a, b, d, a, d, c);
      else idx.push(a, b, c, b, d, c);
    }
  }
  const edge = (e: number, s: number) => (e === 0 ? v(s, 0) : e === 1 ? v(GRID_N, s) : e === 2 ? v(s, GRID_N) : v(0, s));
  for (let e = 0; e < 4; e++) {
    for (let s = 0; s < GRID_N; s++) {
      const g0 = edge(e, s), g1 = edge(e, s + 1);
      const s0 = GRID * GRID + e * GRID + s, s1 = s0 + 1;
      idx.push(g0, g1, s1, g0, s1, s0);
    }
  }
  return new Uint16Array(idx);
}

class PlanetDemo implements Demo {
  private device: GPUDevice;
  private orbit: DemoContext['camera'];
  params: PlanetParams = structuredClone(DEFAULTS);
  cam: PlanetCamera;
  atmo: Atmosphere;
  tree: ChunkTree;

  private width = 1;
  private height = 1;
  private sizeDirty = true;
  private bakedThisFrame = 0;
  private sunLon = DEFAULTS.sunLon;
  private terrainKey = '';

  private frameData = new Float32Array(104);
  private frameUBO: GPUBuffer;
  private slotUBO: GPUBuffer;
  private camTableUBO: GPUBuffer;
  private camTable = new ArrayBuffer(SLOT_COUNT * 32);
  private shapeUBO: GPUBuffer;
  private genBuffer: GPUBuffer;
  private genData = new ArrayBuffer(MAX_BAKE * GEN_STRIDE);
  private vertBuffer: GPUBuffer;
  private drawBuffer: GPUBuffer;
  private drawData = new ArrayBuffer(MAX_DRAWS * DRAW_STRIDE);
  private shadowDraws: { first: number; count: number }[] = [];
  private cascadeDraws: Draw[][] = [[], []];
  private shadowMats: Float32Array<ArrayBuffer>[] = [new Float32Array(16), new Float32Array(16)];
  private shadowTexel = [1, 1];
  private shadowMap: GPUTexture;
  private shadowLayers: GPUTextureView[];
  private shadowUBOs: GPUBuffer[];
  private shadowPipe: GPURenderPipeline;
  private shadowBGs: GPUBindGroup[];
  private shadowSampler: GPUSampler;
  private indexBuf: GPUBuffer;
  private indexCount: number;
  private drawCount = 0;
  private screen: GPUTexture[] = [];
  /** G-buffer texel under the screen centre: camera-relative position (m) and height, read back one frame late. */
  probe: number[] | null = null;
  private probeBuf: GPUBuffer;
  private probeState: 'idle' | 'copied' | 'mapping' = 'idle';
  private views!: { g0: GPUTextureView; g1: GPUTextureView; depth: GPUTextureView; hdr: GPUTextureView };

  private heightsPipe: GPUComputePipeline;
  private morphPipe: GPUComputePipeline;
  private gbufferPipe: GPURenderPipeline;
  private shadePipe: GPUComputePipeline;
  private compositePipe: GPURenderPipeline;
  private heightsBG: GPUBindGroup;
  private morphBG: GPUBindGroup;
  private gbufferBG: GPUBindGroup;
  private shadeBG!: GPUBindGroup;
  private shadeAtmoBG: GPUBindGroup;
  private compositeBG!: GPUBindGroup;

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.orbit = ctx.camera;
    this.atmo = new Atmosphere(device);
    this.atmo.params = { ...structuredClone(EARTH_ATMOSPHERE), groundAlbedo: [0.12, 0.13, 0.14] };
    this.cam = new PlanetCamera(PLANET_RADIUS);
    this.cam.surfaceHeight = (dir) => Math.max(0, terrainHeight(vec3.scale(dir, PLANET_RADIUS), this.shape(), (k) => geometryWeight(k, this.params.maxLevel, this.params.geometryOctaves)));
    this.tree = new ChunkTree(POOL, (n) => this.heightRange(n));
    this.applyPreset('Orbit, day side');
    this.cam.bind(this.orbit);

    buildGui(ctx.gui, this.params, {
      preset: (name) => this.applyPreset(name),
      terrainChanged: () => this.resetTerrain(),
    });

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'planet frame');
    this.slotUBO = uniformBuffer(device, SLOT_COUNT * 64, 'noise slots');
    device.queue.writeBuffer(this.slotUBO, 0, slotStaticData());
    this.camTableUBO = uniformBuffer(device, this.camTable.byteLength, 'camera noise table');
    this.shapeUBO = uniformBuffer(device, 16, 'terrain shape');
    this.genBuffer = device.createBuffer({ label: 'chunk bake list', size: this.genData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.vertBuffer = device.createBuffer({ label: 'chunk vertex pool', size: POOL * VERTS_PER_CHUNK * 32, usage: GPUBufferUsage.STORAGE });
    this.drawBuffer = device.createBuffer({ label: 'chunk draws', size: this.drawData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.probeBuf = device.createBuffer({ label: 'centre probe', size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const indices = chunkIndices();
    this.indexCount = indices.length;
    this.indexBuf = indexBuffer(device, indices, 'chunk indices');

    const consts = terrainConstantsWgsl();
    const chunkModule = createShader(device, 'chunk bake', wgsl(commonWgsl, consts, terrainWgsl, chunkWgsl));
    this.heightsPipe = device.createComputePipeline({ label: 'chunk heights', layout: 'auto', compute: { module: chunkModule, entryPoint: 'heights' } });
    this.morphPipe = device.createComputePipeline({ label: 'chunk morph', layout: 'auto', compute: { module: chunkModule, entryPoint: 'morph' } });
    const gbufModule = createShader(device, 'planet gbuffer', wgsl(commonWgsl, gbufferWgsl));
    this.gbufferPipe = device.createRenderPipeline({
      label: 'planet gbuffer',
      layout: 'auto',
      vertex: { module: gbufModule, entryPoint: 'vs' },
      fragment: { module: gbufModule, entryPoint: 'fs', targets: [{ format: 'rgba32float' }, { format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    });
    const shadeModule = createShader(device, 'planet shade', wgsl(commonWgsl, consts, terrainWgsl, atmosphereWgsl(1), shadeWgsl));
    this.shadePipe = device.createComputePipeline({ label: 'planet shade', layout: 'auto', compute: { module: shadeModule, entryPoint: 'main' } });
    const compositeModule = createShader(device, 'planet composite', wgsl(commonWgsl, compositeWgsl));
    this.compositePipe = device.createRenderPipeline({
      label: 'planet composite',
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vs' },
      fragment: { module: compositeModule, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });

    const shadowModule = createShader(device, 'planet shadow', wgsl(commonWgsl, shadowWgsl));
    this.shadowPipe = device.createRenderPipeline({
      label: 'planet shadow',
      layout: 'auto',
      vertex: { module: shadowModule, entryPoint: 'vs' },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 1, depthBiasSlopeScale: 1.5 },
    });
    this.shadowMap = device.createTexture({
      label: 'sun shadow cascades', size: [SHADOW_SIZE, SHADOW_SIZE, 2], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowLayers = [0, 1].map((i) => this.shadowMap.createView({ dimension: '2d', baseArrayLayer: i, arrayLayerCount: 1 }));
    this.shadowUBOs = [0, 1].map((i) => uniformBuffer(device, 64, 'shadow cascade ' + i));
    this.shadowBGs = this.shadowUBOs.map((u) => bindGroup(device, this.shadowPipe, 0, [u, this.drawBuffer, this.vertBuffer]));
    this.shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });

    this.heightsBG = bindGroup(device, this.heightsPipe, 0, [this.slotUBO, this.genBuffer, this.vertBuffer, this.shapeUBO]);
    this.morphBG = bindGroup(device, this.morphPipe, 0, [null, this.genBuffer, this.vertBuffer]);
    this.gbufferBG = bindGroup(device, this.gbufferPipe, 0, [this.frameUBO, this.drawBuffer, this.vertBuffer]);
    this.shadeAtmoBG = this.atmo.bindGroup(this.shadePipe, 1);
  }

  // ---- Terrain --------------------------------------------------------------

  shape(): TerrainShape {
    const p = this.params;
    return { seaBias: p.seaBias, mountainHeight: p.mountainHeight, detailHeight: p.detailHeight };
  }

  /** Height (m) at latitude / longitude (degrees), at the finest geometry LOD. */
  heightAt(latDeg: number, lonDeg: number): number {
    const la = deg(latDeg), lo = deg(lonDeg);
    const dir: Vec3 = [Math.cos(la) * Math.sin(lo), Math.sin(la), Math.cos(la) * Math.cos(lo)];
    return terrainHeight(vec3.scale(dir, PLANET_RADIUS), this.shape(), (k) => geometryWeight(k, this.params.maxLevel, this.params.geometryOctaves));
  }

  /** Conservative geometry height range of a node from a 3 x 3 CPU sample. */
  private heightRange(n: QNode): [number, number] {
    if (n.level < 4) return [0, 9500];
    const size = 2 / 2 ** n.level;
    const lod = n.level;
    const w = (k: number) => geometryWeight(k, lod, this.params.geometryOctaves);
    let lo = Infinity, hi = -Infinity;
    for (let b = -1; b <= 1; b++) {
      for (let a = -1; a <= 1; a++) {
        const dir = vec3.normalize(cubePoint(n.face, n.uc + (a * size) / 2, n.vc + (b * size) / 2));
        const h = Math.max(0, terrainHeight(vec3.scale(dir, PLANET_RADIUS), this.shape(), w));
        lo = Math.min(lo, h);
        hi = Math.max(hi, h);
      }
    }
    // Unsampled relief between the samples: scale with the node size.
    const margin = Math.min(Math.max(nodeSize(n.level) * 0.15, 20), 4000);
    return [Math.max(0, lo - margin), hi + margin];
  }

  private resetTerrain() {
    this.tree = new ChunkTree(POOL, (n) => this.heightRange(n));
  }

  // ---- Presets and camera ---------------------------------------------------

  applyPreset(name: string) {
    const preset = PRESETS[name];
    if (!preset) return;
    Object.assign(this.params, structuredClone(DEFAULTS), structuredClone(preset.params));
    this.sunLon = this.params.sunLon;
    const c = preset.camera;
    const ground = Math.max(0, this.heightAt(c.lat, c.lon));
    this.cam.setGeo(c.lat, c.lon, Math.max(c.altitude, ground + this.cam.eyeHeight), c.heading, c.pitch);
  }

  private sunDir(): Vec3 {
    const la = deg(this.params.sunLat), lo = deg(this.sunLon);
    return [Math.cos(la) * Math.sin(lo), Math.sin(la), Math.cos(la) * Math.cos(lo)];
  }

  // ---- Resources ------------------------------------------------------------

  private rebuildTargets() {
    const d = this.device;
    this.screen.forEach((t) => t.destroy());
    const size = [this.width, this.height];
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const g0 = d.createTexture({ label: 'gbuffer position', size, format: 'rgba32float', usage: RT | GPUTextureUsage.COPY_SRC });
    const g1 = d.createTexture({ label: 'gbuffer info', size, format: 'rgba8unorm', usage: RT });
    const depth = d.createTexture({ label: 'planet depth', size, format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const hdr = d.createTexture({ label: 'planet hdr', size, format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.screen = [g0, g1, depth, hdr];
    this.views = { g0: g0.createView(), g1: g1.createView(), depth: depth.createView(), hdr: hdr.createView() };
    this.shadeBG = bindGroup(d, this.shadePipe, 0, [
      this.frameUBO, this.slotUBO, this.camTableUBO, this.views.g0, this.views.g1, this.views.hdr,
      this.shadowMap.createView({ dimension: '2d-array' }), this.shadowSampler,
    ]);
    this.compositeBG = bindGroup(d, this.compositePipe, 0, [this.frameUBO, this.views.hdr]);
    this.sizeDirty = false;
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.sizeDirty = true;
  }

  // ---- Per-frame CPU work -----------------------------------------------------

  private writeBakeList(nodes: QNode[]) {
    const f32 = new Float32Array(this.genData);
    const i32 = new Int32Array(this.genData);
    nodes.forEach((n, k) => {
      const o = (k * GEN_STRIDE) / 4;
      const face = FACES[n.face];
      const size = 2 / 2 ** n.level;
      const c0 = cubePoint(n.face, n.uc, n.vc);
      f32.set([...face.n, n.level], o);
      f32.set([...face.a, n.uc], o + 4);
      f32.set([...face.b, n.vc], o + 8);
      f32.set([...c0, size], o + 12);
      // Skirts deep enough to hide the transient gaps when neighbours are
      // several levels apart (the LOD catching up with a fast camera).
      f32.set([n.slot, nodeSize(n.level) * 0.08 + 20, this.params.geometryOctaves, 0], o + 16);
      // The noise reference is the node origin O = R dir(c0), in double.
      writeSlotTable(vec3.scale(vec3.normalize(c0), PLANET_RADIUS), TERRAIN_SLOTS, i32, f32, o + 20);
    });
    if (nodes.length) this.device.queue.writeBuffer(this.genBuffer, 0, this.genData, 0, nodes.length * GEN_STRIDE);
  }

  /** Packs the camera's draws, then each shadow cascade's, into the draw buffer. */
  private writeDraws(lists: Draw[][]) {
    const f32 = new Float32Array(this.drawData);
    const u32 = new Uint32Array(this.drawData);
    const cam = this.cam.position;
    const ranges: { first: number; count: number }[] = [];
    let k = 0;
    for (const list of lists) {
      const first = k;
      for (const d of list) {
        if (k >= MAX_DRAWS) break;
        const n = d.node;
        const o = (k * DRAW_STRIDE) / 4;
        const origin = vec3.scale(vec3.normalize(cubePoint(n.face, n.uc, n.vc)), PLANET_RADIUS);
        // Double-precision subtraction, then float: the only large-to-small step.
        f32.set([origin[0] - cam[0], origin[1] - cam[1], origin[2] - cam[2], n.level], o);
        // Fully morphed where the parent level takes over (2x this split distance).
        const D = d.splitDistance;
        f32.set(n.level === 0 ? [1e30, 2e30] : [1.35 * D, 1.85 * D], o + 4);
        u32[o + 6] = n.slot;
        u32[o + 7] = 0;
        k++;
      }
      ranges.push({ first, count: k - first });
    }
    this.drawCount = ranges[0].count;
    this.shadowDraws = ranges.slice(1);
    if (k) this.device.queue.writeBuffer(this.drawBuffer, 0, this.drawData, 0, k * DRAW_STRIDE);
  }

  /**
   * Two sun cascades around the camera (shifted forward), sized by the
   * height above ground. Sets the light matrices and returns each cascade's
   * camera-relative culling planes. The centre is snapped to the texel grid
   * using the double-precision camera position, so the maps do not shimmer.
   */
  private shadowCascades(sunDir: Vec3): [number, number, number, number][][] {
    const c = this.cam;
    const L = sunDir;
    const ref: Vec3 = Math.abs(L[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const right = vec3.normalize(vec3.cross(ref, L));
    const up = vec3.cross(L, right);
    const e0 = Math.min(Math.max(c.heightAboveGround * 10 + 2500, 2500), 120000);
    const extents = [e0, Math.min(Math.max(e0 * 10, 25000), 800000)];
    const fwd = vec3.normalize(vec3.sub(c.forward, vec3.scale(c.up, vec3.dot(c.forward, c.up))));
    return extents.map((E, i) => {
      const texel = E / SHADOW_SIZE;
      this.shadowTexel[i] = texel;
      const centerAbs = vec3.add(c.position, vec3.scale(fwd, E * 0.3));
      const lx = vec3.dot(centerAbs, right), ly = vec3.dot(centerAbs, up);
      const sx = Math.round(lx / texel) * texel, sy = Math.round(ly / texel) * texel;
      // Camera-relative centre, snapped in light space.
      const ctr = vec3.add(vec3.sub(centerAbs, c.position), vec3.add(vec3.scale(right, sx - lx), vec3.scale(up, sy - ly)));
      const h = E / 2;
      const Z = h + 20000 + (E * E) / (8 * PLANET_RADIUS);
      const m = this.shadowMats[i];
      const rows = [
        [right[0] / h, right[1] / h, right[2] / h, -vec3.dot(ctr, right) / h],
        [up[0] / h, up[1] / h, up[2] / h, -vec3.dot(ctr, up) / h],
        [-L[0] / (2 * Z), -L[1] / (2 * Z), -L[2] / (2 * Z), 0.5 + vec3.dot(ctr, L) / (2 * Z)],
        [0, 0, 0, 1],
      ];
      for (let r = 0; r < 4; r++) for (let col = 0; col < 4; col++) m[col * 4 + r] = rows[r][col];
      this.device.queue.writeBuffer(this.shadowUBOs[i], 0, m);
      const cr = vec3.dot(ctr, right), cu = vec3.dot(ctr, up);
      return [
        [right[0], right[1], right[2], h - cr], [-right[0], -right[1], -right[2], h + cr],
        [up[0], up[1], up[2], h - cu], [-up[0], -up[1], -up[2], h + cu],
      ] as [number, number, number, number][];
    });
  }

  /** Camera-relative frustum side planes (normalised), from the camera-relative view-projection. */
  private frustumPlanes(): [number, number, number, number][] {
    const m = this.cam.viewProj;
    const row = (r: number) => [m[r], m[4 + r], m[8 + r], m[12 + r]];
    const r0 = row(0), r1 = row(1), r3 = row(3);
    const planes = [
      r3.map((v, i) => v + r0[i]), r3.map((v, i) => v - r0[i]),
      r3.map((v, i) => v + r1[i]), r3.map((v, i) => v - r1[i]),
    ];
    return planes.map((p) => {
      const l = Math.hypot(p[0], p[1], p[2]) || 1;
      return [p[0] / l, p[1] / l, p[2] / l, p[3] / l] as [number, number, number, number];
    });
  }

  private writeCameraTable(time: number) {
    const p = this.params;
    const i32 = new Int32Array(this.camTable);
    const f32 = new Float32Array(this.camTable);
    // Clouds drift per octave in their own direction (so they evolve rather
    // than slide); wave octaves travel at their deep-water phase speed.
    const cloudT = time * p.cloudTimeLapse;
    const extra = (s: number): Vec3 | null => {
      if (s >= SLOT_BASE.cloud && s < SLOT_BASE.cloud + CHANNELS.cloud.count) {
        const k = s - SLOT_BASE.cloud;
        const a = k * 2.399963;
        const speed = 9 + 4 * Math.sin(k * 1.7);
        return [Math.cos(a) * speed * cloudT, Math.sin(a * 0.7) * speed * 0.4 * cloudT, Math.sin(a) * speed * cloudT];
      }
      if (s >= SLOT_BASE.wave && s < SLOT_BASE.wave + CHANNELS.wave.count) {
        const lambda = 2 ** (22 - SLOT_LEVEL[s]);
        const c = Math.sqrt((9.81 * lambda) / (2 * Math.PI));
        const a = 0.6 + (s - SLOT_BASE.wave) * 0.35;
        return [Math.cos(a) * c * time, 0, Math.sin(a) * c * time];
      }
      return null;
    };
    writeSlotTable(this.cam.position, SLOT_COUNT, i32, f32, 0, extra);
    this.device.queue.writeBuffer(this.camTableUBO, 0, this.camTable);
  }

  private writeFrame(info: FrameInfo, sunDir: Vec3) {
    const p = this.params;
    const c = this.cam;
    const f = this.frameData;
    f.set(c.viewProj, 0);
    f.set(c.invViewProj, 16);
    f.set([...c.position, info.time], 32);
    f.set([...sunDir, p.exposure], 36);
    const pixelAngle = (2 * Math.tan(c.fovY / 2)) / this.height;
    f.set([this.width, this.height, pixelAngle, p.debugView], 40);
    f.set([p.cloudCoverage, p.cloudAltitude, p.cloudDepth, p.cloudShadow], 44);
    f.set([p.cloudSwirl, p.clouds ? 1 : 0, p.cloudTimeLapse, 0], 48);
    f.set([p.cityIntensity, p.cityDensity, 0, 0], 52);
    f.set([p.waveSlope, p.waterRoughness, 0, 0], 56);
    f.set([c.altitude, p.stars, p.snowLine, p.moisture], 60);
    f.set([p.seaBias, p.mountainHeight, p.detailHeight, 0], 64);
    f.set(this.shadowMats[0], 68);
    f.set(this.shadowMats[1], 84);
    f.set([this.shadowTexel[0], this.shadowTexel[1], p.shadows ? 1 : 0, 0], 100);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    if (this.sizeDirty) this.rebuildTargets();
    const key = JSON.stringify([p.seaBias, p.mountainHeight, p.detailHeight, p.geometryOctaves]);
    if (key !== this.terrainKey) {
      if (this.terrainKey) this.resetTerrain();
      this.terrainKey = key;
      this.device.queue.writeBuffer(this.shapeUBO, 0, new Float32Array([p.seaBias, p.mountainHeight, p.detailHeight, 0]));
    }

    this.cam.fovY = deg(p.fov);
    this.cam.update(this.orbit, this.width / this.height);
    if (p.dayLength > 0) {
      this.sunLon = ((((this.sunLon - (360 * info.dt) / p.dayLength) + 180) % 360) + 360) % 360 - 180;
      p.sunLon = this.sunLon;
    } else {
      this.sunLon = p.sunLon;
    }
    const sunDir = this.sunDir();

    // Bake what the previous selection asked for, then select with it.
    const bake = p.freezeLod ? [] : this.tree.allocate(Math.min(p.bakeBudget, MAX_BAKE), this.cam.position);
    this.bakedThisFrame = bake.length;
    const cascadePlanes = this.shadowCascades(sunDir);
    if (!p.freezeLod) {
      const base = {
        camera: this.cam.position,
        pxPerRadian: this.height / 2 / Math.tan(this.cam.fovY / 2),
        maxErrorPx: p.maxErrorPx,
        maxLevel: p.maxLevel,
      };
      this.tree.beginFrame();
      this.tree.select({ ...base, planes: this.frustumPlanes(), occluderRadius: PLANET_RADIUS - 11000 });
      // Shadow casters: the same LOD metric, culled by each cascade's box
      // instead of the view frustum (mountains behind the camera cast too).
      // Casters use a coarser error: shadows blur detail anyway.
      const casters = { ...base, maxErrorPx: p.maxErrorPx * 2.5, occluderRadius: 0 };
      this.cascadeDraws = p.shadows ? cascadePlanes.map((planes) => this.tree.select({ ...casters, planes }, false)) : [[], []];
    }
    this.writeBakeList(bake);
    this.writeDraws([this.tree.draws, ...this.cascadeDraws]);
    this.writeCameraTable(info.time);
    this.writeFrame(info, sunDir);

    if (bake.length) {
      const cp = encoder.beginComputePass({ label: 'chunk bake' });
      cp.setPipeline(this.heightsPipe);
      cp.setBindGroup(0, this.heightsBG);
      cp.dispatchWorkgroups(Math.ceil((GRID * GRID) / 64), bake.length);
      cp.setPipeline(this.morphPipe);
      cp.setBindGroup(0, this.morphBG);
      cp.dispatchWorkgroups(Math.ceil(VERTS_PER_CHUNK / 64), bake.length);
      cp.end();
    }

    this.atmo.update(encoder, {
      cameraKm: vec3.scale(this.cam.position, 1e-3),
      altitudeKm: this.cam.altitude * 1e-3,
      sunDir,
      invViewProj: this.cam.invViewProj,
    });

    for (let i = 0; i < 2; i++) {
      const sp = encoder.beginRenderPass({
        label: 'sun shadow ' + i,
        colorAttachments: [],
        depthStencilAttachment: { view: this.shadowLayers[i], depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      const r = this.shadowDraws[i];
      if (p.shadows && r && r.count) {
        sp.setPipeline(this.shadowPipe);
        sp.setBindGroup(0, this.shadowBGs[i]);
        sp.setIndexBuffer(this.indexBuf, 'uint16');
        sp.drawIndexed(this.indexCount, r.count, 0, 0, r.first);
      }
      sp.end();
    }

    const clear = (view: GPUTextureView): GPURenderPassColorAttachment => ({ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] });
    const gp = encoder.beginRenderPass({
      label: 'planet gbuffer',
      colorAttachments: [clear(this.views.g0), clear(this.views.g1)],
      depthStencilAttachment: { view: this.views.depth, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 0 },
    });
    if (this.drawCount) {
      gp.setPipeline(this.gbufferPipe);
      gp.setBindGroup(0, this.gbufferBG);
      gp.setIndexBuffer(this.indexBuf, 'uint16');
      gp.drawIndexed(this.indexCount, this.drawCount);
    }
    gp.end();
    if (this.probeState === 'copied') {
      this.probeState = 'mapping';
      this.probeBuf.mapAsync(GPUMapMode.READ).then(() => {
        this.probe = Array.from(new Float32Array(this.probeBuf.getMappedRange(), 0, 4));
        this.probeBuf.unmap();
        this.probeState = 'idle';
      }).catch(() => {});
    } else if (this.probeState === 'idle') {
      encoder.copyTextureToBuffer({ texture: this.screen[0], origin: [this.width >> 1, this.height >> 1] }, { buffer: this.probeBuf, bytesPerRow: 256 }, [1, 1]);
      this.probeState = 'copied';
    }

    const sp = encoder.beginComputePass({ label: 'planet shade' });
    sp.setPipeline(this.shadePipe);
    sp.setBindGroup(0, this.shadeBG);
    sp.setBindGroup(1, this.shadeAtmoBG);
    sp.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
    sp.end();

    const rp = encoder.beginRenderPass({
      label: 'planet composite',
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(this.compositePipe);
    rp.setBindGroup(0, this.compositeBG);
    rp.draw(3);
    rp.end();
  }

  hud() {
    const c = this.cam;
    const g = c.geo;
    const fmt = (m: number) => (m >= 1e5 ? `${(m / 1000).toFixed(0)} km` : m >= 2000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(1)} m`);
    const s = this.tree.stats;
    return `altitude ${fmt(c.altitude)} · above ground ${fmt(c.heightAboveGround)} · ${g.lat.toFixed(3)}°, ${g.lon.toFixed(3)}° · near ${fmt(c.near)}` +
      `\nchunks drawn ${s.drawn} (deepest level ${s.maxLevel}), resident ${s.resident}/${POOL}, baked ${this.bakedThisFrame}, pending ${this.tree.requests.length}` +
      `\ndrag: look · wheel: altitude · WASD/QE: fly (shift: faster)`;
  }

  destroy() {
    this.atmo.destroy();
    this.screen.forEach((t) => t.destroy());
    this.shadowMap.destroy();
    this.shadowUBOs.forEach((b) => b.destroy());
    [this.frameUBO, this.slotUBO, this.camTableUBO, this.shapeUBO, this.genBuffer, this.vertBuffer, this.drawBuffer, this.indexBuf, this.probeBuf].forEach((b) => b.destroy());
  }
}

export const planetEntry: DemoEntry = {
  id: 'planet',
  title: 'Procedural planet',
  tags: ['cube-sphere LOD', 'noise precision', 'biomes', 'clouds', 'atmosphere'],
  info: `
    <h2>Procedural planet</h2>
    <ul>
      <li>No textures: continents, mountains, biomes, clouds and cities from 3D noise on the sphere</li>
      <li>Cube-sphere quadtree, screen-space-error LOD, morphing and skirts</li>
      <li>Per-octave double-precision lattice split: exact noise from 20,000 km to walking height</li>
      <li>Camera-relative rendering, reversed-Z, adaptive near plane</li>
      <li>Per-pixel analytic normals, ocean glint, city lights</li>
      <li>Cloud layer with shadows, sun shadow cascades on the terrain</li>
      <li>Hillaire atmosphere shared with the atmosphere study</li>
    </ul>
    <p><kbd>drag</kbd> look · <kbd>wheel</kbd> altitude<br>
    <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> fly · <kbd>Q</kbd><kbd>E</kbd> down/up · <kbd>shift</kbd> faster</p>`,
  create: (ctx) => new PlanetDemo(ctx),
};
