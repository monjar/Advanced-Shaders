import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, indexBuffer, uniformBuffer } from '../../core/gpu';
import { deg, mat4, vec3, type Mat4, type Vec3 } from '../../core/math';
import {
  BRAZIERS, ENV_STRIDE, OBJ_STRIDE, OBJECT_HEIGHT, buildEnvironment, objectMesh, shapeUniform,
} from './geometry';
import { DEFAULTS, PRESETS, buildGui, type MagicParams } from './params';

import bakeWgsl from './shaders/bake.wgsl?raw';
import commonWgsl from './shaders/common.wgsl?raw';
import compositeWgsl from './shaders/composite.wgsl?raw';
import envWgsl from './shaders/env.wgsl?raw';
import envFilterWgsl from './shaders/envfilter.wgsl?raw';
import fieldWgsl from './shaders/field.wgsl?raw';
import fieldResWgsl from './shaders/fieldres.wgsl?raw';
import mipsWgsl from './shaders/mips.wgsl?raw';
import objectWgsl from './shaders/object.wgsl?raw';
import particlesWgsl from './shaders/particles.wgsl?raw';
import shapeWgsl from './shaders/shape.wgsl?raw';
import warpWgsl from './shaders/warp.wgsl?raw';

const HDR: GPUTextureFormat = 'rgba16float';
const SHAPE_RES = 96;
const CRACK_RES = 128;
/** Half extent (object units) of the baked SDF / crack volume and of the field volume. */
const SHAPE_BOX = 1.25;
const FIELD_BOX = 1.7;
const CUBE_SIZE = 128;
const CUBE_MIPS = 6;
const SCENE_MIPS = 5;
const BLOOM_LEVELS = 6;
const MAX_PARTICLES = 262144;
const OBJECT_SCALE = 0.8;
const SHADOW_SIZE = 2048;

const wgsl = (...parts: string[]) => parts.join('\n');

// Cube faces in WebGPU order: right, up and forward axes of each face camera,
// matching the direction convention in envfilter.wgsl.
const CUBE_FACES: [Vec3, Vec3, Vec3][] = [
  [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
  [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
  [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
  [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
];

function faceView(eye: Vec3, [r, u, f]: [Vec3, Vec3, Vec3]): Mat4 {
  const m = new Float32Array(16);
  m[0] = r[0]; m[4] = r[1]; m[8] = r[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -vec3.dot(r, eye);
  m[13] = -vec3.dot(u, eye);
  m[14] = vec3.dot(f, eye);
  m[15] = 1;
  return m;
}

/** Object → world: translation · rotY · rotX · uniform scale. */
function modelMatrix(pos: Vec3, yaw: number, tilt: number, s: number): Mat4 {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cx = Math.cos(tilt), sx = Math.sin(tilt);
  // R = Ry · Rx, columns.
  const c0: Vec3 = [cy, 0, -sy];
  const c1: Vec3 = [sy * sx, cx, cy * sx];
  const c2: Vec3 = [sy * cx, -sx, cy * cx];
  const m = new Float32Array(16);
  m.set([...vec3.scale(c0, s), 0, ...vec3.scale(c1, s), 0, ...vec3.scale(c2, s), 0, ...pos, 1]);
  return m;
}

class MagicDemo implements Demo {
  private device: GPUDevice;
  private camera: DemoContext['camera'];
  private gui: DemoContext['gui'];
  private params: MagicParams = structuredClone(DEFAULTS);
  private width = 1;
  private height = 1;
  private frameIndex = 0;
  private fieldTime = 0;
  private spinAngle = 0;
  private wallTime = 0;

  private frameData = new Float32Array(120);
  private materialData = new Float32Array(76);
  private viewData = new Float32Array(36);
  private frameUBO: GPUBuffer;
  private materialUBO: GPUBuffer;
  private viewUBO: GPUBuffer;
  private shapeUBO: GPUBuffer;
  private cubeViewUBOs: GPUBuffer[] = [];
  private cubeFilterUBOs: GPUBuffer[] = [];
  private probeBuf: GPUBuffer;
  private particleBuf: GPUBuffer;

  private envVB: GPUBuffer;
  private envIB: GPUBuffer;
  private envCount: number;
  private objVB!: GPUBuffer;
  private objIB!: GPUBuffer;
  private objCount = 0;

  private shapeTex: GPUTexture;
  private crackTex: GPUTexture;
  private fieldA!: GPUTexture;
  private fieldB!: GPUTexture;
  private fieldG!: GPUTexture;
  private fieldRes = 0;
  private envCube: GPUTexture;
  private envDepth: GPUTexture;
  private shadowMap: GPUTexture;
  private sizeTextures: GPUTexture[] = [];
  private sceneHdr!: GPUTexture;
  private hdr!: GPUTexture;
  private sceneTex!: GPUTexture;
  private depth!: GPUTexture;
  private bloomDown: GPUTexture[] = [];
  private bloomUp: GPUTexture[] = [];

  private linearClamp: GPUSampler;
  private shadowSampler: GPUSampler;

  private shapePipe: GPUComputePipeline;
  private crackPipe: GPUComputePipeline;
  private fieldPipe: GPUComputePipeline;
  private probePipe: GPUComputePipeline;
  private simPipe: GPUComputePipeline;
  private shadowPipe: GPURenderPipeline;
  private envPipe: GPURenderPipeline;
  private skyPipe: GPURenderPipeline;
  private flamePipe: GPURenderPipeline;
  private envFilterPipe: GPURenderPipeline;
  private downPipe: GPURenderPipeline;
  private downKarisPipe: GPURenderPipeline;
  private upPipe: GPURenderPipeline;
  private warpPipe: GPURenderPipeline;
  private objectPipe: GPURenderPipeline;
  private particlePipe: GPURenderPipeline;
  private compositePipe: GPURenderPipeline;

  private g: Record<string, GPUBindGroup[]> = {};
  private cubeGroups: { env: GPUBindGroup; sky: GPUBindGroup; flame: GPUBindGroup }[] = [];
  private cubeFilterGroups: GPUBindGroup[] = [];
  private mipGroups: GPUBindGroup[] = [];
  private bloomGroups: { down: GPUBindGroup[]; up: GPUBindGroup[] } = { down: [], up: [] };

  private shapeDirty = true;
  private crackDirty = true;
  private shadowDirty = true;
  private builtShape = -1;
  private builtCrackScale = -1;

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.camera = ctx.camera;
    this.gui = ctx.gui;
    Object.assign(this.camera, {
      target: [0, 2.15, 0] as Vec3, distance: 5.4, yaw: 0.25, pitch: 0.1,
      minDistance: 1.6, minEyeHeight: 0.25, minPitch: -0.25, maxPitch: 1.3, moveSpeed: 3,
    });
    this.camera.fovY = deg(42);
    this.camera.near = 0.05;
    buildGui(ctx.gui, this.params, () => {
      this.crackDirty = true;
    });

    this.frameUBO = uniformBuffer(device, this.frameData.byteLength, 'frame');
    this.materialUBO = uniformBuffer(device, this.materialData.byteLength, 'material');
    this.viewUBO = uniformBuffer(device, this.viewData.byteLength, 'view');
    this.shapeUBO = uniformBuffer(device, shapeUniform(0, SHAPE_BOX).byteLength, 'shape');
    for (let f = 0; f < 6; f++) this.cubeViewUBOs.push(uniformBuffer(device, this.viewData.byteLength, `cube view ${f}`));
    for (let mip = 1; mip < CUBE_MIPS; mip++) {
      for (let f = 0; f < 6; f++) {
        const size = CUBE_SIZE >> mip;
        const b = uniformBuffer(device, 16, `cube filter ${mip}/${f}`);
        device.queue.writeBuffer(b, 0, new Float32Array([f, size, 2.6 / size, 0]));
        this.cubeFilterUBOs.push(b);
      }
    }
    this.probeBuf = device.createBuffer({ label: 'probe', size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.particleBuf = device.createBuffer({ label: 'particles', size: MAX_PARTICLES * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const env = buildEnvironment();
    this.envVB = device.createBuffer({ label: 'env', size: env.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.envVB, 0, env.vertices);
    this.envIB = indexBuffer(device, env.indices, 'env');
    this.envCount = env.count;

    const tex3d = (label: string, size: number, format: GPUTextureFormat) => device.createTexture({
      label, size: [size, size, size], dimension: '3d', format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shapeTex = tex3d('shape sdf', SHAPE_RES, 'rgba16float');
    this.crackTex = tex3d('cracks', CRACK_RES, 'rgba8unorm');
    this.envCube = device.createTexture({
      label: 'env cube', size: [CUBE_SIZE, CUBE_SIZE, 6], format: HDR, mipLevelCount: CUBE_MIPS,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.envDepth = device.createTexture({ label: 'env depth', size: [CUBE_SIZE, CUBE_SIZE], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.shadowMap = device.createTexture({
      label: 'moon shadow', size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.linearClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    this.shadowSampler = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });

    // Pipelines.
    const compute = (label: string, code: string, entryPoint: string) =>
      device.createComputePipeline({ label, layout: 'auto', compute: { module: createShader(device, label, code), entryPoint } });
    this.shapePipe = compute('shape bake', wgsl(commonWgsl, shapeWgsl), 'bakeShape');
    const bakeCode = wgsl(commonWgsl, fieldWgsl, fieldResWgsl, bakeWgsl);
    this.fieldPipe = compute('field bake', bakeCode, 'bakeField');
    this.crackPipe = compute('crack bake', bakeCode, 'bakeCracks');
    this.probePipe = compute('probe', bakeCode, 'probeField');
    const particleCode = wgsl(commonWgsl, fieldWgsl, fieldResWgsl, particlesWgsl);
    this.simPipe = compute('particle sim', particleCode, 'simulate');

    const envModule = createShader(device, 'env', wgsl(commonWgsl, fieldWgsl, envWgsl));
    const envLayout: GPUVertexBufferLayout = {
      arrayStride: ENV_STRIDE,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'float32' },
      ],
    };
    const depthWrite: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' };
    this.envPipe = device.createRenderPipeline({
      label: 'env', layout: 'auto',
      vertex: { module: envModule, entryPoint: 'vs', buffers: [envLayout] },
      fragment: { module: envModule, entryPoint: 'fs', targets: [{ format: HDR }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: depthWrite,
    });
    this.skyPipe = device.createRenderPipeline({
      label: 'sky', layout: 'auto',
      vertex: { module: envModule, entryPoint: 'skyVs' },
      fragment: { module: envModule, entryPoint: 'skyFs', targets: [{ format: HDR }] },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'always' },
    });
    this.flamePipe = device.createRenderPipeline({
      label: 'flames', layout: 'auto',
      vertex: { module: envModule, entryPoint: 'flameVs' },
      fragment: {
        module: envModule, entryPoint: 'flameFs',
        targets: [{
          format: HDR,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater' },
    });
    this.shadowPipe = device.createRenderPipeline({
      label: 'shadow', layout: 'auto',
      vertex: { module: envModule, entryPoint: 'shadowVs', buffers: [{ arrayStride: ENV_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2 },
    });

    const fullscreen = (label: string, module: GPUShaderModule, entryPoint: string, format: GPUTextureFormat, constants?: Record<string, number>) =>
      device.createRenderPipeline({
        label, layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint, targets: [{ format }], constants },
      });
    const filterModule = createShader(device, 'env filter', envFilterWgsl);
    this.envFilterPipe = fullscreen('env filter', filterModule, 'fs', HDR);
    const mipModule = createShader(device, 'mips', mipsWgsl);
    this.downPipe = fullscreen('downsample', mipModule, 'fsDown', HDR, { KARIS: 0 });
    this.downKarisPipe = fullscreen('downsample karis', mipModule, 'fsDown', HDR, { KARIS: 1 });
    this.upPipe = fullscreen('upsample', mipModule, 'fsUp', HDR);
    this.warpPipe = fullscreen('warp', createShader(device, 'warp', wgsl(commonWgsl, fieldWgsl, fieldResWgsl, warpWgsl)), 'fs', HDR);
    this.compositePipe = fullscreen('composite', createShader(device, 'composite', wgsl(commonWgsl, fieldWgsl, fieldResWgsl, compositeWgsl)), 'fs', ctx.format);

    const objModule = createShader(device, 'object', wgsl(commonWgsl, fieldWgsl, fieldResWgsl, objectWgsl));
    this.objectPipe = device.createRenderPipeline({
      label: 'object', layout: 'auto',
      vertex: {
        module: objModule, entryPoint: 'vs',
        buffers: [{
          arrayStride: OBJ_STRIDE,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
            { shaderLocation: 2, offset: 24, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: objModule, entryPoint: 'fs', targets: [{ format: HDR }] },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: depthWrite,
    });
    const partModule = createShader(device, 'particles', particleCode);
    this.particlePipe = device.createRenderPipeline({
      label: 'particles', layout: 'auto',
      vertex: { module: partModule, entryPoint: 'vsParticle' },
      fragment: {
        module: partModule, entryPoint: 'fsParticle',
        targets: [{
          format: HDR,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater' },
    });

    // Cube-face bind groups and the filter chain do not depend on the screen size.
    for (let f = 0; f < 6; f++) {
      this.cubeGroups.push({
        env: bindGroup(device, this.envPipe, 0, [
          this.cubeViewUBOs[f], this.frameUBO, this.materialUBO, this.probeBuf, this.shadowMap.createView(), this.shadowSampler,
        ]),
        sky: bindGroup(device, this.skyPipe, 0, [this.cubeViewUBOs[f], this.frameUBO, null, this.probeBuf]),
        flame: bindGroup(device, this.flamePipe, 0, [this.cubeViewUBOs[f], this.frameUBO]),
      });
    }
    for (let mip = 1; mip < CUBE_MIPS; mip++) {
      const srcView = this.envCube.createView({ dimension: 'cube', baseMipLevel: mip - 1, mipLevelCount: 1 });
      for (let f = 0; f < 6; f++) {
        this.cubeFilterGroups.push(bindGroup(device, this.envFilterPipe, 0, [srcView, this.linearClamp, this.cubeFilterUBOs[(mip - 1) * 6 + f]]));
      }
    }
    this.setShape();
    this.ensureField();
  }

  private setShape() {
    const shape = this.params.shape;
    this.objVB?.destroy();
    this.objIB?.destroy();
    const m = objectMesh(shape);
    this.objVB = this.device.createBuffer({ label: 'object', size: m.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.objVB, 0, m.vertices);
    this.objIB = indexBuffer(this.device, m.indices, 'object');
    this.objCount = m.count;
    this.device.queue.writeBuffer(this.shapeUBO, 0, shapeUniform(shape, SHAPE_BOX));
    this.builtShape = shape;
    this.shapeDirty = true;
  }

  private ensureField() {
    const res = this.params.fieldRes;
    if (res === this.fieldRes) return;
    this.fieldA?.destroy();
    this.fieldB?.destroy();
    this.fieldG?.destroy();
    const make = (label: string) => this.device.createTexture({
      label, size: [res, res, res], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fieldA = make('field A (v, C)');
    this.fieldB = make('field B (advected noise)');
    this.fieldG = make('field G (∇C)');
    this.fieldRes = res;
    if (this.sceneTex) this.buildGroups();
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    const d = this.device;
    this.sizeTextures.forEach((t) => t.destroy());
    const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const make = (label: string, w: number, h: number, format: GPUTextureFormat, usage: number, mipLevelCount = 1) =>
      d.createTexture({ label, size: [Math.max(1, w), Math.max(1, h)], format, usage, mipLevelCount });
    this.sceneHdr = make('scene hdr', width, height, HDR, RT);
    this.hdr = make('hdr', width, height, HDR, RT | GPUTextureUsage.COPY_SRC);
    this.sceneTex = make('scene (refraction)', width, height, HDR, RT | GPUTextureUsage.COPY_DST, SCENE_MIPS);
    this.depth = make('depth', width, height, 'depth32float', GPUTextureUsage.RENDER_ATTACHMENT);
    this.bloomDown = [];
    this.bloomUp = [];
    let w = width;
    let h = height;
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      w = Math.max(1, Math.floor(w / 2));
      h = Math.max(1, Math.floor(h / 2));
      this.bloomDown.push(make(`bloom down ${i}`, w, h, HDR, RT));
      if (i < BLOOM_LEVELS - 1) this.bloomUp.push(make(`bloom up ${i}`, w, h, HDR, RT));
    }
    this.sizeTextures = [this.sceneHdr, this.hdr, this.sceneTex, this.depth, ...this.bloomDown, ...this.bloomUp];
    this.buildGroups();
  }

  private buildGroups() {
    const d = this.device;
    const s = this.linearClamp;
    const A = this.fieldA.createView();
    const B = this.fieldB.createView();
    const S = this.shapeTex.createView();
    const C = this.crackTex.createView();
    const cube = this.envCube.createView({ dimension: 'cube' });
    const fieldAll = (p: GPURenderPipeline | GPUComputePipeline) => bindGroup(d, p, 1, [A, B, S, C, s]);
    const u = [this.frameUBO, this.materialUBO];
    this.g = {
      shape: [bindGroup(d, this.shapePipe, 0, [null, this.shapeUBO, S])],
      field: [
        bindGroup(d, this.fieldPipe, 0, [...u, this.fieldA.createView(), this.fieldB.createView(), null, null, this.fieldG.createView()]),
        bindGroup(d, this.fieldPipe, 1, [null, null, S, null, s]),
      ],
      crack: [bindGroup(d, this.crackPipe, 0, [...u, null, null, C])],
      probe: [bindGroup(d, this.probePipe, 0, [...u, null, null, null, this.probeBuf]), fieldAll(this.probePipe)],
      sim: [bindGroup(d, this.simPipe, 0, [...u, this.particleBuf]), fieldAll(this.simPipe)],
      shadow: [bindGroup(d, this.shadowPipe, 0, [null, this.frameUBO])],
      env: [bindGroup(d, this.envPipe, 0, [this.viewUBO, ...u, this.probeBuf, this.shadowMap.createView(), this.shadowSampler])],
      sky: [bindGroup(d, this.skyPipe, 0, [this.viewUBO, this.frameUBO, null, this.probeBuf])],
      flame: [bindGroup(d, this.flamePipe, 0, [this.viewUBO, this.frameUBO])],
      warp: [
        bindGroup(d, this.warpPipe, 0, [...u, this.sceneHdr.createView(), s]),
        bindGroup(d, this.warpPipe, 1, [A, B, S, null, s]),
      ],
      object: [
        bindGroup(d, this.objectPipe, 0, [...u, this.sceneTex.createView(), cube, s]),
        bindGroup(d, this.objectPipe, 1, [A, B, S, C, s, this.fieldG.createView()]),
      ],
      particles: [bindGroup(d, this.particlePipe, 0, [...u, null, this.particleBuf])],
      composite: [
        bindGroup(d, this.compositePipe, 0, [...u, this.hdr.createView(), this.bloomUp[0].createView(), s, cube]),
        fieldAll(this.compositePipe),
      ],
    };
    this.mipGroups = [];
    for (let mip = 1; mip < SCENE_MIPS; mip++) {
      this.mipGroups.push(bindGroup(d, this.downPipe, 0, [this.sceneTex.createView({ baseMipLevel: mip - 1, mipLevelCount: 1 }), s]));
    }
    this.bloomGroups = { down: [], up: [] };
    this.bloomGroups.down.push(bindGroup(d, this.downKarisPipe, 0, [this.hdr.createView(), s]));
    for (let i = 1; i < BLOOM_LEVELS; i++) {
      this.bloomGroups.down.push(bindGroup(d, this.downPipe, 0, [this.bloomDown[i - 1].createView(), s]));
    }
    // up[i] = down[i] + tent(up[i + 1]), with the smallest level taken from down.
    for (let i = 0; i < BLOOM_LEVELS - 1; i++) {
      const smaller = i + 1 < BLOOM_LEVELS - 1 ? this.bloomUp[i + 1] : this.bloomDown[i + 1];
      this.bloomGroups.up.push(bindGroup(d, this.upPipe, 0, [smaller.createView(), s, this.bloomDown[i].createView()]));
    }
  }

  private objectCenter(): Vec3 {
    const p = this.params;
    return [0, OBJECT_HEIGHT + p.bob * 0.07 * Math.sin(this.fieldTime * 0.8), 0];
  }

  private writeUniforms(info: FrameInfo) {
    const p = this.params;
    const cam = this.camera;
    const center = this.objectCenter();
    const tilt = p.bob * 0.06 * Math.sin(this.fieldTime * 0.53);
    const model = modelMatrix(center, this.spinAngle, tilt, OBJECT_SCALE);
    const moonDir = vec3.normalize([-0.42, 0.34, -0.84]);
    const lightView = mat4.lookAt(vec3.scale(moonDir, 40), [0, 0, 0], [0, 1, 0]);
    const lightViewProj = mat4.multiply(mat4.orthographic(-12, 12, -12, 12, 1, 90), lightView);
    const t = this.wallTime;
    const flicker = (k: number) =>
      0.8 + 0.1 * Math.sin(t * 7.3 + k) + 0.06 * Math.sin(t * 13.1 + 2 * k) + 0.04 * Math.sin(t * 23.7 + 5 * k);

    const f = this.frameData;
    f.set(cam.viewProj, 0);
    f.set(cam.invViewProj, 16);
    f.set(model, 32);
    f.set(mat4.invert(model), 48);
    f.set(lightViewProj, 64);
    f.set([...cam.eye, t], 80);
    f.set([this.width, this.height, this.frameIndex, info.dt], 84);
    f.set([...moonDir, p.exposure], 88);
    f.set([0.5 * p.moon, 0.6 * p.moon, 0.85 * p.moon, p.bloom], 92);
    f.set([...center, OBJECT_SCALE], 96);
    f.set([...BRAZIERS[0], 4 * p.fire * flicker(0)], 100);
    f.set([...BRAZIERS[1], 4 * p.fire * flicker(1.7)], 104);
    f.set([p.debugView, FIELD_BOX, SHAPE_BOX, p.fog], 108);
    f.set([1.0, 0.42, 0.12, this.fieldTime], 112);
    f.set([this.height / (2 * Math.tan(cam.fovY / 2)), 0, 0, 0], 116);
    this.device.queue.writeBuffer(this.frameUBO, 0, f);

    const m = this.materialData;
    m.set([p.kind, p.ior, p.dispersion, p.roughness], 0);
    m.set([p.frost, p.anisotropy, p.runes, p.grooveDepth], 4);
    m.set([...p.surfaceColor, p.steps], 8);
    m.set([...p.absorption, p.energyExtinction], 12);
    m.set([...p.energyLow, p.energyIntensity], 16);
    m.set([...p.energyHigh, p.energySharpness], 20);
    m.set([p.flowSpeed, p.flowScale, p.swirl, p.grin], 24);
    m.set([p.chargeScale, p.chargeSpeed, p.surge, p.crackScale], 28);
    m.set([p.crackWidth, p.crackThreshold, p.crackIntensity, p.crackDepth], 32);
    m.set([...p.crackColor, 0], 36);
    m.set([...p.crackHot, 0], 40);
    m.set([...p.rimColor, p.rimPower], 44);
    m.set([...p.scatterColor, p.scatter], 48);
    m.set([p.spawnRate, p.particleSize, p.particleLife, p.buoyancy], 52);
    m.set([p.flowFollow, p.ejectSpeed, p.particleIntensity, p.streak], 56);
    m.set([...p.particleColor, Math.min(p.particleCount, MAX_PARTICLES)], 60);
    m.set([...p.particleHot, 0], 64);
    m.set([p.haze, p.lens, p.hazeRadius, p.rimIntensity], 68);
    m.set([p.objectLight, p.moon, p.energyDepth, 0], 72);
    this.device.queue.writeBuffer(this.materialUBO, 0, m);

    const v = this.viewData;
    v.set(cam.viewProj, 0);
    v.set(cam.invViewProj, 16);
    v.set([...cam.eye, 0], 32);
    this.device.queue.writeBuffer(this.viewUBO, 0, v);
  }

  private writeCubeViews() {
    const eye = this.objectCenter();
    const proj = mat4.perspectiveReversedInfinite(Math.PI / 2, 1, 0.05);
    CUBE_FACES.forEach((axes, i) => {
      const vp = mat4.multiply(proj, faceView(eye, axes));
      const v = new Float32Array(36);
      v.set(vp, 0);
      v.set(mat4.invert(vp), 16);
      v.set([...eye, CUBE_SIZE], 32);
      this.device.queue.writeBuffer(this.cubeViewUBOs[i], 0, v);
    });
  }

  private drawEnv(pass: GPURenderPassEncoder, env: GPUBindGroup, sky: GPUBindGroup, flame: GPUBindGroup) {
    pass.setPipeline(this.skyPipe);
    pass.setBindGroup(0, sky);
    pass.draw(3);
    pass.setPipeline(this.envPipe);
    pass.setBindGroup(0, env);
    pass.setVertexBuffer(0, this.envVB);
    pass.setIndexBuffer(this.envIB, 'uint32');
    pass.drawIndexed(this.envCount);
    pass.setPipeline(this.flamePipe);
    pass.setBindGroup(0, flame);
    pass.draw(6, 2);
  }

  private fullscreenPass(encoder: GPUCommandEncoder, label: string, view: GPUTextureView, pipe: GPURenderPipeline, group: GPUBindGroup) {
    const pass = encoder.beginRenderPass({ label, colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
  }

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    if (p.autoOrbit) this.camera.yaw += p.orbitSpeed * info.dt;
    this.wallTime += info.dt;
    this.fieldTime += info.dt * p.timeScale;
    this.spinAngle += info.dt * p.timeScale * p.spin;
    if (p.shape !== this.builtShape) this.setShape();
    if (p.crackScale !== this.builtCrackScale) this.crackDirty = true;
    this.ensureField();
    this.writeUniforms(info);
    const g = this.g;

    if (this.shadowDirty) {
      const pass = encoder.beginRenderPass({
        label: 'moon shadow', colorAttachments: [],
        depthStencilAttachment: { view: this.shadowMap.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
      });
      pass.setPipeline(this.shadowPipe);
      pass.setBindGroup(0, g.shadow[0]);
      pass.setVertexBuffer(0, this.envVB);
      pass.setIndexBuffer(this.envIB, 'uint32');
      pass.drawIndexed(this.envCount);
      pass.end();
      this.shadowDirty = false;
    }

    // --- The field: bakes, probe, particles --------------------------------
    const cp = encoder.beginComputePass({ label: 'field' });
    if (this.shapeDirty) {
      cp.setPipeline(this.shapePipe);
      cp.setBindGroup(0, g.shape[0]);
      cp.dispatchWorkgroups(SHAPE_RES / 4, SHAPE_RES / 4, SHAPE_RES / 4);
      this.shapeDirty = false;
    }
    if (this.crackDirty) {
      cp.setPipeline(this.crackPipe);
      cp.setBindGroup(0, g.crack[0]);
      cp.dispatchWorkgroups(CRACK_RES / 4, CRACK_RES / 4, CRACK_RES / 4);
      this.crackDirty = false;
      this.builtCrackScale = p.crackScale;
    }
    const fr = Math.ceil(this.fieldRes / 4);
    cp.setPipeline(this.fieldPipe);
    g.field.forEach((b, i) => cp.setBindGroup(i, b));
    cp.dispatchWorkgroups(fr, fr, fr);
    cp.setPipeline(this.probePipe);
    g.probe.forEach((b, i) => cp.setBindGroup(i, b));
    cp.dispatchWorkgroups(1);
    const count = Math.min(p.particleCount, MAX_PARTICLES);
    cp.setPipeline(this.simPipe);
    g.sim.forEach((b, i) => cp.setBindGroup(i, b));
    cp.dispatchWorkgroups(Math.ceil(count / 256));
    cp.end();

    // --- Reflection cube from the object's centre ---------------------------
    if (this.frameIndex % Math.max(1, Math.round(p.envEvery)) === 0) {
      this.writeCubeViews();
      for (let f = 0; f < 6; f++) {
        const pass = encoder.beginRenderPass({
          label: `env cube ${f}`,
          colorAttachments: [{
            view: this.envCube.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: f, arrayLayerCount: 1 }),
            loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
          }],
          depthStencilAttachment: { view: this.envDepth.createView(), depthLoadOp: 'clear', depthStoreOp: 'discard', depthClearValue: 0 },
        });
        this.drawEnv(pass, this.cubeGroups[f].env, this.cubeGroups[f].sky, this.cubeGroups[f].flame);
        pass.end();
      }
      for (let mip = 1; mip < CUBE_MIPS; mip++) {
        for (let f = 0; f < 6; f++) {
          const view = this.envCube.createView({ dimension: '2d', baseMipLevel: mip, mipLevelCount: 1, baseArrayLayer: f, arrayLayerCount: 1 });
          this.fullscreenPass(encoder, 'env filter', view, this.envFilterPipe, this.cubeFilterGroups[(mip - 1) * 6 + f]);
        }
      }
    }

    // --- Scene, then the field's distortion of it ---------------------------
    const scene = encoder.beginRenderPass({
      label: 'scene',
      colorAttachments: [{ view: this.sceneHdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
      depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 0 },
    });
    this.drawEnv(scene, g.env[0], g.sky[0], g.flame[0]);
    scene.end();

    const warp = encoder.beginRenderPass({
      label: 'warp', colorAttachments: [{ view: this.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
    });
    warp.setPipeline(this.warpPipe);
    g.warp.forEach((b, i) => warp.setBindGroup(i, b));
    warp.draw(3);
    warp.end();

    encoder.copyTextureToTexture({ texture: this.hdr }, { texture: this.sceneTex }, [this.width, this.height]);
    for (let mip = 1; mip < SCENE_MIPS; mip++) {
      const view = this.sceneTex.createView({ baseMipLevel: mip, mipLevelCount: 1 });
      this.fullscreenPass(encoder, 'scene mip', view, this.downPipe, this.mipGroups[mip - 1]);
    }

    // --- Object, then particles --------------------------------------------
    const obj = encoder.beginRenderPass({
      label: 'object',
      colorAttachments: [{ view: this.hdr.createView(), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: 'load', depthStoreOp: 'store' },
    });
    obj.setPipeline(this.objectPipe);
    g.object.forEach((b, i) => obj.setBindGroup(i, b));
    obj.setVertexBuffer(0, this.objVB);
    obj.setIndexBuffer(this.objIB, 'uint32');
    obj.drawIndexed(this.objCount);
    obj.end();

    const parts = encoder.beginRenderPass({
      label: 'particles',
      colorAttachments: [{ view: this.hdr.createView(), loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: this.depth.createView(), depthReadOnly: true },
    });
    parts.setPipeline(this.particlePipe);
    parts.setBindGroup(0, g.particles[0]);
    parts.draw(6, count);
    parts.end();

    // --- Bloom and resolve --------------------------------------------------
    this.fullscreenPass(encoder, 'bloom down 0', this.bloomDown[0].createView(), this.downKarisPipe, this.bloomGroups.down[0]);
    for (let i = 1; i < BLOOM_LEVELS; i++) {
      this.fullscreenPass(encoder, `bloom down ${i}`, this.bloomDown[i].createView(), this.downPipe, this.bloomGroups.down[i]);
    }
    for (let i = BLOOM_LEVELS - 2; i >= 0; i--) {
      this.fullscreenPass(encoder, `bloom up ${i}`, this.bloomUp[i].createView(), this.upPipe, this.bloomGroups.up[i]);
    }
    const comp = encoder.beginRenderPass({ label: 'composite', colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    comp.setPipeline(this.compositePipe);
    g.composite.forEach((b, i) => comp.setBindGroup(i, b));
    comp.draw(3);
    comp.end();

    this.frameIndex++;
  }

  /** For scripted shots and the console: `demo.applyPreset('Cursed obsidian')`. */
  applyPreset(name: string) {
    const p = this.params;
    const keep = { particleCount: p.particleCount, fieldRes: p.fieldRes, debugView: p.debugView, autoOrbit: p.autoOrbit };
    Object.assign(p, structuredClone(DEFAULTS), structuredClone(PRESETS[name] ?? {}), keep);
    this.gui.controllersRecursive().forEach((c) => c.updateDisplay());
    this.crackDirty = true;
  }

  hud() {
    const p = this.params;
    return `field ${this.fieldRes}³ · march ≤ ${p.steps} steps · ${(Math.min(p.particleCount, MAX_PARTICLES) / 1024).toFixed(0)}k particle pool\ndrag: orbit · wheel: zoom`;
  }

  destroy() {
    [...this.sizeTextures, this.shapeTex, this.crackTex, this.fieldA, this.fieldB, this.fieldG, this.envCube, this.envDepth, this.shadowMap].forEach((t) => t.destroy());
    [this.frameUBO, this.materialUBO, this.viewUBO, this.shapeUBO, ...this.cubeViewUBOs, ...this.cubeFilterUBOs,
      this.probeBuf, this.particleBuf, this.envVB, this.envIB, this.objVB, this.objIB].forEach((b) => b.destroy());
  }
}

export const magicEntry: DemoEntry = {
  id: 'magic',
  title: 'Magical materials',
  tags: ['volume march', 'procedural field', 'refraction', 'GPU particles'],
  info: `
    <h2>Procedural magical materials</h2>
    <p>One animated object-space field (curl flow, advected energy filaments,
    a slowly evolving charge and a Voronoi crack network) drives every effect:</p>
    <ul>
      <li>Interior energy ray-marched through the volume with Beer–Lambert absorption and field-bent rays</li>
      <li>Cracks that open where the charge is high, glow, and groove the surface</li>
      <li>Screen-space refraction with dispersion, frost blur and total internal reflection</li>
      <li>Rim corona and subsurface translucency through thin parts</li>
      <li>Heat haze / space warp of the background along the flow</li>
      <li>Up to 256k GPU particles born on open cracks and carried by the curl</li>
    </ul>
    <p><kbd>drag</kbd> orbit · <kbd>wheel</kbd> zoom · try the presets and debug views.</p>`,
  create: (ctx) => new MagicDemo(ctx),
};
