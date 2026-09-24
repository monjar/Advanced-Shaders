import type { Demo, DemoContext, DemoEntry, FrameInfo } from '../../core/demo';
import { bindGroup, createShader, indexBuffer, uniformBuffer, vertexBuffer } from '../../core/gpu';
import { deg, mat4, vec3, type Mat4, type Vec3 } from '../../core/math';
import { animateObjects, type ObjectDraw } from './objects';
import { DEFAULTS, PRESETS, buildGui, type PortalParams } from './params';
import {
  PORTAL_HALF_HEIGHT, PORTAL_HALF_WIDTH, insideOpening, intersectRect, m4, obliqueProjection, planeDist, planeToView, portalRect,
  type Portal, type Rect, type Vec4,
} from './portal-math';
import { LAYOUT, LOCATION_NAMES, SUN_DIRECTIONS, VERTEX_STRIDE, buildScene, locationOf, portalCupMesh, portalRingMesh, type Range } from './scene';

import commonWgsl from './shaders/common.wgsl?raw';
import portalWgsl from './shaders/portal.wgsl?raw';
import postWgsl from './shaders/post.wgsl?raw';
import sceneWgsl from './shaders/scene.wgsl?raw';

const MSAA = 4;
const SLOT = 256; // dynamic-offset alignment
const MAX_VIEWS = 64;
const VIEW_SLOTS = MAX_VIEWS + 2; // + two shadow views
const DRAW_SLOTS = 64;
const PORTAL_SLOT = 3; // draw slots 0-2: static locations, 3-10: portals, 11+: objects
const OBJECT_SLOT = 11;
const SHADOW_SIZE = 2048;
const EYE_HEIGHT = 1.65;
const EYE_OFFSET = 0.01; // the orbit camera becomes first-person at (almost) zero distance
const PLAYER_RADIUS = 0.25;
const NO_CLIP: Vec4 = [0, 0, 0, 1];
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus-stencil8';
// Rim glow, stored ÷ GLOW_SCALE (see common.wgsl): 4 bytes per sample instead of 8.
const GLOW_FORMAT: GPUTextureFormat = 'rgb10a2unorm';
const wgsl = (...parts: string[]) => parts.join('\n');

interface ViewNode {
  slot: number;
  /** Stencil level (0 for the root view). */
  level: number;
  /** Logical recursion depth (differs from `level` only in the reference view). */
  depth: number;
  view: Mat4;
  viewProj: Mat4;
  eye: Vec3;
  rect: Rect;
  location: number;
  /** Portal this view looks out of (not drawn inside it), −1 for the root. */
  exclude: number;
  clip: Vec4;
  fadeColour: Vec3;
  fade: number;
  oblique: boolean;
  children: { portal: number; rect: Rect; node: ViewNode | null }[];
}

interface ObjectInstance extends ObjectDraw {
  slot: number;
  location: number;
}

class PortalsDemo implements Demo {
  private device: GPUDevice;
  private camera: DemoContext['camera'];
  params: PortalParams = structuredClone(DEFAULTS);
  private portals: Portal[] = LAYOUT.portals;
  private width = 1;
  private height = 1;

  private viewData = new Float32Array((VIEW_SLOTS * SLOT) / 4);
  private drawData = new Float32Array((DRAW_SLOTS * SLOT) / 4);
  private globalData = new Float32Array(368);
  private postData = new Float32Array(328);
  private viewUBO: GPUBuffer;
  private drawUBO: GPUBuffer;
  private globalUBO: GPUBuffer;
  private postUBO: GPUBuffer;

  private sceneVB: GPUBuffer;
  private sceneIB: GPUBuffer;
  private cupVB: GPUBuffer;
  private cupIB: GPUBuffer;
  private cupCount: number;
  private ringVB: GPUBuffer;
  private ringIB: GPUBuffer;
  private ringCount: number;
  private locations: Range[];
  private meshes: ReturnType<typeof buildScene>['objects'];

  private shadowMap: GPUTexture;
  private shadowViews: GPUTextureView[];
  private targets: GPUTexture[] = [];
  private attach!: Record<'hdr' | 'hdrResolve' | 'rim' | 'rimResolve' | 'info' | 'infoResolve' | 'glow' | 'glowResolve' | 'depth', GPUTextureView>;

  private pipes: Record<'scene' | 'sky' | 'mask' | 'restore' | 'halo' | 'fill' | 'shadow', GPURenderPipeline>;
  private postPipe: GPURenderPipeline;
  private viewBG: GPUBindGroup;
  private drawBG: GPUBindGroup;
  private globalBG: GPUBindGroup;
  private postBG!: GPUBindGroup;
  private linearSampler: GPUSampler;

  // Camera and world state.
  private prevEye: Vec3 | null = null;
  private teleports = 0;
  private walkedSinceTeleport: number | null = null;
  private objects: ObjectInstance[] = [];
  private straddling = 0;
  private occluderCount = 0;
  private root: ViewNode | null = null;
  private viewCount = 0;
  private drawCalls = 0;
  private maxLevel = 0;
  private rects: { rect: Rect; level: number }[] = [];
  private shadowMats: Mat4[] = [];

  constructor(ctx: DemoContext) {
    const device = (this.device = ctx.device);
    this.camera = ctx.camera;
    Object.assign(this.camera, {
      distance: EYE_OFFSET, minDistance: EYE_OFFSET, minPitch: -1.35, maxPitch: 1.35, near: 0.05, fovY: deg(62), moveSpeed: 3,
    });
    buildGui(ctx.gui, this.params, (name) => this.applyPreset(name), this.portals.map((p) => p.name));

    this.viewUBO = device.createBuffer({ label: 'views', size: this.viewData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawUBO = device.createBuffer({ label: 'draws', size: this.drawData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.globalUBO = uniformBuffer(device, this.globalData.byteLength, 'portal globals');
    this.postUBO = uniformBuffer(device, this.postData.byteLength, 'post');

    const scene = buildScene();
    this.sceneVB = vertexBuffer(device, scene.vertices, 'portal scene');
    this.sceneIB = indexBuffer(device, scene.indices, 'portal scene');
    this.locations = scene.locations;
    this.meshes = scene.objects;
    const cup = portalCupMesh(1);
    this.cupVB = vertexBuffer(device, cup.positions, 'portal cup');
    this.cupIB = indexBuffer(device, cup.indices, 'portal cup');
    this.cupCount = cup.indices.length;
    const ring = portalRingMesh(0.8, 1.45);
    this.ringVB = vertexBuffer(device, ring.positions, 'portal ring');
    this.ringIB = indexBuffer(device, ring.indices, 'portal ring');
    this.ringCount = ring.indices.length;

    this.shadowMap = device.createTexture({
      label: 'sun shadows', size: [SHADOW_SIZE, SHADOW_SIZE, 2], format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowViews = [0, 1].map((layer) => this.shadowMap.createView({ dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 }));
    const shadowSampler = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });
    this.linearSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    // Explicit layouts: views and draws are dynamic-offset uniforms, so one bind
    // group each serves every view and draw (auto layouts can't be dynamic).
    const dynamicUniform = (label: string, minBindingSize: number) => device.createBindGroupLayout({
      label, entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize } }],
    });
    const viewBGL = dynamicUniform('view', 208);
    const drawBGL = dynamicUniform('draw', 96);
    const globalBGL = device.createBindGroupLayout({
      label: 'portal globals',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
      ],
    });
    this.viewBG = device.createBindGroup({ layout: viewBGL, entries: [{ binding: 0, resource: { buffer: this.viewUBO, size: 208 } }] });
    this.drawBG = device.createBindGroup({ layout: drawBGL, entries: [{ binding: 0, resource: { buffer: this.drawUBO, size: 96 } }] });
    this.globalBG = device.createBindGroup({
      layout: globalBGL,
      entries: [
        { binding: 0, resource: { buffer: this.globalUBO } },
        { binding: 1, resource: this.shadowMap.createView({ dimension: '2d-array' }) },
        { binding: 2, resource: shadowSampler },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [viewBGL, drawBGL, globalBGL] });
    const shadowLayout = device.createPipelineLayout({ bindGroupLayouts: [viewBGL, drawBGL] });

    const sceneModule = createShader(device, 'portal scene', wgsl(commonWgsl, sceneWgsl));
    const portalModule = createShader(device, 'portal passes', wgsl(commonWgsl, portalWgsl));
    const sceneVertex: GPUVertexBufferLayout = {
      arrayStride: VERTEX_STRIDE,
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
        { shaderLocation: 2, offset: 24, format: 'unorm8x4' },
      ],
    };
    const portalVertex: GPUVertexBufferLayout = { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] };

    // Every pass inside the main render pass shares its attachments:
    // 0 HDR colour, 1 rim offsets, 2 view info (level, view id, flags), 3 rim glow.
    // The glow has its own target so the rim refraction (which resamples the
    // HDR image in the post pass) does not smear it.
    type Writes = { hdr?: GPUBlendState | true; rim?: GPUBlendState; info?: true; glow?: GPUBlendState | true };
    const target = (w: Writes): GPUColorTargetState[] => {
      const t = (format: GPUTextureFormat, b?: GPUBlendState | true): GPUColorTargetState =>
        ({ format, writeMask: b ? GPUColorWrite.ALL : 0, blend: b === true ? undefined : b });
      return [t(HDR_FORMAT, w.hdr), t('rgba8unorm', w.rim), t('rgba8unorm', w.info), t(GLOW_FORMAT, w.glow)];
    };
    const stencil = (compare: GPUCompareFunction, passOp: GPUStencilOperation): GPUStencilFaceState => ({ compare, passOp, failOp: 'keep', depthFailOp: 'keep' });
    const ds = (depthCompare: GPUCompareFunction, depthWriteEnabled: boolean, passOp: GPUStencilOperation = 'keep'): GPUDepthStencilState => ({
      format: DEPTH_FORMAT, depthCompare, depthWriteEnabled,
      stencilFront: stencil('equal', passOp), stencilBack: stencil('equal', passOp), stencilReadMask: 0xff, stencilWriteMask: 0xff,
    });
    const over: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };
    const add: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };
    const pipe = (label: string, module: GPUShaderModule, vs: string, fs: string, targets: GPUColorTargetState[], depthStencil: GPUDepthStencilState, buffers: GPUVertexBufferLayout[], cullMode: GPUCullMode) =>
      device.createRenderPipeline({
        label, layout,
        vertex: { module, entryPoint: vs, buffers },
        fragment: { module, entryPoint: fs, targets },
        primitive: { topology: 'triangle-list', cullMode },
        depthStencil,
        multisample: { count: MSAA },
      });
    this.pipes = {
      scene: pipe('scene', sceneModule, 'vs', 'fs', target({ hdr: true, info: true }), ds('greater', true), [sceneVertex], 'back'),
      sky: pipe('sky + depth reset', sceneModule, 'skyVs', 'skyFs', target({ hdr: true, info: true, glow: true }), ds('always', true), [], 'none'),
      mask: pipe('portal mask', portalModule, 'portalVs', 'maskFs', target({}), ds('greater', false, 'increment-clamp'), [portalVertex], 'none'),
      restore: pipe('portal restore', portalModule, 'portalVs', 'restoreFs', target({ hdr: over, rim: over, glow: over }), ds('always', true, 'decrement-clamp'), [portalVertex], 'none'),
      halo: pipe('portal halo', portalModule, 'haloVs', 'haloFs', target({ glow: add }), ds('greater', false), [portalVertex], 'none'),
      fill: pipe('recursion end', portalModule, 'fillVs', 'fillFs', target({ hdr: true, info: true, glow: true }), ds('always', false), [], 'none'),
      shadow: device.createRenderPipeline({
        label: 'sun shadow', layout: shadowLayout,
        vertex: { module: sceneModule, entryPoint: 'shadowVs', buffers: [{ arrayStride: VERTEX_STRIDE, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
        fragment: { module: sceneModule, entryPoint: 'shadowFs', targets: [] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { format: 'depth32float', depthCompare: 'less', depthWriteEnabled: true, depthBias: 2, depthBiasSlopeScale: 2.5 },
      }),
    };
    const postModule = createShader(device, 'portal post', postWgsl);
    this.postPipe = device.createRenderPipeline({
      label: 'portal post', layout: 'auto',
      vertex: { module: postModule, entryPoint: 'vs' },
      fragment: { module: postModule, entryPoint: 'fs', targets: [{ format: ctx.format }] },
    });

    this.writeStaticData();
    this.applyPreset('Through one portal');
  }

  resize(width: number, height: number) {
    this.width = width;
    this.height = height;
    this.targets.forEach((t) => t.destroy());
    const d = this.device;
    const ms = (label: string, format: GPUTextureFormat) =>
      d.createTexture({ label, size: [width, height], format, sampleCount: MSAA, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const rs = (label: string, format: GPUTextureFormat) =>
      d.createTexture({ label, size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const t = [
      ms('hdr msaa', HDR_FORMAT), rs('hdr', HDR_FORMAT), ms('rim msaa', 'rgba8unorm'), rs('rim', 'rgba8unorm'),
      ms('info msaa', 'rgba8unorm'), rs('info', 'rgba8unorm'), ms('glow msaa', GLOW_FORMAT), rs('glow', GLOW_FORMAT),
      ms('depth stencil', DEPTH_FORMAT),
    ];
    this.targets = t;
    const [hdr, hdrResolve, rim, rimResolve, info, infoResolve, glow, glowResolve, depth] = t.map((x) => x.createView());
    this.attach = { hdr, hdrResolve, rim, rimResolve, info, infoResolve, glow, glowResolve, depth };
    this.postBG = bindGroup(d, this.postPipe, 0, [this.postUBO, hdrResolve, rimResolve, infoResolve, glowResolve, this.linearSampler]);
  }

  // --- Camera ------------------------------------------------------------------

  /** Places the first-person camera at `eye` looking at `at`. */
  setPose(eye: Vec3, at: Vec3) {
    const d = vec3.normalize(vec3.sub(at, eye));
    const cam = this.camera;
    cam.yaw = Math.atan2(-d[0], -d[2]);
    cam.pitch = -Math.asin(d[1]);
    const cp = Math.cos(cam.pitch);
    cam.target = vec3.sub(eye, vec3.scale([cp * Math.sin(cam.yaw), Math.sin(cam.pitch), cp * Math.cos(cam.yaw)], EYE_OFFSET));
    cam.update(0, this.width / this.height);
    this.prevEye = [...cam.eye];
  }

  applyPreset(name: string) {
    const preset = PRESETS[name];
    Object.assign(this.params, structuredClone(DEFAULTS), structuredClone(preset.params));
    this.walkedSinceTeleport = null;
    this.setPose(preset.camera.eye, preset.camera.at);
  }

  private updateCamera(dt: number) {
    const cam = this.camera;
    const p = this.params;
    const aspect = this.width / this.height;
    cam.distance = EYE_OFFSET;
    if (p.autoWalk !== 0) {
      cam.target = vec3.add(cam.target, vec3.scale([-Math.sin(cam.yaw), 0, -Math.cos(cam.yaw)], p.autoWalk * dt));
      // Walk on for a few metres after going through, then stop.
      if (this.walkedSinceTeleport !== null) {
        this.walkedSinceTeleport += Math.abs(p.autoWalk) * dt;
        if (this.walkedSinceTeleport > 3) {
          p.autoWalk = 0;
          this.walkedSinceTeleport = null;
        }
      }
    }
    if (p.walkMode) cam.target[1] = EYE_HEIGHT - EYE_OFFSET * Math.sin(cam.pitch);
    cam.update(0, aspect);

    const from = this.prevEye ?? cam.eye;
    const to = cam.eye;
    // Teleport when the eye crosses an opening from the front. The camera's
    // whole frame goes through the pair transform, so position and heading
    // are continuous in what the eye sees.
    if (vec3.length(vec3.sub(to, from)) < 2) {
      for (const q of this.portals) {
        if (q.location !== locationOf(from)) continue;
        const d0 = planeDist(q.plane, from);
        const d1 = planeDist(q.plane, to);
        if (d0 < 0 || d1 >= 0) continue;
        const hit = vec3.add(from, vec3.scale(vec3.sub(to, from), d0 / (d0 - d1)));
        if (!insideOpening(m4.point(q.inv, hit))) continue;
        cam.target = m4.point(q.toLinked, cam.target);
        cam.yaw += q.yawDelta;
        cam.update(0, aspect);
        this.teleports++;
        if (p.autoWalk !== 0) this.walkedSinceTeleport = 0;
        break;
      }
    }
    if (p.collisions) this.collide();
    cam.update(0, aspect);
    this.prevEye = [...cam.eye];
  }

  /** Circle-vs-box push-out in xz; the wall holding a portal is ignored while walking through its opening. */
  private collide() {
    const cam = this.camera;
    const loc = locationOf(cam.eye);
    const skip = new Set<number>();
    for (const q of this.portals) {
      if (q.location !== loc) continue;
      const l = m4.point(q.inv, cam.eye);
      const halfWidth = PORTAL_HALF_WIDTH * Math.sqrt(Math.max(0, 1 - (l[1] / PORTAL_HALF_HEIGHT) ** 2));
      if (l[2] > -0.1 && l[2] < 0.7 && Math.abs(l[0]) < halfWidth - 0.12) skip.add(q.host);
    }
    const t = cam.target;
    LAYOUT.colliders[loc].forEach((b, i) => {
      if (skip.has(i)) return;
      const cx = Math.min(Math.max(t[0], b.x0), b.x1);
      const cz = Math.min(Math.max(t[2], b.z0), b.z1);
      const dx = t[0] - cx;
      const dz = t[2] - cz;
      const dist = Math.hypot(dx, dz);
      if (dist >= PLAYER_RADIUS) return;
      if (dist > 1e-6) {
        t[0] = cx + (dx / dist) * PLAYER_RADIUS;
        t[2] = cz + (dz / dist) * PLAYER_RADIUS;
      } else {
        // Centre inside the box: leave through the nearest side.
        const exits = [t[0] - b.x0, b.x1 - t[0], t[2] - b.z0, b.z1 - t[2]];
        const k = exits.indexOf(Math.min(...exits));
        if (k === 0) t[0] = b.x0 - PLAYER_RADIUS; else if (k === 1) t[0] = b.x1 + PLAYER_RADIUS;
        else if (k === 2) t[2] = b.z0 - PLAYER_RADIUS; else t[2] = b.z1 + PLAYER_RADIUS;
      }
    });
    if (loc === 2) {
      const { centre, radius } = LAYOUT.clearing;
      const dx = t[0] - centre[0];
      const dz = t[2] - centre[1];
      const r = Math.hypot(dx, dz);
      if (r > radius) {
        t[0] = centre[0] + (dx / r) * radius;
        t[2] = centre[1] + (dz / r) * radius;
      }
    }
  }

  // --- View tree ------------------------------------------------------------------

  /**
   * Builds the tree of views: the camera, and for every portal visible in a
   * view (in front of its camera, beyond its clip plane, and overlapping its
   * scissor rectangle) a virtual camera behind the linked portal. The
   * recursion stops at the depth limit, when the view budget is used up, or
   * when the portal's rectangle gets smaller than a few pixels.
   */
  private buildViews(): ViewNode {
    const cam = this.camera;
    const p = this.params;
    const full: Rect = [0, 0, this.width, this.height];
    let slot = 0;
    const budget = Math.min(p.maxViews, MAX_VIEWS);

    const makeNode = (view: Mat4, proj: Mat4, eye: Vec3, rect: Rect, level: number, depth: number, exclude: number, clip: Vec4, via: Portal | null, oblique: boolean): ViewNode => ({
      slot: slot++, level, depth, view, viewProj: mat4.multiply(proj, view), eye, rect, location: locationOf(eye), exclude, clip,
      fadeColour: via ? vec3.scale(via.colour, 0.45) : [0, 0, 0],
      fade: Math.min(1, Math.max(0, (depth - 1) / p.maxDepth)) ** 2,
      oblique, children: [],
    });

    /** Virtual camera looking through `q` from a view with matrix `view`. */
    const through = (q: Portal, view: Mat4, eye: Vec3) => {
      const dest = this.portals[q.link];
      // view · toLinked⁻¹; the inverse pair transform is the partner's pair transform.
      const v2 = mat4.multiply(view, dest.toLinked);
      const eye2 = m4.point(q.toLinked, eye);
      let proj: Mat4 | null = cam.proj;
      let oblique = false;
      // Within a near-plane distance of the plane the ordinary near plane
      // already clips everything between the eye and the opening (inside the
      // carved hole), and the oblique matrix would lose its depth precision.
      if (p.obliqueClip && -planeDist(dest.plane, eye2) > cam.near) {
        proj = obliqueProjection(cam.proj, planeToView(v2, dest.plane));
        oblique = true;
      }
      return { view: v2, eye: eye2, proj, oblique, dest };
    };

    const expand = (node: ViewNode) => {
      const candidates: { q: Portal; rect: Rect; full: Rect; dist: number }[] = [];
      for (const q of this.portals) {
        if (q.location !== node.location || q.index === node.exclude) continue;
        if (planeDist(q.plane, node.eye) <= 0) continue;
        if (!q.outline.some((pt) => planeDist(node.clip, pt) > 0)) continue;
        const full = portalRect(node.view, cam.proj, q.outline, this.width, this.height);
        const rect = full && intersectRect(full, node.rect);
        if (!full || !rect) continue;
        candidates.push({ q, rect, full, dist: vec3.length(vec3.sub(q.center, node.eye)) });
      }
      candidates.sort((a, b) => a.dist - b.dist);
      for (const { q, rect, full } of candidates) {
        // Size on screen, not the part left inside the parent's rectangle.
        const small = full[2] - full[0] < p.minRectPx || full[3] - full[1] < p.minRectPx;
        if (node.depth + 1 > p.maxDepth || slot >= budget || small) {
          node.children.push({ portal: q.index, rect, node: null });
          continue;
        }
        const t = through(q, node.view, node.eye);
        if (!t.proj) {
          // No view direction reaches beyond the destination (only at grazing
          // angles, by rounding): fill rather than leave the carved hole open.
          node.children.push({ portal: q.index, rect, node: null });
          continue;
        }
        const child = makeNode(t.view, t.proj, t.eye, rect, node.level + 1, node.depth + 1, t.dest.index, t.dest.plane, q, t.oblique);
        node.children.push({ portal: q.index, rect, node: child });
        expand(child);
      }
    };

    let root: ViewNode;
    const ref = p.referenceView;
    if (ref >= 0 && ref < this.portals.length) {
      // Reference: render full screen from the virtual camera of one portal,
      // exactly as the first recursion level would (to check parallax).
      const q = this.portals[ref];
      const t = through(q, cam.view, cam.eye);
      root = makeNode(t.view, t.proj ?? cam.proj, t.eye, full, 0, 1, t.dest.index, t.dest.plane, null, t.oblique);
    } else {
      root = makeNode(cam.view, cam.proj, cam.eye, full, 0, 0, -1, NO_CLIP, null, false);
    }
    expand(root);
    this.viewCount = slot;
    return root;
  }

  // --- Uniforms --------------------------------------------------------------------

  private writeStaticData() {
    const g = this.globalData;
    this.portals.forEach((q, i) => {
      const o = i * 32;
      g.set(q.inv, o);
      g.set(q.plane, o + 16);
      g.set([...q.center, q.location], o + 20);
      g.set([...q.colour, q.carve], o + 24);
      // Radiance arriving through the portal, a rough average of its destination.
      const dest = this.portals[q.link].location;
      const spill: Vec3 = dest === 0 ? [1.6, 1.1, 0.7] : dest === 1 ? [0.25, 0.33, 0.42] : [0.65, 0.82, 0.68];
      g.set([...spill, 0], o + 28);
    });
    LAYOUT.hangarLights.forEach((l, i) => g.set([...l.pos, 70], 256 + i * 4));

    // Sun shadow maps for the courtyard and the forest.
    this.shadowMats = [
      { centre: [0, 0, 0] as Vec3, r: 15, s: SUN_DIRECTIONS[0] },
      { centre: [-100, 0, 0] as Vec3, r: 30, s: SUN_DIRECTIONS[2] },
    ].map(({ centre, r, s }) => {
      const view = mat4.lookAt(vec3.add(centre, vec3.scale(s, 60)), centre, [0, 1, 0]);
      return mat4.multiply(mat4.orthographic(-r, r, -r, r, 1, 120), view);
    });
    g.set(this.shadowMats[0], 328);
    g.set(this.shadowMats[1], 344);

    // Static draws: one per location, carving portal holes.
    for (let i = 0; i < 3; i++) this.writeDraw(i, mat4.identity(), NO_CLIP, [1, 0, 0, 0]);
    this.portals.forEach((q, i) => this.writeDraw(PORTAL_SLOT + i, q.world, NO_CLIP, [0, 0, i, 0]));
  }

  private writeDraw(slot: number, model: Mat4, clip: Vec4, flags: number[]) {
    const o = (slot * SLOT) / 4;
    this.drawData.set(model, o);
    this.drawData.set(clip, o + 16);
    this.drawData.set(flags, o + 20);
  }

  /** View uniforms; `node` omitted for the shadow views (only the matrix is used). */
  private writeView(slot: number, viewProj: Mat4, view: Mat4, eye: Vec3, node?: ViewNode) {
    const o = (slot * SLOT) / 4;
    const v = this.viewData;
    v.set(viewProj, o);
    v.set(mat4.invert(view), o + 16);
    v.set(eye, o + 32);
    if (!node) return;
    const tanY = Math.tan(this.camera.fovY / 2);
    v[o + 35] = node.level;
    v.set(node.clip, o + 36);
    v.set([...node.fadeColour, node.fade], o + 40);
    v.set([tanY * (this.width / this.height), tanY, node.slot, node.location], o + 44);
    v.set([this.width, this.height, node.oblique ? 1 : 0, 0], o + 48);
  }

  private writeFrameData(time: number) {
    const p = this.params;
    // Views (depth-first) and the scissor rectangles for the overlay.
    this.rects = [];
    this.maxLevel = 0;
    const walk = (n: ViewNode) => {
      this.writeView(n.slot, n.viewProj, n.view, n.eye, n);
      this.maxLevel = Math.max(this.maxLevel, n.level);
      if (n.level > 0) this.rects.push({ rect: n.rect, level: n.level });
      n.children.forEach((c) => c.node && walk(c.node));
    };
    walk(this.root!);
    this.shadowMats.forEach((m, i) => this.writeView(MAX_VIEWS + i, m, mat4.identity(), [0, 0, 0]));
    this.device.queue.writeBuffer(this.viewUBO, 0, this.viewData, 0, (VIEW_SLOTS * SLOT) / 4);

    this.objects.forEach((o) => this.writeDraw(o.slot, o.model, o.clip, [0, o.crossing * 0.5, 0, 1]));
    this.device.queue.writeBuffer(this.drawUBO, 0, this.drawData);

    const g = this.globalData;
    g.set([time, p.lightSpill, p.fog, p.rimDistortion], 360);
    g.set([p.rimWidth, p.glow, this.occluderCount, p.debugView], 364);
    this.device.queue.writeBuffer(this.globalUBO, 0, g);

    const q = this.postData;
    q.set([this.width, this.height, time, p.debugView, p.exposure, p.showRects ? 1 : 0, Math.min(this.rects.length, 64), 0], 0);
    this.rects.slice(0, 64).forEach((r, i) => {
      q.set(r.rect, 8 + i * 4);
      q[264 + i] = r.level;
    });
    this.device.queue.writeBuffer(this.postUBO, 0, q);
  }

  // --- Frame ---------------------------------------------------------------------

  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo) {
    const p = this.params;
    this.updateCamera(info.dt);
    if (p.animate) p.objectTime = (p.objectTime + info.dt * p.objectSpeed) % 1000;

    const anim = animateObjects(p.objectTime, this.meshes, this.portals, p.objectClip);
    this.straddling = anim.straddling;
    this.objects = anim.draws.slice(0, DRAW_SLOTS - OBJECT_SLOT).map((d, i) => ({
      ...d, slot: OBJECT_SLOT + i, location: locationOf(m4.point(d.model, [0, 0.5, 0])),
    }));
    const occ = anim.occluders.slice(0, 12);
    occ.forEach((o, i) => this.globalData.set([...o.centre, o.radius], 280 + i * 4));
    this.occluderCount = occ.length;

    this.root = this.buildViews();
    this.writeFrameData(info.time);
    this.drawCalls = 0;

    // Sun shadows (courtyard, forest).
    [0, 2].forEach((loc, layer) => {
      const pass = encoder.beginRenderPass({
        label: `shadow ${LOCATION_NAMES[loc]}`, colorAttachments: [],
        depthStencilAttachment: { view: this.shadowViews[layer], depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      pass.setPipeline(this.pipes.shadow);
      pass.setBindGroup(0, this.viewBG, [(MAX_VIEWS + layer) * SLOT]);
      pass.setVertexBuffer(0, this.sceneVB);
      pass.setIndexBuffer(this.sceneIB, 'uint32');
      this.drawLocation(pass, loc);
      pass.end();
    });

    const a = this.attach;
    const pass = encoder.beginRenderPass({
      label: 'portal views',
      colorAttachments: [
        { view: a.hdr, resolveTarget: a.hdrResolve, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 1] },
        { view: a.rim, resolveTarget: a.rimResolve, loadOp: 'clear', storeOp: 'discard', clearValue: [128 / 255, 128 / 255, 0, 0] },
        { view: a.info, resolveTarget: a.infoResolve, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] },
        { view: a.glow, resolveTarget: a.glowResolve, loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: {
        view: a.depth, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard',
        stencilClearValue: 0, stencilLoadOp: 'clear', stencilStoreOp: 'discard',
      },
    });
    pass.setBindGroup(2, this.globalBG);
    this.drawView(pass, this.root);
    pass.end();

    const post = encoder.beginRenderPass({ label: 'portal post', colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    post.setPipeline(this.postPipe);
    post.setBindGroup(0, this.postBG);
    post.draw(3);
    post.end();
  }

  /** Static geometry of one location plus the objects currently in it. */
  private drawLocation(pass: GPURenderPassEncoder, loc: number) {
    const r = this.locations[loc];
    pass.setBindGroup(1, this.drawBG, [loc * SLOT]);
    pass.drawIndexed(r.count, 1, r.first);
    this.drawCalls++;
    for (const o of this.objects) {
      if (o.location !== loc) continue;
      pass.setBindGroup(1, this.drawBG, [o.slot * SLOT]);
      pass.drawIndexed(o.range.count, 1, o.range.first);
      this.drawCalls++;
    }
  }

  /**
   * Renders a view inside its stencil mask (stencil == level): sky + depth
   * reset, the location's geometry, then every portal it sees: mask
   * (level → level+1), the portal's view (recursively) or the end-of-recursion
   * fill, restore (level+1 → level, portal depth, fog, rim offsets), halo.
   */
  private drawView(pass: GPURenderPassEncoder, node: ViewNode) {
    const P = this.pipes;
    const setRect = (r: Rect) => pass.setScissorRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
    const bindView = () => pass.setBindGroup(0, this.viewBG, [node.slot * SLOT]);
    const bindPortal = (i: number) => pass.setBindGroup(1, this.drawBG, [(PORTAL_SLOT + i) * SLOT]);
    const cup = () => {
      pass.setVertexBuffer(0, this.cupVB);
      pass.setIndexBuffer(this.cupIB, 'uint16');
      pass.drawIndexed(this.cupCount);
      this.drawCalls++;
    };

    setRect(node.rect);
    pass.setStencilReference(node.level);
    bindView();
    pass.setPipeline(P.sky);
    bindPortal(0); // unused by the sky, but group 1 must be bound
    pass.draw(3);
    pass.setPipeline(P.scene);
    pass.setVertexBuffer(0, this.sceneVB);
    pass.setIndexBuffer(this.sceneIB, 'uint32');
    this.drawLocation(pass, node.location);

    for (const child of node.children) {
      setRect(node.rect);
      pass.setStencilReference(node.level);
      bindView();
      bindPortal(child.portal);
      pass.setPipeline(P.mask);
      cup();
      if (child.node) {
        this.drawView(pass, child.node);
      } else {
        setRect(child.rect);
        pass.setStencilReference(node.level + 1);
        bindView();
        bindPortal(child.portal);
        pass.setPipeline(P.fill);
        pass.draw(3);
        this.drawCalls++;
      }
      setRect(node.rect);
      pass.setStencilReference(node.level + 1);
      bindView();
      bindPortal(child.portal);
      pass.setPipeline(P.restore);
      cup();
      pass.setStencilReference(node.level);
      pass.setPipeline(P.halo);
      pass.setVertexBuffer(0, this.ringVB);
      pass.setIndexBuffer(this.ringIB, 'uint16');
      pass.drawIndexed(this.ringCount);
      this.drawCalls++;
    }
  }

  hud() {
    const loc = LOCATION_NAMES[locationOf(this.camera.eye)];
    return `${this.viewCount} views · deepest level ${this.maxLevel} · ${this.drawCalls} draws\n` +
      `${loc} · teleports ${this.teleports} · objects crossing ${this.straddling}\n` +
      'drag: look · WASD: walk (shift: run) · walk into a portal to go through';
  }

  destroy() {
    [...this.targets, this.shadowMap].forEach((t) => t.destroy());
    [this.viewUBO, this.drawUBO, this.globalUBO, this.postUBO, this.sceneVB, this.sceneIB, this.cupVB, this.cupIB, this.ringVB, this.ringIB].forEach((b) => b.destroy());
  }
}

export const portalsEntry: DemoEntry = {
  id: 'portals',
  title: 'Portals',
  tags: ['stencil recursion', 'oblique clipping', 'teleport'],
  info: `
    <h2>Portals with recursive rendering</h2>
    <ul>
      <li>Three locations joined only by portals; the view through each is rendered from a virtual camera</li>
      <li>Oblique near-plane clipping (Lengyel 2005) for the reversed-Z projection</li>
      <li>Stencil recursion up to 8 levels, scissored to each portal's rectangle</li>
      <li>Objects crossing a portal are clipped and duplicated on the other side</li>
      <li>Walk through: the camera teleports with no visible seam</li>
      <li>Energy rim: refraction only in a thin band at the edge, light spill</li>
    </ul>
    <p><kbd>drag</kbd> look · <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> walk · <kbd>Shift</kbd> run<br>
    Try the presets, <em>Recursion depth</em> view and the oblique-clipping toggle.</p>`,
  create: (ctx) => new PortalsDemo(ctx),
};
