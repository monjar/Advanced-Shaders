import type GUI from 'lil-gui';
import type { Vec3 } from '../../core/math';

export const DEFAULTS = {
  // Recursion
  maxDepth: 5,
  maxViews: 48,
  minRectPx: 6,
  obliqueClip: true,
  // Portal look
  rimWidth: 0.085,
  rimDistortion: 1.0,
  glow: 1.0,
  lightSpill: 1.5,
  fog: 1.0,
  exposure: 1.0,
  // Objects
  animate: true,
  objectSpeed: 1.0,
  objectTime: 0,
  objectClip: true,
  // Camera
  walkMode: true,
  collisions: true,
  autoWalk: 0,
  // Debug
  debugView: 0,
  showRects: false,
  /** Render the root view from the virtual camera behind this portal's partner (−1 = off). */
  referenceView: -1,
};

export type PortalParams = typeof DEFAULTS;

export interface CameraPose {
  eye: Vec3;
  at: Vec3;
}

export interface Preset {
  camera: CameraPose;
  params: Partial<PortalParams>;
}

export const PRESETS: Record<string, Preset> = {
  'Through one portal': { camera: { eye: [-1.2, 1.65, -5.6], at: [-3, 1.4, -9.99] }, params: { maxDepth: 4 } },
  'Recursive corridor': { camera: { eye: [106.35, 1.65, -5.2], at: [106, 1.45, -8.99] }, params: { maxDepth: 8 } },
  'Object crossing': { camera: { eye: [-99.6, 1.65, -3.6], at: [-103, 0.9, -6.64] }, params: { animate: false, objectTime: 8.0 } },
  'Walk through': { camera: { eye: [-3.1, 1.65, -5.5], at: [-3, 1.55, -9.99] }, params: { autoWalk: 1.3 } },
  'Portals in portals': { camera: { eye: [5, 1.65, 6.2], at: [-6, 1.6, -4] }, params: { maxDepth: 6 } },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final': 0,
  'Recursion depth': 1,
  'Stencil masks': 2,
  'Crossing duplicates': 3,
};

export function buildGui(gui: GUI, p: PortalParams, applyPreset: (name: string) => void, portalNames: string[]) {
  const preset = { preset: 'Through one portal' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    applyPreset(name);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  });
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');
  gui.add(p, 'showRects').name('Scissor rectangles');
  gui.add(p, 'obliqueClip').name('Oblique near plane');

  const rec = gui.addFolder('Recursion');
  rec.add(p, 'maxDepth', 1, 8, 1).name('Max depth');
  rec.add(p, 'maxViews', 1, 64, 1).name('View budget');
  rec.add(p, 'minRectPx', 0, 64, 1).name('Min portal size px');
  const refs: Record<string, number> = { 'Off': -1 };
  portalNames.forEach((n, i) => (refs[`Behind ${n}'s partner`] = i));
  rec.add(p, 'referenceView', refs).name('Reference camera');

  const look = gui.addFolder('Portal look');
  look.add(p, 'rimWidth', 0.02, 0.2, 0.005).name('Rim width');
  look.add(p, 'rimDistortion', 0, 2, 0.01).name('Rim refraction');
  look.add(p, 'glow', 0, 3, 0.01).name('Glow');
  look.add(p, 'lightSpill', 0, 3, 0.01).name('Light spill');
  look.add(p, 'fog', 0, 3, 0.01).name('Fog');
  look.add(p, 'exposure', 0.2, 3, 0.01).name('Exposure');

  const obj = gui.addFolder('Objects');
  obj.add(p, 'animate').name('Animate');
  obj.add(p, 'objectSpeed', 0, 3, 0.01).name('Speed');
  obj.add(p, 'objectTime', 0, 30, 0.01).name('Time').listen();
  obj.add(p, 'objectClip').name('Clip planes');

  const cam = gui.addFolder('Camera');
  cam.add(p, 'walkMode').name('Walk (eye height 1.65 m)');
  cam.add(p, 'collisions').name('Collisions');
  cam.add(p, 'autoWalk', 0, 4, 0.01).name('Auto walk m/s');

  for (const f of [look, obj, cam]) f.close();
}
