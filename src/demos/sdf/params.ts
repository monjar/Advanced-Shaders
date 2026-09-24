import type GUI from 'lil-gui';
import type { Vec3 } from '../../core/math';

export const DEFAULTS = {
  // Rendering and performance
  resolutionScale: 0.75,
  supersample: false,
  maxSteps: 200,
  omega: 1.5,
  epsilon: 0.6,
  maxDistance: 650,
  bounds: true,
  lod: true,
  // Lighting
  bounces: 1,
  shadows: true,
  shadowSteps: 64,
  penumbra: 12,
  ao: true,
  aoStrength: 1,
  sss: 1,
  // Sky and atmosphere
  sunElevation: 30,
  sunAzimuth: 50,
  sunIntensity: 9,
  turbidity: 2.5,
  exposure: 0.55,
  fog: 0.0022,
  fogFalloff: 0.03,
  waterMurk: 1,
  // Animation
  timeScale: 1,
  paused: false,
  walkSpeed: 1.1,
  breathing: 1,
  spin: 1,
  warp: 1,
  // Distance field slice
  sliceAxis: 0,
  sliceOffset: 2,
  sliceSpacing: 0.5,
  debugView: 0,
};

export type SdfParams = typeof DEFAULTS;

export interface ViewPreset {
  target: Vec3;
  yaw: number;
  pitch: number;
  distance: number;
  /** Keep the camera target on the walking character. */
  follow?: boolean;
}

/** Scripted viewpoints; the camera flies between them. */
export const VIEWS: Record<string, ViewPreset> = {
  'Temple entrance': { target: [0, 4.5, 4], yaw: 0.18, pitch: 0.08, distance: 24 },
  'Overlook': { target: [-6, 4, 6], yaw: 0.3, pitch: 0.36, distance: 118 },
  'Character close-up': { target: [0, 1.1, 0], yaw: 0.6, pitch: 0.12, distance: 3.6, follow: true },
  'Sculpture': { target: [-24, 6.5, 40], yaw: 0.85, pitch: 0.28, distance: 15 },
  'Rotunda': { target: [-34, 6, -4], yaw: -0.55, pitch: 0.12, distance: 26 },
  'Aqueduct and bridge': { target: [22, 5, 30], yaw: 2.45, pitch: 0.12, distance: 34 },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final': 0,
  'Steps heatmap (primary)': 1,
  'Cost heatmap (all map calls)': 2,
  'Normals': 3,
  'AO only': 4,
  'Shadows only': 5,
  'Material ids': 6,
  'Distance field slice': 7,
};

export interface GuiCallbacks {
  view(name: string): void;
  resized(): void;
}

export function buildGui(gui: GUI, p: SdfParams, cb: GuiCallbacks) {
  const view = { view: 'Temple entrance' };
  gui.add(view, 'view', Object.keys(VIEWS)).name('Viewpoint').onChange((name: string) => cb.view(name));
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');

  const perf = gui.addFolder('Ray marching and performance');
  perf.add(p, 'resolutionScale', 0.25, 1, 0.05).name('Resolution scale').onFinishChange(() => cb.resized());
  perf.add(p, 'supersample').name('2x2 supersampling (in-shader)');
  perf.add(p, 'maxSteps', 16, 512, 1).name('Max steps');
  perf.add(p, 'omega', 1, 1.95, 0.01).name('Over-relaxation ω');
  perf.add(p, 'epsilon', 0.1, 4, 0.05).name('Hit epsilon (pixels)');
  perf.add(p, 'maxDistance', 50, 1500, 10).name('Max distance m');
  perf.add(p, 'bounds').name('Bounding volumes');
  perf.add(p, 'lod').name('fBm level of detail');

  const light = gui.addFolder('Lighting');
  light.add(p, 'bounces', 0, 2, 1).name('Reflection bounces');
  light.add(p, 'shadows').name('Soft shadows');
  light.add(p, 'shadowSteps', 8, 160, 1).name('Shadow steps');
  light.add(p, 'penumbra', 2, 64, 0.5).name('Penumbra k (sharpness)');
  light.add(p, 'ao').name('Ambient occlusion');
  light.add(p, 'aoStrength', 0, 3, 0.01).name('AO strength');
  light.add(p, 'sss', 0, 2, 0.01).name('Subsurface (character)');

  const sky = gui.addFolder('Sky and atmosphere');
  sky.add(p, 'sunElevation', 2, 89, 0.1).name('Sun elevation °');
  sky.add(p, 'sunAzimuth', 0, 360, 0.1).name('Sun azimuth °');
  sky.add(p, 'sunIntensity', 0, 30, 0.1).name('Sun intensity');
  sky.add(p, 'turbidity', 0.5, 10, 0.01).name('Turbidity');
  sky.add(p, 'exposure', 0.05, 4, 0.01).name('Exposure');
  sky.add(p, 'fog', 0, 0.02, 0.0001).name('Fog density /m');
  sky.add(p, 'fogFalloff', 0, 0.2, 0.001).name('Fog height falloff /m');
  sky.add(p, 'waterMurk', 0.1, 4, 0.01).name('Water murkiness');

  const anim = gui.addFolder('Animation');
  anim.add(p, 'paused').name('Pause');
  anim.add(p, 'timeScale', 0, 3, 0.01).name('Time scale');
  anim.add(p, 'walkSpeed', 0, 2.5, 0.01).name('Walk speed m/s');
  anim.add(p, 'breathing', 0, 3, 0.01).name('Breathing');
  anim.add(p, 'spin', -3, 3, 0.01).name('Sculpture spin');
  anim.add(p, 'warp', 0, 3, 0.01).name('Sculpture warp');

  const slice = gui.addFolder('Distance field slice');
  slice.add(p, 'sliceAxis', { 'Horizontal (y)': 0, 'Vertical (z)': 1, 'Vertical (x)': 2 }).name('Plane');
  slice.add(p, 'sliceOffset', -60, 60, 0.01).name('Offset m');
  slice.add(p, 'sliceSpacing', 0.05, 5, 0.01).name('Contour spacing m');

  for (const f of [light, sky, anim, slice]) f.close();
}
