import type GUI from 'lil-gui';

export const DEFAULTS = {
  // Lighting abstraction
  lightThreshold: 0.62,
  shadowThreshold: 0.4,
  bandSoftness: 0.05,
  bandJitter: 0.12,
  lightGlaze: 0.45,
  midGlaze: 0.75,
  shadowGlaze: 0.85,
  turbulence: 0.35,
  shadowTint: [0.62, 0.66, 0.95] as number[],
  shadowTintAmount: 0.65,
  castShadows: 1,
  aerial: 0.0022,
  // Brush
  streaks: 0.35,
  streakSize: 10,
  smear: 10,
  directionJitter: 0.6,
  // Water
  bleedRadius: 14,
  wetAreas: 0.35,
  edgeDarkening: 0.9,
  // Paper
  granulation: 0.6,
  dryBrush: 0.5,
  wobble: 1.2,
  relief: 0.35,
  paperColor: [0.97, 0.95, 0.9] as number[],
  grainSize: 3.5,
  // Ink
  ink: 0,
  inkWidth: 1.1,
  inkBreakup: 0.3,
  inkColor: [0.2, 0.15, 0.12] as number[],
  creaseSensitivity: 0.25,
  // Light
  sunElevation: 38,
  sunAzimuth: 220,
  // Coherence
  noiseSpace: 0,
  paperMode: 0,
  autoOrbit: false,
  orbitSpeed: 0.15,
  measure: true,
  debugView: 0,
};

export type PaintParams = typeof DEFAULTS;

export const PRESETS: Record<string, Partial<PaintParams>> = {
  'Watercolour': {},
  'Pen and wash': { ink: 0.85, bleedRadius: 9, wetAreas: 0.25, midGlaze: 0.45, shadowGlaze: 0.9, edgeDarkening: 0.6 },
  'Wet-in-wet': { bleedRadius: 30, wetAreas: 0.7, edgeDarkening: 1.3, streaks: 0.15, bandSoftness: 0.1, dryBrush: 0.2 },
  'Dry brush': {
    bleedRadius: 4, wetAreas: 0.1, dryBrush: 1, granulation: 1, streaks: 0.6, streakSize: 7, smear: 16, edgeDarkening: 0.5,
  },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Painting': 0,
  'Pigment before bleeding': 1,
  'Light value': 2,
  'Brush direction (screen)': 3,
  'Edges (ink / wash)': 4,
  'Wet areas': 5,
  'Paper height': 6,
  'Coherent noise': 7,
};

export function buildGui(gui: GUI, p: PaintParams) {
  const preset = { preset: 'Watercolour' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    Object.assign(p, structuredClone(DEFAULTS), structuredClone(PRESETS[name]));
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  });
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');

  const coherence = gui.addFolder('Temporal coherence');
  coherence.add(p, 'noiseSpace', { 'World, depth-adaptive (coherent)': 0, 'World, fixed scale': 1, 'Screen space (shower door)': 2 }).name('Noise space');
  coherence.add(p, 'paperMode', { 'Dynamic canvas': 0, 'Fixed to screen': 1, 'Attached to surfaces': 2 }).name('Paper');
  coherence.add(p, 'autoOrbit').name('Auto orbit');
  coherence.add(p, 'orbitSpeed', -1, 1, 0.01).name('Orbit speed');
  coherence.add(p, 'measure').name('Measure reprojection error');

  const light = gui.addFolder('Lighting abstraction');
  light.add(p, 'lightThreshold', 0, 1, 0.01).name('Light threshold');
  light.add(p, 'shadowThreshold', 0, 1, 0.01).name('Shadow threshold');
  light.add(p, 'bandSoftness', 0, 0.3, 0.005).name('Band softness');
  light.add(p, 'bandJitter', 0, 0.5, 0.005).name('Band jitter');
  light.add(p, 'lightGlaze', 0, 1, 0.01).name('Light glaze');
  light.add(p, 'midGlaze', 0, 1.5, 0.01).name('Mid glaze');
  light.add(p, 'shadowGlaze', 0, 2.5, 0.01).name('Shadow glaze');
  light.add(p, 'turbulence', 0, 1, 0.01).name('Pigment turbulence');
  light.addColor(p, 'shadowTint').name('Shadow tint');
  light.add(p, 'shadowTintAmount', 0, 1, 0.01).name('Tint amount');
  light.add(p, 'castShadows', 0, 1, 0.01).name('Cast shadows');
  light.add(p, 'aerial', 0, 0.02, 0.0001).name('Aerial perspective');
  light.add(p, 'sunElevation', 5, 85, 0.5).name('Sun elevation °');
  light.add(p, 'sunAzimuth', 0, 360, 0.5).name('Sun azimuth °');

  const brush = gui.addFolder('Brush');
  brush.add(p, 'streaks', 0, 1, 0.01).name('Streaks');
  brush.add(p, 'streakSize', 2, 40, 0.5).name('Streak size px');
  brush.add(p, 'smear', 0, 40, 0.5).name('Smear length px');
  brush.add(p, 'directionJitter', 0, 2, 0.01).name('Direction jitter');

  const water = gui.addFolder('Water');
  water.add(p, 'bleedRadius', 0, 60, 0.5).name('Bleed radius px');
  water.add(p, 'wetAreas', 0, 0.8, 0.01).name('Wet areas');
  water.add(p, 'edgeDarkening', 0, 3, 0.01).name('Edge darkening');

  const paper = gui.addFolder('Paper');
  paper.add(p, 'granulation', 0, 2, 0.01).name('Granulation');
  paper.add(p, 'dryBrush', 0, 1, 0.01).name('Dry brush');
  paper.add(p, 'wobble', 0, 4, 0.01).name('Wobble');
  paper.add(p, 'relief', 0, 1, 0.01).name('Relief');
  paper.add(p, 'grainSize', 1, 10, 0.1).name('Grain size px');
  paper.addColor(p, 'paperColor').name('Paper colour');

  const ink = gui.addFolder('Ink');
  ink.add(p, 'ink', 0, 1, 0.01).name('Ink strength');
  ink.add(p, 'inkWidth', 0, 4, 0.05).name('Line width px');
  ink.add(p, 'inkBreakup', 0, 1, 0.01).name('Breakup');
  ink.add(p, 'creaseSensitivity', 0, 1, 0.01).name('Crease threshold');
  ink.addColor(p, 'inkColor').name('Ink colour');

  for (const f of [light, brush, water, paper, ink]) f.close();
}
