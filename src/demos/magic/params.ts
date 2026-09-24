import type GUI from 'lil-gui';
import { SHAPES } from './geometry';

// Every visual quantity below is a *response* to the one procedural field
// (flow velocity, energy, charge, crack network). Presets only change how
// the material reads the field, never add an independent animation.
export const DEFAULTS = {
  shape: 1,
  // Surface
  kind: 0,                 // 0 dielectric (refractive), 1 metal (opaque, anisotropic)
  ior: 1.54,
  dispersion: 0.03,
  roughness: 0.03,
  frost: 0,                // blur of the refracted background (scene mip level)
  anisotropy: 0,
  surfaceColor: [0.9, 0.95, 1.0] as number[],
  absorption: [0.55, 0.22, 0.08] as number[],   // Beer–Lambert, per object unit
  // Interior energy
  energyLow: [0.1, 0.75, 1.6] as number[],      // colour at low charge
  energyHigh: [0.75, 0.25, 1.9] as number[],    // colour at high charge
  energyIntensity: 5,
  energySharpness: 7,
  energyExtinction: 0.8,
  energyDepth: 0.05,       // energy hides this deep below the surface (object units): a clear skin over a glowing core
  steps: 40,
  grin: 0.35,              // gradient-index bending of the interior march by the charge field
  // Field
  flowSpeed: 1,
  flowScale: 2.2,
  swirl: 0.6,
  chargeScale: 1.1,
  chargeSpeed: 0.12,
  surge: 0.6,
  // Cracks
  crackScale: 2.6,
  crackWidth: 0.018,
  crackThreshold: 0.74,
  crackColor: [0.35, 0.9, 1.6] as number[],
  crackHot: [1.4, 1.6, 2.0] as number[],
  crackIntensity: 3,
  crackDepth: 0.14,        // how deep the glowing fracture sheets reach into the volume
  grooveDepth: 0.6,
  runes: 0,
  // Rim, scattering
  rimColor: [0.35, 0.6, 1.5] as number[],
  rimPower: 3.5,
  rimIntensity: 0.9,
  scatterColor: [0.5, 0.8, 1.0] as number[],
  scatter: 0.03,
  // Particles
  particleCount: 65536,
  spawnRate: 1,
  particleSize: 0.012,
  particleLife: 2.6,
  buoyancy: 0.05,
  flowFollow: 1,
  ejectSpeed: 0.25,
  particleIntensity: 0.8,
  particleColor: [0.4, 0.85, 1.6] as number[],
  particleHot: [1.2, 0.6, 2.0] as number[],
  streak: 0.8,
  // Distortion
  haze: 0.5,
  lens: 0,
  hazeRadius: 1.7,
  // Object light cast on the altar, derived from the field (see probe)
  objectLight: 1,
  // Scene
  exposure: 1,
  bloom: 0.06,
  moon: 0.9,
  fire: 1,
  fog: 0.035,
  // Motion
  timeScale: 1,
  spin: 0.12,
  bob: 1,
  autoOrbit: false,
  orbitSpeed: 0.12,
  // Performance
  fieldRes: 72,
  envEvery: 2,
  debugView: 0,
};

export type MagicParams = typeof DEFAULTS;

/** Parameters that force a re-bake of the crack network or shape SDF. */
export const CRACK_KEYS: (keyof MagicParams)[] = ['crackScale'];

export const PRESETS: Record<string, Partial<MagicParams>> = {
  'Enchanted crystal': {},
  'Cursed obsidian': {
    shape: 0, ior: 1.49, dispersion: 0.01, roughness: 0.035, surfaceColor: [0.06, 0.05, 0.05],
    absorption: [3.2, 4.2, 5.0],
    energyLow: [1.4, 0.18, 0.02], energyHigh: [2.2, 0.8, 0.12], energyIntensity: 0.5, energyDepth: 0.1, energySharpness: 5, energyExtinction: 0.3,
    grin: 0.2, flowSpeed: 0.7, flowScale: 2.0, swirl: 0.35, chargeScale: 1.2, chargeSpeed: 0.1, surge: 0.8,
    crackScale: 2.1, crackWidth: 0.03, crackThreshold: 0.68, crackColor: [2.4, 0.42, 0.05], crackHot: [3.2, 1.9, 0.7],
    crackIntensity: 3, crackDepth: 0.07, grooveDepth: 1.1,
    rimColor: [1.2, 0.25, 0.05], rimPower: 4, rimIntensity: 0.5, scatterColor: [1.0, 0.3, 0.1], scatter: 0.1,
    spawnRate: 0.5, particleSize: 0.011, particleLife: 3.2, buoyancy: 0.7, flowFollow: 0.5, ejectSpeed: 0.35,
    particleIntensity: 1.2, particleColor: [2.0, 0.35, 0.05], particleHot: [2.4, 1.3, 0.4], streak: 1.0,
    haze: 1.2, lens: 0, hazeRadius: 1.9, objectLight: 0.8,
  },
  'Arcane metal': {
    shape: 2, kind: 1, roughness: 0.2, anisotropy: 0.85, surfaceColor: [0.78, 0.76, 0.8],
    energyLow: [1.6, 0.9, 0.25], energyHigh: [2.2, 1.7, 0.8], energyIntensity: 3, energySharpness: 6,
    flowSpeed: 1.2, flowScale: 2.4, swirl: 1.2, chargeScale: 1.3, chargeSpeed: 0.15, surge: 0.7,
    crackScale: 2.4, crackWidth: 0.009, crackThreshold: 0.5, crackColor: [2.0, 1.1, 0.3], crackHot: [2.6, 2.2, 1.4],
    crackIntensity: 3.5, grooveDepth: 1.4, runes: 1,
    rimColor: [1.0, 0.6, 0.2], rimPower: 5, rimIntensity: 0.6,
    spawnRate: 0.3, particleSize: 0.009, particleLife: 2.2, buoyancy: 0.15, flowFollow: 1.0, ejectSpeed: 0.12,
    particleIntensity: 0.9, particleColor: [2.0, 1.2, 0.35], particleHot: [2.4, 2.1, 1.2], streak: 0.7,
    haze: 0.35, hazeRadius: 1.6,
  },
  'Frozen soul ice': {
    shape: 1, ior: 1.31, dispersion: 0.012, roughness: 0.12, frost: 2.2, surfaceColor: [0.85, 0.95, 1.0],
    absorption: [1.2, 0.45, 0.2],
    energyLow: [0.55, 0.95, 1.1], energyHigh: [0.9, 1.1, 1.3], energyIntensity: 1.6, energySharpness: 9, energyExtinction: 1.6,
    grin: 0.15, flowSpeed: 0.45, flowScale: 1.8, swirl: 1.4, chargeScale: 0.9, chargeSpeed: 0.08, surge: 0.4,
    crackScale: 3.2, crackWidth: 0.01, crackThreshold: 0.4, crackColor: [0.7, 0.9, 1.1], crackHot: [1.0, 1.1, 1.2],
    crackIntensity: 1.2, crackDepth: 0.35, grooveDepth: 0.3,
    rimColor: [0.6, 0.85, 1.2], rimPower: 2.5, rimIntensity: 1.1, scatterColor: [0.6, 0.85, 1.0], scatter: 0.35,
    spawnRate: 0.35, particleSize: 0.01, particleLife: 3, buoyancy: -0.12, flowFollow: 0.8, ejectSpeed: 0.08,
    particleIntensity: 0.5, particleColor: [0.7, 0.9, 1.2], particleHot: [1.0, 1.1, 1.3], streak: 0.4,
    haze: 0.25, hazeRadius: 1.6, objectLight: 0.6,
  },
  'Void stone': {
    shape: 0, ior: 1.7, dispersion: 0.05, roughness: 0.06, surfaceColor: [0.08, 0.06, 0.1],
    absorption: [3.0, 4.0, 2.5],
    energyLow: [0.6, 0.1, 1.4], energyHigh: [1.8, 0.3, 1.2], energyIntensity: 0.6, energyDepth: 0.22, energySharpness: 8, energyExtinction: 0.5,
    grin: 1.2, flowSpeed: 0.8, flowScale: 2.6, swirl: 1.2, chargeScale: 1.0, chargeSpeed: 0.1, surge: 0.5,
    crackScale: 2.4, crackWidth: 0.01, crackThreshold: 0.8, crackColor: [1.2, 0.3, 1.8], crackHot: [1.8, 1.2, 2.2],
    crackIntensity: 2, crackDepth: 0.25, grooveDepth: 0.5,
    rimColor: [0.9, 0.2, 1.6], rimPower: 3, rimIntensity: 1.0, scatterColor: [0.6, 0.2, 1.0], scatter: 0.2,
    spawnRate: 0.6, particleSize: 0.009, particleLife: 3, buoyancy: 0, flowFollow: 0.9, ejectSpeed: -0.05,
    particleIntensity: 0.6, particleColor: [0.9, 0.25, 1.8], particleHot: [1.6, 0.9, 2.0], streak: 0.8,
    haze: 0.3, lens: 1.4, hazeRadius: 2.2, objectLight: 0.7,
  },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final': 0,
  'Field slice (E red, C green, |v| blue)': 1,
  'Crack mask': 2,
  'Thickness': 3,
  'Interior march steps': 4,
  'Particle density': 5,
  'Refraction offsets': 6,
  'Distortion offsets': 7,
  'Reflection env map': 8,
};

export function buildGui(gui: GUI, p: MagicParams, onPreset: () => void) {
  const preset = { preset: 'Enchanted crystal' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    const keep = { particleCount: p.particleCount, fieldRes: p.fieldRes, debugView: p.debugView, autoOrbit: p.autoOrbit };
    Object.assign(p, structuredClone(DEFAULTS), structuredClone(PRESETS[name]), keep);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    onPreset();
  });
  gui.add(p, 'shape', SHAPES).name('Shape');
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');

  const surf = gui.addFolder('Surface');
  surf.add(p, 'kind', { 'Dielectric (refractive)': 0, 'Metal (opaque)': 1 }).name('Kind');
  surf.add(p, 'ior', 1, 2.4, 0.01).name('IOR');
  surf.add(p, 'dispersion', 0, 0.1, 0.001).name('Dispersion');
  surf.add(p, 'roughness', 0.01, 0.8, 0.01).name('Roughness');
  surf.add(p, 'frost', 0, 5, 0.05).name('Frost (refraction blur)');
  surf.add(p, 'anisotropy', 0, 0.95, 0.01).name('Anisotropy');
  surf.addColor(p, 'surfaceColor').name('Tint / F0');
  surf.addColor(p, 'absorption').name('Absorption');

  const inner = gui.addFolder('Interior energy');
  inner.addColor(p, 'energyLow').name('Colour (low charge)');
  inner.addColor(p, 'energyHigh').name('Colour (high charge)');
  inner.add(p, 'energyIntensity', 0, 8, 0.05).name('Intensity');
  inner.add(p, 'energySharpness', 1, 16, 0.05).name('Filament sharpness');
  inner.add(p, 'energyExtinction', 0, 4, 0.05).name('Energy extinction');
  inner.add(p, 'energyDepth', 0, 0.6, 0.005).name('Clear skin depth');
  inner.add(p, 'grin', 0, 3, 0.01).name('Field refraction (GRIN)');
  inner.add(p, 'steps', 8, 96, 1).name('Max march steps');
  inner.addColor(p, 'scatterColor').name('Scatter colour');
  inner.add(p, 'scatter', 0, 3, 0.01).name('Subsurface scatter');

  const field = gui.addFolder('Field');
  field.add(p, 'flowSpeed', 0, 3, 0.01).name('Flow speed');
  field.add(p, 'flowScale', 0.5, 5, 0.01).name('Energy scale');
  field.add(p, 'swirl', 0, 4, 0.01).name('Swirl');
  field.add(p, 'chargeScale', 0.3, 3, 0.01).name('Charge scale');
  field.add(p, 'chargeSpeed', 0, 1, 0.005).name('Charge speed');
  field.add(p, 'surge', 0, 2, 0.01).name('Charge surges');
  field.add(p, 'fieldRes', [48, 64, 72, 96, 128]).name('Bake resolution');

  const cracks = gui.addFolder('Cracks / runes');
  cracks.add(p, 'crackScale', 1, 6, 0.05).name('Cell scale');
  cracks.add(p, 'crackWidth', 0.002, 0.06, 0.001).name('Width');
  cracks.add(p, 'crackThreshold', 0, 1, 0.01).name('Open threshold (charge)');
  cracks.addColor(p, 'crackColor').name('Glow');
  cracks.addColor(p, 'crackHot').name('Core');
  cracks.add(p, 'crackIntensity', 0, 10, 0.05).name('Intensity');
  cracks.add(p, 'crackDepth', 0, 0.6, 0.01).name('Depth into volume');
  cracks.add(p, 'grooveDepth', 0, 3, 0.01).name('Groove (normal)');
  cracks.add(p, 'runes', 0, 1, 0.01).name('Runes / sigils');

  const rim = gui.addFolder('Rim');
  rim.addColor(p, 'rimColor').name('Colour');
  rim.add(p, 'rimPower', 0.5, 10, 0.1).name('Power');
  rim.add(p, 'rimIntensity', 0, 5, 0.05).name('Intensity');

  const parts = gui.addFolder('Particles');
  parts.add(p, 'particleCount', { '16k': 16384, '64k': 65536, '128k': 131072, '256k': 262144 }).name('Count');
  parts.add(p, 'spawnRate', 0, 4, 0.01).name('Spawn rate');
  parts.add(p, 'particleSize', 0.002, 0.05, 0.001).name('Size');
  parts.add(p, 'particleLife', 0.3, 8, 0.05).name('Lifetime s');
  parts.add(p, 'buoyancy', -1, 2, 0.01).name('Buoyancy');
  parts.add(p, 'flowFollow', 0, 4, 0.01).name('Follow curl');
  parts.add(p, 'ejectSpeed', -0.5, 1.5, 0.01).name('Eject speed');
  parts.add(p, 'particleIntensity', 0, 8, 0.05).name('Intensity');
  parts.addColor(p, 'particleColor').name('Colour');
  parts.addColor(p, 'particleHot').name('Colour (young)');
  parts.add(p, 'streak', 0, 4, 0.01).name('Streak');

  const dist = gui.addFolder('Distortion');
  dist.add(p, 'haze', 0, 3, 0.01).name('Heat haze');
  dist.add(p, 'lens', 0, 3, 0.01).name('Space warp (lens)');
  dist.add(p, 'hazeRadius', 1.1, 3, 0.01).name('Radius');

  const scene = gui.addFolder('Scene');
  scene.add(p, 'objectLight', 0, 3, 0.01).name('Object light');
  scene.add(p, 'exposure', 0.1, 4, 0.01).name('Exposure');
  scene.add(p, 'bloom', 0, 0.3, 0.001).name('Bloom');
  scene.add(p, 'moon', 0, 3, 0.01).name('Moonlight');
  scene.add(p, 'fire', 0, 3, 0.01).name('Braziers');
  scene.add(p, 'fog', 0, 0.15, 0.001).name('Fog');
  scene.add(p, 'timeScale', 0, 3, 0.01).name('Time scale');
  scene.add(p, 'spin', -1, 1, 0.01).name('Object spin');
  scene.add(p, 'bob', 0, 2, 0.01).name('Levitation bob');
  scene.add(p, 'autoOrbit').name('Auto orbit');
  scene.add(p, 'orbitSpeed', -1, 1, 0.01).name('Orbit speed');
  scene.add(p, 'envEvery', 1, 8, 1).name('Env map every N frames');

  for (const f of [surf, inner, field, cracks, rim, parts, dist, scene]) f.close();
}
