import type GUI from 'lil-gui';

export const DEFAULTS = {
  // Spectrum (changing these rebuilds h0)
  windSpeed: 8,
  windDirection: 250,
  fetchKm: 120,
  swell: 0.3,
  spreadBlend: 0.9,
  shortWavesFade: 0.03,
  peakEnhancement: 3.3,
  spectrumScale: 1,
  seed: 1,
  // Simulation
  choppiness: 1.1,
  displacementScale: 1,
  normalStrength: 1,
  shallowDepth: 6,
  timeScale: 1,
  paused: false,
  // Foam
  foamThreshold: 0.45,
  foamSharpness: 3,
  foamDecay: 0.45,
  foamIntensity: 1,
  foamScale: 1,
  shoreFoam: 1,
  contactFoam: 0.9,
  // Optics
  absorptionR: 0.42,
  absorptionG: 0.075,
  absorptionB: 0.05,
  scatterColor: [0.0, 0.07, 0.12] as number[],
  foamColor: [0.92, 0.95, 0.97] as number[],
  refraction: 0.035,
  sss: 1.4,
  roughness: 0.075,
  glitter: 1,
  // Shoreline waves
  shoreAmplitude: 0.45,
  shoreWavelength: 16,
  shorePeriod: 6.5,
  shoreRange: 70,
  // Interaction
  rippleSpeed: 4,
  rippleDamping: 0.5,
  rippleStrength: 1.5,
  rippleHeight: 1,
  // Caustics
  causticIntensity: 1,
  causticDepth: 3.5,
  // Sky and light
  sunElevation: 13,
  sunAzimuth: 196,
  sunIntensity: 9,
  turbidity: 2.2,
  rayleigh: 1,
  mieG: 0.8,
  skyBoost: 2.2,
  exposure: 0.85,
  fog: 0.00016,
  debugView: 0,
};

export type OceanParams = typeof DEFAULTS;

export const SPECTRUM_KEYS: (keyof OceanParams)[] = [
  'windSpeed', 'windDirection', 'fetchKm', 'swell', 'spreadBlend', 'shortWavesFade', 'peakEnhancement', 'spectrumScale', 'seed',
];

export const PRESETS: Record<string, Partial<OceanParams>> = {
  'Default': {},
  'Glassy morning': {
    windSpeed: 3, fetchKm: 30, swell: 0.8, choppiness: 0.8, sunElevation: 7, sunAzimuth: 190,
    turbidity: 3.5, shoreAmplitude: 0.3, foamThreshold: 0.6,
  },
  'Trade wind': { windSpeed: 12, fetchKm: 400, swell: 0.2, choppiness: 1.3, sunElevation: 40, sunAzimuth: 150 },
  'Storm': {
    windSpeed: 22, fetchKm: 800, swell: 0.1, choppiness: 1.35, foamThreshold: 0.85, foamDecay: 0.25,
    sunElevation: 25, turbidity: 9, sunIntensity: 4, skyBoost: 1.4, fog: 0.0006, shoreAmplitude: 1.1,
    scatterColor: [0.02, 0.09, 0.1], exposure: 1.1,
  },
  'Sunset': { sunElevation: 2.5, sunAzimuth: 200, turbidity: 4, windSpeed: 6, exposure: 1.2 },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Shaded': 0,
  'Normals': 1,
  'Foam mask': 2,
  'Water thickness': 3,
  'Clipmap levels': 4,
  'Fresnel': 5,
  'Caustics (seabed)': 6,
  'Seabed height': 7,
};

export interface GuiCallbacks {
  spectrumChanged(): void;
  drop(): void;
  resetObjects(): void;
}

export function buildGui(gui: GUI, p: OceanParams, cb: GuiCallbacks) {
  const presetState = { preset: 'Default' };
  gui.add(presetState, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    Object.assign(p, structuredClone(DEFAULTS), structuredClone(PRESETS[name]));
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    cb.spectrumChanged();
  });
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');

  const waves = gui.addFolder('Waves (FFT spectrum)');
  const spec = (c: ReturnType<GUI['add']>) => c.onFinishChange(() => cb.spectrumChanged());
  spec(waves.add(p, 'windSpeed', 1, 30, 0.1).name('Wind speed m/s'));
  spec(waves.add(p, 'windDirection', 0, 360, 1).name('Wind direction °'));
  spec(waves.add(p, 'fetchKm', 1, 1000, 1).name('Fetch km'));
  spec(waves.add(p, 'swell', 0, 1, 0.01).name('Swell'));
  spec(waves.add(p, 'spreadBlend', 0, 1, 0.01).name('Directional spread'));
  spec(waves.add(p, 'shortWavesFade', 0, 0.3, 0.001).name('Short wave fade m'));
  spec(waves.add(p, 'peakEnhancement', 1, 7, 0.1).name('JONSWAP gamma'));
  spec(waves.add(p, 'spectrumScale', 0, 3, 0.01).name('Amplitude'));
  spec(waves.add(p, 'seed', 1, 100, 1).name('Seed'));
  waves.add(p, 'choppiness', 0, 2, 0.01).name('Choppiness');
  waves.add(p, 'displacementScale', 0, 2, 0.01).name('Displacement scale');
  waves.add(p, 'normalStrength', 0, 2, 0.01).name('Normal strength');
  waves.add(p, 'shallowDepth', 0.5, 20, 0.1).name('Shallow fade depth m');
  waves.add(p, 'timeScale', 0, 3, 0.01).name('Time scale');
  waves.add(p, 'paused').name('Pause');

  const foam = gui.addFolder('Foam');
  foam.add(p, 'foamThreshold', 0, 1.5, 0.01).name('Jacobian threshold');
  foam.add(p, 'foamSharpness', 0.5, 10, 0.1).name('Injection');
  foam.add(p, 'foamDecay', 0, 3, 0.01).name('Decay /s');
  foam.add(p, 'foamIntensity', 0, 1.5, 0.01).name('Intensity');
  foam.add(p, 'foamScale', 0.2, 4, 0.01).name('Texture scale');
  foam.add(p, 'shoreFoam', 0, 2, 0.01).name('Breaking foam');
  foam.add(p, 'contactFoam', 0, 4, 0.01).name('Contact width m');
  foam.addColor(p, 'foamColor').name('Colour');

  const optics = gui.addFolder('Water optics');
  optics.add(p, 'absorptionR', 0, 1.5, 0.001).name('Absorption R /m');
  optics.add(p, 'absorptionG', 0, 1, 0.001).name('Absorption G /m');
  optics.add(p, 'absorptionB', 0, 1, 0.001).name('Absorption B /m');
  optics.addColor(p, 'scatterColor').name('Scatter colour');
  optics.add(p, 'refraction', 0, 0.15, 0.001).name('Refraction');
  optics.add(p, 'sss', 0, 5, 0.01).name('Subsurface');
  optics.add(p, 'roughness', 0.01, 0.5, 0.001).name('Roughness');
  optics.add(p, 'glitter', 0, 4, 0.01).name('Sun glitter');

  const shore = gui.addFolder('Shoreline waves');
  shore.add(p, 'shoreAmplitude', 0, 2, 0.01).name('Amplitude m');
  shore.add(p, 'shoreWavelength', 4, 40, 0.1).name('Wavelength m');
  shore.add(p, 'shorePeriod', 2, 15, 0.1).name('Period s');
  shore.add(p, 'shoreRange', 10, 150, 1).name('Range m');

  const interaction = gui.addFolder('Interaction');
  interaction.add(p, 'rippleSpeed', 0.5, 8, 0.1).name('Ripple speed m/s');
  interaction.add(p, 'rippleDamping', 0, 3, 0.01).name('Ripple damping');
  interaction.add(p, 'rippleStrength', 0, 4, 0.01).name('Injection');
  interaction.add(p, 'rippleHeight', 0, 3, 0.01).name('Height scale');
  interaction.add({ drop: () => cb.drop() }, 'drop').name('Drop objects');
  interaction.add({ reset: () => cb.resetObjects() }, 'reset').name('Reset objects');

  const caustics = gui.addFolder('Caustics');
  caustics.add(p, 'causticIntensity', 0, 3, 0.01).name('Intensity');
  caustics.add(p, 'causticDepth', 0.5, 12, 0.1).name('Focus depth m');

  const sky = gui.addFolder('Sky and light');
  sky.add(p, 'sunElevation', -4, 89, 0.1).name('Sun elevation °');
  sky.add(p, 'sunAzimuth', 0, 360, 0.1).name('Sun azimuth °');
  sky.add(p, 'sunIntensity', 0, 30, 0.1).name('Sun intensity');
  sky.add(p, 'turbidity', 0.5, 12, 0.01).name('Turbidity');
  sky.add(p, 'rayleigh', 0, 4, 0.01).name('Rayleigh');
  sky.add(p, 'mieG', 0, 0.99, 0.001).name('Mie g');
  sky.add(p, 'skyBoost', 0, 6, 0.01).name('Sky brightness');
  sky.add(p, 'exposure', 0.05, 4, 0.01).name('Exposure');
  sky.add(p, 'fog', 0, 0.003, 0.00001).name('Fog density');

  for (const f of [foam, optics, shore, interaction, caustics, sky]) f.close();
}
