import type GUI from 'lil-gui';

export const DEFAULTS = {
  // Shape
  coverage: 0.5,
  cloudType: 0.65,
  density: 1.8,
  layerBottom: 1500,
  layerTop: 4200,
  baseScaleKm: 11,
  detailScaleKm: 0.9,
  erosion: 0.55,
  curlAmplitude: 220,
  heightSkew: 700,
  weatherSizeKm: 45,
  seed: 1,
  // Wind
  windSpeed: 15,
  windDirection: 60,
  detailSpeed: 1.6,
  weatherSpeed: 0.25,
  // Lighting
  extinction: 0.04,
  lightAbsorption: 0.45,
  albedo: 0.99,
  powder: 0.6,
  ambient: 1,
  forwardG: 0.8,
  backwardG: 0.25,
  forwardWeight: 0.7,
  octaves: 4,
  msExtinction: 0.4,
  msContribution: 0.65,
  msPhase: 0.55,
  lightSteps: 6,
  lightDistance: 1400,
  // Ray march
  steps: 64,
  horizonSteps: 2,
  maxDistanceKm: 60,
  jitter: true,
  // Temporal
  temporal: true,
  updatePattern: 4,
  historyBlend: 0.12,
  resolutionScale: 0.5,
  // Sky
  sunElevation: 22,
  sunAzimuth: 140,
  sunIntensity: 9,
  turbidity: 2,
  exposure: 0.7,
  fog: 0.000018,
  shadowStrength: 1,
  // Debug
  debugView: 0,
  showWeather: false,
  showShadow: false,
};

export type CloudParams = typeof DEFAULTS;

export const WEATHER_KEYS: (keyof CloudParams)[] = ['seed'];

export const PRESETS: Record<string, Partial<CloudParams>> = {
  'Fair-weather cumulus': {},
  'Scattered': { coverage: 0.35, cloudType: 0.8, windSpeed: 8 },
  'Towering cumulus': {
    coverage: 0.55, cloudType: 1, layerTop: 7500, density: 1.3, erosion: 0.45, sunElevation: 35, exposure: 0.5,
  },
  'Overcast stratus': {
    coverage: 0.95, cloudType: 0.05, layerBottom: 900, layerTop: 2600, density: 0.8, sunElevation: 30, exposure: 0.9,
  },
  'Stratocumulus deck': { coverage: 0.75, cloudType: 0.4, layerBottom: 1200, layerTop: 3200 },
  'Sunset': { sunElevation: 3, sunAzimuth: 110, turbidity: 3.5, coverage: 0.45, exposure: 0.9 },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final': 0,
  'Clouds only': 1,
  'Transmittance': 2,
  'Cloud depth': 3,
  'Raw trace (no reconstruction)': 4,
};

export interface GuiCallbacks {
  weatherChanged(): void;
  resized(): void;
  resetHistory(): void;
}

export function buildGui(gui: GUI, p: CloudParams, cb: GuiCallbacks) {
  const preset = { preset: 'Fair-weather cumulus' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    Object.assign(p, structuredClone(DEFAULTS), structuredClone(PRESETS[name]));
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    cb.weatherChanged();
    cb.resized();
  });
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');
  gui.add(p, 'showWeather').name('Weather map inset');
  gui.add(p, 'showShadow').name('Shadow map inset');

  const shape = gui.addFolder('Shape and weather');
  shape.add(p, 'coverage', 0, 1, 0.01).name('Coverage');
  shape.add(p, 'cloudType', 0, 1, 0.01).name('Type (stratus → cumulus)');
  shape.add(p, 'density', 0.1, 3, 0.01).name('Density');
  shape.add(p, 'layerBottom', 200, 5000, 10).name('Layer bottom m');
  shape.add(p, 'layerTop', 800, 10000, 10).name('Layer top m');
  shape.add(p, 'baseScaleKm', 4, 60, 0.1).name('Base noise tile km');
  shape.add(p, 'detailScaleKm', 0.3, 8, 0.01).name('Detail noise tile km');
  shape.add(p, 'erosion', 0, 1, 0.01).name('Detail erosion');
  shape.add(p, 'curlAmplitude', 0, 800, 1).name('Curl distortion m');
  shape.add(p, 'heightSkew', 0, 3000, 10).name('Wind shear m');
  shape.add(p, 'weatherSizeKm', 10, 150, 1).name('Weather map km');
  shape.add(p, 'seed', 1, 50, 1).name('Weather seed').onFinishChange(() => cb.weatherChanged());

  const wind = gui.addFolder('Wind');
  wind.add(p, 'windSpeed', 0, 80, 0.1).name('Speed m/s');
  wind.add(p, 'windDirection', 0, 360, 1).name('Direction °');
  wind.add(p, 'detailSpeed', 0, 4, 0.01).name('Detail speed ×');
  wind.add(p, 'weatherSpeed', 0, 2, 0.01).name('Weather speed ×');

  const light = gui.addFolder('Lighting');
  light.add(p, 'extinction', 0.005, 0.2, 0.001).name('Extinction /m');
  light.add(p, 'lightAbsorption', 0.05, 2, 0.01).name('Sun-march absorption');
  light.add(p, 'albedo', 0.5, 1, 0.001).name('Scattering albedo');
  light.add(p, 'powder', 0, 1, 0.01).name('Powder');
  light.add(p, 'ambient', 0, 3, 0.01).name('Ambient');
  light.add(p, 'forwardG', 0, 0.95, 0.01).name('Forward g');
  light.add(p, 'backwardG', 0, 0.9, 0.01).name('Backward g');
  light.add(p, 'forwardWeight', 0, 1, 0.01).name('Forward weight');
  light.add(p, 'octaves', 1, 8, 1).name('Scattering octaves');
  light.add(p, 'msExtinction', 0.05, 1, 0.01).name('Octave extinction a');
  light.add(p, 'msContribution', 0.05, 1, 0.01).name('Octave energy b');
  light.add(p, 'msPhase', 0.05, 1, 0.01).name('Octave phase c');
  light.add(p, 'lightSteps', 1, 12, 1).name('Light steps');
  light.add(p, 'lightDistance', 100, 5000, 10).name('Light march m');

  const march = gui.addFolder('Ray march');
  march.add(p, 'steps', 8, 256, 1).name('Primary steps');
  march.add(p, 'horizonSteps', 1, 4, 0.1).name('Horizon step ×');
  march.add(p, 'maxDistanceKm', 5, 150, 1).name('Max distance km');
  march.add(p, 'jitter').name('Jitter start');

  const temporal = gui.addFolder('Temporal reprojection');
  temporal.add(p, 'temporal').name('Enabled').onChange(() => cb.resetHistory());
  temporal.add(p, 'updatePattern', { 'Every pixel': 1, '1 of 4 pixels (2×2)': 4 }).name('Update pattern').onChange(() => cb.resized());
  temporal.add(p, 'historyBlend', 0.02, 1, 0.01).name('New-sample weight');
  temporal.add(p, 'resolutionScale', 0.25, 1, 0.05).name('Cloud resolution').onFinishChange(() => cb.resized());

  const sky = gui.addFolder('Sky and light');
  sky.add(p, 'sunElevation', -3, 89, 0.1).name('Sun elevation °');
  sky.add(p, 'sunAzimuth', 0, 360, 0.1).name('Sun azimuth °');
  sky.add(p, 'sunIntensity', 0, 30, 0.1).name('Sun intensity');
  sky.add(p, 'turbidity', 0.5, 10, 0.01).name('Turbidity');
  sky.add(p, 'exposure', 0.05, 4, 0.01).name('Exposure');
  sky.add(p, 'fog', 0, 0.0002, 0.000001).name('Aerial perspective');
  sky.add(p, 'shadowStrength', 0, 2, 0.01).name('Cloud shadows');

  for (const f of [wind, light, march, sky]) f.close();
}
