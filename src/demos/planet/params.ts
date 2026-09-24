import type GUI from 'lil-gui';

export const DEFAULTS = {
  // Terrain
  seaBias: 0.08,
  mountainHeight: 4400,
  detailHeight: 700,
  snowLine: 0,
  moisture: 0,
  // Sun: sub-solar point, optionally moving (one revolution per dayLength s)
  sunLat: 12,
  sunLon: -20,
  dayLength: 0,
  // Clouds
  clouds: true,
  cloudCoverage: 0.5,
  cloudAltitude: 5500,
  cloudDepth: 30,
  cloudShadow: 0.85,
  cloudSwirl: 120000,
  cloudTimeLapse: 800,
  // City lights and ocean
  cityIntensity: 0.1,
  cityDensity: 0.5,
  waveSlope: 0.16,
  waterRoughness: 0.06,
  // Camera and exposure
  fov: 55,
  exposure: 10,
  stars: 1,
  shadows: true,
  // LOD
  maxErrorPx: 5,
  maxLevel: 19,
  geometryOctaves: 21,
  bakeBudget: 24,
  freezeLod: false,
  debugView: 0,
};

export type PlanetParams = typeof DEFAULTS;

export interface CameraPreset {
  lat: number;
  lon: number;
  /** Metres above sea level; the camera never goes below the ground + eye height. */
  altitude: number;
  heading: number;
  pitch: number;
}

export const PRESETS: Record<string, { params: Partial<PlanetParams>; camera: CameraPreset }> = {
  'Orbit, day side': {
    params: { sunLat: 15, sunLon: 10, exposure: 9 },
    camera: { lat: 12, lon: 30, altitude: 13000000, heading: 0, pitch: -88.5 },
  },
  'Terminator, city lights': {
    params: { sunLat: 5, sunLon: 105, exposure: 100 },
    camera: { lat: 18, lon: -25, altitude: 3000000, heading: 90, pitch: -55 },
  },
  'Low orbit sunset': {
    params: { sunLat: 0, sunLon: 97, exposure: 14 },
    camera: { lat: 10, lon: 0, altitude: 400000, heading: 90, pitch: -15 },
  },
  'Flyover, mountains': {
    params: { sunLat: 5, sunLon: -30, exposure: 10, cloudCoverage: 0.3 },
    camera: { lat: 8.2, lon: 40.3, altitude: 9000, heading: 20, pitch: -10 },
  },
  'Coast': {
    params: { sunLat: -10, sunLon: -150, exposure: 10 },
    camera: { lat: -38.05, lon: -107.5, altitude: 2500, heading: 0, pitch: -12 },
  },
  'Ground level': {
    params: { sunLat: 5, sunLon: -38, exposure: 11 },
    camera: { lat: 8.45, lon: 40.3, altitude: 0, heading: 20, pitch: 2 },
  },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final': 0,
  'Height': 1,
  'Biome albedo': 2,
  'Normals': 3,
  'LOD chunks (level, morph)': 4,
  'Cloud shadow': 5,
  'Clouds only': 6,
  'Habitability (city lights)': 7,
};

export interface GuiCallbacks {
  preset(name: string): void;
  terrainChanged(): void;
}

export function buildGui(gui: GUI, p: PlanetParams, cb: GuiCallbacks) {
  const preset = { preset: 'Orbit, day side' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    cb.preset(name);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  });
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');
  gui.add(p, 'exposure', 0.5, 400, 0.1).name('Exposure');

  const sun = gui.addFolder('Sun');
  sun.add(p, 'sunLat', -30, 30, 0.1).name('Sub-solar latitude °');
  sun.add(p, 'sunLon', -180, 180, 0.1).name('Sub-solar longitude °').listen();
  sun.add(p, 'dayLength', 0, 600, 1).name('Day length s (0 = stop)');

  const terrain = gui.addFolder('Terrain');
  terrain.add(p, 'seaBias', -0.3, 0.3, 0.001).name('Sea level bias').onFinishChange(() => cb.terrainChanged());
  terrain.add(p, 'mountainHeight', 0, 9000, 10).name('Mountains m').onFinishChange(() => cb.terrainChanged());
  terrain.add(p, 'detailHeight', 0, 2000, 10).name('Detail m').onFinishChange(() => cb.terrainChanged());
  terrain.add(p, 'snowLine', -3000, 3000, 10).name('Snow line offset m');
  terrain.add(p, 'moisture', -0.5, 0.5, 0.01).name('Moisture offset');

  const clouds = gui.addFolder('Clouds');
  clouds.add(p, 'clouds').name('Enabled');
  clouds.add(p, 'cloudCoverage', 0, 1, 0.01).name('Coverage');
  clouds.add(p, 'cloudAltitude', 1000, 12000, 10).name('Altitude m');
  clouds.add(p, 'cloudDepth', 1, 80, 0.1).name('Max optical depth');
  clouds.add(p, 'cloudShadow', 0, 1, 0.01).name('Shadows');
  clouds.add(p, 'cloudSwirl', 0, 800000, 1000).name('Swirl m');
  clouds.add(p, 'cloudTimeLapse', 0, 5000, 1).name('Time-lapse ×');

  const surface = gui.addFolder('Cities and ocean');
  surface.add(p, 'cityIntensity', 0, 0.1, 0.0001).name('City lights');
  surface.add(p, 'cityDensity', 0, 1, 0.01).name('City density');
  surface.add(p, 'waveSlope', 0, 0.5, 0.01).name('Wave slope');
  surface.add(p, 'waterRoughness', 0.01, 0.4, 0.001).name('Water roughness');

  const lod = gui.addFolder('LOD');
  lod.add(p, 'maxErrorPx', 1, 30, 0.1).name('Max error px');
  lod.add(p, 'maxLevel', 4, 20, 1).name('Max level');
  lod.add(p, 'geometryOctaves', 10, 23, 1).name('Finest mesh octave').onFinishChange(() => cb.terrainChanged());
  lod.add(p, 'bakeBudget', 1, 256, 1).name('Chunks baked / frame');
  lod.add(p, 'freezeLod').name('Freeze LOD');
  lod.add(p, 'fov', 10, 100, 0.1).name('Field of view °');
  lod.add(p, 'shadows').name('Terrain shadows');
  lod.add(p, 'stars', 0, 4, 0.01).name('Stars');

  for (const f of [terrain, clouds, surface, lod]) f.close();
}
