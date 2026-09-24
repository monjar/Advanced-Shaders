import type GUI from 'lil-gui';
import type { OrbitCamera } from '../../core/camera';

export const DEFAULTS = {
  // Rendering stage
  renderMode: 1,          // 0 artistic, 1 physical, 2 split screen (artistic left)
  split: 0.5,
  debugView: 0,
  // Observer (the camera distance is the orbit camera's; units of r_s)
  fovDeg: 50,
  shiftX: 0,              // off-axis view window centre, degrees
  shiftY: 0,
  observer: 0,            // 0 static, 1 free fall from rest at infinity, 2 circular orbit about the disk axis
  obsRedshift: true,      // the observer sits at r_obs (light from infinity arrives blueshifted)
  // Geodesics
  integrator: 1,          // 0 orbital plane RK4, 1 orbital plane Dormand–Prince 5(4), 2 Cartesian RK4
  dphi: 0.08,
  tolerance: 1e-6,
  maxSteps: 600,
  cartStep: 0.1,
  // Artistic stage
  artPull: 1.0,
  artStep: 0.04,
  artMaxSteps: 400,
  // Accretion disk
  disk: true,
  diskIn: 3,
  diskOut: 14,
  diskTilt: 0,
  diskTemp: 4600,
  opticalDepth: 1.6,
  turbulence: 0.7,
  diskBrightness: 3.0,
  timeScale: 12,          // r_s/c of coordinate time per second
  cycle: 1.0,             // noise cross-fade period in inner-edge orbital periods
  doppler: true,
  gravRedshift: true,
  intensityMode: 0,       // 0 spectral (T → gT), 1 g⁴ (bolometric), 2 g³ (fixed frequency)
  // Sky
  stars: true,
  starBrightness: 1,
  galaxy: true,
  galaxyBrightness: 0.15,
  footprintMode: 1,       // 0 none (unlensed pixel), 1 analytic ray differentials, 2 finite differences
  filterWidth: 0.5,       // Gaussian σ in pixels
  starSizeMicro: 5,       // intrinsic star size, µrad
  beacon: false,
  beaconFlux: 1e-4,
  beaconSizeMicro: 400,
  // Image
  exposure: 1.0,
  bloom: 0.12,
  accumulate: false,
  stats: true,
};

export type BlackHoleParams = typeof DEFAULTS;

interface CameraPose { yaw: number; pitch: number; distance: number }

export const PRESETS: Record<string, { params: Partial<BlackHoleParams>; camera: CameraPose }> = {
  'Inclined': { params: {}, camera: { yaw: 0.35, pitch: 0.2, distance: 26 } },
  'Edge-on (Interstellar-like)': {
    params: { diskOut: 16, opticalDepth: 2.2, fovDeg: 42 },
    camera: { yaw: 0.0, pitch: 0.045, distance: 28 },
  },
  'Face-on': { params: { diskTilt: 90, fovDeg: 60 }, camera: { yaw: 0, pitch: 0, distance: 22 } },
  'Einstein ring alignment': {
    params: { disk: false, beacon: true, galaxyBrightness: 0.08, fovDeg: 45 },
    camera: { yaw: 0, pitch: 0, distance: 30 },
  },
  'Artistic vs physical (split)': { params: { renderMode: 2 }, camera: { yaw: 0.2, pitch: 0.12, distance: 26 } },
  'Shadow measurement': {
    params: { disk: false, galaxyBrightness: 0.08, fovDeg: 24, accumulate: true },
    camera: { yaw: 0.3, pitch: 0.35, distance: 30 },
  },
  'Photon ring close-up': {
    // The window sits on the top of the shadow edge (Synge: 5.623° at 26 r_s).
    params: { fovDeg: 0.8, shiftY: 5.7, opticalDepth: 0.6, debugView: 0 },
    camera: { yaw: 0.35, pitch: 0.2, distance: 26 },
  },
  'Free fall at 5 r_s': {
    params: { observer: 1, fovDeg: 70, exposure: 0.22 },
    camera: { yaw: 0.6, pitch: 0.25, distance: 5 },
  },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final': 0,
  'Steps per pixel': 1,
  'Disk crossings (image order)': 2,
  'Escaped / captured': 3,
  'Redshift factor g': 4,
  'Sky footprint size': 5,
};

export function applyPreset(p: BlackHoleParams, camera: OrbitCamera, name: string) {
  const preset = PRESETS[name];
  Object.assign(p, structuredClone(DEFAULTS), structuredClone(preset.params));
  camera.target = [0, 0, 0];
  Object.assign(camera, preset.camera);
}

export function buildGui(gui: GUI, p: BlackHoleParams, camera: OrbitCamera) {
  const preset = { preset: 'Inclined' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    applyPreset(p, camera, name);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  });
  gui.add(p, 'renderMode', { 'Artistic approximation': 0, 'Schwarzschild geodesics': 1, 'Split: artistic | physical': 2 }).name('Stage');
  gui.add(p, 'split', 0.05, 0.95, 0.01).name('Split position');
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View');

  const obs = gui.addFolder('Observer');
  obs.add(camera, 'distance', 1.6, 200, 0.01).name('Distance r_obs (r_s)').listen();
  obs.add(camera, 'pitch', -1.55, 1.55, 0.001).name('Elevation (rad)').listen();
  obs.add(p, 'fovDeg', 0.05, 120, 0.01).name('Field of view °');
  obs.add(p, 'shiftX', -60, 60, 0.001).name('View shift x °');
  obs.add(p, 'shiftY', -60, 60, 0.001).name('View shift y °');
  obs.add(p, 'observer', { 'Static': 0, 'Free fall from infinity': 1, 'Circular orbit': 2 }).name('Motion (aberration)');
  obs.add(p, 'obsRedshift').name('Observer at r_obs (blueshift)');

  const geo = gui.addFolder('Geodesics');
  geo.add(p, 'integrator', { 'Orbital plane, RK4 fixed dφ': 0, 'Orbital plane, adaptive DP5(4)': 1, 'Cartesian 3D, RK4 step ∝ r': 2 }).name('Integrator');
  geo.add(p, 'dphi', 0.01, 0.5, 0.005).name('RK4 dφ / first step');
  geo.add(p, 'tolerance', 1e-8, 1e-3, 1e-8).name('DP5(4) tolerance');
  geo.add(p, 'cartStep', 0.01, 0.5, 0.005).name('Cartesian step / r');
  geo.add(p, 'maxSteps', 16, 2000, 1).name('Max steps');

  const art = gui.addFolder('Artistic stage');
  art.add(p, 'artPull', 0, 4, 0.01).name('Pull strength');
  art.add(p, 'artStep', 0.005, 0.3, 0.005).name('Step / r');
  art.add(p, 'artMaxSteps', 16, 2000, 1).name('Max steps');

  const disk = gui.addFolder('Accretion disk');
  disk.add(p, 'disk').name('Enabled');
  disk.add(p, 'diskIn', 1.6, 10, 0.01).name('Inner radius (ISCO = 3)');
  disk.add(p, 'diskOut', 4, 40, 0.1).name('Outer radius');
  disk.add(p, 'diskTilt', -90, 90, 0.5).name('Inclination °');
  disk.add(p, 'diskTemp', 1500, 40000, 10).name('Peak temperature K');
  disk.add(p, 'opticalDepth', 0, 6, 0.01).name('Optical depth');
  disk.add(p, 'turbulence', 0, 1.5, 0.01).name('Turbulence');
  disk.add(p, 'diskBrightness', 0, 10, 0.01).name('Brightness');
  disk.add(p, 'timeScale', 0, 100, 0.1).name('Time scale (r_s/c per s)');
  disk.add(p, 'cycle', 0.2, 4, 0.01).name('Noise cycle (inner orbits)');
  disk.add(p, 'doppler').name('Doppler beaming');
  disk.add(p, 'gravRedshift').name('Gravitational redshift');
  disk.add(p, 'intensityMode', { 'Spectral: T → gT': 0, 'Bolometric: × g⁴': 1, 'Fixed frequency: × g³': 2 }).name('Intensity');

  const sky = gui.addFolder('Sky and filtering');
  sky.add(p, 'stars').name('Stars');
  sky.add(p, 'starBrightness', 0, 10, 0.01).name('Star brightness');
  sky.add(p, 'galaxy').name('Galaxy band');
  sky.add(p, 'galaxyBrightness', 0, 3, 0.01).name('Galaxy brightness');
  sky.add(p, 'footprintMode', { 'None (unlensed pixel)': 0, 'Analytic ray differentials': 1, 'Finite differences (+2 rays)': 2 }).name('Footprint');
  sky.add(p, 'filterWidth', 0.1, 2, 0.01).name('Filter σ (px)');
  sky.add(p, 'starSizeMicro', 0, 200, 0.1).name('Star size µrad');
  sky.add(p, 'beacon').name('Beacon star behind hole');
  sky.add(p, 'beaconFlux', 0, 1e-3, 1e-6).name('Beacon flux');
  sky.add(p, 'beaconSizeMicro', 0, 5000, 1).name('Beacon size µrad');

  const img = gui.addFolder('Image');
  img.add(p, 'exposure', 0.05, 10, 0.01).name('Exposure');
  img.add(p, 'bloom', 0, 0.5, 0.001).name('Bloom');
  img.add(p, 'accumulate').name('Accumulate (freezes time)');
  img.add(p, 'stats').name('Statistics in HUD');

  for (const f of [geo, art, sky, img]) f.close();
}
