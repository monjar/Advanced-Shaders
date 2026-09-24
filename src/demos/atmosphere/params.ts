import type GUI from 'lil-gui';

export const DEFAULTS = {
  // Planet and medium (multipliers on the Earth or Mars base values)
  planet: 'Earth' as 'Earth' | 'Mars',
  rayleighScale: 1,
  rayleighHeight: 8,
  mieScale: 1,
  mieAbsorptionScale: 1,
  mieHeight: 1.2,
  mieG: 0.8,
  ozoneScale: 1,
  groundAlbedo: 0.2,
  multiScattering: 1,
  sunIlluminance: 1,
  sunDiskDeg: 0.2679,
  limbDarkening: true,
  // Sun
  timeOfDay: 14,
  latitude: 40,
  sunElevation: 35,
  sunAzimuth: 200,
  // Camera: position as latitude / longitude around the site, log altitude
  altitudeLog: Math.log10(1500),
  fov: 60,
  // Scene
  terrainScale: 1,
  snowLine: 2.7,
  shadows: true,
  stars: 1,
  exposure: 10,
  rayMarchAltitude: 80,
  apDistance: 0,
  // Reference
  reference: false,
  refScale: 4,
  refViewSteps: 256,
  refSunSteps: 32,
  refMultiScattering: true,
  debugView: 0,
  splitX: 0.5,
};

export type AtmosphereDemoParams = typeof DEFAULTS;

// Base-dependent slider values set when switching planets.
const EARTH_BASE: Partial<AtmosphereDemoParams> = { rayleighHeight: 8, mieHeight: 1.2, mieG: 0.8, sunDiskDeg: 0.2679 };
const MARS_BASE: Partial<AtmosphereDemoParams> = { rayleighHeight: 11, mieHeight: 11, mieG: 0.72, sunDiskDeg: 0.175, snowLine: 20 };

/** Camera placement for a preset: latitude / longitude (deg) relative to the site, altitude (m), heading and pitch (deg). */
export interface CameraPreset {
  lat: number;
  lon: number;
  altitude: number;
  heading: number;
  pitch: number;
}

export const PRESETS: Record<string, { params: Partial<AtmosphereDemoParams>; camera: CameraPreset }> = {
  'Noon': {
    params: { sunElevation: 62, sunAzimuth: 170, exposure: 8 },
    camera: { lat: 0.3, lon: 0.6, altitude: 4200, heading: 200, pitch: -6 },
  },
  'Morning, mountains': {
    params: { sunElevation: 8, sunAzimuth: 100, exposure: 14 },
    camera: { lat: 0.2, lon: -0.4, altitude: 3000, heading: 250, pitch: -2 },
  },
  'Afternoon': {
    params: { sunElevation: 25, sunAzimuth: 235, exposure: 10 },
    camera: { lat: -0.6, lon: -0.3, altitude: 2500, heading: 320, pitch: 0 },
  },
  'Sunset': {
    params: { sunElevation: 1.5, sunAzimuth: 275, exposure: 22 },
    camera: { lat: -0.3, lon: -0.1, altitude: 2200, heading: 285, pitch: 1 },
  },
  'Twilight': {
    params: { sunElevation: -4, sunAzimuth: 275, exposure: 260 },
    camera: { lat: -0.3, lon: -0.1, altitude: 2200, heading: 280, pitch: 8 },
  },
  'Earth shadow (anti-sunset)': {
    params: { sunElevation: -1, sunAzimuth: 275, exposure: 60 },
    camera: { lat: -0.3, lon: -0.1, altitude: 2200, heading: 95, pitch: 6 },
  },
  'Stratosphere': {
    params: { sunElevation: 12, sunAzimuth: 250, exposure: 12 },
    camera: { lat: -0.3, lon: 0, altitude: 30000, heading: 250, pitch: -4 },
  },
  'Low orbit, limb': {
    params: { sunElevation: 20, sunAzimuth: 250, exposure: 10 },
    camera: { lat: -3, lon: 0, altitude: 400000, heading: 0, pitch: -14 },
  },
  'Low orbit, sunrise': {
    params: { sunElevation: -12, sunAzimuth: 90, exposure: 25 },
    camera: { lat: 0, lon: -8, altitude: 400000, heading: 90, pitch: -17 },
  },
  'From orbit': {
    params: { sunElevation: 40, sunAzimuth: 240, exposure: 8 },
    camera: { lat: -20, lon: 25, altitude: 15000000, heading: 0, pitch: -88 },
  },
  'Mars, afternoon': {
    params: { planet: 'Mars', sunElevation: 30, sunAzimuth: 210, exposure: 16, ...MARS_BASE, groundAlbedo: 0.3 },
    camera: { lat: -0.4, lon: 0.1, altitude: 1500, heading: 20, pitch: 6 },
  },
  'Mars, sunset': {
    params: { planet: 'Mars', sunElevation: 2, sunAzimuth: 265, exposure: 60, ...MARS_BASE, groundAlbedo: 0.3 },
    camera: { lat: -1.5, lon: -0.3, altitude: 1500, heading: 275, pitch: 6 },
  },
  'Mars from orbit': {
    params: { planet: 'Mars', sunElevation: 40, sunAzimuth: 240, exposure: 14, ...MARS_BASE, groundAlbedo: 0.3 },
    camera: { lat: -20, lon: 25, altitude: 9000000, heading: 0, pitch: -88 },
  },
};

export const DEBUG_VIEWS: Record<string, number> = {
  'Final (LUTs)': 0,
  'Reference (brute force)': 1,
  'Split: LUTs | reference': 2,
  'Relative error (10 % = red)': 3,
  'Transmittance LUT': 4,
  'Multiple-scattering LUT': 5,
  'Sky-view LUT': 6,
  'Aerial perspective: in-scattering': 7,
  'Aerial perspective: transmittance': 8,
  'Sky irradiance LUT': 9,
  'Hit distance': 10,
};

export interface GuiCallbacks {
  preset(name: string): void;
  timeOfDay(): void;
  altitude(): void;
  reference(): void;
}

export function buildGui(gui: GUI, p: AtmosphereDemoParams, cb: GuiCallbacks) {
  const preset = { preset: 'Morning, mountains' };
  gui.add(preset, 'preset', Object.keys(PRESETS)).name('Preset').onChange((name: string) => {
    cb.preset(name);
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  });
  gui.add(p, 'debugView', DEBUG_VIEWS).name('View').onChange(() => cb.reference());

  const sun = gui.addFolder('Sun and camera');
  sun.add(p, 'timeOfDay', 0, 24, 0.01).name('Time of day h').onChange(() => {
    cb.timeOfDay();
    sun.controllers.forEach((c) => c.updateDisplay());
  });
  sun.add(p, 'latitude', -80, 80, 0.1).name('Site latitude °').onChange(() => cb.timeOfDay());
  sun.add(p, 'sunElevation', -20, 90, 0.01).name('Sun elevation °').listen();
  sun.add(p, 'sunAzimuth', 0, 360, 0.1).name('Sun azimuth °').listen();
  sun.add(p, 'altitudeLog', 0.3, 7.5, 0.001).name('Altitude log10 m').onChange(() => cb.altitude()).listen();
  sun.add(p, 'fov', 10, 100, 0.1).name('Field of view °');
  sun.add(p, 'exposure', 0.5, 2000, 0.1).name('Exposure');

  const atmo = gui.addFolder('Atmosphere');
  atmo.add(p, 'planet', ['Earth', 'Mars']).name('Base').onChange((planet: string) => {
    Object.assign(p, planet === 'Mars' ? MARS_BASE : EARTH_BASE);
    atmo.controllers.forEach((c) => c.updateDisplay());
  });
  atmo.add(p, 'rayleighScale', 0, 5, 0.01).name('Rayleigh ×');
  atmo.add(p, 'rayleighHeight', 1, 20, 0.1).name('Rayleigh height km');
  atmo.add(p, 'mieScale', 0, 20, 0.01).name('Mie scattering ×');
  atmo.add(p, 'mieAbsorptionScale', 0, 20, 0.01).name('Mie absorption ×');
  atmo.add(p, 'mieHeight', 0.2, 15, 0.01).name('Mie height km');
  atmo.add(p, 'mieG', 0, 0.99, 0.01).name('Mie g (green)');
  atmo.add(p, 'ozoneScale', 0, 5, 0.01).name('Ozone ×');
  atmo.add(p, 'groundAlbedo', 0, 1, 0.01).name('Ground albedo');
  atmo.add(p, 'multiScattering', 0, 2, 0.01).name('Multiple scattering ×');
  atmo.add(p, 'sunIlluminance', 0.05, 5, 0.01).name('Sun illuminance');
  atmo.add(p, 'sunDiskDeg', 0.05, 3, 0.001).name('Sun radius °');
  atmo.add(p, 'limbDarkening').name('Limb darkening');

  const scene = gui.addFolder('Scene');
  scene.add(p, 'terrainScale', 0, 2, 0.01).name('Mountains ×');
  scene.add(p, 'snowLine', 0, 6, 0.01).name('Snow line km');
  scene.add(p, 'shadows').name('Terrain shadows');
  scene.add(p, 'stars', 0, 4, 0.01).name('Stars');
  scene.add(p, 'rayMarchAltitude', 0, 200, 0.1).name('Ray-march sky above km');
  scene.add(p, 'apDistance', 0, 3000, 1).name('AP depth km (0 auto)');

  const ref = gui.addFolder('Brute-force reference');
  ref.add(p, 'reference').name('Compute + compare').onChange(() => cb.reference());
  ref.add(p, 'refScale', { '1/2': 2, '1/4': 4, '1/8': 8 }).name('Resolution').onChange(() => cb.reference());
  ref.add(p, 'refViewSteps', 16, 2048, 1).name('View steps').onChange(() => cb.reference());
  ref.add(p, 'refSunSteps', 4, 256, 1).name('Sun steps').onChange(() => cb.reference());
  ref.add(p, 'refMultiScattering').name('Include Ψms term').onChange(() => cb.reference());
  ref.add(p, 'splitX', 0, 1, 0.001).name('Split position');

  for (const f of [atmo, scene]) f.close();
}

/**
 * Sun position for a time of day at a latitude around the equinox-ish
 * declination of +10°: elevation and azimuth (from north, clockwise), degrees.
 */
export function solarPosition(hours: number, latDeg: number, declDeg = 10): { elevation: number; azimuth: number } {
  const d2r = Math.PI / 180;
  const H = (hours - 12) * 15 * d2r;
  const lat = latDeg * d2r;
  const dec = declDeg * d2r;
  const sinEl = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(H);
  const el = Math.asin(sinEl);
  const az = Math.atan2(-Math.sin(H) * Math.cos(dec), Math.cos(lat) * Math.sin(dec) - Math.sin(lat) * Math.cos(dec) * Math.cos(H));
  return { elevation: el / d2r, azimuth: ((az / d2r) + 360) % 360 };
}
