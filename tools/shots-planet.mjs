// Scripted shots for the procedural planet (same format as tools/shots.mjs):
// functions are serialised into the page and may only use their arguments
// and the helpers lerp(a, b, u) and ease(u). The bake budget is raised so
// the LOD keeps up with the camera at one fixed step per frame.

export default [
  {
    name: 'orbit, day side',
    seconds: 4,
    warmup: 20,
    setup: (demo) => {
      demo.applyPreset('Orbit, day side');
      demo.params.bakeBudget = 128;
    },
    update: (demo, t, u) => {
      demo.cam.setGeo(12, lerp(20, 45, u), 13000000, 0, -88.5);
    },
  },
  {
    name: 'terminator and city lights',
    seconds: 5,
    warmup: 20,
    setup: (demo) => {
      demo.applyPreset('Terminator, city lights');
      demo.params.bakeBudget = 128;
    },
    update: (demo, t, u) => {
      // The terminator sweeps west over the continent as night falls.
      demo.params.sunLon = lerp(125, 100, ease(u));
      demo.params.exposure = lerp(55, 100, ease(u));
      demo.cam.setGeo(18, -25, 3000000, 90, -55);
    },
  },
  {
    name: 'descent from orbit to the ground',
    seconds: 10,
    warmup: 20,
    setup: (demo) => {
      demo.applyPreset('Ground level');
      demo.params.bakeBudget = 256;
    },
    update: (demo, t, u) => {
      // Height above the ground falls logarithmically from 12,000 km to eye
      // height; the ground track arrives early, then the camera tilts up.
      const e = ease(u);
      const g = ease(Math.min(1, u / 0.75));
      const k = Math.min(Math.max((u - 0.45) / 0.45, 0), 1);
      const tilt = k * k * (3 - 2 * k);
      const lat = lerp(12, 8.45, g);
      const lon = lerp(30, 40.3, g);
      const hag = Math.exp(lerp(Math.log(1.2e7), Math.log(1.7), e));
      const pan = Math.min(Math.max((u - 0.75) / 0.25, 0), 1);
      demo.cam.setGeo(lat, lon, Math.max(0, demo.heightAt(lat, lon)) + hag, lerp(0, 20, g) + 20 * pan * pan * (3 - 2 * pan), lerp(-88.5, 2, tilt));
    },
  },
  {
    name: 'flyover, mountains',
    seconds: 5,
    warmup: 30,
    setup: (demo) => {
      demo.applyPreset('Flyover, mountains');
      demo.params.bakeBudget = 128;
    },
    update: (demo, t, u) => {
      demo.cam.setGeo(lerp(8.0, 8.35, u), lerp(40.2, 40.35, u), 9000, lerp(20, 35, ease(u)), -10);
    },
  },
];
