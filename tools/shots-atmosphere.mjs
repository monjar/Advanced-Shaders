// Scripted shots for the atmosphere study (same format as tools/shots.mjs):
// functions are serialised into the page and may only use their arguments
// and the helpers lerp(a, b, u) and ease(u).

export default [
  {
    name: 'sunset time-lapse over the mountains',
    seconds: 6,
    warmup: 2,
    setup: (demo) => {
      demo.applyPreset('Sunset');
    },
    update: (demo, t, u) => {
      const p = demo.params;
      p.sunElevation = lerp(14, -5, u);
      p.sunAzimuth = lerp(262, 278, u);
      // Exposure follows the fading sky (roughly exponential below 6°).
      p.exposure = 12 * Math.exp(Math.min(Math.max(6 - p.sunElevation, 0), 12) * 0.24);
      demo.cam.setGeo(-0.3, -0.1, 2200, lerp(282, 272, ease(u)), lerp(0, 5, u));
    },
  },
  {
    name: 'climb from the ground to low orbit',
    seconds: 7,
    warmup: 2,
    setup: (demo) => {
      demo.applyPreset('Afternoon');
    },
    update: (demo, t, u) => {
      const e = ease(u);
      // Logarithmic climb: 2.5 km to 900 km.
      const alt = Math.exp(lerp(Math.log(2500), Math.log(900000), e));
      demo.cam.setGeo(lerp(-0.6, -4, e), -0.3, alt, lerp(320, 340, e), lerp(0, -24, e * e));
      demo.params.exposure = lerp(10, 8, e);
    },
  },
  {
    name: 'terminator from orbit',
    seconds: 5,
    warmup: 2,
    setup: (demo) => {
      demo.applyPreset('From orbit');
    },
    update: (demo, t, u) => {
      const p = demo.params;
      p.sunAzimuth = 240;
      p.sunElevation = lerp(40, -35, ease(u));
      demo.cam.setGeo(-20, lerp(25, 35, u), 15000000, 0, -88);
    },
  },
  {
    name: 'Mars: blue sunset',
    seconds: 5,
    warmup: 2,
    setup: (demo) => {
      demo.applyPreset('Mars, sunset');
    },
    update: (demo, t, u) => {
      const p = demo.params;
      p.sunElevation = lerp(9, -1, u);
      p.exposure = 10 * Math.exp(Math.min(Math.max(9 - p.sunElevation, 0), 10) * 0.25);
      demo.cam.setGeo(-1.5, -0.3, 1500, lerp(280, 270, ease(u)), 6);
    },
  },
];
