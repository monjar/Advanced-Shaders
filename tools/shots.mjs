// Scripted camera/parameter shots for tools/record.mjs.
//
// `setup(demo)` runs once when a shot starts; `update(demo, t, u)` runs before
// every frame with t = seconds into the shot and u = t / seconds. Both are
// serialised into the page, so they may only use their arguments and the
// helpers `lerp(a, b, u)` and `ease(u)` (defined in the page by record.mjs).
// `warmup` frames are rendered but not recorded, so simulations and temporal
// history settle before the shot begins.

export const SHOTS = {
  ocean: [
    {
      name: 'open water, sun glitter',
      seconds: 7,
      warmup: 30,
      setup: (demo) => {
        demo.camera.target = [0, 0, -8];
      },
      update: (demo, t, u) => {
        const c = demo.camera;
        c.yaw = lerp(-0.35, 0.25, ease(u));
        c.distance = lerp(38, 24, ease(u));
        c.pitch = lerp(0.2, 0.13, u);
      },
    },
    {
      name: 'shoreline, caustics, breaking waves',
      seconds: 6,
      warmup: 30,
      setup: () => {},
      update: (demo, t, u) => {
        const c = demo.camera;
        c.target = [lerp(0, 20, u), 0, -172];
        c.distance = 34;
        c.pitch = lerp(0.42, 0.3, ease(u));
        c.yaw = lerp(0.25, 0.7, ease(u));
      },
    },
    {
      name: 'objects dropped: buoyancy and ripples',
      seconds: 6,
      warmup: 30,
      setup: (demo) => {
        demo.params.windSpeed = 5;
        demo.spectrumDirty = true;
        demo.camera.target = [0, 0, -10];
        demo.__dropped = false;
      },
      update: (demo, t, u) => {
        if (t > 0.3 && !demo.__dropped) {
          demo.drop();
          demo.__dropped = true;
        }
        const c = demo.camera;
        c.distance = 21;
        c.pitch = lerp(0.52, 0.42, u);
        c.yaw = lerp(-0.2, 0.3, ease(u));
      },
    },
    {
      name: 'storm',
      seconds: 6,
      warmup: 40,
      setup: (demo) => {
        Object.assign(demo.params, {
          windSpeed: 22, fetchKm: 800, swell: 0.1, choppiness: 1.25, foamThreshold: 0.68, foamDecay: 0.8,
          sunElevation: 25, turbidity: 9, sunIntensity: 4, skyBoost: 1.4, fog: 0.0006, shoreAmplitude: 1.1,
          scatterColor: [0.02, 0.09, 0.1], exposure: 1.1,
        });
        demo.spectrumDirty = true;
        demo.camera.target = [0, 0, -8];
      },
      update: (demo, t, u) => {
        const c = demo.camera;
        c.distance = 30;
        c.pitch = 0.15;
        c.yaw = lerp(0.1, -0.3, ease(u));
      },
    },
  ],

  clouds: [
    {
      name: 'fair-weather cumulus, time-lapse wind',
      seconds: 7,
      warmup: 40,
      setup: (demo) => {
        demo.params.windSpeed = 70;
        demo.camera.target = [0, 250, 0];
      },
      update: (demo, t, u) => {
        const c = demo.camera;
        c.pitch = lerp(-0.32, -0.2, u);
        c.yaw = lerp(0.4, 1.0, ease(u));
      },
    },
    {
      name: 'climbing through the layer',
      seconds: 8,
      warmup: 40,
      setup: (demo) => {
        demo.params.windSpeed = 30;
      },
      update: (demo, t, u) => {
        const c = demo.camera;
        c.target = [0, lerp(900, 6500, ease(u)), lerp(0, -3000, u)];
        c.pitch = lerp(-0.15, 0.45, ease(u));
        c.yaw = lerp(0.8, 1.1, u);
      },
    },
    {
      name: 'sunset, looking into the sun',
      seconds: 6,
      warmup: 40,
      setup: (demo) => {
        Object.assign(demo.params, { sunAzimuth: 110, turbidity: 3.5, coverage: 0.45, exposure: 0.9, windSpeed: 50 });
        demo.camera.target = [0, 250, 0];
      },
      update: (demo, t, u) => {
        demo.params.sunElevation = lerp(9, 2, u);
        const c = demo.camera;
        c.pitch = -0.12;
        c.yaw = lerp(-1.0, -1.35, ease(u));
      },
    },
    {
      name: 'overcast clearing up',
      seconds: 5,
      warmup: 40,
      setup: (demo) => {
        Object.assign(demo.params, { sunElevation: 30, sunAzimuth: 140, windSpeed: 40 });
        demo.camera.target = [0, 250, 0];
      },
      update: (demo, t, u) => {
        demo.params.coverage = lerp(0.95, 0.35, ease(u));
        demo.params.cloudType = lerp(0.2, 0.75, ease(u));
        const c = demo.camera;
        c.pitch = -0.25;
        c.yaw = lerp(0.2, 0.5, u);
      },
    },
  ],
};
