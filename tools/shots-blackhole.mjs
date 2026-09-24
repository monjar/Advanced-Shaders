// Scripted shots of the black hole study for tools/record.mjs (same format as
// the arrays in tools/shots.mjs). `setup` and `update` are serialised into the
// page: they may only use their arguments and the page helpers lerp and ease.
// demo.preset(name) applies one of the GUI presets (src/demos/blackhole/params.ts).

export default [
  {
    name: 'edge-on disk, Doppler-bright approaching side',
    seconds: 6,
    warmup: 2,
    setup: (demo) => {
      demo.preset('Edge-on (Interstellar-like)');
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(-0.35, 0.35, ease(u));
      c.distance = lerp(34, 24, ease(u));
      c.pitch = lerp(0.03, 0.09, u);
    },
  },
  {
    name: 'inclination sweep towards face-on',
    seconds: 5,
    warmup: 2,
    setup: (demo) => {
      demo.preset('Inclined');
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.pitch = lerp(0.02, 1.25, ease(u));
      c.yaw = lerp(0.35, 0.8, u);
      c.distance = lerp(26, 22, ease(u));
    },
  },
  {
    name: 'artistic approximation | Schwarzschild geodesics',
    seconds: 4,
    warmup: 2,
    setup: (demo) => {
      demo.preset('Artistic vs physical (split)');
    },
    update: (demo, t, u) => {
      demo.params.split = lerp(0.15, 0.85, ease(u));
      demo.camera.yaw = lerp(0.2, 0.05, u);
    },
  },
  {
    name: 'a star moving behind the hole: arcs close into an Einstein ring',
    seconds: 4,
    warmup: 2,
    setup: (demo) => {
      demo.preset('Einstein ring alignment');
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(0.12, 0.0, ease(Math.min(1, u * 1.4)));
      c.pitch = lerp(0.05, 0.0, ease(Math.min(1, u * 1.4)));
    },
  },
  {
    name: 'zoom into the photon ring',
    seconds: 4,
    warmup: 2,
    setup: (demo) => {
      demo.preset('Photon ring close-up');
    },
    update: (demo, t, u) => {
      const e = ease(u);
      // Log-zoom from the whole hole down to a 0.8° window on the shadow edge.
      demo.params.fovDeg = Math.exp(lerp(Math.log(30), Math.log(0.8), e));
      demo.params.shiftY = lerp(0, 5.7, Math.min(1, e * 1.15));
    },
  },
];
