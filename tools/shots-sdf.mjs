// Scripted shots of the SDF world for tools/record.mjs (same format as the
// arrays in tools/shots.mjs). `setup(demo)` runs once when a shot starts and
// `update(demo, t, u)` before every frame (t = seconds into the shot, u = t /
// seconds). Both are serialised into the page, so they may only use their
// arguments and the page globals `lerp(a, b, u)` and `ease(u)`.
//
// The renderer keeps no history, so a single warm-up frame (for the cost
// meter) is enough. Every setup calls setView() first, which also turns the
// character-follow camera off again.

export default [
  {
    name: 'overlook: the valley, river, temple and aqueduct',
    seconds: 5,
    warmup: 1,
    setup: (demo) => {
      demo.setView('Overlook', true);
      Object.assign(demo.params, { debugView: 0, sunElevation: 30, sunAzimuth: 50 });
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.target = [-6, 4, 6];
      c.yaw = lerp(0.12, 0.46, ease(u));
      c.pitch = lerp(0.4, 0.33, u);
      c.distance = lerp(128, 104, ease(u));
    },
  },
  {
    name: 'temple entrance, dolly in past the walker',
    seconds: 5,
    warmup: 1,
    setup: (demo) => {
      demo.setView('Temple entrance', true);
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.target = [0, 4.5, 4];
      c.yaw = lerp(0.36, 0.04, ease(u));
      c.pitch = lerp(0.1, 0.06, u);
      c.distance = lerp(30, 19, ease(u));
    },
  },
  {
    name: 'character close-up: gait, breathing, subsurface',
    seconds: 4,
    warmup: 1,
    setup: (demo) => {
      demo.setView('Character close-up', true);
    },
    update: (demo, t, u) => {
      // The follow camera keeps the target on the walker; orbit from a
      // three-quarter front view towards its side (heading = -path angle).
      const c = demo.camera;
      const heading = -(demo.simTime * demo.params.walkSpeed / 4.8 + 1);
      c.yaw = heading + lerp(1.0, 0.25, ease(u));
      c.pitch = lerp(0.14, 0.06, u);
      c.distance = lerp(3.4, 2.8, ease(u));
    },
  },
  {
    name: 'floating sculpture over the river',
    seconds: 4,
    warmup: 1,
    setup: (demo) => {
      demo.setView('Sculpture', true);
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.target = [-24, 6.5, 40];
      c.yaw = lerp(0.45, 1.35, ease(u));
      c.pitch = lerp(0.32, 0.14, ease(u));
      c.distance = lerp(16, 13, u);
    },
  },
  {
    name: 'rotunda and the aqueduct behind',
    seconds: 3,
    warmup: 1,
    setup: (demo) => {
      demo.setView('Rotunda', true);
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.target = [-34, 6, -4];
      c.yaw = lerp(-0.85, -0.35, ease(u));
      c.pitch = 0.12;
      c.distance = lerp(28, 24, u);
    },
  },
  {
    name: 'under the hood: steps heatmap, then a distance-field slice sweeping up through the temple',
    seconds: 4,
    warmup: 1,
    setup: (demo) => {
      demo.setView('Temple entrance', true);
      Object.assign(demo.params, { debugView: 1, sliceAxis: 0, sliceSpacing: 0.5 });
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      if (t < 2) {
        demo.params.debugView = 1;
        c.target = [0, 4.5, 4];
        c.yaw = lerp(0.3, 0.1, t / 2);
        c.pitch = 0.08;
        c.distance = 26;
      } else {
        const v = (t - 2) / 2;
        demo.params.debugView = 7;
        demo.params.sliceOffset = lerp(2.0, 9.5, ease(v));
        c.target = [0, 3, 2];
        c.yaw = 0.5;
        c.pitch = 0.95;
        c.distance = 34;
      }
    },
  },
];
