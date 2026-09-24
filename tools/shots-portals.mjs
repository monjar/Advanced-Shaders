// Shots for the portals study (same format as tools/shots.mjs). `setPose`
// places the first-person camera and resets its crossing history, so scripted
// moves never trigger a teleport; the walk-through shot instead lets the demo
// walk the camera (autoWalk) so it really goes through the portal.

export default [
  {
    name: 'parallax through one portal',
    seconds: 5,
    warmup: 5,
    setup: (demo) => {
      demo.applyPreset('Through one portal');
      demo.params.animate = true;
    },
    update: (demo, t, u) => {
      const x = lerp(0.8, -6.2, ease(u));
      demo.setPose([x, 1.65, lerp(-5.2, -6.4, u)], [-3, 1.45, -9.99]);
    },
  },
  {
    name: 'recursive corridor',
    seconds: 5,
    warmup: 5,
    setup: (demo) => {
      demo.applyPreset('Recursive corridor');
      demo.params.animate = false;
    },
    update: (demo, t, u) => {
      demo.params.objectTime = 3.5 + t * 1.2;
      const z = lerp(-3.2, -6.9, ease(u));
      demo.setPose([lerp(106.9, 105.8, u), 1.65, z], [106, 1.4, -8.99]);
    },
  },
  {
    name: 'robot walking out of a portal',
    seconds: 5,
    warmup: 5,
    setup: (demo) => {
      demo.applyPreset('Object crossing');
      demo.params.animate = false;
    },
    update: (demo, t, u) => {
      demo.params.objectTime = 5.6 + t;
      const a = lerp(0.35, 1.05, ease(u));
      demo.setPose([-103 + Math.sin(a) * 3.6, 1.6, -6.64 + Math.cos(a) * 3.6], [-103, 1.0, -6.4]);
    },
  },
  {
    name: 'walking through the portal',
    seconds: 5.5,
    warmup: 5,
    setup: (demo) => {
      demo.applyPreset('Walk through');
    },
    update: (demo, t, u) => {
      // Hold still during warmup (u = 0), then walk: the demo teleports the camera.
      if (u === 0) {
        demo.applyPreset('Walk through');
        demo.params.autoWalk = 0;
      } else {
        demo.params.autoWalk = 1.4;
      }
    },
  },
  {
    name: 'recursion depth and scissor rectangles',
    seconds: 3,
    warmup: 5,
    setup: (demo) => {
      demo.applyPreset('Recursive corridor');
      demo.params.debugView = 1;
      demo.params.showRects = true;
      demo.params.animate = false;
    },
    update: (demo, t, u) => {
      demo.params.objectTime = 9 + t;
      demo.setPose([lerp(106.6, 105.6, u), 1.65, -4.4], [106, 1.4, -8.99]);
    },
  },
];
