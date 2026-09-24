// Scripted shots of the magical-materials study for tools/record.mjs
// (same format as the arrays in tools/shots.mjs). `setup` and `update` are
// serialised into the page and may only use their arguments plus the
// helpers `lerp(a, b, u)` and `ease(u)`. Warm-up frames let the particles
// populate and the field settle before recording starts.

export default [
  {
    name: 'enchanted crystal, orbit through a charge surge',
    seconds: 6,
    warmup: 90,
    setup: (demo) => {
      demo.applyPreset('Enchanted crystal');
      demo.camera.target = [0, 2.3, 0];
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(-0.35, 0.75, ease(u));
      c.distance = lerp(4.4, 3.1, ease(u));
      c.pitch = lerp(0.16, 0.05, u);
    },
  },
  {
    name: 'cursed obsidian, embers from lava cracks',
    seconds: 5,
    warmup: 120,
    setup: (demo) => {
      demo.applyPreset('Cursed obsidian');
      demo.camera.target = [0, 2.4, 0];
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(0.9, 1.5, ease(u));
      c.distance = lerp(3.2, 2.6, ease(u));
      c.pitch = lerp(-0.02, 0.12, u);
    },
  },
  {
    name: 'arcane metal, runes awakening',
    seconds: 5,
    warmup: 90,
    setup: (demo) => {
      demo.applyPreset('Arcane metal');
      demo.camera.target = [0, 2.4, 0];
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(-0.6, 0.1, ease(u));
      c.distance = lerp(2.9, 3.3, u);
      c.pitch = lerp(0.2, 0.08, ease(u));
    },
  },
  {
    name: 'frozen soul ice',
    seconds: 4,
    warmup: 120,
    setup: (demo) => {
      demo.applyPreset('Frozen soul ice');
      demo.camera.target = [0, 2.3, 0];
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(2.4, 2.9, u);
      c.distance = 3.4;
      c.pitch = lerp(0.3, 0.18, ease(u));
    },
  },
  {
    name: 'void stone, space warp',
    seconds: 4,
    warmup: 90,
    setup: (demo) => {
      demo.applyPreset('Void stone');
      demo.camera.target = [0, 2.4, 0];
    },
    update: (demo, t, u) => {
      const c = demo.camera;
      c.yaw = lerp(0.2, -0.25, ease(u));
      c.distance = lerp(3.8, 3.0, ease(u));
      c.pitch = 0.06;
    },
  },
];
