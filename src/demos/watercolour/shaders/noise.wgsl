// Noise lookups. Require bindings noiseTex (3D) and noiseSampler, and uniforms F, P.

fn noiseAt(p: vec3f) -> vec4f {
  return textureSampleLevel(noiseTex, noiseSampler, p, 0.0);
}

// Surface-attached noise whose features stay about `px` pixels wide at any
// distance (after Bénard et al., "Dynamic Solid Textures for Real-Time
// Coherent Stylization", 2009). Two octaves of world-space noise one octave
// apart are blended by the fractional part of log2(feature size); the blend
// keeps the variance constant so there is no visible pulsing. Because the
// lookup is in world space the pattern moves exactly with the surface
// (no "shower door"), and because the octave follows distance it neither
// shrinks to aliasing noise far away nor blows up close to the camera.
//
// Mode 1 (fixed world scale) and mode 2 (screen space) exist to demonstrate
// what goes wrong without this.
fn coherentNoise(world: vec3f, screenPx: vec2f, dist: f32, px: f32) -> vec4f {
  let mode = u32(P.mode.x + 0.5);
  let featurePx = px * F.pxScale;
  if (mode == 2u) {
    return noiseAt(vec3f(screenPx / (featurePx * NOISE_CELLS), 0.37));
  }
  if (mode == 1u) {
    // Fixed world size chosen to look right at 30 m.
    let size = 30.0 * F.pixelAngle * featurePx;
    return noiseAt(world / (size * NOISE_CELLS));
  }
  let size = max(dist * F.pixelAngle * featurePx, 1e-4);
  let level = log2(size);
  let l0 = floor(level);
  let w = level - l0;
  let s0 = exp2(l0);
  let a = noiseAt(world / (s0 * NOISE_CELLS) + hash31(l0));
  let b = noiseAt(world / (2.0 * s0 * NOISE_CELLS) + hash31(l0 + 1.0));
  let n = (a - 0.5) * (1.0 - w) + (b - 0.5) * w;
  return n / sqrt((1.0 - w) * (1.0 - w) + w * w) + 0.5;
}

// Same idea for directions (sky): noise on the view sphere is attached to the
// scene at infinity, so it is stable under rotation and translation.
fn skyNoise(dir: vec3f, screenPx: vec2f, px: f32) -> vec4f {
  if (u32(P.mode.x + 0.5) == 2u) {
    return noiseAt(vec3f(screenPx / (px * F.pxScale * NOISE_CELLS), 0.61));
  }
  let size = F.pixelAngle * px * F.pxScale;
  return noiseAt(dir / (size * NOISE_CELLS) + vec3f(0.3, 0.1, 0.7));
}
