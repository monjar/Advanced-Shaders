## 04 · Atmospheric scattering

A physically based sky and aerial perspective after Hillaire, *A Scalable
and Production Ready Sky and Atmosphere Rendering Technique* (EGSR 2020),
with the transmittance parameterisation of Bruneton & Neyret, *Precomputed
Atmospheric Scattering* (EGSR 2008). The camera goes from a mountain valley
to 20,000 km, over a spherical planet with a 400 km mountain patch, oceans
and continents. Everything reusable lives in `src/shared/atmosphere/` and
the procedural planet (05) uses the same module. Presets: *Noon*, *Morning,
mountains*, *Afternoon*, *Sunset*, *Twilight*, *Earth shadow*,
*Stratosphere*, *Low orbit, limb*, *Low orbit, sunrise*, *From orbit*, and a
Mars-like atmosphere (*Mars, afternoon / sunset / from orbit*).

### Frame graph

```
compute  terrain heights + gradients   1024² mountain patch (once)
compute  transmittance                 256×64, 40 samples        (when the medium changes)
compute  multiScattering               32×32 × 64 directions     (when the medium changes)
compute  irradiance                    64×16 × 64 directions     (when the medium changes)
compute  skyView                       192×108, 24–40 samples    (every frame)
compute  aerial                        32×32×32 froxels          (every frame)
compute  scene                         per pixel: height-field march, ground shading + AP, or sky + sun
compute  reference + compare           brute force at 1/4 res and error statistics (on demand)
render   composite                     exposure, ACES, sRGB, debug views
```

### Techniques and where they live

- **Medium** (`shared/atmosphere/common.wgsl`): Rayleigh (5.802, 13.558,
  33.1)·10⁻⁶ /m with an 8 km scale height; Mie scattering 3.996·10⁻⁶ /m and
  absorption 0.444·10⁻⁶ /m (extinction 4.44·10⁻⁶, the paper's values;
  the absorption slider covers other choices) with 1.2 km and a
  Cornette-Shanks phase, g = 0.8 (per channel, for Mars); ozone (0.650,
  1.881, 0.085)·10⁻⁶ /m in a tent 30 km wide centred at 25 km. Planet
  6360 km, top 6460 km. Units are km so float32 is comfortable, and every
  ray/sphere test from the camera takes the double-precision altitude and
  forms `c = h (2R + h)` instead of `r² − R²`.
- **Transmittance LUT** (`generate.wgsl` `transmittance`): Bruneton's
  (x_μ, x_r) mapping, which puts texels where the horizon changes fastest.
  Only directions above the horizon are stored; the Earth's shadow is a
  separate test, softened over the sun's angular radius so the terminator
  does not alias.
- **Multiple scattering** (`multiScattering`): Hillaire's Ψ_ms. For each
  (sun zenith, altitude) texel one 64-thread workgroup integrates 64
  directions (single scattering with an isotropic phase plus sunlight
  bounced off the ground albedo, and the transfer f_ms), reduces them in
  shared memory and stores L₂ / (1 − f_ms), the sum of all orders.
- **Sky irradiance LUT** (`irradiance`): not in Hillaire's paper, but the
  ground needs skylight. Cosine-weighted hemisphere of 64 directions through
  the LUT-based integrator; used for ground and water.
- **Sky-view LUT** (`skyView`): 192×108 around the camera's zenith, with
  Hillaire's non-linear latitude mapping (sqrt towards the horizon) and
  azimuth measured from the sun (sqrt towards it). Rebuilt every frame.
- **Aerial perspective** (`aerial`, `lookup.wgsl`): 32³ froxels over the
  camera frustum storing in-scattering and chromatic transmittance, with a
  squared slice distribution. The depth grows with altitude (to the horizon
  plus 30 % of the atmosphere's tangent length) and distances are measured
  from the atmosphere entry point, so the same volume works above the
  atmosphere. Beyond its last slice the rest is ray marched.
- **From orbit** (`atmoSkyRadiance`, `atmoAerialPerspective`): above
  80 km (a slider) sky and aerial perspective are ray marched per pixel
  (16–48 samples) from the entry point, as the paper does for space views:
  the atmosphere is then a shell a few pixels thick and the LUTs' angular
  resolution would smear the limb.
- **Sun disk**: radiance normalised so the disk integrates to the sun's
  illuminance, with Hestroffer & Magnan's power-law limb darkening
  (α = 0.40 / 0.50 / 0.65 for R / G / B), attenuated by the camera's
  transmittance.
- **Brute-force reference** (`reference.wgsl`, `compare.wgsl`): the same
  scene at 1/4 resolution with nothing looked up: the view ray uses 256
  samples and every sample marches its own 32-sample optical depth to the
  sun. The Ψ_ms term is kept (it is part of the model being compared; a
  checkbox drops it from both sides). A compute pass compares matching
  pixels and accumulates relative luminance errors with atomics; the HUD
  and `demo.stats` report them.
- **Scene** (`scene_lib.wgsl`, `terrain.wgsl`): a 1024² height field baked
  once over a 400 km gnomonic patch (domain-warped ridged multifractal plus
  derivative-damped fBm), ray marched with clearance-scaled steps and a
  bisection refine, soft height-field shadows, grass/rock/snow by height and
  slope. Continents and oceans on the sphere are value noise; water has a
  GGX glint. `src/shared/planet-camera.ts` is the camera: position in
  double precision, heading parallel-transported over the sphere, speed
  proportional to the height above the ground.

### Measured against the brute-force reference

640×360, reference at 160×90 (luminance error relative to the reference,
plus 1 % of display white so the night sky does not dominate; 8-bit is the
mean absolute difference after tone mapping):

| Preset | sky mean / max | ground mean / max | ground 8-bit |
|---|---|---|---|
| Morning, mountains | 0.09 % / 0.28 % | 0.13 % / 0.6 % | 0.16 |
| Noon | 0.11 % / 0.37 % | 0.05 % / 1.8 % | 0.05 |
| Sunset | 0.05 % / 0.27 % | 0.28 % / 5.6 % | 0.21 |
| Twilight | 0.13 % / 10 % (stars) | 0.07 % / 1.7 % | 0.07 |
| Stratosphere (30 km) | 0.25 % / 1.5 % | 1.1 % / 8.8 % | 1.10 |
| Low orbit (ray marched) | 0.00 % / 0.31 % | 0.51 % / 2.1 % | 0.36 |
| From orbit (ray marched) | 0.00 % / 0.22 % | 0.88 % / 3.3 % | 0.61 |
| Mars, sunset | 0.30 % / 1.4 % | 0.19 % / 2.4 % | 0.12 |

The reference itself is converged: 128, 512 and 1024 view steps agree to
0.06 %. Three changes came out of these measurements:

- **Sample position.** Hillaire places each segment's sample at 30 % of the
  segment. At the sky-view LUT's 24–40 steps that made the sky a steady
  2.5 % too bright (max 3.8 %); the midpoint gives 0.09 %. 200 steps at
  30 % would be needed for 0.4 %.
- **Froxels past the ground.** Clamping each froxel ray at its own ground
  hit (as in the paper) is wrong for the pixels of that froxel whose ground
  is further away: they read too little in-scattering, in bands one froxel
  row high. The sunset ground error was 3.6 % mean (22 % max), the
  stratosphere 6 %. Integrating on through a virtual sea-level medium
  gives 0.28 % and 1.1 %.
- **Interpolating slices linearly in distance** rather than in the squared
  slice coordinate cut the ground error a further 1.5–4× (sunset 0.74 % →
  0.28 %, noon 0.19 % → 0.05 %).

The remaining errors are the froxels' 32×32 screen resolution on grazing
views from high up (stratosphere), and the thin terminator from orbit.
Multiple scattering itself is not validated against a path tracer here.

### Cost

Per frame the atmosphere evaluates ~0.6 M integrator samples for the
sky-view LUT and 65 k for the froxels (each: one exp, two LUT fetches); the
parameter-dependent LUTs add ~3.5 M samples once. On SwiftShader (CPU) the
whole frame at 480×270 takes 0.6–1.5 s, dominated by the terrain march; it
has not been timed on a GPU, but the LUT work is small next to one
full-resolution pass of the scene.

### Debug views

LUT path, brute-force reference, split view with a movable divider,
relative error heat map (10 % = red), and each LUT: transmittance,
multiple scattering (×50), sky view, aerial perspective in-scattering and
transmittance (32 slices as an 8×4 mosaic), sky irradiance; hit distance.

### Known limitations

- The Mars preset is "Mars-like": the dust parameters are tuned by eye for a
  butterscotch day and a blue sunset, not fitted to measurements.
- No volumetric shadows (light shafts) from terrain or clouds in the
  in-scattering; the terrain only shadows the ground.
- Ψ_ms assumes isotropic higher orders and uniform lighting around each
  point (Hillaire's approximation); twilight colours are therefore
  plausible rather than validated.
- Surface albedo in the irradiance and multiple-scattering LUTs is one
  constant, independent of the continents and oceans drawn.
- Verified on SwiftShader only.
