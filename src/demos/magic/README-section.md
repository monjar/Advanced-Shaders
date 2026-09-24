## 04 · Magical materials

Materials meant to look supernatural rather than "emissive + noise". The
rule of the study: **one procedural field, every effect derived from it**.
An animated object-space field (curl flow, advected energy filaments, a
slowly evolving charge potential and a Voronoi crack network) drives the
interior light, the cracks, the rim, the scattering, the heat haze, the
particles and even the light the object casts on the altar, so everything
moves together. Presets: *Enchanted crystal*, *Cursed obsidian*, *Arcane
metal*, *Frozen soul ice*, *Void stone*. Shapes: sphere, crystal cluster,
torus knot. The scene is a ruined moonlit temple (pillars, braziers with
flames, engraved rune circle, fog) so refraction and distortion have
something to bend.

### Frame graph

```
compute  bakeShape     96³ SDF + normal of the current shape            (on shape change)
compute  bakeCracks    128³ crack distances, two scales, cell id        (on crack-scale change)
compute  bakeField     N³ (default 72³): v + C, advected noise n, ∇C     (every frame)
compute  probeField    one workgroup: the object's emitted light         (every frame)
compute  simulate      particles: spawn on open cracks, curl advection   (every frame)
render   shadow        moonlight depth map of the static temple          (once)
render   env cube ×6   temple from the object's centre, 128², + 5 prefiltered mips (every 2nd frame)
render   scene         sky, temple, flames → HDR, alpha = view distance
render   warp          heat haze / space warp of the scene around the object
copy + render mips     scene mip chain (13-tap), sampled by frosted refraction
render   object        surface + interior march + refraction
render   particles     additive streaks, depth-tested against scene and object
render   bloom         6 levels down (Karis on the first), 5 up (tent)
render   composite     bloom, exposure, ACES, vignette, sRGB, dither, debug views
```

### The field (`field.wgsl`, baked by `bake.wgsl`)

- **Flow v**: curl noise (Bridson et al., *Curl-Noise for Procedural Fluid
  Flow*, 2007) from analytic-derivative gradient noise, plus a
  divergence-free swirl about the object's axis. The potential is
  multiplied by Bridson's ramp of the shape's distance, so the velocity is
  tangent to the surface: energy circulates inside, air streams around.
- **Energy E**: two advected noise values whose ridges are multiplied, which
  is large only near the intersection curves of two zero sets, i.e. thin
  filaments. Advection is two-phase flow noise (Neyret, *Advected
  Textures*, 2003): each phase is displaced along v for a bounded time and
  re-seeded, and the phases are blended with variance-preserving weights, so
  filaments morph and reconnect instead of cross-fading. The *smooth* noise
  values are baked and the ridge nonlinearity is applied per sample, which
  keeps filaments sub-voxel sharp at a 72³ bake (the same reason SDFs
  interpolate well). Baking the finished E instead gave blotches.
- **Charge C**: slow drifting noise plus surges that sweep through the
  object along a turning axis; its gradient is baked separately.
- **Cracks**: exact distance to Voronoi cell borders (two-pass, Quilez) at
  two scales with a low-frequency domain warp; static in object space like
  real fractures. `crackOpen(C, E)` decides how open each crack is.

### Techniques and where they live

- **Interior energy** (`object.wgsl` `marchInterior`): the view ray is
  refracted in, the exit distance is found by sphere tracing −SDF, then the
  chord is marched with a step count proportional to its length (4…max,
  jittered with IGN). Each step: Beer–Lambert absorption (preset absorption
  plus energy-dependent extinction), exact per-step integral of emission,
  energy hidden under a clear "skin" depth so it floats inside with
  parallax, glowing fracture *sheets* continuing the crack network into the
  volume (fading with depth), and in-scattered moonlight estimated from the
  depth below the surface. The ray bends along ∇C: a gradient-index medium
  whose index follows the charge, so the energy itself refracts light.
  Early out at 1 % transmittance.
- **Refraction** (`refractedBackground`): the scene is rendered first
  (distance in alpha, warped by the haze). At the exit point the ray is
  refracted out per colour channel (dispersion), extended by the scene depth
  behind the pixel, projected, and looked up in the scene mip chain at the
  frost level. Total internal reflection falls back to the reflection cube.
- **Cracks on the surface**: the same crack function evaluated analytically
  per pixel (pixel-exact, anti-aliased with `fwidth`), width = f(opening);
  hot core, glow, and a small light leak around them; closed cracks remain
  as faint hairlines. The groove tilts the normal along ∇(crack distance),
  which is exact for the bisector-plane distance.
- **Fresnel and rim**: Schlick Fresnel against a prefiltered reflection cube
  rendered from the object's centre every other frame. The rim corona is
  powered by the field sampled just *outside* the surface, so it flickers
  and streams with the flow instead of being a constant rim.
- **Subsurface look**: translucency after Barré-Brisebois & Bouchard (GDC
  2011) using the thickness the march just measured, wrap lighting, and the
  in-scattering term inside the march. Inner energy bleeds through thin
  parts naturally because thin chords absorb less.
- **Arcane metal**: anisotropic GGX with height-correlated visibility
  (Heitz 2014), brushed along the mesh tangent, and Filament's bent normal
  for the environment. The crack network becomes engraved channels, and
  each crack cell carries a procedural sigil (rings, a k-gon, spokes, ticks)
  in the cell's tangent plane that *draws itself* around its circle as the
  charge at the cell's feature point rises. Channels and sigils are windows
  into the same interior energy: a short 10-step march beneath them.
- **Heat haze / space warp** (`warp.wgsl`): rays through a halo sphere
  integrate the flow (weighted by energy, charge and closeness to the
  surface) on 6 jittered samples outside the object and project it to a
  screen offset, with a slight chromatic split. Only background farther
  than the halo entry is warped. *Void stone* adds a lens term ∝ 1/impact²
  that pulls the background around the object.
- **Particles** (`particles.wgsl`): a dead particle proposes a random
  surface point (projected onto the SDF) and is born with probability ∝
  crack opening × energy there, so emission follows cracks as they open.
  Alive particles relax toward the curl velocity (plus preset buoyancy), are
  pushed out if they enter the object, and fade over their lifetime. Drawn
  as additive sprites stretched along their screen motion; sub-pixel sprites
  are widened to 1 px with the same energy to avoid sparkle.
- **The object as a light** (`probeField`): one workgroup averages the
  field's emission over a 6³ lattice inside the shape; the temple, the
  engraved rune circle, the fog glow and the reflection cube are lit by
  that, so the light on the altar pulses with the interior energy.
- **Scene** (`env.wgsl`): concentric flagstones, masonry courses, moonlight
  with a PCF shadow map, a soft spherical occluder for the object's shadow
  (as opaque as its material), flickering braziers with billboard flames,
  height fog with closed-form point-light single scattering, starry sky with
  moon and mountain silhouettes. Bloom is the Call of Duty 13-tap chain
  (Jimenez 2014), then ACES (Narkowicz fit).

### Debug views

Field slice (E red, C green, |v| blue, SDF outline, open cracks), crack
mask (primary, secondary, opening), thickness, interior march steps, particle
density, refraction offsets, distortion offsets, reflection cube.

### Performance

Costs scale with the object's screen coverage (the march), the particle
pool and the bake resolution. Knobs: max march steps (40), bake resolution
(72³), particle pool (64k, up to 256k; the spawn probability is normalised
by the pool size, so the pool is a capacity and *Spawn rate* sets the
density), env cube interval (every 2nd frame). The interior
march is bounded: the exit is found first, so the step count adapts to the
chord length, and it ends at 1 % transmittance. Per frame at defaults the
field bake is 373k voxels × ~11 noise evaluations; the surface pays two
27+27-cell Voronoi passes per pixel.

Measured in headless Chromium on SwiftShader (CPU WebGPU) at 640×360, on
a 4-core machine shared with other jobs (load average 10–16, so timings
are noisy): 3.4–8 s per frame depending on preset and how much of the
screen the object covers. Toggling passes on that setup: updating the
reflection cube every frame costs about as much as the main scene
(≈5.0 → 2.2 s/frame with it frozen), and a 131k particle pool against 16k
(before spawn normalisation, when most of the pool was alive) was
≈5.7 → 2.4 s/frame. The one-off crack bake (128³, two Voronoi scales) takes
a few seconds on SwiftShader. None of this has been timed on a real GPU;
the budget reasoning is: one 72³ bake of ~11 noise evaluations per voxel,
≤40 steps × 5 texture fetches per covered pixel, and a few hundred
thousand small additive sprites.

### Known limitations

- The crack displacement is a normal perturbation (and the interior fracture
  sheets give depth); the silhouette is not displaced.
- Refraction is screen-space: what is off screen or hidden behind the
  object falls back to the reflection cube, and there is one refraction
  event in and one out (internal bounces are approximated by the cube).
- The reflection cube is rendered from the object's centre, so reflections
  of nearby geometry (the pedestal) are parallax-approximate, and the cube
  filter is a cone blur, not a true GGX prefilter.
- Particles do not collide with the temple and are not sorted (additive
  blending does not need it); they are depth-tested but not soft.
- The moon shadow of the object uses a spherical occluder, not the shape.
- Only verified in headless Chromium on SwiftShader (CPU WebGPU).
