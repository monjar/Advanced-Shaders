# Advanced Shaders

A WebGPU/WGSL playground for studying advanced real-time shading techniques.
Each study is a self-contained "demo" registered in a shared scene shell
(sidebar navigation, orbit camera, parameter GUI, HDR pipeline).

| # | Study | Techniques |
|---|-------|-----------|
| 01 | [Deep-water ocean](src/demos/ocean) | compute-shader FFT, vertex displacement, procedural animation, BRDFs, screen-space refraction, caustics, GPU buoyancy |
| 02 | [Volumetric clouds](src/demos/clouds) | ray marching, tileable 3D noise, density shaping, multiple scattering, weather maps, temporal reprojection |

![Open water](docs/ocean-open-water.jpg)
![Shoreline](docs/ocean-shoreline.jpg)
![Cumulus from the ground](docs/clouds-ground.jpg)
![Above the cloud layer](docs/clouds-above.jpg)

## Running

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle in dist/
```

It needs a browser with WebGPU: Chrome/Edge 113+, Safari 26+, or Firefox with
WebGPU enabled. `?scale=0.5` renders at half resolution on slow GPUs.

Controls: drag to orbit, right-drag to pan, wheel to zoom, `WASD` to move,
`Q`/`E` to go down/up, and `Shift`+drag on the water to move a floating object.
The active demo is exposed as `window.demo` in the devtools console.

## Layout

```
src/
  main.ts              scene shell: WebGPU init, routing (#/<id>), render loop, HUD
  core/                camera, math, GPU helpers, Demo interface, registry
  shared/              analytic sky (WGSL + CPU mirror) used by every study
  demos/ocean/         01 · deep-water ocean
    index.ts           resources and frame graph
    params.ts          defaults, presets, GUI
    shaders/*.wgsl
  demos/clouds/        02 · volumetric clouds (same structure)
```

To add a study, implement `DemoEntry` from `src/core/demo.ts` and append it to
`src/core/registry.ts`.

## 01 · Deep-water ocean

### Frame graph

```
compute  spectrum_init + spectrum_conjugate   (only when spectrum parameters change)
compute  spectrum_update      h(k,t) → 8 packed complex fields × 3 cascades
compute  fft (rows) → fft (columns)            256-point IFFT in workgroup memory
compute  assemble             displacement, derivatives, Jacobian foam (ping-pong)
compute  mipgen               mip chains for displacement / derivatives
compute  ripples ×2           wave equation, driven by floating bodies
compute  buoyancy             rigid bodies sample the water, write model matrices
render   caustics             photon-area caustic texture (additive)
render   scene                sky, island/seabed, floating objects → HDR + distance
copy     HDR → scene copy     input for refraction
render   ocean                clipmap surface, full water shading
render   composite            exposure, ACES, sRGB
```

### Techniques and where they live

- **FFT waves** (`spectrum_*.wgsl`, `fft.wgsl`, `assemble.wgsl`): JONSWAP
  spectrum with finite-depth dispersion, Donelan/Hasselmann-style directional
  spreading plus swell, short-wave suppression. Three cascades (250 m, 34 m,
  7.3 m) with non-overlapping wavenumber bands. The time update packs Dx+iDz,
  Dy+i∂Dz/∂x, ∂Dy/∂x+i∂Dy/∂z and ∂Dx/∂x+i∂Dz/∂z, so 8 real fields cost 4
  complex IFFTs. The IFFT runs one 256-invocation workgroup per line with a
  bit-reversed load and 8 radix-2 stages in shared memory.
- **Vertex displacement** (`ocean.wgsl` `vs`, `waves.wgsl`): a camera-centred
  geometry clipmap of 10 levels (129² vertices each, spacing doubling per
  level). Levels snap to twice their spacing. Odd vertices morph onto even ones
  near each level's outer edge, and the displacement mip is matched across the
  seam, so levels meet without cracks. The fragment shader discards each
  level's inner hole.
- **Multi-scale normals**: slopes from all cascades are summed and divided by
  the horizontal-displacement Jacobian terms. Mip-mapped, anisotropically
  filtered derivative textures handle distance, and roughness rises with
  distance to keep the sun highlight energy.
- **Fresnel reflection**: Schlick (F0 = 0.02) blending between the analytic sky
  (Rayleigh + Mie single scattering, `sky.wgsl`) and the refracted water.
- **Sun glitter**: GGX with height-correlated Smith visibility, plus sparse
  sharp micro-facet glints that re-randomise over time.
- **Refraction and depth colour**: the scene is rendered first with a linear
  distance target. The water pass offsets the lookup by the normal, rejects
  samples in front of the surface, and applies Beer–Lambert absorption per
  channel over the refracted path length, plus in-scattering from the water body.
- **Subsurface scattering**: the wave-crest back-lit term and view term from
  the Atlas GDC 2019 talk.
- **Foam**: Jacobian-driven whitecaps that accumulate and decay in the
  simulation, breaking foam from shoreline waves, contact foam where the water
  gets thin against geometry (screen-space depth difference), and ripple
  foam, all masked by an animated noise pattern.
- **Caustics** (`caustics.wgsl`, `lighting.wgsl`): a 256² grid over one
  cascade tile is refracted to a plane at the focus depth. Each triangle is
  rasterised where its light lands, with intensity = original area / projected
  area (Evan Wallace's method), and summed additively over 3×3 instances so the
  result tiles. The seabed and submerged objects trace back to the surface
  along the refracted sun direction and sample it with slight chromatic spread.
  Underwater irradiance accounts for Fresnel transmission and beam spreading.
- **Shoreline waves** (`terrain_bake.wgsl`, `waves.wgsl`): the island bake
  stores the signed distance to the shoreline and the direction to shore.
  Gerstner waves travel down that field, grow and steepen as the water shoals,
  and break into foam. Open-ocean waves attenuate in shallow water, and troughs
  are soft-limited so they cannot dig below the seabed.
- **Object interaction** (`buoyancy.wgsl`, `ripples.wgsl`): floating bodies
  invert the horizontal displacement (fixed-point iteration) to find the water
  height under them, then integrate buoyancy, drag and orbital drift and
  align to the surface normal, all on the GPU. Their model matrices go straight
  to the object vertex shader. A damped wave equation on a 128 m grid is driven
  by each body's vertical motion (displaced water rises in a ring around the
  hull) and horizontal motion (bow wave/trough), and treats land as a wall. Its
  height and slope feed back into the ocean surface.

### Debug views

`View` in the GUI: normals, foam mask, water thickness, clipmap levels,
Fresnel, seabed caustics, seabed height.

### Known limitations

- Reflections are sky-only: the island and objects are not reflected (no SSR
  or planar reflection yet).
- No shadows.
- Caustics come from one cascade at one focus depth, so their sharpness only
  approximates depth-dependent focusing.
- The ripple domain is a fixed 128 m square around the objects, and objects are
  clamped to it when dragged.
- The camera is kept above the water; there is no underwater rendering.
- Development was verified in headless Chromium on SwiftShader (CPU WebGPU),
  not benchmarked on real GPUs.

## 02 · Volumetric clouds

Clouds built entirely from procedural 3D noise and ray marched through a
spherical shell over the planet (1.5–4.2 km by default). Fly with `WASD`/`QE`
(hold `Shift` for 4×) through, under or above the layer.

### Frame graph

```
compute  noise_base         128³ Perlin-Worley + Worley fBm, 16 slices per frame (first 8 frames)
compute  noise_small        32³ Worley detail, 128² curl, 512² weather map (weather regenerates on seed change)
compute  mip3d              mip chains for both 3D textures
compute  shadow             256² top-down cloud transmittance around the camera
compute  trace              ray march (1 of 4 pixels per frame by default, half resolution)
compute  resolve            reprojection + variance clipping into ping-pong history
render   composite          sky / ground with cloud shadows, clouds over, ACES, sRGB
```

### Techniques and where they live

- **3D noise** (`noise_lib.wgsl`, `noise_base.wgsl`, `noise_small.wgsl`):
  periodic Perlin and Worley noise evaluated with wrapped lattice hashes, so
  every texture tiles. The base texture packs Perlin-Worley (Perlin fBm
  remapped into Worley fBm) with three Worley fBm frequencies. The detail
  texture has three higher Worley frequencies. The 128³ base is generated
  in 8 slabs over 8 frames so no single dispatch runs long enough to trip a
  GPU watchdog.
- **Weather map**: coverage (Perlin shaped by Worley), cloud type and
  local density. It scrolls with the wind at its own speed. The global
  coverage slider moves a threshold across it.
- **Density shaping** (`density.wgsl`): stratus, stratocumulus and cumulus
  height profiles blended by type. The base shape is eroded by its own Worley
  fBm, then remapped by coverage (Schneider's `remap(base, 1 − coverage, 1)`),
  so sparse areas keep only the densest cores. Detail erosion uses
  curl-distorted Worley, inverted near the cloud base for wispy bottoms and
  billowy tops. Wind shear leans tops downwind, and detail uses the next mip
  with distance.
- **Light absorption** (`trace.wgsl`): Beer–Lambert transmittance to the sun
  from a 6-sample cone march with quadratic spacing. Distant samples skip
  detail. A sun-march absorption factor below 1 stands in for light that
  multiple scattering carries deeper than single scattering would.
- **Scattering**: dual-lobe Henyey–Greenstein (forward silver lining plus a
  weak back lobe), Beer–powder darkening of edges seen away from the sun, and
  height-graded sky ambient.
- **Multiple scattering**: octave approximation (Wrenninge et al. 2013, as in
  Hillaire 2016). Each octave scales extinction by *a*, energy by *b* and
  phase eccentricity by *c*, with *a ≤ b*.
- **Integration**: Hillaire's energy-conserving step integral, cheap
  empty-space probing (base density only) before any lighting work, early
  exit at 1% transmittance, and more steps on long grazing paths.
- **Temporal reprojection** (`resolve.wgsl`): the march start is jittered per
  pixel with interleaved gradient noise plus a golden-ratio sequence, which
  swaps banding for noise that averages out over time. With the 2×2 pattern
  each frame traces only one pixel per block (Horizon Zero Dawn-style), and the
  rest come from history. History is reprojected through a
  transmittance-weighted cloud depth and clipped to the mean ± γσ of this
  frame's fresh samples, which rejects ghosting when clouds or the camera
  move. History is dropped on resize or mode change.
- **Planet-scale precision**: ray/sphere tests with 6,360 km radii are done
  in a camera-centred frame, with `|o|² − r²` factored from altitudes, and
  the stable quadratic form, so float32 does not cancel out the cloud layer.
  Wind offsets wrap to their noise tiles.
- **Cloud shadows and aerial perspective**: ground transmittance comes from a
  shadow map marched along the sun direction. Clouds fade towards the
  horizon sky by their weighted depth.

The analytic sky shared with the ocean now includes ozone absorption on the
sun path, and its reddening falls off quadratically with view elevation.
That keeps the twilight zenith blue instead of teal.

### Performance and stability

The knobs that matter, all in the GUI: cloud buffer resolution (default ½),
update pattern (every pixel, or 1 of 4), primary steps (64), horizon step
multiplier, light steps (6), max distance (60 km), jitter, and new-sample
weight. At defaults each frame traces ¼ × ¼ = 1/16 of the full-resolution
pixel count.

Measured frame-to-frame change with a static camera and wind off, at 16
primary steps (headless Chromium, SwiftShader; lower is more stable):

| Configuration | mean \|ΔI\| | pixels changing > 8/255 |
|---|---|---|
| no jitter, no temporal | 0.34 / 255 | 0.0 % (stable, but visibly banded) |
| jitter, no temporal | 1.32 / 255 | 3.1 % (no banding, noisy) |
| jitter + temporal (2×2 pattern) | 0.59 / 255 | 0.2 % |

![Banding vs noise vs temporal](docs/clouds-temporal-comparison.jpg)

At 64 steps the no-temporal jitter noise is already small (0.38 vs 0.34 /255);
the temporal pass earns its cost mainly by allowing fewer steps and 1/4
updates.

### Debug views

Clouds only, transmittance, cloud depth, raw trace (no reconstruction), plus
weather-map and shadow-map insets.

### Known limitations

- The sun march ignores the Earth's shadow, so after sunset clouds go dark
  rather than keeping their high-altitude glow.
- The weather map is a single tile (45 km by default), so it repeats over
  long flights.
- Reprojection uses one depth per pixel, so thin clouds in front of distant
  ones can smear slightly when the camera moves fast.
- No lightning, precipitation shafts, or light shafts (god rays) through gaps.
- Only verified on SwiftShader (CPU WebGPU); real-GPU frame times have not
  been measured.
