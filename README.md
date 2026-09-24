# Advanced Shaders

A WebGPU/WGSL playground for studying advanced real-time shading techniques.
Each study is a self-contained "demo" registered in a shared scene shell
(sidebar navigation, orbit camera, parameter GUI, HDR pipeline).

| # | Study | Techniques |
|---|-------|-----------|
| 01 | [Deep-water ocean](src/demos/ocean) | compute-shader FFT, vertex displacement, procedural animation, BRDFs, screen-space refraction, caustics, GPU buoyancy |
| 02 | [Volumetric clouds](src/demos/clouds) | ray marching, tileable 3D noise, density shaping, multiple scattering, weather maps, temporal reprojection |
| 03 | [Watercolour](src/demos/watercolour) | G-buffer stylisation, lighting abstraction, world-space brush direction, edge extraction, colour bleeding, paper, temporal coherence |

![Open water](docs/ocean-open-water.jpg)
![Shoreline](docs/ocean-shoreline.jpg)
![Cumulus from the ground](docs/clouds-ground.jpg)
![Above the cloud layer](docs/clouds-above.jpg)
![Watercolour](docs/watercolour.jpg)
![Pen and wash](docs/watercolour-pen-and-wash.jpg)

## Videos

60 fps, 1280×720 captures of each study (H.264):

- [`docs/videos/ocean.mp4`](docs/videos/ocean.mp4): open water and sun glitter,
  shoreline with caustics and breaking waves, dropped objects splashing, storm.
- [`docs/videos/clouds.mp4`](docs/videos/clouds.mp4): time-lapse cumulus, a
  climb through the cloud layer, sunset into the sun, overcast clearing up.
- [`docs/videos/watercolour.mp4`](docs/videos/watercolour.mp4): an orbit around
  the cottage, a pen-and-wash dolly-in, wet-in-wet under a low sun.

They are rendered offline, one fixed 1/60 s step per frame, so they are smooth
regardless of how fast the machine renders. To re-record (for example after
changing a shader or `tools/shots.mjs`):

```sh
npm i -D playwright            # the recorder drives Chromium through Playwright
npm run dev                    # keep running in another terminal
node tools/record.mjs all      # or: ocean | clouds
```

It needs `ffmpeg` with libx264 on `PATH` (or `FFMPEG=/path/to/ffmpeg`). Useful
flags: `--size 1920x1080`, `--fps 120`, `--crf 18`, `--headed` (some platforms
only expose the GPU to a visible browser), `--swiftshader` for machines
without a GPU, and `--final-crf 26` to re-encode the joined file smaller (the
committed ocean video uses it). Each shot is cached as a segment in `docs/videos/.segments`, so
an interrupted run resumes where it stopped. With a real GPU a full run takes
minutes. The committed videos were rendered on SwiftShader (CPU), at about
7 s per ocean frame and 1 s per clouds frame.

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

## 03 · Watercolour

A real medium rather than a cel shader: the scene is painted as transparent
watercolour washes on cold-press paper, with an optional pen-and-ink layer
(pen and wash). Presets: *Watercolour*, *Pen and wash*, *Wet-in-wet*, *Dry
brush*.

### Frame graph

```
render   shadow        sun depth map (cast shadows become a glaze, not a lighting term)
render   gbuffer       sky wash + scene → pigment absorbance, normal + wetness, depth + id, brush direction + light
compute  edges         ink edges (depth / id / crease) and wash boundaries
compute  bleed ×2      separable wet-in-wet bleeding of absorbance
compute  smear         line-integral smear along the brush direction
compute  composite     paper, wobble, edge darkening, granulation, dry brush, ink, Beer–Lambert
compute  coherence     reprojection error against the previous frame (optional)
render   blit          to the canvas
```

### Techniques and where they live

- **Lighting abstraction** (`gbuffer.wgsl`): wrapped diffuse and a PCF cast
  shadow are reduced to three glazes: paper left light, a mid wash, and a
  darker shadow wash tinted cool, as painters mix blue into shadows. Glaze
  boundaries are soft and jittered by surface noise so they read as brushed
  edges. Distant washes get paler and bluer (aerial perspective).
- **Pigment model**: each wash is stored as absorbance, `−log(pigment) ×
  density`, and the final colour is `paper × exp(−absorbance)` in linear
  light. Bleeding, smearing and edge darkening all operate on absorbance, so
  colours mix subtractively like pigment instead of averaging to grey.
- **World-space brush direction**: every material chooses a stroke direction
  on the surface. Walls and slopes follow the contour (`N × up`), roofs and
  trunks run downhill or vertically, water is horizontal. The direction gets
  a small noise-driven rotation. Streak noise is stretched ~7× along it in
  world space, and a screen-space smear follows its projection, stopping at
  object boundaries.
- **Edge extraction** (`edges.wgsl`): from the G-buffer rather than colour,
  so it is noise-free and moves with the geometry. Only the nearer side of a
  depth or id discontinuity draws, and creases come from normal differences
  within an object. Ink width, pressure and breaks vary with surface noise.
- **Colour bleeding** (`bleed.wgsl`): separable blur of absorbance where
  each tap is weighted by the wetness of both the centre and the sample.
  Pigment only runs between washes that are both wet, while dry washes keep
  hard edges. Wet areas are blobs of surface noise per material (foliage,
  water and sky wettest), so blooms are irregular but stable.
- **Paper** (`composite.wgsl`): a procedural cold-press sheet (round tooth
  plus faint fibres). Pigment interacts with it through granulation (more
  pigment in the valleys), dry brush (thin washes skip the peaks), wobble
  (the image is displaced by the grain) and edge darkening (pigment collects
  at the rim of each wash). Raking light over the relief finishes it.

### Temporal coherence

Stylisation noise is where camera motion breaks NPR. Screen-space noise
stays put while the scene moves under it (the "shower door" effect). Plain
world-space noise follows surfaces, but shrinks into aliasing far away and
blows up close to the camera.

- **Depth-adaptive world noise** (`noise.wgsl`, after Bénard et al.,
  *Dynamic Solid Textures for Real-Time Coherent Stylization*, 2009): two
  octaves of world-space noise one octave apart, chosen from the pixel's
  distance so features stay a fixed number of pixels wide, and blended by
  the fractional octave with a variance-preserving weight. Pigment
  turbulence, glaze jitter, streaks, wet areas, ink pressure and the
  pigment-paper grain all use it. The sky uses the same idea on view
  directions.
- **Dynamic canvas** (after Cunzi et al., *Dynamic Canvas for
  Non-Photorealistic Walkthroughs*, 2003): the paper sheet is neither glued
  to the screen nor to surfaces. Each frame a 12×7 grid of rays is cast on the
  CPU against the terrain, the hits are reprojected into the previous frame,
  and a least-squares 2D similarity (shift + zoom) is composed into the paper
  transform. The grain uses two octaves blended by the fractional zoom level
  (infinite zoom), so its size never drifts.
- **Split paper**: the sheet's relief stays on the dynamic canvas, but how
  pigment sat on that sheet (granulation, dry brush, wobble) belongs to the
  painted surface and uses surface-attached grain. The *Attached to
  surfaces* paper mode moves the relief there too.
- **Measured, not eyeballed** (`coherence.wgsl`): every 4th pixel is
  reprojected into the previous frame through its depth, and the stylised
  images are compared with bilinear lookup. The HUD shows the mean
  difference. The *Temporal coherence* folder switches noise space and
  paper mode to reproduce each failure case.

Reprojection error during a 0.6 rad/s orbit (640×360, 30 fps steps,
SwiftShader), averaged over 40 frames:

| Noise | Paper | Error (/255) | vs naive |
|---|---|---|---|
| screen space | fixed to screen | 8.95 | naive baseline |
| world, fixed scale | dynamic canvas | 7.05 | −21 % |
| world, depth-adaptive | dynamic canvas (default) | 6.45 | −28 % |
| world, depth-adaptive | attached to surfaces | 5.45 | −39 % |

With every paper effect turned off, the same orbit measures 2.1 (world) vs
3.6 (screen). The rest of the default's error is mostly the sheet relief
sliding under a moving scene. That is the price of a sheet that still reads
as paper; attaching it to surfaces removes most of it. A static camera
measures 0.

### Debug views

Pigment before bleeding, light value, brush direction, edges, wet areas,
paper height, coherent noise.

### Known limitations

- The dynamic canvas raycasts only the terrain height field on the CPU;
  buildings and trees don't influence the fitted paper motion.
- Pixel-sized effects (bleed radius, smear, ink width) are in screen space.
  They scale with resolution, but a bloom's footprint changes a little as
  objects approach.
- The smear direction is the projection of a world direction, so strokes on
  surfaces seen edge-on can turn noticeably between frames.
- No pigment simulation (no fluid solve as in Curtis et al. 1997); bleeding
  and backruns are image-space approximations.
- Verified in headless Chromium on SwiftShader only.

