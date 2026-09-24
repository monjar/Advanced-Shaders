# Advanced Shaders

A WebGPU/WGSL playground for studying advanced real-time shading techniques.
Each study is a self-contained "demo" registered in a shared scene shell
(sidebar navigation, orbit camera, parameter GUI, HDR pipeline).

| # | Study | Techniques |
|---|-------|-----------|
| 01 | [Deep-water ocean](src/demos/ocean) | compute-shader FFT, vertex displacement, procedural animation, BRDFs, screen-space refraction, caustics, GPU buoyancy |
| 02 | [Volumetric clouds](src/demos/clouds) | ray marching, tileable 3D noise, density shaping, multiple scattering, weather maps, temporal reprojection |
| 03 | [Watercolour](src/demos/watercolour) | G-buffer stylisation, lighting abstraction, world-space brush direction, edge extraction, colour bleeding, paper, temporal coherence |
| 04 | [Portals](src/demos/portals) | recursive stencil portals, oblique near-plane clipping, objects crossing the plane, camera teleport, boundary distortion |
| 05 | [SDF world](src/demos/sdf) | one fullscreen fragment shader: smooth CSG, repetition, deformation, domain warping, soft shadows, AO, reflections |

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

## 04 · Portals

Portal-game style portals between three procedural locations: a warm stone
courtyard at golden hour, a cool sci-fi hangar lit by ceiling panels, and a
misty forest clearing. They sit 100 m apart in one world and are enclosed, so
nothing connects them except four portal pairs (A courtyard ↔ hangar, B hangar
↔ forest, C two facing portals in a hangar bay, D courtyard ↔ forest). Each
opening shows its destination from a virtual camera, recursively (portals
seen through portals, up to 8 levels). Objects pass through, and so does the
camera. Walk with `WASD` (`Shift` to run) and drag to look. Walk into an
opening to go through it. Presets: *Through one portal*, *Recursive
corridor*, *Object crossing*, *Walk through*, *Portals in portals*.

![Recursive corridor](docs/portals-corridor.jpg)
![Courtyard with portals to the forest and the hangar](docs/portals-courtyard.jpg)

### Frame graph

```
CPU      camera: collisions, teleport through the pair transform when the eye crosses an opening
CPU      objects: unfolded animation, clip planes, duplicates for anything straddling a portal
CPU      view tree: virtual camera, oblique projection and scissor rectangle per visible portal
render   shadow ×2      sun depth maps for the courtyard and the forest (clip planes applied)
render   portal views   one 4× MSAA pass, every view of the tree, depth first:
           sky + depth reset   (stencil == L)
           location geometry   (stencil == L)
           per visible portal: mask (L → L+1) · child view or fill · restore (L+1 → L) · halo
render   post           rim refraction (resample), + glow, debug views, ACES, sRGB
```

### Techniques and where they live

- **Virtual cameras** (`portal-math.ts`): each portal has a world matrix
  (x right, y up, z = normal into its room). The pair transform is
  `T = destination · rotateY(180°) · source⁻¹`, so a point in front of the
  source lands behind the destination. A view through the source uses
  `view · T⁻¹`, and `T⁻¹` is simply the partner portal's pair transform.
  The projection's x/y rows are the camera's own, so the opening shows the
  destination in screen space, like a window, with full parallax.
- **Oblique near-plane clipping** (`obliqueProjection`, after Lengyel,
  *Modifying the Projection Matrix to Perform Oblique Near-Plane Clipping*,
  2005), re-derived for WebGPU's `0 ≤ z ≤ w` and the repo's reversed-Z
  infinite projection (rows `r2 = (0,0,0,n)`, `r3 = (0,0,−1,0)`; near is
  `z ≤ w`, far is `z ≥ 0`). Replace `r2` by `r3 − a·C`, with `C` the
  destination plane in view space (`C·v ≥ 0` on the visible side). The new
  near plane `(r3 − r2')·v = a·C·v ≥ 0` is exactly the portal plane, at depth
  1. Depth becomes `1 − a·(C·v)/(−v_z)`. Along a ray, `(C·v)/t` rises towards
  `C·d` (the camera is behind the plane, `C_w < 0`), so nothing visible is
  clipped by the tilted far plane if `a ≤ 1/max(C·d)`. That maximum is at a
  frustum corner: `a = 1/(|Cx|·tanX + |Cy|·tanY − Cz)`. The far plane then
  touches the frustum only at infinity, depth stays in [0, 1], and the
  clear-to-0 / `greater` convention is unchanged. When the virtual camera is
  closer to the plane than the near distance, the plain projection is used:
  its near plane already removes everything between eye and opening (that
  space is the carved hole), and the oblique matrix would lose its depth
  resolution as `C_w → 0`.
- **Stencil recursion** (`drawView` in `index.ts`, `portal.wgsl`): all views
  share one render pass, one depth-stencil buffer and a dynamic-offset
  uniform slot each. A view at level L draws inside `stencil == L`: first a
  fullscreen sky that writes depth 0 (the depth reset), then its location's
  geometry. Then for every portal it sees, nearest first:
  *mask* the opening (depth-tested against level L, so occluders in front
  keep their pixels; stencil L → L+1), render the child view inside
  `stencil == L+1`, then *restore*. Restore writes the portal surface's depth
  in level L's projection (`always`, stencil L+1 → L), so later portals and
  geometry of level L are occluded correctly by this opening. Because every
  restore leaves true surface depth behind, overlapping portals come out
  right in either order. Nearest-first only means the far one's hidden part
  fails its mask test and is never shaded. The destination portal a view looks
  out of is excluded from that view.
- **Culling and scissors** (`portalRect`, `buildViews`): a child view is
  created only for portals in the view's location that face its camera, lie
  at least partly beyond its clip plane, and overlap its scissor rectangle.
  The rectangle is the opening's 48-gon outline, clipped against a plane
  just in front of the eye (not the near plane, see below), projected, then
  intersected with the parent's rectangle. It shrinks down the recursion, and
  every draw of a view is scissored to it.
- **End of recursion**: at the depth limit (1–8), when the view budget (48
  views by default) is used up, or when a portal is smaller than 6 px on
  screen, the opening is filled with its rim colour. The last levels fade
  towards that colour (`((depth − 1)/maxDepth)²`), so the cut is hard to see.
  Trade-off: a fade costs nothing and never shows stale content, but the
  far end of a corridor is a flat colour instead of more corridor. Portal 2
  reuses the previous frame's image there instead, which looks
  deeper but lags and smears when the camera moves.
- **Crossing without flicker** (`portalCupMesh`, `scene.wgsl`): the opening
  is drawn as a thin cup (front ellipse, side wall and back cap up to 30 cm
  deep). Static geometry discards fragments inside a hole carved behind every
  opening (slightly shallower than the host wall, so from behind the wall is
  solid). When the near plane cuts the front cap, the cup's sides and back
  still cover every ray through the opening. The restore and rim passes map
  those fragments back onto the opening plane.
- **Camera teleport** (`updateCamera`): when the eye's path from the last
  frame crosses an opening from the front, inside the ellipse, the camera
  target goes through the pair transform and its yaw turns by the pair's
  yaw difference. The camera is an `OrbitCamera` at 1 cm distance, i.e.
  first person. Collisions are circle-vs-box in xz. The wall holding a portal
  is ignored while the eye is inside its doorway.
- **Objects crossing portals** (`objects.ts`): a bouncing ball (pair A, real
  g parabolas timed so the middle hop peaks in the opening), a walking robot
  (pair B, seven animated parts) and a spinning cube that loops through the
  facing pair C forever. Each is animated in unfolded coordinates of one
  location. Once its centre is past a plane, the canonical transform goes
  through the pair. While its bounds straddle an opening it is drawn twice:
  the original with a per-draw clip plane keeping the source side, and a
  duplicate through the pair transform keeping the destination side. The
  clip is a fragment `discard`, because `clip-distances` is an optional
  feature. With the oblique near plane and the masks, the clip mostly
  matters where hosts are thin and for shadows. Portal B's forest stone is a
  22 cm slab, and with *Clip planes* off the robot visibly pokes out of its back.
- **Energy rim** (`restoreFs`, `haloFs`, `post.wgsl`): refraction happens only
  in a band at the edge (8.5 % of the radius). Restore writes a screen-space
  offset (rgba8, 128 = 0, ¼ px steps) that pulls the sample radially inwards
  and swirls it with seamless angular noise. The offset is 0 at the band's
  inner edge, so the interior is untouched. The post pass resamples the HDR
  image by it. The rim line, the halo on the wall and a faint inner glow go
  into a separate additive target that is added after the resample, so the
  glow itself stays sharp.
- **Physically consistent touches**: lighting is a function of the world
  position only (courtyard sun with PCF shadows, hangar point lights,
  forest sun and mist), so a surface looks the same directly and through
  any chain of portals. Fog is piecewise: each view fogs only beyond its
  entry plane, and its restore pass fogs the opening by the air in front of
  it, so light is attenuated by each location's air in turn. Each opening
  also acts as a small area light carrying its destination's average
  radiance plus its rim colour (*Light spill*). The moving objects add
  analytic sphere occlusion to the floor.
- **Locations** (`scene.ts`): boxes, cylinders, cones, jittered spheres
  and a height grid, with procedural materials in `scene.wgsl`: running-bond
  brick and flagstones, metal panels with seams and rivets, hazard stripes in
  front of every hangar portal, grass, bark, mossy stone, rippled water.
  Joint lines are filtered by the pixel footprint so distant walls don't
  shimmer. Each location is its own index range, so a view draws only
  the location it looks into (2.4 k / 0.6 k / 32.5 k triangles).

![Objects crossing](docs/portals-crossing.jpg)
![Oblique clipping](docs/portals-oblique.jpg)
![Debug views](docs/portals-debug.jpg)

### Verification

Measured in headless Chromium on SwiftShader at 640×360:

| Check | Result |
|---|---|
| Opening vs. a full-screen render from the virtual camera (*Reference camera*, fog and rim off), two camera positions | 0/255 difference over 12,509 and 16,769 pixels (bit-identical) |
| Hangar point projected through the portal vs. the same point moved behind the courtyard wall by `T⁻¹` | same NDC to 1e-6 |
| Oblique matrix, 41×41 frustum directions × 5 distances (0.1 m – 10 km) | 0 points beyond the plane outside [0, 1] (range 0.0005–0.9997), 0 points in front of it kept |
| Walking forward through A at 1.3 m/s, 30 fps: mean frame-to-frame change | teleport frame 2.73/255, neighbouring frames 2.55–2.88 |
| Walking backwards through A (lands 1.3 cm in front of the destination, looking into it) | teleport frame 2.54/255, neighbours 2.11–2.66 |
| Eye 2 cm from the opening, looking straight in and along the wall | no gap or flicker |

Building the parallax check exposed two bugs. The rim target's clear value
0.5 quantised to 128/255 and shifted the whole image by ⅛ px, and a portal
partly outside its parent's rectangle was judged "too small" and filled.

### Cost

Per view: one sky draw, one location draw and the objects in it. Per
visible portal: mask, restore and halo (cup 384 triangles, ring 192). The
corridor preset (8 levels) renders 9 views in 121 draws. One portal renders
2 views in 18 draws, and *Portals in portals* 3 views in 22 draws. Each view's
fragment work is bounded by its scissor rectangle and mask, so deep levels
are cheap. The attachments cost 24 bytes per sample (4× MSAA: HDR
rgba16float, rim offsets and view info rgba8, glow rgb10a2, depth24 +
stencil8), about 240 MB at 1080p including resolves. That is the main memory
cost, and dropping MSAA would cut it to about 50 MB. SwiftShader takes
1.0 s/frame without portals and 1.5–1.8 s/frame for the courtyard and
hangar presets (2.2–3.0 s in the forest). Most of that is the fixed cost of
the 4× MSAA scene pass and two 2048² shadow maps, not the recursion: the
corridor at depth 1 and at depth 8 take the same time. Real GPUs
have not been measured.

### Debug views

`View`: *Recursion depth* (level 0 grey, then one hue per level), *Stencil
masks* (a flat colour per view, the recursion end hatched), *Crossing
duplicates* (original half cyan, duplicate magenta). There are also toggles
for *Scissor rectangles*, *Oblique near plane* (off shows the leak: the far
side of the destination wall), *Clip planes*, and a *Reference camera* that
renders full screen from one portal's virtual camera.

### Known limitations

- The view tree is built depth first, so a long chain can use up the view
  budget before its siblings. They then get the rim-colour fill.
- Light does not travel through portals except as the spill term, which
  uses a fixed average radiance per destination (not the actual image) and
  casts no shadows. Sun shadows stop at the opening, and the hangar has none.
- Portals are vertical. The pair math handles any orientation, but the
  camera (yaw/pitch) and the collisions assume vertical openings.
- Objects are scripted, not simulated, and can straddle one portal at a
  time. An object larger than the opening would cut into the wall.
- The rim refraction is a screen-space resample, so near an occluder in
  front of the rim it can pull in a few pixels of that occluder.
- The oblique far plane is tilted, and depth resolution falls as the eye
  approaches the opening. Below the near distance the plain projection
  takes over.
- Verified in headless Chromium on SwiftShader only.

## 05 · SDF world

A whole world from signed distance functions: a warped-fBm valley with
cliffs and a river, a partly collapsed temple, a domed rotunda, a plaza with
stairs into the water, a two-tier aqueduct that runs to the horizon, an arched
footbridge, a walking character and a floating gyroid sculpture. It is all
sphere traced in one fullscreen fragment shader. Viewpoints: *Temple entrance*,
*Overlook*, *Character close-up* (follows the walker), *Sculpture*, *Rotunda*,
*Aqueduct and bridge*; the camera flies between them.

**The constraint (the extra suffering).** One render pipeline and one draw
of a fullscreen triangle write the final image. There are no meshes, textures,
compute passes, extra render passes or temporal accumulation buffers. Every
pixel rebuilds the world from maths every frame, and all noise is evaluated
analytically. The CPU fills one 256-byte uniform block (camera, sun, time, GUI
values). Everything else, including the character's skeleton, is derived in the
shader. The only other binding is an 8-counter atomic buffer for the cost meter,
written by every 16th pixel on measured frames. Resolution scale shrinks the
canvas backing store and lets the browser upscale, because an upsampling pass
would be a second pass.

### Frame graph

```
render   world      fullscreen triangle, one fragment shader (src/demos/sdf/shaders/world.wgsl + src/shared/sky.wgsl):
                      per pixel: solve character skeleton → sphere trace → shade (shadow, AO, SSS) →
                      0–2 traced reflection bounces (or water: Fresnel + refracted march) → height fog →
                      ACES, sRGB, dither; optional 2×2 in-shader supersampling; debug views
```

### Techniques and where they live

All in `world.wgsl`, which is split into numbered sections.

- **Primitives and operators** (§3–4): boxes, rounded boxes (rounding
  operator), cylinders, capsules, tori, ellipsoids, capped and rounded cones.
  The polynomial smooth min returns its blend weight, and `opSmoothU` uses it
  to blend materials too: skin into cloth on the character, chrome into gold
  on the sculpture's metaballs, echinus into shaft on the columns. Smooth
  subtraction/intersection (`smax`) builds the gyroid lattice and its opening.
  Onion/shelling makes the cella walls, the rotunda drum, the dome and the
  lattice sphere. Elongation (an elongated circle is a stadium) gives every
  round-headed arch: doors, windows, both aqueduct tiers.
- **Domain repetition**:
  - *Infinite*: the aqueduct's arches along x, so it recedes into the haze
    in both directions.
  - *Limited*: the colonnades (11 per side, 4 per front), triglyph grooves,
    cella windows and bridge balusters. The ghat stairs evaluate the two
    neighbouring cells as well, because a taller step next door can be nearer
    than the step in the current cell.
  - *Polar*: 20 flutes per column (carved by subtraction), the rotunda's 14
    columns, 8 windows and 16 dome ribs.
  - *Mirroring*: the temple (|x|, |z|), the twisted columns and the railings.
    The collapsed corner breaks the temple's symmetry by subtracting a lumpy
    ellipsoid from the columns, entablature, roof and cella, leaving broken
    shafts. Fallen drums lie beside it.
- **Deformations**:
  - *Twist*: Solomonic columns, a rounded square twisted 1.25 rad/m, with the
    distance scaled by 1/sqrt(1 + (k·r)²).
  - *Twisted ring*: a square section rotated by 1.5× its sweep angle, so it
    closes seamlessly.
  - *Bend*: the footbridge is a straight deck, railings and balusters bent into
    an arch.
  - *Domain warping*: the terrain (analytic Jacobian, see below) and the
    sculpture. For the sculpture, w = A sin(Bq + φt) is bounded by dividing the
    field by 1 + A·B·√3.
- **Terrain** (§5): the height field is built in four steps.
  - A valley profile across a low-frequency, domain-warped axis: floor, two
    cliff bands with a ledge, mountains.
  - Two noise octaves that wander the cliff line, carving buttresses and
    gullies.
  - Up to 9 octaves of value-noise fBm with analytic derivatives (rotated and
    offset per octave).
  - A flattened plateau under the buildings, and a meandering river channel
    carved through everything. The water is a plane at y = 0.

  **Lipschitz-safe step**: p.y − h is a vertical offset, not a distance. On a
  slope with gradient g the true distance can be as small as
  (p.y − h)/sqrt(1 + |g|²). The shader therefore scales by 1/sqrt(1 + L²).
  Near the ground, L is the local slope: the exact gradient is carried through
  the fBm, the warp Jacobian, the cliff and gully profiles, and the river,
  plus 0.3 of margin for curvature. More than a few metres up, it blends
  (smoothstep over 1–16 m) to the global bound L ≈ 5 (k = 0.2). There a cliff
  may be nearer sideways than the ground below. The global bound everywhere
  would crawl over flat ground at grazing angles, where most terrain pixels
  are. A small overstep that still happens is pulled back by the marcher
  (t += d when d < 0).

  **fBm LOD**: octaves with a wavelength under ~4 pixel footprints are
  dropped, and the last one fades out fractionally so there is no popping. The
  loop also exits early once the point is higher above the partial sum than
  the remaining octaves could reach (±2·amplitude). The early exit returns that
  slack as part of the bound. On the plateau the noise is skipped entirely.
- **Character** (§7): about 20 capsules, ellipsoids and round cones,
  smooth-blended.
  - *Gait*: forward kinematics solved once per pixel into private variables
    (it depends only on time). Thighs swing, knees flex in the swing phase,
    arms counter-swing, and the lower foot is planted, which produces the
    vertical bob.
  - The chest breathes, and the figure walks a circle around the plaza.
  - *Subsurface-ish skin*: wrapped diffuse tinted red, plus back-light
    transmission through thin parts, measured by one field sample 10 cm
    inside.
- **Sculpture** (§8): a chrome core with five gold metaballs orbiting through
  it (smooth union with material blend), a gyroid-perforated onion shell
  opened by a smooth subtraction, and a twisted chrome ring. The whole thing
  is domain warped, spins and bobs over the river.
- **Ray marching** (§10): over-relaxed sphere tracing (Keinert et al.,
  *Enhanced Sphere Tracing*, 2014).
  - Steps are ω·d. When consecutive unbounding spheres don't overlap, the
    marcher steps back and continues with ω = 1.
  - The hit test is a cone, d < ε·t with ε in pixels: a distance-dependent
    epsilon.
  - Rays are clipped to the slab below the scene top and to the water plane.
  - A ray that runs out of steps inside the scene uses its best (smallest
    d/t) sample instead of leaving a hole.
- **Bounding volumes** (§9): each group (temple, plaza, rotunda, aqueduct,
  bridge, character, sculpture) is evaluated only when its box, cylinder or
  sphere is nearer than the best distance so far. That is exact, because the
  bound is a lower bound. Sub-bounds inside groups skip the colonnade, the
  column flutes and the triglyph carving.
- **Lighting** (§11–12):
  - *Soft shadows*: Quilez's improved penumbra estimate, which triangulates
    the closest approach from consecutive spheres. It falls back to k·h/t when
    the spheres don't overlap (h ≥ 2·h_prev after a clamped step or a bound
    jump). Without that fallback the triangulation returns 0 and speckles the
    columns and cliffs black.
  - *AO*: 5 samples along the normal (3 cm to 0.9 m, quadratic spacing), 2 in
    reflections.
  - *Specular*: GGX with height-correlated Smith visibility and Schlick
    Fresnel, and a split-sum environment term (Karis' analytic fit).
  - *Materials*: procedural albedo, roughness and metalness for marble,
    polished tiles, sandstone courses, terracotta, gold, chrome, skin, cloth,
    straw, and terrain (grass/rock/sand/wet/snow by slope and height).
  - *Reflections*: smooth surfaces (polished floors, gold, chrome, water)
    continue the path along the mirror direction. Up to 2 bounces, with ⅓ of
    the primary steps and cheaper shadows and AO. When the bounces run out,
    the path closes with the sky.
- **Water** (§13): an analytic plane, which is the one SDF cheaper to
  intersect than to march.
  - Normals from four directional waves plus drifting noise ripples, faded
    with distance.
  - Fresnel reflection is traced by the path loop.
  - A 28-step refracted march finds what lies beneath (riverbed, the
    submerged stairs). Beer–Lambert absorption acts along both the view path
    and the sun path, with in-scattering and cheap caustics.
- **Sky and fog** (§14): the shared analytic sky (`src/shared/sky.wgsl`) and
  exponential height fog integrated in closed form. The fog colour is the sky
  radiance in the same direction, so geometry fades into exactly the sky
  behind it. A fade over the last quarter of the march distance hides the far
  clip: there is no seam between terrain and sky.

### Performance

GUI: resolution scale (default 0.75), 2×2 in-shader supersampling, max steps
(200), over-relaxation ω (1.5), hit epsilon (0.6 px), max distance (650 m),
bounding volumes and fBm LOD on/off, reflection bounces, shadow steps (64),
penumbra k, AO. The HUD shows the live cost meter.

Measured per pixel with the shader's counters, at the defaults, 480×270, 1
bounce (headless Chromium on SwiftShader; counts are hardware independent):

| Viewpoint | primary steps | shadow steps | reflection steps | map() calls | fBm octaves | group evals | out of steps |
|---|---|---|---|---|---|---|---|
| Temple entrance | 25.6 | 32.5 | 8.7 | 77 | 48 | 43 | 0 % |
| Overlook | 27.1 | 31.5 | 3.2 | 72 | 139 | 4 | 0 % |
| Character close-up | 25.8 | 38.9 | 16.6 | 93 | 62 | 63 | 0.04 % |
| Sculpture | 27.4 | 32.5 | 11.4 | 81 | 176 | 12 | 0 % |
| Rotunda | 25.5 | 24.1 | 1.3 | 60 | 54 | 14 | 0 % |
| Aqueduct and bridge | 29.2 | 25.4 | 3.1 | 67 | 157 | 5 | 0.01 % |

At 960×540 the same views need 5–8 % more (entrance: 27.5 primary steps,
79 map calls), because the epsilon is in pixels. What each optimisation buys:

| | Temple entrance | Overlook |
|---|---|---|
| primary steps, ω = 1.0 / 1.3 / **1.5** / 1.8 | 28.6 / 24.3 / **25.6** / 29.1 | 37.7 / 29.7 / **27.1** / 30.6 |
| group evaluations per pixel, bounds off → on | 542 → 43 (12.7×) | 504 → 4 (126×) |
| fBm octaves per pixel, LOD off → on | 56 → 48 | 198 → 139 (−30 %) |

Over-relaxation pays most on long grazing terrain rays (−28 % steps on the
overlook). Above ω ≈ 1.5 the failed steps (which step back) eat the gain.
The steps heatmap shows where it helps: the distant slopes and the sky
just above the mountains, the red band where rays graze a ridge.

**Cost estimate (not measured on a GPU).** About 60–95 map() calls per pixel.
A call averages very roughly 200–400 ALU operations: plateau terrain is ~30,
warped terrain ~150 plus ~50 per octave, and the temple group ~400. That is
about 20–30 k operations per pixel, or 45–65 G per 1080p frame. On a
~10 TFLOPS mid-range GPU at a realistic 30–50 % efficiency (divergent loops),
that is roughly 10–20 ms at full 1080p and 5–11 ms at the default 0.75 scale.
It is only verified on SwiftShader (CPU), where 480×270 takes 5–20 s per frame.

### Debug views

`View` in the GUI:
- **Steps heatmap**: primary steps, 0 → 100, turbo colour map with a
  legend bar along the bottom.
- **Cost heatmap**: every map() call of the pixel (primary, normals, shadow,
  AO, reflection, water), 0 → 400.
- **Normals**, **AO only**, **Shadows only**.
- **Material ids**, blended where smooth unions blend materials.
- **Distance field slice**: a horizontal or vertical plane at a chosen
  offset, coloured by map() (orange outside, blue inside). It has contour lines
  every *spacing* metres, antialiased by the pixel footprint, and a white
  zero iso-line. It shows what the tracer really sees, including the
  Lipschitz-scaled terrain distance and the lower bounds returned by
  bounding volumes.

### Known limitations

- One sample per pixel by default. Fine detail (flutes, balusters, triglyph
  grooves, distant aqueduct arches) aliases and shimmers in motion. The
  in-shader 2×2 supersampling fixes that at 4× the cost. Temporal
  anti-aliasing is ruled out by the no-history constraint.
- The terrain's local Lipschitz estimate is first order. On sharply convex
  bumps it can overstep a few centimetres before the pull-back.
- Shadows in terrain penumbras come out softer and darker than they should,
  because the Lipschitz scaling makes terrain distances underestimate.
- Soft shadows, AO and the gyroid are approximations. The gyroid sheet
  distance is only a bound, and the ellipsoids use the usual bound, not an
  exact distance.
- Reflections are mirror-only, blended to a sky-based term for rough
  surfaces; there are no glossy traced reflections. Water refraction ignores
  total internal reflection, and the caustics are a noise pattern, not
  photon-derived.
- The walker's feet are not solved by IK: the planted foot can slide a little
  over the stride.
- There is no collision: the camera can fly into the geometry. The eye is
  only kept above the water.
- Numbers come from headless Chromium on SwiftShader. Real-GPU frame times
  are an estimate, not a measurement.
