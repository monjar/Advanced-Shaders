## 05 · Procedural planet

An Earth-sized planet (radius 6,360 km) with no textures at all: continents,
shelves, mountain chains, biomes, snow, oceans, clouds and city lights all
come from 3D noise evaluated on the sphere, so there are no UV seams and
nothing pinches at the poles. The atmosphere is the shared module from
study 04. One continuous camera goes from 20,000 km to walking height
(1.7 m) with no loading steps and no precision jitter. Presets: *Orbit, day
side*, *Terminator, city lights*, *Low orbit sunset*, *Flyover, mountains*,
*Coast*, *Ground level*.

### Frame graph

```
CPU      double-precision camera; LOD selection for the view and for two
         sun cascades; bake queue; per-octave noise reference tables
compute  chunk heights + morph      nodes baked this frame (33×33 + skirts each)
compute  atmosphere LUTs            shared module (04)
render   sun shadow ×2              depth only, chunks selected per cascade box
render   gbuffer                    every visible chunk in one instanced draw, reversed-Z
compute  shade                      per pixel: terrain again, biomes, ocean, clouds, cities, atmosphere
render   composite                  exposure, ACES, sRGB, debug views
```

### Techniques and where they live

- **Precision: per-octave lattice split** (`terrain.ts`, `terrain.wgsl`).
  Every octave is 3D gradient noise with analytic derivatives on a cubic
  lattice of spacing 2^(22−level) m (level 0: 4,194 km, level 23: 0.5 m).
  float32 cannot hold a 6,360 km coordinate to better than 0.5 m, so a
  point is never passed in absolute form. For each octave the CPU splits a
  reference point, rotated and scaled in double precision, into an integer
  lattice cell and a float fraction; the GPU evaluates
  `noise(cell + floor(q), q − floor(q))` with
  `q = frac + R·(x − ref)·f`, where `x − ref` is small. The reference is
  the chunk origin while baking vertices and the camera while shading, so
  noise inputs keep ~mm precision at walking height. Each octave ("slot",
  100 in total) has its own rotation and hash seed; because the rotation is
  applied to the reference in double, octaves can be decorrelated by
  rotation without breaking the split, and a non-orthonormal map works too
  (cloud octaves are squashed north–south into zonal bands). The same
  height function runs on the CPU in double for the camera's ground clamp
  and the LOD bounds.
- **Terrain** (`terrain.wgsl` `terrainEval`): continents are 9 octaves of
  fBm under a 3-component continent-scale domain warp (with the warp's
  Jacobian in the chain rule); a base profile maps the continent value to
  abyssal plain, continental slope, a shelf ramping to the coast and rising
  land. Mountain chains follow the zero lines of a low-frequency mask
  (`1 − 6|m|`), carry a 13-octave ridged multifractal (Musgrave, each
  octave weighted by the previous; squared to deepen valleys) and fade
  towards coasts; everywhere on land a low version of it makes hills.
  "Erosion" is Quilez's derivative-damped fBm (14 octaves down to 0.5 m),
  damped by the accumulated slope in each octave's own lattice units so all
  LODs damp identically. Every term returns its gradient analytically.
- **Cube-sphere quadtree** (`quadtree.ts`): six faces with the equal-angle
  warp `tan(u π/4)` (cells within ~1.4× in area). A node splits when one of
  its 32×32 quads would cover more than 5 px; it only splits once its
  children in view are baked (or can be covered by baked descendants), so
  the drawn set always covers the sphere without overlaps. Culling: frustum
  side planes and a horizon test against a sphere 11 km below sea level.
  Bounds come from a 3×3 CPU height sample plus a size-dependent margin.
  1,536 pool slots (60 MB of vertices), LRU eviction of chunks not drawn
  this frame, a bake budget per frame (24 by default), coarsest and nearest
  first. Levels go to 19: 19 m chunks with 0.6 m vertex spacing.
- **Chunk bake** (`chunk.wgsl`): vertices are written relative to the node
  origin O = R·dir(centre), which stays in double on the CPU. The offset
  `dir − dir0` is formed as `dc/|c| + c0 (1/|c| − 1/|c0|)` with
  `1/|c| − 1/|c0| = −(2 c0·dc + dc·dc)/(|c||c0|(|c| + |c0|))` and
  `tan a − tan b = sin(a − b)/(cos a cos b)`: every term is proportional to
  the small offset, so float32 keeps ~10⁻⁷ of the chunk size (2 µm for a
  20 m chunk). Neighbouring chunks therefore agree on shared edge vertices
  to that precision even though their origins differ. Each vertex also
  stores the position its parent LOD would have there (the parent's octave
  set at even vertices, the parent's triangle for odd ones).
- **Morphing and skirts** (`gbuffer.wgsl`): CDLOD-style. The vertex shader
  blends to the parent position between 1.35 and 1.85 times the node's split
  distance, so a level is fully morphed where the next coarser one begins,
  and the triangulation (main diagonal where i + j is even) nests exactly
  in the parent's triangles. Skirts (8 % of the node size + 20 m) hide the
  gaps that remain while the LOD is still catching up with a fast camera.
- **Camera-relative rendering** (`src/shared/planet-camera.ts`,
  `index.ts` `writeDraws`): the camera position is a double; each draw gets
  `O − camera` subtracted in double and then rounded to float, the only
  large-to-small step. The view matrix has no translation. Depth is
  reversed-Z with an infinite far plane (float32 depth keeps the same
  relative precision at every distance); the near plane only has to stay in
  front of the nearest geometry and follows 2 % of the height above ground
  (5 cm at walking height, 20 km from orbit). Speed is proportional to the
  height above ground and the heading is parallel-transported.
- **Per-pixel shading** (`shade.wgsl`): the G-buffer holds only the
  camera-relative position and the interpolated height, so the sphere point
  is `P − h·up`. The full height function is evaluated again per pixel in
  the camera's noise frame, with octaves weighted by the pixel footprint
  (full while a wavelength spans ≥ 4 footprints, gone at 2; the footprint
  is measured from the neighbouring G-buffer texels, so slopes seen edge-on
  get fewer octaves): normal, height, biome and snow line never depend on
  the mesh LOD, so LOD changes only move silhouettes.
- **Biomes**: temperature from latitude, a 6.5 K/km lapse rate and noise;
  moisture from circulation cells (wet equator and ~60°, dry ~30° and
  poles), distance to the coast and noise; they blend desert, savanna,
  jungle, grassland, forest, taiga, tundra and ice. Rock on steep slopes and
  high ground, beaches near sea level, snow above a latitude-dependent snow
  line (5,200 m at the equator, sea level near 70°) except on steep slopes,
  sea ice near the poles. A 13-octave albedo channel with a flatter
  spectrum (gain 0.78) keeps patches visible at every distance and makes
  the snow line and slope threshold patchy.
- **Ocean**: sea level is the mesh clamped at 0; the per-pixel height gives
  the depth, so coastlines are exact per pixel. Beer–Lambert over a sandy
  seabed (turquoise shallows, navy deep water), Schlick Fresnel with sky
  reflection, GGX sun glint; six octaves of animated wave slopes, with the
  slope variance the footprint cannot resolve moved into the roughness, so
  the glint widens smoothly from orbit. Reflections are kept above the
  horizon.
- **Clouds**: one layer at 5.5 km with a vertical optical depth from
  coverage noise: the three lowest octaves (2,000–520 km) make the systems,
  a curl-like displacement along `up × ∇n` swirls them, latitude bands add
  the ITCZ, subtropical highs and storm tracks, and each octave drifts in
  its own direction (so clouds evolve rather than slide). Lit by the
  atmosphere's sun transmittance at the cloud point with a two-stream-style
  slab reflectance that rises towards grazing sun, diffuse transmission when
  seen from the other side, a forward-scattering term for thin edges, and
  sky irradiance; composited with the aerial perspective to the cloud.
  Shadows on the ground march the sun ray to the layer.
- **Filtering thresholds**: coverage and city masks are thresholds of noise
  that the footprint removes octaves from. The removed variance is tracked
  and the step widened by it (`filteredStep`), so thresholded patterns fade
  to their correct mean instead of vanishing when seen from far away.
- **City lights**: cities are 3D Gaussian "balls" in cells of three
  lattice levels (33 km, 8 km and 2 km cells), present with a probability
  from a regional population field times habitability (temperate, moist,
  low, not snow or rock, biased to coasts), sliced by the surface. The
  footprint blurs each ball with its mass conserved, and once it spans a
  cell the sum becomes the cell's mean density, so cities are sharp spots
  close up and the right average glow from orbit, with metropolises
  staying distinct points. Street texture from the finest octaves; they
  switch on as the sun sets.
- **Terrain shadows**: two orthographic sun cascades (2,048², extents
  from 2.5 km to 800 km with the height above ground), each with its own LOD
  selection culled by its box rather than the view frustum (so mountains
  behind the camera still cast), a coarser error for casters, texel-snapped
  in double precision so they do not shimmer, 3×3 PCF with a normal offset.
- **Atmosphere** (`src/shared/atmosphere`): sky and limb glow (ray marched
  per pixel above 80 km, LUT below), aerial perspective over terrain and
  clouds, sun transmittance for ground, water and clouds (which gives the
  red terminator), sky irradiance for ambient light.

### Measured

- **No precision jitter at walking height.** Two frames rendered with the
  LOD frozen, the camera 1.7 m above a mountainside, then moved along its
  heading (480×270, SwiftShader): no move, 0 difference; 1 mm, mean
  0.10/255 and 0.013 % of pixels changing by more than 2/255; 1 cm,
  0.38/255 and 0.43 %; 10 cm, 2.3/255 and 29 % (the real parallax grows
  with the move; a precision problem would not). The difference images are
  smooth parallax plus a few silhouette pixels. The first version of this
  test also caught normals flipping on ridge crests (`1 − |n|` has a
  crease, 0.058 % of pixels at 1 mm): the crease is now rounded over ~1.5
  pixel footprints (`sqrt(n² + ε²)`, ε = 0.01 for the mesh and the CPU).
- **CPU and GPU agree**: looking straight down from the clamped eye height
  (1.7 m above the CPU terrain), the G-buffer probe measures 1.69 m to the
  rendered surface.
- **LOD** (640×360, 5 px error): 27 chunks from 13,000 km, ~100 from low
  orbit, ~300 flying at 9 km, 470–520 at ground level in the mountains
  (levels 0–19). The pool of 1,536 slots fills near the ground and evicts
  without stalling. In the scripted orbit-to-ground descent (10 s at one
  fixed step per frame) the LOD keeps up with a budget of 256 bakes per
  frame; level 19 is in place while the camera is still ~8 m above the
  ground.
- No seams across cube-face edges or at the poles in the height, normal and
  LOD views; checked at the (1, 1, 1) cube corner and over the north pole.

### Cost

Per pixel the shading pass evaluates up to ~110 noise octaves with
derivatives (48 terrain, 21 biome, ~26 cloud and cloud shadow, 6 waves,
city lattices), fewer from orbit where the footprint removes them. At
1080p that is on the order of 30 GFLOP per frame, i.e. a few milliseconds
on a mid-range GPU. Geometry is ~520 chunks × 2,304 triangles for the view
plus the casters. On SwiftShader (CPU) a 480×270 frame takes 1.5–4 s. Not
timed on a real GPU.

### Debug views

Height, biome albedo, normals, LOD chunks (level colour, per-chunk tint,
morph darkening), cloud shadow (cloud / cloud × terrain / terrain in RGB),
clouds only, habitability. The HUD shows altitude, height above ground,
latitude/longitude, near plane, chunks drawn / resident / baked / pending.

### Known limitations

- Clouds are a single slab (optical depth, no volumetric parallax); peaks
  above the layer are simply in front of it. Clouds do not shadow each
  other and there are no light shafts.
- Terrain shadows are limited to the two cascades (80 km and 800 km at the
  most); beyond them terrain is unshadowed. The atmosphere's in-scattering
  ignores terrain shadows.
- A camera that moves faster than the bake budget allows leaves coarse
  chunks for a few frames; the skirts hide the resulting gaps, but the
  detail visibly pops in. Shots raise the budget to 256.
- The finest mesh has 0.6 m vertex spacing and the finest normal octave a
  0.5 m wavelength; below that surfaces are smooth. No rocks, grass or
  rivers; "erosion" is derivative-damped noise, not a simulation.
- Mountains cannot reach the coast (the mask fades them over the first
  part of the continent value), so coasts are low plains.
- Biome rules, colours and the city model are hand-tuned; ocean waves are
  normal-mapped noise, not a spectrum.
- Verified in headless Chromium on SwiftShader only.
