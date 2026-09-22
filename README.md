# Advanced Shaders

A WebGPU/WGSL playground for studying advanced real-time shading techniques.
Each study is a self-contained "demo" registered in a shared scene shell
(sidebar navigation, orbit camera, parameter GUI, HDR pipeline).

| # | Study | Techniques |
|---|-------|-----------|
| 01 | [Deep-water ocean](src/demos/ocean) | compute-shader FFT, vertex displacement, procedural animation, BRDFs, screen-space refraction, caustics, GPU buoyancy |

![Open water](docs/ocean-open-water.jpg)
![Shoreline](docs/ocean-shoreline.jpg)

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
  demos/ocean/         01 · deep-water ocean
    index.ts           resources and frame graph
    params.ts          defaults, presets, GUI
    sky.ts             CPU mirror of the sky model (sun colour, ambient)
    shaders/*.wgsl
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
