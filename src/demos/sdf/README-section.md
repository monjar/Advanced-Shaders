## 04 · SDF world

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
