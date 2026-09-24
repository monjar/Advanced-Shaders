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
