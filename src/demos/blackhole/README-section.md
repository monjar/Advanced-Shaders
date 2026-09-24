## 04 · Black hole lensing

Light bent around a Schwarzschild black hole, with a thin accretion disk and a
procedural sky behind it. The study starts from the usual artistic shortcut
and ends at exact null geodesics. Every stage stays selectable, and a split
screen puts the artistic and physical renders side by side. Units are
Schwarzschild radii (r_s = 2GM/c² = 1) and r_s/c. Presets: *Inclined*,
*Edge-on (Interstellar-like)*, *Face-on*, *Einstein ring alignment*,
*Artistic vs physical (split)*, *Shadow measurement*, *Photon ring close-up*,
*Free fall at 5 r_s*.

![Edge-on disk](docs/blackhole-edge-on.jpg)
![Artistic approximation | Schwarzschild geodesics](docs/blackhole-artistic-vs-physical.jpg)
![Image order m at the shadow edge](docs/blackhole-photon-ring-orders.jpg)

### Frame graph

```
compute  trace        per pixel: camera ray (+ aberration), geodesic, disk crossings,
                      prefiltered sky lookup → HDR, debug data, statistics (atomics)
compute  accumulate   progressive mean of jittered frames (or a copy)
compute  bloom down   6 levels, 13-tap, Karis average on the first
compute  bloom up     5 levels, 3×3 tent
render   composite    bloom mix, exposure, ACES, sRGB, debug views, split-screen line
```

### Techniques and where they live

- **Stage 1, artistic** (`geodesic.wgsl` `traceArtistic`): march in flat space
  with a step ∝ r and pull the direction towards the hole with an
  inverse-square term, `v = normalize(v - k x/r³ ds)`. The horizon is a black
  sphere at r = r_s, and the disk is a plane hit found by linear
  interpolation, with a fixed colour ramp and no redshift. It has no photon
  sphere, so the higher-order images and the shadow size are wrong.
- **Stage 2, exact geodesics in the orbital plane** (`tracePlane`). A photon
  moves in the plane spanned by the hole, the camera and the ray. In that
  plane u = 1/r obeys the Binet-form orbit equation
  `d²u/dφ² + u = 3Mu² = (3/2) r_s u²`. Each ray gets a basis (e1 towards the
  camera, e2 the direction it turns towards, e3 = e1 × e2) and becomes a 1D
  ODE in φ.
  - *Initial slope.* The camera is a static observer, and its local frame
    stretches the radial component by (1 - r_s/r)^(-1/2). A ray at angle θ from
    the outward radial therefore starts with `du/dφ = -u √(1 - u) cot θ`
    (r_s = 1). Check: this satisfies the first integral
    `u'² = 1/b² - u² + u³` with Synge's `b = r sin θ / √(1 - r_s/r)`.
  - *Integration.* RK4 at a fixed dφ, or adaptive Dormand–Prince 5(4) (the
    ode45 pair, FSAL). Its error norm is `|δ(u, u')| / |(u, u')|`, which for a
    straight line `u = sin(φ∞ - φ)/b` is exactly the escape-angle error,
    whatever the impact parameter. A fixed dφ is already a step ∝ r in space.
  - *Exact events.* The photon plane cuts the disk plane along a line of
    nodes, so disk crossings happen at φ = φ_ℓ + kπ. Steps are clipped to land
    on them exactly. Escape is the root u = 0 on the last step's cubic Hermite
    interpolant: an exact asymptotic direction with no escape radius and no
    tail correction. A ray inside the photon sphere with u' > 0 has u'' > 0 and
    can never turn back, so it is stopped there as captured.
- **Stage 2b, Cartesian cross-check** (`traceCartesian`). The same geodesics
  as flat-space motion under a pseudo-force. For a central force f(r), Binet's
  equation is `u'' + u = f/(h²u²)`. Setting `f = (3/2) r_s h² u⁴` gives
  `a = -(3/2) r_s h² r̂ / r⁴` with `h = |r × v|` conserved: the constant is 3/2,
  not 3. It is integrated with RK4 at a step of k·r in 3D, with the same
  metric-corrected start velocity. Disk hits come from a Hermite/Newton plane
  crossing. The r⁻⁴ force decays fast, so the path stops at 2·r_obs and adds
  the remaining bending to first order along a straight line: the integral of
  `(3/2) b³ ∫ ds/(b² + s²)^(5/2)`, with a series form for s ≫ b.
- **Ray differentials** (`tracePlane`, `env.wgsl`). Alongside u the shader
  integrates the variational equation `w'' = (-1 + 3u) w` for w = ∂u/∂p,
  where p = du/dφ at the camera. At escape, `∂φ∞/∂p = -w/u'`. A pixel step
  changes p through θ and rotates the orbital plane about e1. Those two
  derivatives come from the camera's own differentials (central differences
  of the camera and aberration mapping). Together they give the exact
  Jacobian of the escape direction with respect to the pixel, J = [∂D/∂x,
  ∂D/∂y], without tracing extra rays. The same chain rule, with dφ_c from
  `N·e(φ_c) = 0`, gives the hit-point differentials on the disk. The
  alternative *Finite differences* mode traces the neighbouring rays
  instead, falling back to the other side when a neighbour falls in. The
  Cartesian integrator carries no differentials, so it uses finite
  differences too.
- **Prefiltered sky** (`env.wgsl`). Each pixel integrates the sky against a
  Gaussian with covariance `Σ = k² J Jᵀ` on the celestial sphere (EWA
  filtering in the spirit of Heckbert 1989 and Igehy 1999). Lensing conserves
  surface brightness, so this is what a camera records: magnified stars get
  brighter, demagnified sky averages out.
  - *Stars.* Nine levels of an equi-angular cube map, 4·2^L cells per face
    edge, at most one star per cell, 0.3× the flux per level (4× the count). A
    level is summed star by star (2×2 cells, so the kernel's 3σ must fit in
    half a cell) while it is resolved. Once the footprint outgrows it, the
    level fades to its exact mean radiance, `flux · presence · density /
    Ω_cell`: the level's "top mip". Stars in edge cells keep half a cell away
    from the face edge, so no star is ever needed from across a seam.
  - *Galaxy.* A Gaussian band and bulge with fBm structure, dust lanes and Hα
    patches, all functions of the 3D direction (no seams). Noise octaves fade
    with the footprint. The Gaussian profiles are blurred analytically
    (variance v → v + σ²), so right at the photon ring the band spreads over
    the sky instead of aliasing.
  - *Colour.* Stars and galaxy light are blackbodies, so the observer's
    blueshift turns T into g·T for them too.
- **Accretion disk** (`disk.wgsl`). A thin annulus from the ISCO (3 r_s) out.
  - *Temperature.* `T ∝ r^(-3/4) (1 - √(r_in/r))^(1/4)`, peaking at
    49/36 r_in (zero-torque inner edge, so the ISCO rim is dark). The peak
    temperature is a display choice (4600 K by default); a real disk is
    10⁴–10⁷ K.
  - *Blackbody table* (`blackbody.ts`). Planck spectra integrated against the
    Wyman–Sloan–Shirley CIE fits, converted to linear sRGB, and white-balanced
    to 6500 K.
  - *Turbulence.* fBm in (azimuth, ln r), with streaks about 4× longer than
    wide, advected at the Keplerian Ω = √(M/r³). Differential rotation would
    wind any pattern up without bound. Two copies are advected over a finite
    cycle (default: one inner-edge orbit), restarted with fresh seeds, and
    cross-faded with a variance-preserving weight (flow noise with periodic
    reset). Noise octaves are prefiltered with the hit-point differentials,
    measured in the noise's own anisotropic metric.
  - *Coverage.* `1 - exp(-τ ρ)` per crossing, front to back, so gaps show
    the higher-order images and the sky behind.
  - *Light travel time.* The coordinate time `dt/dφ = r²/(b(1 - r_s/r))` is
    integrated with the same stages as u. In the Cartesian form it is
    `dt/dλ = √(1 - r_s/r_obs)/(1 - r_s/r)`. Each crossing samples the disk at
    `t_now - Δt`. The secondary and photon-ring images therefore lag the
    direct image by up to a few tens of r_s/c (the *Light travel time*
    toggle).
- **Redshift** (`diskRedshift`). The gas moves on circular geodesics with
  local speed `β = √(M/(r - 2M))` along φ̂ = N × r̂, as a static observer
  measures it. The photon's direction at the hit comes from the plane state
  `(u, u')`. The Doppler factor is `δ = 1/(γ(1 - β φ̂·k))`, the gravitational
  factor is `√(1 - r_s/r)`, and the observer's blueshift is
  `1/√(1 - r_s/r_obs)`. The product is algebraically the textbook
  `g = √(1 - 3M/r) / (1 - Ω b_z)`: substitute γ, and
  `φ̂·k = (b_z/r) √(1 - r_s/r)`.
  - *Intensity.* I_ν/ν³ is invariant, so a blackbody at T is seen as a
    blackbody at g·T. The default *Spectral* mode looks up g·T. That is
    exact, and it is what the eye sees in the visible band. The Wien side
    makes the brightness change faster than g⁴ at display temperatures.
  - *Other modes.* *× g⁴* is the bolometric (frequency-integrated) law, as
    in Luminet 1979 and most papers. *× g³* is specific intensity at a fixed
    frequency, for a flat spectrum. Both keep the colour at T. The toggles
    switch Doppler and gravity separately.
- **Observer** (`trace.wgsl`). The camera is a static observer at r_obs.
  Optionally it moves: free fall from rest at infinity (β = √(r_s/r) inwards),
  or a circular orbit about the disk axis. Each pixel's direction is aberrated
  into the static frame, and the observer Doppler factor γ(1 + β·n) scales
  every g. An off-axis view window (lens shift) zooms into the photon ring
  without moving the camera.
- **Accumulation** (`accumulate.wgsl`). With *Accumulate* on, time freezes
  and R2-jittered frames are averaged until anything changes. It is used for
  reference images and the sub-pixel shadow measurement. The live image
  relies on the prefiltering.

### Verification

Measured in headless Chromium on SwiftShader. The float64 references come
from a CPU integrator (RK4, dφ = 2·10⁻⁴).

| Quantity | Theory | Rendered |
|---|---|---|
| critical impact parameter b_c (float64 bisection) | 3√3/2 = 2.598076 | 2.598076 |
| shadow b from captured-pixel area, 16 jittered frames, r_obs = 10 / 30 / 100 | 2.5981 | 2.5978 / 2.5980 / 2.5982 (≤ 0.01 %) |
| same, Cartesian integrator, r_obs = 30 | 2.5981 | 2.5980 |
| shadow edge on a 3·10⁻⁶° pixel scan, r_obs = 26 (Synge) | 5.623179° | first captured pixel 5.623177–5.623183° |
| Einstein ring of the beacon star, r_obs = 30 (float64 φ∞ = π; weak field gives 14.79°) | 16.1825° | 16.1826° (brightness-weighted radius) |
| face-on disk, b_z = 0: g = √(1 - 3M/r) / √(1 - r_s/r_obs) at r = 3 / 14 | 0.723 / 0.967 | 0.724 / 0.967 |
| photon ring images: distance of the m = 3, 4, 5 band edges from the shadow edge | ratio e^π = 23.1 | 0.00786°, 0.000335°, ≈ 0.000016°: ratios 23.5 and ≈ 21 |
| artistic stage (pull 1), r_obs = 30 | 4.884° | 4.960° (equivalent b = 2.638) |

The shadow is measured in the app. Captured pixels are counted with GPU
atomics. A disc of angular radius α centred on the view axis covers
π tan²α of the image plane, so α = atan √(N·A_px/π), which is inverted with
Synge's `sin α = (b_c/r_obs) √(1 - r_s/r_obs)`. The HUD shows it live; for
exact numbers use the *Shadow measurement* preset, with the disk off and
accumulation on.

Image order m is the number of disk-plane crossings before escape or capture
(Gralla, Holz and Wald 2019). The *Photon ring close-up* preset shows the m = 2
lensing ring and the m ≥ 3 photon ring, which each shrink by e^π per order.
At an inclination of 11° the images that hit the disk alternate with bands
that cross the plane inside the ISCO.

Accuracy and cost of the integrators (Inclined preset, 480×270, disk on).
The reference is RK4 at dφ = 0.01. Escape-direction errors are against the
float64 integrator over 142 sample pixels. One 1080p pixel at 50° is
8·10⁻⁴ rad.

| Integrator | steps/px (max) | derivative evals/px | mean \|Δ\| vs reference | escape direction error (mean / max) |
|---|---|---|---|---|
| plane RK4, dφ = 0.2 | 15.5 (49) | 62 | 0.064/255 (0.06 % px > 8/255) | |
| plane RK4, dφ = 0.08 | 37.3 (120) | 149 | 0.004/255 | 8·10⁻⁶ / 2·10⁻⁵ rad |
| plane DP5(4), tol 10⁻⁴ | 6.7 (22) | ~40 | 0.046/255 | |
| **plane DP5(4), tol 10⁻⁶ (default)** | 12.5 (31) | ~75 | 0.001/255 | 9·10⁻⁶ / 2·10⁻⁵ rad |
| Cartesian RK4, step 0.1 r | 41.2 (140) | 165 | 0.36/255 (0.03/255 with equal footprints) | 1·10⁻⁶ / 1.5·10⁻⁵ rad |
| artistic, step 0.04 r | 102.6 (388) | 103 | 12.2/255 (20 % px > 8/255) | wrong physics |

All the geodesic integrators agree far below a pixel. The plane
integrators' ~10⁻⁵ rad is a float32 floor that does not shrink with the step.
Coarse RK4 steps show up in the light-travel-time integral first (the disk
pattern shifts), not in the directions. Most of the Cartesian difference
comes from its cruder footprints (finite differences for the sky, unlensed
for the disk). With footprints off in both, it is 0.03/255 whatever its step
(k = 0.2 → 0.05 gives the same). Its escape directions are actually closer
to float64 (10⁻⁶ rad); the lensed galaxy near the Einstein ring is magnified
enough to show the plane reference's 10⁻⁵ rad. The two formulations also
agree on the light-travel delay at the first disk hit: mean difference
2·10⁻⁴ r_s/c over 12,900 pixels, at the f16 storage limit, for delays of
13–60 r_s/c. The finite-difference footprint costs 24 extra steps/px on top
of 12.6 with DP5(4); the analytic one costs two more ODE components.

Sparkle, measured as frame-to-frame change in a slow orbit (0.0005 rad/frame
≈ 0.15 px, 480×270, disk off):

| Footprint | 1–1.25 shadow radii | 1–2 shadow radii | whole frame |
|---|---|---|---|
| none (unlensed pixel, typical shadertoy) | 0.259/255, 0.40 % px > 8/255 | 0.236/255 | 0.258/255 |
| analytic ray differentials | 0.006/255, 0 % | 0.063/255 | 0.230/255 |
| finite differences | 0.006/255, 0 % | 0.063/255 | 0.230/255 |

The whole-frame number is dominated by ordinary unlensed stars moving. Next
to the shadow the flicker drops about 40×. The filter conserves energy. Mean
radiance in annuli, 1 spp prefiltered vs 64 jittered point samples: within
0.3 % at 2–4 shadow radii and 0.9 % at 1.3–2. At 1.1–1.3 it is 12 % higher,
where a handful of demagnified bright stars dominate the 64-sample
reference.

No non-finite pixel appeared in any test (the HUD counts them): static
observers at 1.6 r_s looking in, out and sideways, free fall at 1.2 r_s
(inside the photon sphere), a circular orbit at 3 r_s in the disk plane, and
all integrators and presets. Static observers are kept outside 1.05 r_s.

### Performance

The trace is one compute pass with no textures. Per pixel it runs about 75
derivative evaluations (DP5(4)), up to 9 × 4 star cells, 15 gradient-noise
octaves for the galaxy, and 16 per disk crossing. The procedural noise, not
the integration, dominates. By operation count that should fit 1080p60 on a
mid-range GPU, but real GPUs have not been measured. On SwiftShader (CPU)
the default view takes about 2 s per frame at 480×270 and 12–18 s at 960×540.
The knobs: integrator and tolerance/dφ, footprint mode (finite differences
roughly triples the ray count), disk optical depth (opaque disks stop rays
early), and *Statistics in HUD* (workgroup-reduced atomics).

### Debug views

`View` in the GUI:
- *Steps per pixel*: log scale, including rejected adaptive steps.
- *Disk crossings (image order)*: m = 1 / 2 / 3 / ≥ 4 in orange / green /
  magenta / white; full colour where the disk was hit, dim where it wasn't.
- *Escaped / captured*: also shows the step limit and rays absorbed by the
  disk.
- *Redshift factor g*: at the first crossing, red below 1 and blue above,
  log scale from ½ to 2.
- *Sky footprint size*: log₂ of the footprint's linear size against an
  unlensed pixel.

The HUD adds steps/px, capture fractions, image-order counts, non-finite
pixels and the live shadow measurement.

### Known limitations

- Schwarzschild only: no spin (Kerr), so no frame dragging and no asymmetric
  shadow.
- Light travel time delays when each image samples the disk, but the
  observer's own proper time is not used: the animation runs in coordinate
  time (a factor √(1 - r_s/r_obs) off for a static observer).
- The disk is geometrically thin, with no vertical structure,
  self-illumination, corona or jet. Its opacity is a single 1 - e^(-τ) layer
  per crossing. The temperature is a display temperature.
- The turbulence cross-fade restarts the pattern once per cycle. Far out,
  where it barely turns within a cycle, it can read as slow "breathing"
  rather than rotation.
- The disk texture filter is isotropic: it uses the largest axis of the
  footprint ellipse, so grazing views blur more than an anisotropic multi-tap
  filter would. The artistic and Cartesian paths use an unlensed footprint
  estimate for the disk.
- Geometric edges (the shadow rim, the disk's inner edge, image-order
  boundaries) are not prefiltered. Only accumulation antialiases them, and the
  m ≥ 3 rings are sub-pixel at normal zoom.
- The circular-orbit observer is a geodesic only in the disk plane. The
  camera orientation is taken from the orbit camera, not parallel-transported.
- Verified in headless Chromium on SwiftShader only.
