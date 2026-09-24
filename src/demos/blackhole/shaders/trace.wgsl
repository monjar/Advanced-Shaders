// One compute invocation per pixel: build the camera ray (with the moving
// observer's aberration), trace it with the selected integrator, shade the
// disk crossings, look the escape direction up on the prefiltered sky, and
// write HDR colour, debug data and statistics.

@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>, 16>;
@group(1) @binding(0) var hdrOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var auxOut: texture_storage_2d<rgba16float, write>;

// Statistics slots (index.ts reads them back). Slots 0-3 count ray status.
const ST_CAPTURED: u32 = CAPTURED;
const ST_STEPS: u32 = 4u;
const ST_MAX_STEPS: u32 = 5u;
const ST_NONFINITE: u32 = 6u;
const ST_ORDER1: u32 = 7u;
const ST_ORDER2: u32 = 8u;
const ST_ORDER3: u32 = 9u;
const ST_ACC_CAPTURED: u32 = 10u;
const ST_ACC_FRAMES: u32 = 11u;
const ST_EXTRA_STEPS: u32 = 12u;
const ST_NODES2: u32 = 13u;
const ST_NODES3: u32 = 14u;
const ST_COUNT: u32 = 15u;

var<workgroup> wgStats: array<atomic<u32>, 15>;

// Direction through a pixel in the observer's own frame.
fn cameraLocal(px: vec2f) -> vec3f {
  let ndc = vec2f(px.x / F.resolution.x * 2.0 - 1.0, 1.0 - px.y / F.resolution.y * 2.0);
  // Off-axis window (lens shift): zoom into e.g. the photon ring without
  // moving or turning the camera.
  let shift = F.cart.z * normalize(F.camRight) + F.cart.w * normalize(F.camUp);
  return normalize(F.camFwd + shift + ndc.x * F.camRight + ndc.y * F.camUp);
}

// Special-relativistic aberration from the moving observer's frame to the
// local static frame. With k' = -n' the photon's propagation direction in
// the observer frame (velocity addition for light):
//   k = (k' + β [γ + (γ - 1)(β·k')/β²]) / (γ (1 + β·k')).
fn aberrate(n: vec3f) -> vec3f {
  let b = F.obsVel;
  let b2 = dot(b, b);
  if (b2 < 1e-10) { return n; }
  let g = F.obsGamma;
  let kp = -n;
  let bk = dot(b, kp);
  let k = (kp + b * (g + (g - 1.0) * bk / b2)) / (g * (1.0 + bk));
  return -normalize(k);
}

fn cameraRay(px: vec2f) -> CamRay {
  var cr: CamRay;
  cr.n = aberrate(cameraLocal(px));
  // The camera mapping is smooth at pixel scale: central differences are exact enough.
  cr.dnx = aberrate(cameraLocal(px + vec2f(0.5, 0.0))) - aberrate(cameraLocal(px - vec2f(0.5, 0.0)));
  cr.dny = aberrate(cameraLocal(px + vec2f(0.0, 0.5))) - aberrate(cameraLocal(px - vec2f(0.0, 0.5)));
  // Light from infinity gains energy falling to r_obs (1/√(1 - r_s/r)), then
  // the moving observer sees it Doppler shifted by γ(1 - β·k), k = -n.
  var g = 1.0;
  if (F.obsRedshift > 0.5) { g = inverseSqrt(1.0 - 1.0 / F.rObs); }
  g *= F.obsGamma * (1.0 + dot(F.obsVel, cr.n));
  cr.gObs = g;
  return cr;
}

fn traceWith(cr: CamRay, artistic: bool, disk: bool) -> Ray {
  if (artistic) { return traceArtistic(cr, disk); }
  if (F.integ.x > 1.5) { return traceCartesian(cr, disk); }
  return tracePlane(cr, disk);
}

// Escape direction of a neighbouring ray, for finite-difference footprints.
fn neighbour(px: vec2f, artistic: bool, extra: ptr<function, u32>) -> vec4f {
  let r = traceWith(cameraRay(px), artistic, false);
  *extra += r.steps;
  return vec4f(r.dir, select(0.0, 1.0, r.status == ESCAPED));
}

fn finiteDiff(px: vec2f, o: vec2f, dir: vec3f, artistic: bool, extra: ptr<function, u32>) -> vec3f {
  let a = neighbour(px + o, artistic, extra);
  if (a.w > 0.5) { return a.xyz - dir; }
  let b = neighbour(px - o, artistic, extra);
  if (b.w > 0.5) { return dir - b.xyz; }
  // Both neighbours fall in: the pixel straddles the shadow edge. Treat the
  // footprint as very wide (the sky's mean), the safe choice for aliasing.
  return vec3f(0.3);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3u) {
  let statsOn = F.view.z > 0.5;
  if (statsOn) {
    if (li < ST_COUNT) { atomicStore(&wgStats[li], 0u); }
    workgroupBarrier();
  }

  let size = vec2u(F.resolution);
  if (all(id.xy < size)) {
    let px = vec2f(id.xy) + 0.5 + F.jitter;
    let mode = u32(F.view.x + 0.5);
    let artistic = mode == 0u || (mode == 2u && px.x < F.split * F.resolution.x);
    let cr = cameraRay(px);
    let ray = traceWith(cr, artistic, true);

    var colour = ray.radiance;
    var fpLog = 0.0;
    var extra = 0u;
    if (ray.status == ESCAPED && ray.trans > 0.0) {
      var Jx = cr.dnx;
      var Jy = cr.dny;
      let fm = u32(F.env.z + 0.5);
      if (fm == 1u && ray.analytic) {
        Jx = ray.dDx;
        Jy = ray.dDy;
      } else if (fm == 2u || (fm == 1u && !artistic)) {
        // Finite differences on request, and as the fallback for the
        // Cartesian integrator, which carries no differentials. The artistic
        // stage keeps the naive unlensed footprint unless asked.
        Jx = finiteDiff(px, vec2f(1.0, 0.0), ray.dir, artistic, &extra);
        Jy = finiteDiff(px, vec2f(0.0, 1.0), ray.dir, artistic, &extra);
      }
      colour += ray.trans * environment(ray.dir, Jx, Jy, cr.gObs);
      // log2 of the linear footprint relative to an unlensed pixel (> 0: sky
      // compressed into the pixel, < 0: magnified). Shows the true footprint
      // whenever it is known, even if filtering is switched off.
      let a = select(length(cross(Jx, Jy)), length(cross(ray.dDx, ray.dDy)), ray.analytic);
      let a0 = length(cross(cr.dnx, cr.dny));
      fpLog = 0.5 * log2(max(a, 1e-30) / max(a0, 1e-30));
    }

    let ok = finite3(colour) && finite3(ray.dir) && finite(ray.gFirst) && finite(fpLog);
    if (!ok) { colour = vec3f(0.0); }
    textureStore(hdrOut, id.xy, vec4f(colour, 1.0));
    // Packed small integers (exact in f16): shaded crossings, status, bad, plane crossings m.
    let flags = f32(min(ray.crossings, 7u) + 8u * ray.status + select(32u, 0u, ok) + 64u * min(ray.nodes, 7u));
    textureStore(auxOut, id.xy, vec4f(f32(min(ray.steps, 2047u)), flags, select(0.0, ray.gFirst, ok), select(0.0, fpLog, ok)));

    if (statsOn) {
      atomicAdd(&wgStats[ray.status], 1u);
      atomicAdd(&wgStats[ST_STEPS], ray.steps);
      atomicMax(&wgStats[ST_MAX_STEPS], ray.steps);
      atomicAdd(&wgStats[ST_EXTRA_STEPS], extra);
      if (!ok) { atomicAdd(&wgStats[ST_NONFINITE], 1u); }
      if (ray.crossings >= 1u) { atomicAdd(&wgStats[ST_ORDER1], 1u); }
      if (ray.crossings >= 2u) { atomicAdd(&wgStats[ST_ORDER2], 1u); }
      if (ray.crossings >= 3u) { atomicAdd(&wgStats[ST_ORDER3], 1u); }
      if (ray.nodes >= 2u) { atomicAdd(&wgStats[ST_NODES2], 1u); }
      if (ray.nodes >= 3u) { atomicAdd(&wgStats[ST_NODES3], 1u); }
    }
  }

  if (statsOn) {
    // One global atomic per slot per workgroup instead of one per pixel.
    workgroupBarrier();
    if (li < ST_COUNT && li != ST_ACC_CAPTURED && li != ST_ACC_FRAMES) {
      let v = atomicLoad(&wgStats[li]);
      if (li == ST_MAX_STEPS) {
        atomicMax(&stats[li], v);
      } else if (v != 0u) {
        atomicAdd(&stats[li], v);
      }
      if (li == ST_CAPTURED && v != 0u) { atomicAdd(&stats[ST_ACC_CAPTURED], v); }
    }
    if (li == 0u && all(wid.xy == vec2u(0u))) { atomicAdd(&stats[ST_ACC_FRAMES], 1u); }
  }
}
