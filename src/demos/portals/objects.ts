// Moving objects that pass through portals: a bouncing ball (pair A), a
// walking robot (pair B) and a spinning cube looping through the facing pair
// C forever.
//
// Each object is animated in "unfolded" coordinates of one location, as if the
// portal were a plain doorway into the space behind the wall. Once its centre
// is past the portal plane, the canonical transform is carried through the
// pair transform (the object teleports). While it straddles a plane it is
// drawn twice: the original clipped to the front of the source portal and a
// duplicate, moved through the pair, clipped to the front of the destination.
// The two halves meet exactly at the opening, so it looks continuous both
// directly and through the portal.

import { mat4, type Mat4, type Vec3 } from '../../core/math';
import { insideOpening, m4, planeDist, type Portal, type Vec4 } from './portal-math';
import { locationOf, type ObjectMeshes, type Range } from './scene';

export interface ObjectDraw {
  range: Range;
  model: Mat4;
  clip: Vec4;
  /** 0: not crossing, 1: crossing (original side), 2: crossing (duplicate on the far side). */
  crossing: number;
}

export interface Occluder {
  centre: Vec3;
  radius: number;
}

const NO_CLIP: Vec4 = [0, 0, 0, 1];
const tri = (u: number) => 1 - Math.abs(1 - 2 * (u - Math.floor(u)));
const smooth = (u: number) => u * u * (3 - 2 * u);

interface Body {
  parts: { range: Range; local: Mat4 }[];
  root: Mat4;
  /** Point used for the plane test and its radius along the portal normal. */
  centre: Vec3;
  radius: number;
  occluders: Occluder[];
}

/** Ball: three parabolic hops, the middle one peaking in the opening of portal A. */
function ball(t: number, meshes: ObjectMeshes, portalA: Portal): Body {
  const r = 0.3;
  const hop = 2.5;
  const height = 0.9;
  const speed = hop / Math.sqrt((8 * height) / 9.81); // horizontal speed for a real g bounce
  const length = hop * 3;
  const s = length * tri((t * speed) / (2 * length));
  const zCross = portalA.center[2];
  const z = zCross + length / 2 - s;
  const u = (s - length / 2) / hop + 0.5;
  const f = u - Math.floor(u);
  const y = r + height * 4 * f * (1 - f);
  const root = m4.mul(m4.translation([portalA.center[0], y, z]), m4.rotationX(-s / r));
  return { parts: [{ range: meshes.ball, local: mat4.identity() }], root, centre: [portalA.center[0], y, z], radius: r + 0.02, occluders: [{ centre: [0, 0, 0], radius: r }] };
}

/** Robot: walks through portal B, turns, walks back. */
function robot(t: number, meshes: ObjectMeshes, portalB: Portal): Body {
  const walk = 12;
  const turn = 1.2;
  const period = 2 * (walk + turn);
  const start = portalB.center[0] - 8;
  const tau = ((t % period) + period) % period;
  let x: number, yaw: number, dist: number;
  if (tau < walk) {
    x = start + tau; yaw = Math.PI / 2; dist = tau;
  } else if (tau < walk + turn) {
    x = start + walk; yaw = Math.PI / 2 + Math.PI * smooth((tau - walk) / turn); dist = walk + (tau - walk) * 0.4;
  } else if (tau < 2 * walk + turn) {
    x = start + walk - (tau - walk - turn); yaw = -Math.PI / 2; dist = walk + turn * 0.4 + (tau - walk - turn);
  } else {
    x = start; yaw = -Math.PI / 2 + Math.PI * smooth((tau - 2 * walk - turn) / turn); dist = 2 * walk + turn * 0.4 + (tau - 2 * walk - turn) * 0.4;
  }
  const phase = (dist / 1.1) * Math.PI * 2;
  const swing = Math.sin(phase);
  const bob = 0.025 * Math.abs(Math.cos(phase));
  const root = m4.mul(m4.translation([x, bob, portalB.center[2]]), m4.rotationY(yaw));
  const pivot = (y: number, a: number) => m4.mul(m4.translation([0, y, 0]), m4.rotationX(a), m4.translation([0, -y, 0]));
  const R = meshes.robot;
  const nod = m4.mul(m4.translation([0, 1.1, 0]), m4.rotationY(0.25 * Math.sin(t * 0.9)), m4.translation([0, -1.1, 0]));
  return {
    parts: [
      { range: R.body, local: mat4.identity() },
      { range: R.head, local: nod },
      { range: R.visor, local: nod },
      { range: R.legL, local: pivot(0.58, 0.45 * swing) },
      { range: R.legR, local: pivot(0.58, -0.45 * swing) },
      { range: R.armL, local: pivot(0.98, -0.35 * swing) },
      { range: R.armR, local: pivot(0.98, 0.35 * swing) },
    ],
    root,
    centre: [x, 0.8, portalB.center[2]],
    radius: 0.45,
    occluders: [{ centre: [0, 0.75, 0], radius: 0.32 }, { centre: [0, 1.3, 0], radius: 0.2 }],
  };
}

/** Cube: flies from the bulkhead portal to the north-wall portal of the bay, and round again. */
function cube(t: number, meshes: ObjectMeshes, c0: Portal, c1: Portal): Body {
  const length = c1.center[2] - c0.center[2];
  const s = (((t * 1.1) % length) + length) % length;
  const centre: Vec3 = [c1.center[0], 1.35 + 0.12 * Math.sin(t * 1.3), c1.center[2] - s];
  const root = m4.mul(m4.translation(centre), m4.rotationAxis([1, 0.7, 0.3], t * 0.9));
  return { parts: [{ range: meshes.cube, local: mat4.identity() }], root, centre, radius: 0.45, occluders: [{ centre: [0, 0, 0], radius: 0.3 }] };
}

/** Moves a body through the pair once its centre is behind `portal`. */
function canonical(body: Body, portal: Portal): Body {
  if (planeDist(portal.plane, body.centre) >= 0) return body;
  return { ...body, root: m4.mul(portal.toLinked, body.root), centre: m4.point(portal.toLinked, body.centre) };
}

export function animateObjects(t: number, meshes: ObjectMeshes, portals: Portal[], clipPlanes: boolean) {
  const bodies = [
    canonical(ball(t, meshes, portals[0]), portals[0]),
    canonical(robot(t, meshes, portals[2]), portals[2]),
    cube(t, meshes, portals[4], portals[5]),
  ];
  const draws: ObjectDraw[] = [];
  const occluders: Occluder[] = [];
  let straddling = 0;
  for (const body of bodies) {
    const instances: { root: Mat4; clip: Vec4; crossing: number }[] = [{ root: body.root, clip: NO_CLIP, crossing: 0 }];
    const loc = locationOf(body.centre);
    for (const p of portals) {
      if (p.location !== loc) continue;
      const d = planeDist(p.plane, body.centre);
      if (Math.abs(d) >= body.radius || !insideOpening(m4.point(p.inv, body.centre), -body.radius)) continue;
      const dest = portals[p.link];
      instances[0].clip = p.plane;
      instances[0].crossing = 1;
      instances.push({ root: m4.mul(p.toLinked, body.root), clip: dest.plane, crossing: 2 });
      straddling++;
      break;
    }
    for (const inst of instances) {
      const clip = clipPlanes ? inst.clip : NO_CLIP;
      for (const part of body.parts) draws.push({ range: part.range, model: m4.mul(inst.root, part.local), clip, crossing: inst.crossing });
      for (const o of body.occluders) occluders.push({ centre: m4.point(inst.root, o.centre), radius: o.radius });
    }
  }
  return { draws, occluders, straddling };
}
