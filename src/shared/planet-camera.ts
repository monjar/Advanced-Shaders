import type { OrbitCamera } from '../core/camera';
import { clamp, mat4, vec3, type Mat4, type Vec3 } from '../core/math';

/**
 * A first-person camera around a planet that works from deep space down to
 * walking height.
 *
 * - The position is kept in double precision (JS numbers), in metres,
 *   relative to the planet centre. Only camera-relative quantities reach the
 *   GPU, so float32 never has to represent a 6,360 km coordinate.
 * - Orientation is surface-relative: a heading tangent to the sphere and a
 *   pitch above the local horizon. The heading is parallel-transported as
 *   the camera moves, so crossing a pole does not flip it.
 * - Speed scales with the height above the ground, so one key press crosses
 *   a continent from orbit and a few metres at walking height; the wheel
 *   zooms that height logarithmically.
 *
 * Input comes from the scene shell's OrbitCamera: its yaw/pitch (mouse drag),
 * distance (wheel) and target (WASD/QE) are read as controls and reset each
 * frame.
 *
 * Planet frame: +Y is the north pole, longitude 0 lies along +Z and east is +X.
 */
export class PlanetCamera {
  radius: number;
  position: Vec3;
  heading: Vec3 = [0, 0, -1];
  /** Radians above the local horizon. */
  pitch = 0;
  fovY = (50 * Math.PI) / 180;
  eyeHeight = 1.7;
  /** Fraction of the height above ground travelled per second. */
  speedFactor = 0.8;
  maxAltitude = 6e7;
  /** Surface height (m above `radius`, water counts as 0) under a unit direction. */
  surfaceHeight: (dir: Vec3) => number = () => 0;

  up: Vec3 = [0, 1, 0];
  forward: Vec3 = [0, 0, -1];
  right: Vec3 = [1, 0, 0];
  altitude = 0;
  heightAboveGround = 0;
  near = 0.1;
  /** Camera-relative matrices (the view has no translation). */
  view: Mat4 = mat4.identity();
  proj: Mat4 = mat4.identity();
  viewProj: Mat4 = mat4.identity();
  invViewProj: Mat4 = mat4.identity();

  private lastYaw: number | null = null;
  private appliedPitch = NaN;

  constructor(radius: number) {
    this.radius = radius;
    this.position = [0, radius + 2, 0];
  }

  /** Configures the shell's orbit camera so it only acts as an input device. */
  bind(orbit: OrbitCamera) {
    Object.assign(orbit, { target: [0, 0, 0], distance: 100, minDistance: 1e-6, moveSpeed: 1, minPitch: -1.55, maxPitch: 1.55 });
    orbit.pitch = -this.pitch;
    this.appliedPitch = this.pitch;
    this.lastYaw = orbit.yaw;
  }

  /** Places the camera at a latitude/longitude (degrees), altitude above sea level (m) and heading from north (degrees). */
  setGeo(latDeg: number, lonDeg: number, altitude: number, headingDeg = 0, pitchDeg = 0) {
    const lat = (latDeg * Math.PI) / 180;
    const lon = (lonDeg * Math.PI) / 180;
    const up: Vec3 = [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];
    const north: Vec3 = [-Math.sin(lat) * Math.sin(lon), Math.cos(lat), -Math.sin(lat) * Math.cos(lon)];
    const east: Vec3 = [Math.cos(lon), 0, -Math.sin(lon)];
    const h = (headingDeg * Math.PI) / 180;
    this.position = vec3.scale(up, this.radius + altitude);
    this.heading = vec3.add(vec3.scale(north, Math.cos(h)), vec3.scale(east, Math.sin(h)));
    this.pitch = (pitchDeg * Math.PI) / 180;
  }

  /** Latitude, longitude (degrees) and altitude above sea level (m). */
  get geo() {
    const [x, y, z] = this.position;
    const r = Math.hypot(x, y, z);
    return { lat: (Math.asin(y / r) * 180) / Math.PI, lon: (Math.atan2(x, z) * 180) / Math.PI, altitude: r - this.radius };
  }

  // Motion deltas arrive already scaled by dt (OrbitCamera.update).
  update(orbit: OrbitCamera, aspect: number) {
    // --- Input from the orbit camera.
    if (this.pitch !== this.appliedPitch) orbit.pitch = -this.pitch; // set from script/preset
    this.pitch = clamp(-orbit.pitch, -1.55, 1.55);
    this.appliedPitch = this.pitch;
    const yaw = orbit.yaw;
    const dYaw = this.lastYaw === null ? 0 : yaw - this.lastYaw;
    this.lastYaw = yaw;
    const zoom = orbit.distance / 100;
    orbit.distance = 100;
    const move = orbit.target;
    orbit.target = [0, 0, 0];
    const fwdW: Vec3 = [-Math.sin(yaw), 0, -Math.cos(yaw)];
    const rightW: Vec3 = [Math.cos(yaw), 0, -Math.sin(yaw)];
    const moveF = vec3.dot(move, fwdW);
    const moveR = vec3.dot(move, rightW);
    const moveU = move[1];

    // --- Orientation: rotate the heading about the local up.
    let up = vec3.normalize(this.position);
    this.heading = rotateAbout(this.heading, up, dYaw);

    // --- Motion.
    let r = vec3.length(this.position);
    let ground = this.radius + this.surfaceHeight(up);
    let hag = Math.max(r - ground, 0.01);
    const speed = this.speedFactor * Math.max(hag, 2);
    const right = vec3.normalize(vec3.cross(this.heading, up));
    if (moveF !== 0 || moveR !== 0) {
      // Move along the surface: step in the tangent plane, then restore the
      // radius so horizontal flight does not climb.
      const step = vec3.add(vec3.scale(this.heading, moveF * speed), vec3.scale(right, moveR * speed));
      this.position = vec3.scale(vec3.normalize(vec3.add(this.position, step)), r);
      up = vec3.normalize(this.position);
      ground = this.radius + this.surfaceHeight(up);
      hag = Math.max(r - ground, 0.01);
    }
    hag = hag * zoom + moveU * speed;
    r = clamp(ground + hag, ground + this.eyeHeight, this.radius + this.maxAltitude);
    this.position = vec3.scale(up, r);

    // Parallel transport: keep the heading tangent to the new position.
    const h = vec3.sub(this.heading, vec3.scale(up, vec3.dot(this.heading, up)));
    this.heading = vec3.length(h) > 1e-9 ? vec3.normalize(h) : vec3.normalize(vec3.cross(up, [1, 0, 0]));

    // --- Derived frame and matrices.
    this.up = up;
    this.altitude = r - this.radius;
    this.heightAboveGround = r - ground;
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    this.forward = vec3.add(vec3.scale(this.heading, cp), vec3.scale(up, sp));
    const camUp = vec3.add(vec3.scale(this.heading, -sp), vec3.scale(up, cp));
    this.right = vec3.normalize(vec3.cross(this.forward, camUp));
    // Reversed-Z with an infinite far plane stores depth as near / z in
    // float32, whose relative precision is the same at every distance, so
    // the near plane only has to stay closer than the nearest geometry.
    this.near = clamp(this.heightAboveGround * 0.02, 0.05, 20000);
    this.view = mat4.lookAt([0, 0, 0], this.forward, camUp);
    this.proj = mat4.perspectiveReversedInfinite(this.fovY, aspect, this.near);
    this.viewProj = mat4.multiply(this.proj, this.view);
    this.invViewProj = mat4.invert(this.viewProj);
  }
}

/** Rodrigues rotation of v about the unit axis k. */
export function rotateAbout(v: Vec3, k: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const kv = vec3.cross(k, v);
  const d = vec3.dot(k, v) * (1 - c);
  return [v[0] * c + kv[0] * s + k[0] * d, v[1] * c + kv[1] * s + k[1] * d, v[2] * c + kv[2] * s + k[2] * d];
}
