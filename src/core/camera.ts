import { clamp, mat4, vec3, type Mat4, type Vec3 } from './math';

/**
 * Orbit camera with keyboard fly controls.
 *  - left drag: orbit     - right/middle drag: pan     - wheel: zoom
 *  - WASD: move target    - Q/E: move target down/up
 * Pointer events with Shift held are left to the active demo.
 */
export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  yaw = 0;
  pitch = 0.2;
  distance = 30;
  fovY = (55 * Math.PI) / 180;
  near = 0.1;
  minEyeHeight = -Infinity;

  eye: Vec3 = [0, 0, 0];
  view: Mat4 = mat4.identity();
  proj: Mat4 = mat4.identity();
  viewProj: Mat4 = mat4.identity();
  invViewProj: Mat4 = mat4.identity();

  private keys = new Set<string>();
  private drag: { id: number; button: number; x: number; y: number } | null = null;
  private disposers: (() => void)[] = [];

  attach(el: HTMLElement) {
    const on = <K extends keyof HTMLElementEventMap>(t: K, fn: (e: HTMLElementEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      el.addEventListener(t, fn as EventListener, opts);
      this.disposers.push(() => el.removeEventListener(t, fn as EventListener));
    };
    const onWin = <K extends keyof WindowEventMap>(t: K, fn: (e: WindowEventMap[K]) => void) => {
      window.addEventListener(t, fn as EventListener);
      this.disposers.push(() => window.removeEventListener(t, fn as EventListener));
    };

    on('contextmenu', (e) => e.preventDefault());
    on('pointerdown', (e) => {
      if (e.shiftKey) return;
      el.setPointerCapture(e.pointerId);
      this.drag = { id: e.pointerId, button: e.button, x: e.clientX, y: e.clientY };
    });
    on('pointermove', (e) => {
      if (!this.drag || this.drag.id !== e.pointerId) return;
      const dx = e.clientX - this.drag.x;
      const dy = e.clientY - this.drag.y;
      this.drag.x = e.clientX;
      this.drag.y = e.clientY;
      if (this.drag.button === 0) {
        this.yaw -= dx * 0.005;
        this.pitch = clamp(this.pitch + dy * 0.005, -0.3, 1.5);
      } else {
        const s = this.distance * 0.0015;
        const right: Vec3 = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
        const fwd: Vec3 = [-Math.sin(this.yaw), 0, -Math.cos(this.yaw)];
        this.target = vec3.add(this.target, vec3.add(vec3.scale(right, -dx * s), vec3.scale(fwd, dy * s)));
      }
    });
    const end = (e: PointerEvent) => {
      if (this.drag?.id === e.pointerId) this.drag = null;
    };
    on('pointerup', end);
    on('pointercancel', end);
    on('wheel', (e) => {
      e.preventDefault();
      this.distance = clamp(this.distance * Math.exp(e.deltaY * 0.001), 2, 2000);
    }, { passive: false });
    onWin('keydown', (e) => {
      if ((e.target as HTMLElement)?.closest?.('input, textarea, select')) return;
      this.keys.add(e.key.toLowerCase());
    });
    onWin('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    onWin('blur', () => this.keys.clear());
  }

  detach() {
    this.disposers.forEach((d) => d());
    this.disposers = [];
  }

  get forward(): Vec3 {
    return vec3.normalize(vec3.sub(this.target, this.eye));
  }

  update(dt: number, aspect: number) {
    const speed = (this.keys.has('shift') ? 4 : 1) * Math.max(8, this.distance * 0.6) * dt;
    const right: Vec3 = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];
    const fwd: Vec3 = [-Math.sin(this.yaw), 0, -Math.cos(this.yaw)];
    let move: Vec3 = [0, 0, 0];
    if (this.keys.has('w')) move = vec3.add(move, fwd);
    if (this.keys.has('s')) move = vec3.sub(move, fwd);
    if (this.keys.has('d')) move = vec3.add(move, right);
    if (this.keys.has('a')) move = vec3.sub(move, right);
    if (this.keys.has('e')) move[1] += 1;
    if (this.keys.has('q')) move[1] -= 1;
    this.target = vec3.add(this.target, vec3.scale(move, speed));

    const cp = Math.cos(this.pitch);
    this.eye = vec3.add(this.target, [
      this.distance * cp * Math.sin(this.yaw),
      this.distance * Math.sin(this.pitch),
      this.distance * cp * Math.cos(this.yaw),
    ]);
    if (this.eye[1] < this.minEyeHeight) this.eye[1] = this.minEyeHeight;

    this.view = mat4.lookAt(this.eye, this.target, [0, 1, 0]);
    this.proj = mat4.perspectiveReversedInfinite(this.fovY, aspect, this.near);
    this.viewProj = mat4.multiply(this.proj, this.view);
    this.invViewProj = mat4.invert(this.viewProj);
  }

  /** World-space ray through a point in normalised device coordinates. */
  ray(ndcX: number, ndcY: number): { origin: Vec3; dir: Vec3 } {
    const p = mat4.transformPoint(this.invViewProj, [ndcX, ndcY, 1, 1]);
    const near: Vec3 = [p[0] / p[3], p[1] / p[3], p[2] / p[3]];
    return { origin: this.eye, dir: vec3.normalize(vec3.sub(near, this.eye)) };
  }
}
