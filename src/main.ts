import GUI from 'lil-gui';
import './style.css';
import { OrbitCamera } from './core/camera';
import type { Demo, DemoEntry } from './core/demo';
import { initWebGPU, type GpuContext } from './core/gpu';
import { demos } from './core/registry';

const canvas = document.querySelector<HTMLCanvasElement>('#gfx')!;
const hud = document.querySelector<HTMLDivElement>('#hud')!;
const errorEl = document.querySelector<HTMLDivElement>('#error')!;
const list = document.querySelector<HTMLElement>('#demo-list')!;
const info = document.querySelector<HTMLElement>('#demo-info')!;

const params = new URLSearchParams(location.search);
const renderScale = Math.min(2, Math.max(0.25, Number(params.get('scale')) || 1));

function showError(message: string) {
  errorEl.hidden = false;
  errorEl.textContent = message;
}

function buildNav() {
  list.innerHTML = '';
  demos.forEach((d, i) => {
    const a = document.createElement('a');
    a.href = `#/${d.id}`;
    a.dataset.id = d.id;
    a.innerHTML = `<span class="num">${String(i + 1).padStart(2, '0')}</span>${d.title}<span class="tags">${d.tags.join(' · ')}</span>`;
    list.appendChild(a);
  });
}

async function start() {
  buildNav();
  let gpu: GpuContext;
  try {
    gpu = await initWebGPU(canvas);
  } catch (e) {
    showError((e as Error).message);
    return;
  }

  let current: { entry: DemoEntry; demo: Demo; gui: GUI; camera: OrbitCamera; started: number } | null = null;
  let width = 0;
  let height = 0;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * renderScale;
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (w === width && h === height) return;
    width = canvas.width = w;
    height = canvas.height = h;
    current?.demo.resize(w, h);
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  const select = (id: string) => {
    const entry = demos.find((d) => d.id === id) ?? demos[0];
    if (current?.entry === entry) return;
    if (current) {
      current.demo.destroy();
      current.gui.destroy();
      current.camera.detach();
    }
    for (const a of list.querySelectorAll('a')) a.classList.toggle('active', a.dataset.id === entry.id);
    info.innerHTML = entry.info;

    const gui = new GUI({ title: entry.title, container: canvas.parentElement! });
    const camera = new OrbitCamera();
    camera.attach(canvas);
    const demo = entry.create({ device: gpu.device, canvas, format: gpu.format, gui, camera });
    demo.resize(width, height);
    current = { entry, demo, gui, camera, started: performance.now() };
    // Handy for poking at a demo from the devtools console.
    (window as unknown as { demo: Demo }).demo = demo;
  };

  const route = () => select(location.hash.replace(/^#\/?/, ''));
  window.addEventListener('hashchange', route);
  route();

  let last = performance.now();
  let fpsTime = 0;
  let fpsFrames = 0;
  let fps = 0;
  const loop = (now: number) => {
    requestAnimationFrame(loop);
    if (!current) return;
    const elapsed = (now - last) / 1000;
    const dt = Math.min(elapsed, 1 / 15);
    last = now;
    fpsTime += elapsed;
    fpsFrames++;
    if (fpsTime > 0.5) {
      fps = fpsFrames / fpsTime;
      fpsTime = 0;
      fpsFrames = 0;
    }

    current.camera.update(dt, width / height);
    const encoder = gpu.device.createCommandEncoder();
    current.demo.frame(encoder, gpu.context.getCurrentTexture().createView(), {
      time: (now - current.started) / 1000,
      dt,
      width,
      height,
    });
    gpu.device.queue.submit([encoder.finish()]);

    const extra = current.demo.hud?.() ?? '';
    hud.textContent = `${fps.toFixed(0)} fps  ${width}x${height}${extra ? '\n' + extra : ''}`;
  };
  requestAnimationFrame(loop);
}

start();
