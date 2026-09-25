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
const picker = document.querySelector<HTMLSelectElement>('#demo-select')!;
const narrow = window.matchMedia('(max-width: 720px)');

const params = new URLSearchParams(location.search);
const renderScale = Math.min(2, Math.max(0.25, Number(params.get('scale')) || 1));
// Offline capture: `?capture&w=1280&h=720` renders a fixed-size canvas with no
// UI and no real-time loop. Frames are advanced explicitly through
// `window.capture.step(dt)` (see tools/record.mjs), so a slow GPU still
// produces a smooth, deterministic 60 fps sequence.
const capture = params.has('capture');
if (capture) {
  const w = Number(params.get('w')) || 1280;
  const h = Number(params.get('h')) || 720;
  document.body.classList.add('capture');
  document.documentElement.style.setProperty('--capture-w', `${w}px`);
  document.documentElement.style.setProperty('--capture-h', `${h}px`);
}

function showError(message: string) {
  errorEl.hidden = false;
  errorEl.replaceChildren();
  const text = document.createElement('p');
  text.textContent = message;
  const more = document.createElement('p');
  const link = document.createElement('a');
  link.href = 'https://github.com/monjar/Advanced-Shaders#readme';
  link.textContent = 'screenshots, videos and write-ups of every study';
  more.append('The repository has ', link, '.');
  errorEl.append(text, more);
}

function buildNav() {
  list.innerHTML = '';
  demos.forEach((d, i) => {
    const a = document.createElement('a');
    a.href = `#/${d.id}`;
    a.dataset.id = d.id;
    a.innerHTML = `<span class="num">${String(i + 1).padStart(2, '0')}</span>${d.title}<span class="tags">${d.tags.join(' · ')}</span>`;
    list.appendChild(a);
    picker.add(new Option(`${String(i + 1).padStart(2, '0')} · ${d.title}`, d.id));
  });
  picker.addEventListener('change', () => (location.hash = `#/${picker.value}`));
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
    const dpr = capture ? 1 : Math.min(window.devicePixelRatio || 1, 2) * renderScale;
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
      current = null;
    }
    errorEl.hidden = true;
    for (const a of list.querySelectorAll('a')) a.classList.toggle('active', a.dataset.id === entry.id);
    picker.value = entry.id;
    info.innerHTML = entry.info;

    const gui = new GUI({ title: entry.title, container: canvas.parentElement! });
    if (narrow.matches) gui.close();
    const camera = new OrbitCamera();
    camera.attach(canvas);
    let demo: Demo;
    try {
      demo = entry.create({ device: gpu.device, canvas, format: gpu.format, gui, camera });
      demo.resize(width, height);
    } catch (e) {
      // Keep the page usable: report the failure and let the user pick another scene.
      gui.destroy();
      camera.detach();
      console.error(e);
      showError(`${entry.title} could not start on this device: ${(e as Error).message}`);
      return;
    }
    current = { entry, demo, gui, camera, started: performance.now() };
    // Handy for poking at a demo from the devtools console.
    (window as unknown as { demo: Demo }).demo = demo;
  };

  const route = () => select(location.hash.replace(/^#\/?/, ''));
  window.addEventListener('hashchange', route);
  route();

  const renderFrame = (time: number, dt: number) => {
    if (!current) return;
    current.camera.update(dt, width / height);
    const encoder = gpu.device.createCommandEncoder();
    current.demo.frame(encoder, gpu.context.getCurrentTexture().createView(), { time, dt, width, height });
    gpu.device.queue.submit([encoder.finish()]);
  };

  if (capture) {
    let time = 0;
    (window as unknown as { capture: unknown }).capture = {
      /** Renders one frame `dt` seconds after the previous one and waits for the GPU. */
      async step(dt: number) {
        time += dt;
        renderFrame(time, dt);
        await gpu.device.queue.onSubmittedWorkDone();
      },
    };
    return;
  }

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

    renderFrame((now - current.started) / 1000, dt);
    const extra = current.demo.hud?.() ?? '';
    hud.textContent = `${fps.toFixed(0)} fps  ${width}x${height}${extra ? '\n' + extra : ''}`;
  };
  requestAnimationFrame(loop);
}

start();
