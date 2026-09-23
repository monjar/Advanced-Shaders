#!/usr/bin/env node
// Offline 60 fps recorder for the demos.
//
//   npm run dev                               # in another terminal
//   node tools/record.mjs [ocean|clouds|all] [options]
//
// Options:
//   --url <url>        dev/preview server (default http://localhost:5173/)
//   --out <dir>        output directory (default docs/videos)
//   --fps <n>          frame rate (default 60)
//   --size <WxH>       resolution (default 1280x720)
//   --crf <n>          x264 quality, lower is better (default 20)
//   --frames <n>       stop each shot after n frames (quick tests)
//   --swiftshader      force the CPU WebGPU fallback (for machines without a GPU)
//   --headed           show the browser window (some platforms only expose the GPU headed)
//
// Every frame is advanced by exactly 1/fps seconds through the app's capture
// mode (`?capture`), so the result is smooth no matter how slowly the machine
// renders. Frames are piped straight into ffmpeg (set FFMPEG to override the
// binary; it needs libx264).

import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { SHOTS } from './shots.mjs';

const args = process.argv.slice(2);
const flag = (name) => args.includes(`--${name}`);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const target = args.find((a) => !a.startsWith('--') && !args[args.indexOf(a) - 1]?.startsWith('--')) ?? 'all';
const url = option('url', 'http://localhost:5173/');
const outDir = option('out', 'docs/videos');
const fps = Number(option('fps', 60));
const [width, height] = option('size', '1280x720').split('x').map(Number);
const crf = option('crf', '20');
const maxFrames = Number(option('frames', Infinity));
const ffmpegBin = process.env.FFMPEG ?? 'ffmpeg';

const gpuArgs = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'];
const swiftshaderArgs = ['--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'];

async function record(demoId, shots) {
  mkdirSync(outDir, { recursive: true });
  const file = join(outDir, `${demoId}.mp4`);
  const ffmpeg = spawn(ffmpegBin, [
    '-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', crf, '-pix_fmt', 'yuv420p',
    '-r', String(fps), '-movflags', '+faststart', file,
  ], { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((resolve, reject) => {
    ffmpeg.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited with ${code}`))));
  });

  const browser = await chromium.launch({
    headless: !flag('headed'),
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: [...gpuArgs, ...(flag('swiftshader') ? swiftshaderArgs : [])],
  });
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error') console.error(`[page] ${m.text()}`); });
  page.on('pageerror', (e) => console.error(`[page] ${e.message}`));
  await page.addInitScript(() => {
    window.lerp = (a, b, u) => a + (b - a) * u;
    window.ease = (u) => u * u * (3 - 2 * u);
  });

  const base = new URL(url);
  base.searchParams.set('capture', '');
  base.searchParams.set('w', String(width));
  base.searchParams.set('h', String(height));
  base.hash = `#/${demoId}`;
  await page.goto(base.toString());
  await page.waitForFunction(() => window.capture && window.demo, null, { timeout: 60_000 });

  const canvas = page.locator('#gfx');
  const dt = 1 / fps;
  const run = (fn, t = 0, u = 0) => page.evaluate(([code, t, u]) => (0, eval)(code)(window.demo, t, u), [fn.toString(), t, u]);
  const step = () => page.evaluate((dt) => window.capture.step(dt), dt);

  let written = 0;
  const started = Date.now();
  for (const shot of shots) {
    const frames = Math.min(Math.round(shot.seconds * fps), maxFrames);
    await run(shot.setup);
    for (let i = 0; i < (shot.warmup ?? 0); i++) {
      await run(shot.update, 0, 0);
      await step();
    }
    for (let i = 0; i < frames; i++) {
      const t = i / fps;
      await run(shot.update, t, t / shot.seconds);
      await step();
      const png = await canvas.screenshot({ type: 'png' });
      if (!ffmpeg.stdin.write(png)) await new Promise((r) => ffmpeg.stdin.once('drain', r));
      written++;
      if (written % 30 === 0) {
        const perFrame = (Date.now() - started) / written / 1000;
        process.stdout.write(`\r${demoId}: ${shot.name} ${i + 1}/${frames} · ${written} frames · ${perFrame.toFixed(2)} s/frame   `);
      }
    }
  }
  process.stdout.write('\n');
  ffmpeg.stdin.end();
  await done;
  await browser.close();
  console.log(`${file}: ${written} frames, ${(written / fps).toFixed(1)} s at ${fps} fps, ${width}x${height}`);
}

const ids = target === 'all' ? Object.keys(SHOTS) : [target];
for (const id of ids) {
  if (!SHOTS[id]) throw new Error(`No shots defined for "${id}". Known: ${Object.keys(SHOTS).join(', ')}`);
  await record(id, SHOTS[id]);
}
