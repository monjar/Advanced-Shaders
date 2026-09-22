import type GUI from 'lil-gui';
import type { OrbitCamera } from './camera';

export interface DemoContext {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  format: GPUTextureFormat;
  gui: GUI;
  camera: OrbitCamera;
}

export interface FrameInfo {
  /** Wall-clock seconds since the demo started. */
  time: number;
  /** Seconds since the previous frame, clamped. */
  dt: number;
  width: number;
  height: number;
}

export interface Demo {
  resize(width: number, height: number): void;
  frame(encoder: GPUCommandEncoder, target: GPUTextureView, info: FrameInfo): void;
  /** Extra lines shown in the HUD. */
  hud?(): string;
  destroy(): void;
}

export interface DemoEntry {
  id: string;
  title: string;
  tags: string[];
  /** HTML shown in the sidebar while the demo is active. */
  info: string;
  create(ctx: DemoContext): Demo;
}
