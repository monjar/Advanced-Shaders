export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this browser. Use a recent Chrome, Edge or Safari, or Firefox with WebGPU enabled.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');

  const device = await adapter.requestDevice();
  device.lost.then((info) => console.error(`WebGPU device lost (${info.reason}): ${info.message}`));
  device.addEventListener('uncapturederror', (e) => {
    console.error('[WebGPU]', (e as GPUUncapturedErrorEvent).error.message);
  });

  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not create a WebGPU canvas context.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  return { adapter, device, context, format };
}

/** Creates a shader module and forwards compiler diagnostics to the console. */
export function createShader(device: GPUDevice, label: string, code: string): GPUShaderModule {
  const module = device.createShaderModule({ label, code });
  module.getCompilationInfo().then((info) => {
    for (const m of info.messages) {
      const text = `[${label}] ${m.type} at ${m.lineNum}:${m.linePos}: ${m.message}`;
      if (m.type === 'error') console.error(text);
      else if (m.type === 'warning') console.warn(text);
    }
  });
  return module;
}

type Resource = GPUBuffer | GPUTextureView | GPUSampler | GPUBufferBinding;

/**
 * Builds a bind group for an auto-layout pipeline. Resources are bound to
 * consecutive binding indices starting at 0; `null` skips a binding the
 * entry point doesn't use.
 */
export function bindGroup(
  device: GPUDevice,
  pipeline: GPURenderPipeline | GPUComputePipeline,
  group: number,
  resources: (Resource | null)[],
  label?: string,
): GPUBindGroup {
  const entries: GPUBindGroupEntry[] = [];
  resources.forEach((r, binding) => {
    if (r) entries.push({ binding, resource: r instanceof GPUBuffer ? { buffer: r } : r });
  });
  return device.createBindGroup({ label, layout: pipeline.getBindGroupLayout(group), entries });
}

export function uniformBuffer(device: GPUDevice, size: number, label: string): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.ceil(size / 16) * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function vertexBuffer(device: GPUDevice, data: Float32Array<ArrayBuffer>, label: string): GPUBuffer {
  const buffer = device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function indexBuffer(device: GPUDevice, data: Uint16Array | Uint32Array, label: string): GPUBuffer {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, size);
  return buffer;
}

/** A regular grid of (i, j) integer coordinates in [0, n] with triangle indices. */
export function gridMesh(n: number): { vertices: Float32Array<ArrayBuffer>; indices: Uint16Array<ArrayBuffer> | Uint32Array<ArrayBuffer> } {
  const verts = new Float32Array((n + 1) * (n + 1) * 2);
  let v = 0;
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      verts[v++] = i;
      verts[v++] = j;
    }
  }
  const count = n * n * 6;
  const indices = (n + 1) * (n + 1) > 65535 ? new Uint32Array(count) : new Uint16Array(count);
  let k = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      const b = a + 1;
      const c = a + (n + 1);
      const d = c + 1;
      // Alternate the diagonal so the grid has no directional bias.
      if ((i + j) % 2 === 0) {
        indices.set([a, c, b, b, c, d], k);
      } else {
        indices.set([a, c, d, a, d, b], k);
      }
      k += 6;
    }
  }
  return { vertices: verts, indices };
}
