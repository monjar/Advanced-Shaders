import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset paths so the build works from a GitHub Pages sub-path.
  base: './',
  build: {
    target: 'es2022',
    // The bundle is mostly WGSL source embedded as strings (~230 KB gzipped for
    // all studies); one cached download is fine, so don't warn about it.
    chunkSizeWarningLimit: 1000,
  },
});
