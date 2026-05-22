import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';

// The frontend talks to the inference service through /api so the production
// nginx (task #23) can same-origin-proxy it. In dev we replicate that with
// Vite's proxy → http://localhost:8000 (or whatever DEV_API_URL points at).
const DEV_API = process.env.DEV_API_URL ?? 'http://localhost:8000';

export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, '.'),
    },
  },
  server: {
    // HMR is disabled in AI Studio via DISABLE_HMR env var.
    hmr: process.env.DISABLE_HMR !== 'true',
    watch: process.env.DISABLE_HMR === 'true' ? null : {},
    proxy: {
      '/api': {
        target: DEV_API,
        changeOrigin: true,
        rewrite: p => p.replace(/^\/api/, ''),
      },
    },
  },
}));
