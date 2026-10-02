// The demo is a static site: everything below is about serving files, not
// about running anything. Two things need saying to the dev server.
//
// **Cross-origin isolation.** wllama only gets threads when the page is
// cross-origin isolated (SharedArrayBuffer), which takes these two headers.
// Without them it silently falls back to one thread. The production host has
// to send the same pair -- see README.md.
//
// **The checkpoints.** In development the GGUFs are served straight out of
// `../checkpoints/gguf` rather than copied into `public/`, which would put
// 1.5 GB into every `vite build`. Production points `VITE_MODEL_BASE` at
// wherever the files are hosted instead.

import { createReadStream, statSync } from 'node:fs';
import { resolve, basename } from 'node:path';
import { defineConfig, loadEnv } from 'vite';

const CHECKPOINTS = resolve(__dirname, '../checkpoints/gguf');

const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

function serveCheckpoints() {
  return {
    name: 'serve-checkpoints',
    configureServer(server) {
      server.middlewares.use('/models', (req, res, next) => {
        // basename() so a request cannot walk out of the checkpoint folder.
        const file = resolve(CHECKPOINTS, basename(decodeURIComponent(req.url.split('?')[0])));
        let size;
        try {
          size = statSync(file).size;
        } catch {
          return next();
        }
        // Content-Length is what lets wllama report download progress.
        res.writeHead(200, {
          'Content-Type': 'application/octet-stream',
          'Content-Length': size,
          'Cross-Origin-Resource-Policy': 'same-origin',
          ...isolation,
        });
        createReadStream(file).pipe(res);
      });
    },
  };
}

export default defineConfig(({ command, mode }) => {
  // A production build without a model URL falls back to `/models/`, which
  // only the dev server serves: the page then "downloads" an HTML 404 in an
  // instant and fails. Refuse to build it instead.
  if (command === 'build' && !loadEnv(mode, __dirname).VITE_MODEL_BASE) {
    throw new Error('VITE_MODEL_BASE is not set: see .env.production.');
  }
  return config;
});

const config = {
  plugins: [serveCheckpoints()],
  server: {
    headers: isolation,
    // The explanation templates are read from `../rules`, the same files the
    // Python side renders from, so the two can never disagree.
    fs: { allow: ['..'] },
  },
  preview: { headers: isolation },
  // wllama ships its own worker and wasm; pre-bundling breaks the wasm URL.
  optimizeDeps: { exclude: ['@wllama/wllama'] },
};
