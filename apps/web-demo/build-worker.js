#!/usr/bin/env node
import esbuild from 'esbuild';
import path from 'path';

async function buildWorker() {
  try {
    const result = await esbuild.build({
      entryPoints: ['./public/workers/speech-worker-wrapped.js'],
      bundle: true,
      format: 'esm',
      platform: 'browser',
      outfile: './public/speech-worker-bundled.js',
      minify: false,
      sourcemap: false,
      external: [],
      loader: {
        '.js': 'js',
      },
      define: {
        'process.env.NODE_ENV': '"production"',
      },
      logLevel: 'info',
    });
    
    console.log('Worker bundled successfully!');
  } catch (error) {
    console.error('Failed to bundle worker:', error);
    process.exit(1);
  }
}

buildWorker();