// SoundPad2D: 2D parameter space with looped playback + bilinear crossfade.
//
// Pre-renders an N×N grid of AudioBuffers by calling the space's renderFn
// with the parameter values at each grid point. At runtime, four
// BufferSourceNodes (the cursor's 4 nearest neighbours) play in loop,
// and their gains are interpolated bilinearly as the cursor moves.
//
// This gives real-time, glitch-free morphing through the parameter
// space without needing AudioWorklets or live synthesis.

export class SoundPad2D {
  /**
   * @param {AudioContext} ctx
   * @param {object} space — { xParam, yParam, xRange, yRange, xLabel, yLabel,
   *                           fixedParams, gridSize, durationS, renderFn }
   * @param {(progress: number, label: string) => void} [onProgress]
   */
  constructor(ctx, space, onProgress = null) {
    this.ctx = ctx;
    this.space = space;
    this.onProgress = onProgress;
    this.grid = null;          // 2D array of AudioBuffer
    this.spectralFeatures = null; // 2D array of {centroid, rms, lowEnergy, highEnergy}
    this.sourceNodes = [null, null, null, null];
    this.gainNodes = [null, null, null, null];
    this.activeIndices = [-1, -1, -1, -1]; // grid (ix, iy) of current 4 sources
    this.playing = false;
    this.x = 0.5;  // current cursor position in [0,1]
    this.y = 0.5;
    this.masterGain = ctx.createGain();
    this.masterGain.gain.value = 0.7;
  }

  /** Pre-render the full N×N grid. */
  async prerender() {
    const { gridSize, xRange, yRange, xParam, yParam, fixedParams,
            durationS, renderFn } = this.space;
    this.grid = Array.from({ length: gridSize }, () => new Array(gridSize));
    this.spectralFeatures = Array.from({ length: gridSize }, () => new Array(gridSize));
    const total = gridSize * gridSize;
    let done = 0;
    for (let iy = 0; iy < gridSize; iy++) {
      const y = yRange[0] + (yRange[1] - yRange[0]) * iy / Math.max(gridSize - 1, 1);
      for (let ix = 0; ix < gridSize; ix++) {
        const x = xRange[0] + (xRange[1] - xRange[0]) * ix / Math.max(gridSize - 1, 1);
        const params = {
          ...fixedParams,
          [xParam]: x,
          [yParam]: y,
          duration_s: durationS,
          seed: 42 + ix * 7 + iy * 31,
        };
        const buf = await renderFn(this.ctx, params);
        this.grid[iy][ix] = buf;
        this.spectralFeatures[iy][ix] = this._analyse(buf);
        done++;
        if (this.onProgress) this.onProgress(done / total, `${done}/${total}`);
        // Yield to the browser so the progress bar can repaint
        if (done % 4 === 0) await new Promise(r => setTimeout(r, 0));
      }
    }
  }

  /** Compute crude spectral features for the heatmap. */
  _analyse(buf) {
    const data = buf.getChannelData(0);
    const n = data.length;
    // Crude: split into low/mid/high bands via downsampled energy
    let rmsSum = 0;
    let lowEnergy = 0, midEnergy = 0, highEnergy = 0;
    // Use a simple windowed FFT-like proxy: high-frequency content via
    // first-difference of samples; low-frequency via running mean.
    let prev = 0;
    let runMean = 0;
    const a = 0.999;  // running-mean lpf
    for (let i = 0; i < n; i++) {
      const s = data[i];
      rmsSum += s * s;
      const hf = s - prev;
      runMean = a * runMean + (1 - a) * s;
      const lf = runMean;
      highEnergy += hf * hf;
      lowEnergy += lf * lf;
      midEnergy += (s - lf - hf) * (s - lf - hf);
      prev = s;
    }
    const totalE = lowEnergy + midEnergy + highEnergy + 1e-9;
    return {
      rms: Math.sqrt(rmsSum / n),
      lowRatio: lowEnergy / totalE,
      midRatio: midEnergy / totalE,
      highRatio: highEnergy / totalE,
      centroid: (lowEnergy * 0.25 + midEnergy * 0.5 + highEnergy * 0.85) / totalE,
    };
  }

  /** Begin looped playback at the current cursor position. */
  start(destination = null) {
    if (this.playing || !this.grid) return;
    this.playing = true;
    this.masterGain.connect(destination || this.ctx.destination);
    this._refreshSources();
  }

  /** Stop all playback. */
  stop() {
    this.playing = false;
    for (let k = 0; k < 4; k++) {
      if (this.sourceNodes[k]) {
        try { this.sourceNodes[k].stop(); } catch (e) {}
        this.sourceNodes[k].disconnect();
        this.gainNodes[k].disconnect();
        this.sourceNodes[k] = null;
        this.gainNodes[k] = null;
      }
    }
    this.activeIndices = [-1, -1, -1, -1];
    try { this.masterGain.disconnect(); } catch (e) {}
  }

  /** Update cursor position (x, y in [0,1]) and crossfade gains. */
  moveTo(x, y) {
    this.x = Math.max(0, Math.min(1, x));
    this.y = Math.max(0, Math.min(1, y));
    if (!this.playing) return;
    this._refreshSources();
  }

  /** Set master volume 0..1. */
  setVolume(v) {
    if (this.masterGain) this.masterGain.gain.value = Math.max(0, Math.min(1, v));
  }

  /** Map cursor position to the 4 nearest grid cells and update gains. */
  _refreshSources() {
    const g = this.space.gridSize;
    const fx = this.x * (g - 1);
    const fy = this.y * (g - 1);
    const ix0 = Math.max(0, Math.min(g - 2, Math.floor(fx)));
    const iy0 = Math.max(0, Math.min(g - 2, Math.floor(fy)));
    const ix1 = ix0 + 1;
    const iy1 = iy0 + 1;
    const wx = fx - ix0;
    const wy = fy - iy0;
    // 4 corners with bilinear weights
    const corners = [
      { ix: ix0, iy: iy0, w: (1 - wx) * (1 - wy) },
      { ix: ix1, iy: iy0, w: wx * (1 - wy) },
      { ix: ix0, iy: iy1, w: (1 - wx) * wy },
      { ix: ix1, iy: iy1, w: wx * wy },
    ];
    // Compare to active sources: if same cell, just update gain; otherwise replace
    const newIndices = corners.map(c => c.ix * 10000 + c.iy);
    for (let k = 0; k < 4; k++) {
      const target = corners[k];
      const targetIdx = newIndices[k];
      const currentIdx = this.activeIndices[k];
      if (currentIdx !== targetIdx) {
        // Replace this source
        if (this.sourceNodes[k]) {
          // Fade out and stop old
          const oldGain = this.gainNodes[k];
          const oldSrc = this.sourceNodes[k];
          const now = this.ctx.currentTime;
          oldGain.gain.cancelScheduledValues(now);
          oldGain.gain.setValueAtTime(oldGain.gain.value, now);
          oldGain.gain.linearRampToValueAtTime(0, now + 0.05);
          setTimeout(() => {
            try { oldSrc.stop(); } catch (e) {}
            oldSrc.disconnect();
            oldGain.disconnect();
          }, 80);
        }
        // Start new source for this corner
        const src = this.ctx.createBufferSource();
        src.buffer = this.grid[target.iy][target.ix];
        src.loop = true;
        const gain = this.ctx.createGain();
        gain.gain.value = 0;
        src.connect(gain).connect(this.masterGain);
        src.start();
        const now = this.ctx.currentTime;
        gain.gain.linearRampToValueAtTime(target.w, now + 0.05);
        this.sourceNodes[k] = src;
        this.gainNodes[k] = gain;
        this.activeIndices[k] = targetIdx;
      } else {
        // Same cell, just update gain smoothly
        const now = this.ctx.currentTime;
        this.gainNodes[k].gain.cancelScheduledValues(now);
        this.gainNodes[k].gain.setValueAtTime(this.gainNodes[k].gain.value, now);
        this.gainNodes[k].gain.linearRampToValueAtTime(target.w, now + 0.03);
      }
    }
  }

  /** Get the current "mixed" parameter values for display. */
  getCurrentParams() {
    const { xRange, yRange, xParam, yParam } = this.space;
    return {
      [xParam]: xRange[0] + (xRange[1] - xRange[0]) * this.x,
      [yParam]: yRange[0] + (yRange[1] - yRange[0]) * this.y,
    };
  }
}
