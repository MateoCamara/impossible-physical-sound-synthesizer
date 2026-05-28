// PadCanvas: visual + interaction layer for SoundPad2D.
//
// Draws a heatmap background (one cell per pre-rendered grid point, coloured
// by spectral character: blue=low, cyan=mid, orange=high), grid lines,
// axis labels, and a glowing cursor with a fading trail.
// Mouse/touch events update SoundPad2D's cursor position in real time.

export class PadCanvas {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {SoundPad2D} pad
   */
  constructor(canvas, pad) {
    this.canvas = canvas;
    this.pad = pad;
    this.ctx2d = canvas.getContext("2d");
    this.trail = []; // last N {x, y, t}
    this.trailMax = 30;
    this.cursorActive = false;
    this.dpr = window.devicePixelRatio || 1;
    this._fitCanvas();
    window.addEventListener("resize", () => this._fitCanvas());
    this._bindEvents();
    this._draw = this._draw.bind(this);
    this._rafId = requestAnimationFrame(this._draw);
  }

  /** Resize canvas backing store to match CSS pixels × DPR. */
  _fitCanvas() {
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.cssW = rect.width;
    this.cssH = rect.height;
    this.ctx2d.scale(this.dpr, this.dpr);
  }

  _bindEvents() {
    const onMove = (clientX, clientY, isDown) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = (clientX - rect.left) / rect.width;
      const y = 1 - (clientY - rect.top) / rect.height; // invert Y so up = +
      const cx = Math.max(0, Math.min(1, x));
      const cy = Math.max(0, Math.min(1, y));
      this.pad.moveTo(cx, cy);
      this.trail.push({ x: cx, y: cy, t: performance.now() });
      if (this.trail.length > this.trailMax) this.trail.shift();
      this.cursorActive = isDown;
    };
    this.canvas.addEventListener("mousedown", (e) => {
      this.cursorActive = true;
      onMove(e.clientX, e.clientY, true);
    });
    this.canvas.addEventListener("mousemove", (e) => {
      if (e.buttons === 1) onMove(e.clientX, e.clientY, true);
      else onMove(e.clientX, e.clientY, false);
    });
    this.canvas.addEventListener("mouseup", () => { this.cursorActive = false; });
    this.canvas.addEventListener("mouseleave", () => { this.cursorActive = false; });
    // Touch
    this.canvas.addEventListener("touchstart", (e) => {
      e.preventDefault();
      const t = e.touches[0];
      onMove(t.clientX, t.clientY, true);
    });
    this.canvas.addEventListener("touchmove", (e) => {
      e.preventDefault();
      const t = e.touches[0];
      onMove(t.clientX, t.clientY, true);
    });
    this.canvas.addEventListener("touchend", () => { this.cursorActive = false; });
  }

  /** Heatmap colour from spectral features. */
  _cellColor(feat) {
    if (!feat) return "rgb(20, 25, 40)";
    // Map low/mid/high ratios to RGB-like colour space
    const r = Math.min(255, Math.floor(40 + 215 * feat.highRatio));
    const g = Math.min(255, Math.floor(40 + 180 * feat.midRatio));
    const b = Math.min(255, Math.floor(60 + 195 * feat.lowRatio));
    return `rgb(${r}, ${g}, ${b})`;
  }

  /** Render loop. */
  _draw() {
    const ctx = this.ctx2d;
    const w = this.cssW;
    const h = this.cssH;
    ctx.clearRect(0, 0, w, h);

    // Background heatmap (grid cells)
    const features = this.pad.spectralFeatures;
    if (features) {
      const g = features.length;
      const cellW = w / g;
      const cellH = h / g;
      for (let iy = 0; iy < g; iy++) {
        for (let ix = 0; ix < g; ix++) {
          // iy=0 is bottom (y axis inverted)
          const screenY = h - (iy + 1) * cellH;
          ctx.fillStyle = this._cellColor(features[iy][ix]);
          ctx.fillRect(ix * cellW, screenY, cellW + 1, cellH + 1);
        }
      }
      // Subtle grid lines at cell boundaries
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= g; i++) {
        const px = i * cellW;
        ctx.beginPath();
        ctx.moveTo(px, 0); ctx.lineTo(px, h);
        ctx.stroke();
        const py = i * cellH;
        ctx.beginPath();
        ctx.moveTo(0, py); ctx.lineTo(w, py);
        ctx.stroke();
      }
    } else {
      // Pre-render state: empty dark background
      ctx.fillStyle = "rgb(20, 25, 40)";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
      ctx.font = "16px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Click \"Initialize\" to pre-render the grid", w / 2, h / 2);
      this._rafId = requestAnimationFrame(this._draw);
      return;
    }

    // Cursor trail
    const now = performance.now();
    for (let i = 0; i < this.trail.length; i++) {
      const p = this.trail[i];
      const age = (now - p.t) / 1000;
      if (age > 1.5) continue;
      const alpha = (1 - age / 1.5) * 0.4;
      const radius = 4 + 8 * (i / this.trail.length);
      const sx = p.x * w;
      const sy = (1 - p.y) * h;
      ctx.fillStyle = `rgba(255, 230, 180, ${alpha})`;
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Cursor (glowing)
    const cx = this.pad.x * w;
    const cy = (1 - this.pad.y) * h;
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 25);
    grad.addColorStop(0, "rgba(255, 240, 200, 0.95)");
    grad.addColorStop(0.5, "rgba(255, 180, 80, 0.5)");
    grad.addColorStop(1, "rgba(255, 100, 30, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, 25, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fill();

    // Axis labels (overlay text)
    const space = this.pad.space;
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
    ctx.textAlign = "left";
    // X axis: bottom centre
    ctx.textBaseline = "bottom";
    ctx.fillText(`X: ${space.xLabel || space.xParam}`, 12, h - 8);
    // Y axis: top left rotated... easier: top left
    ctx.textBaseline = "top";
    ctx.fillText(`Y: ${space.yLabel || space.yParam}`, 12, 8);
    // Current params
    const cur = this.pad.getCurrentParams();
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    const xv = cur[space.xParam];
    const yv = cur[space.yParam];
    ctx.fillText(
      `${xv.toFixed(2)}, ${yv.toFixed(2)}`,
      w - 12, h - 8,
    );

    this._rafId = requestAnimationFrame(this._draw);
  }

  destroy() {
    if (this._rafId) cancelAnimationFrame(this._rafId);
  }
}
