/* One Euro filter (Casiez, Roussel & Vogel, CHI 2012).
   Smooths jitter heavily while the gaze is still, and follows quickly when it moves fast,
   so fixations look steady but jumps between lines are not blurred away. */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});

  function smoothingFactor(dt, cutoffHz) {
    const r = 2 * Math.PI * cutoffHz * dt;
    return r / (r + 1);
  }

  class OneEuroFilter {
    constructor(opts = {}) {
      this.minCutoff = opts.minCutoff ?? 1.0; // Hz: lower = smoother when still
      this.beta = opts.beta ?? 2.5; // how fast the cutoff rises with speed
      this.dCutoff = opts.dCutoff ?? 1.0; // Hz: smoothing of the speed estimate
      this.reset();
    }

    reset() {
      this.tPrev = null;
      this.xPrev = 0;
      this.dxPrev = 0;
    }

    filter(x, t) {
      if (this.tPrev === null) {
        this.tPrev = t;
        this.xPrev = x;
        this.dxPrev = 0;
        return x;
      }
      let dt = t - this.tPrev;
      if (!(dt > 0)) dt = 1 / 60;
      if (dt > 0.5) {
        // Long gap (face lost, blink run): start fresh instead of dragging the old value along.
        this.tPrev = t;
        this.xPrev = x;
        this.dxPrev = 0;
        return x;
      }
      const dx = (x - this.xPrev) / dt;
      const aD = smoothingFactor(dt, this.dCutoff);
      const dxHat = aD * dx + (1 - aD) * this.dxPrev;
      const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
      const a = smoothingFactor(dt, cutoff);
      const xHat = a * x + (1 - a) * this.xPrev;
      this.tPrev = t;
      this.xPrev = xHat;
      this.dxPrev = dxHat;
      return xHat;
    }
  }

  GT.OneEuroFilter = OneEuroFilter;
  if (typeof module !== "undefined" && module.exports) module.exports = { OneEuroFilter };
})(typeof window !== "undefined" ? window : globalThis);
