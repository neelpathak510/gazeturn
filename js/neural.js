/* Neural gaze engine: BlazeGaze (WebEyeTrack, MIT) on TensorFlow.js.
   Loads lazily; the rest of the lab works without it. */
(function () {
  "use strict";
  const GT = (window.GT = window.GT || {});

  const TF_URL = "vendor/tfjs/tf.min.js";
  const MODEL_URL = "vendor/blazegaze/model.json";

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (window.tf) return resolve();
      const s = document.createElement("script");
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("couldn't load " + src));
      document.head.append(s);
    });
  }

  const N = {
    state: "idle", // idle | loading | ready | error
    error: "",
    backend: "",
    model: null,
    patch: null, // reusable 512×128 ImageData
    ms: NaN, // smoothed time per prediction
    loading: null,

    load() {
      if (this.loading) return this.loading;
      this.state = "loading";
      this.loading = (async () => {
        try {
          await loadScript(TF_URL);
          const tf = window.tf;
          try {
            await tf.setBackend(window.GT_TF_BACKEND || "webgl"); // GT_TF_BACKEND: test override
          } catch {
            /* falls back to the default backend */
          }
          await tf.ready();
          this.backend = tf.getBackend();
          this.model = await tf.loadLayersModel(MODEL_URL);
          this.patch = new ImageData(GT.blazePrep.PATCH_W, GT.blazePrep.PATCH_H);
          // Warm up so the first real frame isn't slow.
          this.predict(this.patch.data, [0, 0, -1], [0, 0, 60]);
          this.state = "ready";
          window.dispatchEvent(new Event("gt-neural-ready"));
        } catch (e) {
          console.warn("Neural gaze engine unavailable", e);
          this.state = "error";
          this.error = String((e && e.message) || e);
          window.dispatchEvent(new Event("gt-neural-error"));
        }
      })();
      return this.loading;
    },

    /** patchData: RGBA of the 512×128 eye patch; returns the raw gaze point (screen-centred, −0.5…0.5). */
    predict(patchData, head, origin) {
      const tf = window.tf;
      const t0 = performance.now();
      if (patchData !== this.patch.data) this.patch.data.set(patchData);
      const out = tf.tidy(() => {
        const img = tf.browser.fromPixels(this.patch, 3).toFloat().div(255).expandDims(0);
        const h = tf.tensor2d([head], [1, 3]);
        const o = tf.tensor2d([origin], [1, 3]);
        let y = this.model.predict([img, h, o]);
        if (Array.isArray(y)) y = y[0];
        return y;
      });
      const v = out.dataSync();
      out.dispose();
      const dt = performance.now() - t0;
      this.ms = Number.isFinite(this.ms) ? 0.9 * this.ms + 0.1 * dt : dt;
      return [v[0], v[1]];
    },
  };

  GT.neural = N;
})();
