/* Gaze mapping: ridge regression from eye features to screen position.
   The penalty (lambda) is chosen by leave-one-dot-out cross-validation, so the
   reported calibration error is an honest estimate rather than a training score. */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});

  // Solve A·X = B for a symmetric positive-definite A (Cholesky). A is d×d, B is d×m.
  function cholSolve(A, B) {
    const n = A.length;
    const m = B[0].length;
    const L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-12));
        else L[i][j] = s / L[j][j];
      }
    }
    const X = Array.from({ length: n }, () => new Float64Array(m));
    const y = new Float64Array(n);
    for (let c = 0; c < m; c++) {
      for (let i = 0; i < n; i++) {
        let s = B[i][c];
        for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
        y[i] = s / L[i][i];
      }
      for (let i = n - 1; i >= 0; i--) {
        let s = y[i];
        for (let k = i + 1; k < n; k++) s -= L[k][i] * X[k][c];
        X[i][c] = s / L[i][i];
      }
    }
    return X;
  }

  // Features are standardized; the intercept is not penalized (targets are centered).
  function fitRidge(X, Y, lambda) {
    const n = X.length;
    const d = X[0].length;
    const m = Y[0].length;
    const mu = new Array(d).fill(0);
    const sd = new Array(d).fill(0);
    for (const r of X) for (let j = 0; j < d; j++) mu[j] += r[j];
    for (let j = 0; j < d; j++) mu[j] /= n;
    for (const r of X) for (let j = 0; j < d; j++) sd[j] += (r[j] - mu[j]) ** 2;
    for (let j = 0; j < d; j++) {
      sd[j] = Math.sqrt(sd[j] / n);
      if (!(sd[j] > 1e-9)) sd[j] = Infinity; // constant column: contributes nothing
    }
    const yMu = new Array(m).fill(0);
    for (const r of Y) for (let c = 0; c < m; c++) yMu[c] += r[c];
    for (let c = 0; c < m; c++) yMu[c] /= n;

    const A = Array.from({ length: d }, () => new Float64Array(d));
    const B = Array.from({ length: d }, () => new Float64Array(m));
    const z = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) z[j] = (X[i][j] - mu[j]) / sd[j];
      for (let a = 0; a < d; a++) {
        const za = z[a];
        if (!za) continue;
        for (let b = 0; b <= a; b++) A[a][b] += za * z[b];
        for (let c = 0; c < m; c++) B[a][c] += za * (Y[i][c] - yMu[c]);
      }
    }
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < a; b++) A[b][a] = A[a][b];
      A[a][a] += lambda * n + 1e-9;
    }
    const W = cholSolve(A, B);
    return {
      mu,
      sd: sd.map((v) => (Number.isFinite(v) ? v : 0)),
      W: W.map((row) => Array.from(row)),
      yMu,
    };
  }

  function predict(model, x) {
    const { mu, sd, W, yMu } = model;
    const out = yMu.slice();
    for (let j = 0; j < mu.length; j++) {
      if (!sd[j]) continue;
      const z = (x[j] - mu[j]) / sd[j];
      for (let c = 0; c < out.length; c++) out[c] += z * W[j][c];
    }
    // Undo the squeeze toward the middle that regression produces (see fitGaze).
    if (model.post) for (let c = 0; c < out.length; c++) out[c] = model.post[c][0] + model.post[c][1] * out[c];
    return out;
  }

  /**
   * Fit the gaze map.
   * X: feature vectors, Y: [x, y] targets (0–1 of the viewport), groups: dot index per sample.
   * opts.weights converts target units to physical size (e.g. viewport width and height in cm)
   * so x and y errors are compared fairly.
   */
  function fitGaze(X, Y, groups, opts = {}) {
    const lambdas = opts.lambdas || [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
    const w = opts.weights || [1, 1];
    const err = (p, y) => Math.hypot((p[0] - y[0]) * w[0], (p[1] - y[1]) * w[1]);
    const uniq = Array.from(new Set(groups));

    let best = null;
    for (const lam of lambdas) {
      let sum = 0;
      let n = 0;
      for (const g of uniq) {
        const tr = [];
        const te = [];
        for (let i = 0; i < X.length; i++) (groups[i] === g ? te : tr).push(i);
        if (!tr.length || !te.length) continue;
        const model = fitRidge(
          tr.map((i) => X[i]),
          tr.map((i) => Y[i]),
          lam
        );
        for (const i of te) {
          sum += err(predict(model, X[i]), Y[i]);
          n++;
        }
      }
      const e = n ? sum / n : Infinity;
      if (!best || e < best.cvError) best = { lambda: lam, cvError: e };
    }

    // Fit on everything, drop wild samples once (glances away, half-blinks), refit.
    let model = fitRidge(X, Y, best.lambda);
    const res = X.map((x, i) => err(predict(model, x), Y[i]));
    const keep = [];
    for (const g of uniq) {
      const idx = [];
      for (let i = 0; i < X.length; i++) if (groups[i] === g) idx.push(i);
      const med = GT.median(idx.map((i) => res[i]));
      for (const i of idx) if (res[i] <= Math.max(2.5 * med, 1e-9)) keep.push(i);
    }
    if (keep.length >= X.length * 0.6) {
      model = fitRidge(
        keep.map((i) => X[i]),
        keep.map((i) => Y[i]),
        best.lambda
      );
    }

    // Regression pulls predictions toward the middle, so the edges of the screen (where a page
    // starts and ends) come out squeezed inward. Measure the squeeze on held-out dots and
    // stretch it back, per axis, half-way (held-out predictions overstate it a little).
    const held = [];
    for (const g of uniq) {
      const tr = [];
      const te = [];
      for (let i = 0; i < X.length; i++) (groups[i] === g ? te : tr).push(i);
      if (!tr.length || !te.length) continue;
      const m = fitRidge(tr.map((i) => X[i]), tr.map((i) => Y[i]), best.lambda);
      for (const i of te) held.push({ p: predict(m, X[i]), y: Y[i] });
    }
    const post = [];
    for (let c = 0; c < 2; c++) {
      const ps = held.map((h) => h.p[c]);
      const ys = held.map((h) => h.y[c]);
      const pm = GT.mean(ps);
      const ym = GT.mean(ys);
      let sxy = 0;
      let sxx = 0;
      for (let i = 0; i < ps.length; i++) {
        sxy += (ps[i] - pm) * (ys[i] - ym);
        sxx += (ps[i] - pm) ** 2;
      }
      const bFull = sxx > 1e-9 ? sxy / sxx : 1;
      const b = Math.min(1.4, Math.max(0.95, 1 + 0.5 * (bFull - 1)));
      post.push([ym - b * pm, b]); // y = a + b·p, centred so the middle stays put
    }
    model.post = post;
    return { model, lambda: best.lambda, cvError: best.cvError, kept: keep.length, total: X.length, stretch: post.map((q) => q[1]) };
  }

  GT.fitRidge = fitRidge;
  GT.predict = predict;
  GT.fitGaze = fitGaze;
  if (typeof module !== "undefined" && module.exports) module.exports = { fitRidge, predict, fitGaze };
})(typeof window !== "undefined" ? window : globalThis);
