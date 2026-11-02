/* Inputs for the BlazeGaze neural gaze model (WebEyeTrack, Davalos et al. 2025, MIT licence).
   A plain-JavaScript port of the preprocessing in WebEyeTrack's js/src/utils/mathUtils.ts:
   the 512×128 eye image, the head direction vector and the 3D position of the eyes.
   It follows the original step for step (the model was trained on exactly these inputs) but
   avoids per-landmark matrix objects, so it runs in about a millisecond.
   Pure functions: no DOM, so it is tested in Node against the original code. */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});

  const PATCH_W = 512;
  const PATCH_H = 128;
  const CROP = 512;
  const IRIS_CM = 1.2;
  const VFOV = 60;
  const NEAR = 1;
  const FAR = 10000;

  // ---------------------------------------------------------------- small linear algebra
  /** Solve A·x = b (n×n) by Gaussian elimination with partial pivoting. */
  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const d = M[c][c] || 1e-12;
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / d;
        if (!f) continue;
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / (M[r][r] || 1e-12);
    }
    return x;
  }

  /** 3×3 homography mapping 4 src points to 4 dst points (h33 = 1). */
  function homography(src, dst) {
    const A = [];
    const b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
      b.push(u);
      A.push([0, 0, 0, x, y, 1, -x * v, -y * v]);
      b.push(v);
    }
    const h = solve(A, b);
    return [
      [h[0], h[1], h[2]],
      [h[3], h[4], h[5]],
      [h[6], h[7], 1],
    ];
  }
  function applyH(H, x, y) {
    const d = H[2][0] * x + H[2][1] * y + H[2][2];
    return [(H[0][0] * x + H[0][1] * y + H[0][2]) / d, (H[1][0] * x + H[1][1] * y + H[1][2]) / d];
  }

  // ---------------------------------------------------------------- landmarks
  function landmarks2d(lm, W, H) {
    const out = new Array(lm.length);
    for (let i = 0; i < lm.length; i++) out[i] = [Math.floor(lm[i].x * W), Math.floor(lm[i].y * H)];
    return out;
  }

  /** The face width in cm, from the iris (≈1.2 cm across) as a ruler. */
  function faceWidthCm(l2) {
    const d = (a, b) => Math.hypot(l2[a][0] - l2[b][0], l2[a][1] - l2[b][1]);
    const iris = (d(476, 474) + d(471, 469)) / 2;
    const face = d(356, 127);
    return IRIS_CM / (iris / face);
  }

  // ---------------------------------------------------------------- eye patch
  /** Geometry of the eye patch: which frame pixels each patch pixel samples. */
  function eyePatchPlan(l2) {
    const c = l2[4];
    const pad = (p, kx, ky) => [p[0] + (p[0] - c[0]) * kx, p[1] + (p[1] - c[1]) * ky];
    const src = [l2[103], l2[150], l2[379], l2[332]].map((p) => pad(p, 0.4, 0.2));
    const dst = [[0, 0], [0, CROP], [CROP, CROP], [CROP, 0]];
    const Hf = homography(src, dst); // frame → 512×512 face crop
    const Hb = homography(dst, src); // face crop → frame (backward sampling)
    const t = applyH(Hf, l2[151][0], l2[151][1]);
    const b = applyH(Hf, l2[195][0], l2[195][1]);
    const top = Math.round(t[1]);
    const eh = Math.round(b[1] - t[1]);
    // Region of the frame the patch can touch, so only that part needs reading.
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const [px, py] of [[0, top], [CROP, top], [0, top + eh], [CROP, top + eh]]) {
      const [sx, sy] = applyH(Hb, px, py);
      x0 = Math.min(x0, sx);
      y0 = Math.min(y0, sy);
      x1 = Math.max(x1, sx);
      y1 = Math.max(y1, sy);
    }
    return { Hb, top, eh, region: { x: Math.floor(x0) - 2, y: Math.floor(y0) - 2, w: Math.ceil(x1 - x0) + 5, h: Math.ceil(y1 - y0) + 5 } };
  }

  /**
   * Build the 512×128 RGBA eye patch (nearest-neighbour, like the original).
   * img: {data, width, height, x, y}: pixels of a frame region whose top-left is (x, y) in the frame.
   * frameW/frameH: full frame size (pixels outside it stay black, as in the original).
   */
  function eyePatch(plan, img, frameW, frameH, out) {
    const data = out || new Uint8ClampedArray(PATCH_W * PATCH_H * 4);
    data.fill(0);
    const { Hb, top, eh } = plan;
    if (!(eh > 0)) return data;
    const src = img.data;
    const rw = img.width;
    const rh = img.height;
    // The original resizes through an SVD-estimated homography whose rounding noise makes
    // floor() land just below exact integers. The model was deployed with that, so mirror it.
    const EPS = 1e-7;
    for (let y = 0; y < PATCH_H; y++) {
      const ry0 = Math.floor((y * eh) / PATCH_H - EPS); // row inside the eye band
      if (ry0 < 0 || ry0 >= eh) continue;
      const cy = top + ry0; // row in the 512×512 crop
      if (cy < 0 || cy >= CROP) continue;
      for (let x = 0; x < PATCH_W; x++) {
        const cx = Math.floor(x - EPS); // column in the crop
        if (cx < 0) continue;
        const d = Hb[2][0] * cx + Hb[2][1] * cy + Hb[2][2];
        const fx = Math.floor((Hb[0][0] * cx + Hb[0][1] * cy + Hb[0][2]) / d);
        const fy = Math.floor((Hb[1][0] * cx + Hb[1][1] * cy + Hb[1][2]) / d);
        if (fx < 0 || fy < 0 || fx >= frameW || fy >= frameH) continue;
        const rx = fx - img.x;
        const ry = fy - img.y;
        if (rx < 0 || ry < 0 || rx >= rw || ry >= rh) continue;
        const si = (ry * rw + rx) * 4;
        const di = (y * PATCH_W + x) * 4;
        data[di] = src[si];
        data[di + 1] = src[si + 1];
        data[di + 2] = src[si + 2];
        data[di + 3] = src[si + 3];
      }
    }
    return data;
  }

  // ---------------------------------------------------------------- head pose
  /** MediaPipe's 4×4 face transform, read the way WebEyeTrack reads it (data[i*4 + j]). */
  function rot3(m) {
    const d = m.data;
    return [
      [d[0], d[1], d[2]],
      [d[4], d[5], d[6]],
      [d[8], d[9], d[10]],
    ];
  }
  function toEuler(R) {
    return [Math.asin(-R[2][0]), Math.atan2(R[2][1], R[2][2]), Math.atan2(R[1][0], R[0][0])];
  }
  function fromEuler(p, y, r) {
    const cp = Math.cos(p), sp = Math.sin(p);
    const cy = Math.cos(y), sy = Math.sin(y);
    const cr = Math.cos(r), sr = Math.sin(r);
    // R = Rz · Ry · Rx
    return [
      [cr * cy, cr * sy * sp - sr * cp, cr * sy * cp + sr * sp],
      [sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp],
      [-sy, cy * sp, cy * cp],
    ];
  }

  function headVector(m) {
    const [pitch, yaw, roll] = toEuler(rot3(m));
    const hp = -yaw;
    const hy = pitch;
    const x = Math.cos(hp) * Math.sin(hy);
    const y = Math.sin(hp);
    const z = -Math.cos(hp) * Math.cos(hy);
    const c = Math.cos(roll);
    const s = Math.sin(roll);
    return [c * x - s * y, s * x + c * y, z];
  }

  // ---------------------------------------------------------------- 3D face position
  /**
   * Where the eyes are in 3D (cm, camera space): WebEyeTrack's faceReconstruction +
   * computeFaceOrigin3D. lm: normalized landmarks; m: face transform; W, H: frame size.
   */
  function faceOrigin3D(lm, m, W, H, widthCm, z0 = 60) {
    const n = lm.length;
    const aspect = W / H;
    const f = 1 / Math.tan(((VFOV / 2) * Math.PI) / 180);
    const den = 1 / (NEAR - FAR);
    const Pa = f / aspect;
    const Pb = f;
    const Pc = (NEAR + FAR) * den;
    const Pe = 2 * FAR * NEAR * den;
    // WebEyeTrack's convertUvToXyz: inv(P)·[nx, ny, -1, 1] for
    // P = [[Pa,0,0,0],[0,Pb,0,0],[0,0,Pc,-1],[0,0,Pe,0]] gives x = nx/Pa, y = ny/Pb,
    // z = 1/Pe, w = Pc·z + 1; the point is (−x/w, y/w) at the guessed depth.
    const wz = 1 / Pe;
    const ww = Pc * wz + 1;
    // Un-project, centre on the nose (landmark 4), negate x and y, as the original does.
    const rel = new Array(n);
    for (let i = 0; i < n; i++) {
      const nx = 2 * lm[i].x - 1;
      const ny = 1 - 2 * lm[i].y;
      rel[i] = [-(nx / Pa) / ww, (ny / Pb) / ww];
    }
    const nose = rel[4];
    const cen = new Array(n);
    for (let i = 0; i < n; i++) cen[i] = [-(rel[i][0] - nose[0]), -(rel[i][1] - nose[1]), 0];
    const dl = cen[356];
    const dr = cen[127];
    const span = Math.hypot(dl[0] - dr[0], dl[1] - dr[1], dl[2] - dr[2]) || 1e-9;
    const k = widthCm / span;

    const R0 = rot3(m);
    let [pitch, yaw, roll] = toEuler(R0);
    [pitch, yaw] = [-yaw, pitch];
    const R = fromEuler(pitch, yaw, roll);
    const sc = [0, 1, 2].map((j) => Math.hypot(R0[0][j], R0[1][j], R0[2][j]));
    const S = (sc[0] + sc[1] + sc[2]) / 3;

    // canonical = Rᵀ·p, then the initial transform maps it to (R/S)·canonical + t.
    const can = new Array(n);
    for (let i = 0; i < n; i++) {
      const p = [cen[i][0] * k, cen[i][1] * k, cen[i][2] * k];
      can[i] = [
        R[0][0] * p[0] + R[1][0] * p[1] + R[2][0] * p[2],
        R[0][1] * p[0] + R[1][1] * p[1] + R[2][1] * p[2],
        R[0][2] * p[0] + R[1][2] * p[1] + R[2][2] * p[2],
      ];
    }
    const M = R.map((row) => row.map((v) => v / S));
    const fx = W;
    const fy = W;
    const cx = W / 2;
    const cy = H / 2;
    const xf = (p, t) => [
      M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + t[0],
      M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + t[1],
      M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + t[2],
    ];
    const proj = (q) => {
      const z = Math.abs(q[2]) < 1e-6 ? 1e-6 : q[2];
      return [Math.round((fx * q[0] + cx * q[2]) / z), Math.round((fy * q[1] + cy * q[2]) / z)];
    };

    const det = new Array(n);
    for (let i = 0; i < n; i++) det[i] = [lm[i].x * W, lm[i].y * H];
    const p4 = proj(xf(can[4], [0, 0, z0]));
    const s2 = [det[4][0] - p4[0], det[4][1] - p4[1]];
    const first = [s2[0] * (z0 / fx), s2[1] * (z0 / fy), z0];
    const t = first.slice();

    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
      mx += det[i][0] / n;
      my += det[i][1] / n;
    }
    for (let it = 0; it < 10; it++) {
      let total = 0;
      for (let i = 0; i < n; i++) {
        const pp = proj(xf(can[i], t));
        const vx = det[i][0] - pp[0];
        const vy = det[i][1] - pp[1];
        const vn = Math.hypot(vx, vy);
        const dot = vx * (mx - pp[0]) + vy * (my - pp[1]);
        total += dot < 0 ? -vn : vn;
      }
      const delta = Math.max(-5, Math.min(5, 0.1 * (total / n)));
      const nz = t[2] + delta;
      if (Math.abs(nz - t[2]) < 0.25) break;
      t[0] = first[0] * (nz / z0);
      t[1] = first[1] * (nz / z0);
      t[2] = nz;
    }
    const pt = (i) => xf(can[i], t);
    const a = pt(362);
    const b = pt(263);
    const c = pt(33);
    const d = pt(133);
    return [(a[0] + b[0] + c[0] + d[0]) / 4, (a[1] + b[1] + c[1] + d[1]) / 4, (a[2] + b[2] + c[2] + d[2]) / 4];
  }
  GT.blazePrep = {
    PATCH_W, PATCH_H, homography, applyH, landmarks2d, faceWidthCm, eyePatchPlan, eyePatch, headVector, faceOrigin3D,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = GT.blazePrep;
})(typeof window !== "undefined" ? window : globalThis);
