/* Turns one MediaPipe Face Landmarker result into the numbers the lab uses:
   iris position inside each eye, eyelid opening, head position and rotation,
   blendshape scores (gaze direction, blinks, facial gestures) and a distance estimate. */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});

  // Indices in MediaPipe's 478-point face mesh. "Right" and "left" are the person's own eyes;
  // in the unmirrored camera image the person's right eye is on the image's left.
  // a = corner on the image's left, b = corner on the image's right.
  const EYES = {
    right: { a: 33, b: 133, top: 159, bottom: 145, iris: 468, ring: [469, 470, 471, 472] },
    left: { a: 362, b: 263, top: 386, bottom: 374, iris: 473, ring: [474, 475, 476, 477] },
  };
  const NOSE_TIP = 1;
  const IRIS_DIAMETER_CM = 1.17; // adult iris is close to 11.7 mm across, which makes it a ruler

  function pt(lm, i, W, H) {
    return [lm[i].x * W, lm[i].y * H];
  }

  function eyeFeatures(lm, e, W, H) {
    const p0 = pt(lm, e.a, W, H);
    const p1 = pt(lm, e.b, W, H);
    const len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]) || 1;
    const ux = (p1[0] - p0[0]) / len;
    const uy = (p1[1] - p0[1]) / len;
    const nx = -uy; // perpendicular, pointing down the image
    const ny = ux;

    const ring = e.ring.map((i) => pt(lm, i, W, H));
    const c0 = pt(lm, e.iris, W, H);
    let cx = 0;
    let cy = 0;
    for (const p of ring) {
      cx += p[0];
      cy += p[1];
    }
    cx = (cx / ring.length + c0[0]) / 2;
    cy = (cy / ring.length + c0[1]) / 2;

    const mx = (p0[0] + p1[0]) / 2;
    const my = (p0[1] + p1[1]) / 2;
    const along = (px, py) => (px - p0[0]) * ux + (py - p0[1]) * uy;
    const across = (px, py) => (px - mx) * nx + (py - my) * ny;

    const top = pt(lm, e.top, W, H);
    const bot = pt(lm, e.bottom, W, H);
    const tN = across(top[0], top[1]);
    const bN = across(bot[0], bot[1]);
    const cN = across(cx, cy);
    const gap = bN - tN;

    const d1 = Math.hypot(ring[0][0] - ring[2][0], ring[0][1] - ring[2][1]);
    const d2 = Math.hypot(ring[1][0] - ring[3][0], ring[1][1] - ring[3][1]);

    return {
      h: along(cx, cy) / len, // 0 = image-left corner, 1 = image-right corner
      v: cN / len, // iris above (−) or below (+) the corner line, in eye widths
      vl: gap > 1e-6 ? (cN - tN) / gap : 0.5, // iris between the lids: 0 top lid, 1 bottom lid
      open: gap / len, // lid opening relative to eye width
      irisPx: Math.max(d1, d2), // the larger diameter is the one eyelids cover least
      widthPx: len,
    };
  }

  // Four rotation terms from the 4×4 face transform. The layout (row- or column-major) is
  // detected from where the translation sits, so this works either way.
  function rotationFeatures(m) {
    if (!m || !m.data || m.data.length < 16) return [0, 0, 0, 0];
    const d = m.data;
    const colMajor = Math.abs(d[14]) >= Math.abs(d[11]);
    const g = (r, c) => (colMajor ? d[c * 4 + r] : d[r * 4 + c]);
    return [g(0, 2), g(1, 2), g(2, 0), g(2, 1)];
  }

  function extractFrame(res, W, H) {
    const fr = { ok: false };
    if (!res || !res.faceLandmarks || !res.faceLandmarks.length) return fr;
    const lm = res.faceLandmarks[0];
    if (!lm || lm.length < 478) return fr;

    const right = eyeFeatures(lm, EYES.right, W, H);
    const left = eyeFeatures(lm, EYES.left, W, H);
    const bs = {};
    const cats = res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories;
    if (cats) {
      for (const c of cats) bs[c.categoryName] = c.score;
      fr.bsNames = cats.map((c) => c.categoryName);
    }
    const m0 = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
    if (m0 && m0.data) fr.mat = m0.data;

    const nose = lm[NOSE_TIP];
    const oR = lm[EYES.right.a];
    const oL = lm[EYES.left.b];
    const scale = Math.hypot((oL.x - oR.x) * W, (oL.y - oR.y) * H) / W;
    const roll = Math.atan2((oL.y - oR.y) * H, (oL.x - oR.x) * W);

    fr.ok = true;
    fr.W = W;
    fr.H = H;
    fr.lm = lm;
    fr.eyes = { right, left };
    fr.bs = bs;
    fr.head = { x: nose.x, y: nose.y, scale, roll };
    fr.rot = rotationFeatures(res.facialTransformationMatrixes && res.facialTransformationMatrixes[0]);
    fr.blink = ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) / 2;
    fr.open = (right.open + left.open) / 2;
    fr.irisPx = (right.irisPx + left.irisPx) / 2;
    return fr;
  }

  // The vector the gaze regression learns from. Bump FEATURE_VERSION when it changes:
  // saved calibrations from an older version no longer fit and must be redone.
  const FEATURE_VERSION = 3;
  function gazeVector(fr) {
    const R = fr.eyes.right;
    const L = fr.eyes.left;
    const g = (k) => fr.bs[k] || 0;
    const h = (R.h + L.h) / 2;
    const v = (R.v + L.v) / 2;
    const vl = (R.vl + L.vl) / 2;
    // Looking toward the person's left: right eye turns in, left eye turns out.
    const lookH = g("eyeLookInRight") + g("eyeLookOutLeft") - (g("eyeLookOutRight") + g("eyeLookInLeft"));
    const lookUp = g("eyeLookUpLeft") + g("eyeLookUpRight");
    const lookDown = g("eyeLookDownLeft") + g("eyeLookDownRight");
    const lookV = lookUp - lookDown;
    return [
      R.h, L.h, R.v, L.v, vl, fr.open,
      h * h, v * v, h * v,
      lookH, lookUp, lookDown, lookV,
      fr.head.x, fr.head.y, fr.head.scale, fr.head.roll,
      fr.rot[0], fr.rot[1], fr.rot[2], fr.rot[3],
      // Looking down closes the lids and partly hides the iris, so the vertical signals
      // bend rather than move in a straight line. These terms let the fit follow the bend.
      fr.open * fr.open, vl * vl, lookV * lookV, v * fr.open,
    ];
  }
  // Neural engine: the BlazeGaze output (fr.nn), mapped to this person's screen with a
  // quadratic, which covers the affine personalisation WebEyeTrack does and a little bend.
  function neuralVector(fr) {
    const [x, y] = fr.nn;
    return [x, y, x * x, y * y, x * y];
  }
  /** The feature vector an engine uses, or null when this frame can't serve it. */
  function engineVector(engine, fr) {
    if (!fr || !fr.ok) return null;
    if (engine === "classic") return gazeVector(fr);
    if (!fr.nn) return null;
    if (engine === "neural") return neuralVector(fr);
    return gazeVector(fr).concat(neuralVector(fr)); // fusion
  }

  // Features that move when the gaze leaves the screen; checked against the calibration range.
  const RANGE_FEATURES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12];

  // Rough camera-to-eye distance from the iris size. Depends on the camera's field of view,
  // which varies by laptop, so treat it as ±20%.
  function estimateDistanceCm(fr, hfovDeg = 65) {
    if (!fr.ok || !(fr.irisPx > 2)) return NaN;
    const f = fr.W / 2 / Math.tan((hfovDeg * Math.PI) / 360);
    return (f * IRIS_DIAMETER_CM) / fr.irisPx;
  }

  function gestureScores(fr) {
    const g = (k) => fr.bs[k] || 0;
    return {
      brow: Math.max(g("browInnerUp"), (g("browOuterUpLeft") + g("browOuterUpRight")) / 2),
      jaw: g("jawOpen"),
      blink: Math.min(g("eyeBlinkLeft"), g("eyeBlinkRight")),
      smile: (g("mouthSmileLeft") + g("mouthSmileRight")) / 2,
    };
  }

  /** Feature ranges seen during calibration (2nd–98th percentile), used to spot off-screen gaze. */
  function featureRanges(X) {
    return RANGE_FEATURES.map((j) => {
      const col = X.map((x) => x[j]);
      return { j, lo: GT.quantile(col, 0.02), hi: GT.quantile(col, 0.98) };
    });
  }
  /** How far outside the calibrated range a feature vector is (0 = inside), in range-widths. */
  function outOfRange(vec, ranges) {
    let worst = 0;
    for (const r of ranges || []) {
      const w = Math.max(r.hi - r.lo, 1e-6);
      const x = vec[r.j];
      const d = x < r.lo ? (r.lo - x) / w : x > r.hi ? (x - r.hi) / w : 0;
      if (d > worst) worst = d;
    }
    return worst;
  }

  GT.EYES = EYES;
  GT.FEATURE_VERSION = FEATURE_VERSION;
  GT.extractFrame = extractFrame;
  GT.gazeVector = gazeVector;
  GT.neuralVector = neuralVector;
  GT.engineVector = engineVector;
  GT.ENGINES = ["classic", "neural", "fusion"];
  GT.featureRanges = featureRanges;
  GT.outOfRange = outOfRange;
  GT.estimateDistanceCm = estimateDistanceCm;
  GT.gestureScores = gestureScores;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { EYES, extractFrame, gazeVector, estimateDistanceCm, gestureScores };
  }
})(typeof window !== "undefined" ? window : globalThis);
