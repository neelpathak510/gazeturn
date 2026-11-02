/* GazeTurn Lab: UI controller.
   Camera → MediaPipe Face Landmarker → features → calibrated gaze → detector → page turns.
   The maths lives in js/core (pure, tested in Node); this file is the browser glue. */
(function () {
  "use strict";
  const GT = window.GT;

  const MP_VERSION = "1.1.0";
  const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
  const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

  // ---------------------------------------------------------------- utilities
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  // GT._clock lets the automated test drive time faster than real time; unused in normal use.
  const now = () => (GT._clock ? GT._clock() : performance.now() / 1000);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmt = (v, d = 1) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));
  const pct = (v) => (Number.isFinite(v) ? Math.round(v * 100) + "%" : "–");
  const median = GT.median;
  const mean = GT.mean;

  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids) if (k != null) e.append(k);
    return e;
  }

  function deepMerge(base, over) {
    const out = Object.assign({}, base);
    if (!over || typeof over !== "object") return out;
    for (const k of Object.keys(base)) {
      if (!(k in over)) continue;
      const b = base[k];
      if (b && typeof b === "object" && !Array.isArray(b)) out[k] = deepMerge(b, over[k]);
      else if (typeof over[k] === typeof b) out[k] = over[k];
    }
    return out;
  }

  const store = {
    get(k, fallback) {
      try {
        const v = localStorage.getItem("gazeturn." + k);
        return v ? JSON.parse(v) : fallback;
      } catch {
        return fallback;
      }
    },
    set(k, v) {
      try {
        localStorage.setItem("gazeturn." + k, JSON.stringify(v));
        return true;
      } catch {
        return false;
      }
    },
  };

  // Full session data (samples) can be several MB, so it goes to IndexedDB, not localStorage.
  const idb = {
    db: null,
    open() {
      if (this.db) return Promise.resolve(this.db);
      return new Promise((resolve, reject) => {
        try {
          const req = indexedDB.open("gazeturn-lab", 1);
          req.onupgradeneeded = () => req.result.createObjectStore("sessions", { keyPath: "id" });
          req.onsuccess = () => resolve((this.db = req.result));
          req.onerror = () => reject(req.error);
        } catch (e) {
          reject(e);
        }
      });
    },
    async tx(mode, fn) {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        const t = db.transaction("sessions", mode);
        const r = fn(t.objectStore("sessions"));
        t.oncomplete = () => resolve(r && r.result);
        t.onerror = () => reject(t.error);
      });
    },
    put(d) { return this.tx("readwrite", (s) => s.put(d)).catch(() => null); },
    get(id) { return this.tx("readonly", (s) => s.get(id)).catch(() => null); },
    del(id) { return this.tx("readwrite", (s) => s.delete(id)).catch(() => null); },
  };

  // ---------------------------------------------------------------- settings
  const SCREENS = [
    { id: "mba13", label: "MacBook Air 13.6″", diag: 13.6 },
    { id: "mba15", label: "MacBook Air 15.3″", diag: 15.3 },
    { id: "mba133", label: "MacBook Air 13.3″ (2020 and older)", diag: 13.3 },
    { id: "mbp14", label: "MacBook Pro 14.2″", diag: 14.2 },
    { id: "mbp16", label: "MacBook Pro 16.2″", diag: 16.2 },
    { id: "custom", label: "Other: enter the width" },
  ];
  const GESTURES = {
    brow: { label: "Raise eyebrows", prompt: "raise your eyebrows", hold: null },
    jaw: { label: "Open mouth", prompt: "open your mouth", hold: null },
    blink: { label: "Long blink", prompt: "close both eyes for about a second", hold: 0.6, note: "hold about 0.6 s" },
    smile: { label: "Smile", prompt: "smile", hold: null },
  };
  const DEFAULTS = {
    screen: "mba13",
    screenWidthCm: 29.0,
    distanceMode: "auto",
    distanceCm: 50,
    fontSize: 22,
    mode: "shadow",
    trigger: "auto",
    layoutAware: true,
    showHud: true,
    showGaze: true,
    shadowFlash: true,
    paperLight: false,
    look: "keynote",
    method: "hybrid", // hybrid = last word or look up; endzone = last word; lookup = look up
    engine: "auto", // auto = whichever tracker scored best at calibration; classic | neural | fusion
    autoRecenter: true,
    driftGain: 0.5,
    book: "sample",
    gestureNext: "brow",
    gestureBack: "none",
    thresholds: { brow: 0.45, jaw: 0.35, blink: 0.6, smile: 0.5 },
    gestureHold: 0.3,
    dwellTime: 0.9,
    blinkThreshold: 0.6,
    detector: Object.assign({}, GT.DETECTOR_DEFAULTS),
    endzone: Object.assign({ zoneScale: 1 }, GT.ENDZONE_DEFAULTS),
    filter: { minCutoff: 1.0, beta: 2.5 },
    settingsVersion: 3,
  };

  function loadSettings() {
    const saved = store.get("settings", null);
    const s = deepMerge(DEFAULTS, saved);
    if (!saved || saved.settingsVersion !== DEFAULTS.settingsVersion) {
      // New defaults from this version: faint gaze light on, last-word turning.
      s.showGaze = true;
      s.method = "hybrid";
      s.settingsVersion = DEFAULTS.settingsVersion;
    }
    return s;
  }

  // ---------------------------------------------------------------- runtime state
  const S = {
    settings: loadSettings(),
    video: null,
    stream: null,
    landmarker: null,
    lastVideoTime: -1,
    frame: null,
    gaze: null,
    fps: 0,
    fpsCount: 0,
    fpsT0: 0,
    calib: null,
    offset: { x: 0, y: 0 },
    scale: { x: 0, y: 0 }, // stretch around the screen centre, learned while reading
    filt: null,
    listeners: new Set(),
    distHist: [],
    headHist: [],
    light: NaN,
    sessions: store.get("sessions", []),
    full: {},
    selected: null,
    view: "setup",
    stageCancelled: false,
    stageFS: false,
    stageCancel: null,
    stageExit: null,
    book: { fileName: "", fileText: "" },
  };
  function saveSettings() {
    store.set("settings", S.settings);
  }

  // ---------------------------------------------------------------- units
  function screenAspect() {
    const w = screen.width || 1470;
    const h = screen.height || 956;
    return w / h;
  }
  function screenWidthCm() {
    const s = S.settings;
    const p = SCREENS.find((x) => x.id === s.screen);
    if (!p || !p.diag) return s.screenWidthCm;
    const a = screenAspect();
    return (p.diag * 2.54 * a) / Math.hypot(a, 1);
  }
  function cmPerPx() {
    return screenWidthCm() / (screen.width || innerWidth);
  }
  function estDistance() {
    return median(S.distHist);
  }
  function distanceCm() {
    if (S.settings.distanceMode === "manual") return S.settings.distanceCm;
    const d = estDistance();
    return Number.isFinite(d) ? d : S.settings.distanceCm;
  }
  const cmToDeg = (cm, D = distanceCm()) => (Math.atan2(cm, D) * 180) / Math.PI;
  function normToDeg(dx, dy, vw = innerWidth, vh = innerHeight) {
    const c = cmPerPx();
    return cmToDeg(Math.hypot(dx * vw * c, dy * vh * c));
  }

  // ---------------------------------------------------------------- toast
  let toastTimer = 0;
  function toast(msg, ms = 3600) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), ms);
  }

  // ---------------------------------------------------------------- navigation
  const VIEWS = ["intro", "setup", "calibrate", "check", "read", "results", "tuning"];
  const reducedMotion = () => window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  function showView(v) {
    if (!VIEWS.includes(v)) v = "intro";
    const prev = S.view;
    const swap = () => {
      S.view = v;
      document.body.dataset.view = v;
      for (const id of VIEWS) $("#view-" + id).hidden = id !== v;
      for (const b of $$(".step")) {
        if (b.dataset.view === v) b.setAttribute("aria-current", "page");
        else b.removeAttribute("aria-current");
      }
      if (v === "results") renderResults();
      if (v === "tuning") renderTuning();
      if (v === "calibrate" && S.calib) showCalResult();
      updateNeeds();
      window.scrollTo(0, 0);
      // Keynote-style reveal: children rise in one after another.
      const view = $("#view-" + v);
      Array.from(view.children).forEach((c, i) => c.style.setProperty("--i", i));
      view.classList.remove("enter");
      void view.offsetWidth;
      view.classList.add("enter");
    };
    try {
      history.replaceState(null, "", "#" + v);
    } catch {
      /* not important */
    }
    if (prev !== v && document.startViewTransition && !reducedMotion()) document.startViewTransition(swap);
    else swap();
  }

  // ---------------------------------------------------------------- status
  function setState(sel, state, text) {
    const n = $(sel);
    if (!n) return;
    n.dataset.state = state;
    if (text != null) n.textContent = text;
  }
  function movedSinceCalib() {
    if (!S.calib || !S.calib.head || S.headHist.length < 10) return false;
    const hx = median(S.headHist.map((h) => h.x));
    const hy = median(S.headHist.map((h) => h.y));
    const hs = median(S.headHist.map((h) => h.scale));
    const b = S.calib.head;
    return Math.abs(hx - b.x) > 0.06 || Math.abs(hy - b.y) > 0.06 || hs / b.scale < 0.88 || hs / b.scale > 1.13;
  }
  function currentAcc() {
    if (!S.calib) return null;
    return S.calib.latest || S.calib.acc;
  }
  function updatePills() {
    const on = !!S.stream;
    const f = S.frame;
    const face = on && f && f.ok;
    const fs = $("#st-face");
    fs.dataset.state = !on ? "off" : face ? "good" : "bad";
    fs.title = !on ? "Camera off" : face ? `Face found · ${Math.round(S.fps)} fps` : "No face";
    const acc = currentAcc();
    const sc = $("#st-calib");
    if (acc) {
      const a = acc.accDeg;
      const moved = on && movedSinceCalib();
      sc.dataset.state = moved ? "warn" : a <= 2 ? "good" : a <= 3.5 ? "warn" : "bad";
      sc.textContent = moved ? `${fmt(a, 1)}° · moved` : `${fmt(a, 1)}°`;
      sc.title = moved ? "You moved since calibrating. Recalibrate for accuracy." : "Calibration accuracy";
    } else {
      sc.dataset.state = "off";
      sc.textContent = "–";
    }
    $("#cam-wrap").classList.toggle("live", !!face);
    $("#btn-setup-next").disabled = !face && !S.calib;
    $(".step[data-view=setup]").dataset.done = String(on);
    $(".step[data-view=calibrate]").dataset.done = String(!!S.calib);
    $(".step[data-view=results]").dataset.done = String(S.sessions.length > 0);
  }
  function updateNeeds() {
    $("#cal-need").textContent = "";
    $("#check-need").textContent = S.calib ? "" : "Calibrate first.";
    $("#btn-check").disabled = !S.calib;
    $("#btn-cursor").disabled = !S.calib;
    $("#read-need").textContent = S.calib ? "" : "Calibrate first.";
    $("#btn-read").disabled = !S.calib;
  }

  // ---------------------------------------------------------------- camera + model
  function waitForMP(timeoutMs = 45000) {
    return new Promise((resolve, reject) => {
      if (GT.mp) return resolve(GT.mp);
      if (GT.mpError) return reject(new Error(GT.mpError));
      const to = setTimeout(() => reject(new Error("timed out loading MediaPipe")), timeoutMs);
      window.addEventListener("gt-mp-ready", () => {
        clearTimeout(to);
        resolve(GT.mp);
      }, { once: true });
      window.addEventListener("gt-mp-error", () => {
        clearTimeout(to);
        reject(new Error(GT.mpError || "MediaPipe failed to load"));
      }, { once: true });
    });
  }

  async function ensureLandmarker() {
    if (S.landmarker) return;
    const mp = await waitForMP();
    const fileset = await mp.FilesetResolver.forVisionTasks(WASM_URL);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    try {
      S.landmarker = await mp.FaceLandmarker.createFromOptions(fileset, opts("GPU"));
      S.delegate = "GPU";
    } catch (e) {
      console.warn("GPU delegate unavailable, using CPU", e);
      S.landmarker = await mp.FaceLandmarker.createFromOptions(fileset, opts("CPU"));
      S.delegate = "CPU";
    }
  }

  let cameraStarting = null;
  function startCamera() {
    if (S.stream && S.landmarker) return Promise.resolve(true);
    if (!cameraStarting) cameraStarting = doStartCamera().finally(() => (cameraStarting = null));
    return cameraStarting;
  }
  async function doStartCamera() {
    const msg = $("#cam-msg");
    const btn = $("#btn-camera");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      msg.textContent = "No camera access here. Open http://localhost:8765 in Chrome.";
      return false;
    }
    btn.disabled = true;
    if (!S.stream) {
      msg.textContent = "Allow the camera.";
      try {
        S.stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
        });
      } catch (e) {
        btn.disabled = false;
        const name = e && e.name;
        msg.textContent =
          name === "NotAllowedError"
            ? "Camera blocked. Allow it from the camera icon in the address bar, and in System Settings › Privacy & Security › Camera."
            : name === "NotFoundError"
              ? "No camera found."
              : name === "NotReadableError"
                ? "Another app is using the camera. Close it and try again."
                : "Camera didn't start: " + ((e && e.message) || e);
        toast(msg.textContent, 7000);
        return false;
      }
      S.video.srcObject = S.stream;
      try {
        await S.video.play();
      } catch {
        /* autoplay of a muted stream is allowed; ignore */
      }
    }
    msg.textContent = "Loading the face model…";
    try {
      await ensureLandmarker();
    } catch (e) {
      btn.disabled = false;
      msg.textContent = "Face model didn't load. Check your internet and reload. (" + (e.message || e) + ")";
      toast(msg.textContent, 7000);
      return false;
    }
    msg.textContent = "";
    $("#cam-empty").hidden = true;
    startLoop();
    updatePills();
    updateNeeds();
    return true;
  }

  let looping = false;
  function startLoop() {
    if (looping) return;
    looping = true;
    S.fpsT0 = now();
    const v = S.video;
    const step = () => {
      if (v.readyState >= 2 && v.currentTime !== S.lastVideoTime && S.landmarker) {
        S.lastVideoTime = v.currentTime;
        let res = null;
        try {
          res = S.landmarker.detectForVideo(v, performance.now());
        } catch (e) {
          console.warn(e);
        }
        processResult(res, v.videoWidth, v.videoHeight);
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function processResult(res, W, H) {
    const t = now();
    const fr = GT.extractFrame(res, W, H);
    fr.t = t;
    S.fpsCount++;
    if (t - S.fpsT0 >= 1) {
      S.fps = S.fpsCount / (t - S.fpsT0);
      S.fpsCount = 0;
      S.fpsT0 = t;
      updatePills();
    }
    if (fr.ok) {
      const d = GT.estimateDistanceCm(fr);
      if (Number.isFinite(d)) {
        S.distHist.push(d);
        if (S.distHist.length > 90) S.distHist.shift();
      }
      S.headHist.push(fr.head);
      if (S.headHist.length > 30) S.headHist.shift();
    }
    if (fr.ok && wantNeural()) computeNeural(fr, res, W, H);
    S.frame = fr;
    S.gaze = computeGaze(fr, t);
    for (const fn of S.listeners) {
      try {
        fn(fr, S.gaze);
      } catch (e) {
        console.error(e);
      }
    }
  }

  // ---------------------------------------------------------------- neural engine
  const NN = { canvas: null, ctx: null, widthHist: [], widthCm: NaN, z: 60, ms: NaN, frame: 0, last: null, lastT: -1 };
  function wantNeural() {
    if (!GT.neural || GT.neural.state !== "ready") return false;
    if (S.calibrating || S.cursorTest) return true;
    return !!(S.calib && S.calib.engine && S.calib.engine !== "classic");
  }
  /** Run BlazeGaze on this frame: eye patch + head pose → raw gaze point in fr.nn. */
  function computeNeural(fr, res, W, H) {
    // On a slow machine run the model every 2nd or 3rd frame and reuse its last answer briefly,
    // so the rest of the tracker keeps its frame rate.
    const every = !(NN.ms > 30) ? 1 : NN.ms > 60 ? 3 : 2;
    NN.frame++;
    if (every > 1 && NN.frame % every !== 0) {
      if (NN.last && fr.t - NN.lastT < 0.25) fr.nn = NN.last;
      return;
    }
    try {
      const t0 = performance.now();
      const P = GT.blazePrep;
      const mat = res && res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
      if (!mat || !mat.data || mat.data.length < 16) return;
      const l2 = P.landmarks2d(fr.lm, W, H);
      if (NN.widthHist.length < 60) {
        const w = P.faceWidthCm(l2);
        if (Number.isFinite(w) && w > 5 && w < 30) NN.widthHist.push(w);
        NN.widthCm = median(NN.widthHist);
      }
      if (!Number.isFinite(NN.widthCm)) return;
      const plan = P.eyePatchPlan(l2);
      if (!(plan.eh > 0)) return;
      const r = plan.region;
      const x0 = clamp(r.x, 0, W);
      const y0 = clamp(r.y, 0, H);
      const rw = clamp(r.x + r.w, 0, W) - x0;
      const rh = clamp(r.y + r.h, 0, H) - y0;
      if (rw < 8 || rh < 8) return;
      if (!NN.canvas) {
        NN.canvas = document.createElement("canvas");
        NN.ctx = NN.canvas.getContext("2d", { willReadFrequently: true });
      }
      if (NN.canvas.width < rw || NN.canvas.height < rh) {
        NN.canvas.width = Math.max(NN.canvas.width, rw);
        NN.canvas.height = Math.max(NN.canvas.height, rh);
      }
      NN.ctx.drawImage(S.video, x0, y0, rw, rh, 0, 0, rw, rh);
      const img = NN.ctx.getImageData(0, 0, rw, rh);
      const patch = P.eyePatch(plan, { data: img.data, width: rw, height: rh, x: x0, y: y0 }, W, H, GT.neural.patch.data);
      const origin = P.faceOrigin3D(fr.lm, mat, W, H, NN.widthCm, NN.z);
      if (!origin.every(Number.isFinite)) return;
      NN.z = clamp(origin[2], 20, 150);
      const out = GT.neural.predict(patch, P.headVector(mat), origin);
      if (Number.isFinite(out[0]) && Number.isFinite(out[1])) {
        fr.nn = out;
        NN.last = out;
        NN.lastT = fr.t;
      }
      const dt = performance.now() - t0;
      NN.ms = Number.isFinite(NN.ms) ? 0.9 * NN.ms + 0.1 * dt : dt;
    } catch (e) {
      console.warn("neural step failed", e);
    }
  }

  // Which feature vector a stored sample gives for an engine (samples keep vec and nn).
  function sampleVec(engine, smp) {
    if (engine === "classic") return smp.vec;
    if (!smp.nn) return null;
    const nv = GT.neuralVector({ nn: smp.nn });
    return engine === "neural" ? nv : smp.vec.concat(nv);
  }
  function engineModel(engine) {
    const C = S.calib;
    if (!C) return null;
    if (C.models) return C.models[engine] ? C.models[engine].model : null;
    return engine === "classic" ? C.model : null;
  }
  /** Calibrated screen position for a stored sample, with the engine in use (classic as fallback). */
  function predictSample(smp) {
    const eng = (S.calib && S.calib.engine) || "classic";
    const v = sampleVec(eng, smp);
    const m = engineModel(eng);
    if (v && m) return GT.predict(m, v);
    return GT.predict(engineModel("classic"), smp.vec);
  }

  // ---------------------------------------------------------------- gaze
  function resetFilters() {
    S.filt = { x: new GT.OneEuroFilter(S.settings.filter), y: new GT.OneEuroFilter(S.settings.filter) };
  }
  function isBlink(fr) {
    if (!fr.ok) return true;
    if (fr.blink >= S.settings.blinkThreshold) return true;
    if (S.calib && S.calib.baseOpen && fr.open < 0.4 * S.calib.baseOpen) return true;
    return false;
  }
  /** Calibrated prediction → screen, with the running re-centre (offset and stretch). */
  function applyCorrection(p) {
    return [p[0] + S.offset.x + S.scale.x * (p[0] - 0.5), p[1] + S.offset.y + S.scale.y * (p[1] - 0.5)];
  }
  function computeGaze(fr, t) {
    const g = { t, valid: false, off: false, x: NaN, y: NaN, rx: NaN, ry: NaN };
    if (!fr.ok || !S.calib) return g;
    const cvec = GT.gazeVector(fr);
    const eng = S.calib.engine || "classic";
    let vec = eng === "classic" ? cvec : GT.engineVector(eng, fr);
    let model = engineModel(eng);
    if (!vec || !model) {
      vec = cvec; // neural not available for this frame: fall back to the classic model
      model = engineModel("classic");
      g.fallback = true;
    }
    const q = applyCorrection(GT.predict(model, vec));
    g.rx = q[0];
    g.ry = q[1];
    // Off the screen (keyboard, desk, window): the eyes look unlike anything seen while calibrating,
    // or the estimate lands outside the screen. Don't pretend it's on the page.
    const oor = S.calib.ranges ? GT.outOfRange(cvec, S.calib.ranges) : 0;
    g.off = oor > 0.35 || g.rx < -0.08 || g.rx > 1.08 || g.ry < -0.08 || g.ry > 1.08;
    if (isBlink(fr) || g.off) {
      const last = S.gaze;
      if (last) {
        g.x = last.x;
        g.y = last.y;
      }
      return g;
    }
    if (!S.filt) resetFilters();
    g.x = S.filt.x.filter(g.rx, t);
    g.y = S.filt.y.filter(g.ry, t);
    g.valid = true;
    return g;
  }

  // ---------------------------------------------------------------- setup view
  const preview = { ctx: null, small: null, sctx: null, lastDraw: 0, lastChecks: 0, lastLight: 0 };
  function drawPreview(fr) {
    const cv = $("#cam-canvas");
    if (!preview.ctx) preview.ctx = cv.getContext("2d");
    const ctx = preview.ctx;
    const v = S.video;
    const W = cv.width;
    const H = cv.height;
    const vw = v.videoWidth || 16;
    const vh = v.videoHeight || 9;
    const s = Math.max(W / vw, H / vh);
    const dw = vw * s;
    const dh = vh * s;
    const ox = (W - dw) / 2;
    const oy = (H - dh) / 2;
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1); // mirror, like a selfie
    ctx.drawImage(v, ox, oy, dw, dh);
    if (fr.ok) {
      const cs = getComputedStyle(document.documentElement);
      const lime = cs.getPropertyValue("--lime").trim() || "#c8ff3d";
      const P = (i) => [ox + fr.lm[i].x * dw, oy + fr.lm[i].y * dh];
      for (const e of [GT.EYES.right, GT.EYES.left]) {
        ctx.strokeStyle = "rgba(255,255,255,0.55)";
        ctx.lineWidth = 2;
        ctx.shadowBlur = 0;
        const a = P(e.a);
        const b = P(e.b);
        const tp = P(e.top);
        const bt = P(e.bottom);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.quadraticCurveTo(tp[0], tp[1] - 4, b[0], b[1]);
        ctx.quadraticCurveTo(bt[0], bt[1] + 4, a[0], a[1]);
        ctx.stroke();
        const c = P(e.iris);
        const r = Math.hypot(P(e.ring[0])[0] - P(e.ring[2])[0], P(e.ring[0])[1] - P(e.ring[2])[1]) / 2;
        ctx.strokeStyle = lime;
        ctx.shadowColor = lime;
        ctx.shadowBlur = 14;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(c[0], c[1], Math.max(3, r), 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = lime;
        ctx.beginPath();
        ctx.arc(c[0], c[1], 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }
    ctx.restore();
  }

  function measureLight() {
    const v = S.video;
    if (!preview.small) {
      preview.small = document.createElement("canvas");
      preview.small.width = 32;
      preview.small.height = 18;
      preview.sctx = preview.small.getContext("2d", { willReadFrequently: true });
    }
    try {
      preview.sctx.drawImage(v, 0, 0, 32, 18);
      const d = preview.sctx.getImageData(0, 0, 32, 18).data;
      let sum = 0;
      for (let i = 0; i < d.length; i += 4) sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
      S.light = sum / (d.length / 4);
    } catch {
      S.light = NaN;
    }
  }

  function checkRow(sel, state, text) {
    const n = $(sel);
    n.dataset.state = state;
    $(".v", n).textContent = text;
  }
  function updateChecks(fr) {
    if (!fr.ok) {
      checkRow("#chk-face", "bad", "Not found");
      checkRow("#chk-pos", "off", "–");
    } else {
      checkRow("#chk-face", "good", "Found");
      const mx = 1 - fr.head.x; // mirrored, as in the preview
      const y = fr.head.y;
      if (mx < 0.3) checkRow("#chk-pos", "warn", "Move right");
      else if (mx > 0.7) checkRow("#chk-pos", "warn", "Move left");
      else if (y < 0.25) checkRow("#chk-pos", "warn", "Tilt screen up");
      else if (y > 0.72) checkRow("#chk-pos", "warn", "Sit up");
      else checkRow("#chk-pos", "good", "Centred");
    }
    const L = S.light;
    if (!Number.isFinite(L)) checkRow("#chk-light", "off", "–");
    else if (L < 55) checkRow("#chk-light", "bad", "Too dark");
    else if (L < 85) checkRow("#chk-light", "warn", "Dim");
    else if (L > 215) checkRow("#chk-light", "warn", "Backlit?");
    else checkRow("#chk-light", "good", "Good");
    const d = estDistance();
    if (!Number.isFinite(d)) checkRow("#chk-dist", "off", "–");
    else if (d < 35) checkRow("#chk-dist", "warn", `${Math.round(d)} cm · close`);
    else if (d > 80) checkRow("#chk-dist", "warn", `${Math.round(d)} cm · far`);
    else checkRow("#chk-dist", "good", `${Math.round(d)} cm`);
    $("#out-dist-est").textContent = Number.isFinite(d) ? `≈ ${Math.round(d)} cm` : "–";
    const f = S.fps;
    checkRow("#chk-fps", f >= 24 ? "good" : f >= 15 ? "warn" : "bad", `${Math.round(f)} fps`);
  }

  function setupListener(fr) {
    if (S.view !== "setup" || document.hidden || !$("#reader").hidden || !$("#stage").hidden) return;
    const t = fr.t;
    if (t - preview.lastDraw > 1 / 31) {
      preview.lastDraw = t;
      drawPreview(fr);
    }
    if (t - preview.lastLight > 0.5) {
      preview.lastLight = t;
      measureLight();
    }
    if (t - preview.lastChecks > 0.25) {
      preview.lastChecks = t;
      updateChecks(fr);
      updateGestureMeters(fr);
    }
  }

  function renderScreenSettings() {
    const sel = $("#set-screen");
    if (!sel.options.length) for (const s of SCREENS) sel.append(el("option", { value: s.id, text: s.label }));
    sel.value = S.settings.screen;
    $("#screen-custom-wrap").hidden = S.settings.screen !== "custom";
    $("#set-screen-cm").value = S.settings.screenWidthCm;
    $("#out-screen").textContent = `${fmt(screenWidthCm(), 1)} cm`;
    $("#dist-auto").checked = S.settings.distanceMode === "auto";
    $("#dist-manual").checked = S.settings.distanceMode === "manual";
    $("#set-dist-cm").value = S.settings.distanceCm;
  }

  // Face gesture meters
  const gestureRows = {};
  function renderGestures() {
    const host = $("#gestures");
    host.innerHTML = "";
    for (const [name, g] of Object.entries(GESTURES)) {
      const thr = S.settings.thresholds[name];
      const fill = el("div", { class: "meter-fill" });
      const mark = el("div", { class: "meter-thr" });
      mark.style.left = thr * 100 + "%";
      const range = el("input", { type: "range", min: "0.05", max: "0.95", step: "0.01", value: String(thr), "aria-label": g.label + " threshold" });
      const meter = el("div", { class: "meter" }, fill, mark, range);
      const out = el("output", { class: "mono", text: thr.toFixed(2) });
      range.addEventListener("input", () => {
        S.settings.thresholds[name] = +range.value;
        mark.style.left = range.value * 100 + "%";
        out.textContent = (+range.value).toFixed(2);
        saveSettings();
      });
      const btn = el("button", { class: "btn ghost small", text: "Auto-set", onclick: () => autoSetGesture(name) });
      const label = el("div", { class: "gesture-name" }, g.label, g.note ? el("small", { text: g.note }) : null);
      host.append(el("div", { class: "gesture" }, label, meter, out, btn));
      gestureRows[name] = { fill, meter };
    }
  }
  function updateGestureMeters(fr) {
    if (!fr.ok) return;
    const sc = GT.gestureScores(fr);
    for (const [name, row] of Object.entries(gestureRows)) {
      const v = clamp(sc[name] || 0, 0, 1);
      row.fill.style.width = v * 100 + "%";
      row.meter.classList.toggle("on", v >= S.settings.thresholds[name]);
    }
  }
  async function autoSetGesture(name) {
    if (!S.stream) {
      const ok = await startCamera();
      if (!ok) return;
    }
    const msg = $("#gesture-msg");
    const collect = async (ms) => {
      const vals = [];
      const fn = (fr) => {
        if (fr.ok) vals.push(GT.gestureScores(fr)[name]);
      };
      S.listeners.add(fn);
      await sleep(ms);
      S.listeners.delete(fn);
      return vals;
    };
    msg.textContent = "Relax your face…";
    const base = await collect(2000);
    msg.textContent = `Now ${GESTURES[name].prompt}, a few times…`;
    const act = await collect(3500);
    if (base.length < 10 || act.length < 10) {
      msg.textContent = "Couldn't see your face. Try again.";
      return;
    }
    const b = GT.quantile(base, 0.9);
    const pk = GT.quantile(act, 0.97);
    if (pk - b < 0.08) {
      msg.textContent = `No clear change. Try another gesture.`;
      return;
    }
    const thr = Math.round((b + 0.55 * (pk - b)) * 100) / 100;
    S.settings.thresholds[name] = thr;
    saveSettings();
    renderGestures();
    msg.textContent = `${GESTURES[name].label}: set to ${thr.toFixed(2)}.`;
  }

  // ---------------------------------------------------------------- full-screen stage
  class Cancelled extends Error {}
  async function requestFS() {
    try {
      if (!document.fullscreenElement) await document.documentElement.requestFullscreen({ navigationUI: "hide" });
      return true;
    } catch {
      return !!document.fullscreenElement;
    }
  }
  async function exitFS() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
    } catch {
      /* ignore */
    }
  }
  async function openStage(kind, fsPromise) {
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    const st = $("#stage");
    st.innerHTML = "";
    st.dataset.kind = kind;
    st.hidden = false;
    S.stageCancelled = false;
    S.stageCancel = null;
    S.stageExit = null;
    await (fsPromise || requestFS());
    await sleep(450); // let the window settle at its full-screen size
    S.stageFS = !!document.fullscreenElement;
  }
  function closeStage() {
    const st = $("#stage");
    st.hidden = true;
    st.innerHTML = "";
    S.stageCancel = null;
    S.stageExit = null;
    S.stageFS = false;
    exitFS();
  }
  function checkCancel() {
    if (S.stageCancelled) throw new Cancelled();
  }
  function stagePrompt(title, text, buttons) {
    return new Promise((resolve) => {
      const row = el("div", { class: "actions" });
      const card = el("div", { class: "stage-card" }, el("h2", { text: title }), el("p", { text }), row);
      buttons.forEach(([val, label], i) => row.append(el("button", { class: "btn" + (i ? " ghost" : ""), text: label, onclick: () => done(val) })));
      $("#stage").append(card);
      const key = (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          done(buttons[0][0]);
        } else if (e.key === "Escape") done("cancel");
      };
      window.addEventListener("keydown", key);
      S.stageCancel = () => done("cancel");
      function done(v) {
        window.removeEventListener("keydown", key);
        S.stageCancel = null;
        card.remove();
        resolve(v);
      }
    });
  }

  /** Show a shrinking dot and collect eye features while the person looks at it. */
  function collectAt(x, y, o = {}) {
    const dur = o.dur ?? 1.5;
    const skip = o.skip ?? 0.55;
    const host = o.host || $("#stage");
    const dot = el("div", { class: "cal-dot" });
    dot.style.left = x * 100 + "%";
    dot.style.top = y * 100 + "%";
    dot.style.setProperty("--dur", dur + "s");
    host.append(dot);
    return new Promise((resolve) => {
      const t0 = now();
      const out = [];
      const fn = (fr) => {
        const t = fr.t - t0;
        if (t < skip || t > dur || isBlink(fr)) return;
        out.push({ vec: GT.gazeVector(fr), nn: fr.nn ? fr.nn.slice() : null, open: fr.open, head: fr.head, t });
      };
      S.listeners.add(fn);
      setTimeout(() => {
        S.listeners.delete(fn);
        dot.remove();
        resolve(out);
      }, dur * 1000 + 40);
    });
  }

  /** A light glides left to right like a line of text and the eyes follow it (smooth pursuit).
      This gives many more samples, right where reading happens. */
  function collectPursuit(y, o = {}) {
    const x0 = o.x0 ?? 0.14;
    const x1 = o.x1 ?? 0.86;
    const hold = o.hold ?? 0.6;
    const dur = o.dur ?? 2.8;
    const lag = 0.1; // the eyes trail a moving target slightly
    const host = $("#stage");
    const dot = el("div", { class: "cal-dot glide" });
    dot.style.left = x0 * 100 + "%";
    dot.style.top = y * 100 + "%";
    host.append(dot);
    return new Promise((resolve) => {
      const out = [];
      let tm = null;
      setTimeout(() => {
        dot.style.transition = `left ${dur}s linear`;
        dot.style.left = x1 * 100 + "%";
        tm = now();
      }, hold * 1000);
      const fn = (fr) => {
        if (tm === null) return;
        const tt = fr.t - tm - lag;
        if (tt < 0.35 || tt > dur || isBlink(fr)) return;
        const x = x0 + (x1 - x0) * (tt / dur);
        out.push({ vec: GT.gazeVector(fr), nn: fr.nn ? fr.nn.slice() : null, x, y, seg: Math.min(2, Math.floor((tt / dur) * 3)), head: fr.head, open: fr.open });
      };
      S.listeners.add(fn);
      setTimeout(() => {
        S.listeners.delete(fn);
        dot.remove();
        resolve(out);
      }, (hold + dur + 0.15) * 1000);
    });
  }

  // ---------------------------------------------------------------- calibration
  const PURSUIT_ROWS = [0.2, 0.45, 0.7, 0.9];
  const CAL_POINTS = [
    [0.5, 0.5], [0.06, 0.06], [0.5, 0.06], [0.94, 0.06], [0.72, 0.28], [0.28, 0.28], [0.06, 0.5],
    [0.94, 0.5], [0.72, 0.72], [0.28, 0.72], [0.06, 0.94], [0.5, 0.94], [0.94, 0.94],
  ];
  const VAL_POINTS = [[0.5, 0.15], [0.15, 0.5], [0.85, 0.5], [0.5, 0.85], [0.62, 0.4]];

  function accuracyFrom(val, vw, vh) {
    const c = cmPerPx();
    const pts = [];
    const sampleDeg = [];
    for (const v of val) {
      if (v.preds.length < 3) continue;
      const mx = median(v.preds.map((p) => p[0]));
      const my = median(v.preds.map((p) => p[1]));
      const ex = (mx - v.target[0]) * vw * c;
      const ey = (my - v.target[1]) * vh * c;
      const spread = Math.sqrt(mean(v.preds.map((p) => ((p[0] - mx) * vw * c) ** 2 + ((p[1] - my) * vh * c) ** 2)));
      for (const p of v.preds) sampleDeg.push(cmToDeg(Math.hypot((p[0] - v.target[0]) * vw * c, (p[1] - v.target[1]) * vh * c)));
      pts.push({
        target: v.target, mean: [mx, my], preds: v.preds,
        accDeg: cmToDeg(Math.hypot(ex, ey)), xDeg: cmToDeg(Math.abs(ex)), yDeg: cmToDeg(Math.abs(ey)),
        precDeg: cmToDeg(spread), yCm: Math.abs(ey),
      });
    }
    const m = (k) => mean(pts.map((p) => p[k]));
    return {
      accDeg: m("accDeg"), xDeg: m("xDeg"), yDeg: m("yDeg"), precDeg: m("precDeg"), yCm: m("yCm"),
      sampleMedianDeg: median(sampleDeg), points: pts, n: pts.length, at: Date.now(), distanceCm: distanceCm(),
    };
  }

  const ENGINE_LABEL = { classic: "Classic", neural: "Neural", fusion: "Fusion" };

  /** Leave-one-dot-out error (cm on screen) for each calibration dot, with the classic model. */
  function dotErrors(cal, wcm, hcm) {
    const groups = Array.from(new Set(cal.map((c) => c.group)));
    const out = {};
    for (const g of groups) {
      const tr = cal.filter((c) => c.group !== g);
      const te = cal.filter((c) => c.group === g);
      if (tr.length < 20 || !te.length) continue;
      const m = GT.fitRidge(tr.map((c) => c.s.vec), tr.map((c) => c.target), 0.01);
      const errs = te.map((c) => {
        const p = GT.predict(m, c.s.vec);
        return Math.hypot((p[0] - c.target[0]) * wcm, (p[1] - c.target[1]) * hcm);
      });
      out[g] = median(errs);
    }
    return out;
  }

  /** Fit one engine on the calibration samples; null if it can't be served. */
  function fitEngine(eng, cal, wcm, hcm) {
    const rows = cal.filter((c) => sampleVec(eng, c.s));
    if (rows.length < 40 || (eng !== "classic" && rows.length < 0.7 * cal.length)) return null;
    const fit = GT.fitGaze(
      rows.map((c) => sampleVec(eng, c.s)),
      rows.map((c) => c.target),
      rows.map((c) => c.group),
      { weights: [wcm, hcm] }
    );
    return { model: fit.model, lambda: fit.lambda, cvDeg: cmToDeg(fit.cvError), stretch: fit.stretch, n: rows.length };
  }

  function chooseEngine(C, want) {
    const ok = (e) => C.models[e] && C.accs[e] && C.accs[e].n >= 4;
    if (want && want !== "auto" && ok(want)) return want;
    let best = "classic";
    for (const e of GT.ENGINES) if (ok(e) && C.accs[e].accDeg < C.accs[best].accDeg) best = e;
    return best;
  }
  function applyEngine(engine) {
    const C = S.calib;
    C.engine = engine;
    C.model = C.models[engine].model;
    C.acc = C.accs[engine];
    C.latest = null;
    S.offset = { x: 0, y: 0 };
    S.scale = { x: 0, y: 0 };
    resetFilters();
  }

  async function runCalibration() {
    const fsPromise = requestFS(); // must start inside the click
    const ok = await startCamera();
    if (!ok) {
      exitFS();
      return;
    }
    await openStage("calibrate", fsPromise);
    S.calibrating = true;
    try {
      const go = await stagePrompt("Follow the light.", "Keep your head still.", [["start", "Start"], ["cancel", "Cancel"]]);
      if (go !== "start") throw new Cancelled();
      if (GT.neural && GT.neural.state === "loading") await Promise.race([GT.neural.loading, sleep(8000)]);
      await sleep(300);
      const vw = innerWidth;
      const vh = innerHeight;
      const c = cmPerPx();
      const wcm = vw * c;
      const hcm = vh * c;
      const cal = []; // { s: sample, target: [x, y], group }

      // 1) Fixed dots, each checked: a dot with too few samples is shown again.
      const dotSamples = {};
      for (let k = 0; k < CAL_POINTS.length; k++) {
        const [x, y] = CAL_POINTS[k];
        let got = await collectAt(x, y);
        checkCancel();
        if (got.length < 8) got = got.concat(await collectAt(x, y));
        checkCancel();
        dotSamples[k] = got;
      }
      const rebuild = () => {
        cal.length = 0;
        for (const [k, got] of Object.entries(dotSamples)) for (const smp of got) cal.push({ s: smp, target: CAL_POINTS[k], group: +k });
      };
      rebuild();
      if (new Set(cal.map((q) => q.group)).size < 10) {
        throw new Error("Lost your eyes on too many dots. Check the light and try again.");
      }
      // 2) Quality pass: dots that disagree with the rest (a blink, a glance away) are redone once.
      const errs = dotErrors(cal, wcm, hcm);
      const med = median(Object.values(errs));
      const limit = Math.max(2.5 * med, Math.tan((2.5 * Math.PI) / 180) * distanceCm());
      const redo = Object.entries(errs).filter(([, e]) => e > limit).sort((a, b) => b[1] - a[1]).slice(0, 3);
      for (const [k] of redo) {
        const [x, y] = CAL_POINTS[k];
        const got = await collectAt(x, y);
        checkCancel();
        if (got.length >= 8) dotSamples[k] = got;
      }
      if (redo.length) rebuild();

      // 3) Reading rows: the light glides across like lines of text.
      for (let r = 0; r < PURSUIT_ROWS.length; r++) {
        const got = await collectPursuit(PURSUIT_ROWS[r]);
        checkCancel();
        for (const smp of got) cal.push({ s: smp, target: [smp.x, smp.y], group: 100 + r * 3 + smp.seg });
      }

      // 4) Fit every tracker, then score each on five new check dots.
      const models = {};
      for (const eng of GT.ENGINES) {
        const m = fitEngine(eng, cal, wcm, hcm);
        if (m) models[eng] = m;
      }
      if (!models.classic) throw new Error("Not enough data. Try again with more light on your face.");
      const valSamples = [];
      for (const [x, y] of VAL_POINTS) {
        const got = await collectAt(x, y);
        checkCancel();
        valSamples.push({ target: [x, y], got });
      }
      const accs = {};
      for (const eng of Object.keys(models)) {
        const val = valSamples.map((v) => ({
          target: v.target,
          preds: v.got.filter((smp) => sampleVec(eng, smp)).map((smp) => GT.predict(models[eng].model, sampleVec(eng, smp))),
        }));
        const a = accuracyFrom(val, vw, vh);
        if (a.n) accs[eng] = a;
      }
      if (!accs.classic) throw new Error("No usable data. Try again with more light on your face.");

      const all = cal.filter((q) => q.group < 100);
      S.calib = {
        models,
        accs,
        engine: "classic",
        model: models.classic.model,
        acc: accs.classic,
        latest: null,
        at: Date.now(),
        viewport: { w: vw, h: vh },
        screenWidthCm: screenWidthCm(),
        distanceCm: distanceCm(),
        baseOpen: median(all.filter((q) => q.group === 0).map((q) => q.s.open)),
        head: {
          x: median(all.map((q) => q.s.head.x)),
          y: median(all.map((q) => q.s.head.y)),
          scale: median(all.map((q) => q.s.head.scale)),
        },
        samples: cal.length,
        redone: redo.length,
        ranges: GT.featureRanges(cal.map((q) => q.s.vec)),
        neural: GT.neural ? GT.neural.state : "none",
        featVersion: GT.FEATURE_VERSION,
      };
      S.calib.best = chooseEngine(S.calib, "auto");
      applyEngine(chooseEngine(S.calib, S.settings.engine));
      store.set("calib", S.calib);
      S.calibData = cal.map((q) => ({ s: { vec: q.s.vec, nn: q.s.nn }, target: q.target, group: q.group }));
      store.set("calibData", { featVersion: GT.FEATURE_VERSION, data: compactCal(S.calibData) });
      closeStage();
      updatePills();
      updateNeeds();
      showView("calibrate");
      showCalResult();
    } catch (e) {
      closeStage();
      if (e instanceof Cancelled) toast("Cancelled");
      else toast(e.message || String(e), 7000);
    } finally {
      S.calibrating = false;
    }
  }

  function tile(k, v, s, state) {
    return el("div", { class: "tile", "data-state": state || null }, el("span", { class: "tile-k", text: k }), el("span", { class: "tile-v", text: v }), s ? el("span", { class: "tile-s", text: s }) : null);
  }
  const accState = (d) => (d <= 2 ? "good" : d <= 3.5 ? "warn" : "bad");

  function showCalResult() {
    const C = S.calib;
    if (!C) return;
    const A = C.latest || C.acc;
    $("#cal-result").hidden = false;
    $("#btn-calibrate").textContent = "Recalibrate";
    $("#btn-calibrate").classList.add("ghost");
    const big = $("#cal-big");
    big.textContent = `${fmt(A.accDeg, 1)}°`;
    big.dataset.state = accState(A.accDeg);
    const lineCm = S.settings.fontSize * 1.55 * cmPerPx();
    const sub = $("#cal-sub");
    sub.innerHTML = "";
    const part = (v, label) => [el("b", { text: v }), " " + label];
    sub.append(
      ...part(`${fmt(A.xDeg, 1)}°`, "across"), "   ·   ",
      ...part(`${fmt(A.yDeg, 1)}°`, `up–down (≈ ${fmt(A.yCm / lineCm, 1)} lines)`), "   ·   ",
      ...part(`${fmt(A.precDeg, 1)}°`, "jitter")
    );
    $("#cal-note").textContent =
      A.accDeg > 3.5 ? "Too rough. Better light, then redo." : A.accDeg > 2 ? "Normal for a webcam." : "Excellent for a webcam.";
    renderEngines();
    drawCalPlot(A, C.viewport);
  }

  /** One row per tracker: its accuracy on the same check dots; the one in use is lit. */
  function renderEngines() {
    const host = $("#cal-engines");
    const C = S.calib;
    host.innerHTML = "";
    if (!C || !C.accs) return;
    for (const e of GT.ENGINES) {
      const a = C.accs[e];
      const b = el(
        "button",
        { class: "eng", "aria-pressed": String(C.engine === e), disabled: a ? null : true, title: a ? `Use ${ENGINE_LABEL[e]}` : "Not available" },
        el("span", { class: "eng-name", text: ENGINE_LABEL[e] + (C.best === e ? " · best" : "") }),
        el("span", { class: "eng-val", text: a ? `${fmt(a.accDeg, 1)}°` : "–" })
      );
      if (a) b.addEventListener("click", () => setEngine(e));
      host.append(b);
    }
    if (!C.accs.neural && C.neural !== "ready") host.append(el("span", { class: "hint", text: "Neural model didn't load" }));
  }
  function setEngine(e) {
    if (!S.calib || !S.calib.models) return;
    S.settings.engine = e;
    saveSettings();
    applyEngine(chooseEngine(S.calib, e));
    store.set("calib", S.calib);
    updatePills();
    showCalResult();
    renderEngineSetting();
  }
  function renderEngineSetting() {
    for (const r of $$("input[name=engine]")) r.checked = r.value === S.settings.engine;
    const st = $("#nn-status");
    if (!st) return;
    const N = GT.neural;
    const ms = Number.isFinite(NN.ms) ? ` · ${NN.ms.toFixed(0)} ms per frame` : "";
    st.textContent = !N ? "" : N.state === "ready" ? `Neural model ready (${N.backend})${ms}` : N.state === "loading" ? "Loading neural model…" : N.state === "error" ? "Neural model unavailable: " + N.error : "";
  }

  function drawCalPlot(A, vp) {
    const cv = $("#cal-plot");
    const cs = getComputedStyle(document.documentElement);
    const col = (n) => cs.getPropertyValue(n).trim();
    const W = cv.width;
    const H = Math.round((W * vp.h) / vp.w);
    cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = col("--bg-2");
    ctx.fillRect(0, 0, W, H);
    const lime = col("--lime");
    for (const p of A.points) {
      const tx = p.target[0] * W;
      const ty = p.target[1] * H;
      ctx.fillStyle = lime;
      ctx.globalAlpha = 0.16;
      for (const q of p.preds) {
        ctx.beginPath();
        ctx.arc(q[0] * W, q[1] * H, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(p.mean[0] * W, p.mean[1] * H);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(tx, ty, 16, 0, Math.PI * 2);
      ctx.stroke();
      ctx.shadowColor = lime;
      ctx.shadowBlur = 22;
      ctx.fillStyle = lime;
      ctx.beginPath();
      ctx.arc(p.mean[0] * W, p.mean[1] * H, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = col("--ink-3");
      ctx.font = `22px ${cs.getPropertyValue("--f-mono")}`;
      ctx.fillText(`${p.accDeg.toFixed(1)}°`, tx + 24, ty - 16);
    }
  }

  // ---------------------------------------------------------------- re-centre and re-check
  async function recenter(host) {
    if (!S.calib) return false;
    const cross = el("div", { class: "chk-cross" });
    host.append(cross);
    const got = await collectAt(0.5, 0.5, { host, dur: 1.4, skip: 0.5 });
    cross.remove();
    if (got.length < 6) {
      toast("Couldn't see your eyes. Try again.");
      return false;
    }
    const preds = got.map((s) => predictSample(s));
    S.offset = { x: 0.5 - median(preds.map((p) => p[0])), y: 0.5 - median(preds.map((p) => p[1])) };
    S.scale = { x: 0, y: 0 };
    resetFilters();
    toast(`Re-centred · ${fmt(normToDeg(S.offset.x, S.offset.y), 1)}° shift`);
    return true;
  }
  async function accuracyTest(host) {
    const val = [];
    for (const [x, y] of VAL_POINTS) {
      const got = await collectAt(x, y, { host });
      val.push({ target: [x, y], preds: got.map((s) => applyCorrection(predictSample(s))) });
    }
    const acc = accuracyFrom(val, innerWidth, innerHeight);
    if (!acc.n) {
      toast("Not enough data. Try again.");
      return;
    }
    S.calib.latest = acc;
    store.set("calib", S.calib);
    updatePills();
    toast(`${fmt(acc.accDeg, 1)}° · across ${fmt(acc.xDeg, 1)}° · up–down ${fmt(acc.yDeg, 1)}°`, 6000);
  }

  // ---------------------------------------------------------------- gaze check
  function place(node, x, y) {
    node.style.transform = `translate(${x * innerWidth}px, ${y * innerHeight}px)`;
  }
  async function runCheck() {
    if (!S.calib) {
      toast("Calibrate first.");
      showView("calibrate");
      return;
    }
    const fsPromise = requestFS();
    const ok = await startCamera();
    if (!ok) {
      exitFS();
      return;
    }
    await openStage("check", fsPromise);
    const st = $("#stage");
    for (const y of [0.1, 0.5, 0.9]) {
      for (const x of [0.1, 0.5, 0.9]) {
        const t = el("div", { class: "chk-target" });
        t.style.left = x * 100 + "%";
        t.style.top = y * 100 + "%";
        st.append(t);
      }
    }
    const raw = el("div", { class: "gaze-raw" });
    const dot = el("div", { class: "gaze-dot" });
    const read = el("span", { class: "stage-read mono", text: "" });
    let busy = false;
    const doRecenter = async () => {
      if (busy) return;
      busy = true;
      dot.hidden = raw.hidden = true;
      await recenter(st);
      dot.hidden = raw.hidden = false;
      busy = false;
    };
    const doTest = async () => {
      if (busy) return;
      busy = true;
      dot.hidden = raw.hidden = true;
      await accuracyTest(st);
      dot.hidden = raw.hidden = false;
      busy = false;
    };
    const bar = el(
      "div",
      { class: "stage-bar" },
      read,
      el("button", { class: "btn ghost small", text: "Re-centre", onclick: doRecenter }),
      el("button", { class: "btn ghost small", text: "Test", onclick: doTest }),
      el("button", { class: "btn small", text: "Done", onclick: () => close() })
    );
    st.append(raw, dot, bar);
    let lastRead = 0;
    const fn = (fr, g) => {
      if (!Number.isFinite(g.x)) {
        dot.classList.add("lost");
        return;
      }
      dot.classList.toggle("lost", !g.valid);
      dot.classList.toggle("off", !!g.off);
      place(dot, g.x, g.y);
      if (Number.isFinite(g.rx)) place(raw, g.rx, g.ry);
      if (fr.t - lastRead > 0.25) {
        lastRead = fr.t;
        let best = Infinity;
        for (const ty of [0.1, 0.5, 0.9]) for (const tx of [0.1, 0.5, 0.9]) best = Math.min(best, normToDeg(g.x - tx, g.y - ty));
        read.textContent = g.off ? "off screen" : g.valid ? (best < 4 ? `${best.toFixed(1)}° off` : "·") : "no eyes";
      }
    };
    S.listeners.add(fn);
    const key = (e) => {
      const k = e.key.toLowerCase();
      if (k === "c") doRecenter();
      else if (k === "t") doTest();
      else if (k === "escape") close();
    };
    window.addEventListener("keydown", key);
    function close() {
      S.listeners.delete(fn);
      window.removeEventListener("keydown", key);
      closeStage();
      updatePills();
      if (S.calib && S.calib.latest) showCalResult();
    }
    S.stageExit = close;
  }

  // ---------------------------------------------------------------- cursor test
  // The mouse cursor is the ground truth: point where you look, and the lab measures how far
  // each tracker lands from it, all over the screen. Samples are recorded only while the cursor
  // rests (people look at a cursor they've just placed) or while Space is held.
  const CT_COLS = 4;
  const CT_ROWS = 3;
  const CT_COLOR = { classic: "#f5f5f7", neural: "#5ee7ff", fusion: "#c8ff3d" };
  const ctCell = (c) => clamp(Math.floor(c[1] * CT_ROWS), 0, CT_ROWS - 1) * CT_COLS + clamp(Math.floor(c[0] * CT_COLS), 0, CT_COLS - 1);
  /** Signed offset gaze − cursor in degrees, per axis. */
  function degXY(dx, dy) {
    const c = cmPerPx();
    const sx = Math.sign(dx) * cmToDeg(Math.abs(dx) * innerWidth * c);
    const sy = Math.sign(dy) * cmToDeg(Math.abs(dy) * innerHeight * c);
    return [sx, sy];
  }
  function biasText(bx, by) {
    const part = (v, pos, neg) => (Math.abs(v) < 0.15 ? "" : `${Math.abs(v).toFixed(1)}° ${v > 0 ? pos : neg}`);
    const t = [part(by, "below", "above"), part(bx, "right", "left")].filter(Boolean).join(", ");
    return t || "centred";
  }

  async function runCursorTest() {
    if (!S.calib) {
      toast("Calibrate first.");
      showView("calibrate");
      return;
    }
    const fsPromise = requestFS();
    const ok = await startCamera();
    if (!ok) {
      exitFS();
      return;
    }
    await openStage("cursor", fsPromise);
    S.cursorTest = true;
    const C = S.calib;
    const st = $("#stage");
    const T = { samples: [], cursor: null, lastMove: -1e9, overUI: false, space: false, compare: true, map: true, filters: {}, t0: now(), busy: false };
    const engines = () => GT.ENGINES.filter((e) => engineModel(e));

    const heat = el("canvas", { class: "ct-heat", "aria-hidden": "true" });
    const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgEl.setAttribute("class", "ct-svg");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    svgEl.append(line);
    const rings = {};
    for (const e of GT.ENGINES) {
      rings[e] = el("div", { class: "ct-ring", title: ENGINE_LABEL[e] });
      rings[e].style.borderColor = CT_COLOR[e];
      rings[e].hidden = true;
    }
    const dot = el("div", { class: "gaze-dot" });
    const rec = el("div", { class: "ct-rec" });
    const nowEl = el("span", { class: "ct-now", text: "–" });
    const medEl = el("span", { class: "ct-stat", text: "" });
    const biasEl = el("span", { class: "ct-stat", text: "" });
    const cntEl = el("span", { class: "ct-stat", text: "" });
    const engEl = el("div", { class: "ct-engines" });
    const hud = el(
      "div",
      { class: "ct-hud" },
      el("div", { class: "ct-row" }, nowEl, el("span", { class: "ct-unit", text: "off now" })),
      el("div", { class: "ct-row small" }, medEl, biasEl, cntEl),
      engEl
    );
    const intro = el(
      "div",
      { class: "ct-intro" },
      el("h2", { text: "Point where you look." }),
      el("p", { text: "Rest the cursor on something, look at its tip, and hold your gaze. Move around the whole screen." })
    );
    const btn = (text, fn, cls = "btn ghost small") => el("button", { class: cls, text, onclick: fn });
    const bCompare = btn("Compare", () => toggle("compare"));
    const bMap = btn("Map", () => toggle("map"));
    const bLearn = btn("Learn", () => learn(), "btn small");
    const bar = el(
      "div",
      { class: "stage-bar ct-bar" },
      bCompare,
      bMap,
      btn("Clear", () => clearSamples()),
      btn("Export", () => exportTest()),
      bLearn,
      btn("Done", () => close())
    );
    st.append(heat, svgEl, ...Object.values(rings), dot, rec, hud, intro, bar);
    const syncButtons = () => {
      bCompare.setAttribute("aria-pressed", String(T.compare));
      bMap.setAttribute("aria-pressed", String(T.map));
    };
    syncButtons();

    function sizeHeat() {
      const dpr = window.devicePixelRatio || 1;
      heat.width = Math.round(innerWidth * dpr);
      heat.height = Math.round(innerHeight * dpr);
    }
    sizeHeat();

    const onMove = (e) => {
      const c = [e.clientX / innerWidth, e.clientY / innerHeight];
      if (!T.cursor || Math.hypot((c[0] - T.cursor[0]) * innerWidth, (c[1] - T.cursor[1]) * innerHeight) > 3) T.lastMove = now();
      T.cursor = c;
      T.overUI = !!(e.target && e.target.closest && e.target.closest(".stage-bar, .ct-panel"));
      rec.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`;
    };
    st.addEventListener("pointermove", onMove);

    function toggle(k) {
      T[k] = !T[k];
      syncButtons();
      if (k === "map") drawHeat();
      if (k === "compare") for (const e of GT.ENGINES) rings[e].hidden = !T.compare || e === C.engine || !engineModel(e);
    }
    for (const e of GT.ENGINES) rings[e].hidden = !T.compare || e === C.engine || !engineModel(e);

    // Per-tracker predictions, smoothed the same way as the live light.
    function predsFor(fr) {
      const out = {};
      for (const e of engines()) {
        const v = GT.engineVector(e, fr);
        if (!v) continue;
        let p = GT.predict(engineModel(e), v);
        if (e === C.engine) p = applyCorrection(p);
        const f = T.filters[e] || (T.filters[e] = { x: new GT.OneEuroFilter(S.settings.filter), y: new GT.OneEuroFilter(S.settings.filter) });
        out[e] = [f.x.filter(p[0], fr.t), f.y.filter(p[1], fr.t)];
      }
      return out;
    }

    let lastHud = 0;
    let lastHeat = 0;
    const fn = (fr, g) => {
      if (T.busy) return;
      const preds = fr.ok ? predsFor(fr) : {};
      if (Number.isFinite(g.x)) {
        dot.classList.toggle("lost", !g.valid);
        dot.classList.toggle("off", !!g.off);
        place(dot, g.x, g.y);
      }
      for (const e of GT.ENGINES) {
        if (rings[e].hidden || !preds[e]) continue;
        place(rings[e], preds[e][0], preds[e][1]);
      }
      const c = T.cursor;
      if (c && Number.isFinite(g.x)) {
        line.setAttribute("x1", c[0] * innerWidth);
        line.setAttribute("y1", c[1] * innerHeight);
        line.setAttribute("x2", g.x * innerWidth);
        line.setAttribute("y2", g.y * innerHeight);
      }
      const stillFor = fr.t - T.lastMove;
      const recording = !!c && g.valid && !g.off && !T.overUI && (T.space || (stillFor >= 0.35 && stillFor <= 4));
      rec.classList.toggle("on", recording);
      if (recording) {
        T.samples.push({
          t: Math.round((fr.t - T.t0) * 1000) / 1000,
          c: [c[0], c[1]],
          live: [g.x, g.y],
          preds,
          vec: GT.gazeVector(fr),
          nn: fr.nn ? fr.nn.slice() : null,
        });
        if (T.samples.length === 30) intro.classList.add("gone");
      }
      if (fr.t - lastHud > 0.2) {
        lastHud = fr.t;
        nowEl.textContent = c && g.valid && !g.off ? `${normToDeg(g.x - c[0], g.y - c[1]).toFixed(1)}°` : g.off ? "off" : "–";
        updateStats();
      }
      if (T.map && fr.t - lastHeat > 1) {
        lastHeat = fr.t;
        drawHeat();
      }
    };
    S.listeners.add(fn);

    function errsOf(key) {
      const out = [];
      for (const s of T.samples) {
        const p = key === "live" ? s.live : s.preds[key];
        if (p) out.push(normToDeg(p[0] - s.c[0], p[1] - s.c[1]));
      }
      return out;
    }
    function updateStats() {
      const n = T.samples.length;
      cntEl.textContent = `${n} samples`;
      if (n < 10) {
        medEl.textContent = "";
        biasEl.textContent = "";
        engEl.innerHTML = "";
        return;
      }
      medEl.textContent = `typical ${median(errsOf("live")).toFixed(1)}°`;
      const bx = median(T.samples.map((s) => degXY(s.live[0] - s.c[0], 0)[0]));
      const by = median(T.samples.map((s) => degXY(0, s.live[1] - s.c[1])[1]));
      biasEl.textContent = `sits ${biasText(bx, by)}`;
      engEl.innerHTML = "";
      for (const e of engines()) {
        const er = errsOf(e);
        if (er.length < 10) continue;
        engEl.append(
          el(
            "span",
            { class: "ct-eng" + (e === C.engine ? " on" : "") },
            el("i", { style: `background:${CT_COLOR[e]}` }),
            `${ENGINE_LABEL[e]} ${median(er).toFixed(1)}°`
          )
        );
      }
    }

    function drawHeat() {
      const ctx = heat.getContext("2d");
      ctx.clearRect(0, 0, heat.width, heat.height);
      if (!T.map) return;
      const cw = heat.width / CT_COLS;
      const ch = heat.height / CT_ROWS;
      const cells = Array.from({ length: CT_COLS * CT_ROWS }, () => []);
      for (const s of T.samples) cells[ctCell(s.c)].push(normToDeg(s.live[0] - s.c[0], s.live[1] - s.c[1]));
      const dpr = window.devicePixelRatio || 1;
      ctx.font = `${Math.round(13 * dpr)}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      cells.forEach((errs, i) => {
        const x = (i % CT_COLS) * cw;
        const y = Math.floor(i / CT_COLS) * ch;
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
        if (errs.length < 8) return;
        const m = median(errs);
        const col = m <= 1.5 ? "200,255,61" : m <= 3 ? "255,179,64" : "255,91,79";
        ctx.fillStyle = `rgba(${col},${Math.min(0.16, 0.04 + m * 0.03)})`;
        ctx.fillRect(x, y, cw, ch);
        ctx.fillStyle = `rgba(${col},0.75)`;
        ctx.fillText(`${m.toFixed(1)}°`, x + cw / 2, y + ch / 2);
      });
    }

    function clearSamples() {
      T.samples = [];
      drawHeat();
      updateStats();
    }

    function exportTest() {
      const c = cmPerPx();
      const data = {
        kind: "gazeturn-cursor-test",
        version: 1,
        at: new Date().toISOString(),
        viewport: { w: innerWidth, h: innerHeight },
        cmPerPx: c,
        distanceCm: distanceCm(),
        engine: C.engine,
        neural: GT.neural ? GT.neural.state : "none",
        featVersion: GT.FEATURE_VERSION,
        calibration: S.calibData || null,
        samples: T.samples,
      };
      const name = `gazeturn-cursor-${data.at.slice(0, 16).replace(/[:T]/g, "-")}.json`;
      download(name, JSON.stringify(data), "application/json");
      toast(`Exported ${T.samples.length} samples`);
    }

    // Learn: retrain every tracker on the calibration plus these cursor samples, and measure
    // before and after on samples held back from training (every 4th two-second stretch).
    async function learn() {
      if (T.busy) return;
      const train = T.samples.filter((s) => Math.floor(s.t / 2) % 4 !== 3);
      const test = T.samples.filter((s) => Math.floor(s.t / 2) % 4 === 3);
      const cellsHit = new Set(T.samples.map((s) => ctCell(s.c))).size;
      if (train.length < 120 || test.length < 30 || cellsHit < 6) {
        toast("Record a bit more first: about a minute, all over the screen.", 5000);
        return;
      }
      T.busy = true;
      bLearn.textContent = "Learning…";
      await sleep(30);
      try {
        const res = learnFromCursor(train, test);
        showLearnPanel(res);
      } catch (e) {
        console.error(e);
        toast("Couldn't learn from this data: " + (e.message || e), 6000);
      } finally {
        T.busy = false;
        bLearn.textContent = "Learn";
      }
    }

    function showLearnPanel(res) {
      const rows = el("div", { class: "ct-lrows" });
      for (const e of GT.ENGINES) {
        const r = res.rows[e];
        if (!r) continue;
        const better = r.after < r.before;
        rows.append(
          el(
            "div",
            { class: "ct-lrow" + (e === res.best ? " best" : "") },
            el("span", { class: "n" }, el("i", { style: `background:${CT_COLOR[e]}` }), ENGINE_LABEL[e]),
            el("span", { class: "b mono", text: Number.isFinite(r.before) ? `${r.before.toFixed(1)}°` : "–" }),
            el("span", { class: "a mono" + (better ? " up" : ""), text: `${r.after.toFixed(1)}°` })
          )
        );
      }
      const cur = res.rows[C.engine];
      const gain = cur && Number.isFinite(cur.before) ? cur.before - res.rows[res.best].after : NaN;
      const panel = el(
        "div",
        { class: "ct-panel" },
        el("h2", { text: Number.isFinite(gain) && gain > 0.05 ? `${gain.toFixed(1)}° better.` : "No clear gain." }),
        el("p", { class: "hint", text: `Measured on ${res.nTest} samples it didn't learn from.` }),
        el("div", { class: "ct-lhead" }, el("span", { text: "" }), el("span", { text: "before" }), el("span", { text: "after" })),
        rows
      );
      const actions = el("div", { class: "actions" });
      actions.append(
        el("button", {
          class: "btn",
          text: `Use ${ENGINE_LABEL[res.best]}`,
          onclick: () => {
            applyLearned(res);
            panel.remove();
            clearSamples();
            toast("Updated. Recording again from zero, so the new numbers are fair.", 5000);
          },
        }),
        el("button", { class: "btn ghost", text: "Keep current", onclick: () => panel.remove() })
      );
      panel.append(actions);
      st.append(panel);
    }

    const key = (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        T.space = e.type === "keydown";
        return;
      }
      if (e.type !== "keydown") return;
      const k = e.key.toLowerCase();
      if (k === "escape") close();
      else if (k === "l") learn();
      else if (k === "m") toggle("map");
      else if (k === "v") toggle("compare");
    };
    window.addEventListener("keydown", key);
    window.addEventListener("keyup", key);
    window.addEventListener("resize", sizeHeat);
    function close() {
      S.cursorTest = false;
      S.listeners.delete(fn);
      window.removeEventListener("keydown", key);
      window.removeEventListener("keyup", key);
      window.removeEventListener("resize", sizeHeat);
      closeStage();
      updatePills();
      if (S.view === "calibrate") showCalResult();
    }
    S.stageExit = close;
    GT._lab.ct = T; // test hook
  }

  /** Retrain all trackers on calibration + cursor samples; score old and new on held-out samples. */
  function learnFromCursor(train, test) {
    const C = S.calib;
    const c = cmPerPx();
    const wcm = innerWidth * c;
    const hcm = innerHeight * c;
    const step = Math.max(1, Math.ceil(train.length / 1500));
    const extra = train.filter((_, i) => i % step === 0).map((s) => ({ s: { vec: s.vec, nn: s.nn }, target: s.c, group: 300 + ctCell(s.c) }));
    const cal = (S.calibData || []).concat(extra);
    const score = (fnPred) => {
      const er = [];
      for (const s of test) {
        const p = fnPred(s);
        if (p) er.push(normToDeg(p[0] - s.c[0], p[1] - s.c[1]));
      }
      return er.length >= test.length * 0.6 ? median(er) : NaN;
    };
    const models = {};
    const rows = {};
    for (const e of GT.ENGINES) {
      const m = fitEngine(e, cal, wcm, hcm);
      if (!m) continue;
      models[e] = m;
      const old = engineModel(e);
      const before = old
        ? score((s) => {
            const v = sampleVec(e, s);
            if (!v) return null;
            const p = GT.predict(old, v);
            return e === C.engine ? applyCorrection(p) : p;
          })
        : NaN;
      const after = score((s) => {
        const v = sampleVec(e, s);
        return v ? GT.predict(m.model, v) : null;
      });
      if (Number.isFinite(after)) rows[e] = { before, after };
    }
    let best = null;
    for (const e of Object.keys(rows)) if (!best || rows[e].after < rows[best].after) best = e;
    if (!best) throw new Error("not enough usable samples");
    return { models, rows, best, cal, nTest: test.length };
  }
  function applyLearned(res) {
    const C = S.calib;
    for (const [e, m] of Object.entries(res.models)) {
      C.models[e] = m;
      const a = res.rows[e];
      if (a) C.accs[e] = Object.assign({}, C.accs[e] || {}, { accDeg: a.after, n: 5, source: "cursor" });
    }
    C.best = res.best;
    S.calibData = res.cal;
    try {
      store.set("calibData", { featVersion: GT.FEATURE_VERSION, data: compactCal(res.cal) });
    } catch {
      /* optional */
    }
    applyEngine(S.settings.engine === "auto" ? res.best : chooseEngine(C, S.settings.engine));
    store.set("calib", C);
    updatePills();
  }
  /** Calibration samples kept for later learning (rounded to keep storage small). */
  function compactCal(cal) {
    const r = (v) => Math.round(v * 1e5) / 1e5;
    return cal.map((q) => ({ s: { vec: q.s.vec.map(r), nn: q.s.nn ? q.s.nn.map(r) : null }, target: q.target.map(r), group: q.group }));
  }

  // ---------------------------------------------------------------- reading test: config
  function renderReadSettings() {
    const s = S.settings;
    for (const r of $$("input[name=mode]")) r.checked = r.value === s.mode;
    for (const r of $$("input[name=trigger]")) r.checked = r.value === s.trigger;
    for (const r of $$("input[name=book]")) r.checked = r.value === s.book;
    const gn = $("#set-g-next");
    const gb = $("#set-g-back");
    if (!gn.options.length) {
      for (const [k, g] of Object.entries(GESTURES)) gn.append(el("option", { value: k, text: g.label }));
      gb.append(el("option", { value: "none", text: "None (use ←)" }));
      for (const [k, g] of Object.entries(GESTURES)) gb.append(el("option", { value: k, text: g.label }));
    }
    gn.value = s.gestureNext;
    gb.value = s.gestureBack;
    $("#set-font").value = s.fontSize;
    $("#out-font").textContent = s.fontSize + " px";
    $("#set-layout").checked = s.layoutAware;
    $("#set-hud").checked = s.showHud;
    $("#set-gaze").checked = s.showGaze;
    $("#set-flash").checked = s.shadowFlash;
    $("#set-paper").checked = s.paperLight;
    $("#set-recentre").checked = s.autoRecenter;
    for (const r of $$("input[name=method]")) r.checked = r.value === s.method;
    $("#method-caption").textContent = {
      hybrid: "Rest on the last word, or look up to the first line. Whichever comes first.",
      endzone: "Rest your eyes on the last word for a moment.",
      lookup: "Finish the page, then look up to its first line.",
    }[s.method];
    $("#layout-wrap").hidden = s.method === "endzone";
    $("#live-opts").hidden = s.mode !== "live";
    $("#gesture-map").hidden = !(s.mode === "live" && (s.trigger === "confirm" || s.trigger === "gesture"));
    $("#book-paste").hidden = s.book !== "paste";
    $("#book-file-wrap").hidden = s.book !== "file";
    $("#book-file-name").textContent = S.book.fileName;
    $("#mode-caption").textContent =
      s.mode === "shadow" ? "You turn with →. It predicts silently and scores itself." : "It turns for you. ← if early, → if missed.";
    $("#trigger-caption").textContent = {
      auto: "Turns by itself.",
      confirm: "Waits for your gesture to confirm.",
      dwell: "Hold your gaze on the arrow beside the page.",
      gesture: "Your gesture turns the page.",
    }[s.trigger];
  }

  function parseBook(raw) {
    let txt = String(raw || "").replace(/\r\n?/g, "\n");
    const start = txt.search(/\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n/i);
    if (start >= 0) txt = txt.slice(txt.indexOf("\n", start) + 1);
    const end = txt.search(/\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG/i);
    if (end >= 0) txt = txt.slice(0, end);
    let paras = txt
      .split(/\n\s*\n+/)
      .map((p) => p.replace(/\s*\n\s*/g, " ").replace(/\s+/g, " ").trim())
      .filter((p) => p.length > 0);
    let total = 0;
    const out = [];
    for (const p of paras) {
      if (total > 45000) break; // about 25 pages is plenty for a test
      out.push(p);
      total += p.length;
    }
    paras = out;
    return paras;
  }
  function getBook() {
    const s = S.settings;
    if (s.book === "file") {
      if (!S.book.fileText) return { error: "Choose a .txt file first." };
      return { title: S.book.fileName.replace(/\.txt$/i, ""), paragraphs: parseBook(S.book.fileText), heading: null };
    }
    if (s.book === "paste") {
      const t = $("#book-paste").value;
      if (!t.trim()) return { error: "Paste some text first." };
      return { title: "Pasted text", paragraphs: parseBook(t), heading: null };
    }
    return { title: GT.SAMPLE_BOOK.title, paragraphs: GT.SAMPLE_BOOK.paragraphs, heading: GT.SAMPLE_BOOK.title };
  }

  // ---------------------------------------------------------------- reading test: runtime
  const R = { active: false };
  const rt = () => now() - R.t0;
  const det = () => (R.mode === "shadow" ? R.runner.det : R.detector);
  const curTotal = () => (R.lines && R.lines[R.pageIndex] ? R.lines[R.pageIndex].length : 0);
  const atLastPage = () => R.pageIndex >= R.pageCount - 1;

  async function startReading() {
    if (!S.calib) {
      toast("Calibrate first.");
      return;
    }
    const book = getBook();
    if (book.error) {
      toast(book.error);
      return;
    }
    if (!book.paragraphs.length) {
      toast("That text is empty.");
      return;
    }
    const fsPromise = requestFS(); // must start inside the click
    const ok = await startCamera();
    if (!ok) {
      exitFS();
      return;
    }
    const s = S.settings;
    Object.assign(R, {
      active: false,
      mode: s.mode,
      trigger: s.mode === "live" ? s.trigger : "auto",
      method: s.method,
      cfg: { mode: s.mode, trigger: s.mode === "live" ? s.trigger : "auto", method: s.method, layoutAware: s.layoutAware, autoRecenter: s.autoRecenter, fontSize: s.fontSize, gestureNext: s.gestureNext, gestureBack: s.gestureBack },
      bookTitle: book.title,
      pageIndex: 0,
      pageCount: 1,
      lines: [],
      pages: [],
      events: [],
      lineTimes: [],
      rects: [],
      paused: false,
      pausedTotal: 0,
      pauseStart: 0,
      lastSystemTurnT: -1e9,
      armedUntil: -1,
      hudT: 0,
      validCount: 0,
      sampleCount: 0,
      startedAt: new Date().toISOString(),
      zoneN: null,
      snapLine: null,
      pendingStart: null,
      log: { t: [], px: [], py: [], gx: [], gy: [], rx: [], ry: [], v: [], page: [], brow: [], jaw: [], blink: [], smile: [] },
    });
    R.params = Object.assign({}, s.detector, s.endzone, { useLayout: s.layoutAware });
    if (R.mode === "shadow") R.runner = new GT.ShadowRunner(R.params, {}, R.method);
    else {
      R.detector = GT.makeDetector(R.method, R.params);
      R.live = new GT.LiveEvaluator();
    }
    const gHold = (k) => GESTURES[k] && GESTURES[k].hold ? GESTURES[k].hold : s.gestureHold;
    R.gNext = new GT.HoldGesture(s.thresholds[s.gestureNext], gHold(s.gestureNext));
    R.gBack = s.gestureBack !== "none" ? new GT.HoldGesture(s.thresholds[s.gestureBack], gHold(s.gestureBack)) : null;
    R.dwell = new GT.DwellDetector(s.dwellTime);

    // Build the book.
    const flow = $("#r-flow");
    flow.innerHTML = "";
    if (book.heading) flow.append(el("h2", { text: book.heading }));
    book.paragraphs.forEach((p, i) => flow.append(el("p", { class: i === 0 ? "first" : null, text: p })));
    $("#r-title").textContent = book.title;
    $("#r-mode").textContent = (R.mode === "shadow" ? "shadow · → to turn" : "live · " + triggerLabel(R.trigger, R.method)).toLowerCase();
    $("#r-endmark").hidden = R.method === "lookup" || (R.mode === "live" && (R.trigger === "dwell" || R.trigger === "gesture"));
    $("#reader").classList.toggle("paper-light", !!s.paperLight);
    $("#r-hud").hidden = !s.showHud;
    $("#r-gaze").hidden = !s.showGaze;
    $("#r-banner").hidden = true;
    $("#r-pause").hidden = true;
    $("#r-armed").hidden = true;
    $("#reader").hidden = false;

    R.fs = await fsPromise;
    await sleep(450);
    try {
      await document.fonts.ready;
    } catch {
      /* fonts are optional */
    }
    R.t0 = now();
    layoutReader();
    openPageRecord(0, "start");
    const tg0 = pageTargets(0);
    R.pendingStart = tg0 ? { from: 0.4, at: 1.2, target: tg0.start } : null;
    checkViewport();
    R.active = true;
    S.listeners.add(onReadFrame);
    window.addEventListener("keydown", onReadKey);
  }

  const METHOD_LABEL = { hybrid: "Last word or look up", endzone: "Last word", lookup: "Look up" };
  function triggerLabel(t, method) {
    const m = METHOD_LABEL[method] || METHOD_LABEL.lookup;
    return { auto: m, confirm: m + " + gesture", dwell: "Look at next", gesture: "Gesture" }[t] || t;
  }

  function layoutReader() {
    const s = S.settings;
    const fs = s.fontSize;
    const dwell = R.mode === "live" && R.trigger === "dwell";
    $("#r-zone-next").hidden = !dwell;
    $("#r-zone-back").hidden = !dwell;
    const body = $("#r-body");
    const br = body.getBoundingClientRect();
    const zones = dwell ? 2 * (110 + 28) : 0;
    const pw = Math.floor(Math.min(fs * 31, br.width - zones - 40));
    const ph = Math.floor(br.height - 44);
    R.pw = pw;
    R.ph = ph;
    R.gap = 80;
    const clip = $("#r-clip");
    const flow = $("#r-flow");
    clip.style.width = pw + "px";
    clip.style.height = ph + "px";
    flow.style.fontSize = fs + "px";
    flow.style.width = pw + "px";
    flow.style.height = ph + "px";
    flow.style.columnWidth = pw + "px";
    flow.style.columnGap = R.gap + "px";
    flow.style.transform = "translateX(0px)";
    R.lines = measureLines();
    R.pageCount = Math.max(1, R.lines.length);
    goTo(clamp(R.pageIndex, 0, R.pageCount - 1));
    const cr = clip.getBoundingClientRect();
    R.rectN = { x0: cr.left / innerWidth, x1: cr.right / innerWidth, y0: cr.top / innerHeight, y1: cr.bottom / innerHeight };
    R.rects.push({ t: R.t0 ? rt() : 0, rect: Object.assign({}, R.rectN) });
    if (dwell) {
      const zn = (n) => {
        const r = n.getBoundingClientRect();
        return { x0: r.left / innerWidth, x1: r.right / innerWidth, y0: r.top / innerHeight, y1: r.bottom / innerHeight };
      };
      R.dwell.setZones([Object.assign({ id: "next" }, zn($("#r-zone-next"))), Object.assign({ id: "back" }, zn($("#r-zone-back")))]);
    }
  }

  /** Line boxes per page, read straight from the browser's layout. */
  function measureLines() {
    const flow = $("#r-flow");
    const fr = flow.getBoundingClientRect();
    const step = R.pw + R.gap;
    const pages = [];
    const walker = document.createTreeWalker(flow, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    let node;
    while ((node = walker.nextNode())) {
      if (!node.textContent.trim()) continue;
      range.selectNodeContents(node);
      for (const r of range.getClientRects()) {
        if (r.width < 1 || r.height < 1) continue;
        const x = r.left - fr.left;
        const y = r.top - fr.top;
        const pi = Math.max(0, Math.floor((x + 2) / step));
        const px = x - pi * step;
        const page = pages[pi] || (pages[pi] = []);
        let line = page.find((l) => Math.abs(l.top - y) < 4);
        if (!line) page.push({ top: y, bottom: y + r.height, left: px, right: px + r.width });
        else {
          line.left = Math.min(line.left, px);
          line.right = Math.max(line.right, px + r.width);
          line.bottom = Math.max(line.bottom, y + r.height);
        }
      }
    }
    for (let i = 0; i < pages.length; i++) {
      if (!pages[i]) pages[i] = [];
      pages[i].sort((a, b) => a.top - b.top);
    }
    return pages;
  }

  /** Where the page's last word is, padded by the measured tracking error: tight left–right,
      generous up–down (webcams are weaker vertically). Page coordinates, 0–1. */
  function zoneFor(i) {
    const lines = R.lines[i] || [];
    if (!lines.length || !R.pw) return null;
    const L = lines[lines.length - 1];
    const fs = S.settings.fontSize;
    const lh = L.bottom - L.top;
    const wx0 = Math.max(L.left, L.right - 3 * fs);
    const A = currentAcc();
    const pxPerDeg = (Math.tan(Math.PI / 180) * distanceCm()) / cmPerPx();
    const sc = S.settings.endzone.zoneScale || 1;
    const ax = A && Number.isFinite(A.xDeg) ? A.xDeg : 1.5;
    const ay = A && Number.isFinite(A.yDeg) ? A.yDeg : 1.5;
    const mx = clamp(ax * pxPerDeg * sc, 0.5 * fs, 3 * fs);
    const my = clamp(ay * pxPerDeg * 1.5 * sc, 1.5 * lh, 4 * lh);
    return {
      x0: (wx0 - mx) / R.pw, x1: (L.right + 2 * mx + fs) / R.pw,
      y0: (L.top - my) / R.ph, y1: (L.bottom + my) / R.ph,
      wordX0: wx0 / R.pw,
      word: { left: Math.round(wx0), right: Math.round(L.right), top: Math.round(L.top), bottom: Math.round(L.bottom) },
    };
  }
  /** The first and last word of a page in viewport coordinates: known places the eyes visit. */
  function pageTargets(i) {
    const lines = R.lines[i] || [];
    if (!lines.length || !R.rectN) return null;
    const fs = S.settings.fontSize;
    const first = lines[0];
    const last = lines[lines.length - 1];
    const r = R.rectN;
    const toV = (x, y) => [r.x0 + (x / R.pw) * (r.x1 - r.x0), r.y0 + (y / R.ph) * (r.y1 - r.y0)];
    return {
      start: toV(first.left + 1.2 * fs, (first.top + first.bottom) / 2),
      end: toV(Math.max(last.left, last.right - 1.2 * fs), (last.top + last.bottom) / 2),
    };
  }
  /** Auto re-centre: compare where the gaze was against a known target and nudge the offset. */
  function correctDrift(t0, t1, target, xq, why) {
    if (!S.settings.autoRecenter || !target || !S.calib) return;
    const L = R.log;
    const xs = [];
    const ys = [];
    for (let i = L.t.length - 1; i >= 0 && L.t[i] >= t0; i--) {
      if (L.t[i] > t1 || !L.v[i] || L.rx[i] == null || L.ry[i] == null) continue;
      xs.push(L.rx[i]);
      ys.push(L.ry[i]);
    }
    if (xs.length < 5) return;
    const ex = target[0] - GT.quantile(xs, xq);
    const ey = target[1] - median(ys);
    const deg = normToDeg(ex, ey);
    if (deg > 2.5) {
      logEvent({ t: t1, type: "driftSkip", why, deg: round2(deg) });
      return;
    }
    // Two known places per page (first word near the top-left, last word near the bottom-right)
    // pin down both a shift and a stretch. Split each correction between the two.
    const g = S.settings.driftGain;
    const mx = GT.quantile(xs, xq);
    const my = median(ys);
    S.offset.x += 0.5 * g * ex;
    S.offset.y += 0.5 * g * ey;
    S.scale.x = clamp(S.scale.x + 0.5 * g * ex * ((mx - 0.5) / 0.09), -0.2, 0.6);
    S.scale.y = clamp(S.scale.y + 0.5 * g * ey * ((my - 0.5) / 0.09), -0.2, 0.6);
    logEvent({ t: t1, type: "drift", why, ex: round3(ex), ey: round3(ey), sx: round3(S.scale.x), sy: round3(S.scale.y) });
  }
  function placeEndmark() {
    const m = $("#r-endmark");
    const z = R.zoneN;
    if (!z) {
      m.style.width = "0px";
      return;
    }
    m.style.left = z.word.left + "px";
    m.style.top = z.word.top + "px";
    m.style.width = Math.max(8, z.word.right - z.word.left) + "px";
    m.style.height = z.word.bottom - z.word.top + "px";
    m.style.setProperty("--p", 0);
  }

  function goTo(i) {
    R.pageIndex = i;
    R.snapLine = null;
    $("#r-flow").style.transform = `translateX(${-i * (R.pw + R.gap)}px)`;
    $("#r-page").textContent = `${i + 1} / ${R.pageCount}`;
    const tot = curTotal();
    R.zoneN = zoneFor(i);
    placeEndmark();
    if (R.mode === "shadow") R.runner.setPage(tot, R.zoneN);
    else R.detector.setPage(tot, R.zoneN);
  }
  function openPageRecord(t, enter) {
    R.pages.push({ i: R.pageIndex, total: curTotal(), t0: t, t1: null, enter, exit: null, zone: R.zoneN });
  }
  function closePageRecord(t, exit) {
    const p = R.pages[R.pages.length - 1];
    if (p && p.t1 == null) {
      p.t1 = t;
      p.exit = exit;
    }
  }
  function changePage(dir, t, how) {
    closePageRecord(t, dir > 0 ? how : "back");
    goTo(R.pageIndex + dir);
    openPageRecord(t, how);
    disarm();
    // Right after a page appears, the eyes go to its first word: a free re-centre target.
    const tg = pageTargets(R.pageIndex);
    R.pendingStart = tg ? { from: t + 0.35, at: t + 0.95, target: tg.start } : null;
  }
  function logEvent(e) {
    R.events.push(Object.assign({}, e, { t: Math.round(e.t * 1000) / 1000 }));
  }

  function checkViewport() {
    const vp = S.calib && S.calib.viewport;
    const b = $("#r-banner");
    if (vp && (Math.abs(vp.w - innerWidth) > vp.w * 0.03 || Math.abs(vp.h - innerHeight) > vp.h * 0.03)) {
      b.textContent = "Window size changed since calibration. Recalibrate for a fair test.";
      b.hidden = false;
      logEvent({ t: rt(), type: "viewportMismatch", w: innerWidth, h: innerHeight });
    } else b.hidden = true;
  }

  function onReadFrame(fr, g) {
    if (!R.active) return;
    const t = g.t - R.t0;
    const r = R.rectN;
    const px = (g.x - r.x0) / (r.x1 - r.x0);
    const py = (g.y - r.y0) / (r.y1 - r.y0);
    const valid = g.valid && !R.paused;
    const gs = fr.ok ? GT.gestureScores(fr) : null;

    const L = R.log;
    L.t.push(round3(t));
    L.px.push(round3(px));
    L.py.push(round3(py));
    L.gx.push(round3(g.x));
    L.gy.push(round3(g.y));
    L.rx.push(round3(g.rx));
    L.ry.push(round3(g.ry));
    L.v.push(valid ? 1 : 0);
    L.page.push(R.pageIndex);
    L.brow.push(gs ? round2(gs.brow) : null);
    L.jaw.push(gs ? round2(gs.jaw) : null);
    L.blink.push(gs ? round2(gs.blink) : null);
    L.smile.push(gs ? round2(gs.smile) : null);
    if (!R.paused) {
      R.sampleCount++;
      if (g.valid) R.validCount++;
    }

    if (S.settings.showGaze) showReaderGaze(g);
    if (R.paused) return;
    if (R.pendingStart && t >= R.pendingStart.at) {
      const ps = R.pendingStart;
      R.pendingStart = null;
      correctDrift(ps.from, ps.at, ps.target, 0.15, "start");
    }

    const s = { t, x: px, y: py, valid };
    if (R.mode === "shadow") {
      for (const e of R.runner.sample(s)) handleDetEvent(e, t);
    } else {
      for (const e of R.detector.update(s)) handleDetEvent(e, t);
      if (gs) {
        if ((R.trigger === "gesture" || R.trigger === "confirm") && R.gNext.update(t, gs[S.settings.gestureNext])) onGestureNext(t);
        if (R.gBack && R.gBack.update(t, gs[S.settings.gestureBack])) {
          logEvent({ t, type: "gestureBack" });
          backAction(t, "gesture");
        }
      }
      if (R.trigger === "dwell") {
        const res = R.dwell.update(t, g.x, g.y, g.valid);
        $("#r-zone-next").style.setProperty("--p", res.progress.next || 0);
        $("#r-zone-back").style.setProperty("--p", res.progress.back || 0);
        if (res.fired === "next") systemTurn(t, "dwell");
        else if (res.fired === "back") backAction(t, "dwell");
      }
      if (R.armedUntil > 0 && t > R.armedUntil) {
        logEvent({ t, type: "armedExpired" });
        disarm();
      }
    }
    const pr = det().progress;
    if (Number.isFinite(pr)) $("#r-endmark").style.setProperty("--p", pr.toFixed(3));
    hud(t);
  }

  /** The faint gaze light while reading, settled onto the nearest text line so it doesn't
      flicker between neighbouring lines. Display only: the detectors use the raw estimate. */
  function showReaderGaze(g) {
    const dot = $("#r-gaze");
    dot.classList.toggle("off", !!g.off);
    if (!Number.isFinite(g.x) || !Number.isFinite(g.y)) return;
    dot.classList.toggle("lost", !g.valid);
    let y = g.y;
    const lines = R.lines[R.pageIndex];
    const r = R.rectN;
    if (lines && lines.length && r) {
      const py = ((g.y - r.y0) / (r.y1 - r.y0)) * R.ph;
      const centre = (l) => (l.top + l.bottom) / 2;
      let best = 0;
      let bd = Infinity;
      lines.forEach((l, i) => {
        const d = Math.abs(centre(l) - py);
        if (d < bd) {
          bd = d;
          best = i;
        }
      });
      const lh = lines[0].bottom - lines[0].top;
      const cur = R.snapLine == null ? best : Math.min(R.snapLine, lines.length - 1);
      if (best !== cur && Math.abs(centre(lines[cur]) - py) - bd > 0.35 * lh) R.snapLine = best;
      else if (R.snapLine == null) R.snapLine = best;
      const l = lines[Math.min(R.snapLine, lines.length - 1)];
      if (bd < 2.5 * lh) y = r.y0 + (centre(l) / R.ph) * (r.y1 - r.y0);
    }
    place(dot, g.x, y);
  }
  const round3 = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);
  const round2 = (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : null);

  function handleDetEvent(e, t) {
    if (e.type === "line") {
      if (R.mode !== "shadow") R.lineTimes.push(e.t); // the shadow runner keeps its own list
      return;
    }
    logEvent(e);
    if (e.type !== "pageEnd") return;
    const viaWord = e.via === "lastWord" || R.method === "endzone";
    if (R.mode === "shadow") {
      if (S.settings.shadowFlash) flash(viaWord ? "last word" : "look up");
    } else if (R.trigger === "auto") {
      if (viaWord) {
        const tg = pageTargets(R.pageIndex);
        correctDrift(t - 0.45, t, tg && tg.end, 0.5, "end");
      }
      systemTurn(t, viaWord ? "lastWord" : "lookUp");
    } else if (R.trigger === "confirm") arm(t);
  }

  function onGestureNext(t) {
    logEvent({ t, type: "gestureNext" });
    if (R.trigger === "gesture") systemTurn(t, "gesture");
    else if (R.trigger === "confirm" && R.armedUntil > 0) systemTurn(t, "confirm");
  }
  function arm(t) {
    R.armedUntil = t + 4;
    const a = $("#r-armed");
    a.textContent = `Ready: ${GESTURES[S.settings.gestureNext].label.toLowerCase()} to turn`;
    a.hidden = false;
  }
  function disarm() {
    R.armedUntil = -1;
    $("#r-armed").hidden = true;
  }
  function flash(label) {
    const f = $("#r-flash");
    f.textContent = "would turn · " + (label || "");
    f.hidden = true;
    void f.offsetWidth; // restart the animation
    f.hidden = false;
    clearTimeout(R.flashTimer);
    R.flashTimer = setTimeout(() => (f.hidden = true), 1250);
  }

  function systemTurn(t, source) {
    if (atLastPage()) {
      logEvent({ t, type: "turn", source, endOfBook: true });
      endReading("finished");
      return;
    }
    R.detector.turned(t);
    R.live.systemTurn(t, source);
    R.lastSystemTurnT = t;
    changePage(+1, t, source);
    logEvent({ t, type: "turn", source, total: curTotal() });
  }
  function backAction(t, source) {
    if (R.pageIndex === 0) return;
    const wasFalse = R.live.back(t);
    if (wasFalse) R.detector.undo(t);
    else {
      R.detector.resetPage(t);
      R.detector.lastTurnT = t;
    }
    changePage(-1, t, "back");
    logEvent({ t, type: "back", source, falseTurn: wasFalse, total: curTotal() });
  }

  function manualNext() {
    const t = rt();
    if (R.paused) return;
    if (R.mode === "shadow") {
      if (atLastPage()) {
        R.runner.manualNext(t, 0);
        logEvent({ t, type: "manualNext", total: 0, endOfBook: true });
        endReading("finished");
        return;
      }
      const tg = pageTargets(R.pageIndex);
      correctDrift(t - 0.6, t - 0.15, tg && tg.end, 0.5, "end"); // you were on the last word
      changePage(+1, t, "manual");
      R.runner.manualNext(t, curTotal(), R.zoneN);
      logEvent({ t, type: "manualNext", total: curTotal(), zone: R.zoneN });
    } else {
      if (t - R.lastSystemTurnT < 1.5) {
        R.live.late(t);
        logEvent({ t, type: "late" });
        return;
      }
      if (atLastPage()) {
        endReading("finished");
        return;
      }
      R.live.manualNext(t);
      const tg = pageTargets(R.pageIndex);
      correctDrift(t - 0.6, t - 0.15, tg && tg.end, 0.5, "end");
      R.detector.turned(t);
      changePage(+1, t, "manual");
      logEvent({ t, type: "manualNext", total: curTotal(), zone: R.zoneN });
    }
  }
  function manualBack() {
    const t = rt();
    if (R.paused || R.pageIndex === 0) return;
    if (R.mode === "shadow") {
      changePage(-1, t, "back");
      R.runner.manualBack(t, curTotal(), R.zoneN);
      logEvent({ t, type: "manualBack", total: curTotal(), zone: R.zoneN });
    } else backAction(t, "key");
  }

  function onReadKey(e) {
    if (!R.active) return;
    const k = e.key;
    if (!$("#r-pause").hidden) {
      if (k === "Enter") resumeReading();
      return;
    }
    if (k === "ArrowRight" || k === " " || k === "PageDown") {
      e.preventDefault();
      manualNext();
    } else if (k === "ArrowLeft" || k === "PageUp") {
      e.preventDefault();
      manualBack();
    } else if (k === "Escape") pauseReading("Time while paused isn't scored.");
    else if (k === "p" || k === "P") pauseReading("Time while paused isn't scored.");
    else if (k === "h" || k === "H") {
      S.settings.showHud = !S.settings.showHud;
      $("#r-hud").hidden = !S.settings.showHud;
      saveSettings();
    } else if (k === "g" || k === "G") {
      S.settings.showGaze = !S.settings.showGaze;
      $("#r-gaze").hidden = !S.settings.showGaze;
      saveSettings();
    } else if (k === "c" || k === "C") readerRecenter();
  }

  async function readerRecenter() {
    if (R.recentering) return;
    R.recentering = true;
    const t = rt();
    R.paused = true;
    R.pauseStart = t;
    await recenter($("#reader"));
    R.pausedTotal += rt() - R.pauseStart;
    R.paused = false;
    logEvent({ t, type: "recenter", dx: round3(S.offset.x), dy: round3(S.offset.y) });
    R.recentering = false;
  }

  function pauseReading(msg) {
    if (!R.active || R.paused) return;
    R.paused = true;
    R.pauseStart = rt();
    logEvent({ t: R.pauseStart, type: "pause" });
    $("#r-pause-msg").textContent = msg;
    $("#r-pause").hidden = false;
    $("#r-resume").focus();
  }
  async function resumeReading() {
    if (!R.active) return;
    $("#r-pause").hidden = true;
    if (R.fs && !document.fullscreenElement) await requestFS();
    await sleep(350);
    layoutReader();
    checkViewport();
    const t = rt();
    R.pausedTotal += t - R.pauseStart;
    R.paused = false;
    logEvent({ t, type: "resume" });
  }

  function hud(t) {
    if (!S.settings.showHud || t - R.hudT < 0.2) return;
    R.hudT = t;
    const d = det();
    let s = R.method === "lookup" ? `${d.lines}/${Math.max(0, curTotal() - 1)} lines · turn at ${d.linesNeeded()}` : `last word ${Math.round((d.progress || 0) * 100)}%`;
    s += S.gaze && S.gaze.valid ? " · ●" : " · ○";
    if (R.mode === "shadow") {
      const ev = R.runner.ev;
      s += ` · lab ${ev.preds.length} · you ${ev.manuals.length}`;
    } else {
      s += ` · turns ${R.live.turns.length}`;
    }
    $("#r-hud").textContent = s;
  }

  async function endReading(reason) {
    if (!R.active) return;
    const tEnd = R.paused ? R.pauseStart : rt();
    if (R.paused) R.pausedTotal += rt() - R.pauseStart;
    R.active = false;
    S.listeners.delete(onReadFrame);
    window.removeEventListener("keydown", onReadKey);
    closePageRecord(tEnd, "end");
    $("#reader").hidden = true;
    $("#r-pause").hidden = true;
    exitFS();

    const evalResult = R.mode === "shadow" ? R.runner.result() : R.live.result();
    const data = {
      id: "s" + Date.now().toString(36),
      version: 1,
      startedAt: R.startedAt,
      endedAt: new Date().toISOString(),
      reason,
      cfg: R.cfg,
      book: R.bookTitle,
      screenWidthCm: screenWidthCm(),
      distanceCm: distanceCm(),
      params: R.params,
      filter: Object.assign({}, S.settings.filter),
      calib: S.calib ? { accDeg: currentAcc().accDeg, xDeg: currentAcc().xDeg, yDeg: currentAcc().yDeg, at: S.calib.at, engine: S.calib.engine || "classic" } : null,
      viewport: { w: innerWidth, h: innerHeight },
      rects: R.rects,
      pages: R.pages,
      events: R.events,
      lineTimes: R.mode === "shadow" ? R.runner.lineTimes.slice() : R.lineTimes,
      detStats: Object.assign({}, det().stats),
      evalResult,
      durationS: Math.max(0, tEnd - R.pausedTotal),
      fps: R.log.t.length / Math.max(tEnd, 1),
      validPct: R.sampleCount ? R.validCount / R.sampleCount : NaN,
      samples: R.log,
    };
    data.summary = summarizeData(data);
    S.full[data.id] = data;
    await idb.put(data);
    S.sessions.unshift(rowFrom(data));
    S.sessions = S.sessions.slice(0, 80);
    store.set("sessions", S.sessions);
    S.selected = data.id;
    updatePills();
    showView("results");
  }

  function summarizeData(data, over = {}) {
    return GT.summarize({
      kind: data.cfg.mode,
      trigger: data.cfg.trigger,
      durationS: data.durationS,
      validPct: data.validPct,
      fps: data.fps,
      calibDeg: data.calib ? data.calib.accDeg : NaN,
      pages: data.pages,
      lineTimes: over.lineTimes || data.lineTimes,
      detStats: over.detStats || data.detStats,
      evalResult: over.evalResult || data.evalResult,
    });
  }
  function rowFrom(data) {
    return { id: data.id, startedAt: data.startedAt, book: data.book, cfg: data.cfg, summary: data.summary, rescored: data.rescored || null };
  }

  // ---------------------------------------------------------------- results
  function testLabel(cfg) {
    const how = cfg.mode === "shadow" ? METHOD_LABEL[cfg.method] || METHOD_LABEL.lookup : triggerLabel(cfg.trigger, cfg.method);
    return (cfg.mode === "shadow" ? "Shadow" : "Live") + " · " + how;
  }
  const VERDICT_WORD = { good: "Promising", ok: "Workable", bad: "Not yet", more: "Too short" };
  function verdictChip(v) {
    return el("span", { class: "verdict", "data-level": v ? v.level : "more", text: v ? VERDICT_WORD[v.level] : "–" });
  }

  function renderResults() {
    const tb = $("#res-table tbody");
    tb.innerHTML = "";
    $("#res-empty").hidden = S.sessions.length > 0;
    $("#res-table").hidden = !S.sessions.length;
    for (const row of S.sessions) {
      const s = row.summary;
      const d = new Date(row.startedAt);
      const tr = el(
        "tr",
        { tabindex: "0", class: row.id === S.selected ? "sel" : null },
        el("td", { text: d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) }),
        el("td", { text: testLabel(row.cfg) }),
        el("td", { class: "num", text: fmt(s.minutes, 1) }),
        el("td", { class: "num", text: String(s.pagesTurned) }),
        el("td", { class: "num", text: pct(s.recall) }),
        el("td", { class: "num", text: fmt(s.falsePerHour, 1) }),
        el("td", { class: "num", text: String(s.misses) }),
        el("td", { class: "num", text: Number.isFinite(s.lineRatio) ? pct(s.lineRatio) : "–" }),
        el("td", { class: "num", text: Number.isFinite(s.calibDeg) ? fmt(s.calibDeg, 1) + "°" : "–" }),
        el("td", {}, verdictChip(s.verdict))
      );
      tr.addEventListener("click", () => selectSession(row.id));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter") selectSession(row.id);
      });
      tb.append(tr);
    }
    const id = S.selected || (S.sessions[0] && S.sessions[0].id);
    if (id) renderDetail(id);
    else $("#res-detail").innerHTML = "";
  }
  function selectSession(id) {
    S.selected = id;
    renderResults();
    $("#res-detail").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function getData(id) {
    if (S.full[id]) return S.full[id];
    const d = await idb.get(id);
    if (d) S.full[id] = d;
    return d || null;
  }

  async function renderDetail(id) {
    const host = $("#res-detail");
    const row = S.sessions.find((r) => r.id === id);
    if (!row) {
      host.innerHTML = "";
      return;
    }
    const data = await getData(id);
    if (S.selected && S.selected !== id) return;
    const s = row.summary;
    host.innerHTML = "";
    const d = new Date(row.startedAt);
    const lvl = s.verdict.level;
    host.append(
      el(
        "div",
        { class: "res-hero" },
        el("span", { class: "res-kicker", text: `${testLabel(row.cfg)} · ${d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}` }),
        el("h1", { class: "res-verdict", "data-level": lvl, text: VERDICT_WORD[lvl] + "." }),
        el("p", { class: "res-line", text: s.verdict.text })
      )
    );

    const tiles = el("div", { class: "tiles" });
    const caughtState = !Number.isFinite(s.recall) ? null : s.recall >= 0.9 ? "good" : s.recall >= 0.7 ? "warn" : "bad";
    tiles.append(tile("caught", pct(s.recall), `${s.hits} of ${s.hits + s.misses}`, caughtState));
    tiles.append(tile("false / hour", fmt(s.falsePerHour, 1), `${s.falseTurns} in ${fmt(s.minutes, 1)} min`, s.falsePerHour <= 2 ? "good" : s.falsePerHour <= 6 ? "warn" : "bad"));
    tiles.append(tile("missed", String(s.misses), null));
    tiles.append(
      tile(
        "line counting",
        Number.isFinite(s.lineRatio) ? pct(s.lineRatio) : "–",
        Number.isFinite(s.lineErrPct) ? `±${pct(s.lineErrPct)} per page` : null,
        Number.isFinite(s.lineErrPct) ? (s.lineErrPct <= 0.15 ? "good" : s.lineErrPct <= 0.3 ? "warn" : "bad") : null
      )
    );
    host.append(tiles);

    const extra = [];
    if (row.cfg.mode === "shadow" && Number.isFinite(s.medianOffset)) extra.push(`timing ${s.medianOffset >= 0 ? "+" : ""}${fmt(s.medianOffset, 1)} s`);
    if (row.cfg.mode !== "shadow") extra.push(`${s.late} late presses`);
    extra.push(`tracking ${pct(s.validPct)} at ${fmt(s.fps, 0)} fps`);
    if (Number.isFinite(s.calibDeg)) extra.push(`calibration ${fmt(s.calibDeg, 1)}°` + (data && data.calib && data.calib.engine ? ` (${data.calib.engine})` : ""));
    const why = el("p", { class: "res-why mono", text: extra.join("   ·   ") });
    host.append(why);

    if (s.rejects) {
      const r = s.rejects;
      const parts = [];
      if (r.viaLastWord != null) parts.push(`${r.viaLastWord} by the last word, ${r.viaLookUp} by the look-up`);
      const why = [];
      if (r.rejectApproach) why.push(`${r.rejectApproach} didn't read in`);
      if (r.rejectLines) why.push(`${r.rejectLines} too few lines`);
      if (r.rejectLow) why.push(`${r.rejectLow} gaze not low`);
      if (r.rejectReturned) why.push(`${r.rejectReturned} glances`);
      if (r.rejectLanding) why.push(`${r.rejectLanding} not on the left`);
      if (r.rejectTime) why.push(`${r.rejectTime} too soon`);
      if (why.length) parts.push(`ignored: ${why.join(", ")}`);
      if (parts.length) host.append(el("p", { class: "res-why" }, parts.join(" · ") + "."));
    }

    if (s.linePages && s.linePages.length) {
      const blk = el("div", { class: "res-block" }, el("h3", { text: "Lines per page" }), pageChart(s.linePages));
      blk.append(el("div", { class: "legend" }, el("span", {}, el("i", { style: "background:var(--bg-3)" }), "real"), el("span", {}, el("i", { style: "background:var(--lime)" }), "counted")));
      host.append(blk);
    }
    if (data) {
      const blk = el("div", { class: "res-block" }, el("h3", { text: "Timeline" }), timelineChart(data));
      blk.append(
        el(
          "div",
          { class: "legend" },
          el("span", {}, el("i", { style: "background:var(--ink)" }), row.cfg.mode === "shadow" ? "you" : "you pressed →"),
          el("span", {}, el("i", { style: "background:var(--lime)" }), row.cfg.mode === "shadow" ? "predicted" : "turned"),
          el("span", {}, el("i", { style: "background:var(--bad)" }), "false"),
          el("span", {}, el("i", { style: "background:transparent;box-shadow:inset 0 0 0 2px var(--warn)" }), "missed")
        )
      );
      host.append(blk);
    }

    if (row.rescored) host.append(el("div", { class: "res-block" }, compareTable(s, row.rescored)));

    const actions = el("div", { class: "actions res-actions" });
    if (data && row.cfg.mode === "shadow") actions.append(el("button", { class: "btn", text: "Re-score", onclick: () => rescore(id) }));
    if (data) actions.append(el("button", { class: "btn ghost", text: "Export", onclick: () => exportSession(id) }));
    actions.append(el("button", { class: "btn danger small", text: "Delete", onclick: () => deleteSession(id) }));
    host.append(actions);
  }

  function compareTable(a, b) {
    const rows = [
      ["Turns caught", pct(a.recall), pct(b.recall)],
      ["False turns / hour", fmt(a.falsePerHour, 1), fmt(b.falsePerHour, 1)],
      ["Missed turns", String(a.misses), String(b.misses)],
      ["Line counting", pct(a.lineRatio), pct(b.lineRatio)],
      ["Verdict", VERDICT_WORD[a.verdict.level], VERDICT_WORD[b.verdict.level]],
    ];
    const t = el("table", { class: "compare" }, el("thead", {}, el("tr", {}, el("th", { text: "" }), el("th", { text: "Before" }), el("th", { text: "After" }))));
    const tb = el("tbody");
    for (const r of rows) tb.append(el("tr", {}, el("td", { text: r[0] }), el("td", { class: "num", text: r[1] }), el("td", { class: "num", text: r[2] })));
    t.append(tb);
    return el("div", { class: "table-wrap" }, el("h3", { text: "Re-scored" }), t);
  }

  const SVGNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs = {}, text) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  }
  function pageChart(per) {
    const W = 800;
    const H = 190;
    const L = 34;
    const B = 24;
    const T = 10;
    const max = Math.max(4, ...per.map((p) => Math.max(p.expected, p.detected)));
    const sv = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Expected and counted line returns per page" });
    const y = (v) => T + (H - T - B) * (1 - v / max);
    const ticks = [0, Math.round(max / 2), max];
    for (const tk of ticks) {
      sv.append(svg("line", { x1: L, x2: W, y1: y(tk), y2: y(tk), class: "axis" }));
      sv.append(svg("text", { x: L - 6, y: y(tk) + 4, "text-anchor": "end" }, String(tk)));
    }
    const slot = (W - L - 8) / per.length;
    const bw = Math.min(16, slot * 0.36);
    per.forEach((p, i) => {
      const cx = L + 4 + slot * (i + 0.5);
      sv.append(svg("rect", { x: cx - bw - 1, y: y(p.expected), width: bw, height: Math.max(0, y(0) - y(p.expected)), class: "exp" }));
      sv.append(svg("rect", { x: cx + 1, y: y(p.detected), width: bw, height: Math.max(0, y(0) - y(p.detected)), class: "det" }));
      if (per.length <= 30 || i % 2 === 0) sv.append(svg("text", { x: cx, y: H - 6, "text-anchor": "middle" }, String(p.i + 1)));
    });
    return el("div", { class: "table-wrap" }, sv);
  }
  function timelineChart(data) {
    const W = 800;
    const H = 96;
    const L = 70;
    const R2 = 12;
    const end = Math.max(1, ...data.samples.t.filter(Number.isFinite));
    const x = (t) => L + ((W - L - R2) * t) / end;
    const sv = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart", role: "img", "aria-label": "Timeline of page turns" });
    const rows = { you: 28, lab: 64 };
    sv.append(svg("text", { x: 0, y: rows.you + 4 }, data.cfg.mode === "shadow" ? "you" : "you →"));
    sv.append(svg("text", { x: 0, y: rows.lab + 4 }, "lab"));
    for (const r of Object.values(rows)) sv.append(svg("line", { x1: L, x2: W - R2, y1: r, y2: r, class: "axis" }));
    const tick = (t, row, cls) => sv.append(svg("line", { x1: x(t), x2: x(t), y1: row - 10, y2: row + 10, class: cls }));
    const ring = (t, row) => sv.append(svg("circle", { cx: x(t), cy: row, r: 7, class: "tk-miss" }));
    const r = data.evalResult;
    if (data.cfg.mode === "shadow") {
      for (const h of r.hits) {
        tick(h.manualT, rows.you, "tk-you");
        tick(h.predT, rows.lab, "tk-lab");
      }
      for (const m of r.misses) {
        tick(m, rows.you, "tk-you");
        ring(m, rows.you);
      }
      for (const f of r.falseTurns) tick(f, rows.lab, "tk-false");
    } else {
      for (const tn of r.turns) tick(tn.t, rows.lab, tn.false ? "tk-false" : "tk-lab");
      for (const m of r.misses) {
        tick(m, rows.you, "tk-you");
        ring(m, rows.you);
      }
    }
    for (let m = 1; m * 60 < end; m++) sv.append(svg("text", { x: x(m * 60), y: H - 2, "text-anchor": "middle" }, m + " min"));
    return el("div", { class: "table-wrap" }, sv);
  }

  async function rescore(id) {
    const data = await getData(id);
    if (!data) return;
    const params = Object.assign({}, S.settings.detector, S.settings.endzone, { useLayout: data.cfg.layoutAware });
    const rep = GT.replayShadow(data, params, S.settings.filter);
    const sum = summarizeData(data, rep);
    const row = S.sessions.find((r) => r.id === id);
    row.rescored = sum;
    store.set("sessions", S.sessions);
    renderResults();
    toast("Re-scored");
  }

  async function exportSession(id) {
    const data = await getData(id);
    if (!data) return;
    const name = `gazeturn-${data.cfg.mode}-${data.startedAt.slice(0, 16).replace(/[:T]/g, "-")}.json`;
    download(name, JSON.stringify(data), "application/json");
  }
  function downloadBlob(name, blob) {
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = el("a", { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  async function deleteSession(id) {
    const row = S.sessions.find((r) => r.id === id);
    if (!row) return;
    const btn = $("#res-detail .btn.danger");
    if (btn && btn.dataset.confirm !== "1") {
      btn.dataset.confirm = "1";
      btn.textContent = "Press again to delete";
      setTimeout(() => {
        if (btn.isConnected) {
          btn.dataset.confirm = "";
          btn.textContent = "Delete";
        }
      }, 3000);
      return;
    }
    S.sessions = S.sessions.filter((r) => r.id !== id);
    store.set("sessions", S.sessions);
    delete S.full[id];
    await idb.del(id);
    S.selected = null;
    renderResults();
    updatePills();
    toast("Deleted");
  }
  function exportCSV() {
    const head = ["started", "test", "trigger", "book", "minutes", "pages", "caught", "false_per_hour", "false_turns", "missed", "line_ratio", "line_err", "calib_deg", "tracking_valid", "fps", "verdict"];
    const lines = [head.join(",")];
    for (const r of S.sessions) {
      const s = r.summary;
      const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
      lines.push([r.startedAt, r.cfg.mode, r.cfg.trigger, q(r.book), fmt(s.minutes, 2), s.pagesTurned, fmt(s.recall, 3), fmt(s.falsePerHour, 2), s.falseTurns, s.misses, fmt(s.lineRatio, 3), fmt(s.lineErrPct, 3), fmt(s.calibDeg, 2), fmt(s.validPct, 3), fmt(s.fps, 1), s.verdict.level].join(","));
    }
    download("gazeturn-results.csv", lines.join("\n"), "text/csv");
  }
  async function importSession(file) {
    try {
      const data = JSON.parse(await file.text());
      if (!data || !data.id || !data.samples || !data.cfg) throw new Error("not a GazeTurn session file");
      data.summary = summarizeData(data);
      S.full[data.id] = data;
      await idb.put(data);
      if (!S.sessions.some((r) => r.id === data.id)) S.sessions.unshift(rowFrom(data));
      store.set("sessions", S.sessions);
      S.selected = data.id;
      renderResults();
      toast("Session imported");
    } catch (e) {
      toast("Couldn't import: " + (e.message || e), 6000);
    }
  }

  // ---------------------------------------------------------------- tuning
  const TUNE = [
    { group: "Line returns", note: "How a jump from the end of one line to the start of the next is recognised." },
    { k: "returnAmp", label: "Jump size", help: "How far left the gaze must move, as a share of the text width.", min: 0.15, max: 0.6, step: 0.01 },
    { k: "returnWindow", label: "Jump time window", help: "The jump must happen within this time.", min: 0.2, max: 0.8, step: 0.05, unit: "s" },
    { k: "returnRefractory", label: "Minimum time between lines", min: 0.3, max: 1.5, step: 0.05, unit: "s" },
    { k: "returnFromMin", label: "Jump starts right of", help: "Share of the text width. Lower values also count short paragraph-ending lines.", min: 0.1, max: 0.7, step: 0.01 },
    { k: "returnToMax", label: "Jump lands left of", min: 0.3, max: 0.95, step: 0.01 },
    { group: "Page end", note: "When the lab decides you've finished the page." },
    { k: "nearEndFrac", label: "Lines needed first", help: "Share of the page's line returns that must be counted before a turn.", min: 0.3, max: 1, step: 0.05 },
    { k: "yLowMin", label: "Gaze low on the page first", help: "Median gaze height in the 1.5 s before the jump: 0 top, 1 bottom.", min: 0, max: 0.9, step: 0.05 },
    { k: "upAmp", label: "Upward jump size", help: "Share of the page height.", min: 0.15, max: 0.8, step: 0.01 },
    { k: "settleTime", label: "Stay up for", help: "The gaze must stay near the top this long, so a quick glance back doesn't count.", min: 0.1, max: 1.5, step: 0.05, unit: "s" },
    { k: "landLeftMax", label: "Rest left of", help: "After the jump the gaze should rest where a page begins (share of the text width).", min: 0.3, max: 1, step: 0.05 },
    { k: "minPageTime", label: "Minimum time on a page", min: 1, max: 30, step: 1, unit: "s" },
    { k: "cooldown", label: "Pause after a turn", min: 0.5, max: 5, step: 0.1, unit: "s" },
    { group: "Last word" },
    { k: "endzone.endDwell", label: "Rest on the last word for", min: 0.2, max: 1.5, step: 0.05, unit: "s" },
    { k: "endzone.stillMax", label: "Stillness", help: "How much the gaze may still drift right while resting (share of text width).", min: 0.01, max: 0.2, step: 0.01 },
    { k: "endzone.approachMin", label: "Read in from", help: "The gaze must arrive from at least this far left (share of text width).", min: 0, max: 0.6, step: 0.05 },
    { k: "endzone.zoneScale", label: "Target size", help: "Scales the area around the last word, which is sized from your calibration.", min: 0.5, max: 2.5, step: 0.1 },
    { group: "Re-centre" },
    { k: "driftGain", label: "Correction strength", help: "How far each page start and end pulls the gaze back onto the text.", min: 0, max: 1, step: 0.05 },
    { group: "Smoothing", note: "One Euro filter on the gaze position. Applies live, and to re-scoring." },
    { k: "filter.minCutoff", label: "Steadiness", help: "Lower is steadier when your eyes are still, but laggier.", min: 0.2, max: 5, step: 0.1, unit: "Hz" },
    { k: "filter.beta", label: "Speed response", help: "Higher follows fast jumps sooner.", min: 0, max: 10, step: 0.1 },
    { group: "Gestures and dwell" },
    { k: "dwellTime", label: "Look-at-button time", min: 0.4, max: 2, step: 0.05, unit: "s" },
    { k: "gestureHold", label: "Hold a gesture for", help: "Long blink always needs about 0.6 s.", min: 0.1, max: 1, step: 0.05, unit: "s" },
    { k: "blinkThreshold", label: "Blink cut-off", help: "Gaze is ignored while eyes are this closed (0–1).", min: 0.3, max: 0.9, step: 0.05 },
  ];
  function tuneGet(k) {
    if (k.startsWith("endzone.")) return S.settings.endzone[k.slice(8)];
    if (k.startsWith("filter.")) return S.settings.filter[k.slice(7)];
    if (k in S.settings.detector) return S.settings.detector[k];
    return S.settings[k];
  }
  function tuneSet(k, v) {
    if (k.startsWith("endzone.")) {
      S.settings.endzone[k.slice(8)] = v;
      saveSettings();
      return;
    }
    if (k.startsWith("filter.")) {
      S.settings.filter[k.slice(7)] = v;
      resetFilters();
    } else if (k in S.settings.detector) S.settings.detector[k] = v;
    else S.settings[k] = v;
    saveSettings();
  }
  function renderTuning() {
    const host = $("#tune-groups");
    host.innerHTML = "";
    let panel = null;
    for (const t of TUNE) {
      if (t.group) {
        panel = el("div", { class: "tune-group" }, el("h2", { text: t.group }));
        host.append(panel);
        continue;
      }
      const v = tuneGet(t.k);
      const out = el("output", { class: "mono", text: fmtTune(v, t) });
      const input = el("input", { type: "range", min: String(t.min), max: String(t.max), step: String(t.step), value: String(v), id: "tune-" + t.k.replace(".", "-"), "aria-label": t.label });
      input.addEventListener("input", () => {
        tuneSet(t.k, +input.value);
        out.textContent = fmtTune(+input.value, t);
      });
      const label = el("div", {}, el("div", { class: "tune-label", text: t.label }), t.help ? el("div", { class: "tune-help", text: t.help }) : null);
      panel.append(el("div", { class: "tune-row" }, label, input, out));
    }
  }
  function fmtTune(v, t) {
    const d = t.step >= 1 ? 0 : t.step >= 0.1 ? 1 : 2;
    return v.toFixed(d) + (t.unit ? " " + t.unit : "");
  }

  // ---------------------------------------------------------------- wiring
  function openDrawer() {
    $("#drawer").hidden = false;
    renderScreenSettings();
    renderEngineSetting();
  }
  function closeDrawer() {
    $("#drawer").hidden = true;
  }
  function applyLook(look) {
    S.settings.look = ["keynote", "glass", "halo"].includes(look) ? look : "keynote";
    document.documentElement.dataset.look = S.settings.look;
    for (const b of $$(".look")) b.setAttribute("aria-pressed", String(b.dataset.look === S.settings.look));
    saveSettings();
  }

  // The intro light drifts toward the pointer: a small preview of what the camera will do.
  function introOrb() {
    const field = $("#orb-field");
    const orb = $("#intro-orb");
    let tx = 0;
    let ty = 0;
    let x = 0;
    let y = 0;
    let t0 = performance.now();
    window.addEventListener("pointermove", (e) => {
      if (S.view !== "intro") return;
      const r = field.getBoundingClientRect();
      tx = clamp((e.clientX - (r.left + r.width / 2)) / (window.innerWidth / 2), -1, 1) * (r.width / 2 - 20);
      ty = clamp((e.clientY - (r.top + r.height / 2)) / (window.innerHeight / 2), -1, 1) * (r.height / 2 - 10);
      t0 = performance.now();
    });
    const tick = (now) => {
      if (S.view === "intro" && !reducedMotion()) {
        const idle = now - t0 > 2500;
        const gx = idle ? Math.sin(now / 1700) * 120 : tx;
        const gy = idle ? Math.sin(now / 1100) * 18 : ty;
        x += (gx - x) * 0.06;
        y += (gy - y) * 0.06;
        orb.style.transform = `translate(${x}px, ${y}px)`;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function bind() {
    for (const b of $$(".step, .wordmark")) b.addEventListener("click", () => showView(b.dataset.view));
    for (const b of $$("[data-go]")) b.addEventListener("click", () => showView(b.dataset.go));
    $("#btn-begin").addEventListener("click", () => {
      showView("setup");
      startCamera();
    });
    $("#btn-setup-next").addEventListener("click", () => showView("calibrate"));
    $("#btn-settings").addEventListener("click", openDrawer);
    for (const c of $$("#drawer [data-close]")) c.addEventListener("click", closeDrawer);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("#drawer").hidden) closeDrawer();
    });
    for (const b of $$(".look")) b.addEventListener("click", () => applyLook(b.dataset.look));
    for (const r of $$("input[name=engine]")) {
      r.addEventListener("change", () => {
        S.settings.engine = r.value;
        saveSettings();
        if (S.calib && S.calib.models) {
          applyEngine(chooseEngine(S.calib, r.value));
          store.set("calib", S.calib);
          updatePills();
          if (S.view === "calibrate") showCalResult();
        }
      });
    }
    window.addEventListener("gt-neural-ready", renderEngineSetting);
    window.addEventListener("gt-neural-error", renderEngineSetting);
    $("#set-paper").addEventListener("change", (e) => {
      S.settings.paperLight = e.target.checked;
      saveSettings();
    });
    $("#btn-camera").addEventListener("click", () => startCamera());
    $("#btn-calibrate").addEventListener("click", runCalibration);
    $("#btn-recal").addEventListener("click", runCalibration);
    $("#btn-check").addEventListener("click", runCheck);
    $("#btn-cursor").addEventListener("click", runCursorTest);
    $("#btn-capture").addEventListener("click", () => GT.capture.run());
    $("#btn-read").addEventListener("click", startReading);

    $("#set-screen").addEventListener("change", (e) => {
      S.settings.screen = e.target.value;
      saveSettings();
      renderScreenSettings();
    });
    $("#set-screen-cm").addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      if (v > 10 && v < 200) {
        S.settings.screenWidthCm = v;
        saveSettings();
        $("#out-screen").textContent = `${fmt(screenWidthCm(), 1)} cm`;
      }
    });
    for (const r of $$("input[name=dist]")) {
      r.addEventListener("change", () => {
        S.settings.distanceMode = r.value;
        saveSettings();
      });
    }
    $("#set-dist-cm").addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      if (v >= 20 && v <= 150) {
        S.settings.distanceCm = v;
        S.settings.distanceMode = "manual";
        $("#dist-manual").checked = true;
        saveSettings();
      }
    });

    const onRead = () => {
      saveSettings();
      renderReadSettings();
    };
    for (const r of $$("input[name=mode]")) r.addEventListener("change", () => { S.settings.mode = r.value; onRead(); });
    for (const r of $$("input[name=trigger]")) r.addEventListener("change", () => { S.settings.trigger = r.value; onRead(); });
    for (const r of $$("input[name=book]")) r.addEventListener("change", () => { S.settings.book = r.value; onRead(); });
    for (const r of $$("input[name=method]")) r.addEventListener("change", () => { S.settings.method = r.value; onRead(); });
    $("#set-recentre").addEventListener("change", (e) => { S.settings.autoRecenter = e.target.checked; onRead(); });
    $("#set-g-next").addEventListener("change", (e) => { S.settings.gestureNext = e.target.value; onRead(); });
    $("#set-g-back").addEventListener("change", (e) => { S.settings.gestureBack = e.target.value; onRead(); });
    $("#set-font").addEventListener("input", (e) => { S.settings.fontSize = +e.target.value; onRead(); });
    $("#set-layout").addEventListener("change", (e) => { S.settings.layoutAware = e.target.checked; onRead(); });
    $("#set-hud").addEventListener("change", (e) => { S.settings.showHud = e.target.checked; onRead(); });
    $("#set-gaze").addEventListener("change", (e) => { S.settings.showGaze = e.target.checked; onRead(); });
    $("#set-flash").addEventListener("change", (e) => { S.settings.shadowFlash = e.target.checked; onRead(); });
    $("#book-file").addEventListener("change", async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      S.book.fileName = f.name;
      S.book.fileText = await f.text();
      S.settings.book = "file";
      onRead();
    });
    $("#book-paste").addEventListener("focus", () => {
      if (S.settings.book !== "paste") {
        S.settings.book = "paste";
        onRead();
      }
    });

    $("#r-end").addEventListener("click", () => pauseReading("End to see your scores."));
    $("#r-resume").addEventListener("click", resumeReading);
    $("#r-finish").addEventListener("click", () => endReading("ended"));

    $("#btn-import").addEventListener("click", () => $("#file-import").click());
    $("#file-import").addEventListener("change", (e) => {
      const f = e.target.files && e.target.files[0];
      if (f) importSession(f);
      e.target.value = "";
    });
    $("#btn-csv").addEventListener("click", exportCSV);
    $("#btn-tune-reset").addEventListener("click", () => {
      S.settings.detector = Object.assign({}, GT.DETECTOR_DEFAULTS);
      S.settings.endzone = Object.assign({ zoneScale: 1 }, GT.ENDZONE_DEFAULTS);
      S.settings.driftGain = DEFAULTS.driftGain;
      S.settings.filter = Object.assign({}, DEFAULTS.filter);
      S.settings.dwellTime = DEFAULTS.dwellTime;
      S.settings.gestureHold = DEFAULTS.gestureHold;
      S.settings.blinkThreshold = DEFAULTS.blinkThreshold;
      saveSettings();
      resetFilters();
      renderTuning();
      toast("Reset");
    });

    document.addEventListener("fullscreenchange", () => {
      if (document.fullscreenElement) return;
      if (!$("#stage").hidden && S.stageFS) {
        S.stageCancelled = true;
        if (S.stageCancel) S.stageCancel();
        if (S.stageExit) S.stageExit();
      }
      if (R.active && R.fs && !R.paused) pauseReading("Full screen closed.");
    });
    let resizeTimer = 0;
    window.addEventListener("resize", () => {
      if (!R.active) return;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        layoutReader();
        checkViewport();
        logEvent({ t: rt(), type: "layout", w: innerWidth, h: innerHeight });
      }, 250);
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden && R.active) pauseReading("Tab was hidden.");
    });
  }

  function init() {
    S.video = $("#cam");
    const saved = store.get("calib", null);
    if (saved && saved.model && saved.acc && saved.featVersion === GT.FEATURE_VERSION) {
      S.calib = saved;
      S.calib.latest = null;
      const cd = store.get("calibData", null);
      if (cd && cd.featVersion === GT.FEATURE_VERSION && Array.isArray(cd.data)) S.calibData = cd.data;
    } else if (saved) {
      setTimeout(() => toast("Tracker upgraded. Please calibrate again.", 6000), 1200);
    }
    resetFilters();
    applyLook(S.settings.look);
    renderScreenSettings();
    renderGestures();
    renderReadSettings();
    bind();
    introOrb();
    S.listeners.add(setupListener);
    setTimeout(() => GT.neural && GT.neural.load(), 400);
    const h = (location.hash || "").slice(1);
    showView(VIEWS.includes(h) ? h : "intro");
    updatePills();
    // Shared with js/capture.js.
    GT.labApi = {
      S, el, $, sleep, now, clamp, toast, requestFS, exitFS, startCamera, openStage, closeStage, stagePrompt, Cancelled,
      cmPerPx, distanceCm, screenWidthCm, downloadBlob,
    };
    // Test hook used by the automated smoke test (no effect in normal use).
    GT._lab = { S, R, processResult, layoutReader, endReading, manualNext, summarizeData, showView };
  }

  init();
})();
