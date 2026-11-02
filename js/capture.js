/* Record test data: about 3 minutes of looking at known spots, saved as one .zip so trackers can
   be replayed and scored offline (tools/bench.js) on held-out spots.

   Protocol
     grid1    25 dots in random order, 1.8 s each        → train
     pursuit  a light glides along 5 rows and 2 columns  → train (and lag check)
     grid2    25 dots again, sitting relaxed             → test: new posture, new order
     read     one page of text, read normally            → reading detector

   Saved per camera frame: face landmarks (478, int16), head transform, blendshapes, target.
   About 10 times a second also a JPEG of the eye band at full camera resolution, and twice a
   second a small face picture. Nothing leaves the computer unless the file is shared. */
(function () {
  "use strict";
  const GT = (window.GT = window.GT || {});

  const GRID = [];
  for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) GRID.push([0.06 + c * 0.22, 0.06 + r * 0.22]);
  const DOT_S = 1.8;
  const SETTLE_S = 0.6;
  const ROWS = [0.12, 0.31, 0.5, 0.69, 0.88];
  const COLS = [0.25, 0.75];

  function shuffled(arr, seed) {
    let s = seed >>> 0 || 1;
    const rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296;
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function b64(bytes) {
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function dataUrlBytes(url) {
    const bin = atob(url.slice(url.indexOf(",") + 1));
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  const r4 = (v) => Math.round(v * 1e4) / 1e4;

  async function run() {
    const L = GT.labApi;
    const { S, el, sleep, now, clamp, toast } = L;
    const fsPromise = L.requestFS();
    const ok = await L.startCamera();
    if (!ok) {
      L.exitFS();
      return;
    }
    await L.openStage("capture", fsPromise);
    const st = document.querySelector("#stage");
    const video = S.video;
    const W = video.videoWidth;
    const H = video.videoHeight;

    const frames = [];
    const images = []; // {name, url}
    let bsNames = null;
    const cur = { ph: null, tg: null, st: 0, dot: null };
    let n = 0;
    const eyeCanvas = document.createElement("canvas");
    const faceCanvas = document.createElement("canvas");
    const t0 = now();

    const fn = (fr) => {
      n++;
      if (!cur.ph) return;
      if (cur.dot && cur.ph !== "read") {
        // the light's real on-screen position this frame (gliding lights move continuously)
        const r = cur.dot.getBoundingClientRect();
        cur.tg = [(r.left + r.width / 2) / innerWidth, (r.top + r.height / 2) / innerHeight];
      }
      const rec = { t: r4(fr.t - t0), ph: cur.ph, tg: cur.tg ? [r4(cur.tg[0]), r4(cur.tg[1])] : null, st: cur.st ? 1 : 0, ok: fr.ok ? 1 : 0 };
      if (fr.ok) {
        const q = new Int16Array(fr.lm.length * 3);
        for (let i = 0; i < fr.lm.length; i++) {
          q[i * 3] = Math.round(fr.lm[i].x * 16384);
          q[i * 3 + 1] = Math.round(fr.lm[i].y * 16384);
          q[i * 3 + 2] = Math.round(fr.lm[i].z * 16384);
        }
        rec.lm = b64(new Uint8Array(q.buffer));
        if (fr.mat) rec.mat = Array.from(fr.mat, (v) => Math.round(v * 1e5) / 1e5);
        if (!bsNames && fr.bsNames) bsNames = fr.bsNames;
        if (fr.bsNames) rec.bs = bsNames.map((k) => Math.round((fr.bs[k] || 0) * 1000) / 1000);
        // Eye band at full camera resolution, ~10 per second.
        if (n % 3 === 0) {
          const P = GT.blazePrep;
          const plan = P.eyePatchPlan(P.landmarks2d(fr.lm, W, H));
          const g = plan.region;
          const x0 = clamp(g.x, 0, W);
          const y0 = clamp(g.y, 0, H);
          const w = clamp(g.x + g.w, 0, W) - x0;
          const h = clamp(g.y + g.h, 0, H) - y0;
          if (w > 8 && h > 8) {
            eyeCanvas.width = w;
            eyeCanvas.height = h;
            eyeCanvas.getContext("2d").drawImage(video, x0, y0, w, h, 0, 0, w, h);
            const name = `eyes/${String(frames.length).padStart(6, "0")}.jpg`;
            images.push({ name, url: eyeCanvas.toDataURL("image/jpeg", 0.9) });
            rec.eyes = name;
            rec.eyesBox = [x0, y0, w, h];
          }
        }
        // Small face picture, twice a second (for face-based models later).
        if (n % 15 === 0) {
          let x0 = Infinity;
          let y0 = Infinity;
          let x1 = -Infinity;
          let y1 = -Infinity;
          for (const p of fr.lm) {
            x0 = Math.min(x0, p.x * W);
            y0 = Math.min(y0, p.y * H);
            x1 = Math.max(x1, p.x * W);
            y1 = Math.max(y1, p.y * H);
          }
          const pad = 0.15 * (x1 - x0);
          x0 = clamp(Math.floor(x0 - pad), 0, W);
          y0 = clamp(Math.floor(y0 - pad), 0, H);
          x1 = clamp(Math.ceil(x1 + pad), 0, W);
          y1 = clamp(Math.ceil(y1 + pad), 0, H);
          const s = Math.min(1, 256 / (x1 - x0));
          faceCanvas.width = Math.round((x1 - x0) * s);
          faceCanvas.height = Math.round((y1 - y0) * s);
          faceCanvas.getContext("2d").drawImage(video, x0, y0, x1 - x0, y1 - y0, 0, 0, faceCanvas.width, faceCanvas.height);
          const name = `face/${String(frames.length).padStart(6, "0")}.jpg`;
          images.push({ name, url: faceCanvas.toDataURL("image/jpeg", 0.85) });
          rec.face = name;
          rec.faceBox = [x0, y0, x1 - x0, y1 - y0, r4(s)];
        }
      }
      frames.push(rec);
    };

    const progress = el("div", { class: "cap-progress" }, el("i"));
    const setProgress = (p) => (progress.firstChild.style.width = `${Math.round(p * 100)}%`);
    const check = () => {
      if (S.stageCancelled) throw new L.Cancelled();
    };

    async function dotAt(x, y) {
      const dot = el("div", { class: "cal-dot" });
      dot.style.left = x * 100 + "%";
      dot.style.top = y * 100 + "%";
      dot.style.setProperty("--dur", DOT_S + "s");
      st.append(dot);
      cur.dot = dot;
      cur.tg = [x, y];
      cur.st = 0;
      await sleep(SETTLE_S * 1000);
      cur.st = 1;
      await sleep((DOT_S - SETTLE_S) * 1000);
      cur.st = 0;
      cur.dot = null;
      dot.remove();
    }
    async function glide(from, to, dur) {
      const dot = el("div", { class: "cal-dot glide" });
      dot.style.left = from[0] * 100 + "%";
      dot.style.top = from[1] * 100 + "%";
      st.append(dot);
      cur.st = 0;
      cur.dot = dot;
      await sleep(600);
      dot.style.transition = `left ${dur}s linear, top ${dur}s linear`;
      dot.style.left = to[0] * 100 + "%";
      dot.style.top = to[1] * 100 + "%";
      await sleep(250);
      cur.st = 1;
      await sleep(dur * 1000 - 250);
      cur.st = 0;
      cur.dot = null;
      dot.remove();
    }

    const steps = 25 + ROWS.length + COLS.length + 25;
    let done = 0;
    const tick = () => setProgress(++done / steps);

    S.listeners.add(fn);
    S.calibrating = true; // keep the neural model running so its timing is realistic
    try {
      const go = await L.stagePrompt(
        "Record test data.",
        "About 3 minutes. Sit how you read, and keep your eyes on the light the whole time.",
        [["start", "Start"], ["cancel", "Cancel"]]
      );
      if (go !== "start") throw new L.Cancelled();
      st.append(progress);
      await sleep(400);

      cur.ph = "grid1";
      for (const [x, y] of shuffled(GRID, 11)) {
        await dotAt(x, y);
        check();
        tick();
      }
      cur.ph = "pursuit";
      for (const y of ROWS) {
        await glide([0.1, y], [0.9, y], 3);
        check();
        tick();
      }
      for (const x of COLS) {
        await glide([x, 0.1], [x, 0.9], 2.6);
        check();
        tick();
      }
      cur.ph = null;
      const go2 = await L.stagePrompt("Now sit back a little.", "Relax the way you would while reading. Same thing again.", [["start", "Continue"], ["cancel", "Stop"]]);
      if (go2 !== "start") throw new L.Cancelled();
      await sleep(400);
      cur.ph = "grid2";
      for (const [x, y] of shuffled(GRID, 97)) {
        await dotAt(x, y);
        check();
        tick();
      }
      cur.ph = null;

      // One page of reading.
      const go3 = await L.stagePrompt("Last: read one page.", "Read normally. Press → when you reach the end.", [["start", "Start"], ["skip", "Skip"], ["cancel", "Stop"]]);
      if (go3 === "cancel") throw new L.Cancelled();
      let reading = null;
      if (go3 === "start") reading = await readPage(st, cur);
      check();

      progress.remove();
      const card = el("div", { class: "stage-card" }, el("h2", { text: "Saving…" }));
      st.append(card);
      await sleep(50);
      S.listeners.delete(fn);

      const meta = {
        kind: "gazeturn-capture",
        version: 1,
        at: new Date().toISOString(),
        camera: { w: W, h: H },
        viewport: { w: innerWidth, h: innerHeight },
        screen: { w: screen.width, h: screen.height, dpr: window.devicePixelRatio || 1 },
        screenWidthCm: L.screenWidthCm(),
        cmPerPx: L.cmPerPx(),
        distanceCm: L.distanceCm(),
        distanceMode: S.settings.distanceMode,
        userAgent: navigator.userAgent,
        featVersion: GT.FEATURE_VERSION,
        landmarkScale: 16384,
        bsNames,
        protocol: { grid: GRID, dotS: DOT_S, settleS: SETTLE_S, rows: ROWS, cols: COLS },
        reading,
        tracker: S.calib ? { engine: S.calib.engine, accs: Object.fromEntries(Object.entries(S.calib.accs || {}).map(([k, v]) => [k, v.accDeg])) } : null,
        neural: GT.neural ? { state: GT.neural.state, backend: GT.neural.backend } : null,
        frames: frames.length,
      };
      const files = [
        { name: "meta.json", data: JSON.stringify(meta) },
        { name: "frames.json", data: JSON.stringify(frames) },
      ];
      for (const im of images) files.push({ name: im.name, data: dataUrlBytes(im.url) });
      const blob = new Blob(GT.zip.buildParts(files), { type: "application/zip" });
      const name = `gazeturn-capture-${meta.at.slice(0, 16).replace(/[:T]/g, "-")}.zip`;
      L.downloadBlob(name, blob);
      const mb = (blob.size / 1e6).toFixed(1);
      card.innerHTML = "";
      card.append(
        el("h2", { text: "Saved." }),
        el("p", { text: `${name} · ${mb} MB · ${frames.length} frames · ${images.length} pictures. It's in your Downloads folder.` }),
        el("div", { class: "actions" }, el("button", { class: "btn", text: "Done", onclick: () => finish() }))
      );
      GT._lab.lastCapture = { name, size: blob.size, frames: frames.length, images: images.length, meta };
    } catch (e) {
      if (!(e instanceof L.Cancelled)) {
        console.error(e);
        toast("Recording failed: " + (e.message || e), 7000);
      } else toast("Stopped. Nothing was saved.");
      finish();
    } finally {
      S.listeners.delete(fn);
      S.calibrating = false;
    }
    function finish() {
      L.closeStage();
    }
  }

  /** Show one page of the sample story; record its line layout; wait for → (max 2 minutes). */
  function readPage(st, cur) {
    const { el } = GT.labApi;
    return new Promise((resolve) => {
      const fs = 22;
      const page = el("div", { class: "cap-page" });
      page.style.fontSize = fs + "px";
      page.style.width = Math.min(fs * 31, innerWidth - 80) + "px";
      const paras = GT.SAMPLE_BOOK.paragraphs;
      page.append(el("h2", { text: GT.SAMPLE_BOOK.title }));
      for (let i = 0; i < 9; i++) page.append(el("p", { class: i === 0 ? "first" : null, text: paras[i] }));
      st.append(page);
      // Trim to what fits on screen, then record each line's box.
      requestAnimationFrame(() => {
        const maxY = innerHeight - 40;
        for (const p of Array.from(page.querySelectorAll("p")).reverse()) if (p.getBoundingClientRect().bottom > maxY) p.remove();
        const lines = [];
        const range = document.createRange();
        const walker = document.createTreeWalker(page, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
          if (!node.textContent.trim()) continue;
          range.selectNodeContents(node);
          for (const r of range.getClientRects()) {
            if (r.width < 1) continue;
            const y = r.top;
            let ln = lines.find((l) => Math.abs(l.top - y) < 4);
            if (!ln) lines.push({ top: y, bottom: r.bottom, left: r.left, right: r.right });
            else {
              ln.left = Math.min(ln.left, r.left);
              ln.right = Math.max(ln.right, r.right);
            }
          }
        }
        lines.sort((a, b) => a.top - b.top);
        const norm = lines.map((l) => [l.left / innerWidth, l.top / innerHeight, l.right / innerWidth, l.bottom / innerHeight].map((v) => Math.round(v * 1e4) / 1e4));
        cur.ph = "read";
        cur.tg = null;
        cur.dot = null;
        const t0 = performance.now();
        const done = (how) => {
          window.removeEventListener("keydown", key);
          clearTimeout(timer);
          cur.ph = null;
          page.remove();
          resolve({ lines: norm, fontSize: fs, ms: Math.round(performance.now() - t0), ended: how });
        };
        const key = (e) => {
          if (e.key === "ArrowRight" || e.key === " ") {
            e.preventDefault();
            done("key");
          }
        };
        window.addEventListener("keydown", key);
        const timer = setTimeout(() => done("timeout"), 120000);
      });
    });
  }

  GT.capture = { run };
})();
