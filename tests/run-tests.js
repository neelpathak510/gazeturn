/* Run with:  node tests/run-tests.js
   No dependencies. Loads the same files the browser loads. */
"use strict";
const path = require("path");
const core = (f) => require(path.join(__dirname, "..", "js", "core", f));
core("stats.js");
core("onefilter.js");
core("ridge.js");
core("features.js");
core("detector.js");
core("blazeprep.js");
const GT = globalThis.GT;
const { simulate, zoneFor, rng, gauss } = require("./simulate.js");

let failures = 0;
function check(name, cond, detail = "") {
  console.log(`${cond ? "  ok " : "FAIL "} ${name}${detail ? "  (" + detail + ")" : ""}`);
  if (!cond) failures++;
}

// ---------- ridge regression ----------
(function testRidge() {
  console.log("Ridge regression");
  const r = rng(7);
  const X = [];
  const Y = [];
  const groups = [];
  for (let g = 0; g < 13; g++) {
    const tx = r();
    const ty = r();
    for (let k = 0; k < 20; k++) {
      // features = noisy linear mix of the target plus junk columns
      const f = [tx + 0.01 * gauss(r), ty + 0.01 * gauss(r), tx * 0.5 - ty + 0.01 * gauss(r), r(), 3];
      X.push(f);
      Y.push([tx, ty]);
      groups.push(g);
    }
  }
  const fit = GT.fitGaze(X, Y, groups, { weights: [30, 20] });
  const p = GT.predict(fit.model, [0.25, 0.75, 0.125 - 0.75, 0.5, 3]);
  check("recovers a linear map", Math.abs(p[0] - 0.25) < 0.03 && Math.abs(p[1] - 0.75) < 0.03, p.map((v) => v.toFixed(3)).join(", "));
  check("cross-validated error is small", fit.cvError < 1.0, `${fit.cvError.toFixed(3)} cm, lambda ${fit.lambda}`);
  check("constant column does not break the fit", Number.isFinite(p[0]) && Number.isFinite(p[1]));
  check("stretch correction stays sensible", fit.stretch.every((b) => b >= 0.95 && b <= 1.4), fit.stretch.map((b) => b.toFixed(2)).join(", "));
})();

// ---------- neural preprocessing (port of WebEyeTrack) ----------
(function testBlazePrep() {
  console.log("Neural preprocessing vs the original WebEyeTrack code (real face landmarks)");
  const P = GT.blazePrep;
  const F = JSON.parse(require("fs").readFileSync(path.join(__dirname, "fixtures", "face-landmarks.json")));
  const E = F.expected_from_original_webeyetrack;
  const l2 = P.landmarks2d(F.lm, F.W, F.H);
  const w = P.faceWidthCm(l2);
  check("face width matches", Math.abs(w - E.faceWidthCm) < 0.01, w.toFixed(4));
  const hv = P.headVector({ data: F.M });
  check("head vector matches", hv.every((v, i) => Math.abs(v - E.headVector[i]) < 1e-3), hv.map((v) => v.toFixed(4)).join(", "));
  const o = P.faceOrigin3D(F.lm, { data: F.M }, F.W, F.H, E.faceWidthCm, 60);
  check("3D eye position matches", o.every((v, i) => Math.abs(v - E.faceOrigin3D[i]) < 0.05), o.map((v) => v.toFixed(2)).join(", "));
  const H = P.homography([[0, 0], [0, 10], [10, 10], [10, 0]], [[3, 4], [2, 30], [40, 33], [37, 5]]);
  const q = P.applyH(H, 10, 10);
  check("homography maps corners", Math.abs(q[0] - 40) < 1e-6 && Math.abs(q[1] - 33) < 1e-6);
  const plan = P.eyePatchPlan(l2);
  check("eye band found", plan.eh > 20 && plan.eh < 300, `band ${plan.eh} rows`);
})();

// ---------- recording file format ----------
(function testZip() {
  console.log("Recording file (zip)");
  const Z = require(path.join(__dirname, "..", "js", "zip.js"));
  const img = new Uint8Array(5000).map((_, i) => (i * 37) & 255);
  const parts = Z.buildParts([{ name: "meta.json", data: '{"a":1}' }, { name: "eyes/000001.jpg", data: img }]);
  const total = parts.reduce((n, p) => n + p.length, 0);
  const buf = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    buf.set(p, o);
    o += p.length;
  }
  const files = Z.read(buf);
  check("zip round trip", new TextDecoder().decode(files.get("meta.json")) === '{"a":1}' && files.get("eyes/000001.jpg").every((v, i) => v === img[i]));
  check("crc32 matches the standard value", Z.crc32(new TextEncoder().encode("123456789")) === 0xcbf43926);
})();

// ---------- off-screen detection ----------
(function testOffScreen() {
  console.log("Off-screen detection");
  const r = rng(9);
  const X = [];
  for (let i = 0; i < 300; i++) {
    const v = Array(25).fill(0);
    for (let j = 0; j < 13; j++) v[j] = 0.3 + 0.4 * r();
    X.push(v);
  }
  const ranges = GT.featureRanges(X);
  const inside = X[5];
  const below = inside.slice();
  below[5] = 0.0; // eyelids nearly shut, as when looking down at a keyboard
  below[11] = 1.2; // looking far down
  check("on-screen sample is not flagged", GT.outOfRange(inside, ranges) < 0.35, GT.outOfRange(inside, ranges).toFixed(2));
  check("looking at the keyboard is flagged", GT.outOfRange(below, ranges) > 0.35, GT.outOfRange(below, ranges).toFixed(2));
})();

// ---------- One Euro filter ----------
(function testFilter() {
  console.log("One Euro filter");
  const r = rng(3);
  const f = new GT.OneEuroFilter({ minCutoff: 1, beta: 2.5 });
  let rawVar = 0;
  let outVar = 0;
  for (let k = 0; k < 300; k++) {
    const raw = 0.5 + 0.03 * gauss(r);
    const out = f.filter(raw, k / 30);
    if (k > 30) {
      rawVar += (raw - 0.5) ** 2;
      outVar += (out - 0.5) ** 2;
    }
  }
  check("reduces jitter while still", outVar < rawVar * 0.3, `variance ×${(outVar / rawVar).toFixed(2)}`);
  let k = 300;
  let out = 0;
  for (let j = 0; j < 6; j++, k++) out = f.filter(0.05, k / 30); // a 0.45 jump (a line return)
  check("follows a big jump within 0.2 s", out < 0.15, `at ${out.toFixed(3)} after 6 frames`);
})();

// ---------- gestures and dwell ----------
(function testGestures() {
  console.log("Gestures and dwell");
  const g = new GT.HoldGesture(0.5, 0.3);
  let fired = 0;
  for (let k = 0; k < 60; k++) if (g.update(k / 30, k > 10 && k < 25 ? 0.8 : 0.1)) fired++;
  check("held gesture fires once", fired === 1);
  const g2 = new GT.HoldGesture(0.5, 0.3);
  let fired2 = 0;
  for (let k = 0; k < 60; k++) if (g2.update(k / 30, k > 10 && k < 15 ? 0.8 : 0.1)) fired2++;
  check("short twitch does not fire", fired2 === 0);
  const d = new GT.DwellDetector(0.9);
  d.setZones([{ id: "next", x0: 0.8, x1: 1, y0: 0, y1: 1 }]);
  let firedAt = null;
  for (let k = 0; k < 90; k++) {
    const res = d.update(k / 30, 0.9, 0.5, true);
    if (res.fired && firedAt === null) firedAt = k / 30;
  }
  check("dwell fires after about 0.9 s", firedAt !== null && Math.abs(firedAt - 0.93) < 0.1, `fired at ${firedAt}`);
})();

// ---------- simulated reading ----------
function runSim(simOpts, params = {}, method = "lookup") {
  const sim = simulate(simOpts);
  const acc = (simOpts.biasDeg || 1) + 0.5 * (simOpts.sigmaDeg || 1);
  const zone = (i) => (method === "endzone" && sim.pageInfo[i] ? zoneFor(sim.pageInfo[i], acc, sim.degX, sim.degY) : null);
  const run = new GT.ShadowRunner(Object.assign({ useLayout: true }, params), {}, method);
  run.setPage(sim.linesPerPage, zone(0));
  // Filter in viewport units like the app does (column ≈ half the viewport width).
  const fx = new GT.OneEuroFilter({ minCutoff: 1, beta: 2.5 });
  const fy = new GT.OneEuroFilter({ minCutoff: 1, beta: 2.5 });
  let mi = 0;
  // Optional auto re-centre, as in the site: known targets at each page end (last word,
  // just before the press) and page start (first word, just after) nudge a running offset.
  const recentre = !!simOpts.recentre;
  const gain = 0.5;
  let ox = 0;
  let oy = 0;
  const hist = [];
  const pendingStarts = [];
  // xq: which quantile of x to compare (0.5 at the last word; 0.15 at a page start, where the
  // eyes are already moving right along the first line).
  const correct = (t0, t1, tx, ty, xq) => {
    const xs = [];
    const ys = [];
    for (const h of hist) if (h.t >= t0 && h.t <= t1) {
      xs.push(h.x);
      ys.push(h.y);
    }
    if (xs.length < 4) return;
    const ex = tx - GT.quantile(xs, xq);
    const ey = ty - GT.median(ys);
    if (Math.abs(ex) > 0.15 || Math.abs(ey) > 0.15) return; // looking elsewhere: skip
    ox += gain * ex;
    oy += gain * ey;
  };
  for (const s of sim.samples) {
    while (mi < sim.manual.length && sim.manual[mi] <= s.t) {
      const M = sim.manual[mi];
      if (recentre) {
        const info = sim.pageInfo[mi];
        correct(M - 0.6, M - 0.15, info.lastLineEnd - 0.04, info.lastLineY, 0.5);
        pendingStarts.push({ t: M + 0.95, from: M + 0.35, x: 0.05, y: 0.02 + 0.5 / sim.linesPerPage * 0.96 });
      }
      run.manualNext(M, sim.linesPerPage, zone(mi + 1));
      mi++;
    }
    while (pendingStarts.length && pendingStarts[0].t <= s.t) {
      const ps = pendingStarts.shift();
      correct(ps.from, ps.t, ps.x, ps.y, 0.15);
    }
    let x = s.x + ox;
    let y = s.y + oy;
    if (s.valid) {
      hist.push({ t: s.t, x, y });
      if (hist.length > 120) hist.shift();
      x = (fx.filter(0.25 + x * 0.5, s.t) - 0.25) / 0.5;
      y = (fy.filter(0.06 + y * 0.88, s.t) - 0.06) / 0.88;
    }
    run.sample({ t: s.t, x, y, valid: s.valid });
  }
  const pages = [];
  for (let i = 0; i < sim.manual.length; i++) {
    pages.push({ i, total: sim.linesPerPage, t0: i ? sim.manual[i - 1] : 0, t1: sim.manual[i], exit: "manual" });
  }
  const sum = GT.summarize({
    kind: "shadow", trigger: "auto", durationS: sim.tEnd, pages,
    lineTimes: run.lineTimes, evalResult: run.result(), detStats: run.det.stats,
  });
  // Early predictions (before the press) matter for the last-word method: they mean the page
  // would have turned before the reader finished.
  sum.earlyBig = run.result().hits.filter((h) => h.offset < -1).length;
  return sum;
}

function simTable(title, method, simExtra) {
  console.log(title);
  console.log("  jitter  drift | turns caught | false turns | >1 s early");
  const rows = [[0.5, 0.5], [1, 1], [1.5, 2], [2, 3], [3, 4]];
  const table = [];
  for (const [sigma, bias] of rows) {
    let hits = 0, misses = 0, falses = 0, early = 0;
    const ratios = [];
    for (const seed of [11, 22, 33]) {
      const s = runSim(Object.assign({ sigmaDeg: sigma, biasDeg: bias, seed }, simExtra), {}, method);
      hits += s.hits;
      misses += s.misses;
      falses += s.falseTurns;
      early += s.earlyBig;
      ratios.push(s.lineRatio);
    }
    const recall = hits / (hits + misses);
    table.push({ sigma, bias, recall, falses, early, ratio: GT.mean(ratios) });
    console.log(
      `  ${sigma.toFixed(1)}°   ${bias.toFixed(1)}°  |   ${(recall * 100).toFixed(0).padStart(3)}%       |    ${String(falses).padStart(2)} / 30    |  ${early}`
    );
  }
  return table;
}

(function testSimulation() {
  const look = simTable("Simulated reading, look-up method (10 pages × 22 lines, 3 seeds per row)", "lookup", {});
  check("look-up: clean tracking (1°) catches ≥ 90% of turns", look[1].recall >= 0.9, `${(look[1].recall * 100).toFixed(0)}%`);
  check("look-up: clean tracking (1°) has ≤ 1 false turn in 30 pages", look[1].falses <= 1, `${look[1].falses}`);
  check("line counting at 1° is within 25%", Math.abs(look[1].ratio - 1) < 0.25, `ratio ${look[1].ratio.toFixed(2)}`);

  const lw = simTable("Simulated reading, last-word method (rests ~0.7 s on the last word)", "endzone", { lingerLast: 0.7 });
  check("last word: 1° catches ≥ 90% of turns", lw[1].recall >= 0.9, `${(lw[1].recall * 100).toFixed(0)}%`);
  check("last word: 1° has ≤ 2 false turns in 30 pages", lw[1].falses <= 2, `${lw[1].falses}`);

  const hy = simTable("Both (last word or look up), reader rests on the last word", "hybrid", { lingerLast: 0.7 });
  check("both: ≥ 95% caught up to 1.5°/2°", hy[2].recall >= 0.95, `${(hy[2].recall * 100).toFixed(0)}%`);
  check("both: ≤ 3 false turns in 30 pages at 1.5°/2°", hy[2].falses <= 3, `${hy[2].falses}`);

  const hyn = simTable("Both, reader who doesn't rest on the last word (just looks up)", "hybrid", {});
  check("both, no resting: still ≥ 95% at 1°", hyn[1].recall >= 0.95, `${(hyn[1].recall * 100).toFixed(0)}%`);

  const hyr = simTable("Both + auto re-centre", "hybrid", { lingerLast: 0.7, recentre: true });
  void hyr;
})();

// ---------- replay matches live ----------
(function testReplay() {
  console.log("Replay");
  const sim = simulate({ sigmaDeg: 1, biasDeg: 1, seed: 5 });
  const run = new GT.ShadowRunner({ useLayout: true });
  run.setPage(sim.linesPerPage);
  const data = { samples: { t: [], px: [], py: [], v: [] }, events: [], pages: [{ total: sim.linesPerPage }], rects: [] };
  let mi = 0;
  for (const s of sim.samples) {
    while (mi < sim.manual.length && sim.manual[mi] <= s.t) {
      run.manualNext(sim.manual[mi], sim.linesPerPage);
      data.events.push({ t: sim.manual[mi], type: "manualNext", total: sim.linesPerPage });
      mi++;
    }
    run.sample(s);
    data.samples.t.push(s.t);
    data.samples.px.push(s.x);
    data.samples.py.push(s.y);
    data.samples.v.push(s.valid ? 1 : 0);
  }
  const live = run.result();
  const rep = GT.replayShadow(data, { useLayout: true });
  check(
    "re-scoring a session reproduces the live result",
    rep.evalResult.hits.length === live.hits.length &&
      rep.evalResult.misses.length === live.misses.length &&
      rep.evalResult.falseTurns.length === live.falseTurns.length,
    `${live.hits.length}/${live.misses.length}/${live.falseTurns.length}`
  );
})();

console.log(failures ? `\n${failures} check(s) failed` : "\nAll checks passed");
process.exit(failures ? 1 : 0);
