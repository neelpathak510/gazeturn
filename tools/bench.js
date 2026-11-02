#!/usr/bin/env node
/* GazeTurn test bench: replays a recording (from "Record test data") through every tracker and
   scores each on dots it never trained on. Uses the site's own code from js/core, so a result
   here is what the site would do.

   Usage:  node tools/bench.js <gazeturn-capture-….zip> [--no-neural] [--json report.json]

   Two tests:
     same sitting   train on 13 of the first 25 dots + the gliding lights, test on the other 12
     new posture    train on all of the first part,  test on the second 25 dots (sitting back)
   Scores are in degrees of visual angle: per frame (what the light does) and per dot (the
   median of a dot's frames, i.e. the steady offset once jitter is averaged out). */
"use strict";
const fs = require("fs");
const path = require("path");

const SITE = path.join(__dirname, "..", "js");
for (const f of ["core/stats.js", "core/onefilter.js", "core/ridge.js", "core/features.js", "core/detector.js", "core/blazeprep.js", "zip.js"]) {
  require(path.join(SITE, f));
}
const GT = globalThis.GT;
const median = GT.median;
const mean = GT.mean;

// ---------------------------------------------------------------- load
function loadCapture(file) {
  const files = GT.zip.read(new Uint8Array(fs.readFileSync(file)));
  const dec = new TextDecoder();
  const meta = JSON.parse(dec.decode(files.get("meta.json")));
  const frames = JSON.parse(dec.decode(files.get("frames.json")));
  if (meta.kind !== "gazeturn-capture") throw new Error("not a GazeTurn recording");
  return { meta, frames, files };
}

function decodeLandmarks(b64, scale) {
  const buf = Buffer.from(b64, "base64");
  const q = new Int16Array(buf.buffer, buf.byteOffset, buf.length / 2);
  const lm = new Array(q.length / 3);
  for (let i = 0; i < lm.length; i++) lm[i] = { x: q[i * 3] / scale, y: q[i * 3 + 1] / scale, z: q[i * 3 + 2] / scale };
  return lm;
}

/** Rebuild MediaPipe-shaped results and run the site's feature code on every frame. */
function prepare(cap) {
  const { meta, frames } = cap;
  const W = meta.camera.w;
  const H = meta.camera.h;
  for (const f of frames) {
    if (!f.ok || !f.lm) continue;
    const lm = decodeLandmarks(f.lm, meta.landmarkScale || 16384);
    const res = {
      faceLandmarks: [lm],
      faceBlendshapes: meta.bsNames && f.bs ? [{ categories: meta.bsNames.map((n, i) => ({ categoryName: n, score: f.bs[i] })) }] : [],
      facialTransformationMatrixes: f.mat ? [{ rows: 4, columns: 4, data: f.mat }] : [],
    };
    const fr = GT.extractFrame(res, W, H);
    if (!fr.ok) continue;
    f.fr = fr;
    f.vec = GT.gazeVector(fr);
  }
  const opens = frames.filter((f) => f.fr && f.ph === "grid1" && f.st).map((f) => f.fr.open);
  const baseOpen = median(opens);
  for (const f of frames) if (f.fr) f.blink = f.fr.blink >= 0.6 || f.fr.open < 0.4 * baseOpen;
}

// ---------------------------------------------------------------- neural (BlazeGaze), cached
async function runNeural(cap, file) {
  const cacheFile = file.replace(/\.zip$/i, "") + ".nn.json";
  if (fs.existsSync(cacheFile)) {
    const c = JSON.parse(fs.readFileSync(cacheFile));
    for (const [i, nn] of Object.entries(c)) cap.frames[+i].nn = nn;
    return { cached: true, n: Object.keys(c).length };
  }
  const tf = require("@tensorflow/tfjs");
  require("@tensorflow/tfjs-backend-wasm");
  await tf.setBackend("wasm");
  await tf.ready();
  const jpeg = require("jpeg-js");
  const dir = path.join(__dirname, "..", "vendor", "blazegaze");
  const mj = JSON.parse(fs.readFileSync(path.join(dir, "model.json")));
  const wb = fs.readFileSync(path.join(dir, "group1-shard1of1.bin"));
  const model = await tf.loadLayersModel(
    tf.io.fromMemory({ modelTopology: mj.modelTopology, weightSpecs: mj.weightsManifest[0].weights, weightData: wb.buffer.slice(wb.byteOffset, wb.byteOffset + wb.byteLength) })
  );
  const P = GT.blazePrep;
  const { meta, frames, files } = cap;
  const W = meta.camera.w;
  const H = meta.camera.h;
  const widths = [];
  for (const f of frames) {
    if (!f.fr || widths.length >= 60) continue;
    const w = P.faceWidthCm(P.landmarks2d(f.fr.lm, W, H));
    if (w > 5 && w < 30) widths.push(w);
  }
  const widthCm = median(widths);
  let z = 60;
  const cache = {};
  const t0 = Date.now();
  let done = 0;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    if (!f.fr || !f.eyes || !f.mat) continue;
    const jp = files.get(f.eyes);
    if (!jp) continue;
    const img = jpeg.decode(jp, { useTArray: true, formatAsRGBA: true });
    const [bx, by] = f.eyesBox;
    const plan = P.eyePatchPlan(P.landmarks2d(f.fr.lm, W, H));
    const patch = P.eyePatch(plan, { data: img.data, width: img.width, height: img.height, x: bx, y: by }, W, H);
    const m = { data: f.mat };
    const origin = P.faceOrigin3D(f.fr.lm, m, W, H, widthCm, z);
    if (!origin.every(Number.isFinite)) continue;
    z = Math.max(20, Math.min(150, origin[2]));
    const head = P.headVector(m);
    const out = tf.tidy(() => {
      const x = tf.tensor(patch, [1, 128, 512, 4], "int32").slice([0, 0, 0, 0], [1, 128, 512, 3]).toFloat().div(255);
      return model.predict([x, tf.tensor2d([head]), tf.tensor2d([origin])]);
    });
    const v = out.dataSync();
    out.dispose();
    f.nn = [v[0], v[1]];
    cache[i] = f.nn;
    if (++done % 200 === 0) process.stderr.write(`  neural ${done} frames (${((Date.now() - t0) / done).toFixed(0)} ms each)\n`);
  }
  fs.writeFileSync(cacheFile, JSON.stringify(cache));
  return { cached: false, n: done, ms: (Date.now() - t0) / Math.max(1, done) };
}

// ---------------------------------------------------------------- evaluation
function makeUnits(meta) {
  const c = meta.cmPerPx;
  const vw = meta.viewport.w;
  const vh = meta.viewport.h;
  const D = meta.distanceCm || 50;
  const toDeg = (cm) => (Math.atan2(Math.abs(cm), D) * 180) / Math.PI;
  return {
    weights: [vw * c, vh * c],
    err: (p, t) => toDeg(Math.hypot((p[0] - t[0]) * vw * c, (p[1] - t[1]) * vh * c)),
    dx: (p, t) => Math.sign(p[0] - t[0]) * toDeg((p[0] - t[0]) * vw * c),
    dy: (p, t) => Math.sign(p[1] - t[1]) * toDeg((p[1] - t[1]) * vh * c),
  };
}

const key = (t) => `${t[0].toFixed(3)},${t[1].toFixed(3)}`;

function samples(frames, ph, needNN) {
  return frames.filter((f) => f.ph === ph && f.st && f.tg && f.fr && !f.blink && (!needNN || f.nn));
}

/** Score a prediction function on test samples grouped by dot. */
function score(test, predict, U) {
  const perFrame = [];
  const byDot = new Map();
  const dxs = [];
  const dys = [];
  const bands = [[], [], []];
  for (const s of test) {
    const p = predict(s);
    if (!p) continue;
    const e = U.err(p, s.tg);
    perFrame.push(e);
    dxs.push(U.dx(p, s.tg));
    dys.push(U.dy(p, s.tg));
    bands[Math.min(2, Math.floor(s.tg[1] * 3))].push(e);
    const k = key(s.tg);
    if (!byDot.has(k)) byDot.set(k, { t: s.tg, ps: [] });
    byDot.get(k).ps.push(p);
  }
  const dotErr = [];
  for (const d of byDot.values()) {
    if (d.ps.length < 3) continue;
    dotErr.push(U.err([median(d.ps.map((p) => p[0])), median(d.ps.map((p) => p[1]))], d.t));
  }
  return {
    n: perFrame.length,
    frame: median(perFrame),
    frameMean: mean(perFrame),
    dot: median(dotErr),
    biasX: median(dxs),
    biasY: median(dys),
    top: median(bands[0]),
    mid: median(bands[1]),
    bottom: median(bands[2]),
  };
}

function vecFor(engine, f) {
  if (engine === "classic") return f.vec;
  if (!f.nn) return null;
  const nv = GT.neuralVector({ nn: f.nn });
  return engine === "neural" ? nv : f.vec.concat(nv);
}

function groupOf(f) {
  if (f.ph === "pursuit") return 1000 + Math.round(f.tg[1] * 20) * 3 + Math.min(2, Math.floor(f.tg[0] * 3));
  return key(f.tg);
}

/** The trackers to compare. Each: name, engine, and an optional fit override. */
const METHODS = [
  { name: "classic (site)", engine: "classic" },
  { name: "neural (site)", engine: "neural" },
  { name: "fusion (site)", engine: "fusion" },
];

function fitMethod(m, train, U) {
  const rows = train.filter((f) => vecFor(m.engine, f));
  if (rows.length < 40) return null;
  const fit = (m.fit || GT.fitGaze)(rows.map((f) => vecFor(m.engine, f)), rows.map((f) => f.tg), rows.map(groupOf), { weights: U.weights });
  return (f) => {
    const v = vecFor(m.engine, f);
    return v ? GT.predict(fit.model, v) : null;
  };
}

function evaluate(cap, opts) {
  const { meta, frames } = cap;
  const U = makeUnits(meta);
  const needNN = opts.neural; // compare everything on the same frames when the neural model ran
  const g1 = samples(frames, "grid1", needNN);
  const pur = samples(frames, "pursuit", needNN);
  const g2 = samples(frames, "grid2", needNN);
  const dots = Array.from(new Set(g1.map((f) => key(f.tg))));
  const half = new Set(dots.filter((_, i) => i % 2 === 0));
  const tests = {
    "same sitting": { train: g1.filter((f) => half.has(key(f.tg))).concat(pur), test: g1.filter((f) => !half.has(key(f.tg))) },
    "new posture": { train: g1.concat(pur), test: g2 },
  };
  const out = [];
  for (const m of METHODS) {
    if (m.engine !== "classic" && !opts.neural) continue;
    const row = { method: m.name };
    for (const [tname, T] of Object.entries(tests)) {
      if (!T.test.length) continue;
      const pred = fitMethod(m, T.train, U);
      row[tname] = pred ? score(T.test, pred, U) : null;
    }
    out.push(row);
  }
  return { results: out, counts: { grid1: g1.length, pursuit: pur.length, grid2: g2.length } };
}

// ---------------------------------------------------------------- report
function printReport(cap, ev, info) {
  const m = cap.meta;
  const f = (v) => (Number.isFinite(v) ? v.toFixed(2).padStart(5) : "   – ");
  const sg = (v) => (Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(1) : "–");
  console.log(`\nRecording ${m.at}  camera ${m.camera.w}×${m.camera.h}  window ${m.viewport.w}×${m.viewport.h}  distance ${Math.round(m.distanceCm)} cm`);
  console.log(`Frames: ${cap.frames.length} (face found in ${cap.frames.filter((x) => x.fr).length}); used: grid1 ${ev.counts.grid1}, pursuit ${ev.counts.pursuit}, grid2 ${ev.counts.grid2}`);
  if (info.neural) console.log(`Neural: ${info.neural.cached ? "from cache" : `${info.neural.n} frames at ${info.neural.ms.toFixed(0)} ms`}`);
  for (const t of ["same sitting", "new posture"]) {
    console.log(`\n${t.toUpperCase()}  (degrees; lower is better)`);
    console.log("  method              per frame  per dot   top   mid  bottom   bias x  bias y");
    for (const r of ev.results) {
      const s = r[t];
      if (!s) continue;
      console.log(`  ${r.method.padEnd(18)}   ${f(s.frame)}    ${f(s.dot)}  ${f(s.top)} ${f(s.mid)} ${f(s.bottom)}   ${sg(s.biasX).padStart(5)}  ${sg(s.biasY).padStart(5)}`);
    }
  }
  console.log("\nbias: + means the gaze lands right of / below the target.");
}

async function main() {
  const args = process.argv.slice(2);
  const file = args.find((a) => !a.startsWith("--"));
  if (!file) {
    console.error("usage: node tools/bench.js <capture.zip> [--no-neural] [--json out.json]");
    process.exit(2);
  }
  const cap = loadCapture(file);
  prepare(cap);
  const info = {};
  const neural = !args.includes("--no-neural");
  if (neural) info.neural = await runNeural(cap, file);
  const ev = evaluate(cap, { neural: neural && cap.frames.some((x) => x.nn) });
  printReport(cap, ev, info);
  const j = args.indexOf("--json");
  if (j >= 0 && args[j + 1]) fs.writeFileSync(args[j + 1], JSON.stringify({ meta: cap.meta, ...ev, info }, null, 1));
}

module.exports = { loadCapture, prepare, runNeural, evaluate, score, makeUnits, METHODS, vecFor, groupOf, samples };
if (require.main === module) main().catch((e) => {
  console.error(e);
  process.exit(1);
});
