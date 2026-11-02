/* Synthetic reader: produces a webcam-like gaze stream for a person reading pages,
   so the detectors can be tested without a camera. Units are page-normalized
   (x across the text column, y down the page). Noise is specified in degrees. */
"use strict";

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function gauss(r) {
  let u = 0;
  while (!u) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

/**
 * opts: pages, linesPerPage, sigmaDeg (per-sample jitter), biasDeg (slow drift limit),
 * yNoiseMult (vertical is noisier on webcams), fps, distCm, colCm, pageCm, seed,
 * glanceProb (chance per page of looking up to the top mid-page), rereadProb (per line),
 * lingerLast (s resting on the last word before turning; 0 = none),
 * pauseProb (chance per line of a long pause at the line's end, e.g. thinking).
 */
function simulate(opts = {}) {
  const o = Object.assign(
    {
      pages: 10, linesPerPage: 22, sigmaDeg: 1, biasDeg: 1, yNoiseMult: 1.6, fps: 30,
      distCm: 50, colCm: 14.8, pageCm: 16.5, seed: 1, glanceProb: 0.35, rereadProb: 0.03,
      lingerLast: 0, pauseProb: 0,
    },
    opts
  );
  const r = rng(o.seed);
  const degX = (d) => (Math.tan((d * Math.PI) / 180) * o.distCm) / o.colCm;
  const degY = (d) => (Math.tan((d * Math.PI) / 180) * o.distCm) / o.pageCm;
  const lineY = (l) => 0.02 + ((l + 0.5) / o.linesPerPage) * 0.96;

  // 1) True fixation sequence.
  const fix = [];
  const manual = [];
  const pageInfo = [];
  let t = 0.5;
  const add = (x, y, dur) => {
    fix.push({ x, y, t0: t, t1: t + dur });
    t += dur + 0.035; // saccade
  };
  for (let p = 0; p < o.pages; p++) {
    const N = o.linesPerPage;
    const glanceLine = r() < o.glanceProb ? Math.floor(N * (0.3 + 0.4 * r())) : -1;
    let lastEnd = 0.9;
    for (let l = 0; l < N; l++) {
      const y = lineY(l);
      const short = r() < 0.15;
      const xEnd = short ? 0.15 + r() * 0.7 : 0.88 + r() * 0.1;
      if (l === N - 1) lastEnd = xEnd;
      let x = 0.02 + r() * 0.05;
      while (x < xEnd) {
        add(Math.min(x, xEnd - 0.02), y, 0.18 + r() * 0.14);
        if (r() < 0.08) add(Math.max(0.02, x - 0.08 - r() * 0.1), y, 0.2);
        x += 0.07 + r() * 0.08;
      }
      if (l < N - 1 && r() < o.pauseProb) add(xEnd - 0.03, y, 0.5 + r() * 0.5);
      if (l > 4 && r() < o.rereadProb) {
        const back = 2 + Math.floor(r() * 4);
        add(0.2 + r() * 0.5, lineY(l - back), 0.5 + r() * 0.5);
      }
      if (l === glanceLine) add(0.3 + r() * 0.4, 0.03 + r() * 0.05, 0.7 + r() * 0.6);
    }
    if (o.lingerLast > 0) add(lastEnd - 0.04, lineY(N - 1), o.lingerLast);
    manual.push(t + 0.15 + r() * 0.25); // presses the key just after
    pageInfo.push({ lastLineEnd: lastEnd, lastLineY: lineY(N - 1), lineH: 0.96 / N });
    t += 0.4;
  }
  // Start reading the page after the last one, so the final turn has a look-up too.
  for (let x = 0.03; x < 0.6; x += 0.1) add(x, 0.04, 0.25);
  const tEnd = t + 1;

  // 2) Sample it like a 30 fps camera with jitter, slow drift and blinks.
  const samples = [];
  let fi = 0;
  let bx = 0;
  let by = 0;
  let nextBlink = 2 + r() * 3;
  let blinkEnd = -1;
  for (let k = 0; k * (1 / o.fps) < tEnd; k++) {
    const ts = k / o.fps;
    while (fi < fix.length - 1 && fix[fi + 1].t0 <= ts) fi++; // hold each fixation until the next begins
    const f = fix[fi];
    bx = Math.max(-o.biasDeg, Math.min(o.biasDeg, bx + gauss(r) * 0.05));
    by = Math.max(-o.biasDeg * o.yNoiseMult, Math.min(o.biasDeg * o.yNoiseMult, by + gauss(r) * 0.05 * o.yNoiseMult));
    if (ts >= nextBlink) {
      blinkEnd = ts + 0.12 + r() * 0.08;
      nextBlink = ts + 2.5 + r() * 3;
    }
    const valid = ts > blinkEnd;
    const x = f.x + degX(bx + o.sigmaDeg * gauss(r));
    const y = f.y + degY(by + o.sigmaDeg * o.yNoiseMult * gauss(r));
    samples.push({ t: ts, x, y, valid });
  }
  return { samples, manual, pageInfo, linesPerPage: o.linesPerPage, pages: o.pages, tEnd, degX, degY };
}

/** The same zone the site draws (see zoneFor in js/app.js): tight around the last word
    left–right, generous up–down. */
function zoneFor(info, accDeg, degX, degY) {
  const em = 22 / 682; // 1 em as a share of the text width at the default size
  const lh = info.lineH;
  const word = { x0: Math.max(0, info.lastLineEnd - 3 * em), x1: info.lastLineEnd };
  const mx = Math.min(Math.max(degX(accDeg), 0.5 * em), 3 * em);
  const my = Math.min(Math.max(degY(accDeg) * 1.5, 1.5 * lh), 4 * lh);
  const top = info.lastLineY - lh / 2;
  const bot = info.lastLineY + lh / 2;
  return { x0: word.x0 - mx, x1: word.x1 + 2 * mx + em, y0: top - my, y1: bot + my, wordX0: word.x0 };
}

module.exports = { simulate, zoneFor, rng, gauss };
