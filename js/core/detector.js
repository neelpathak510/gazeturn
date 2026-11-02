/* Reading-pattern page-turn detector, plus gestures, dwell targets and scoring.
   Everything here is pure (no DOM, no camera), so the same code runs live in the browser,
   in Node tests, and when re-scoring a saved session with new settings.

   Coordinates: x 0→1 across the text column, y 0→1 down the page.

   Idea: webcam gaze is too noisy to tell which line you're on, but the jump from the end of
   one line back to the start of the next (a "line return") is a big, fast leftward move that
   survives the noise. Counting those tells us how far down the page you are. A page ends
   when enough lines are counted, your gaze has been low on the page, and then it jumps up. */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});
  const median = (a) => GT.median(a);

  const DETECTOR_DEFAULTS = {
    returnAmp: 0.3, // leftward jump, as a share of the text width, that counts as a line return
    returnWindow: 0.45, // s: the jump must happen within this time
    returnRefractory: 0.6, // s: minimum time between two line returns
    returnMaxUp: 0.2, // a line return may not move the gaze up by more than this share of the page
    returnFromMin: 0.3, // the jump must start right of this point…
    returnToMax: 0.7, // …and land left of this one
    upAmp: 0.3, // upward jump (share of page height) that signals "I'm at the top again"
    settleTime: 0.5, // s the gaze must stay up after the jump (a quick glance back up doesn't count)
    landLeftMax: 0.6, // after the jump the gaze should rest on the left part (where a page starts)
    nearEndFrac: 0.8, // share of the page's line returns that must be counted before a turn
    yLowMin: 0.5, // median gaze height in the 1.5 s before the jump (0 top, 1 bottom)
    minLines: 3,
    minPageTime: 5, // s on a page before it can turn
    cooldown: 1.5, // s after any turn
    useLayout: true, // use the real line count of the page when the reader knows it
  };

  class ReadingDetector {
    constructor(params = {}) {
      this.p = Object.assign({}, DETECTOR_DEFAULTS, params);
      this.learned = [];
      this.totalLines = 0;
      this.lastTurnT = -1e9;
      this.prev = null;
      this.stats = { upJumps: 0, accepted: 0, rejectLines: 0, rejectLow: 0, rejectTime: 0, rejectReturned: 0, rejectLanding: 0 };
      this.resetPage(0);
    }

    setParams(p) {
      Object.assign(this.p, p);
    }

    setPage(totalLines) {
      this.totalLines = totalLines || 0;
    }

    resetPage(t) {
      this.lines = 0;
      this.pageStartT = t;
      this.buf = [];
      this.lastReturnT = -1e9;
      this.lastJumpT = -1e9;
      this.cand = null;
    }

    medianIn(t0, t1, key) {
      const v = [];
      for (const b of this.buf) if (b.t >= t0 && b.t <= t1) v.push(b[key]);
      return { m: median(v), n: v.length };
    }

    learnedLines() {
      return this.learned.length ? median(this.learned) : 0;
    }

    linesNeeded() {
      const base = this.p.useLayout && this.totalLines ? this.totalLines - 1 : this.learnedLines();
      if (!base) return this.p.minLines;
      return Math.max(this.p.minLines, Math.floor(this.p.nearEndFrac * base));
    }

    /** s = {t, x, y, valid}. Returns events: line, pageEnd, reject. */
    update(s) {
      const p = this.p;
      const t = s.t;
      const out = [];
      const buf = this.buf;
      while (buf.length && buf[0].t < t - 2.5) buf.shift();
      if (!s.valid || !Number.isFinite(s.x) || !Number.isFinite(s.y)) return out;
      buf.push({ t, x: s.x, y: s.y });

      // 1) Line return: fast leftward jump from the right part of the column to the left part.
      let iMax = -1;
      let maxX = -Infinity;
      for (let i = buf.length - 1; i >= 0 && buf[i].t >= t - p.returnWindow; i--) {
        if (buf[i].x > maxX) {
          maxX = buf[i].x;
          iMax = i;
        }
      }
      if (iMax >= 0) {
        const peak = buf[iMax];
        if (
          maxX - s.x >= p.returnAmp &&
          peak.t > this.lastReturnT &&
          t - this.lastReturnT >= p.returnRefractory &&
          s.y - peak.y >= -p.returnMaxUp &&
          maxX >= p.returnFromMin &&
          s.x <= p.returnToMax
        ) {
          this.lines++;
          this.lastReturnT = t;
          out.push({ type: "line", t, n: this.lines });
        }
      }

      // 2) Upward jump. Medians of short windows before and after, so vertical jitter alone
      //    can't fake one. A jump only becomes a candidate end of page.
      if (!this.cand && t - this.lastJumpT >= 1.0) {
        const before = this.medianIn(t - 1.1, t - 0.45, "y");
        const now = this.medianIn(t - 0.2, t, "y");
        if (before.n >= 5 && now.n >= 3 && before.m - now.m >= p.upAmp) {
          this.lastJumpT = t;
          this.cand = {
            t,
            before: before.m,
            jump: before.m - now.m,
            yLow: this.medianIn(t - 2.0, t - 0.45, "y").m,
            lines: this.lines,
          };
        }
      }

      // 3) Candidate settles: the gaze must stay up and rest on the left, where a page begins.
      if (this.cand && t - this.cand.t >= p.settleTime) {
        const c = this.cand;
        this.cand = null;
        const after = this.medianIn(c.t, t, "y");
        const afterX = this.medianIn(c.t, t, "x");
        const need = this.linesNeeded();
        const reasons = [];
        if (!(after.m <= c.before - 0.6 * p.upAmp)) reasons.push("returned");
        if (!(afterX.m <= p.landLeftMax)) reasons.push("landing");
        if (c.lines < need) reasons.push("lines");
        if (!(c.yLow >= p.yLowMin)) reasons.push("low");
        if (c.t - this.pageStartT < p.minPageTime || c.t - this.lastTurnT < p.cooldown) reasons.push("time");
        this.stats.upJumps++;
        const info = { t, jumpT: c.t, lines: c.lines, need, yLow: round(c.yLow), jump: round(c.jump) };
        if (!reasons.length) {
          this.stats.accepted++;
          out.push(Object.assign({ type: "pageEnd" }, info));
        } else {
          for (const r of reasons) {
            const k = "reject" + r[0].toUpperCase() + r.slice(1);
            this.stats[k] = (this.stats[k] || 0) + 1;
          }
          out.push(Object.assign({ type: "reject", reasons }, info));
        }
      }
      return out;
    }

    /** The page changed forward. carry = line returns already seen on the new page. */
    turned(t, carry = 0) {
      this.prev = { lines: this.lines, pageStartT: this.pageStartT, learned: this.learned.slice() };
      const counted = this.lines - carry;
      if (counted >= this.p.minLines) {
        this.learned.push(counted);
        if (this.learned.length > 7) this.learned.shift();
      }
      this.lastTurnT = t;
      this.resetPage(t);
      this.lines = carry;
    }

    /** A turn was wrong and the reader went back: restore that page's progress. */
    undo(t) {
      if (!this.prev) {
        this.resetPage(t);
        this.lastTurnT = t;
        return;
      }
      const prev = this.prev;
      this.prev = null;
      this.learned = prev.learned;
      this.resetPage(prev.pageStartT);
      this.lines = prev.lines;
      this.lastTurnT = t;
    }
  }

  function round(v) {
    return Number.isFinite(v) ? Math.round(v * 1000) / 1000 : v;
  }

  /* ------------------------------------------------------------------ last word
     The page turns when the gaze rests on the page's last word. The reader knows exactly
     where that word is. Webcams are good left–right and poor up–down (error spans a line or
     two), so the zone is tight horizontally and generous vertically. What separates the last
     line from the ones above is behaviour: at the end of an ordinary line the eyes jump back
     left within ~0.3 s (which resets the timer); at the end of the page they stay. */
  const ENDZONE_DEFAULTS = {
    endDwell: 0.65, // s resting at the end of the text before the page turns
    approachWindow: 3, // s: look back this far to check the gaze arrived by reading…
    approachMin: 0.25, // …from at least this far left of the last word (share of text width)
    stillWindow: 0.45, // s: the gaze must have stopped moving right over this window…
    stillMax: 0.05, // …by no more than this share of the text width (reading through vs resting)
    lineGate: false, // also require most line returns to be counted (off: position decides)
  };

  class EndZoneDetector {
    constructor(params = {}) {
      this.p = Object.assign({}, DETECTOR_DEFAULTS, ENDZONE_DEFAULTS, params);
      this.lineDet = new ReadingDetector(this.p); // only used for line counting / optional gate
      this.zone = null;
      this.lastTurnT = -1e9;
      this.prev = null;
      this.stats = { dwells: 0, accepted: 0, rejectApproach: 0, rejectTime: 0, rejectLines: 0 };
      this.resetPage(0);
    }
    get lines() {
      return this.lineDet.lines;
    }
    get totalLines() {
      return this.lineDet.totalLines;
    }
    setParams(p) {
      Object.assign(this.p, p);
      this.lineDet.setParams(p);
    }
    /** zone: {x0, x1, y0, y1, wordX0} in page coordinates (x across the text, y down the page). */
    setPage(totalLines, zone) {
      this.lineDet.setPage(totalLines);
      this.zone = zone || null;
    }
    linesNeeded() {
      return this.lineDet.linesNeeded();
    }
    resetPage(t) {
      this.pageStartT = t;
      this.buf = [];
      this.acc = 0;
      this.enterT = null;
      this.lastT = null;
      this.armed = true;
      this.progress = 0;
    }
    update(s) {
      const out = [];
      let returned = false;
      for (const e of this.lineDet.update(s)) {
        if (e.type === "line") {
          out.push(e);
          returned = true;
        }
      }
      const p = this.p;
      const t = s.t;
      const dt = this.lastT === null ? 0 : Math.min(0.2, Math.max(0, t - this.lastT));
      this.lastT = t;
      while (this.buf.length && this.buf[0].t < t - p.approachWindow - 1.5) this.buf.shift();
      const z = this.zone;
      const valid = s.valid && Number.isFinite(s.x) && Number.isFinite(s.y);
      if (valid) this.buf.push({ t, x: s.x, y: s.y });
      if (!z) return out;
      const inside = valid && s.x >= z.x0 && s.x <= z.x1 && s.y >= z.y0 && s.y <= z.y1;
      if (returned) {
        // A jump back to the start of a line: this was the end of an ordinary line, not the page.
        this.acc = 0;
        this.armed = true;
      } else if (inside) {
        if (this.acc === 0) this.enterT = t;
        this.acc += dt;
      } else if (valid) {
        this.acc = Math.max(0, this.acc - 2 * dt);
        if (this.acc === 0) this.armed = true;
      } // blinks leave the timer where it is
      this.progress = Math.min(1, this.acc / p.endDwell);
      let still = true;
      if (this.armed && this.acc >= p.endDwell) {
        // Resting, not reading through: compare the early and late halves of the last window.
        const w0 = t - p.stillWindow;
        const mid = t - p.stillWindow / 2;
        const a = [];
        const b = [];
        for (const q of this.buf) {
          if (q.t < w0) continue;
          (q.t < mid ? a : b).push(q.x);
        }
        still = a.length >= 3 && b.length >= 3 && Math.abs(median(b) - median(a)) <= p.stillMax;
        if (!still) this.progress = 0.95;
      }
      if (this.armed && this.acc >= p.endDwell && still) {
        this.armed = false;
        this.stats.dwells++;
        const reasons = [];
        let minX = Infinity;
        for (const b of this.buf) if (b.t >= this.enterT - p.approachWindow && b.t <= this.enterT) minX = Math.min(minX, b.x);
        const readIn = minX <= z.wordX0 - p.approachMin || this.enterT - this.lineDet.lastReturnT <= p.approachWindow;
        if (!readIn) reasons.push("approach");
        if (t - this.pageStartT < p.minPageTime || t - this.lastTurnT < p.cooldown) reasons.push("time");
        if (p.lineGate && this.lineDet.totalLines && this.lineDet.lines < this.lineDet.linesNeeded()) reasons.push("lines");
        const info = { t, lines: this.lineDet.lines, need: this.lineDet.linesNeeded(), dwell: round(this.acc) };
        if (!reasons.length) {
          this.stats.accepted++;
          out.push(Object.assign({ type: "pageEnd" }, info));
        } else {
          for (const r of reasons) {
            const k = "reject" + r[0].toUpperCase() + r.slice(1);
            this.stats[k] = (this.stats[k] || 0) + 1;
          }
          out.push(Object.assign({ type: "reject", reasons }, info));
        }
      }
      return out;
    }
    turned(t, carry = 0) {
      this.prev = { pageStartT: this.pageStartT };
      this.lineDet.turned(t, carry);
      this.lastTurnT = t;
      this.resetPage(t);
    }
    undo(t) {
      this.lineDet.undo(t);
      const ps = this.prev ? this.prev.pageStartT : t;
      this.prev = null;
      this.resetPage(ps);
      this.lastTurnT = t;
    }
  }

  /** Both: turn on the last word when tracking is sharp enough, or on the look-up otherwise. */
  class HybridDetector {
    constructor(params = {}) {
      this.ez = new EndZoneDetector(params);
      this.lk = new ReadingDetector(Object.assign({}, DETECTOR_DEFAULTS, params));
      this.via = { lastWord: 0, lookUp: 0 };
    }
    get lines() {
      return this.lk.lines;
    }
    get totalLines() {
      return this.lk.totalLines;
    }
    get progress() {
      return this.ez.progress;
    }
    get lastTurnT() {
      return this.lk.lastTurnT;
    }
    set lastTurnT(v) {
      this.lk.lastTurnT = v;
      this.ez.lastTurnT = v;
    }
    get stats() {
      const a = this.lk.stats;
      const b = this.ez.stats;
      return Object.assign({}, a, {
        dwells: b.dwells,
        dwellAccepted: b.accepted,
        rejectApproach: b.rejectApproach,
        viaLastWord: this.via.lastWord,
        viaLookUp: this.via.lookUp,
      });
    }
    setParams(p) {
      this.ez.setParams(p);
      this.lk.setParams(p);
    }
    setPage(totalLines, zone) {
      this.ez.setPage(totalLines, zone);
      this.lk.setPage(totalLines);
    }
    linesNeeded() {
      return this.lk.linesNeeded();
    }
    resetPage(t) {
      this.ez.resetPage(t);
      this.lk.resetPage(t);
    }
    update(s) {
      const a = this.ez.update(s);
      const b = this.lk.update(s);
      const out = b.filter((e) => e.type === "line");
      const endA = a.find((e) => e.type === "pageEnd");
      const endB = b.find((e) => e.type === "pageEnd");
      for (const e of a) if (e.type === "reject") out.push(Object.assign({ via: "lastWord" }, e));
      for (const e of b) if (e.type === "reject") out.push(Object.assign({ via: "lookUp" }, e));
      if (endA) {
        this.via.lastWord++;
        out.push(Object.assign({ via: "lastWord" }, endA));
      } else if (endB) {
        this.via.lookUp++;
        out.push(Object.assign({ via: "lookUp" }, endB));
      }
      return out;
    }
    turned(t, carry = 0) {
      this.ez.turned(t, carry);
      this.lk.turned(t, carry);
    }
    undo(t) {
      this.ez.undo(t);
      this.lk.undo(t);
    }
  }

  function makeDetector(method, params) {
    if (method === "endzone") return new EndZoneDetector(params);
    if (method === "hybrid") return new HybridDetector(params);
    return new ReadingDetector(params);
  }

  /** Fires once when a score stays above threshold for `hold` seconds; re-arms when it drops. */
  class HoldGesture {
    constructor(threshold, hold) {
      this.threshold = threshold;
      this.hold = hold;
      this.start = null;
      this.fired = false;
    }
    update(t, score) {
      if (!Number.isFinite(score)) return false;
      if (score >= this.threshold) {
        if (this.start === null) this.start = t;
        if (!this.fired && t - this.start >= this.hold) {
          this.fired = true;
          return true;
        }
      } else if (score < this.threshold * 0.75) {
        this.start = null;
        this.fired = false;
      }
      return false;
    }
  }

  /** Look at a target long enough and it fires. Short excursions only slow the timer down. */
  class DwellDetector {
    constructor(dwellTime = 0.9) {
      this.dwellTime = dwellTime;
      this.zones = [];
      this.acc = {};
      this.locked = {};
      this.outside = {};
      this.lastT = null;
    }
    setZones(zones) {
      this.zones = zones;
      for (const z of zones) {
        this.acc[z.id] = 0;
        this.locked[z.id] = false;
        this.outside[z.id] = 0;
      }
    }
    update(t, x, y, valid) {
      const dt = this.lastT === null ? 0 : Math.min(0.2, Math.max(0, t - this.lastT));
      this.lastT = t;
      let fired = null;
      const progress = {};
      for (const z of this.zones) {
        const inside = valid && x >= z.x0 && x <= z.x1 && y >= z.y0 && y <= z.y1;
        if (inside) {
          this.outside[z.id] = 0;
          if (!this.locked[z.id]) this.acc[z.id] += dt;
        } else {
          this.outside[z.id] += dt;
          this.acc[z.id] = Math.max(0, this.acc[z.id] - 1.5 * dt);
          if (this.outside[z.id] > 0.4) this.locked[z.id] = false;
        }
        if (!this.locked[z.id] && this.acc[z.id] >= this.dwellTime) {
          fired = z.id;
          this.locked[z.id] = true;
          this.acc[z.id] = 0;
        }
        progress[z.id] = Math.min(1, this.acc[z.id] / this.dwellTime);
      }
      return { fired, progress };
    }
  }

  /** Shadow test scoring: the reader turns pages; predictions are matched to their presses. */
  class ShadowEvaluator {
    constructor(o = {}) {
      this.early = o.early ?? 3; // a prediction up to 3 s before the press still counts
      this.late = o.late ?? 2; // …or up to 2 s after it
      this.preds = [];
      this.manuals = [];
      this.hits = [];
    }
    predict(t) {
      const p = { t, m: null };
      this.preds.push(p);
      for (const m of this.manuals) {
        if (!m.p && t >= m.t && t - m.t <= this.late) {
          m.p = p;
          p.m = m;
          this.hits.push({ manualT: m.t, predT: t, offset: t - m.t });
          return true;
        }
      }
      return false;
    }
    manual(t) {
      const m = { t, p: null };
      this.manuals.push(m);
      for (const p of this.preds) {
        if (!p.m && p.t <= t && t - p.t <= this.early) {
          p.m = m;
          m.p = p;
          this.hits.push({ manualT: t, predT: p.t, offset: p.t - t });
          return true;
        }
      }
      return false;
    }
    result() {
      return {
        hits: this.hits.map((h) => Object.assign({}, h)),
        misses: this.manuals.filter((m) => !m.p).map((m) => m.t),
        falseTurns: this.preds.filter((p) => !p.m).map((p) => p.t),
      };
    }
  }

  /** Detector + evaluator for shadow mode. Used live and for re-scoring saved sessions. */
  class ShadowRunner {
    constructor(params = {}, evalOpts = {}, method = "lookup") {
      this.det = makeDetector(method, params);
      this.ev = new ShadowEvaluator(evalOpts);
      this.pending = null;
      this.lineTimes = [];
    }
    setPage(total, zone) {
      this.det.setPage(total, zone);
    }
    sample(s) {
      const out = this.det.update(s);
      for (const e of out) {
        if (e.type === "line") this.lineTimes.push(e.t);
        else if (e.type === "pageEnd") {
          e.matched = this.ev.predict(e.t);
          this.det.turned(e.t); // behave as if the page had turned, like the live tool would
          this.pending = null;
        }
      }
      if (this.pending && s.t > this.pending.t + this.ev.late) {
        // No prediction came: the detector missed this page. Start the new page from the press.
        const from = this.pending.t;
        const carry = this.lineTimes.filter((x) => x > from).length;
        this.det.turned(from + this.ev.late, carry);
        this.pending = null;
      }
      return out;
    }
    manualNext(t, total, zone) {
      const matched = this.ev.manual(t);
      this.det.setPage(total, zone);
      if (!matched) this.pending = { t };
      return matched;
    }
    manualBack(t, total, zone) {
      this.pending = null;
      this.det.setPage(total, zone);
      this.det.resetPage(t);
      this.det.lastTurnT = t;
    }
    result() {
      return this.ev.result();
    }
  }

  /** Live test scoring: the tool turns pages; the reader corrects it with the arrow keys. */
  class LiveEvaluator {
    constructor(o = {}) {
      this.undoWindow = o.undoWindow ?? 10;
      this.turns = [];
      this.misses = [];
      this.lates = [];
    }
    systemTurn(t, source) {
      this.turns.push({ t, source, false: false });
    }
    /** Returns true when this "back" marks the last automatic turn as a false turn. */
    back(t) {
      const last = this.turns[this.turns.length - 1];
      if (last && !last.false && t - last.t <= this.undoWindow) {
        last.false = true;
        last.backT = t;
        return true;
      }
      return false;
    }
    manualNext(t) {
      this.misses.push(t);
    }
    late(t) {
      this.lates.push(t);
    }
    result() {
      return {
        turns: this.turns.map((x) => Object.assign({}, x)),
        misses: this.misses.slice(),
        late: this.lates.slice(),
      };
    }
  }

  /** One scoring function for every kind of session. */
  function summarize(c) {
    const hours = Math.max(c.durationS || 0, 1) / 3600;
    const out = {
      kind: c.kind,
      trigger: c.trigger,
      minutes: (c.durationS || 0) / 60,
      validPct: c.validPct,
      fps: c.fps,
      calibDeg: c.calibDeg,
    };
    const pages = c.pages || [];
    out.pagesTurned = pages.filter((p) => p.exit && p.exit !== "back" && p.exit !== "end").length;
    const r = c.evalResult || {};
    if (c.kind === "shadow") {
      out.hits = (r.hits || []).length;
      out.misses = (r.misses || []).length;
      out.falseTurns = (r.falseTurns || []).length;
      out.medianOffset = median((r.hits || []).map((h) => h.offset));
      out.within1s = out.hits ? (r.hits || []).filter((h) => Math.abs(h.offset) <= 1).length / out.hits : NaN;
    } else {
      const turns = r.turns || [];
      out.systemTurns = turns.length;
      out.falseTurns = turns.filter((x) => x.false).length;
      out.hits = out.systemTurns - out.falseTurns;
      out.misses = (r.misses || []).length;
      out.late = (r.late || []).length;
    }
    out.recall = out.hits + out.misses ? out.hits / (out.hits + out.misses) : NaN;
    out.falsePerHour = out.falseTurns / hours;

    // Line counting accuracy: on a page with N lines there are N − 1 line returns.
    const lineTimes = c.lineTimes || [];
    const per = [];
    for (const p of pages) {
      if (!(p.t1 > p.t0) || p.t1 - p.t0 < 6 || !(p.total >= 4)) continue;
      if (!p.exit || p.exit === "back" || p.exit === "end") continue;
      const detected = lineTimes.filter((t) => t > p.t0 && t <= p.t1).length;
      per.push({ i: p.i, total: p.total, expected: p.total - 1, detected });
    }
    out.linePages = per;
    out.lineRatio = per.length ? median(per.map((p) => p.detected / p.expected)) : NaN;
    out.lineErrPct = per.length ? GT.mean(per.map((p) => Math.abs(p.detected - p.expected) / p.expected)) : NaN;
    out.rejects = c.detStats || null;
    out.verdict = verdict(out);
    return out;
  }

  function verdict(o) {
    if (o.hits + o.misses < 5) {
      return { level: "more", text: "Read six pages or more to score it." };
    }
    if (o.recall >= 0.9 && o.falsePerHour <= 2) {
      return { level: "good", text: "Reliable, with almost no false turns. Worth building on." };
    }
    if (o.recall >= 0.7 && o.falsePerHour <= 6) {
      return { level: "ok", text: "Mostly works. The details below show what's failing." };
    }
    return { level: "bad", text: "Too many misses or false turns. Check calibration first." };
  }

  /** Re-run a saved shadow session with different settings. */
  function replayShadow(data, params, filterOpts) {
    const run = new ShadowRunner(Object.assign({}, params), {}, (data.cfg && data.cfg.method) || "lookup");
    const S = data.samples;
    const evs = (data.events || [])
      .filter((e) => e.type === "manualNext" || e.type === "manualBack")
      .sort((a, b) => a.t - b.t);
    const rects = (data.rects || []).slice().sort((a, b) => a.t - b.t);
    let rect = rects.length ? rects[0].rect : null;
    let ri = 0;
    const refilter = filterOpts && rect && S.rx && S.ry;
    const fx = refilter ? new GT.OneEuroFilter(filterOpts) : null;
    const fy = refilter ? new GT.OneEuroFilter(filterOpts) : null;
    run.setPage(data.pages && data.pages[0] ? data.pages[0].total : 0, data.pages && data.pages[0] ? data.pages[0].zone : null);
    let k = 0;
    for (let i = 0; i < S.t.length; i++) {
      const t = S.t[i];
      while (k < evs.length && evs[k].t <= t) {
        const e = evs[k++];
        if (e.type === "manualNext") run.manualNext(e.t, e.total, e.zone);
        else run.manualBack(e.t, e.total, e.zone);
      }
      while (ri < rects.length && rects[ri].t <= t) rect = rects[ri++].rect;
      let x = S.px[i];
      let y = S.py[i];
      const valid = !!S.v[i];
      if (refilter && valid) {
        const gx = fx.filter(S.rx[i], t);
        const gy = fy.filter(S.ry[i], t);
        x = (gx - rect.x0) / (rect.x1 - rect.x0);
        y = (gy - rect.y0) / (rect.y1 - rect.y0);
      }
      run.sample({ t, x, y, valid });
    }
    while (k < evs.length) {
      const e = evs[k++];
      if (e.type === "manualNext") run.manualNext(e.t, e.total, e.zone);
      else run.manualBack(e.t, e.total, e.zone);
    }
    return { evalResult: run.result(), lineTimes: run.lineTimes, detStats: Object.assign({}, run.det.stats) };
  }

  GT.DETECTOR_DEFAULTS = DETECTOR_DEFAULTS;
  GT.ENDZONE_DEFAULTS = ENDZONE_DEFAULTS;
  GT.ReadingDetector = ReadingDetector;
  GT.EndZoneDetector = EndZoneDetector;
  GT.HybridDetector = HybridDetector;
  GT.makeDetector = makeDetector;
  GT.HoldGesture = HoldGesture;
  GT.DwellDetector = DwellDetector;
  GT.ShadowEvaluator = ShadowEvaluator;
  GT.ShadowRunner = ShadowRunner;
  GT.LiveEvaluator = LiveEvaluator;
  GT.summarize = summarize;
  GT.replayShadow = replayShadow;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      DETECTOR_DEFAULTS, ENDZONE_DEFAULTS, ReadingDetector, EndZoneDetector, HybridDetector, makeDetector, HoldGesture, DwellDetector,
      ShadowEvaluator, ShadowRunner, LiveEvaluator, summarize, replayShadow,
    };
  }
})(typeof window !== "undefined" ? window : globalThis);
