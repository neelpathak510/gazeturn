/* Small statistics helpers shared by every module. */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});

  function sortedFinite(a) {
    return Float64Array.from(a.filter(Number.isFinite)).sort();
  }

  GT.median = function median(a) {
    const s = sortedFinite(a);
    if (!s.length) return NaN;
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  GT.quantile = function quantile(a, q) {
    const s = sortedFinite(a);
    if (!s.length) return NaN;
    const i = (s.length - 1) * q;
    const lo = Math.floor(i);
    const hi = Math.ceil(i);
    return s[lo] + (s[hi] - s[lo]) * (i - lo);
  };

  GT.mean = function mean(a) {
    let n = 0;
    let sum = 0;
    for (const v of a) {
      if (Number.isFinite(v)) {
        sum += v;
        n++;
      }
    }
    return n ? sum / n : NaN;
  };

  if (typeof module !== "undefined" && module.exports) module.exports = GT;
})(typeof window !== "undefined" ? window : globalThis);
