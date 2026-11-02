/* Minimal ZIP writer and reader (stored, no compression). JPEGs don't compress anyway.
   Works in the browser (writer → Blob) and in Node (reader, used by tools/bench.js). */
(function (root) {
  "use strict";
  const GT = (root.GT = root.GT || {});

  const TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();
  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  const enc = (s) => new TextEncoder().encode(s);

  /** files: [{name, data: Uint8Array | string}] → array of Uint8Array parts forming a .zip */
  function buildParts(files) {
    const parts = [];
    const central = [];
    let offset = 0;
    const now = new Date();
    const dosTime = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1);
    const dosDate = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
    for (const f of files) {
      const name = enc(f.name);
      const data = typeof f.data === "string" ? enc(f.data) : f.data;
      const crc = crc32(data);
      const h = new DataView(new ArrayBuffer(30));
      h.setUint32(0, 0x04034b50, true);
      h.setUint16(4, 20, true);
      h.setUint16(6, 0x0800, true); // UTF-8 names
      h.setUint16(8, 0, true); // stored
      h.setUint16(10, dosTime, true);
      h.setUint16(12, dosDate, true);
      h.setUint32(14, crc, true);
      h.setUint32(18, data.length, true);
      h.setUint32(22, data.length, true);
      h.setUint16(26, name.length, true);
      h.setUint16(28, 0, true);
      parts.push(new Uint8Array(h.buffer), name, data);
      const c = new DataView(new ArrayBuffer(46));
      c.setUint32(0, 0x02014b50, true);
      c.setUint16(4, 20, true);
      c.setUint16(6, 20, true);
      c.setUint16(8, 0x0800, true);
      c.setUint16(10, 0, true);
      c.setUint16(12, dosTime, true);
      c.setUint16(14, dosDate, true);
      c.setUint32(16, crc, true);
      c.setUint32(20, data.length, true);
      c.setUint32(24, data.length, true);
      c.setUint16(28, name.length, true);
      c.setUint32(42, offset, true);
      central.push(new Uint8Array(c.buffer), name);
      offset += 30 + name.length + data.length;
    }
    let csize = 0;
    for (const p of central) csize += p.length;
    const e = new DataView(new ArrayBuffer(22));
    e.setUint32(0, 0x06054b50, true);
    e.setUint16(8, files.length, true);
    e.setUint16(10, files.length, true);
    e.setUint32(12, csize, true);
    e.setUint32(16, offset, true);
    return parts.concat(central, [new Uint8Array(e.buffer)]);
  }

  /** Read a stored ZIP (Uint8Array) → Map(name → Uint8Array). */
  function read(buf) {
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    let eo = buf.length - 22;
    while (eo >= 0 && dv.getUint32(eo, true) !== 0x06054b50) eo--;
    if (eo < 0) throw new Error("not a zip file");
    const count = dv.getUint16(eo + 10, true);
    let p = dv.getUint32(eo + 16, true);
    const out = new Map();
    const dec = new TextDecoder();
    for (let i = 0; i < count; i++) {
      if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("bad central directory");
      const method = dv.getUint16(p + 10, true);
      const size = dv.getUint32(p + 20, true);
      const nlen = dv.getUint16(p + 28, true);
      const elen = dv.getUint16(p + 30, true);
      const clen = dv.getUint16(p + 32, true);
      const lo = dv.getUint32(p + 42, true);
      const name = dec.decode(buf.subarray(p + 46, p + 46 + nlen));
      if (method !== 0) throw new Error(`${name}: compressed entries not supported`);
      const lnlen = dv.getUint16(lo + 26, true);
      const lelen = dv.getUint16(lo + 28, true);
      const start = lo + 30 + lnlen + lelen;
      out.set(name, buf.subarray(start, start + size));
      p += 46 + nlen + elen + clen;
    }
    return out;
  }

  GT.zip = { crc32, buildParts, read };
  if (typeof module !== "undefined" && module.exports) module.exports = GT.zip;
})(typeof window !== "undefined" ? window : globalThis);
