#!/usr/bin/env node
/* Tiny local web server for GazeTurn Lab. Browsers only allow the camera on
   https:// pages or on http://localhost, so this serves the folder on localhost.
   Uses only Node's built-in modules: no npm install needed.
   Usage:  node server.js        (then open the address it prints) */
"use strict";
const http = require("http");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

const ROOT = __dirname;
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".txt": "text/plain; charset=utf-8",
  ".md": "text/plain; charset=utf-8",
  ".wasm": "application/wasm",
  ".task": "application/octet-stream",
};

function handler(req, res) {
  let p;
  try {
    p = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
  } catch {
    res.writeHead(400).end("Bad request");
    return;
  }
  if (p.endsWith("/")) p += "index.html";
  const file = path.normalize(path.join(ROOT, p));
  if (!file.startsWith(ROOT + path.sep) && file !== ROOT) {
    res.writeHead(403).end("Forbidden");
    return;
  }
  fs.readFile(file, (err, buf) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not found");
      return;
    }
    res.writeHead(200, {
      "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(buf);
  });
}

function listen(port, triesLeft) {
  const server = http.createServer(handler);
  server.once("error", (e) => {
    if (e.code === "EADDRINUSE" && triesLeft > 0) listen(port + 1, triesLeft - 1);
    else {
      console.error("Could not start the server:", e.message);
      process.exit(1);
    }
  });
  server.listen(port, "127.0.0.1", () => {
    const url = `http://localhost:${port}/`;
    console.log(`\n  GazeTurn Lab is running at ${url}`);
    console.log("  Keep this window open while you use it. Press Ctrl+C to stop.\n");
    if (!process.env.NO_OPEN) {
      const cmd =
        process.platform === "darwin"
          ? `open -a "Google Chrome" "${url}" || open "${url}"`
          : process.platform === "win32"
            ? `start "" "${url}"`
            : `xdg-open "${url}"`;
      exec(cmd, () => {});
    }
  });
}

listen(Number(process.env.PORT) || 8765, 10);
