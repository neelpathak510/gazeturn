#!/bin/bash
# Double-click to start GazeTurn Lab on a Mac. It serves this folder on http://localhost
# (browsers only allow the camera on https or localhost) and opens Chrome.
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
[ -s "$HOME/.nvm/nvm.sh" ] && . "$HOME/.nvm/nvm.sh" >/dev/null 2>&1

if command -v node >/dev/null 2>&1; then
  exec node server.js
elif xcode-select -p >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  echo "Node.js not found; using Python instead."
  (sleep 1; open -a "Google Chrome" "http://localhost:8765/" 2>/dev/null || open "http://localhost:8765/") &
  echo "GazeTurn Lab is running at http://localhost:8765/  (Ctrl+C to stop)"
  exec python3 -m http.server 8765 --bind 127.0.0.1
else
  echo "Neither Node.js nor Python 3 was found."
  echo "Install Node.js from https://nodejs.org (the LTS version), then double-click this file again."
  read -r -p "Press Return to close."
fi
