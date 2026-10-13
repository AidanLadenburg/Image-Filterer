#!/usr/bin/env bash
# Double-click launcher: asks for tonight's password, then starts the server.
# The password is never saved to disk — you set it fresh each time you run
# this. Sessions last at most 18h and survive restarts with the same password;
# choosing a different password invalidates previously signed-in sessions.
set -uo pipefail
# Everything in one block: bash reads it all before running, so updating this
# file (e.g. git pull) while the app is live can't break the running launcher.
{
shopt -s nullglob

cd "$(dirname "${BASH_SOURCE[0]}")"

# Keep native state and downloaded models in this checkout unless explicitly overridden.
export IMAGE_FILTERER_DATA_ROOT="${IMAGE_FILTERER_DATA_ROOT:-$PWD/data}"
export HF_HOME="${HF_HOME:-$IMAGE_FILTERER_DATA_ROOT/huggingface}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$IMAGE_FILTERER_DATA_ROOT/ultralytics}"
mkdir -p "$IMAGE_FILTERER_DATA_ROOT" "$HF_HOME" "$YOLO_CONFIG_DIR/Ultralytics"

pause_on_exit() {
  echo
  read -r -p "Press Enter to close this window."
}
trap pause_on_exit EXIT

echo "=== Image Filterer ==="
echo

# Double-clicking spawns a fresh, non-login shell, so PATH never picks up
# whatever a conda/venv activation added to ~/.bashrc. Rather than relying on
# PATH, look for the installed binary directly in the usual places.
find_binary() {
  local name="$1"
  if [ -x "./.venv/bin/$name" ]; then
    echo "./.venv/bin/$name"; return 0
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"; return 0
  fi
  local base envdir
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
    [ -x "$base/bin/$name" ] && { echo "$base/bin/$name"; return 0; }
    for envdir in "$base"/envs/*; do
      [ -x "$envdir/bin/$name" ] && { echo "$envdir/bin/$name"; return 0; }
    done
  done
  return 1
}

BIN="$(find_binary image-filterer-server)" || {
  echo "Couldn't find 'image-filterer-server' anywhere — checked PATH, a"
  echo "./.venv/ next to this script, and every miniconda/anaconda/miniforge"
  echo "environment under $HOME."
  echo "Make sure the project is installed (see README: pip install -e .)"
  exit 1
}

# Already set (e.g. a scripted start)? Use it; otherwise ask — never saved to disk.
while [ -z "${IMAGE_FILTERER_PASSWORD:-}" ]; do
  read -r -s -p "Set tonight's password (viewers will need this to open the page): " PW1
  echo
  if [ -z "$PW1" ]; then
    echo "Password can't be empty."
    echo
    continue
  fi
  read -r -s -p "Confirm password: " PW2
  echo
  if [ "$PW1" != "$PW2" ]; then
    echo "Those didn't match — try again."
    echo
    continue
  fi
  IMAGE_FILTERER_PASSWORD="$PW1"
done

export IMAGE_FILTERER_PASSWORD
unset PW1 PW2

echo
echo "Starting the server — leave this window open while people are using it."
echo "Close it (or Ctrl+C) to stop the server. If the server crashes or stops"
echo "answering, it is restarted automatically within seconds (logged to"
echo "$IMAGE_FILTERER_DATA_ROOT/launcher.log, with a desktop notification)."
echo
echo "Serving over self-signed HTTPS so Export (Chrome/Edge) works for everyone,"
echo "not just people on this machine. Each browser will show a 'not private'"
echo "warning the first time — that's expected for a self-signed certificate;"
echo "click through it ('Advanced' -> 'Proceed')."
echo

PORT="${IMAGE_FILTERER_PORT:-8600}"
LOG="$IMAGE_FILTERER_DATA_ROOT/launcher.log"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
alert() {
  note "$*"
  command -v notify-send >/dev/null 2>&1 && notify-send -u critical "Image Filterer" "$*" 2>/dev/null || true
  if [ -n "${IMAGE_FILTERER_ALERT_WEBHOOK:-}" ]; then
    curl -s -m 10 -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"text": "Image Filterer: " + sys.argv[1]}))' "$*")" \
      "$IMAGE_FILTERER_ALERT_WEBHOOK" >/dev/null 2>&1 || true
  fi
}

# Supervise: restart the server whenever it exits without Ctrl+C, or stops
# answering. The password stays in this window's memory, so restarts need no
# one at the keyboard; signed-in viewers stay signed in (same password), and
# an active folder watch resumes into the same run.
stopping=0; pid=""
trap 'stopping=1; [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null' INT TERM
fast_fails=0
while :; do
  started=$(date +%s)
  "$BIN" --host 0.0.0.0 --port "$PORT" --open-firewall --https &
  pid=$!
  # Hang check: the page must answer. 3 misses in a row → restart (~2 min worst case).
  ( sleep 90; misses=0
    while kill -0 "$pid" 2>/dev/null; do
      if curl -sk -o /dev/null -m 15 "https://127.0.0.1:$PORT/"; then misses=0; else misses=$((misses + 1)); fi
      if [ "$misses" -ge 3 ]; then
        echo "[$(date '+%F %T')] server not answering — restarting it" >> "$LOG"
        kill -TERM "$pid" 2>/dev/null; sleep 5; kill -KILL "$pid" 2>/dev/null; break
      fi
      sleep 20
    done ) &
  probe=$!
  while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
  wait "$pid" 2>/dev/null; rc=$?
  kill "$probe" 2>/dev/null; wait "$probe" 2>/dev/null
  pid=""
  if [ "$stopping" -eq 1 ]; then note "stopped (Ctrl+C)"; break; fi
  if [ $(( $(date +%s) - started )) -lt 120 ]; then fast_fails=$((fast_fails + 1)); else fast_fails=0; fi
  delay=$(( fast_fails > 3 ? 30 : 3 ))
  alert "server stopped unexpectedly (exit $rc) — restarting in ${delay}s"
  sleep "$delay" || true
  [ "$stopping" -eq 1 ] && { note "stopped (Ctrl+C)"; break; }
done
exit
}
