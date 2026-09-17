#!/usr/bin/env bash
# Double-click launcher: asks for tonight's password, then starts the server.
# The password is never saved to disk — you set it fresh each time you run
# this. Sessions last at most 18h and survive restarts with the same password;
# choosing a different password invalidates previously signed-in sessions.
set -uo pipefail
shopt -s nullglob

cd "$(dirname "${BASH_SOURCE[0]}")"

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
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"; return 0
  fi
  if [ -x "./.venv/bin/$name" ]; then
    echo "./.venv/bin/$name"; return 0
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

while true; do
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
  break
done

export IMAGE_FILTERER_PASSWORD="$PW1"
unset PW1 PW2

echo
echo "Starting the server — leave this window open while people are using it."
echo "Close it (or Ctrl+C) to stop the server."
echo
echo "Serving over self-signed HTTPS so Export (Chrome/Edge) works for everyone,"
echo "not just people on this machine. Each browser will show a 'not private'"
echo "warning the first time — that's expected for a self-signed certificate;"
echo "click through it ('Advanced' -> 'Proceed')."
echo

"$BIN" --host 0.0.0.0 --port 8600 --open-firewall --https
