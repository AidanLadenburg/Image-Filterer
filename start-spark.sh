#!/usr/bin/env bash
# Run with bash start-spark.sh <build|setup|check|up|down|logs|status>.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -f .env.spark ]]; then
  echo 'Copy .env.spark.example to .env.spark and configure it first.' >&2
  exit 1
fi
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker compose version >/dev/null
compose=(docker compose --env-file .env.spark -f compose.spark.yaml)
action="${1:-up}"
if [[ $# -gt 0 ]]; then shift; fi

prompt_for_password() {
  if [[ -n "${IMAGE_FILTERER_PASSWORD:-}" ]]; then
    return
  fi
  if [[ ! -t 0 ]]; then
    echo 'Set IMAGE_FILTERER_PASSWORD when starting non-interactively.' >&2
    exit 1
  fi
  local first second
  while true; do
    read -r -s -p "Set tonight's password (viewers will need this to open the page): " first
    echo
    if [[ -z "$first" ]]; then
      echo "Password can't be empty."
      echo
      continue
    fi
    read -r -s -p "Confirm password: " second
    echo
    if [[ "$first" != "$second" ]]; then
      echo "Those didn't match — try again."
      echo
      continue
    fi
    export IMAGE_FILTERER_PASSWORD="$first"
    unset first second
    return
  done
}

case "$action" in
  build) "${compose[@]}" build "$@" ;;
  setup) "${compose[@]}" run --rm --no-deps image-filterer setup "$@" ;;
  check) "${compose[@]}" run --rm --no-deps image-filterer check "$@" ;;
  up) prompt_for_password; "${compose[@]}" up -d "$@" ;;
  down) "${compose[@]}" down "$@" ;;
  logs) "${compose[@]}" logs -f --tail=100 "$@" ;;
  status) "${compose[@]}" ps "$@" ;;
  *) echo 'Usage: bash start-spark.sh {build|setup|check|up|down|logs|status}' >&2; exit 2 ;;
esac
