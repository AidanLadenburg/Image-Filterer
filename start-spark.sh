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
case "$action" in
  build) "${compose[@]}" build "$@" ;;
  setup) "${compose[@]}" run --rm --no-deps image-filterer setup "$@" ;;
  check) "${compose[@]}" run --rm --no-deps image-filterer check "$@" ;;
  up) "${compose[@]}" up -d "$@" ;;
  down) "${compose[@]}" down "$@" ;;
  logs) "${compose[@]}" logs -f --tail=100 "$@" ;;
  status) "${compose[@]}" ps "$@" ;;
  *) echo 'Usage: bash start-spark.sh {build|setup|check|up|down|logs|status}' >&2; exit 2 ;;
esac
