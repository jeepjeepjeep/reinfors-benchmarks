#!/usr/bin/env bash
# The V1 campaign, end to end: every leg runs through benchmarks/runner.py (strict
# preflight, append-only evidence, start/completion manifests). Phases run in order;
# a failure stops the campaign — resume the failed family with
#   .venv23/bin/python benchmarks/runner.py benchmarks/specs/<family>.json \
#     --set tag=$TAG [more --set ...] --resume runs/<session>
# and re-run this script: completed families are skipped by their session markers.
#
#   TAG=v1 bash scripts/run_v1_campaign.sh
#
# Not covered here (manual): the binary-smoke pytest gate
#   H2H_SMOKE_OS_PATH=<os_train out> H2H_SMOKE_OS_CKPT=<n> \
#     .venv23/bin/python -m pytest benchmarks/openspiel/test_h2h_mirror.py
set -euo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:?set TAG=<frozen tag>; preflight refuses anything not built at it}"
PY=.venv23/bin/python

run_family() { # spec-name, extra --set args...
  local name=$1; shift
  if compgen -G "runs/*_${name}/manifest.json" >/dev/null &&
     grep -l '"status": "ok"' runs/*_"${name}"/manifest.json >/dev/null 2>&1; then
    echo "=== $name: already complete, skipping ==="
    return 0
  fi
  echo "=== $name ==="
  "$PY" benchmarks/runner.py "benchmarks/specs/${name}.json" --set "tag=$TAG" "$@"
}

latest_session() { ls -td runs/*_"$1" 2>/dev/null | head -1; }
rf_ckpt() { ls -t "$1"/out/ckpt* 2>/dev/null | head -1; }
os_ckpt_n() {
  ls "$1"/out/checkpoint-*.pt 2>/dev/null |
    sed -E 's/.*checkpoint-(-?[0-9]+)\.pt/\1/' | sort -n | tail -1
}

run_family v1_smoke
smoke=$(latest_session v1_smoke)
run_family v1_smoke_h2h \
  --set "rf_ckpt=$(rf_ckpt "$smoke/rf_train_smoke/cycle1")" \
  --set "os_path=$smoke/os_train_smoke/cycle1/out" \
  --set "os_ckpt=$(os_ckpt_n "$smoke/os_train_smoke/cycle1")"

run_family v1_grid
run_family v1_training

training=$(latest_session v1_training)
h2h_sets=()
for k in 1 2 3; do
  h2h_sets+=(--set "rf_ckpt_$k=$(rf_ckpt "$training/rf_train/cycle$k")")
  h2h_sets+=(--set "os_path_$k=$training/os_train/cycle$k/out")
  h2h_sets+=(--set "os_ckpt_$k=$(os_ckpt_n "$training/os_train/cycle$k")")
done
run_family v1_h2h "${h2h_sets[@]}"

run_family v1_internal
echo "=== campaign complete; sessions under runs/ ==="
