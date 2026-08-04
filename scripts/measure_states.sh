#!/usr/bin/env bash
# States/s per actor count — the TRAINING-relevant rate (states from games COMPLETED inside
# an interior window; in-flight games at the kill count nowhere, mirroring what a hard-killed
# round loses). Selects OpenSpiel's round config honestly: rows/s and states/s disagree
# (more actors => more redundant rows but fewer finished games).
#
# Round-true workload: cache ON, checkpoint_freq=1, learner running — windows are long enough
# (default 20 min) to include learn steps + checkpoint writes, which share the GPU with actors.
# ACTORS entries are "N" (batch=N) or "N:B" (decoupling probe, e.g. 64:32).
#
#   CORES=0-3 GAME=chess WIDTH=256 DEPTH=8 ACTORS="16 32 64 64:32" bash scripts/measure_states.sh
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=open_spiel_cpp/open_spiel/build/examples/alpha_zero_torch_example
CORES="${CORES:-0-3}"
GAME="${GAME:-chess}"
WIDTH="${WIDTH:-256}"
DEPTH="${DEPTH:-8}"
ACTORS="${ACTORS:-16 32 64 64:32}"
CACHE="${CACHE:-262144}"  # their default; sparse entries, so their 2^18 costs ~nothing
WARMUP="${WARMUP:-300}"
WINDOW="${WINDOW:-1200}"
OUT_ROOT=results/states_measure
mkdir -p "$OUT_ROOT"
ACTIVE_PID=""
trap '[ -n "$ACTIVE_PID" ] && kill -9 "$ACTIVE_PID" 2>/dev/null || true' EXIT

for entry in $ACTORS; do
  actors="${entry%%:*}"
  batch="${entry#*:}"; [ "$batch" = "$entry" ] && batch="$actors"
  tag="${GAME}_w${WIDTH}_d${DEPTH}_a${actors}"
  [ "$batch" != "$actors" ] && tag="${tag}_b${batch}"
  out="$OUT_ROOT/$tag"
  echo "=== $tag (${WARMUP}s warmup + ${WINDOW}s window) ==="
  rm -rf "$out"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$CORES" "$BIN" --game="$GAME" --path="$out" \
    --actors="$actors" --evaluators=0 --devices=/cuda:0 \
    --max_simulations=64 --uct_c=2 --policy_alpha=0.3 --policy_epsilon=0.25 \
    --temperature=1 --temperature_drop=10 \
    --nn_model=resnet --nn_width="$WIDTH" --nn_depth="$DEPTH" \
    --inference_batch_size="$batch" --inference_threads=1 --inference_cache="$CACHE" \
    --replay_buffer_size=65536 --replay_buffer_reuse=3 --train_batch_size=1024 \
    --learning_rate=0.0001 --weight_decay=0.0001 --checkpoint_freq=1 \
    --evaluation_window=100 --eval_levels=7 --cutoff_probability=0 --cutoff_value=0.95 \
    --explicit_learning=false --max_steps=0 > "${out}.stdout" 2>&1 &
  pid=$!; ACTIVE_PID=$pid
  sleep "$WARMUP"
  t1=$(date +"%Y-%m-%d %H:%M:%S")
  r1=$(grep -o "rows=[0-9]*" "${out}.stdout" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  sleep "$WINDOW"
  t2=$(date +"%Y-%m-%d %H:%M:%S")
  r2=$(grep -o "rows=[0-9]*" "${out}.stdout" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  ACTIVE_PID=""
  python3 - "$out" "$t1" "$t2" "$WINDOW" "${r1:-0}" "${r2:-0}" <<'PYEOF'
import sys, glob
from datetime import datetime
out, t1s, t2s, window, r1, r2 = sys.argv[1:7]
t1 = datetime.strptime(t1s, "%Y-%m-%d %H:%M:%S")
t2 = datetime.strptime(t2s, "%Y-%m-%d %H:%M:%S")
states = games = steps = 0
for f in glob.glob(f"{out}/log-learner*"):
    for line in open(f, errors="ignore"):
        if "Step" not in line or not line.startswith("["):
            continue
        try:
            ts = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if t1 <= ts <= t2:
            steps += 1
for f in glob.glob(f"{out}/log-actor*"):
    for line in open(f, errors="ignore"):
        if "Actions:" not in line or not line.startswith("["):
            continue
        try:
            ts = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if t1 <= ts <= t2:
            games += 1
            states += len(line.split("Actions:")[1].split())
w = float(window)
rows_s = (int(r2) - int(r1)) / w if int(r2) > int(r1) else float("nan")
print(f"{out.split('/')[-1]}  states/s={states / w:7.1f}  (games={games}, avg_len={states / max(games, 1):.0f})  rows/s={rows_s:8.1f}  learn_steps={steps}  rows_per_state={rows_s / max(states / w, 1e-9):.0f}")
PYEOF
done
