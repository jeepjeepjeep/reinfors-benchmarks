#!/usr/bin/env bash
# reinfors-side companion to measure_states.sh: states/s under the ROUND-TRUE workload —
# the actual train_reinfors_az trainer (learner + checkpoints sharing the GPU), interior-window
# counter deltas from learner.jsonl (cumulative wall/states), hard SIGKILL at the deadline.
# reinfors emits records on episode completion, so states/s here is completed-game states
# (directly comparable to measure_states.sh's number, NOT to search rows/s).
#
#   CORES=0-3 WIDTH=256 DEPTH=8 NGAMES="64 128" MINUTES=20 bash scripts/measure_states_rf.sh
set -uo pipefail
cd "$(dirname "$0")/.."
# SMT resets to ON every boot; measurements/rounds are defined at SMT-off, cores 0-3.
if [[ -r /sys/devices/system/cpu/smt/active && "$(cat /sys/devices/system/cpu/smt/active)" != "0" ]]; then
  echo "SMT is ON — condition mismatch. Fix: sudo bash -c 'echo off > /sys/devices/system/cpu/smt/control'" >&2
  exit 1
fi
PY=.venv23/bin/python
CORES="${CORES:-0-3}"
WIDTH="${WIDTH:-256}"
DEPTH="${DEPTH:-8}"
NGAMES="${NGAMES:-64 128}"
NGROUPS="${NGROUPS:-1}"
CACHE="${CACHE:-262144}"
MINUTES="${MINUTES:-20}"
WARMUP="${WARMUP:-300}"
OUT_ROOT=results/states_measure
mkdir -p "$OUT_ROOT"
ACTIVE_PID=""
trap '[ -n "$ACTIVE_PID" ] && kill -9 "$ACTIVE_PID" 2>/dev/null || true' EXIT

for n in $NGAMES; do
  tag="chess_w${WIDTH}_d${DEPTH}_rf_n${n}_g${NGROUPS}"
  out="$OUT_ROOT/$tag"
  echo "=== $tag (${WARMUP}s warmup, kill at ${MINUTES}m) ==="
  rm -rf "$out"
  taskset -c "$CORES" "$PY" benchmarks/openspiel/train_reinfors_az.py \
    --minutes $((MINUTES + 10)) --device cuda --game chess --out "$out" \
    --seed 0 --n-games "$n" --n-groups "$NGROUPS" --sims 64 --c-puct 2.0 \
    --width "$WIDTH" --depth "$DEPTH" \
    --infer-cache "$CACHE" --collect-size 21845 --checkpoint-every 60 \
    > "${out}.stdout" 2>&1 &
  pid=$!; ACTIVE_PID=$pid
  sleep $((MINUTES * 60))
  kill -9 "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  ACTIVE_PID=""
  python3 - "$out" "$WARMUP" "$((MINUTES * 60 - 30))" <<'PYEOF'
import json, sys
out, lo, hi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
first = last = None
for line in open(f"{out}/learner.jsonl", errors="ignore"):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if lo <= d["wall"] <= hi:
        if first is None:
            first = d
        last = d
tag = out.split("/")[-1]
if first is None or last is first:
    print(f"{tag}  FAILED-NO-INTERIOR-SAMPLES (learner.jsonl had <2 rows in [{lo:.0f}s, {hi:.0f}s])")
    sys.exit(0)
dt = last["wall"] - first["wall"]
ds = last["states"] - first["states"]
dr = last["infer_rows"] - first["infer_rows"]
dstep = last["steps"] - first["steps"]
print(f"{tag}  states/s={ds / dt:7.1f}  net_rows/s={dr / dt:8.1f}  learn_steps={dstep}  "
      f"(window {first['wall']:.0f}s..{last['wall']:.0f}s)")
PYEOF
done
