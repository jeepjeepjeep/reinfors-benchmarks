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
OUT_ROOT="${OUT_ROOT:-results/states_measure}"
mkdir -p "$OUT_ROOT"
ACTIVE_PID=""
trap '[ -n "$ACTIVE_PID" ] && kill -9 "$ACTIVE_PID" 2>/dev/null || true' EXIT

for n in $NGAMES; do
  tag="chess_w${WIDTH}_d${DEPTH}_rf_n${n}_g${NGROUPS}"
  out="$OUT_ROOT/$tag"
  echo "=== $tag (${WARMUP}s warmup, kill at ${MINUTES}m) ==="
  if [ -e "$out" ]; then
    echo "refusing to overwrite $out — move it aside or pick a new OUT_ROOT" >&2
    exit 1
  fi
  mkdir -p "$out"
  CMD="taskset -c $CORES $PY benchmarks/openspiel/train_reinfors_az.py \
--minutes $((MINUTES + 10)) --device cuda --game chess --out $out \
--seed 0 --n-games $n --n-groups $NGROUPS --sims 64 --c-puct 2.0 \
--width $WIDTH --depth $DEPTH --infer-cache $CACHE --collect-size 21845 --checkpoint-every 60"
  python3 benchmarks/openspiel/manifest.py --out "$out" --command "$CMD" \
    run_kind=measure_cell tag="$tag" deadline_seconds=$((MINUTES * 60)) completed=false >/dev/null
  $CMD > "${out}.stdout" 2>&1 &
  pid=$!; ACTIVE_PID=$pid
  sleep $((MINUTES * 60))
  if kill -0 "$pid" 2>/dev/null; then
    deadline_kill=true
    kill -9 "$pid" 2>/dev/null || true
  else
    deadline_kill=false
  fi
  wait "$pid" 2>/dev/null
  child_rc=$?
  ACTIVE_PID=""
  if [ "$deadline_kill" != true ]; then
    echo "$tag  CRASHED before the deadline (exit $child_rc) — see ${out}.stdout" >&2
    python3 -c "import sys; sys.path.insert(0, 'benchmarks/openspiel'); import manifest; manifest.finalize('$out', status='crashed', exit_code=$child_rc, intended_deadline_kill=False)"
    exit 1
  fi
  python3 - "$out" "$WARMUP" "$((MINUTES * 60 - 30))" "$child_rc" <<'PYEOF'
import json, sys
sys.path.insert(0, "benchmarks/openspiel")
import manifest
out, lo, hi, child_rc = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
hashes = {
    "learner.jsonl": manifest.sha256(f"{out}/learner.jsonl"),
    "stdout": manifest.sha256(f"{out}.stdout"),
}
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
    manifest.finalize(out, status="no-interior-window", intended_deadline_kill=True,
                      exit_code=child_rc, output_sha256=hashes)
    sys.exit(2)
dt = last["wall"] - first["wall"]
ds = last["states"] - first["states"]
dr = last["infer_rows"] - first["infer_rows"]
dc = last["infer_calls"] - first["infer_calls"]
dstep = last["steps"] - first["steps"]
metrics = {
    "window_seconds": [first["wall"], last["wall"]],
    "states_per_sec": ds / dt,
    "net_rows_per_sec": dr / dt,
    "rows_per_call": (dr / dc) if dc > 0 else None,
    "learn_steps": dstep,
}
manifest.finalize(out, status="deadline", intended_deadline_kill=True,
                  exit_code=child_rc, metrics=metrics, output_sha256=hashes)
print(f"{tag}  states/s={ds / dt:7.1f}  net_rows/s={dr / dt:8.1f}  "
      f"rows/call={(dr / dc) if dc > 0 else float('nan'):6.1f}  learn_steps={dstep}  "
      f"(window {first['wall']:.0f}s..{last['wall']:.0f}s)")
PYEOF
  reduce_rc=$?
  if [ "$reduce_rc" -ne 0 ]; then exit "$reduce_rc"; fi
done
