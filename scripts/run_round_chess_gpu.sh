#!/usr/bin/env bash
# The chess GPU matched round: one training run per side, sequentially, on the A10G box.
# Protocol (all measured/decided in the phase-0 + drain-correction campaign, 2026-08):
#   - operating point: w256 d8 resnet (AZResnetReplica <-> their --nn_width/--nn_depth), CUDA
#   - each side its own best parallelism: OS_ACTORS (from the states/s measurement — set it
#     explicitly below), reinfors n_games=64
#   - infer cache 262144 BOTH sides (their default; cache is architecture, not a matched
#     knob — hit rate is monotone in capacity and host RAM is not a constraint here)
#   - matched refresh cadence: one weight refresh + cache clear per replay_buffer_size/reuse
#     = 21,845 collected states (their learner's own pacing; ours via --collect-size)
#   - checkpoint_freq=1 (their side): at chess-w256 rates a learn step is ~15+ min apart, so
#     per-step checkpoints keep the last-before-T artifact fresh; ours writes ckpt_<s>s.pt
#     every collect-loop iteration (~2 min at chess collect-size)
#   - HARD wall-clock stop: SIGKILL at T on BOTH sides (no drain — see PR #1); the h2h
#     artifact is the last PERIODIC checkpoint written before T on each side
#   - device string is /cuda:0 (libtorch c10 parses it; "/gpu:0" is TF naming and crashes)
#
#   MINUTES=120 OS_ACTORS=16 RF_NGAMES=128 RF_NGROUPS=2 bash scripts/run_round_chess_gpu.sh
set -uo pipefail
cd "$(dirname "$0")/.."
# SMT resets to ON every boot; measurements/rounds are defined at SMT-off, cores 0-3.
if [[ -r /sys/devices/system/cpu/smt/active && "$(cat /sys/devices/system/cpu/smt/active)" != "0" ]]; then
  echo "SMT is ON — condition mismatch. Fix: sudo bash -c 'echo off > /sys/devices/system/cpu/smt/control'" >&2
  exit 1
fi
MINUTES="${MINUTES:-120}"
OS_ACTORS="${OS_ACTORS:?set OS_ACTORS from the states/s measurement}"
OS_BATCH="${OS_BATCH:-$OS_ACTORS}"  # decoupled if the measurement picked e.g. 64:32
RF_NGAMES="${RF_NGAMES:-64}"
RF_NGROUPS="${RF_NGROUPS:-1}"
case "$RF_NGROUPS" in 1|2) ;; *) echo "RF_NGROUPS must be 1 or 2 (got $RF_NGROUPS)" >&2; exit 1 ;; esac
for v in OS_ACTORS OS_BATCH RF_NGAMES; do
  case "${!v}" in ''|*[!0-9]*|0) echo "$v must be a positive integer (got '${!v}')" >&2; exit 1 ;; esac
done
if [ "$RF_NGROUPS" = 2 ] && [ "$RF_NGAMES" -lt 2 ]; then
  echo "RF_NGROUPS=2 needs RF_NGAMES >= 2" >&2; exit 1
fi
WIDTH=256
DEPTH=8
# Cache is ARCHITECTURE, not a matched knob: their evaluator issues Prior+Evaluate as two
# Inference() calls per node and relies on the cache to merge them (sparse legal-only
# entries, cleared per learn step) — they keep their 2^18 default. Ours also runs 2^18:
# hit rate is monotone in capacity (13.6% @32k -> 14.1% @262k), the ~4.3GB of dense entries
# is host RAM the 32GB box doesn't need elsewhere, and throughput is the only criterion.
OS_CACHE=262144
RF_CACHE=262144
SECS=$((MINUTES * 60))
BIN=open_spiel_cpp/open_spiel/build/examples/alpha_zero_torch_example
PY=.venv23/bin/python
OS_OUT="results/round_chess_os_${MINUTES}m_a${OS_ACTORS}_b${OS_BATCH}"
RF_OUT="results/round_chess_rf_${MINUTES}m_n${RF_NGAMES}_g${RF_NGROUPS}"

ACTIVE_PID=""
trap '[ -n "$ACTIVE_PID" ] && kill -9 "$ACTIVE_PID" 2>/dev/null || true' EXIT

echo "=== round plan: ${MINUTES}m/side, w${WIDTH} d${DEPTH} — openspiel actors=${OS_ACTORS} batch=${OS_BATCH} -> ${OS_OUT} | reinfors n_games=${RF_NGAMES} n_groups=${RF_NGROUPS} -> ${RF_OUT} ==="
echo "=== openspiel leg starting ==="
rm -rf "$OS_OUT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-3 "$BIN" --game=chess --path="$OS_OUT" \
  --actors="$OS_ACTORS" --evaluators=0 --devices=/cuda:0 \
  --max_simulations=64 --uct_c=2 --policy_alpha=0.3 --policy_epsilon=0.25 \
  --temperature=1 --temperature_drop=10 \
  --nn_model=resnet --nn_width=$WIDTH --nn_depth=$DEPTH \
  --inference_batch_size="$OS_BATCH" --inference_threads=1 --inference_cache=$OS_CACHE \
  --replay_buffer_size=65536 --replay_buffer_reuse=3 --train_batch_size=1024 \
  --learning_rate=0.0001 --weight_decay=0.0001 \
  --checkpoint_freq=1 --evaluation_window=100 --eval_levels=7 \
  --cutoff_probability=0 --cutoff_value=0.95 --explicit_learning=false \
  --max_steps=0 > "${OS_OUT}.stdout" 2>&1 &
OS_PID=$!; ACTIVE_PID=$OS_PID
sleep "$SECS"
kill -9 "$OS_PID" 2>/dev/null || true
wait "$OS_PID" 2>/dev/null || true
ACTIVE_PID=""
echo "=== openspiel done (hard deadline); checkpoints: ==="
ls -t "$OS_OUT"/checkpoint-* 2>/dev/null | head -3 || echo "WARNING: no checkpoints found"

echo "=== reinfors: chess ${MINUTES}m n_games=${RF_NGAMES} n_groups=${RF_NGROUPS} -> ${RF_OUT} ==="
rm -rf "$RF_OUT"
taskset -c 0-3 $PY benchmarks/openspiel/train_reinfors_az.py \
  --minutes "$MINUTES" --out "$RF_OUT" --device cuda --game chess \
  --seed 0 --n-games "$RF_NGAMES" --n-groups "$RF_NGROUPS" --sims 64 --c-puct 2.0 \
  --width $WIDTH --depth $DEPTH \
  --infer-cache $RF_CACHE --collect-size 21845 --checkpoint-every 60 \
  > "${RF_OUT}.stdout" 2>&1 &
RF_PID=$!; ACTIVE_PID=$RF_PID
sleep "$SECS"
kill -9 "$RF_PID" 2>/dev/null || true
wait "$RF_PID" 2>/dev/null || true
ACTIVE_PID=""
echo "=== reinfors done (hard deadline); checkpoints: ==="
ls -t "$RF_OUT"/ckpt* 2>/dev/null | head -3 || echo "WARNING: no checkpoints found"
