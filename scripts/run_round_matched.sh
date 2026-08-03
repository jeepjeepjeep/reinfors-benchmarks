#!/usr/bin/env bash
# One matched training round per side, sequentially: OpenSpiel C++ AZ then reinfors AZ.
# Cache ON both (262144 entries), libtorch/torch 2.3.0 both, CPU both, matched search/learner
# knobs, and matched refresh cadence: one weight refresh + cache clear per
# replay_buffer_size/reuse = 21,845 collected states on both sides (their learner outer-step
# pacing; ours via --collect-size). Run under `caffeinate -dis` on macOS.
#
#   MINUTES=120 bash scripts/run_round_matched.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MINUTES="${MINUTES:-120}"
SECS=$((MINUTES * 60))
BIN=open_spiel_cpp/open_spiel/build/examples/alpha_zero_torch_example
OS_OUT="results/os${MINUTES}_cache"
RF_OUT="results/rf${MINUTES}_cache"

echo "=== openspiel: ${MINUTES}m -> ${OS_OUT} ==="
rm -rf "$OS_OUT"
"$BIN" --game=connect_four --path="$OS_OUT" \
  --actors=8 --evaluators=0 --devices=/cpu:0 \
  --max_simulations=64 --uct_c=2 --policy_alpha=0.3 --policy_epsilon=0.25 \
  --temperature=1 --temperature_drop=10 \
  --nn_model=resnet --nn_width=32 --nn_depth=1 \
  --inference_batch_size=8 --inference_threads=1 --inference_cache=262144 \
  --replay_buffer_size=65536 --replay_buffer_reuse=3 --train_batch_size=1024 \
  --learning_rate=0.0001 --weight_decay=0.0001 \
  --checkpoint_freq=5 --evaluation_window=100 --eval_levels=7 \
  --cutoff_probability=0 --cutoff_value=0.95 --explicit_learning=false \
  --max_steps=0 > "${OS_OUT}.stdout" 2>&1 &
OS_PID=$!
sleep "$SECS"
# HARD wall-clock stop (2026-08-03): SIGINT + grace let actors DRAIN in-flight games,
# giving this side extra collection minutes beyond the nominal budget — unfair in any
# equal-wall-clock protocol. SIGKILL at T; both sides checkpoint periodically, so the h2h
# artifact is "the last checkpoint written before T" — symmetric across stacks.
kill -9 "$OS_PID" 2>/dev/null || true
wait "$OS_PID" 2>/dev/null || true
echo "=== openspiel done ==="

echo "=== reinfors: ${MINUTES}m -> ${RF_OUT} ==="
rm -rf "$RF_OUT"
# SAME external hard deadline as the OpenSpiel side: the internal --minutes check only runs
# BETWEEN stream batches (a 21,845-record next() can block far past T, then write a final
# checkpoint the other side was denied). SIGKILL at T on both stacks makes the h2h artifact
# genuinely symmetric: the last PERIODIC checkpoint written before T (--checkpoint-every).
.venv23/bin/python benchmarks/openspiel/train_reinfors_az.py \
  --minutes "$MINUTES" --out "$RF_OUT" \
  --seed 0 --n-games 8 --sims 64 --c-puct 2.0 \
  --infer-cache 262144 --collect-size 21845 \
  > "${RF_OUT}.stdout" 2>&1 &
RF_PID=$!
sleep "$SECS"
kill -9 "$RF_PID" 2>/dev/null || true
wait "$RF_PID" 2>/dev/null || true
echo "=== reinfors done (hard deadline) ==="
tail -4 "${RF_OUT}.stdout"
