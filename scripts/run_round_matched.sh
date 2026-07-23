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
kill -INT "$OS_PID" 2>/dev/null || true       # graceful: their StopToken saves and exits
for _ in $(seq 1 60); do kill -0 "$OS_PID" 2>/dev/null || break; sleep 5; done
kill -TERM "$OS_PID" 2>/dev/null || true
wait "$OS_PID" 2>/dev/null || true
echo "=== openspiel done ==="

echo "=== reinfors: ${MINUTES}m -> ${RF_OUT} ==="
rm -rf "$RF_OUT"
.venv23/bin/python benchmarks/openspiel/train_reinfors_az.py \
  --minutes "$MINUTES" --out "$RF_OUT" \
  --seed 0 --n-games 8 --sims 64 --c-puct 2.0 \
  --infer-cache 262144 --collect-size 21845 \
  > "${RF_OUT}.stdout" 2>&1
echo "=== reinfors done ==="
tail -4 "${RF_OUT}.stdout"
