#!/usr/bin/env bash
# Phase 0, OpenSpiel side: short self-play legs of their C++ AZ across (nn_width, nn_depth) x
# (actors x inference_batch_size) x device, to locate where their stack wants the GPU. Run on
# the bench box AFTER building open_spiel with CUDA libtorch (scripts/setup_openspiel_cpp.sh).
#
# Flag names match run_round_matched.sh / their alpha_zero_torch_example. Device strings pass
# through (slash stripped) to libtorch's c10::Device parser, so CUDA is "/cuda:0" — "/gpu:0"
# is TF naming and is rejected (measured on the bench box).
#
# Isolation: the taskset below pins their whole process tree to CORES; keep it identical to the
# core set used for the reinfors sweep legs. One leg at a time, nothing else on the box.
#
#   CORES=0-3 LEG_SECONDS=120 GAME=chess bash scripts/phase0_openspiel_sweep.sh
#
# MEASUREMENT (fixed 2026-08-03): throughput is computed from INTERIOR counter deltas — the
# [inst] rows counter sampled at two timestamps mid-run, steady rows/s = drows/dt. The old
# method (total rows / nominal LEG_SECONDS) systematically INFLATED OpenSpiel's numbers by
# ~30-40%: the SIGINT drain let actors keep generating rows past the nominal cutoff while the
# denominator stayed fixed (reinfors legs are internally timed, so the comparison was biased).
# Raw logs + learner summary are kept for the record.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=open_spiel_cpp/open_spiel/build/examples/alpha_zero_torch_example
CORES="${CORES:-0-3}"
LEG_SECONDS="${LEG_SECONDS:-120}"   # interior measurement window
WARMUP="${WARMUP:-45}"              # settle time before the first counter sample
GAME="${GAME:-chess}"           # their game name: chess | connect_four
WIDTHS="${WIDTHS:-32 64 128}"
DEPTHS="${DEPTHS:-1 4}"
ACTORS="${ACTORS:-1 4 8}"       # inference_batch_size is tied to actors below (full batch)
DEVICES="${DEVICES:-/cpu:0 /cuda:0}"
OUT_ROOT=results/phase0_os
mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.txt"

for device in $DEVICES; do
  for width in $WIDTHS; do
    for depth in $DEPTHS; do
      for actors in $ACTORS; do
        tag="$(echo "${GAME}_${device}_w${width}_d${depth}_a${actors}" | tr -c 'A-Za-z0-9_\n' '_')"
        out="$OUT_ROOT/$tag"
        echo "=== $tag (${LEG_SECONDS}s) ==="
        rm -rf "$out"
        taskset -c "$CORES" "$BIN" --game="$GAME" --path="$out" \
          --actors="$actors" --evaluators=0 --devices="$device" \
          --max_simulations=64 --uct_c=2 --policy_alpha=0.3 --policy_epsilon=0.25 \
          --temperature=1 --temperature_drop=10 \
          --nn_model=resnet --nn_width="$width" --nn_depth="$depth" \
          --inference_batch_size="$actors" --inference_threads=1 --inference_cache=0 \
          --replay_buffer_size=65536 --replay_buffer_reuse=3 --train_batch_size=1024 \
          --learning_rate=0.0001 --weight_decay=0.0001 \
          --checkpoint_freq=1000000 --evaluation_window=100 --eval_levels=7 \
          --cutoff_probability=0 --cutoff_value=0.95 --explicit_learning=false \
          --max_steps=0 > "${out}.stdout" 2>&1 &
        pid=$!
        sleep "$WARMUP"
        t1=$(date +%s.%N); r1=$(grep -o "rows=[0-9]*" "${out}.stdout" 2>/dev/null | tail -1 | cut -d= -f2)
        sleep "$LEG_SECONDS"
        t2=$(date +%s.%N); r2=$(grep -o "rows=[0-9]*" "${out}.stdout" 2>/dev/null | tail -1 | cut -d= -f2)
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        # Primary metric: the instrumented evaluator's cumulative counters on stdout
        # ("[inst] req=.. hits=.. fwd=.. rows=.."): rows/leg = net rows/s, the same metric as
        # the reinfors sweep, and robust to games not finishing within the leg (chess CPU legs
        # complete ZERO games). Learner "Collected" summary + ply count are appended when
        # present. Every grep is no-match-safe: set -euo pipefail aborts the sweep otherwise.
        inst=$(grep -h '\[inst\]' "${out}.stdout" 2>/dev/null | tail -1 || true)
        rows=$(echo "$inst" | grep -oE 'rows=[0-9]+' | grep -oE '[0-9]+' || true)
        collected=$(grep -h 'Collected' "$out"/log-learner.txt 2>/dev/null | tail -1 || true)
        plies=$(grep -h 'Actions:' "$out"/log-actor* 2>/dev/null | sed 's/.*Actions://' | wc -w || true)
        if [[ -n "${r1:-}" && -n "${r2:-}" && "${r2:-0}" -gt "${r1:-0}" ]]; then
          rows_s=$(awk "BEGIN{printf \"%.1f\", ($r2 - $r1) / ($t2 - $t1)}")
          method="steady(dt=$(awk "BEGIN{printf \"%.0f\", $t2 - $t1}")s)"
        else
          # no interior samples (counter prints every 8192 requests; very slow legs may miss the
          # window) — fall back to total/nominal, EXPLICITLY marked as drain-inflated.
          rows_s=$(awk "BEGIN{printf \"%.1f\", ${rows:-0}/$LEG_SECONDS}")
          method="DRAIN-INFLATED-FALLBACK"
        fi
        echo "$tag  method=$method  net_rows=${rows:-0}  rows_s=$rows_s  plies=${plies:-0}${collected:+  theirs: ${collected#*] }}" | tee -a "$SUMMARY"
      done
    done
  done
done
echo "done -> $SUMMARY (verify one leg's raw logs before trusting the parse)"
