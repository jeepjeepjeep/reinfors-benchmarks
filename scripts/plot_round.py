"""Tensorboard-style comparison panels for one matched round, from the two learners' own
structured logs:

  reinfors   <rf_dir>/telemetry.jsonl (one line per minibatch -> aggregated per collection cycle)
  openspiel  <os_dir>/learner.jsonl   (one record per learn step, DataLogger jsonl)

Both series are shown at the SAME aggregation: one point per collection/learning cycle
(reinfors minibatches are grouped by their cumulative-states plateau — the `states` counter
only advances when a collect batch lands, so equal values delimit one cycle's burst).
Cache hit rates are per-interval on both sides (theirs clears per step; ours from counter
deltas between cycles). Losses are definitionally aligned (masked CE, value MSE on z) but
measured on each side's OWN self-play distribution — per-system learning progress, not
head-to-head quality.

  uv run python scripts/plot_round.py runs/v1_training/rf_train/cycle1/training \
      runs/v1_training/os_train/cycle1/training -o /tmp/round_panels.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rf(path: Path) -> dict[str, np.ndarray]:
    """One point per collection cycle: rows grouped by their cumulative-states plateau."""
    rows = [json.loads(line) for line in (path / "telemetry.jsonl").open()]
    cycles: list[dict[str, float]] = []
    for r in rows:
        if not cycles or r["states"] != cycles[-1]["states"]:
            cycles.append({"states": r["states"], "policy": [], "value": [], "last": r})
        c = cycles[-1]
        c["policy"].append(r["policy_loss"])
        c["value"].append(r["value_loss"])
        c["last"] = r
    out = {
        "wall": np.array([c["last"]["wall"] for c in cycles]),
        "states": np.array([c["states"] for c in cycles], dtype=float),
        "policy_loss": np.array([np.mean(c["policy"]) for c in cycles]),
        "value_loss": np.array([np.mean(c["value"]) for c in cycles]),
    }
    # per-interval hit rate from cumulative counter deltas (counters start at 0, so the
    # first cycle's cumulative value IS its interval value)
    hits = np.array([c["last"].get("cache_hits", 0) for c in cycles], dtype=float)
    looks = np.array([c["last"].get("cache_lookups", 0) for c in cycles], dtype=float)
    dh, dl = np.diff(hits, prepend=0.0), np.diff(looks, prepend=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["hit_rate"] = np.where(dl > 0, 100.0 * dh / dl, np.nan)
    return out


def load_os(path: Path) -> dict[str, np.ndarray]:
    """One point per learn step from their DataLogger jsonl (exact, unshifted, unrounded)."""
    rows = [json.loads(line) for line in (path / "learner.jsonl").open()]
    return {
        "wall": np.array([r["time_rel"] for r in rows]),
        "states": np.array([r["total_states"] for r in rows], dtype=float),
        "policy_loss": np.array([r["loss"]["policy"] for r in rows]),
        "value_loss": np.array([r["loss"]["value"] for r in rows]),
        "hit_rate": np.array([100.0 * r["cache"]["hit_rate"] for r in rows]),
        "game_len": np.array([r["game_length"]["avg"] for r in rows]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rf_dir", type=Path)
    ap.add_argument("os_dir", type=Path)
    ap.add_argument("-o", "--out", default="round_panels.png")
    args = ap.parse_args()

    rf, os_ = load_rf(args.rf_dir), load_os(args.os_dir)
    rf_h, os_h = rf["wall"] / 3600, os_["wall"] / 3600
    style = dict(marker="o", ms=2.5, lw=1)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "matched 2h round — learner telemetry, one point per collection cycle "
        "(losses on each side's own self-play data)"
    )

    a = ax[0][0]
    a.plot(rf_h, rf["policy_loss"], label="reinfors", **style)
    a.plot(os_h, os_["policy_loss"], label="openspiel", **style)
    a.set(title="policy loss (masked CE) vs wall-clock", xlabel="hours")
    a.legend()

    a = ax[0][1]
    a.plot(rf_h, rf["value_loss"], label="reinfors", **style)
    a.plot(os_h, os_["value_loss"], label="openspiel", **style)
    a.set(title="value loss (MSE on z) vs wall-clock", xlabel="hours")
    a.legend()

    a = ax[0][2]
    a.plot(rf["states"] / 1e6, rf["policy_loss"], label="reinfors", **style)
    a.plot(os_["states"] / 1e6, os_["policy_loss"], label="openspiel", **style)
    a.set(title="policy loss vs states collected (learning per data)", xlabel="states (M)")
    a.legend()

    a = ax[1][0]
    a.plot(rf_h, rf["states"] / 1e6, label="reinfors", **style)
    a.plot(os_h, os_["states"] / 1e6, label="openspiel", **style)
    a.set(title="cumulative states collected", xlabel="hours", ylabel="states (M)")
    a.legend()

    a = ax[1][1]
    a.plot(rf_h, rf["hit_rate"], label="reinfors", **style)
    a.plot(os_h, os_["hit_rate"], label="openspiel", **style)
    a.set(title="infer-cache hit rate, per interval (both sides)", xlabel="hours", ylabel="%")
    a.legend()

    a = ax[1][2]
    a.plot(os_h, os_["game_len"], label="openspiel", color="tab:orange", **style)
    a.set(title="self-play game length (openspiel only — rf gap, not logged)",
          xlabel="hours", ylabel="plies")
    a.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
