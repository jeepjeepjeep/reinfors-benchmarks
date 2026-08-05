"""Tensorboard-style comparison panels for one matched round, from the two learners' own logs:

  reinfors   <rf_dir>/learner.jsonl        (one line per minibatch)
  openspiel  <os_dir>/log-learner.txt      (one block per learn step)

Losses are definitionally aligned (masked CE over legal actions, value MSE on z in [-1,1];
their separate l2 term is ignored) but each side's loss is measured on ITS OWN self-play
distribution — read the curves as per-system learning progress, not head-to-head quality.

  uv run python scripts/plot_round.py results/round_chess_rf_120m_n64 \
      results/round_chess_os_120m_a16_b16 -o results/round1_panels.png
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
COLLECTED = re.compile(r"Collected (\d+) states from (\d+) games, ([\d.]+) states/s.*game length: ([\d.]+)")
SEEN = re.compile(r"States seen: (\d+)")
LOSSES = re.compile(r"Losses: policy: ([\d.]+), value: ([\d.]+)")
HITRATE = re.compile(r"hit rate: ([\d.]+)%")


def load_rf(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in (path / "learner.jsonl").open()]
    out = {k: np.array([r[k] for r in rows]) for k in ("wall", "states", "policy_loss", "value_loss")}
    hits = np.array([r.get("cache_hits", 0) for r in rows], dtype=float)
    lookups = np.array([r.get("cache_lookups", 0) for r in rows], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["hit_rate"] = np.where(lookups > 0, 100.0 * hits / lookups, np.nan)  # cumulative
    return out


def load_os(path: Path) -> dict[str, np.ndarray]:
    """One sample per learn step: the values most recently seen when a Losses line arrives."""
    t0 = None
    cur: dict[str, float] = {}
    samples = []
    for line in (path / "log-learner.txt").open(errors="ignore"):
        m = TS.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            t0 = t0 or ts
            cur["wall"] = (ts - t0).total_seconds()
        m = COLLECTED.search(line)
        if m:
            cur["states_s"] = float(m.group(3))
            cur["game_len"] = float(m.group(4))
        m = SEEN.search(line)
        if m:
            cur["states"] = int(m.group(1))
        m = HITRATE.search(line)
        if m:
            cur["hit_rate"] = float(m.group(1))  # per learn interval (their cache clears per step)
        m = LOSSES.search(line)
        if m:
            cur["policy_loss"], cur["value_loss"] = float(m.group(1)), float(m.group(2))
            samples.append(dict(cur))
    keys = ("wall", "states", "policy_loss", "value_loss", "hit_rate", "game_len", "states_s")
    return {k: np.array([s.get(k, np.nan) for s in samples]) for k in keys}


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if len(y) <= k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rf_dir", type=Path)
    ap.add_argument("os_dir", type=Path)
    ap.add_argument("-o", "--out", default="round_panels.png")
    ap.add_argument("--smooth", type=int, default=51, help="rf minibatch-loss rolling window")
    args = ap.parse_args()

    rf, os_ = load_rf(args.rf_dir), load_os(args.os_dir)
    k = args.smooth
    rf_h = rf["wall"] / 3600
    rf_h_s = rf_h[k - 1:] if len(rf_h) > k else rf_h
    os_h = os_["wall"] / 3600

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("matched 2h round — each side's learner telemetry (losses on own self-play data)")

    a = ax[0][0]
    a.plot(rf_h_s, smooth(rf["policy_loss"], k), label="reinfors")
    a.plot(os_h, os_["policy_loss"], label="openspiel", marker="o", ms=3)
    a.set(title="policy loss (masked CE) vs wall-clock", xlabel="hours")
    a.legend()

    a = ax[0][1]
    a.plot(rf_h_s, smooth(rf["value_loss"], k), label="reinfors")
    a.plot(os_h, os_["value_loss"], label="openspiel", marker="o", ms=3)
    a.set(title="value loss (MSE on z) vs wall-clock", xlabel="hours")
    a.legend()

    a = ax[0][2]
    rf_states = rf["states"][k - 1:] if len(rf["states"]) > k else rf["states"]
    a.plot(rf_states / 1e6, smooth(rf["policy_loss"], k), label="reinfors")
    a.plot(os_["states"] / 1e6, os_["policy_loss"], label="openspiel", marker="o", ms=3)
    a.set(title="policy loss vs states consumed (learning per data)", xlabel="states (M)")
    a.legend()

    a = ax[1][0]
    a.plot(rf_h, rf["states"] / 1e6, label="reinfors")
    a.plot(os_h, os_["states"] / 1e6, label="openspiel", marker="o", ms=3)
    a.set(title="cumulative training states", xlabel="hours", ylabel="states (M)")
    a.legend()

    a = ax[1][1]
    a.plot(rf_h, rf["hit_rate"], label="reinfors (cumulative)")
    a.plot(os_h, os_["hit_rate"], label="openspiel (per step)", marker="o", ms=3)
    a.set(title="infer-cache hit rate", xlabel="hours", ylabel="%")
    a.legend()

    a = ax[1][2]
    a.plot(os_h, os_["game_len"], label="openspiel", marker="o", ms=3, color="tab:orange")
    a.set(title="self-play game length (openspiel only — rf gap, not logged)", xlabel="hours", ylabel="plies")
    a.legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
