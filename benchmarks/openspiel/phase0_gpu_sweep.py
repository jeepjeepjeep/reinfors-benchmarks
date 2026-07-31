"""Phase 0 of the GPU round: find the (net size x batch) regime where CUDA actually beats CPU.

A GPU head-to-head is only meaningful at an operating point where BOTH stacks want the GPU;
this sweep locates that point before any framework comparison. Two parts:

  net    — pure forwards of a width/depth-parameterized replica of open_spiel's AZ resnet
           family (same family their --nn_width/--nn_depth select), swept over batch rows and
           devices. Kernels are the same ATen either way, so this surface applies to both
           stacks; per-row wrapper costs (known constants from the CPU rounds) shift it only
           slightly.
  engine — reinfors end-to-end AZ self-play (Engine.collect + callback net), swept over
           n_games and devices: realized net rows/s, achieved batch rows/call, and % of wall
           in the net. This is where batch-1 GPU inference goes to die (measured on MPS) and
           pooled n_games rescues it — the sweep finds the CUDA crossover.

The verdict table reports, per net config, the smallest batch (net) / n_games (engine) where
the CUDA:CPU rows/s ratio clears --gpu-threshold (default 2.0). Pick the head-to-head
operating point from there.

Isolation (do BOTH on the bench box):
  - pin the process to fixed physical cores:   taskset -c 0-3 .venv23/bin/python ...
  - intra-op threads are pinned here via --torch-threads (default 4); the process affinity
    and thread count are echoed into every result row.

    taskset -c 0-3 .venv23/bin/python benchmarks/openspiel/phase0_gpu_sweep.py \
        --mode both --game chess --devices cpu,cuda --out results/phase0/sweep.jsonl
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import reinfors as rf
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import seed_all  # noqa: E402


class SweepResnet(nn.Module):
    """open_spiel alpha_zero_torch resnet family (their model.cc), width/depth parameterized —
    the same net their --nn_width/--nn_depth flags build, so one (width, depth) point here maps
    directly onto their side of the head-to-head. Structure per AZResnetReplica (input conv+BN,
    `depth` residual blocks, both heads; BN eps/momentum match model.cc)."""

    def __init__(self, in_channels: int, h: int, w: int, n_actions: int, width: int, depth: int) -> None:
        super().__init__()
        self.in_channels, self.h, self.w, self.n_actions = in_channels, h, w, n_actions
        bn = dict(eps=0.001, momentum=0.01)
        self.in_conv = nn.Conv2d(in_channels, width, 3, padding=1)
        self.in_bn = nn.BatchNorm2d(width, **bn)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    dict(
                        conv1=nn.Conv2d(width, width, 3, padding=1),
                        bn1=nn.BatchNorm2d(width, **bn),
                        conv2=nn.Conv2d(width, width, 3, padding=1),
                        bn2=nn.BatchNorm2d(width, **bn),
                    )
                )
            )
        self.value_conv = nn.Conv2d(width, 1, 1)
        self.value_bn = nn.BatchNorm2d(1, **bn)
        self.value_l1 = nn.Linear(h * w, width)
        self.value_l2 = nn.Linear(width, 1)
        self.policy_conv = nn.Conv2d(width, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2, **bn)
        self.policy_lin = nn.Linear(2 * h * w, n_actions)

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.in_bn(self.in_conv(x)))
        for b in self.blocks:
            y = torch.relu(b["bn1"](b["conv1"](x)))
            y = b["bn2"](b["conv2"](y))
            x = torch.relu(x + y)
        v = torch.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = torch.tanh(self.value_l2(torch.relu(self.value_l1(v)))).squeeze(-1)
        p = torch.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        return self.policy_lin(p), v


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def device_ok(device: str) -> bool:
    if device.startswith("cuda"):
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def build_game(name: str):
    if name == "chess":
        game = rf.games.Chess(encoder=rf.encoders.OpenSpielChess(), max_ticks=None)
        return game, 4674  # their head width; pi occupies 0..4671, top two dead (see ledger)
    game = rf.games.Connect4()
    return game, game.action_space().n


def bench_net(args, shape, head_actions, results) -> None:
    c, h, w = shape
    for width, depth, device in itertools.product(args.widths, args.depths, args.devices):
        if not device_ok(device):
            print(f"net    w{width} d{depth} [{device}]  SKIP (device unavailable)")
            continue
        seed_all()
        net = SweepResnet(c, h, w, head_actions, width, depth).to(device).eval()
        for batch in args.batches:
            x = torch.randn(batch, c, h, w, device=device)
            with torch.no_grad():
                for _ in range(args.warmup_calls):
                    net.heads(x)
                sync(device)
                t0 = time.perf_counter()
                calls = 0
                while time.perf_counter() - t0 < args.net_leg_seconds:
                    net.heads(x)
                    calls += 1
                sync(device)
                wall = time.perf_counter() - t0
            rows_s = calls * batch / wall
            row = dict(
                part="net", game=args.game, width=width, depth=depth, device=device,
                batch=batch, rows_s=rows_s, us_per_row=1e6 * wall / (calls * batch),
                calls=calls, wall=wall,
            )
            results.append(row)
            print(f"net    w{width:<4} d{depth} [{device:4}] batch {batch:<5} {rows_s:>10.0f} rows/s  {row['us_per_row']:>8.1f} us/row")


def bench_engine(args, game, head_actions, results) -> None:
    shape = game.observation_space().shape
    c, h, w = shape
    actions = game.action_space().n
    for width, depth, device in itertools.product(args.widths, args.depths, args.devices):
        if not device_ok(device):
            print(f"engine w{width} d{depth} [{device}]  SKIP (device unavailable)")
            continue
        for n_games in args.n_games:
            seed_all()
            net = SweepResnet(c, h, w, head_actions, width, depth).to(device).eval()

            def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                with torch.no_grad():
                    x = torch.from_numpy(np.ascontiguousarray(obs_batch)).reshape(-1, c, h, w).to(device)
                    logits, values = net.heads(x)
                return logits[:, :actions].cpu().double().numpy(), values.cpu().double().numpy()

            engine = rf.Engine(
                game,
                rf.Reward(win=1.0, loss=-1.0),
                rf.policies.AlphaZero(
                    num_simulations=args.sims,
                    c_puct=args.c_puct,
                    noise=rf.noise.Dirichlet(epsilon=0.25, alpha=0.3),
                    temperature=1.0,
                    temperature_drop=10,
                ),
                rf.learners.AlphaZero(gamma=1.0),
                n_games=n_games,
                seed=args.seed,
                infer_cache=args.infer_cache,
            )
            engine.collect(args.warmup_records, infer)  # torch/device warmup outside the clock
            t0 = time.perf_counter()
            rows = calls = decisions = records = 0
            net_seconds = 0.0
            while time.perf_counter() - t0 < args.engine_leg_seconds:
                batch = engine.collect(args.chunk_records, infer)
                tel = batch.telemetry
                rows += int(tel["infer_rows"])
                calls += int(tel["infer_calls"])
                decisions += int(tel["decisions"])
                net_seconds += float(tel["infer_seconds"])
                records += batch.obs.shape[0]
            wall = time.perf_counter() - t0
            row = dict(
                part="engine", game=args.game, width=width, depth=depth, device=device,
                n_games=n_games, rows_s=rows / wall, moves_s=decisions / wall,
                achieved_batch=rows / max(calls, 1), net_share=net_seconds / wall,
                records=records, wall=wall, sims=args.sims, infer_cache=args.infer_cache,
            )
            results.append(row)
            print(
                f"engine w{width:<4} d{depth} [{device:4}] n_games {n_games:<4} "
                f"{row['rows_s']:>9.0f} rows/s  {row['moves_s']:>7.1f} moves/s  "
                f"batch {row['achieved_batch']:>6.1f}  net {row['net_share'] * 100:5.1f}%"
            )


def verdict(results, args) -> None:
    """Per net config: the smallest batch / n_games where CUDA clears the threshold vs CPU."""
    print(f"\n--- verdict (cuda >= {args.gpu_threshold}x cpu rows/s) ---")
    for part, lever in (("net", "batch"), ("engine", "n_games")):
        rows = [r for r in results if r["part"] == part]
        for width, depth in itertools.product(args.widths, args.depths):
            by_dev: dict[str, dict[int, float]] = {}
            for r in rows:
                if r["width"] == width and r["depth"] == depth:
                    by_dev.setdefault(r["device"], {})[r[lever]] = r["rows_s"]
            cpu, cuda = by_dev.get("cpu", {}), by_dev.get("cuda", {})
            if not cpu or not cuda:
                continue
            crossing = [b for b in sorted(cuda) if b in cpu and cuda[b] >= args.gpu_threshold * cpu[b]]
            at = f"{lever} >= {crossing[0]}" if crossing else "NEVER in swept range"
            best = max((cuda[b] / cpu[b] for b in cuda if b in cpu), default=float("nan"))
            print(f"{part:6} w{width:<4} d{depth}: cuda wins at {at}  (best ratio {best:.2f}x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["net", "engine", "both"], default="both")
    ap.add_argument("--game", choices=["chess", "connect4"], default="chess")
    ap.add_argument("--devices", type=str, default="cpu,cuda")
    ap.add_argument("--widths", type=str, default="32,64,128")
    ap.add_argument("--depths", type=str, default="1,4")
    ap.add_argument("--batches", type=str, default="1,8,32,128,512", help="net part: rows per forward")
    ap.add_argument("--n-games", type=str, default="1,8,32", help="engine part: parallel games")
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--infer-cache", type=int, default=0, help="0 = off; the round itself runs cache ON")
    ap.add_argument("--net-leg-seconds", type=float, default=3.0)
    ap.add_argument("--engine-leg-seconds", type=float, default=20.0)
    ap.add_argument("--warmup-calls", type=int, default=20)
    ap.add_argument("--warmup-records", type=int, default=32)
    ap.add_argument("--chunk-records", type=int, default=64)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--gpu-threshold", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="", help="append result rows as jsonl")
    args = ap.parse_args()
    args.devices = args.devices.split(",")
    args.widths = [int(x) for x in args.widths.split(",")]
    args.depths = [int(x) for x in args.depths.split(",")]
    args.batches = [int(x) for x in args.batches.split(",")]
    args.n_games = [int(x) for x in args.n_games.split(",")]

    torch.set_num_threads(args.torch_threads)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a (macOS)"
    header = dict(
        game=args.game, torch=torch.__version__, build=rf._reinfors.core_build_profile(),
        torch_threads=torch.get_num_threads(), affinity=str(affinity),
        cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    print(" ".join(f"{k}={v}" for k, v in header.items()))
    if header["build"] != "release":
        print("WARNING: reinfors is a DEBUG build — numbers are meaningless (~10x slow)")

    game, head_actions = build_game(args.game)
    results: list[dict] = []
    if args.mode in ("net", "both"):
        bench_net(args, game.observation_space().shape, head_actions, results)
    if args.mode in ("engine", "both"):
        bench_engine(args, game, head_actions, results)
    verdict(results, args)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(dict(header=header)) + "\n")
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} rows -> {out}")


if __name__ == "__main__":
    main()
