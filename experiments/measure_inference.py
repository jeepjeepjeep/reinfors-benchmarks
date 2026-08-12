"""Batch/size curves: find the (net size x batch) regime where CUDA actually beats CPU.

(Historically "phase 0" of the GPU round — the sweep that located the head-to-head
operating point before any framework comparison.)

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

    taskset -c 0-3 .venv23/bin/python experiments/measure_inference.py \
        --mode both --game chess --devices cpu,cuda --out /tmp/inference
    # --devices is a sweep: each point runs per device; --mode picks the surface
    # (kernel = pure forwards, engine = data-gen loop)
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

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import manifest
import protocol
from common import SweepResnet, seed_all


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
        return (
            game,
            4674,
        )  # their head width; pi occupies 0..4671, top two dead (see ledger)
    game = rf.games.Connect4()
    return game, game.action_space().n


def bench_net(args, shape, head_actions, results) -> None:
    c, h, w = shape
    for width, depth, device in itertools.product(
        args.widths, args.depths, args.devices
    ):
        if not device_ok(device):
            print(f"net    w{width} d{depth} [{device}]  SKIP (device unavailable)")
            continue
        seed_all()
        net = SweepResnet(c, h, w, head_actions, width, depth).to(device).eval()
        heads = (
            torch.compile(net.heads, mode="reduce-overhead")
            if args.callback == "compiled"
            else net.heads
        )
        for batch in args.batches:
            x = torch.randn(batch, c, h, w, device=device)
            with torch.no_grad():
                for _ in range(args.warmup_calls):
                    heads(x)
                sync(device)
                t0 = time.perf_counter()
                calls = 0
                while time.perf_counter() - t0 < args.net_leg_seconds:
                    heads(x)
                    calls += 1
                sync(device)
                wall = time.perf_counter() - t0
            rows_s = calls * batch / wall
            row = dict(
                part="net",
                game=args.game,
                width=width,
                depth=depth,
                device=device,
                batch=batch,
                rows_s=rows_s,
                us_per_row=1e6 * wall / (calls * batch),
                calls=calls,
                wall=wall,
            )
            results.append(row)
            print(
                f"net    w{width:<4} d{depth} [{device:4}] batch {batch:<5} {rows_s:>10.0f} rows/s  {row['us_per_row']:>8.1f} us/row"
            )


def bench_engine(args, game, head_actions, results) -> None:
    import threading

    shape = game.observation_space().shape
    c, h, w = shape
    actions = game.action_space().n
    for width, depth, device in itertools.product(
        args.widths, args.depths, args.devices
    ):
        if not device_ok(device):
            print(f"engine w{width} d{depth} [{device}]  SKIP (device unavailable)")
            continue
        for n_games, engines in itertools.product(args.n_games, args.engines):
            seed_all()
            net = SweepResnet(c, h, w, head_actions, width, depth).to(device).eval()
            heads = (
                torch.compile(net.heads, mode="reduce-overhead")
                if args.callback == "compiled"
                else net.heads
            )

            noop_l = np.zeros(
                (max(n_games * max(engines, 1) + 8, 8), actions), dtype=np.float32
            )
            noop_v = np.zeros((noop_l.shape[0],), dtype=np.float32)

            # per-THREAD pinned staging: with --engines > 1 the callback runs
            # concurrently from several engine threads, and a shared buffer would be
            # a data race between one thread's H2D and another's staging copy
            pin_cap = n_games * max(engines, 1)
            pin_tls = (
                threading.local()
                if (args.callback in ("fast", "compiled") and device.startswith("cuda"))
                else None
            )

            def pinned_staging() -> "torch.Tensor":
                buf = getattr(pin_tls, "buf", None)
                if buf is None:
                    buf = torch.empty((pin_cap, c * h * w), pin_memory=True)
                    pin_tls.buf = buf
                return buf

            def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                if args.callback == "noop":
                    # No torch at all: measures engine + Rust roundtrip + binding pack/widen,
                    # isolating the device-independent CPU residual (RF_INFER_TIMING gives pack).
                    m = obs_batch.shape[0]
                    return noop_l[:m], noop_v[:m]
                # Shared eval-mode net; concurrent forwards from several engine threads are
                # the POINT when engines > 1 (one engine's CPU phase overlaps another's
                # forward — a script-level approximation of the collect_async double-buffer).
                if args.callback in ("fast", "compiled"):
                    # Plumbing-only fast path (kernels identical; needs the PR #136 build):
                    # inference_mode; pinned staging for the H2D; NO pre-transfer slice (the
                    # binding accepts padded logits — a slice forces a device-side gather);
                    # ONE packed D2H (logits+value), handed over with zero python slicing:
                    # the packed array IS the padded logits, values ride as its last column.
                    n = obs_batch.shape[0]
                    with torch.inference_mode():
                        src = torch.from_numpy(np.ascontiguousarray(obs_batch))
                        if pin_tls is not None and n <= pin_cap:
                            staging = pinned_staging()[:n]
                            staging.copy_(src.reshape(n, -1))
                            x = staging.to(device, non_blocking=True).reshape(
                                n, c, h, w
                            )
                        else:
                            x = src.reshape(-1, c, h, w).to(device)
                        logits, values = heads(x)
                        out = torch.cat([logits, values.unsqueeze(1)], dim=1)
                        if args.infer_dtype == "f64":
                            # the A/B's f64 arm: device-side widen + double-width
                            # D2H — the pre-f32-contract cost profile
                            out = out.double()
                        packed = out.cpu().numpy()
                    return packed, packed[:, -1]
                with torch.no_grad():
                    x = (
                        torch.from_numpy(np.ascontiguousarray(obs_batch))
                        .reshape(-1, c, h, w)
                        .to(device)
                    )
                    logits, values = heads(x)
                if (
                    args.infer_dtype == "f32"
                ):  # native f32: the binding widens exactly (PR #136)
                    return logits[:, :actions].cpu().numpy(), values.cpu().numpy()
                return logits[
                    :, :actions
                ].cpu().double().numpy(), values.cpu().double().numpy()

            def make_engine(idx: int) -> "rf.Engine":
                return rf.Engine(
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
                    seed=args.seed + idx,
                    infer_cache=args.infer_cache,
                )

            engs = [make_engine(e) for e in range(engines)]
            for eng in engs:
                eng.collect(
                    args.warmup_records, infer
                )  # torch/device warmup outside the clock

            totals = [
                dict(rows=0, calls=0, decisions=0, records=0, net_seconds=0.0)
                for _ in engs
            ]
            deadline = time.perf_counter() + args.engine_leg_seconds

            def run(eng, tot) -> None:
                while time.perf_counter() < deadline:
                    batch = eng.collect(args.chunk_records, infer)
                    tel = batch.telemetry
                    tot["rows"] += int(tel["infer_rows"])
                    tot["calls"] += int(tel["infer_calls"])
                    tot["decisions"] += int(tel["decisions"])
                    tot["net_seconds"] += float(tel["infer_seconds"])
                    tot["records"] += batch.obs.shape[0]

            t0 = time.perf_counter()
            threads = [
                threading.Thread(target=run, args=(eng, tot))
                for eng, tot in zip(engs, totals)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            rows = sum(t["rows"] for t in totals)
            calls = sum(t["calls"] for t in totals)
            decisions = sum(t["decisions"] for t in totals)
            net_seconds = sum(
                t["net_seconds"] for t in totals
            )  # summed thread-time, > wall when overlapping
            row = dict(
                part="engine",
                game=args.game,
                width=width,
                depth=depth,
                device=device,
                n_games=n_games,
                engines=engines,
                rows_s=rows / wall,
                moves_s=decisions / wall,
                achieved_batch=rows / max(calls, 1),
                net_share=net_seconds / wall,
                records=sum(t["records"] for t in totals),
                wall=wall,
                sims=args.sims,
                infer_cache=args.infer_cache,
                infer_dtype=args.infer_dtype,
                callback=args.callback,
            )
            results.append(row)
            print(
                f"engine w{width:<4} d{depth} [{device:4}] {engines}x{n_games:<4} "
                f"{row['rows_s']:>9.0f} rows/s  {row['moves_s']:>7.1f} moves/s  "
                f"batch {row['achieved_batch']:>6.1f}  net-thread {row['net_share'] * 100:5.1f}%"
            )


def verdict(results, args) -> None:
    """Per net config: the smallest batch / n_games where CUDA clears the threshold vs CPU."""
    print(f"\n--- verdict (cuda >= {args.gpu_threshold}x cpu rows/s) ---")
    for part, lever in (("net", "batch"), ("engine", "n_games")):
        rows = [r for r in results if r["part"] == part and r.get("engines", 1) == 1]
        for width, depth in itertools.product(args.widths, args.depths):
            by_dev: dict[str, dict[int, float]] = {}
            for r in rows:
                if r["width"] == width and r["depth"] == depth:
                    by_dev.setdefault(r["device"], {})[r[lever]] = r["rows_s"]
            cpu, cuda = by_dev.get("cpu", {}), by_dev.get("cuda", {})
            if not cpu or not cuda:
                continue
            crossing = [
                b
                for b in sorted(cuda)
                if b in cpu and cuda[b] >= args.gpu_threshold * cpu[b]
            ]
            at = f"{lever} >= {crossing[0]}" if crossing else "NEVER in swept range"
            best = max(
                (cuda[b] / cpu[b] for b in cuda if b in cpu), default=float("nan")
            )
            print(
                f"{part:6} w{width:<4} d{depth}: cuda wins at {at}  (best ratio {best:.2f}x)"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["kernel", "engine", "both"],
        default="both",
        help="measurement surface: kernel = pure net forwards (no engine), "
        "engine = the self-play data-gen loop with the net as callback",
    )
    ap.add_argument("--game", choices=["chess", "connect4"], default="chess")
    ap.add_argument(
        "--devices",
        type=str,
        default="cpu,cuda",
        help="SWEEP list: every point runs once per device; the verdict compares "
        "them (pass a single device to skip the crossover comparison)",
    )
    ap.add_argument("--widths", type=str, default="32,64,128")
    ap.add_argument("--depths", type=str, default="1,4")
    ap.add_argument(
        "--batches",
        type=str,
        default="1,8,32,128,512",
        help="net part: rows per forward",
    )
    ap.add_argument(
        "--n-games",
        type=str,
        default="1,8,32",
        help="engine part: parallel games PER ENGINE",
    )
    ap.add_argument(
        "--engines",
        type=str,
        default="1",
        help="engine part: concurrent engines (threads) sharing the net",
    )
    ap.add_argument(
        "--infer-dtype",
        choices=["f64", "f32"],
        default="f64",
        help="callback output dtype (f32 needs the f32-contract build)",
    )
    ap.add_argument(
        "--callback",
        choices=["legacy", "fast", "compiled", "noop"],
        default="legacy",
        help="fast = inference_mode + pinned H2D + no slice + single packed D2H; "
        'compiled = the fast path with torch.compile(mode="reduce-overhead") on the '
        "forward (targets the per-call dispatch residual); noop = no-torch "
        "(residual decomposition)",
    )
    ap.add_argument("--sims", type=int, default=protocol.SIMS)
    ap.add_argument("--c-puct", type=float, default=protocol.C_PUCT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--infer-cache",
        type=int,
        default=0,
        help="0 = off; the round itself runs cache ON",
    )
    ap.add_argument("--net-leg-seconds", type=float, default=3.0)
    ap.add_argument("--engine-leg-seconds", type=float, default=20.0)
    ap.add_argument("--warmup-calls", type=int, default=20)
    ap.add_argument("--warmup-records", type=int, default=32)
    ap.add_argument("--chunk-records", type=int, default=64)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--gpu-threshold", type=float, default=2.0)
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="fresh run directory (rows.jsonl + manifest)",
    )
    args = ap.parse_args()
    out_dir = None
    if args.out:
        out_dir = Path(args.out).resolve()
        if out_dir.exists():
            raise SystemExit(f"refusing to overwrite {out_dir} — pick a fresh --out")
        out_dir.mkdir(parents=True)
        manifest.write(
            out_dir,
            command=[sys.executable, *sys.argv],
            run_kind="inference",
            config=vars(args),
            completed=False,
        )
    args.devices = args.devices.split(",")
    args.widths = [int(x) for x in args.widths.split(",")]
    args.depths = [int(x) for x in args.depths.split(",")]
    args.batches = [int(x) for x in args.batches.split(",")]
    args.n_games = [int(x) for x in args.n_games.split(",")]
    args.engines = [int(x) for x in args.engines.split(",")]

    torch.set_num_threads(args.torch_threads)
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else "n/a (macOS)"
    )
    header = dict(
        game=args.game,
        torch=torch.__version__,
        build=rf._reinfors.core_build_profile(),
        torch_threads=torch.get_num_threads(),
        affinity=str(affinity),
        cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    print(" ".join(f"{k}={v}" for k, v in header.items()))
    if header["build"] != "release":
        print(
            "WARNING: reinfors is a DEBUG build — numbers are meaningless (~10x slow)"
        )

    game, head_actions = build_game(args.game)
    results: list[dict] = []
    if args.mode in ("kernel", "both"):
        bench_net(args, game.observation_space().shape, head_actions, results)
    if args.mode in ("engine", "both"):
        bench_engine(args, game, head_actions, results)
    verdict(results, args)

    if out_dir is not None:
        rows_path = out_dir / "rows.jsonl"
        with rows_path.open("x") as f:
            f.write(json.dumps(dict(header=header)) + "\n")
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(results)} rows -> {rows_path}")
        manifest.finalize(
            out_dir,
            status="ok",
            result_rows=len(results),
            output_sha256={"rows.jsonl": manifest.sha256(rows_path)},
        )


if __name__ == "__main__":
    main()
