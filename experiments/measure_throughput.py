"""One topology-grid point, either engine: launch the trainer pinned, measure the
pre-registered interior window, kill, reduce, record.

Timeline:
    [0, W)       warmup — excluded (no rf telemetry even exists before ~160-224s)
    [W, W+T]     the measurement window, identical for both sides
    W+T+tail     SIGKILL of the child's process group (early exit = crashed)

Both sides reduce from the same artifact, `<out>/telemetry.jsonl`, rows of cumulative
counters {wall, states, infer_rows, infer_calls, steps}. The rf trainer writes it
natively with its own clock; for os this harness samples the binary's instrumented
stderr counters and its actor/learner logs into the identical schema (the manifest's
`telemetry_source` records which). The metric is a pure function of that artifact and
(W, T): re-running the reduce on an archived cell reproduces the published number.

    measure_throughput.py --side rf --n-games 128 --n-groups 2 --out runs/x/throughput
    measure_throughput.py --side os --actors 64 --batch 32 --out runs/x/throughput
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import manifest
import protocol
import run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=["rf", "os"])
    ap.add_argument("--n-games", type=int, help="rf: parallel games")
    ap.add_argument("--n-groups", type=int, default=1, help="rf: collection groups")
    ap.add_argument("--actors", type=int, help="os: actor count")
    ap.add_argument("--batch", type=int, help="os: inference batch (default: actors)")
    ap.add_argument(
        "--rf-infer",
        choices=["fast", "compiled"],
        default="compiled",
        help="rf: trainer callback implementation (compiled = the operating config)",
    )
    ap.add_argument(
        "--rf-pad-rows-to",
        type=int,
        default=-1,
        help="rf: fixed call rows (-1 = trainer auto rule)",
    )
    ap.add_argument("--warmup-seconds", type=float, default=protocol.WARMUP_SECONDS)
    ap.add_argument("--window-seconds", type=float, default=protocol.WINDOW_SECONDS)
    ap.add_argument("--cache", type=int, default=protocol.CACHE)
    ap.add_argument("--cores", default="0-3", help="taskset pin for the child")
    ap.add_argument("--out", required=True, help="fresh grid-point directory")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--tail-seconds", type=float, default=30.0)
    args = ap.parse_args(argv)
    if args.side == "rf" and args.n_games is None:
        ap.error("--side rf requires --n-games")
    if args.side == "os" and args.actors is None:
        ap.error("--side os requires --actors")
    if args.side == "rf" and args.actors is not None:
        ap.error("--actors is an os parameter")
    if args.side == "os" and args.n_games is not None:
        ap.error("--n-games is an rf parameter")
    if args.side == "os" and args.rf_infer != "compiled":
        ap.error("--rf-infer is an rf parameter")
    if args.side == "os" and args.rf_pad_rows_to != -1:
        ap.error("--rf-pad-rows-to is an rf parameter")
    if args.side == "os" and args.batch is None:
        args.batch = args.actors
    return args


def build_child_argv(args: argparse.Namespace, out: Path) -> list[str]:
    if args.side == "rf":
        # give the trainer no reason to exit before our kill
        minutes = (
            args.warmup_seconds + args.window_seconds + args.tail_seconds
        ) / 60 + 10
        child = protocol.rf_train_argv(
            out,
            args.n_games,
            args.n_groups,
            args.cache,
            minutes=round(minutes, 1),
            infer=args.rf_infer,
            pad_rows_to=args.rf_pad_rows_to,
        )
    else:
        child = protocol.os_train_argv(out, args.actors, args.batch, args.cache)
    return run.pin(child, args.cores)


def reduce_window(path: Path, lo: float, hi: float) -> dict | None:
    """Delta of cumulative counters between the first and last rows inside [lo, hi];
    None if the window holds fewer than two rows."""
    first = last = None
    if not Path(path).exists():
        return None
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if lo <= d["wall"] <= hi:
            if first is None:
                first = d
            last = d
    if first is None or last is first:
        return None
    dt = last["wall"] - first["wall"]
    dr = last["infer_rows"] - first["infer_rows"]
    dc = last["infer_calls"] - first["infer_calls"]
    # os telemetry carries no padded_rows; physical == useful there
    dp = last.get("padded_rows", 0) - first.get("padded_rows", 0)
    return {
        "window_seconds": [first["wall"], last["wall"]],
        "states_per_sec": (last["states"] - first["states"]) / dt,
        "net_rows_per_sec": dr / dt,
        "rows_per_call": dr / dc if dc > 0 else None,
        "physical_rows_per_call": (dr + dp) / dc if dc > 0 else None,
        "learn_steps": last["steps"] - first["steps"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run.refuse_smt()
    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"refusing to overwrite {out} — pick a fresh --out")
    out.mkdir(parents=True)

    child_argv = build_child_argv(args, out)
    topology = (
        {"n_games": args.n_games, "n_groups": args.n_groups}
        if args.side == "rf"
        else {"actors": args.actors, "batch": args.batch}
    )
    manifest.write(
        out,
        command=child_argv,
        run_kind="throughput",
        side=args.side,
        topology=topology,
        warmup_seconds=args.warmup_seconds,
        window_seconds=args.window_seconds,
        cache=args.cache,
        telemetry_source="trainer" if args.side == "rf" else "harness-sampler",
        completed=False,
    )

    sampler = run.OsSampler(out) if args.side == "os" else None
    child = run.launch(
        child_argv, out, extra_env=protocol.OS_CHILD_ENV if args.side == "os" else None
    )
    try:
        early_exit = run.run_scheduled(
            child,
            args.warmup_seconds + args.window_seconds + args.tail_seconds,
            args.poll_seconds,
            on_poll=sampler.sample if sampler else None,
        )
    finally:
        if sampler:
            sampler.close()
    if early_exit is not None:
        manifest.finalize(
            out, status="crashed", child_exit_code=early_exit, scheduled_kill=False
        )
        print(
            f"CRASHED before the scheduled kill (exit {early_exit}) — "
            f"see {out}/child.log",
            file=sys.stderr,
        )
        return 1

    hashes = {
        "telemetry.jsonl": manifest.sha256(out / "telemetry.jsonl"),
        "child.log": manifest.sha256(out / "child.log"),
    }
    metrics = reduce_window(
        out / "telemetry.jsonl",
        args.warmup_seconds,
        args.warmup_seconds + args.window_seconds,
    )
    if metrics is None:
        manifest.finalize(
            out,
            status="no-interior-window",
            scheduled_kill=True,
            child_exit_code=child.returncode,
            output_sha256=hashes,
        )
        print(
            f"FAILED-NO-INTERIOR-SAMPLES in [{args.warmup_seconds:.0f}s, "
            f"{args.warmup_seconds + args.window_seconds:.0f}s] — see {out}",
            file=sys.stderr,
        )
        return 2
    manifest.finalize(
        out,
        status="ok",
        scheduled_kill=True,
        child_exit_code=child.returncode,
        metrics=metrics,
        output_sha256=hashes,
    )
    lo, hi = metrics["window_seconds"]
    rpc = metrics["rows_per_call"]
    rpc_str = f"{rpc:6.1f}" if rpc is not None else "   n/a"
    print(
        f"{args.side} {topology}  states/s={metrics['states_per_sec']:7.1f}  "
        f"net_rows/s={metrics['net_rows_per_sec']:8.1f}  rows/call={rpc_str}  "
        f"learn_steps={metrics['learn_steps']}  (window {lo:.0f}s..{hi:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
