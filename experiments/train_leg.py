"""One matched-cadence training leg, either engine: launch the trainer pinned, run it
for exactly --minutes of wall clock, then SIGKILL the process group — the hard stop IS
the experiment (an identical wall-clock budget on both sides; whatever checkpoints
survived it are the leg's product).

No measurement happens here: throughput and strength analysis are post-hoc from the
archived telemetry. Both sides leave the same artifacts — `<out>/telemetry.jsonl`
(native from the rf trainer; sampled from the os binary's counters and logs by this
harness) plus checkpoints, with the newest checkpoint recorded in the manifest so
downstream evaluation never has to guess at filenames.

    train_leg.py --side rf --n-games 128 --n-groups 2 --minutes 120 --seed 1 --out runs/x/training
    train_leg.py --side os --actors 16 --minutes 120 --out runs/x/training
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import manifest
import protocol
import run

_OS_CKPT = re.compile(r"checkpoint-(-?\d+)\.pt$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", required=True, choices=["rf", "os"])
    ap.add_argument("--n-games", type=int, help="rf: parallel games")
    ap.add_argument("--n-groups", type=int, default=1, help="rf: collection groups")
    ap.add_argument("--actors", type=int, help="os: actor count")
    ap.add_argument("--batch", type=int, help="os: inference batch (default: actors)")
    ap.add_argument("--minutes", type=float, default=120.0, help="wall-clock budget")
    ap.add_argument(
        "--seed", type=int, help="rf only: their trainer has no seed surface"
    )
    ap.add_argument("--cache", type=int, default=protocol.CACHE)
    ap.add_argument("--cores", default="0-3", help="taskset pin for the child")
    ap.add_argument("--out", required=True, help="fresh training-leg directory")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args(argv)
    if args.side == "rf" and args.n_games is None:
        ap.error("--side rf requires --n-games")
    if args.side == "os" and args.actors is None:
        ap.error("--side os requires --actors")
    if args.side == "rf" and args.actors is not None:
        ap.error("--actors is an os parameter")
    if args.side == "os" and args.n_games is not None:
        ap.error("--n-games is an rf parameter")
    if args.side == "os" and args.seed is not None:
        ap.error("--seed is an rf parameter (their trainer has no seed surface)")
    if args.side == "rf" and args.seed is None:
        args.seed = 0
    if args.side == "os" and args.batch is None:
        args.batch = args.actors
    return args


def verify_checkpoint(path: str | Path, side: str) -> bool:
    """The deadline kill can tear a mid-write checkpoint; only one that LOADS counts
    as complete. rf checkpoints are torch state_dicts; theirs are libtorch archives."""
    import torch

    try:
        if side == "rf":
            torch.load(path, map_location="cpu")
        else:
            torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:
        return False


def latest_checkpoint(out: Path, side: str) -> tuple[str | None, int | None, list[str]]:
    """(newest COMPLETE checkpoint, os checkpoint number, torn candidates skipped).
    Walks newest -> oldest, load-verifying each candidate."""
    if side == "rf":
        candidates = [
            (None, p)
            for p in sorted(
                out.glob("ckpt*"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        ]
    else:
        candidates = sorted(
            (
                (int(m.group(1)), p)
                for p in out.glob("checkpoint-*.pt")
                if (m := _OS_CKPT.search(p.name))
            ),
            reverse=True,
        )
    skipped: list[str] = []
    for number, path in candidates:
        if verify_checkpoint(path, side):
            return str(path), number, skipped
        skipped.append(path.name)
    return None, None, skipped


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run.refuse_smt()
    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"refusing to overwrite {out} — pick a fresh --out")
    out.mkdir(parents=True)

    if args.side == "rf":
        # trainer must never self-exit before our kill
        child = protocol.rf_train_argv(
            out,
            args.n_games,
            args.n_groups,
            args.cache,
            minutes=args.minutes + 10,
            seed=args.seed,
        )
        topology = {"n_games": args.n_games, "n_groups": args.n_groups}
    else:
        child = protocol.os_train_argv(out, args.actors, args.batch, args.cache)
        topology = {"actors": args.actors, "batch": args.batch}
    child_argv = run.pin(child, args.cores)

    manifest.write(
        out,
        command=child_argv,
        run_kind="training",
        side=args.side,
        topology=topology,
        minutes=args.minutes,
        seed=args.seed,
        cache=args.cache,
        telemetry_source="trainer" if args.side == "rf" else "harness-sampler",
        completed=False,
    )

    sampler = run.OsSampler(out) if args.side == "os" else None
    proc = run.launch(
        child_argv, out, extra_env=protocol.OS_CHILD_ENV if args.side == "os" else None
    )
    try:
        early_exit = run.run_scheduled(
            proc,
            args.minutes * 60,
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
            f"CRASHED before the {args.minutes:g}-minute stop (exit {early_exit}) — "
            f"see {out}/child.log",
            file=sys.stderr,
        )
        return 1

    ckpt, ckpt_number, torn = latest_checkpoint(out, args.side)
    if ckpt and args.side == "rf":
        # stable alias so downstream configs reference a knowable path (their side
        # cannot get one: the os loader takes a directory + checkpoint NUMBER and
        # builds the filename itself — eval_h2h resolves `latest` instead)
        shutil.copy2(ckpt, out / "model.pt")
    hashes = {
        "telemetry.jsonl": manifest.sha256(out / "telemetry.jsonl"),
        "child.log": manifest.sha256(out / "child.log"),
    }
    if ckpt:
        hashes[Path(ckpt).name] = manifest.sha256(ckpt)
        if args.side == "rf":
            hashes["model.pt"] = manifest.sha256(out / "model.pt")
    manifest.finalize(
        out,
        status="ok" if ckpt else "no-checkpoint",
        scheduled_kill=True,
        child_exit_code=proc.returncode,
        latest_checkpoint=ckpt,
        checkpoint_number=ckpt_number,
        torn_checkpoints=torn,
        model="model.pt" if ckpt and args.side == "rf" else None,
        output_sha256=hashes,
    )
    if not ckpt:
        print(f"no checkpoint survived the leg — see {out}", file=sys.stderr)
        return 2
    print(f"{args.side} {topology}  {args.minutes:g}m leg done  checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
