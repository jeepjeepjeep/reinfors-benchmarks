"""Experiment runner: executes a checked-in session spec with auditable evidence.

Responsibilities: strict freeze preflight, unique append-only run directories, launching
every benchmark subprocess itself (exact argv and environment captured), start/completion
manifests finalized atomically, raw-output hashing, and a session index.

    python benchmarks/runner.py spec.json [--resume runs/<session>] [--skip-preflight]

Spec:
{
  "session": "v1-campaign",
  "expect_tag": "v1",
  "cycles": 3,                      # repeats, interleaved: cycle over all cells, repeat
  "cells": [
    {"name": "rf_n64_g1",
     "argv": ["python", "benchmarks/openspiel/train_reinfors_az.py", "--out", "{run_dir}/out", ...],
     "deadline_seconds": 1200,      # SIGKILL the process group here (recorded as intended)
     "cores": "0-3",                # taskset pinning (omit off-box)
     "env": {"OMP_NUM_THREADS": "1"},
     "outputs": ["out/learner.jsonl"],          # hashed at completion
     "cycles": 1                                # optional per-cell override
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "openspiel"))
import manifest  # noqa: E402
import preflight  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]


def _rep_dir(session_dir: Path, cell: dict[str, Any], cycle: int) -> Path:
    return session_dir / cell["name"] / f"cycle{cycle}"


def _completed(rep_dir: Path) -> bool:
    path = rep_dir / "manifest.json"
    try:
        return bool(json.loads(path.read_text()).get("completed"))
    except (OSError, json.JSONDecodeError):
        return False


def _update_index(session_dir: Path, entry: dict[str, Any]) -> None:
    path = session_dir / "index.json"
    try:
        entries = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        entries = []
    entries = [
        e for e in entries if (e["cell"], e["cycle"]) != (entry["cell"], entry["cycle"])
    ]
    entries.append(entry)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp, path)


def run_cell(session_dir: Path, cell: dict[str, Any], cycle: int) -> dict[str, Any]:
    rep_dir = _rep_dir(session_dir, cell, cycle)
    if rep_dir.exists() and any(rep_dir.iterdir()) and not _completed(rep_dir):
        raise FileExistsError(
            f"{rep_dir} exists but is not a completed run; refusing to touch it"
        )
    rep_dir.mkdir(parents=True, exist_ok=True)

    argv = [a.replace("{run_dir}", str(rep_dir)) for a in cell["argv"]]
    if cell.get("cores"):
        argv = ["taskset", "-c", cell["cores"], *argv]
    env = os.environ.copy() | {k: str(v) for k, v in cell.get("env", {}).items()}
    deadline = cell.get("deadline_seconds")

    manifest.write(
        rep_dir,
        command=argv,
        run_kind="cell",
        cell=cell["name"],
        cycle=cycle,
        deadline_seconds=deadline,
        env_overrides=cell.get("env", {}),
        completed=False,
    )

    started = time.monotonic()
    with (
        open(rep_dir / "stdout.log", "wb") as out,
        open(rep_dir / "stderr.log", "wb") as err,
    ):
        proc = subprocess.Popen(
            argv, cwd=_REPO, env=env, stdout=out, stderr=err, start_new_session=True
        )
        intended_kill = False
        try:
            proc.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            intended_kill = True
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=30)
    elapsed = time.monotonic() - started

    outputs = {rel: manifest.sha256(rep_dir / rel) for rel in cell.get("outputs", [])}
    status = (
        "deadline" if intended_kill else ("ok" if proc.returncode == 0 else "failed")
    )
    manifest.finalize(
        rep_dir,
        exit_code=proc.returncode,
        intended_deadline_kill=intended_kill,
        elapsed_seconds=round(elapsed, 1),
        output_sha256=outputs,
        status=status,
    )
    entry = {
        "cell": cell["name"],
        "cycle": cycle,
        "status": status,
        "elapsed_seconds": round(elapsed, 1),
        "dir": str(rep_dir.relative_to(session_dir)),
    }
    _update_index(session_dir, entry)
    if status == "failed":
        raise RuntimeError(
            f"cell {cell['name']} cycle {cycle} failed (exit {proc.returncode}); "
            f"see {rep_dir}/stderr.log"
        )
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--resume", default=None, help="existing session dir to continue")
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="harness testing only; never for publication runs",
    )
    ap.add_argument("--allow-host", action="store_true")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())

    if not args.skip_preflight:
        errors = preflight.check(spec["expect_tag"], allow_host=args.allow_host)
        for e in errors:
            print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        if errors:
            return 1

    if args.resume:
        session_dir = Path(args.resume)
        assert session_dir.is_dir(), f"no session at {session_dir}"
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        session_dir = _REPO / "runs" / f"{stamp}_{spec['session']}"
        session_dir.mkdir(parents=True, exist_ok=False)  # unique, append-only
        manifest.write(
            session_dir,
            command=sys.argv,
            run_kind="session",
            spec=spec,
            completed=False,
        )

    cycles = spec.get("cycles", 1)
    for cycle in range(1, cycles + 1):
        for cell in spec["cells"]:
            if cycle > cell.get("cycles", cycles):
                continue
            rep_dir = _rep_dir(session_dir, cell, cycle)
            if _completed(rep_dir):
                print(f"skip {cell['name']} cycle {cycle} (completed)")
                continue
            print(f"run  {cell['name']} cycle {cycle}", flush=True)
            run_cell(session_dir, cell, cycle)
    manifest.finalize(session_dir, status="ok")
    print(f"session complete: {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
