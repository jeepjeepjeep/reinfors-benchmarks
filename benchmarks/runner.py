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
     "argv": ["benchmarks/openspiel/train_reinfors_az.py", ...],
                                    # a .py argv[0] runs under the runner's own
                                    # (preflighted) interpreter — never name one
     "args": {"n-games": 64, "seed": "{cycle}"},  # dict-style config, appended as --key value
     "deadline_seconds": 1200,      # SIGKILL the process group here. For deadline-driven
                                    # cells (matched-cadence training) set "deadline_expected":
                                    # true and an EXACT value; for self-terminating payloads it
                                    # is an APPROXIMATE hang backstop (payload + margin) and its
                                    # firing records the cell as hung, failing the session.
     "deadline_expected": false,
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
import re
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


def _finalized_status(rep_dir: Path) -> str | None:
    """The finalized status of a prior attempt, or None if never finalized."""
    path = rep_dir / "manifest.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return str(data.get("status", "unknown")) if data.get("completed") else None


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


def run_cell(
    session_dir: Path,
    cell: dict[str, Any],
    cycle: int,
    sets: dict[str, str] | None = None,
) -> dict[str, Any]:
    sets = sets or {}
    rep_dir = _rep_dir(session_dir, cell, cycle)
    if rep_dir.exists() and any(rep_dir.iterdir()):
        raise FileExistsError(f"{rep_dir} already holds evidence; refusing to touch it")
    rep_dir.mkdir(parents=True, exist_ok=True)

    def sub(value: str) -> str:
        value = value.replace("{run_dir}", str(rep_dir)).replace("{cycle}", str(cycle))
        for key, replacement in sets.items():
            value = value.replace("{" + key + "}", replacement)
        unresolved = re.findall(r"\{[a-z_]+\}", value)
        if unresolved:
            raise ValueError(
                f"unresolved placeholders {unresolved} in cell {cell['name']}; "
                f"supply them with --set key=value"
            )
        return value

    argv = [sub(a) for a in cell["argv"]]
    if argv[0].endswith(".py"):
        # python cells never name an interpreter: the runner prepends its own, so
        # the env preflight validated is, by construction, the env cells execute in
        argv = [sys.executable, *argv]
    for key, value in cell.get("args", {}).items():
        argv.append(f"--{key}")
        if value is not True:  # a literal true is a bare flag
            argv.append(sub(str(value)))
    if cell.get("cores"):
        argv = ["taskset", "-c", cell["cores"], *argv]
    env_overrides = {k: sub(str(v)) for k, v in cell.get("env", {}).items()}
    env = os.environ.copy() | env_overrides
    deadline = cell.get("deadline_seconds")

    manifest.write(
        rep_dir,
        command=argv,
        run_kind="cell",
        cell=cell["name"],
        cycle=cycle,
        deadline_seconds=deadline,
        env_overrides=env_overrides,
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
    if intended_kill:
        status = "deadline" if cell.get("deadline_expected") else "hung"
    elif proc.returncode == 0:
        status = "ok"
    else:
        status = "failed"
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
    if status == "hung":
        raise RuntimeError(
            f"cell {cell['name']} cycle {cycle} hit its hang backstop "
            f"({deadline}s) — a self-terminating payload never should; see {rep_dir}"
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
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="fill a {key} placeholder in the spec (recorded in the session manifest)",
    )
    args = ap.parse_args()
    sets = dict(kv.split("=", 1) for kv in getattr(args, "set"))
    spec = json.loads(Path(args.spec).read_text())

    expect_tag = spec["expect_tag"]
    for key, value in sets.items():
        expect_tag = expect_tag.replace("{" + key + "}", value)
    if "{" in expect_tag:
        print(f"unresolved expect_tag {expect_tag!r}; supply --set", file=sys.stderr)
        return 1

    if not args.skip_preflight:
        errors = preflight.check(expect_tag, allow_host=args.allow_host)
        for e in errors:
            print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
        if errors:
            return 1

    if args.resume:
        session_dir = Path(args.resume)
        assert session_dir.is_dir(), f"no session at {session_dir}"
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        base = _REPO / "runs" / f"{stamp}_{spec['session']}"
        session_dir = next(
            d
            for d in (
                base,
                *(base.with_name(f"{base.name}-{k}") for k in range(2, 100)),
            )
            if not d.exists()
        )
        session_dir.mkdir(parents=True, exist_ok=False)  # unique, append-only
        manifest.write(
            session_dir,
            command=sys.argv,
            full=True,
            run_kind="session",
            spec=spec,
            spec_sha256=manifest.sha256(args.spec),
            substitutions=sets,
            completed=False,
        )

    cycles = spec.get("cycles", 1)
    for cycle in range(1, cycles + 1):
        for cell in spec["cells"]:
            if cycle > cell.get("cycles", cycles):
                continue
            rep_dir = _rep_dir(session_dir, cell, cycle)
            status = _finalized_status(rep_dir)
            if status in ("ok", "deadline"):
                print(f"skip {cell['name']} cycle {cycle} ({status})")
                continue
            if status is not None:
                raise RuntimeError(
                    f"{rep_dir} finished with status '{status}'; a failed attempt "
                    f"never counts as done — move that directory aside (e.g. to "
                    f"{rep_dir}.attempt1) to retry it"
                )
            print(f"run  {cell['name']} cycle {cycle}", flush=True)
            run_cell(session_dir, cell, cycle, sets)
    manifest.finalize(session_dir, status="ok")
    print(f"session complete: {session_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
