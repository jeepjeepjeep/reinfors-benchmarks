"""Shared child runtime for the measurement/training harnesses: pinned launch into a
fresh process group, scheduled SIGKILL with early-exit (crash) detection, SMT guard,
and the os-telemetry sampler that normalizes their binary's raw sources into the rf
learner.jsonl schema."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import protocol

_INST = re.compile(r"\[inst\].*?rows=(\d+).*?fwd=(\d+)")
_TS = re.compile(r"^\[")


def refuse_smt() -> None:
    smt = Path("/sys/devices/system/cpu/smt/active")
    if smt.exists() and smt.read_text().strip() != "0":
        sys.exit(
            "SMT is ON — the protocol is defined at SMT-off. "
            "Fix: sudo bash -c 'echo off > /sys/devices/system/cpu/smt/control'"
        )


def pin(argv: list[str], cores: str) -> list[str]:
    return ["taskset", "-c", cores, *argv] if shutil.which("taskset") else argv


def launch(
    argv: list[str], out: Path, extra_env: dict[str, str] | None = None
) -> subprocess.Popen:
    """Start the child in its own process group, stdout+stderr -> <out>/child.log."""
    with open(out / "child.log", "wb") as log:
        return subprocess.Popen(
            argv,
            cwd=protocol.REPO,
            env=os.environ | (extra_env or {}),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def run_scheduled(
    child: subprocess.Popen, seconds: float, poll: float, on_poll=None
) -> int | None:
    """Run the child for exactly `seconds`, then SIGKILL its process group.
    Returns None after the scheduled kill, or the exit code if it died early."""
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < seconds:
        time.sleep(min(poll, seconds - elapsed))
        if on_poll is not None:
            on_poll(time.monotonic() - start)
        if child.poll() is not None:
            return child.returncode
    os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    child.wait(timeout=30)
    return None


class OsSampler:
    """Normalizes the os binary's three raw telemetry sources into rf's learner.jsonl
    schema: cumulative {wall, states, infer_rows, infer_calls, steps} per poll."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.offsets: dict[Path, int] = {}
        self.states = 0
        self.games = 0
        self.steps = 0
        self.rows = 0
        self.calls = 0
        self.sink = open(out / "learner.jsonl", "a")

    def _new_lines(self, path: Path):
        try:
            with open(path, errors="ignore") as f:
                f.seek(self.offsets.get(path, 0))
                lines = f.readlines()
                self.offsets[path] = f.tell()
                return lines
        except OSError:
            return []

    def sample(self, wall: float) -> None:
        for line in self._new_lines(self.out / "child.log"):
            m = _INST.search(line)
            if m:  # cumulative already; latest wins
                self.rows, self.calls = int(m.group(1)), int(m.group(2))
        for path in sorted(self.out.glob("log-actor*")):
            for line in self._new_lines(path):
                if "Actions:" in line and _TS.match(line):
                    self.games += 1
                    self.states += len(line.split("Actions:", 1)[1].split())
        for path in sorted(self.out.glob("log-learner*")):
            for line in self._new_lines(path):
                if "Step" in line and _TS.match(line):
                    self.steps += 1
        row = {
            "wall": round(wall, 3),
            "states": self.states,
            "games": self.games,
            "infer_rows": self.rows,
            "infer_calls": self.calls,
            "steps": self.steps,
        }
        self.sink.write(json.dumps(row) + "\n")
        self.sink.flush()

    def close(self) -> None:
        self.sink.close()
