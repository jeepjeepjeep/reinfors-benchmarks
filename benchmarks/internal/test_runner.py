"""Runner semantics: unique dirs, captured argv, deadline vs crash, resume."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "benchmarks" / "runner.py"


def _spec(tmp_path: Path, cells) -> Path:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"session": "t", "expect_tag": "v1", "cycles": 2, "cells": cells})
    )
    return spec


def _run(spec: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), str(spec), "--skip-preflight", *extra],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def _session_dir() -> Path:
    runs = sorted((REPO / "runs").glob("*_t"))
    return runs[-1]


@pytest.fixture(autouse=True)
def _clean_runs():
    yield
    import shutil

    for d in (REPO / "runs").glob("*_t"):
        shutil.rmtree(d)


def test_cells_run_interleaved_with_manifests_and_hashes(tmp_path: Path) -> None:
    out = _run(
        _spec(
            tmp_path,
            [
                {
                    "name": "a",
                    "argv": [
                        "python3",
                        "-c",
                        "open('{run_dir}/x.txt','w').write('hi')",
                    ],
                    "outputs": ["x.txt"],
                },
                {"name": "b", "argv": ["python3", "-c", "print('ok')"]},
            ],
        )
    )
    assert out.returncode == 0, out.stderr
    session = _session_dir()
    order = [
        line.split()[1:] for line in out.stdout.splitlines() if line.startswith("run")
    ]
    assert order == [
        ["a", "cycle", "1"],
        ["b", "cycle", "1"],
        ["a", "cycle", "2"],
        ["b", "cycle", "2"],
    ]
    m = json.loads((session / "a" / "cycle1" / "manifest.json").read_text())
    assert m["completed"] and m["status"] == "ok" and m["exit_code"] == 0
    assert m["command"][:1] == ["python3"] and "cycle1" in m["command"][2]
    assert len(m["output_sha256"]["x.txt"]) == 64
    index = json.loads((session / "index.json").read_text())
    assert len(index) == 4


def test_deadline_kill_is_recorded_as_intended(tmp_path: Path) -> None:
    out = _run(
        _spec(
            tmp_path,
            [
                {
                    "name": "slow",
                    "argv": ["python3", "-c", "import time; time.sleep(60)"],
                    "deadline_seconds": 1,
                    "cycles": 1,
                },
            ],
        )
    )
    assert out.returncode == 0, out.stderr
    m = json.loads((_session_dir() / "slow" / "cycle1" / "manifest.json").read_text())
    assert m["status"] == "deadline" and m["intended_deadline_kill"] is True
    assert m["exit_code"] != 0


def test_crash_fails_the_session(tmp_path: Path) -> None:
    out = _run(
        _spec(
            tmp_path,
            [
                {
                    "name": "boom",
                    "argv": ["python3", "-c", "raise SystemExit(3)"],
                    "cycles": 1,
                },
            ],
        )
    )
    assert out.returncode != 0
    m = json.loads((_session_dir() / "boom" / "cycle1" / "manifest.json").read_text())
    assert m["status"] == "failed" and m["exit_code"] == 3


def test_resume_skips_completed_reps(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        [
            {"name": "a", "argv": ["python3", "-c", "print('ok')"]},
            {"name": "boom", "argv": ["python3", "-c", "raise SystemExit(3)"]},
        ],
    )
    first = _run(spec)
    assert first.returncode != 0
    session = _session_dir()
    (session / "boom" / "cycle1").rename(session / "boom" / "cycle1.failed")
    second = _run(spec, "--resume", str(session))
    assert "skip a cycle 1 (completed)" in second.stdout
