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


def test_expected_deadline_kill_is_success(tmp_path: Path) -> None:
    out = _run(
        _spec(
            tmp_path,
            [
                {
                    "name": "slow",
                    "argv": ["python3", "-c", "import time; time.sleep(60)"],
                    "deadline_seconds": 1,
                    "deadline_expected": True,
                    "cycles": 1,
                },
            ],
        )
    )
    assert out.returncode == 0, out.stderr
    m = json.loads((_session_dir() / "slow" / "cycle1" / "manifest.json").read_text())
    assert m["status"] == "deadline" and m["intended_deadline_kill"] is True
    assert m["exit_code"] != 0


def test_unexpected_deadline_is_hung_and_blocks_resume(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        [
            {
                "name": "wedged",
                "argv": ["python3", "-c", "import time; time.sleep(60)"],
                "deadline_seconds": 1,
                "cycles": 1,
            },
        ],
    )
    out = _run(spec)
    assert out.returncode != 0 and "hang backstop" in out.stderr
    session = _session_dir()
    m = json.loads((session / "wedged" / "cycle1" / "manifest.json").read_text())
    assert m["status"] == "hung" and m["intended_deadline_kill"] is True
    blocked = _run(spec, "--resume", str(session))
    assert blocked.returncode != 0 and "status 'hung'" in blocked.stderr


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


def test_resume_blocks_on_failed_cell_until_archived(tmp_path: Path) -> None:
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

    # a failed attempt must never count as done: resume refuses until it is archived
    blocked = _run(spec, "--resume", str(session))
    assert blocked.returncode != 0
    assert (
        "status 'failed'" in blocked.stderr
        and "move that directory aside" in blocked.stderr
    )

    (session / "boom" / "cycle1").rename(session / "boom" / "cycle1.attempt1")
    second = _run(spec, "--resume", str(session))
    assert "skip a cycle 1 (ok)" in second.stdout
    assert "run  boom cycle 1" in second.stdout


def test_placeholder_substitution_and_unresolved_rejection(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        [
            {
                "name": "sub",
                "argv": ["python3", "-c", "print('cycle={cycle} extra={extra}')"],
                "cycles": 1,
            },
        ],
    )
    missing = _run(spec)
    assert missing.returncode != 0 and "unresolved placeholders" in missing.stderr

    ok = _run(spec, "--set", "extra=42")
    assert ok.returncode == 0, ok.stderr
    session = _session_dir()
    log = (session / "sub" / "cycle1" / "stdout.log").read_text()
    assert "cycle=1 extra=42" in log
    m = json.loads((session / "manifest.json").read_text())
    assert m["substitutions"] == {"extra": "42"}
    assert m["packages"] and m["cpu_model"] is not None


def test_args_dict_expands_into_the_command(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        [
            {
                "name": "dictargs",
                "argv": ["python3", "-c", "import sys; print(sys.argv[1:])"],
                "args": {"n-games": 64, "seed": "{cycle}", "quick": True},
                "cycles": 1,
            },
        ],
    )
    out = _run(spec)
    assert out.returncode == 0, out.stderr
    session = _session_dir()
    m = json.loads((session / "dictargs" / "cycle1" / "manifest.json").read_text())
    assert m["command"][-5:] == ["--n-games", "64", "--seed", "1", "--quick"]


def test_unresolved_expect_tag_is_rejected(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"session": "t", "expect_tag": "{tag}", "cells": []}))
    out = _run(spec)
    assert out.returncode != 0 and "unresolved expect_tag" in out.stderr


def test_checked_in_specs_are_well_formed() -> None:
    specs = sorted((REPO / "benchmarks" / "specs").glob("*.json"))
    assert specs, "no checked-in specs found"
    known = {"session", "expect_tag", "cycles", "cells"}
    cell_known = {
        "name",
        "argv",
        "args",
        "deadline_seconds",
        "deadline_expected",
        "cores",
        "env",
        "outputs",
        "cycles",
    }
    for path in specs:
        spec = json.loads(path.read_text())
        assert spec["session"] == path.stem  # campaign driver maps sessions <-> files
        assert set(spec) <= known, f"{path.name}: unknown keys {set(spec) - known}"
        names = [c["name"] for c in spec["cells"]]
        assert len(names) == len(set(names)), f"{path.name}: duplicate cell names"
        for cell in spec["cells"]:
            assert set(cell) <= cell_known, (
                f"{path.name}:{cell['name']}: unknown keys {set(cell) - cell_known}"
            )
            assert cell["argv"] and all(isinstance(a, str) for a in cell["argv"])
            assert all(
                isinstance(v, (str, int, float, bool))
                for v in cell.get("args", {}).values()
            ), f"{path.name}:{cell['name']}: args values must be scalars"
            if "measure_grid.py" in cell["argv"][1]:
                # deadlines on self-terminating cells are DERIVED, never hand-set:
                # warmup + window + 30s tail + 120s margin
                args = cell["args"]
                w, t = args["warmup-seconds"], args["window-seconds"]
                assert cell["deadline_seconds"] == w + t + 30 + 120, cell["name"]
                assert not cell.get("deadline_expected"), cell["name"]
