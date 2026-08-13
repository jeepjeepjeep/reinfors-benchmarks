"""measure_inference: run-directory lifecycle on a tiny cpu net sweep."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "measure_inference.py"

TINY = [
    "--mode",
    "kernel",
    "--game",
    "connect4",
    "--devices",
    "cpu",
    "--widths",
    "16",
    "--depths",
    "1",
    "--batches",
    "2",
    "--net-leg-seconds",
    "0.1",
    "--warmup-calls",
    "1",
]


def _run(out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *TINY, "--out", str(out)],
        capture_output=True,
        text=True,
    )


def test_out_dir_lifecycle_and_overwrite_refusal(tmp_path: Path) -> None:
    out = tmp_path / "inference"
    first = _run(out)
    assert first.returncode == 0, first.stderr

    rows = [json.loads(x) for x in open(out / "rows.jsonl")]
    assert "header" in rows[0] and len(rows) >= 2  # header + one measured cell

    m = json.loads((out / "manifest.json").read_text())
    assert m["run_kind"] == "inference" and m["completed"] and m["status"] == "ok"
    assert m["result_rows"] == len(rows) - 1
    assert len(m["output_sha256"]["rows.jsonl"]) == 64
    assert m["config"]["mode"] == "kernel"

    second = _run(out)
    assert second.returncode != 0 and "refusing to overwrite" in second.stderr


def test_engine_mode_runs_both_dtype_arms(tmp_path: Path) -> None:
    # the f32/f64 A/B is only real if the fast callback actually honors the flag;
    # exercise both arms end-to-end on a tiny cpu engine leg
    for dtype in ("f64", "f32"):
        out = tmp_path / f"arm_{dtype}"
        r = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--mode",
                "engine",
                "--game",
                "connect4",
                "--devices",
                "cpu",
                "--widths",
                "8",
                "--depths",
                "1",
                "--n-games",
                "2",
                "--engine-leg-seconds",
                "0.2",
                "--sims",
                "2",
                "--callback",
                "fast",
                "--infer-dtype",
                dtype,
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        rows = [json.loads(x) for x in open(out / "rows.jsonl")]
        assert len(rows) >= 2  # header + the engine cell


def test_compiled_arm_wiring(tmp_path: Path) -> None:
    # covers the compiled arm's plumbing on both surfaces; TORCHDYNAMO_DISABLE makes
    # torch.compile a passthrough so the test doesn't pay cpu compile time — the
    # actual overhead measurement is a box/CUDA question
    import os

    env = os.environ | {"TORCHDYNAMO_DISABLE": "1"}
    for mode, extra in [
        (
            "kernel",
            ["--batches", "2", "--net-leg-seconds", "0.1", "--warmup-calls", "1"],
        ),
        ("engine", ["--n-games", "2", "--engine-leg-seconds", "0.2", "--sims", "2"]),
    ]:
        out = tmp_path / mode
        r = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--mode",
                mode,
                "--game",
                "connect4",
                "--devices",
                "cpu",
                "--widths",
                "8",
                "--depths",
                "1",
                "--callback",
                "compiled",
                "--out",
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert r.returncode == 0, r.stderr
        rows = [json.loads(x) for x in open(out / "rows.jsonl")]
        assert len(rows) >= 2
        if mode == "engine":
            # compiled engines run the production fixed-shape path: every call is
            # exactly n_games rows, so physical rows are calls * pad
            cell = next(x for x in rows if x.get("part") == "engine")
            assert cell["physical_batch"] == 2.0
            assert cell["padded_rows"] >= 0
