"""measure_inference: run-directory lifecycle on a tiny cpu net sweep."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "measure_inference.py"

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
