"""rereduce: metrics-v2 sidecars for archived cells."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import rereduce


def _cell(root: Path, name: str, rows: list[dict]) -> Path:
    d = root / name / "cycle1" / "throughput"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "warmup_seconds": 300,
                "window_seconds": 900,
                "metrics": {"states_per_sec": 1.0},
            }
        )
    )
    (d / "telemetry.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _row(wall: float, states: int) -> dict:
    return {
        "wall": wall,
        "states": states,
        "infer_rows": states,
        "infer_calls": states // 10,
        "steps": states // 100,
    }


def test_writes_sidecar_and_leaves_manifest(tmp_path: Path, capsys) -> None:
    d = _cell(tmp_path, "cell", [_row(310, 1000), _row(400, 2000), _row(500, 3000)])
    assert rereduce.main([str(tmp_path)]) == 0
    v2 = json.loads((d / "metrics-v2.json").read_text())
    assert v2["reduce"] == "event-aligned-v2"
    assert v2["metrics"]["states_per_sec"] == 10.0
    assert v2["metrics_v1"] == {"states_per_sec": 1.0}
    assert json.loads((d / "manifest.json").read_text())["metrics"] == {
        "states_per_sec": 1.0
    }


def test_fewer_than_two_events_fails(tmp_path: Path, capsys) -> None:
    # two in-window rows but a single state-advance event: the old two-row
    # reducer accepted this shape, so status=ok does not rule it out
    d = _cell(tmp_path, "cell", [_row(310, 1000), _row(400, 2000), _row(500, 2000)])
    assert rereduce.main([str(tmp_path)]) == 1
    assert not (d / "metrics-v2.json").exists()
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "failed 1" in out
