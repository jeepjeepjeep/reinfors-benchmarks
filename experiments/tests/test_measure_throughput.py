"""measure_throughput: unified reduce, os telemetry normalization, harness lifecycle."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import measure_throughput
import protocol
import run


def _rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_reduce_deltas_inside_window_only(tmp_path: Path) -> None:
    _rows(
        tmp_path / "learner.jsonl",
        [
            {"wall": 100, "states": 999, "infer_rows": 9, "infer_calls": 1, "steps": 0},
            {
                "wall": 300,
                "states": 1000,
                "infer_rows": 100,
                "infer_calls": 10,
                "steps": 1,
            },
            {
                "wall": 900,
                "states": 2200,
                "infer_rows": 700,
                "infer_calls": 40,
                "steps": 7,
            },
            {
                "wall": 1200,
                "states": 2800,
                "infer_rows": 1000,
                "infer_calls": 55,
                "steps": 10,
            },
            {
                "wall": 1300,
                "states": 9999,
                "infer_rows": 9999,
                "infer_calls": 99,
                "steps": 99,
            },
        ],
    )
    m = measure_throughput.reduce_window(tmp_path / "learner.jsonl", 300, 1200)
    assert m["window_seconds"] == [300, 1200]
    assert m["states_per_sec"] == pytest.approx(1800 / 900)
    assert m["net_rows_per_sec"] == pytest.approx(900 / 900)
    assert m["rows_per_call"] == pytest.approx(900 / 45)
    assert m["learn_steps"] == 9


def test_reduce_needs_two_rows_in_window(tmp_path: Path) -> None:
    _rows(
        tmp_path / "learner.jsonl",
        [{"wall": 400, "states": 1, "infer_rows": 1, "infer_calls": 1, "steps": 1}],
    )
    assert measure_throughput.reduce_window(tmp_path / "learner.jsonl", 300, 1200) is None


def test_os_sampler_normalizes_all_three_sources(tmp_path: Path) -> None:
    (tmp_path / "child.log").write_text(
        "noise\n[inst] rows=100 fwd=10\n[inst] rows=250 fwd=25\n"
    )
    (tmp_path / "log-actor-0.txt").write_text(
        "[2026-01-01 00:00:01] Actions: e4 e5 Nf3\n"
        "[2026-01-01 00:00:02] not a game line\n"
    )
    (tmp_path / "log-learner-0.txt").write_text("[2026-01-01 00:00:03] Step 1 done\n")
    sampler = run.OsSampler(tmp_path)
    sampler.sample(10.0)
    # second poll: only NEW lines counted (incremental offsets, cumulative counters)
    with open(tmp_path / "log-actor-0.txt", "a") as f:
        f.write("[2026-01-01 00:00:04] Actions: d4 d5\n")
    sampler.sample(20.0)
    sampler.close()
    rows = [json.loads(x) for x in open(tmp_path / "learner.jsonl")]
    assert rows[0] == {
        "wall": 10.0,
        "states": 3,
        "games": 1,
        "infer_rows": 250,
        "infer_calls": 25,
        "steps": 1,
    }
    assert rows[1]["states"] == 5 and rows[1]["games"] == 2


def test_child_argv_carries_the_matched_protocol() -> None:
    rf = protocol.rf_train_argv(Path("/x"), 128, 2, protocol.CACHE, minutes=30)
    os_ = protocol.os_train_argv(Path("/x"), 64, 32, protocol.CACHE)
    assert ["--width", "256"] == rf[rf.index("--width") : rf.index("--width") + 2]
    assert ["--sims", "64"] in [rf[i : i + 2] for i in range(len(rf))]
    assert "--nn_width=256" in os_ and "--max_simulations=64" in os_
    assert "--inference_batch_size=32" in os_ and "--actors=64" in os_


def _fake_rf_trainer(tmp_path: Path) -> Path:
    """Writes rf-schema telemetry fast, then sleeps forever (killed by harness)."""
    script = tmp_path / "fake_trainer.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            out = sys.argv[sys.argv.index("--out") + 1]
            with open(out + "/learner.jsonl", "w") as f:
                for i in range(40):
                    f.write(json.dumps({"wall": i * 0.1, "states": i * 10,
                                        "infer_rows": i * 100, "infer_calls": i,
                                        "steps": i}) + "\\n")
                f.flush()
            time.sleep(600)
            """
        )
    )
    return script


def test_harness_end_to_end_rf(tmp_path: Path, monkeypatch) -> None:
    fake = _fake_rf_trainer(tmp_path)
    monkeypatch.setattr(
        protocol,
        "rf_train_argv",
        lambda out, *a, **k: [sys.executable, str(fake), "--out", str(out)],
    )
    out = tmp_path / "throughput"
    rc = measure_throughput.main(
        [
            "--side",
            "rf",
            "--n-games",
            "8",
            "--out",
            str(out),
            "--warmup-seconds",
            "1",
            "--window-seconds",
            "2",
            "--tail-seconds",
            "1",
            "--poll-seconds",
            "0.2",
        ]
    )
    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["completed"] and m["status"] == "ok" and m["scheduled_kill"] is True
    assert m["side"] == "rf" and m["topology"] == {"n_games": 8, "n_groups": 1}
    assert m["metrics"]["states_per_sec"] == pytest.approx(100, rel=0.01)
    assert m["child_exit_code"] != 0  # SIGKILL
    assert len(m["output_sha256"]["learner.jsonl"]) == 64
    # append-only: a rerun into the same directory must refuse
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        measure_throughput.main(["--side", "rf", "--n-games", "8", "--out", str(out)])


def test_harness_records_a_crash(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        protocol,
        "rf_train_argv",
        lambda out, *a, **k: [sys.executable, "-c", "raise SystemExit(7)"],
    )
    out = tmp_path / "throughput"
    rc = measure_throughput.main(
        [
            "--side",
            "rf",
            "--n-games",
            "8",
            "--out",
            str(out),
            "--warmup-seconds",
            "1",
            "--window-seconds",
            "2",
            "--tail-seconds",
            "1",
            "--poll-seconds",
            "0.2",
        ]
    )
    assert rc == 1
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "crashed" and m["child_exit_code"] == 7
    assert m["scheduled_kill"] is False


def test_no_interior_window_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        protocol,
        "rf_train_argv",
        lambda out, *a, **k: [sys.executable, "-c", "import time; time.sleep(600)"],
    )
    out = tmp_path / "throughput"
    rc = measure_throughput.main(
        [
            "--side",
            "rf",
            "--n-games",
            "8",
            "--out",
            str(out),
            "--warmup-seconds",
            "1",
            "--window-seconds",
            "2",
            "--tail-seconds",
            "1",
            "--poll-seconds",
            "0.2",
        ]
    )
    assert rc == 2
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "no-interior-window"


def test_harness_end_to_end_os_sampler(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "fake_binary.py"
    fake.write_text(
        textwrap.dedent(
            """
            import sys, time
            from pathlib import Path
            out = Path(sys.argv[1])
            actor = open(out / "log-actor-0.txt", "a")
            for i in range(1, 200):
                print(f"[inst] rows={i * 100} fwd={i * 10}", flush=True)
                actor.write(f"[2026-01-01 00:00:{i:02d}] Actions: e4 e5 Nf3\\n")
                actor.flush()
                time.sleep(0.05)
            """
        )
    )
    monkeypatch.setattr(
        protocol,
        "os_train_argv",
        lambda out, *a, **k: [sys.executable, str(fake), str(out)],
    )
    out = tmp_path / "throughput"
    rc = measure_throughput.main(
        [
            "--side",
            "os",
            "--actors",
            "8",
            "--out",
            str(out),
            "--warmup-seconds",
            "1",
            "--window-seconds",
            "2",
            "--tail-seconds",
            "1",
            "--poll-seconds",
            "0.2",
        ]
    )
    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "ok" and m["telemetry_source"] == "harness-sampler"
    assert m["topology"] == {"actors": 8, "batch": 8}
    # sampler-normalized telemetry: ~3 states per 0.05s child tick
    assert m["metrics"]["states_per_sec"] == pytest.approx(60, rel=0.35)
    assert m["metrics"]["net_rows_per_sec"] > 0
    rows = [json.loads(x) for x in open(out / "learner.jsonl")]
    assert all(r["states"] <= s2["states"] for r, s2 in zip(rows, rows[1:]))


def test_side_topology_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        measure_throughput.parse_args(["--side", "rf", "--actors", "8", "--out", "x"])
    with pytest.raises(SystemExit):
        measure_throughput.parse_args(["--side", "os", "--n-games", "8", "--out", "x"])
    args = measure_throughput.parse_args(["--side", "os", "--actors", "64", "--out", "x"])
    assert args.batch == 64  # defaults to full-fill
