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
                "padded_rows": 5,
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
                "padded_rows": 50,
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
    # padded_rows may be absent on interior rows (os telemetry): .get defaults apply
    assert m["physical_rows_per_call"] == pytest.approx((900 + 45) / 45)
    assert m["learn_steps"] == 9


def test_reduce_is_burst_quantization_free(tmp_path: Path) -> None:
    # counters publish in bursts; between bursts the sampler repeats the last value.
    # 9 bursts of 1000 states arrive every 100s at walls 350..1150; filler rows at
    # 305 and 1195 repeat stale values. Edge-row endpoints would divide 8000 states
    # by the 890s row span (=8.99/s); the true rate over whole inter-event
    # intervals is 8000/800 = 10.0/s exactly.
    rows = [{"wall": 100.0, "states": 0, "infer_rows": 0, "infer_calls": 0, "steps": 0}]
    for i in range(9):
        t = 350.0 + i * 100
        st = (i + 1) * 1000
        rows.append(
            {
                "wall": t,
                "states": st,
                "infer_rows": st * 10,
                "infer_calls": st // 100,
                "steps": i + 1,
            }
        )
        rows.append(
            {
                "wall": t + 45,
                "states": st,
                "infer_rows": st * 10,
                "infer_calls": st // 100,
                "steps": i + 1,
            }
        )
    rows.insert(
        1, {"wall": 305.0, "states": 0, "infer_rows": 0, "infer_calls": 0, "steps": 0}
    )
    rows.append(
        {
            "wall": 1195.0,
            "states": 9000,
            "infer_rows": 90000,
            "infer_calls": 90,
            "steps": 9,
        }
    )
    _rows(tmp_path / "learner.jsonl", sorted(rows, key=lambda r: r["wall"]))
    m = measure_throughput.reduce_window(tmp_path / "learner.jsonl", 300, 1200)
    assert m["window_seconds"] == [350.0, 1150.0]
    assert m["states_per_sec"] == pytest.approx(10.0)
    assert m["net_rows_per_sec"] == pytest.approx(100.0)
    assert m["rows_per_call"] == pytest.approx(1000.0)
    assert m["learn_steps"] == 8


def test_reduce_needs_two_rows_in_window(tmp_path: Path) -> None:
    _rows(
        tmp_path / "learner.jsonl",
        [{"wall": 400, "states": 1, "infer_rows": 1, "infer_calls": 1, "steps": 1}],
    )
    assert (
        measure_throughput.reduce_window(tmp_path / "learner.jsonl", 300, 1200) is None
    )


def test_os_sampler_reads_their_learner_counters(tmp_path: Path) -> None:
    # states/games/steps MUST come from their learner.jsonl (its counters see every
    # actor); log-actor files are NEVER a source — their binary caps them at 20
    # actors, the undercount behind the retracted pre-V1 grid numbers
    # the REAL emission order of instrument_vpevaluator.patch: fwd before rows
    (tmp_path / "child.log").write_text(
        "noise\n"
        "[inst] req=1000 hits=400 fwd=10 rows=100 fwd_ms=12.8 eval=0 prior=0\n"
        "[inst] req=2000 hits=800 fwd=25 rows=250 fwd_ms=32.1 eval=0 prior=0\n"
    )
    (tmp_path / "learner.jsonl").write_text(
        json.dumps({"total_states": 1024, "total_trajectories": 1, "step": 1}) + "\n"
    )
    # a stray log-actor file must have no effect on the counts
    (tmp_path / "log-actor-0.txt").write_text(
        "[2026-01-01 00:00:01] Actions: e4 e5 Nf3\n"
    )
    sampler = run.OsSampler(tmp_path)
    sampler.sample(10.0)
    # second poll: only NEW lines read (incremental offsets, cumulative counters)
    with open(tmp_path / "learner.jsonl", "a") as f:
        f.write(
            json.dumps({"total_states": 3072, "total_trajectories": 2, "step": 2})
            + "\n"
        )
    sampler.sample(20.0)
    sampler.close()
    rows = [json.loads(x) for x in open(tmp_path / "telemetry.jsonl")]
    assert rows[0] == {
        "wall": 10.0,
        "states": 1024,
        "games": 1,
        "infer_rows": 250,
        "infer_calls": 25,
        # their step = a sweep of min(total_states, buffer)//1024 minibatches
        "steps": 1,
    }
    assert rows[1]["states"] == 3072 and rows[1]["steps"] == 1 + 3


def test_child_argv_carries_the_matched_protocol() -> None:
    rf = protocol.rf_train_argv(Path("/x"), 128, 2, protocol.CACHE, minutes=30)
    os_ = protocol.os_train_argv(Path("/x"), 64, 32, protocol.CACHE)
    assert ["--width", "256"] == rf[rf.index("--width") : rf.index("--width") + 2]
    assert ["--sims", "64"] in [rf[i : i + 2] for i in range(len(rf))]
    # the operating configuration IS the default: compiled callback, no padding
    assert rf[rf.index("--infer") + 1] == "compiled"
    assert rf[rf.index("--pad-rows-to") + 1] == "-1"
    assert "--nn_width=256" in os_ and "--max_simulations=64" in os_
    assert "--inference_batch_size=32" in os_ and "--actors=64" in os_


def test_rf_infer_flag_reaches_the_trainer() -> None:
    args = measure_throughput.parse_args(
        [
            "--side",
            "rf",
            "--n-games",
            "128",
            "--n-groups",
            "2",
            "--rf-infer",
            "compiled",
            "--out",
            "x",
        ]
    )
    child = measure_throughput.build_child_argv(args, Path("/x"))
    assert child[child.index("--infer") + 1] == "compiled"
    padded = measure_throughput.parse_args(
        [
            "--side",
            "rf",
            "--n-games",
            "128",
            "--n-groups",
            "2",
            "--rf-pad-rows-to",
            "64",
            "--out",
            "x",
        ]
    )
    child = measure_throughput.build_child_argv(padded, Path("/x"))
    assert child[child.index("--pad-rows-to") + 1] == "64"
    with pytest.raises(SystemExit):
        measure_throughput.parse_args(
            ["--side", "os", "--actors", "8", "--rf-infer", "fast", "--out", "x"]
        )
    with pytest.raises(SystemExit):
        measure_throughput.parse_args(
            ["--side", "os", "--actors", "8", "--rf-pad-rows-to", "64", "--out", "x"]
        )


def _fake_rf_trainer(tmp_path: Path) -> Path:
    """Writes rf-schema telemetry fast, then sleeps forever (killed by harness)."""
    script = tmp_path / "fake_trainer.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys, time
            out = sys.argv[sys.argv.index("--out") + 1]
            with open(out + "/telemetry.jsonl", "w") as f:
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
    assert len(m["output_sha256"]["telemetry.jsonl"]) == 64
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
            import json, sys, time
            from pathlib import Path
            out = Path(sys.argv[1])
            learner = open(out / "learner.jsonl", "a")
            for i in range(1, 200):
                print(f"[inst] req={i * 200} hits=0 fwd={i * 10} rows={i * 100} fwd_ms=0.1", flush=True)
                learner.write(json.dumps({"total_states": i * 3,
                                          "total_trajectories": i,
                                          "step": i}) + "\\n")
                learner.flush()
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
    rows = [json.loads(x) for x in open(out / "telemetry.jsonl")]
    assert all(r["states"] <= s2["states"] for r, s2 in zip(rows, rows[1:]))


def test_side_topology_flags_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        measure_throughput.parse_args(["--side", "rf", "--actors", "8", "--out", "x"])
    with pytest.raises(SystemExit):
        measure_throughput.parse_args(["--side", "os", "--n-games", "8", "--out", "x"])
    args = measure_throughput.parse_args(
        ["--side", "os", "--actors", "64", "--out", "x"]
    )
    assert args.batch == 64  # defaults to full-fill
