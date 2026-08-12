"""train.py: matched-cadence leg lifecycle, checkpoint recording, crash detection."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import protocol
import train


def _fake_trainer(tmp_path: Path, body: str) -> None:
    script = tmp_path / "fake_trainer.py"
    script.write_text(body)
    return script


def test_leg_records_the_newest_checkpoint(tmp_path: Path, monkeypatch) -> None:
    fake = _fake_trainer(
        tmp_path,
        "import sys, time\n"
        "out = sys.argv[sys.argv.index('--out') + 1]\n"
        "open(out + '/learner.jsonl', 'w').write('{}\\n')\n"
        "open(out + '/ckpt_60s.pt', 'w').write('a')\n"
        "time.sleep(0.3)\n"
        "open(out + '/ckpt_120s.pt', 'w').write('b')\n"
        "time.sleep(600)\n",
    )
    captured = {}

    def fake_argv(out, n_games, n_groups, cache, minutes, seed=0, device="cuda"):
        captured.update(minutes=minutes, seed=seed)
        return [sys.executable, str(fake), "--out", str(out)]

    monkeypatch.setattr(protocol, "rf_train_argv", fake_argv)
    out = tmp_path / "training"
    rc = train.main(
        [
            "--side",
            "rf",
            "--n-games",
            "128",
            "--n-groups",
            "2",
            "--seed",
            "3",
            "--minutes",
            "0.02",
            "--out",
            str(out),
            "--poll-seconds",
            "0.1",
        ]
    )
    assert rc == 0
    assert captured["seed"] == 3
    assert captured["minutes"] > 0.02  # child outlives the scheduled stop
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "ok" and m["scheduled_kill"] is True
    assert m["run_kind"] == "training" and m["minutes"] == 0.02 and m["seed"] == 3
    assert m["latest_checkpoint"].endswith("ckpt_120s.pt")
    assert m["checkpoint_number"] is None
    assert len(m["output_sha256"]["ckpt_120s.pt"]) == 64
    # the stable alias downstream configs reference
    assert m["model"] == "model.pt"
    assert (out / "model.pt").read_text() == "b"
    assert m["output_sha256"]["model.pt"] == m["output_sha256"]["ckpt_120s.pt"]


def test_os_leg_samples_telemetry_and_numbers_the_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    fake = _fake_trainer(
        tmp_path,
        "import sys, time\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[1])\n"
        "(out / 'checkpoint--1.pt').write_text('init')\n"
        "(out / 'checkpoint-3.pt').write_text('later')\n"
        "actor = open(out / 'log-actor-0.txt', 'a')\n"
        "for i in range(1, 200):\n"
        "    print(f'[inst] rows={i * 10} fwd={i}', flush=True)\n"
        "    actor.write(f'[2026-01-01 00:00:{i % 60:02d}] Actions: e4 e5\\n')\n"
        "    actor.flush()\n"
        "    time.sleep(0.02)\n",
    )
    monkeypatch.setattr(
        protocol,
        "os_train_argv",
        lambda out, *a, **k: [sys.executable, str(fake), str(out)],
    )
    out = tmp_path / "training"
    rc = train.main(
        [
            "--side",
            "os",
            "--actors",
            "16",
            "--minutes",
            "0.02",
            "--out",
            str(out),
            "--poll-seconds",
            "0.1",
        ]
    )
    assert rc == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["telemetry_source"] == "harness-sampler" and m["seed"] is None
    assert m["latest_checkpoint"].endswith("checkpoint-3.pt")
    assert m["checkpoint_number"] == 3 and m["model"] is None
    assert not (out / "model.pt").exists()  # their loader needs dir + number
    rows = [json.loads(x) for x in open(out / "learner.jsonl")]
    assert rows and rows[-1]["states"] > 0 and rows[-1]["infer_rows"] > 0


def test_crash_before_the_stop_is_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        protocol,
        "rf_train_argv",
        lambda out, *a, **k: [sys.executable, "-c", "raise SystemExit(9)"],
    )
    out = tmp_path / "training"
    rc = train.main(
        [
            "--side",
            "rf",
            "--n-games",
            "8",
            "--minutes",
            "0.02",
            "--out",
            str(out),
            "--poll-seconds",
            "0.1",
        ]
    )
    assert rc == 1
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "crashed" and m["child_exit_code"] == 9


def test_leg_without_checkpoints_is_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        protocol,
        "rf_train_argv",
        lambda out, *a, **k: [sys.executable, "-c", "import time; time.sleep(600)"],
    )
    out = tmp_path / "training"
    rc = train.main(
        [
            "--side",
            "rf",
            "--n-games",
            "8",
            "--minutes",
            "0.02",
            "--out",
            str(out),
            "--poll-seconds",
            "0.1",
        ]
    )
    assert rc == 2
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "no-checkpoint"


def test_seed_is_rf_only() -> None:
    with pytest.raises(SystemExit):
        train.parse_args(
            ["--side", "os", "--actors", "16", "--seed", "1", "--out", "x"]
        )
    args = train.parse_args(["--side", "rf", "--n-games", "8", "--out", "x"])
    assert args.seed == 0  # rf default
