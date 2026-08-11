"""Run-manifest collection: every measurement records what produced it.

`collect()` gathers source identity (both repositories), environment, and isolation
state; callers add run-specific fields and write the result next to the run's output.
Also invocable as a CLI for shell harnesses:

    python benchmarks/openspiel/manifest.py --out <run_dir> key=value ...
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=10
        )
    except OSError:
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and value else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def collect(**extra: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv,
        "host": platform.node(),
        "python": platform.python_version(),
        "benchmarks_sha": _git("rev-parse", "HEAD"),
        "benchmarks_tag": _git("describe", "--tags", "--exact-match"),
        "benchmarks_dirty": bool(_git("status", "--porcelain")),
        "openspiel_pin": _read(_REPO / "open_spiel_cpp" / "PIN"),
        "smt_active": _read(Path("/sys/devices/system/cpu/smt/active")),
    }
    try:
        import reinfors as rf

        manifest["reinfors"] = getattr(rf, "build_info", dict)()
    except ImportError:
        manifest["reinfors"] = None
    try:
        import torch

        manifest["torch"] = torch.__version__
        if torch.cuda.is_available():
            manifest["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    manifest.update(extra)
    return manifest


def write(out_dir: str | Path, **extra: Any) -> Path:
    path = Path(out_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collect(**extra), indent=2) + "\n")
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    assert args[0] == "--out", "usage: manifest.py --out <dir> [key=value ...]"
    extras = dict(kv.split("=", 1) for kv in args[2:])
    print(write(args[1], **extras))
