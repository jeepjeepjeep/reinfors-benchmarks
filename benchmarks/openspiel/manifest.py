"""Run-manifest collection: every measurement records what produced it.

`collect()` gathers source identity for both repositories, environment, and isolation
state. The benchmark COMMAND must be supplied by the caller (`command=[...]`); this
module never guesses it. Shell harnesses invoke the CLI with the real command string:

    python benchmarks/openspiel/manifest.py --out <run_dir> --command "<exact command>" k=v ...
"""

from __future__ import annotations

import hashlib
import json
import os
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
    return value if out.returncode == 0 else None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def sha256(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def collect(command: list[str] | str | None = None, **extra: Any) -> dict[str, Any]:
    porcelain = _git("status", "--porcelain")
    manifest: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "host": platform.node(),
        "python": platform.python_version(),
        "benchmarks_sha": _git("rev-parse", "HEAD") or "unknown",
        "benchmarks_tag": _git("describe", "--tags", "--exact-match"),
        "benchmarks_dirty": bool(porcelain) if porcelain is not None else "unknown",
        "openspiel_pin": (
            (_read(_REPO / "open_spiel_cpp" / "PIN") or "").splitlines() or [None]
        )[0],
        "smt_active": _read(Path("/sys/devices/system/cpu/smt/active")),
    }
    try:
        import reinfors as rf

        info = getattr(rf, "build_info", dict)()
        info["extension_sha256"] = sha256(rf._reinfors.__file__)
        manifest["reinfors"] = info
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


_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_NUM_THREADS",
)


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def collect_full(
    command: list[str] | str | None = None, **extra: Any
) -> dict[str, Any]:
    """collect() plus the once-per-session environment record: full package freeze,
    OS/kernel, CPU model, affinity, CUDA driver/runtime, inherited thread/device env."""
    data = collect(command=command, **extra)
    import importlib.metadata

    data["packages"] = sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
    )
    data["os"] = {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
    }
    data["cpu_model"] = _cpu_model()
    if hasattr(os, "sched_getaffinity"):
        data["affinity"] = sorted(os.sched_getaffinity(0))
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data["cuda_driver"] = (
            out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
        )
    except (OSError, IndexError):
        data["cuda_driver"] = None
    try:
        import torch

        data["cuda_runtime"] = torch.version.cuda
    except ImportError:
        pass
    data["env"] = {k: os.environ.get(k) for k in _ENV_KEYS if k in os.environ}
    return data


def _manifest_path(out: str | Path) -> Path:
    """A run dir (-> <dir>/manifest.json) or an explicit *.json manifest file."""
    path = Path(out)
    return path if path.suffix == ".json" else path / "manifest.json"


def write(
    out_dir: str | Path,
    command: list[str] | str | None = None,
    full: bool = False,
    **extra: Any,
) -> Path:
    """Atomically write the manifest (refuses to clobber a finalized one)."""
    path = _manifest_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            finalized = bool(json.loads(path.read_text()).get("completed"))
        except (json.JSONDecodeError, OSError):
            finalized = False
        if finalized:
            raise FileExistsError(f"{path} is a finalized manifest; refusing overwrite")
    tmp = path.with_suffix(".json.tmp")
    gather = collect_full if full else collect
    tmp.write_text(json.dumps(gather(command=command, **extra), indent=2) + "\n")
    os.replace(tmp, path)
    return path


def merge(
    out_dir: str | Path, command: list[str] | str | None = None, **fields: Any
) -> Path:
    """Fill manifest gaps without clobbering harness-written fields; creates if absent.

    Python run surfaces call this at startup: standalone runs get a full manifest, runs
    launched by a shell harness keep the harness's fields (its command records the real
    taskset/env wrapper) and only gain what is missing (e.g. the parsed config)."""
    path = _manifest_path(out_dir)
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text())
    else:
        data = collect(command=command)
    if data.get("command") is None and command is not None:
        data["command"] = command
    for key, value in fields.items():
        data.setdefault(key, value)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def finalize(out_dir: str | Path, **fields: Any) -> Path:
    """Atomically merge completion fields into the run's manifest."""
    path = _manifest_path(out_dir)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(fields)
    if "completed" not in fields:
        data["completed"] = True
    data["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    assert args and args[0] == "--out", (
        "usage: manifest.py --out <dir> [--command '<cmd>'] [key=value ...]"
    )
    out, rest = args[1], args[2:]
    command = None
    if rest[:1] == ["--command"]:
        command, rest = rest[1], rest[2:]
    extras: dict[str, Any] = {}
    coerce = {"null": None, "true": True, "false": False}
    for kv in rest:
        key, value = kv.split("=", 1)
        extras[key] = coerce.get(value, value)
    print(write(out, command=command, **extras))
