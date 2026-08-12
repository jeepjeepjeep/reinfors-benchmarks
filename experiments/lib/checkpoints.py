"""Checkpoint completeness and resolution, shared by training and head-to-head.

The deadline kill can land between writes, so a checkpoint only counts as complete if
its artifacts LOAD. rf checkpoints are single torch state_dicts. OpenSpiel checkpoints
are a PAIR — their vpnet.cc writes `checkpoint-N.pt` then `checkpoint-N-optimizer.pt`,
and its loader loads both — so a kill between the two writes leaves a loadable model
whose checkpoint is nonetheless unusable; the pair is the unit of completeness.
"""

import json
import re
import zipfile
from pathlib import Path

OS_CKPT = re.compile(r"checkpoint-(-?\d+)\.pt$")


def optimizer_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name[: -len(".pt")] + "-optimizer.pt")


def verify_rf(path: str | Path) -> bool:
    import torch

    try:
        torch.load(path, map_location="cpu")
        return True
    except Exception:
        return False


def verify_os_pair(model_path: str | Path) -> bool:
    """Model archive must jit-load; the optimizer archive (a serialized ivalue, not a
    module) must exist and be a complete zip archive."""
    import torch

    model_path = Path(model_path)
    try:
        torch.jit.load(str(model_path), map_location="cpu")
    except Exception:
        return False
    opt = optimizer_path(model_path)
    try:
        with zipfile.ZipFile(opt) as z:
            return z.testzip() is None
    except Exception:
        return False


def resolve_os(os_dir: str | Path, requested: str | int) -> int:
    """The checkpoint number to load from an OpenSpiel model directory.

    An explicit number is honored as-is. For "latest": a finalized training-leg
    manifest is authoritative (it recorded the verified selection — re-resolving by
    filename could pick a torn checkpoint the leg already skipped); without one, walk
    the numbered pairs newest-first and take the first that verifies.
    """
    os_dir = Path(os_dir)
    if requested != "latest":
        return int(requested)
    manifest_path = os_dir / "manifest.json"
    if manifest_path.exists():
        try:
            number = json.loads(manifest_path.read_text()).get("checkpoint_number")
        except (json.JSONDecodeError, OSError):
            number = None
        if number is not None:
            return int(number)
    numbered = sorted(
        (
            (int(m.group(1)), p)
            for p in os_dir.glob("checkpoint-*.pt")
            if (m := OS_CKPT.search(p.name))
        ),
        reverse=True,
    )
    for number, path in numbered:
        if verify_os_pair(path):
            return number
    raise SystemExit(f"no complete checkpoint pair found in {os_dir}")
