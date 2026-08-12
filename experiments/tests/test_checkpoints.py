"""Shared checkpoint resolver: pair completeness and manifest-first resolution."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import checkpoints


def _os_pair(tmp_path: Path, n: int, optimizer: str = "valid") -> Path:
    model = tmp_path / f"checkpoint-{n}.pt"
    torch.jit.save(torch.jit.script(torch.nn.Linear(2, 2)), str(model))
    opt = checkpoints.optimizer_path(model)
    if optimizer == "valid":
        torch.save({"state": torch.zeros(2)}, opt)
    elif optimizer == "torn":
        torch.save({"state": torch.zeros(2)}, opt)
        opt.write_bytes(opt.read_bytes()[: opt.stat().st_size // 2])
    # optimizer == "missing": write nothing
    return model


def test_os_pair_requires_both_archives(tmp_path: Path) -> None:
    assert checkpoints.verify_os_pair(_os_pair(tmp_path, 1, "valid"))
    assert not checkpoints.verify_os_pair(_os_pair(tmp_path, 2, "missing"))
    assert not checkpoints.verify_os_pair(_os_pair(tmp_path, 3, "torn"))
    # torn model, valid optimizer
    model = _os_pair(tmp_path, 4, "valid")
    model.write_bytes(model.read_bytes()[: model.stat().st_size // 2])
    assert not checkpoints.verify_os_pair(model)


def test_resolve_prefers_the_leg_manifest(tmp_path: Path) -> None:
    # the leg verified 19 and skipped a torn 20; H2H must follow that selection
    _os_pair(tmp_path, 19, "valid")
    _os_pair(tmp_path, 20, "missing")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"completed": True, "checkpoint_number": 19})
    )
    assert checkpoints.resolve_os(tmp_path, "latest") == 19


def test_resolve_without_manifest_walks_verified_pairs(tmp_path: Path) -> None:
    _os_pair(tmp_path, 19, "valid")
    _os_pair(tmp_path, 20, "missing")  # newest, but incomplete
    assert checkpoints.resolve_os(tmp_path, "latest") == 19
    assert checkpoints.resolve_os(tmp_path, 20) == 20  # explicit numbers are honored


def test_resolve_fails_closed_when_nothing_verifies(tmp_path: Path) -> None:
    _os_pair(tmp_path, 5, "torn")
    with pytest.raises(SystemExit, match="no complete checkpoint pair"):
        checkpoints.resolve_os(tmp_path, "latest")
