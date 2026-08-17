"""Harness tests must run on SMT-on CI hosts; the SMT guard protects measurement runs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import run


@pytest.fixture(autouse=True)
def _allow_smt(monkeypatch):
    monkeypatch.setattr(run, "refuse_smt", lambda: None)
