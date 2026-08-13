"""Preflight tag rules: benchmarks patch tags keep the frozen wheel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import preflight


def test_wheel_tag_accepts_exact_and_dotted_base() -> None:
    assert preflight.wheel_tag_ok("v1", "v1")
    assert preflight.wheel_tag_ok("v1.0.1", "v1")
    assert preflight.wheel_tag_ok("v1.0.2", "v1")


def test_wheel_tag_rejects_other_lines_and_missing() -> None:
    assert not preflight.wheel_tag_ok("v1.0.1", "v2")
    # a dotted wheel tag means reinfors itself changed mid-campaign: that demands an
    # exact re-freeze at the new tag, never a lenient pass under a later patch tag
    assert not preflight.wheel_tag_ok("v1.0.2", "v1.0.1")
    assert not preflight.wheel_tag_ok("v2", "v1")
    assert not preflight.wheel_tag_ok("v10.1", "v1")  # prefix != dotted base
    assert not preflight.wheel_tag_ok("v1.0.1", None)
    assert not preflight.wheel_tag_ok("v1.0.1", "")
