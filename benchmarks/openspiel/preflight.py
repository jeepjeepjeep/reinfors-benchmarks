"""Freeze preflight: fail-closed validation before any publication run.

Every check must pass; recording bad state is not a substitute for refusing to run.

    python benchmarks/openspiel/preflight.py --expect-tag v1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import manifest

_REPO = Path(__file__).resolve().parents[2]


def check(expect_tag: str, allow_host: bool = False) -> list[str]:
    errors: list[str] = []
    m = manifest.collect()

    rf = m.get("reinfors")
    if not rf:
        errors.append("reinfors is not importable in this environment")
    else:
        if rf.get("git_sha") in (None, "unknown"):
            errors.append(
                "reinfors build has unknown git identity (built outside a checkout?)"
            )
        if rf.get("git_dirty") is not False:
            errors.append(
                f"reinfors build is not from a clean checkout (dirty={rf.get('git_dirty')!r})"
            )
        if rf.get("git_tag") != expect_tag:
            errors.append(
                f"reinfors build tag is {rf.get('git_tag')!r}, expected {expect_tag!r}"
            )
        if rf.get("profile") != "release":
            errors.append(
                f"reinfors build profile is {rf.get('profile')!r}, need release"
            )
        if not rf.get("extension_sha256"):
            errors.append("reinfors extension could not be hashed")

    if m["benchmarks_sha"] == "unknown":
        errors.append("benchmarks repository SHA unknown")
    if m["benchmarks_dirty"] is not False:
        errors.append(
            f"benchmarks repository is not clean (dirty={m['benchmarks_dirty']!r})"
        )
    if m["benchmarks_tag"] != expect_tag:
        errors.append(
            f"benchmarks HEAD tag is {m['benchmarks_tag']!r}, expected {expect_tag!r}"
        )

    pin = m["openspiel_pin"]
    if not pin:
        errors.append("open_spiel_cpp/PIN missing (run scripts/setup_openspiel_cpp.sh)")
    else:
        try:
            head = subprocess.run(
                [
                    "git",
                    "-C",
                    str(_REPO / "open_spiel_cpp" / "open_spiel"),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except OSError:
            head = ""
        if not head.startswith(pin):
            errors.append(
                f"open_spiel checkout {head[:12]!r} does not match PIN {pin!r}"
            )

    if not allow_host:
        if m["smt_active"] is None:
            errors.append("SMT state unreadable — not a Linux measurement host?")
        elif m["smt_active"] != "0":
            errors.append(
                f"SMT is active ({m['smt_active']}); disable it before measuring"
            )
        if "gpu" not in m:
            errors.append("CUDA GPU not visible to torch")

    torch_version = m.get("torch", "")
    if not torch_version.startswith("2.3."):
        errors.append(
            f"torch is {torch_version!r}; the measurement environment pins 2.3.x"
        )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-tag", required=True)
    ap.add_argument(
        "--allow-host",
        action="store_true",
        help="skip host-only checks (SMT, GPU) for harness tests off the box",
    )
    args = ap.parse_args()
    errors = check(args.expect_tag, allow_host=args.allow_host)
    for e in errors:
        print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
    if not errors:
        print("preflight ok")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
