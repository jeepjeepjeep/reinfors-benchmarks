"""Recompute metrics for archived cells with the current reduce_window.

Cells measured before a reducer fix carry old-definition metrics in their
finalized manifests; those stay untouched. This writes `metrics-v2.json`
beside each manifest, recomputed from the archived telemetry with the cell's
own recorded window parameters.

    python experiments/rereduce.py runs/v1_grid runs/v1_curves
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import manifest as manifest_lib
import measure_throughput


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", help="run directories to walk")
    args = ap.parse_args(argv)
    done = skipped = 0
    for root in args.roots:
        for mf in sorted(Path(root).glob("**/throughput/manifest.json")):
            d = json.loads(mf.read_text())
            tel = mf.parent / "telemetry.jsonl"
            if d.get("status") != "ok" or not tel.exists():
                skipped += 1
                continue
            lo = d["warmup_seconds"]
            metrics = measure_throughput.reduce_window(
                tel, lo, lo + d["window_seconds"]
            )
            out = {
                "reduce": "event-aligned-v2",
                "telemetry_sha256": manifest_lib.sha256(tel),
                "metrics": metrics,
                "metrics_v1": d.get("metrics"),
            }
            (mf.parent / "metrics-v2.json").write_text(json.dumps(out, indent=2) + "\n")
            done += 1
            name = mf.parent.parent.parent.name
            v1 = (d.get("metrics") or {}).get("states_per_sec")
            v2 = (metrics or {}).get("states_per_sec")
            if v1 and v2:
                print(f"{name:16s} {mf.parent.parent.name}  {v1:7.1f} -> {v2:7.1f}")
    print(f"re-reduced {done}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
