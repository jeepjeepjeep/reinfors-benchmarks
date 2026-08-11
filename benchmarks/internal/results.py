"""Machine-readable sink for the internal harnesses (--out).

First line = shared run manifest (see benchmarks/openspiel/manifest.py), one JSON
result row per line after it. Refuses to overwrite an existing file.
"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openspiel"))
import manifest  # noqa: E402


class Sink:
    def __init__(self, out: str, **fields: Any) -> None:
        self.rows: list[dict[str, Any]] = []
        self.out = Path(out) if out else None
        if self.out is not None:
            if self.out.exists():
                raise FileExistsError(f"refusing to overwrite {self.out}")
            self.out.parent.mkdir(parents=True, exist_ok=True)
            self.manifest = manifest.collect(
                command=[sys.executable, *sys.argv], **fields
            )

    def record(self, **row: Any) -> None:
        if self.out is not None:
            self.rows.append(row)

    def write(self) -> None:
        if self.out is None:
            return
        self.manifest["completed"] = True
        with open(self.out, "w") as f:
            f.write(json.dumps({"manifest": self.manifest}) + "\n")
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nwrote {len(self.rows)} result rows -> {self.out}")
