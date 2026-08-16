"""Render documentation figures from the Markdown result tables."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DOC = ROOT / "docs" / "configuring-the-engines.md"
OUTPUT = ROOT / "docs" / "assets" / "operating-points.svg"


def operating_point_rows() -> list[tuple[int, float, float, float, float]]:
    row = re.compile(
        r"^\| ([\d,]+) \| (.+?) \| (.+?) \| (.+?) \| (.+?) \|$",
        re.MULTILINE,
    )

    def number(value: str) -> float:
        return float(value.replace("*", "").replace(",", ""))

    rows = [
        (int(call.replace(",", "")), *(number(value) for value in values))
        for call, *values in row.findall(CONFIG_DOC.read_text())
        if call.replace(",", "").isdigit()
    ]
    if len(rows) != 6:
        raise ValueError(f"expected six operating-point rows, found {len(rows)}")
    return rows


def main() -> None:
    rows = operating_point_rows()
    calls = [row[0] for row in rows]
    series = {
        "OpenSpiel · full-fill (solid)": (
            [row[1] for row in rows],
            "#d97706",
            "-",
        ),
        "OpenSpiel · half-fill (dashed)": (
            [row[2] for row in rows],
            "#d97706",
            "--",
        ),
        "reinfors · ungrouped (solid)": (
            [row[3] for row in rows],
            "#2563eb",
            "-",
        ),
        "reinfors · grouped (dashed)": (
            [row[4] for row in rows],
            "#2563eb",
            "--",
        ),
    }

    mpl.rcParams["svg.hashsalt"] = "reinfors-benchmarks"
    mpl.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, (values, color, style) in series.items():
        ax.plot(
            calls,
            values,
            color=color,
            linestyle=style,
            marker="o",
            linewidth=2.2,
            label=label,
        )

    ax.scatter(
        [256, 256],
        [rows[3][1], rows[3][4]],
        marker="*",
        s=180,
        color=["#d97706", "#2563eb"],
        zorder=5,
    )
    ax.annotate(
        "236.2",
        (256, rows[3][1]),
        xytext=(-8, -21),
        textcoords="offset points",
        ha="right",
        color="#92400e",
    )
    ax.annotate(
        "265.7",
        (256, rows[3][4]),
        xytext=(8, 8),
        textcoords="offset points",
        color="#1d4ed8",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(calls, [f"{call:,}" for call in calls])
    ax.set_xlabel("Nominal inference call size")
    ax.set_ylabel("Completed-game states/s")
    ax.set_ylim(100, 285)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        ncol=2,
        frameon=False,
        handlelength=3.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg", metadata={"Date": None})
    OUTPUT.write_text(
        "\n".join(line.rstrip() for line in OUTPUT.read_text().splitlines()) + "\n"
    )


if __name__ == "__main__":
    main()
