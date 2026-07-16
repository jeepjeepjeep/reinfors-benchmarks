"""Sequential (1 game / 1 actor) decomposition of both frameworks into
    T_move = T_engine + forwards/move x us/forward
with net architecture, kernels, and self-play noise removed as variables.

  reinfors rows:  run here directly (engine telemetry gives forwards + net seconds).
                  --net replica  -> exact torch mirror of their resnet(32,1) (see common.py)
                  --net zero     -> near-free net (pure engine cost)
  open_spiel row: run the instrumented alpha_zero_torch_example separately (1 actor,
                  --policy_epsilon 0), then:  decompose_sequential.py parse-az <stdout> <logdir>
                  ([inst] stderr lines carry requests/cache-hits/forwards/forward-ns).
"""

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_reinfors import build_engine, make_infer  # noqa: E402
from common import AZResnetReplica, ZeroNet, seed_all  # noqa: E402

NETS = {"replica": AZResnetReplica, "zero": ZeroNet}


def run_reinfors(net_name: str, device: str, n_records: int) -> None:
    seed_all()
    net = NETS[net_name](in_channels=2).to(device).eval()
    engine = build_engine(n_games=1)
    infer = make_infer(net, device)
    engine.collect(min(500, n_records), infer)  # warmup
    t0 = time.perf_counter()
    batch = engine.collect(n_records, infer)
    wall = time.perf_counter() - t0
    tel = batch[len(batch) - 1]
    moves, fwds, net_s = int(tel["decisions"]), int(tel["infer_calls"]), float(tel["infer_seconds"])
    print(f"reinfors [{net_name} net, {device}, torch {torch.__version__}]:")
    print(f"  moves/s        {moves / wall:8.1f}")
    print(f"  forwards/move  {fwds / moves:8.1f}   (rows/call {tel['infer_rows'] / fwds:.2f})")
    print(f"  us/forward     {net_s / fwds * 1e6:8.1f}   (net {net_s / wall * 100:.1f}% of wall)")
    print(f"  engine us/move {(wall - net_s) / moves * 1e6:8.1f}")


def parse_az(stdout_file: str, logdir: str) -> None:
    # last [inst] line = cumulative counters (includes warmup; run long enough that it washes out)
    inst = None
    for line in Path(stdout_file).read_text().splitlines():
        m = re.match(r"\[inst\] req=(\d+) hits=(\d+) fwd=(\d+) rows=(\d+) fwd_ms=([\d.]+)", line)
        if m:
            inst = [float(g) for g in m.groups()]
    if inst is None:
        sys.exit("no [inst] lines found — instrumented build not used?")
    req, hits, fwds, rows, fwd_ms = inst

    moves, t0, t1 = 0, None, None
    for f in Path(logdir).glob("log-actor-*.txt"):
        for line in f.read_text().splitlines():
            m = re.match(r"\[([\d\- :.]+)\] Game \d+: Returns: [^;]+; Actions: (.+)", line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            t0 = ts if t0 is None or ts < t0 else t0
            t1 = ts if t1 is None or ts > t1 else t1
            moves += len(m.group(2).split())
    wall = (t1 - t0).total_seconds()
    net_s = fwd_ms / 1e3
    print(f"open_spiel C++ AZ [{stdout_file}]:")
    print(f"  moves/s        {moves / wall:8.1f}   ({moves} moves, {wall:.0f}s)")
    print(f"  requests/move  {req / moves:8.1f}   cache hit rate {hits / req * 100:.1f}%")
    print(f"  forwards/move  {fwds / moves:8.1f}   (rows/call {rows / max(fwds, 1):.2f})")
    print(f"  us/forward     {net_s / fwds * 1e6:8.1f}   (net {net_s / wall * 100:.1f}% of wall)")
    print(f"  engine us/move {(wall - net_s) / moves * 1e6:8.1f}   (approx: counters include warmup)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reinfors")
    r.add_argument("--net", choices=list(NETS), default="replica")
    r.add_argument("--device", default="cpu")
    r.add_argument("--records", type=int, default=2000)
    p = sub.add_parser("parse-az")
    p.add_argument("stdout_file")
    p.add_argument("logdir")
    args = ap.parse_args()
    if args.cmd == "reinfors":
        run_reinfors(args.net, args.device, args.records)
    else:
        parse_az(args.stdout_file, args.logdir)


if __name__ == "__main__":
    main()
