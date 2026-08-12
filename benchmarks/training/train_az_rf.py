"""reinfors-side AlphaZero training driver for the like-for-like comparison, mirroring the
open_spiel alpha_zero_torch learner's hyperparameters and topology:

  net        AZResnetReplica (exact torch mirror of their resnet(width=32, depth=1))
  actors     engine.collect_stream (continuous by default — their actors+learner topology)
  learner    replay buffer (65,536 states) -> minibatch 1,024, Adam lr 1e-4, wd 1e-4;
             pacing: one learn step per (batch_size / reuse) new states (their replay_buffer_reuse)
  search     PUCT, matched sims / c_puct / noise eps+alpha / temperature+drop (set on BOTH sides)

Outputs one JSONL line per learn step (wall, states collected, steps, losses) plus periodic
state_dict checkpoints for the offline strength eval — the analogues of their learner.jsonl and
checkpoint files.

  uv run python benchmarks/training/train_az_rf.py --minutes 3 --out results/rf_az_smoke
"""

import argparse
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
import manifest
import numpy as np
import reinfors as rf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SweepResnet, seed_all


class ReplayBuffer:
    """Uniform ring buffer over (obs, pi, z) rows — their replay_buffer_size/sampling."""

    def __init__(self, capacity: int, dim: int, actions: int, seed: int) -> None:
        self.obs = np.zeros((capacity, dim), dtype=np.float32)
        self.pi = np.zeros((capacity, actions), dtype=np.float64)
        self.legal = np.zeros((capacity, actions), dtype=bool)
        self.z = np.zeros(capacity, dtype=np.float64)
        self.capacity = capacity
        self.size = 0
        self.head = 0
        self.rng = np.random.default_rng(seed)

    def push(
        self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray, legal: np.ndarray
    ) -> None:
        for i in range(obs.shape[0]):
            self.obs[self.head] = obs[i]
            self.pi[self.head] = pi[i]
            self.legal[self.head] = legal[i]
            self.z[self.head] = z[i]
            self.head = (self.head + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=n)
        return self.obs[idx], self.pi[idx], self.z[idx], self.legal[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--game", choices=["connect4", "chess"], default="connect4")
    ap.add_argument(
        "--out", type=str, required=True, help="output dir (jsonl + checkpoints)"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    # matched search knobs — set the SAME values on the open_spiel side
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--noise-eps", type=float, default=0.25)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--temperature-drop", type=int, default=10)
    # matched learner knobs (their defaults)
    ap.add_argument("--buffer-size", type=int, default=2**16)
    ap.add_argument("--batch-size", type=int, default=2**10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--reuse", type=float, default=3.0, help="their replay_buffer_reuse"
    )
    # topology knobs
    ap.add_argument(
        "--n-games", type=int, default=8, help="parallel games (their --actors)"
    )
    ap.add_argument(
        "--n-groups",
        type=int,
        default=1,
        help="2 = double-buffered collect (size n-games as groups x 64 to keep "
        "each group at the A10G batch-64 sweet spot)",
    )
    ap.add_argument(
        "--collect-size", type=int, default=512, help="records per stream batch"
    )
    ap.add_argument("--stream-depth", default="none", help="stream depth: int | none")
    ap.add_argument(
        "--infer-cache",
        type=int,
        default=0,
        help="engine infer-cache entries (0 = off)",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=float,
        default=60.0,
        help="seconds between checkpoints",
    )
    # net architecture — MUST match the OpenSpiel side's --nn_width/--nn_depth (the round
    # launcher passes both sides from one env var; the printed param count is the check)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=1)
    args = ap.parse_args()

    if rf.core_build_profile() != "release" and not os.environ.get(
        "REINFORS_ALLOW_DEBUG"
    ):
        sys.exit(
            "reinfors is a DEBUG build — numbers would be garbage. "
            "maturin develop --release, or REINFORS_ALLOW_DEBUG=1 for wiring tests."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    versions = {
        "reinfors": rf.core_version(),
        "reinfors_profile": rf.core_build_profile(),
        "torch": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(vars(args) | versions, indent=2))
    manifest.merge(
        out,
        command=[sys.executable, *sys.argv],
        run_kind="training",
        config=vars(args),
    )
    print(f"versions: {versions}")
    seed_all()
    torch.manual_seed(args.seed)

    if args.game == "chess":
        # Their observation exactly (parity-gated OpenSpielChess encoder) and THEIR head width:
        # 4674 = 4672 + their two dedicated castling actions. Our pi occupies slots 0..4671; the
        # top two are dead — mirroring their own dead in-grid king-slide slots. Exact net-shape
        # identity; index semantics stay native per framework (see the ledger).
        game = rf.games.Chess(encoder=rf.encoders.OpenSpielChess(), max_ticks=None)
        head_actions = 4674
    else:
        game = rf.games.Connect4()
        head_actions = rf.games.Connect4().action_space().n
    obs_c, obs_h, obs_w = game.observation_space().shape
    dim = int(np.prod(game.observation_space().shape))
    actions = game.action_space().n
    net = SweepResnet(obs_c, obs_h, obs_w, head_actions, args.width, args.depth).to(
        args.device
    )
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"net: SweepResnet w{args.width} d{args.depth} head={head_actions} params={n_params:,}"
    )
    # value head participates here (unlike the review benchmark where it was discarded)
    # Weight decay mirrors their manual L2 term (model.cc: skip any param whose NAME contains
    # "bias" — so BN gammas, named "weight", DO decay). Adam's coupled weight_decay adds
    # wd*w to the grad, the exact gradient of their wd*Σw²/2 loss term.
    decay = [q for name, q in net.named_parameters() if "bias" not in name]
    no_decay = [q for name, q in net.named_parameters() if "bias" in name]
    optimizer = torch.optim.Adam(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    buffer = ReplayBuffer(args.buffer_size, dim, actions, args.seed)

    engine = rf.Engine(
        game,
        rf.Reward(win=1.0, loss=-1.0),
        rf.policies.AlphaZero(
            num_simulations=args.sims,
            c_puct=args.c_puct,
            noise=rf.noise.Dirichlet(epsilon=args.noise_eps, alpha=args.noise_alpha),
            temperature=1.0,
            temperature_drop=args.temperature_drop,
        ),
        rf.learners.AlphaZero(gamma=1.0),
        n_games=args.n_games,
        seed=args.seed,
        infer_cache=args.infer_cache,
        n_groups=args.n_groups,
    )

    collector_net = copy.deepcopy(net)
    collector_net.eval()
    sync_lock = threading.Lock()
    c, h, w = obs_c, obs_h, obs_w

    def infer(obs_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with sync_lock, torch.inference_mode():
            x = (
                torch.from_numpy(np.ascontiguousarray(obs_batch))
                .reshape(-1, c, h, w)
                .to(args.device)
            )
            logits, values = collector_net.heads(x)
        # f32 contract (reinfors PR #136): native f32, padded logits returned WHOLE — the
        # binding accepts width >= A and ignores the tail (a pre-transfer slice costs a
        # device-side gather). This is the measured operating-point path.
        return logits.cpu().numpy(), values.cpu().numpy()

    depth = (
        None
        if str(args.stream_depth).lower() in ("none", "inf")
        else int(args.stream_depth)
    )
    log = (out / "learner.jsonl").open("w")
    t0 = time.perf_counter()
    deadline = t0 + args.minutes * 60.0
    states = 0
    steps = 0
    infer_calls = 0
    infer_rows = 0
    infer_seconds = 0.0
    cache_hits = 0
    cache_lookups = 0
    debt = 0.0
    next_ckpt = args.checkpoint_every
    net.train()

    with engine.collect_stream(args.collect_size, infer, depth=depth) as stream:
        while time.perf_counter() < deadline:
            batch = stream.next()
            with sync_lock:
                collector_net.load_state_dict(net.state_dict())
            engine.weights_updated()  # cache contract: entries from the old weights must not serve
            obs, pi, z = batch.obs, batch.policy_targets, batch.value_targets
            counts = np.diff(batch.legal_offsets)
            rows = np.repeat(np.arange(obs.shape[0]), counts)
            legal = np.zeros((obs.shape[0], actions), dtype=bool)
            legal[rows, batch.legal_ids] = True
            buffer.push(obs, pi, z, legal)
            states += obs.shape[0]
            # engine-side net telemetry: rows = forwards (no cache), calls = pooled batches
            infer_calls += batch.telemetry["infer_calls"]
            infer_rows += batch.telemetry["infer_rows"]
            infer_seconds += batch.telemetry["infer_seconds"]
            cache_hits += batch.telemetry["cache_hits"]
            cache_lookups += batch.telemetry["cache_lookups"]
            if buffer.size < args.batch_size:
                continue
            # their pacing: reuse learn-passes per state -> one step per batch_size/reuse new states
            debt += args.reuse * obs.shape[0]
            while debt >= args.batch_size and time.perf_counter() < deadline:
                debt -= args.batch_size
                bo, bp, bz, bl = buffer.sample(args.batch_size)
                x = torch.from_numpy(bo).reshape(-1, c, h, w).to(args.device)
                target_pi = torch.from_numpy(bp).float().to(args.device)
                mask = torch.from_numpy(bl).to(args.device)
                if (
                    head_actions != actions
                ):  # pad pi/mask with zeros over the dead head slots
                    target_pi = torch.nn.functional.pad(
                        target_pi, (0, head_actions - actions)
                    )
                    mask = torch.nn.functional.pad(mask, (0, head_actions - actions))
                target_z = torch.from_numpy(bz).float().to(args.device)
                logits, values = net.heads(x)
                # their loss exactly: illegal logits to -(1<<16) before log_softmax (model.cc
                # masks in forward) => softmax support = legal actions only
                logits = logits.masked_fill(~mask, -(2.0**16))
                policy_loss = (
                    -(target_pi * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
                )
                value_loss = torch.nn.functional.mse_loss(values, target_z)
                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                optimizer.step()
                steps += 1
                log.write(
                    json.dumps(
                        {
                            "wall": time.perf_counter() - t0,
                            "states": states,
                            "steps": steps,
                            "policy_loss": float(policy_loss.item()),
                            "value_loss": float(value_loss.item()),
                            "pending": stream.pending(),
                            "infer_calls": infer_calls,
                            "infer_rows": infer_rows,
                            "infer_seconds": infer_seconds,
                            "cache_hits": cache_hits,
                            "cache_lookups": cache_lookups,
                        }
                    )
                    + "\n"
                )
            elapsed = time.perf_counter() - t0
            if elapsed >= next_ckpt:
                # label with ACTUAL elapsed time — the check only runs once per collect batch
                # (~133s at chess collect-size), so the next_ckpt counter lags wall-clock and
                # its labels would drift to ~half of real time over a long round
                torch.save(net.state_dict(), out / f"ckpt_{int(elapsed)}s.pt")
                next_ckpt = elapsed + args.checkpoint_every
    torch.save(net.state_dict(), out / "ckpt_final.pt")
    log.close()
    wall = time.perf_counter() - t0
    print(
        f"done: {wall:.0f}s  states {states} ({states / wall:.0f}/s)  learn steps {steps} "
        f"({steps / wall:.2f}/s)"
    )
    if states:
        print(
            f"  infer: {infer_rows} rows in {infer_calls} calls ({infer_rows / max(infer_calls, 1):.2f} rows/call, "
            f"{infer_rows / wall:.0f} rows/s)  rows/state {infer_rows / states:.1f}  "
            f"net share {infer_seconds / wall * 100:.0f}%"
        )
        if cache_lookups:
            print(
                f"  cache: {cache_hits}/{cache_lookups} hits ({cache_hits / cache_lookups * 100:.0f}%)"
            )


if __name__ == "__main__":
    main()
