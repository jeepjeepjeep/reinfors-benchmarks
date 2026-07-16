"""Shared, matched settings for the reinfors vs OpenSpiel connect4 systems benchmark.

This is a SYSTEMS benchmark (engine efficiency: leaf evaluations/s, moves/s, % time in the net),
NOT an algorithm benchmark. The two sides run different algorithms (reinfors: UCT over per-action
Q values + TreeStrap targets; OpenSpiel: (Alpha)MCTS with a prior+value evaluator), so sample
efficiency / playing strength are out of scope by design. What is matched:
  - game: connect4
  - search budget: SIMULATIONS simulations per move, same uct_c
  - net compute: the same conv trunk (only the input channels and output head differ, both
    negligible next to the trunk); eval-mode forwards, identical device handling
Known, accepted mismatches (documented so results are read honestly):
  - obs encoding: reinfors (2,6,7) vs OpenSpiel observation_tensor (3,6,7) -> first conv differs
    by one input channel (<5% of trunk FLOPs)
  - output head: A Q-values (reinfors) vs A priors + 1 value (OpenSpiel)
  - reinfors' collect additionally assembles TreeStrap training targets; OpenSpiel's bot only
    plays. This biases AGAINST reinfors, which is the safe direction.
"""

import time

import numpy as np
import torch
from torch import nn

SIMULATIONS = 64
UCT_C = 1.4
SEED = 0

CONNECT4_ACTIONS = 7
TRUNK_CHANNELS = 32
TRUNK_FEATURES = 64


class Trunk(nn.Module):
    """Conv(in,32,3) - ReLU - Conv(32,32,3) - ReLU - Flatten - Linear(->64) - ReLU."""

    def __init__(self, in_channels: int, h: int = 6, w: int = 7) -> None:
        super().__init__()
        c = TRUNK_CHANNELS
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(c * h * w, TRUNK_FEATURES),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class QNet(nn.Module):
    """reinfors side: per-action Q values, (N, C*H*W) flat float32 in -> (N, 1, A)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.trunk = Trunk(in_channels)
        self.head = nn.Linear(TRUNK_FEATURES, CONNECT4_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x)).reshape(-1, 1, CONNECT4_ACTIONS)


class PriorValueNet(nn.Module):
    """OpenSpiel side: AZ-style evaluator net -> (priors logits (N, A), value (N, 1))."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.trunk = Trunk(in_channels)
        self.prior_head = nn.Linear(TRUNK_FEATURES, CONNECT4_ACTIONS)
        self.value_head = nn.Linear(TRUNK_FEATURES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return self.prior_head(z), torch.tanh(self.value_head(z))


class Meter:
    """Wall-clock + counters for one benchmark run."""

    def __init__(self) -> None:
        self.net_seconds = 0.0
        self.net_calls = 0
        self.net_rows = 0
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        return time.perf_counter() - self._t0

    def count(self, rows: int, seconds: float) -> None:
        self.net_seconds += seconds
        self.net_calls += 1
        self.net_rows += rows


def report(label: str, wall: float, moves: int, evals: int, net_seconds: float, extra: str = "") -> dict:
    row = dict(label=label, wall=wall, moves=moves, evals=evals, net_seconds=net_seconds)
    print(
        f"{label:44s} {evals / wall:>9.0f} evals/s {moves / wall:>7.1f} moves/s"
        f"  net {net_seconds / wall * 100:5.1f}% of wall  ({evals / max(moves, 1):.1f} evals/move){extra}"
    )
    return row


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class AZResnetReplica(nn.Module):
    """Exact torch mirror of open_spiel alpha_zero_torch's resnet(width=32, depth=1) — input
    conv+BN, one residual block, and BOTH output heads (value computed and discarded; policy
    logits returned as the Q output, with the same masking op). Used to remove net architecture
    as a variable in the sequential decomposition. BN eps/momentum match their model.cc."""

    def __init__(self, in_channels: int, h: int = 6, w: int = 7) -> None:
        super().__init__()
        self.in_channels = in_channels
        c = TRUNK_CHANNELS
        bn = dict(eps=0.001, momentum=0.01)
        self.in_conv = nn.Conv2d(in_channels, c, 3, padding=1)
        self.in_bn = nn.BatchNorm2d(c, **bn)
        self.res_conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(c, **bn)
        self.res_conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(c, **bn)
        self.value_conv = nn.Conv2d(c, 1, 1)
        self.value_bn = nn.BatchNorm2d(1, **bn)
        self.value_l1 = nn.Linear(h * w, c)
        self.value_l2 = nn.Linear(c, 1)
        self.policy_conv = nn.Conv2d(c, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2, **bn)
        self.policy_lin = nn.Linear(2 * h * w, CONNECT4_ACTIONS)
        self.register_buffer("mask", torch.ones(1, CONNECT4_ACTIONS, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.in_bn(self.in_conv(x)))
        y = torch.relu(self.res_bn1(self.res_conv1(x)))
        y = self.res_bn2(self.res_conv2(y))
        x = torch.relu(x + y)
        v = torch.relu(self.value_bn(self.value_conv(x))).flatten(1)
        _ = torch.tanh(self.value_l2(torch.relu(self.value_l1(v))))  # computed, discarded
        p = torch.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        p = self.policy_lin(p)
        p = torch.where(self.mask, p, torch.full_like(p, -65536.0))
        return p.reshape(-1, 1, CONNECT4_ACTIONS)


class ZeroNet(nn.Module):
    """Near-free net (single 7-out linear on 3 inputs) — engine-cost isolation."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.lin = nn.Linear(3, CONNECT4_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x.flatten(1)[:, :3]).reshape(-1, 1, CONNECT4_ACTIONS)
