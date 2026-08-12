"""The shared benchmark network.

SweepResnet is a torch replica of OpenSpiel's alpha_zero_torch resnet family (their
model.cc): the same net their --nn_width/--nn_depth flags build, layer for layer, so
one (width, depth) point here maps directly onto their side of the comparison
(parameter counts are asserted equal at startup by the trainer).
"""

import numpy as np
import torch
from torch import nn

SEED = 0


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class SweepResnet(nn.Module):
    """open_spiel alpha_zero_torch resnet family (their model.cc), width/depth parameterized —
    the same net their --nn_width/--nn_depth flags build, so one (width, depth) point here maps
    directly onto their side of the head-to-head. Structure per AZResnetReplica (input conv+BN,
    `depth` residual blocks, both heads; BN eps/momentum match model.cc)."""

    def __init__(
        self, in_channels: int, h: int, w: int, n_actions: int, width: int, depth: int
    ) -> None:
        super().__init__()
        self.in_channels, self.h, self.w, self.n_actions = in_channels, h, w, n_actions
        bn = dict(eps=0.001, momentum=0.01)
        self.in_conv = nn.Conv2d(in_channels, width, 3, padding=1)
        self.in_bn = nn.BatchNorm2d(width, **bn)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    dict(
                        conv1=nn.Conv2d(width, width, 3, padding=1),
                        bn1=nn.BatchNorm2d(width, **bn),
                        conv2=nn.Conv2d(width, width, 3, padding=1),
                        bn2=nn.BatchNorm2d(width, **bn),
                    )
                )
            )
        self.value_conv = nn.Conv2d(width, 1, 1)
        self.value_bn = nn.BatchNorm2d(1, **bn)
        self.value_l1 = nn.Linear(h * w, width)
        self.value_l2 = nn.Linear(width, 1)
        self.policy_conv = nn.Conv2d(width, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2, **bn)
        self.policy_lin = nn.Linear(2 * h * w, n_actions)

    def heads(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.in_bn(self.in_conv(x)))
        for b in self.blocks:
            y = torch.relu(b["bn1"](b["conv1"](x)))
            y = b["bn2"](b["conv2"](y))
            x = torch.relu(x + y)
        v = torch.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = torch.tanh(self.value_l2(torch.relu(self.value_l1(v)))).squeeze(-1)
        p = torch.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        return self.policy_lin(p), v
