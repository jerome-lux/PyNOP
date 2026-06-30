import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
from pynop.core.blocks import MLPBlock, TransolverBlock, LinearNOBlock, TransformerBlock, TransolverBlockv3
from pynop.core.norm import AdaptiveLayerNorm, AdaRMSNorm
from pynop.core.activations import Sine, TaylorSoftmax, gumbel_softmax
from pynop.core.utils import print_stats


class Transolver(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        slice_num: int = 64,
        n_layers: int = 5,
        n_hidden: int = 256,
        dropout: float = 0,
        n_head: int = 8,
        activation: Union[Callable, None] = nn.GELU,
        mode: str = "linear",
        mlp_ratio: int = 1,
        dim: int = 2,
        cond_dim: int = None,
    ):
        super(Transolver, self).__init__()
        assert n_hidden % n_head == 0, "Hidden dim must be divisible by the number o fheads"
        self.lifting = nn.Linear(in_ch, n_hidden)
        self.pe = MLPBlock(dim, n_hidden, hidden_dim=n_hidden // 2, num_layers=1, activation=activation)
        self.pre_norm = nn.RMSNorm((n_hidden))
        self.mixer = nn.Linear(2 * n_hidden, n_hidden)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.n_hidden = n_hidden
        self.dim = dim
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim_head = n_hidden // n_head

        if cond_dim is not None:
            self.embedding = nn.Linear(cond_dim, n_hidden)

        if mode == "linear":
            self.blocks = nn.ModuleList(
                [
                    LinearNOBlock(
                        dim=n_hidden,
                        n_heads=n_head,
                        n_tokens=slice_num,
                        dropout=dropout,
                        activation=activation,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(n_layers)
                ]
            )
        elif "t3":
            self.blocks = nn.ModuleList(
                [
                    TransolverBlockv3(
                        in_features=n_hidden,
                        hidden_features=n_hidden,
                        num_heads=n_head,
                        num_layers=1,
                        num_slices=slice_num,
                        dim_feedforward=mlp_ratio * n_hidden,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransolverBlock(
                        dim=n_hidden,
                        heads=n_head,
                        dim_head=self.dim_head,
                        dropout=dropout,
                        slice_num=slice_num,
                        activation=activation,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(n_layers)
                ]
            )
        self.projection = MLPBlock(
            n_hidden, out_ch=out_ch, hidden_dim=n_hidden * mlp_ratio, num_layers=1, activation=activation
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, cond=None, **kwargs):

        # cond MUST be [B, H*W, cond_dim] (field) or [B, cond_dim] (scalar)

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).view(B, H * W, C).contiguous()

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0)  # [B, H, W, 2] - VRAM: 0
        pe = self.pe(coords).view(1, H * W, -1).expand(B, -1, -1)

        if cond is not None:
            cond = self.embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, :]
            x = x + cond

        x = self.lifting(x)
        x = self.pre_norm(x)
        x = torch.concat([x, pe], dim=-1)
        x = self.mixer(x)

        x = x + self.placeholder[None, None, :]

        for _, block in enumerate(self.blocks):
            x = block(x)

        x = self.projection(x)
        x = x.reshape(B, H, W, self.out_ch).permute(0, 3, 1, 2).contiguous()

        return x
