import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
from pynop.core.blocks import MLPBlock, TransolverBlock


class Transolver(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        slice_num=64,
        n_layers=5,
        n_hidden=256,
        dropout=0,
        n_head=8,
        activation=nn.GELU,
        mlp_ratio=1,
        dim=2,
        cond_dim=None,
    ):
        super(Transolver, self).__init__()
        assert n_hidden % n_head == 0, "Hidden dim must be divisible by the number o fheads"
        self.preprocess = MLPBlock(in_ch + dim, n_hidden, hidden_dim=n_hidden * 2, num_layers=1, activation=activation)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.n_hidden = n_hidden
        self.dim = dim
        self.out_ch = out_ch
        self.dim_head = n_hidden // n_head
        if cond_dim is not None:
            self.embedding = nn.Linear(cond_dim, n_hidden)
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

    def forward(self, x, time=None, cond=None, **kwargs):

        # cond MUST be [B, H*W, cond_dim] (field) or [B, cond_dim] (scalar)

        B, C, H, W = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2] - VRAM: 0
        coords = coords.reshape(B, H * W, 2)

        if time is not None:
            t = time.view(B, 1, 1).expand(-1, H * W, -1)
            coords = torch.cat([coords, t], dim=-1)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        x = torch.concat([x, coords], dim=-1)

        x = self.preprocess(x)
        x = x + self.placeholder[None, None, :]

        if cond is not None:
            cond = self.embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, :]
            x = x + cond

        for _, block in enumerate(self.blocks):
            x = block(x)

        x = self.projection(x)
        x = x.reshape(B, H, W, self.out_ch).permute(0, 3, 1, 2).contiguous()

        return x
