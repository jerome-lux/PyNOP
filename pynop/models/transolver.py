import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
from pynop.core.blocks import MLPBlock, TransolverBlock, LinearNOBlock, TransformerBlock
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
        dt: float = 1,
    ):
        super(Transolver, self).__init__()
        assert n_hidden % n_head == 0, "Hidden dim must be divisible by the number o fheads"
        self.preprocess = MLPBlock(in_ch + dim, n_hidden, hidden_dim=n_hidden * 2, num_layers=1, activation=activation)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.n_hidden = n_hidden
        self.dim = dim
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim_head = n_hidden // n_head
        self.dt = dt

        if cond_dim is not None:
            self.embedding = nn.Linear(cond_dim, n_hidden)

        self.time_mlp = MLPBlock(
            in_ch=1,
            out_ch=n_hidden,
            hidden_dim=n_hidden,
            num_layers=1,
            activation=activation,
        )

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

    def forward(self, x, time=None, cond=None, return_derivative=True, **kwargs):

        if return_derivative:
            if self.in_ch > self.out_ch:
                shortcut = x[:, -self.out_ch :, ...]
            elif self.in_ch == self.out_ch:
                shortcut = x
            else:
                return_derivative = False

        # cond MUST be [B, H*W, cond_dim] (field) or [B, cond_dim] (scalar)

        B, C, H, W = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2] - VRAM: 0
        coords = coords.reshape(B, H * W, 2)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        x = torch.concat([x, coords], dim=-1)

        x = self.preprocess(x)
        x = x + self.placeholder[None, None, :]

        if cond is not None:
            cond = self.embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, :]
            x = x + cond

        # time modulation
        if time is not None:
            t = self.time_mlp(time).unsqueeze(1)
            x = x + t

        for _, block in enumerate(self.blocks):
            x = block(x)

        x = self.projection(x)
        x = x.reshape(B, H, W, self.out_ch).permute(0, 3, 1, 2).contiguous()

        if return_derivative:
            return
        else:
            return x * self.dt + shortcut

        return x


class LatentTransolver(nn.Module):
    # TODO: ADD suppor for multiple heads to the slicing/deslicing operation
    """ """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 64,
        dt: float = 1.0,
        num_blocks: int = 4,
        hidden_channels: int = 256,
        num_heads: int = 4,
        linear_kernel: bool = True,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act=nn.GELU,
        mlp_factor=4,
        dropout=0,
        dim=2,
        cond_dim=None,
        verbose=False,
        rmsnorm=True,
        std_init=1e-2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.dt = dt
        self.verbose = verbose
        self.num_heads = num_heads

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, modes * num_heads)
        # torch.nn.init.trunc_normal_(self.linear2.weight, std=std_init)
        # nn.init.constant_(self.linear2.bias, 0.0)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.layer_norm_latent = nn.LayerNorm(hidden_channels)

        self.proj_temperature = nn.Sequential(
            nn.Linear(hidden_channels, modes, bias=False), activation(), nn.Linear(modes, num_heads), activation()
        )
        self.proj_temperature.apply(partial(self.init_linear_weights, std=1e-3))
        self.bias = nn.Parameter(torch.ones([1, 1, num_heads]))

        # self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.pe_layer = MLPBlock(
            out_ch=hidden_channels,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.time_scaling = MLPBlock(
            out_ch=out_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)
        self.norm2 = nn.RMSNorm(hidden_channels)
        self.norm3 = nn.RMSNorm(hidden_channels)

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.alpha = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.01))

        # List of attention modules
        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                TransformerBlock(
                    dim=hidden_channels,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=mlp_factor * hidden_channels,
                    dropout=dropout,
                    rmsnorm=rmsnorm,
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def init_linear_weights(self, m, std=1e-3):
        if isinstance(m, nn.Linear):
            # Xavier initialization for tanh/sigmoid
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
        return_derivative: bool = True,
    ):

        if not return_derivative:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                return_derivative = True

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)  # [B, H, W, 2]

        pe = self.pe_layer(coords).expand(B, -1, -1, -1)
        x = self.lifting(x)
        x = x + pe

        time_scaling = F.softplus(self.time_scaling(time))
        encoded_time = self.timsetep_embedding(time)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        # adds temporal encoding and time conditionning
        x = x.view(B, H * W, -1)  # N = H * W
        x = self.norm1(x + encoded_time[:, None, :], encoded_time)

        logits = self.linear2(x).view(B, -1, self.num_heads, self.modes)

        temperature = self.proj_temperature(x).view(B, -1, self.num_heads).mean(dim=1, keepdim=True)
        temperature = torch.clamp(temperature + self.bias, min=0.1)  # self.bias: [1, 1, H]

        # 3. Softmax on the latent dimension M (dim=-1)
        # Each point N in each head H distributes its "mass" over M tokens
        w = nn.Softmax(dim=-1)(logits / temperature.unsqueeze(-1))  # [B, N, H, M]
        v = x.view(B, -1, self.num_heads, x.shape[-1] // self.num_heads)

        # 5. Aggregation: [B, H, M, N] @ [B, H, N, head_dim] -> [B, H, M, head_dim]
        w_t = w.permute(0, 2, 3, 1)  # [B, H, M, N]
        v_t = v.permute(0, 2, 1, 3)  # [B, H, N, head_dim]
        s = torch.matmul(w_t, v_t)  # [B, H, M, head_dim]

        # 6. Mass normalization per head
        d = w_t.sum(dim=-1, keepdim=True) + 1e-6  # [B, H, M, 1]
        # s = s / d

        s = s.permute(0, 2, 1, 3).reshape(B, self.modes, -1)
        s = self.layer_norm_latent(s)

        s = self.linear1(s)

        if self.verbose:
            print_stats(s, -1, "after after projection:")

        # Add Positional encoding in latent representation before the self-attention modules
        m_coords = torch.linspace(-1, 1, steps=self.modes, device=s.device).unsqueeze(-1)
        PE = m_coords.unsqueeze(0).expand(B, -1, -1)
        PE = self.latent_pe(PE)
        s = s + PE

        # Multi-head attention with time conditioning in transformed domain
        for op in self.ops:
            s = op(s, encoded_time)
        if self.verbose:
            print_stats(s, -1, "after ATT modules:")

        s = self.linear3(s)
        s = s.view(B, self.modes, self.num_heads, -1)  # B M H head_dim

        x_rec = torch.einsum("bnhm, bmhd -> bnhd", w, s)
        x_rec = x_rec.reshape(B, H, W, -1)

        if self.verbose:
            print_stats(x_rec, -1, "after REC:")

        # Reshape pour la suite (Conv2d attend B, C, H, W)
        x_rec = x_rec.permute(0, 3, 1, 2).contiguous()

        # Mixing channels and multiplying by the time scaling
        output = time_scaling.view(B, self.out_channels, 1, 1) * self.projection(x_rec)

        if self.verbose:
            print_stats(output, 1, "Final:")

        if return_derivative:
            return output
        else:
            return output * self.dt + shortcut
