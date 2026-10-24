"""UNO — U-Net Neural Operator.

A symmetric U-Net architecture built from Swin-style local attention blocks:

- **Self-attention**: :class:`~pynop.core.blocks.AdaptiveSwinBlock`
  (local window attention with optional shifted windows).
- **Downsampling**: :class:`~pynop.core.blocks.WindowCrossAttention`
  (learnable query compression per window, reducing spatial resolution).
- **Upsampling**: :class:`~pynop.core.blocks.BlockCrossAttentionUpsampler`
  (cross-attention upsampling guided by the high-resolution encoder features).
- **Skip connections**: explicit fusion (add or concat) of the encoder features
  into the decoder at every level, giving the self-attention blocks direct
  access to the high-frequency encoder information.

The model is **resolution invariant**: the window size adapts to the input
resolution (``win = H // num_windows``), so the same weights work at any
spatial size divisible by the number of windows.

Architecture overview (``N = len(grid_resolutions)`` stages)::

    input (B, Cin, H, W)
    -> Stem          : PE (adaptive) + Linear -> (B, H, W, D0)
    -> Enc stage 0   : repeats x (SwinBlock + ShiftSwinBlock)  -> skip[0]
    -> Down 0        : WindowCrossAttention  -> (B, H1, W1, D1)
    -> Enc stage 1   : repeats x (SwinBlock + ShiftSwinBlock)  -> skip[1]
    -> ...
    -> Enc stage N-1 : repeats x (SwinBlock + ShiftSwinBlock)  -> skip[N-1]
    -> Down N-1      : WindowCrossAttention  -> (B, Hb, Wb, Db)
    -> Bottleneck    : repeats x (SwinBlock + ShiftSwinBlock)
    -> Up 0          : BlockCrossAttentionUpsampler (uses skip[N-1])
    -> Dec stage 0   : skip fusion + repeats x (SwinBlock + ShiftSwinBlock)
    -> Up 1          : BlockCrossAttentionUpsampler (uses skip[N-2])
    -> Dec stage 1   : skip fusion + repeats x (SwinBlock + ShiftSwinBlock)
    -> ...
    -> Up N-1        : BlockCrossAttentionUpsampler (uses skip[0])
    -> Dec stage N-1 : skip fusion + repeats x (SwinBlock + ShiftSwinBlock)
    -> Head          : RMSNorm + Linear + Act + Linear -> (B, Cout, H, W)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from pynop.core.blocks import (
    AdaptiveSwinBlock,
    BlockCrossAttentionUpsampler,
    PEBlock,
    TransformerBlock,
    WindowCrossAttention,
)

__all__ = ["UNO"]


def _make_tuple_or_list(value: Union[int, list, tuple], length: int, name: str) -> list:
    """Expand a scalar or validate a sequence to a list of the given length.

    Args:
        value: A scalar int or a sequence of ints.
        length: Expected sequence length.
        name: Parameter name for error messages.

    Returns:
        A list of ints with the given length.

    Raises:
        ValueError: If a sequence has the wrong length.
    """
    if isinstance(value, int):
        return [value] * length
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"{name} has length {len(value)}, expected {length}.")
        return list(value)
    raise TypeError(f"{name} must be an int or a sequence, got {type(value).__name__}.")


class UNO(nn.Module):
    """U-Net Neural Operator (UNO).

    A symmetric encoder/decoder U-Net built entirely from Swin-style local
    attention blocks, with explicit skip connections between encoder and
    decoder.

    Compared to :class:`~pynop.models.SWIN.MultiscaleSWINNO` (which also uses
    the same three block types), UNO treats the skip connection as a
    first-class architectural element:

    - The skip is fused **after** the decoder self-attention blocks (giving
      them a clean, skip-free input to attend over), not only inside the
      upsampler.
    - The fusion mode is configurable: ``"add"`` (residual, default) or
      ``"concat"`` (channel concatenation + 1x1 projection).
    - The architecture is fully symmetric and each stage is independently
      configurable (windows, repeats, dimension).

    The model is **resolution invariant**: window size is derived from the
    input resolution at forward time, so the same weights work at any spatial
    size that is divisible by the per-stage window count and by the
    cumulative downsampling factor.

    Args:
        in_channels: Number of input channels in the physical field.
        out_channels: Number of output channels in the predicted field.
        grid_resolutions: Target spatial resolutions for each encoder stage,
            ordered from high to low (e.g. ``[(64, 64), (32, 32), (16, 16)]``
            for 3 downsample stages). The input resolution must be
            divisible by the first entry.
        enc_num_windows: Number of windows per axis for each encoder stage,
            e.g. ``[(4, 4), (8, 8), (16, 16)]``. Use ``None`` for a stage to
            apply a global :class:`TransformerBlock` instead of Swin blocks.
            Length must equal ``len(grid_resolutions)``.
        dec_num_windows: Number of windows per axis for each decoder stage
            (same order as ``grid_resolutions`` but applied in reverse
            during decoding). Use ``None`` for a global stage.
            Length must equal ``len(grid_resolutions)``.
        model_dim: Hidden dimension(s). A single int applies to all stages;
            a list of length ``len(grid_resolutions) + 1`` gives per-level
            dimensions (level 0 = input, level N = bottleneck).
        enc_repeats: Number of Swin block **pairs** (unshifted + shifted) per
            encoder stage. Scalar or list matching ``len(grid_resolutions)``.
        dec_repeats: Number of Swin block pairs per decoder stage.
            Scalar or list matching ``len(grid_resolutions)``.
        bottleneck_repeats: Number of Swin block pairs in the bottleneck.
        num_heads: Number of attention heads in all attention blocks.
        mlp_factor: Expansion factor for the MLP hidden size.
        rmsnorm: If True, use ``nn.RMSNorm``; otherwise ``nn.LayerNorm``.
            (Used by the global transformer fallback and the head.)
        activation: Activation class for the MLP layers (default GELU).
        skip_fusion: Skip-connection fusion mode:

            - ``"add"`` (default): residual addition. The upsampler already
              adds the shortcut internally, so this mode is equivalent to
              MultiscaleSWINNO.
            - ``"concat"``: after the decoder self-attention blocks, the raw
              encoder skip is concatenated along the channel axis and
              projected back with a 1x1 linear layer. This gives the
              self-attention blocks a skip-free input while still making the
              encoder information available before the next upsampling.

        cond_dim: If set, a conditioning tensor of this many channels is
            concatenated to the input before the stem.

    Raises:
        ValueError: On inconsistent lengths, invalid fusion mode, or
            dimension/head divisibility violations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_resolutions: List[Tuple[int, int]],
        enc_num_windows: List[Union[Tuple[int, int], None]],
        dec_num_windows: List[Union[Tuple[int, int], None]],
        model_dim: Union[int, List[int]] = 128,
        enc_repeats: Union[int, List[int]] = 1,
        dec_repeats: Union[int, List[int]] = 1,
        bottleneck_repeats: int = 1,
        num_heads: int = 4,
        mlp_factor: int = 4,
        rmsnorm: bool = True,
        activation: Callable = nn.GELU,
        skip_fusion: str = "add",
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Validate and normalize parameters
        # ------------------------------------------------------------------
        n_stages = len(grid_resolutions)
        if n_stages < 1:
            raise ValueError(f"UNO: grid_resolutions must have at least 1 stage, got {n_stages}.")
        if len(enc_num_windows) != n_stages:
            raise ValueError(
                f"UNO: enc_num_windows length ({len(enc_num_windows)}) must match "
                f"grid_resolutions length ({n_stages})."
            )
        if len(dec_num_windows) != n_stages:
            raise ValueError(
                f"UNO: dec_num_windows length ({len(dec_num_windows)}) must match "
                f"grid_resolutions length ({n_stages})."
            )
        if skip_fusion not in ("add", "concat"):
            raise ValueError(f"UNO: skip_fusion must be 'add' or 'concat', got {skip_fusion!r}.")

        # Per-level dimensions (n_stages + 1 levels: input, ..., bottleneck).
        if isinstance(model_dim, int):
            self.dims = [model_dim] * (n_stages + 1)
        else:
            if len(model_dim) != n_stages + 1:
                raise ValueError(
                    f"UNO: model_dim list must have {n_stages + 1} entries "
                    f"(one per level), got {len(model_dim)}."
                )
            self.dims = list(model_dim)

        self.enc_repeats = _make_tuple_or_list(enc_repeats, n_stages, "enc_repeats")
        self.dec_repeats = _make_tuple_or_list(dec_repeats, n_stages, "dec_repeats")
        self.bottleneck_repeats = bottleneck_repeats
        self.num_heads = num_heads
        self.skip_fusion = skip_fusion
        self.grid_resolutions = grid_resolutions
        self.enc_num_windows = enc_num_windows
        self.dec_num_windows = dec_num_windows
        self.n_stages = n_stages

        # Divisibility check: each dim must be a multiple of num_heads.
        for i, d in enumerate(self.dims):
            if d % num_heads != 0:
                raise ValueError(
                    f"UNO: model_dim[{i}] = {d} must be divisible by num_heads = {num_heads}."
                )

        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm
        in_ch = in_channels if cond_dim is None else in_channels + cond_dim

        # ------------------------------------------------------------------
        # Stem: positional encoding + input projection
        # ------------------------------------------------------------------
        self.pos_encoder = PEBlock(in_features=2, out_features=self.dims[0], method="adaptive")
        self.input_proj = nn.Linear(in_ch, self.dims[0])
        self.input_norm = norm_layer(self.dims[0])
        self.stem_mixer = nn.Linear(2 * self.dims[0], self.dims[0])

        # ------------------------------------------------------------------
        # Encoder stages
        # ------------------------------------------------------------------
        self.encoder_stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for i in range(n_stages):
            curr_dim = self.dims[i]
            next_dim = self.dims[i + 1]
            windows = enc_num_windows[i]
            repeats = self.enc_repeats[i]

            # Self-attention blocks (Swin pairs or global transformer).
            self_attn = nn.ModuleList()
            if windows is not None:
                for _ in range(repeats):
                    self_attn.append(
                        AdaptiveSwinBlock(
                            dim=curr_dim,
                            grid_windows=windows,
                            num_heads=num_heads,
                            mlp_factor=mlp_factor,
                            activation=activation,
                            shift=False,
                        )
                    )
                    self_attn.append(
                        AdaptiveSwinBlock(
                            dim=curr_dim,
                            grid_windows=windows,
                            num_heads=num_heads,
                            mlp_factor=mlp_factor,
                            activation=activation,
                            shift=True,
                        )
                    )
            else:
                for _ in range(repeats):
                    self_attn.append(
                        TransformerBlock(
                            dim=curr_dim,
                            n_heads=num_heads,
                            mlp_dim=mlp_factor * curr_dim,
                            activation=activation,
                            rmsnorm=rmsnorm,
                        )
                    )

            # Downsampling: WindowCrossAttention compresses spatial resolution.
            down = WindowCrossAttention(in_ch=curr_dim, dim=next_dim, num_heads=num_heads, m=1)
            down_norm = norm_layer(next_dim)

            self.encoder_stages.append(nn.ModuleDict({"self_attn": self_attn}))
            self.downsamplers.append(nn.ModuleDict({"cross_attn": down, "cross_norm": down_norm}))

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        bn_dim = self.dims[-1]
        # The bottleneck operates at the lowest resolution. Use a global
        # transformer (the resolution is small, so global attention is cheap
        # and provides full context).
        self.bottleneck = nn.ModuleList(
            [
                TransformerBlock(
                    dim=bn_dim,
                    n_heads=num_heads,
                    mlp_dim=mlp_factor * bn_dim,
                    activation=activation,
                    rmsnorm=rmsnorm,
                )
                for _ in range(bottleneck_repeats)
            ]
        )
        self.bottleneck_norm = norm_layer(bn_dim)

        # ------------------------------------------------------------------
        # Decoder stages
        # ------------------------------------------------------------------
        self.decoder_stages = nn.ModuleList()

        for i in range(n_stages):
            # Decoder stage i goes from level (n_stages - i) to (n_stages - i - 1).
            # In the reversed order: from dims[n_stages - i] to dims[n_stages - i - 1].
            low_dim = self.dims[n_stages - i]  # current (low-res) dimension
            high_dim = self.dims[n_stages - i - 1]  # target (high-res) dimension
            windows = dec_num_windows[n_stages - 1 - i]  # reversed window config
            repeats = self.dec_repeats[n_stages - 1 - i]  # reversed repeats

            # Upsampling: BlockCrossAttentionUpsampler.
            upscaler = BlockCrossAttentionUpsampler(
                latent_dim=low_dim,
                shortcut_dim=high_dim,
                num_heads=num_heads,
                activation=activation,
                mlp_factor=mlp_factor,
            )

            # Self-attention blocks at the upsampled resolution.
            self_attn = nn.ModuleList()
            if windows is not None:
                for _ in range(repeats):
                    self_attn.append(
                        AdaptiveSwinBlock(
                            dim=high_dim,
                            grid_windows=windows,
                            num_heads=num_heads,
                            mlp_factor=mlp_factor,
                            activation=activation,
                            shift=False,
                        )
                    )
                    self_attn.append(
                        AdaptiveSwinBlock(
                            dim=high_dim,
                            grid_windows=windows,
                            num_heads=num_heads,
                            mlp_factor=mlp_factor,
                            activation=activation,
                            shift=True,
                        )
                    )
            else:
                for _ in range(repeats):
                    self_attn.append(
                        TransformerBlock(
                            dim=high_dim,
                            n_heads=num_heads,
                            mlp_dim=mlp_factor * high_dim,
                            activation=activation,
                            rmsnorm=rmsnorm,
                        )
                    )

            # Skip fusion projection (only needed for "concat" mode).
            if skip_fusion == "concat":
                skip_proj = nn.Linear(2 * high_dim, high_dim)
            else:
                skip_proj = None

            self.decoder_stages.append(
                nn.ModuleDict(
                    {
                        "upscaler": upscaler,
                        "self_attn": self_attn,
                        "skip_proj": skip_proj,
                    }
                )
            )

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            norm_layer(self.dims[0]),
            nn.Linear(self.dims[0], self.dims[0]),
            activation(),
            nn.Linear(self.dims[0], out_channels),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_coords(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate normalized 2D coordinate grid in [-1, 1].

        Args:
            H: Grid height.
            W: Grid width.
            device: Target device.

        Returns:
            Tensor of shape (1, H, W, 2) with coordinates in [-1, 1]^2.
        """
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_y, grid_x], dim=-1).view(1, H, W, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the UNO model.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.
            cond: Optional conditioning tensor of shape
                ``(B, cond_dim, H, W)``, concatenated to the input.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.

        Raises:
            ValueError: If the input spatial size is not divisible by the
                cumulative downsampling factor.
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        B, C, H_in, W_in = x.shape
        device = x.device
        n_stages = self.n_stages

        # Check cumulative downsampling divisibility.
        H_curr, W_curr = H_in, W_in
        for i in range(n_stages):
            H_tgt, W_tgt = self.grid_resolutions[i]
            if H_curr % H_tgt != 0 or W_curr % W_tgt != 0:
                raise ValueError(
                    f"UNO: resolution {H_curr}x{W_curr} at stage {i} is not divisible "
                    f"by target {H_tgt}x{W_tgt}."
                )
            H_curr, W_curr = H_tgt, W_tgt

        # ------------------------------------------------------------------
        # Stem
        # ------------------------------------------------------------------
        coords = self._get_coords(H_in, W_in, device)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = self.input_proj(x)
        x = self.input_norm(x)

        pe = self.pos_encoder(coords).expand(B, -1, -1, -1)
        x = self.stem_mixer(torch.cat([x, pe], dim=-1))

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------
        skips: List[torch.Tensor] = []
        H_curr, W_curr = H_in, W_in

        for i in range(n_stages):
            # Self-attention at the current resolution.
            windows = self.enc_num_windows[i]
            if windows is not None:
                for blk in self.encoder_stages[i]["self_attn"]:
                    x = blk(x)
            else:
                D = x.shape[-1]
                x = x.view(B, H_curr * W_curr, D)
                for blk in self.encoder_stages[i]["self_attn"]:
                    x = blk(x)
                x = x.view(B, H_curr, W_curr, D)

            # Save the skip (encoder feature at this resolution).
            skips.append(x)

            # Downsample to the next resolution.
            H_tgt, W_tgt = self.grid_resolutions[i]
            D_curr = self.dims[i]
            D_next = self.dims[i + 1]

            win_h = H_curr // H_tgt
            win_w = W_curr // W_tgt

            # Reshape into windows for WindowCrossAttention.
            kv = x.view(B, H_tgt, win_h, W_tgt, win_w, D_curr)
            kv = kv.permute(0, 1, 3, 2, 4, 5).contiguous()
            kv = kv.view(B * H_tgt * W_tgt, win_h * win_w, D_curr)

            coords_win = self._get_coords(H_curr, W_curr, device)
            coords_win = coords_win.view(1, H_tgt, win_h, W_tgt, win_w, 2).expand(B, -1, -1, -1, -1, -1).contiguous()
            coords_win = coords_win.permute(0, 1, 3, 2, 4, 5).contiguous()
            coords_win = coords_win.view(B * H_tgt * W_tgt, win_h * win_w, 2)

            x = self.downsamplers[i]["cross_attn"](kv, coords_win)
            x = self.downsamplers[i]["cross_norm"](x)
            x = x.view(B, H_tgt, W_tgt, D_next)

            H_curr, W_curr = H_tgt, W_tgt

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        D_bn = self.dims[-1]
        x = x.view(B, H_curr * W_curr, D_bn)
        for blk in self.bottleneck:
            x = blk(x)
        x = self.bottleneck_norm(x)
        x = x.view(B, H_curr, W_curr, D_bn)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        # Skips are in order: [high-res, ..., low-res]. The decoder goes
        # from low-res to high-res, so we reverse them.
        skips_rev = list(reversed(skips))

        for i in range(n_stages):
            shortcut = skips_rev[i]  # (B, H_high, W_high, D_high)
            B_s, H_high, W_high, D_high = shortcut.shape

            # Upsample using the shortcut as the spatial guide.
            x = self.decoder_stages[i]["upscaler"](x, shortcut)  # (B, H_high, W_high, D_high)

            # Self-attention at the upsampled resolution.
            windows = self.dec_num_windows[n_stages - 1 - i]
            if windows is not None:
                for blk in self.decoder_stages[i]["self_attn"]:
                    x = blk(x)
            else:
                D = x.shape[-1]
                x = x.view(B, H_high * W_high, D)
                for blk in self.decoder_stages[i]["self_attn"]:
                    x = blk(x)
                x = x.view(B, H_high, W_high, D)

            # Explicit skip fusion (the key UNO feature).
            if self.skip_fusion == "concat":
                # Concatenate the raw encoder skip and project back.
                x = torch.cat([x, shortcut], dim=-1)  # (B, H, W, 2*D)
                x = self.decoder_stages[i]["skip_proj"](x)  # (B, H, W, D)
            # For "add", the BlockCrossAttentionUpsampler already added the
            # shortcut internally (out = out + shortcut), so no extra op needed.

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        x = self.head(x)  # (B, H_in, W_in, out_channels)
        return x.permute(0, 3, 1, 2).contiguous()
