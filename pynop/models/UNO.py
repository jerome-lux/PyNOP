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
spatial size divisible by the per-stage window count and by
``downsample_factor ** n_stages``.

Architecture overview (``N`` stages, ``downsample_factor = f``)::

    input (B, Cin, H, W)
    -> Stem          : PE (adaptive) + Linear -> (B, H, W, D0)
    -> Enc stage 0   : repeats x (SwinBlock + ShiftSwinBlock)  -> skip[0]
    -> Down 0        : WindowCrossAttention (f x f) -> (B, H/f, W/f, D1)
    -> Enc stage 1   : repeats x (SwinBlock + ShiftSwinBlock)  -> skip[1]
    -> ...
    -> Enc stage N-1 : repeats x (SwinBlock + ShiftSwinBlock)  -> skip[N-1]
    -> Down N-1      : WindowCrossAttention (f x f) -> (B, H/f^N, W/f^N, DN)
    -> Bottleneck    : repeats x TransformerBlock (global)
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
    size that is divisible by the per-stage window count and by
    ``downsample_factor ** n_stages``.

    Args:
        in_channels: Number of input channels in the physical field.
        out_channels: Number of output channels in the predicted field.
        enc_num_windows: Number of windows per axis for each encoder stage,
            ordered from highest to lowest resolution. The **number of
            entries** determines the number of stages ``N``.

            The window size at stage ``i`` is
            ``resolution_i / num_windows_i``, where
            ``resolution_i = input_res / downsample_factor**i``.
            Two natural strategies:

            - **Constant window size** (recommended): decrease
              ``num_windows`` by ``downsample_factor`` at each stage so the
              spatial extent of each window stays the same.
              Example for ``downsample_factor=2``, ``N=3``:
              ``[(16, 16), (8, 8), (4, 4)]``.
            - **Constant window count**: keep ``num_windows`` the same at
              every stage; windows shrink as resolution decreases (more local
              at deeper stages). Example: ``[(8, 8), (8, 8), (8, 8)]``.

            Use ``None`` for a stage to apply a global
            :class:`~pynop.core.blocks.TransformerBlock` instead of Swin
            blocks (useful at the lowest resolution where global attention is
            cheap).

        dec_num_windows: Number of windows per axis for each decoder stage.
            Same semantics as ``enc_num_windows``. Must have the same length.
            Note the indexing: ``dec_num_windows[0]`` applies to the
            **highest-resolution** decoder stage (just before the output),
            and ``dec_num_windows[N-1]`` applies to the **lowest-resolution**
            decoder stage (just after the bottleneck).

        downsample_factor: Spatial downsampling factor applied by each
            :class:`~pynop.core.blocks.WindowCrossAttention`. A factor of 2
            means each stage halves the spatial resolution (standard U-Net).
            Must be a positive integer (default 2).
        model_dim: Hidden dimension(s). A single int applies to all stages;
            a list of length ``N + 1`` gives per-level dimensions
            (level 0 = input resolution, level N = bottleneck).
        enc_repeats: Number of Swin block **pairs** (unshifted + shifted) per
            encoder stage. Scalar or list of length ``N``.
        dec_repeats: Number of Swin block pairs per decoder stage.
            Scalar or list of length ``N``.
        bottleneck_repeats: Number of Transformer blocks in the bottleneck.
        num_heads: Number of attention heads in all attention blocks.
        mlp_factor: Expansion factor for the MLP hidden size.
        rmsnorm: If True, use ``nn.RMSNorm``; otherwise ``nn.LayerNorm``.
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
        ValueError: At forward time, if the input spatial size is not
            divisible by ``downsample_factor ** N`` or by the per-stage
            window count.

    Example:
        >>> model = UNO(
        ...     in_channels=1, out_channels=1,
        ...     enc_num_windows=[(8, 8), (4, 4), (2, 2)],
        ...     dec_num_windows=[(2, 2), (4, 4), (8, 8)],
        ...     downsample_factor=2,
        ...     model_dim=128,
        ...     num_heads=4,
        ... )
        >>> x = torch.randn(1, 1, 64, 64)  # 64 / 2^3 = 8 (bottleneck)
        >>> out = model(x)  # (1, 1, 64, 64)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        enc_num_windows: List[Union[Tuple[int, int], None]],
        dec_num_windows: List[Union[Tuple[int, int], None]],
        downsample_factor: int = 2,
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
        n_stages = len(enc_num_windows)
        if n_stages < 1:
            raise ValueError(f"UNO: enc_num_windows must have at least 1 stage, got {n_stages}.")
        if len(dec_num_windows) != n_stages:
            raise ValueError(
                f"UNO: dec_num_windows length ({len(dec_num_windows)}) must match "
                f"enc_num_windows length ({n_stages})."
            )
        if downsample_factor < 1:
            raise ValueError(
                f"UNO: downsample_factor must be a positive integer, got {downsample_factor}."
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
        self.downsample_factor = downsample_factor
        self.enc_num_windows = list(enc_num_windows)
        self.dec_num_windows = list(dec_num_windows)
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
            low_dim = self.dims[n_stages - i]
            high_dim = self.dims[n_stages - i - 1]
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
            ValueError: If the input spatial size is not divisible by
                ``downsample_factor ** n_stages`` or by the per-stage
                window count.
        """
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        B, C, H_in, W_in = x.shape
        device = x.device
        n_stages = self.n_stages
        f = self.downsample_factor

        # Check cumulative downsampling divisibility.
        total_factor = f ** n_stages
        if H_in % total_factor != 0 or W_in % total_factor != 0:
            raise ValueError(
                f"UNO: input resolution {H_in}x{W_in} must be divisible by "
                f"downsample_factor^n_stages = {f}^{n_stages} = {total_factor}."
            )

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

            # Downsample by the fixed factor.
            H_tgt = H_curr // f
            W_tgt = W_curr // f
            D_curr = self.dims[i]
            D_next = self.dims[i + 1]

            # Reshape into f x f windows for WindowCrossAttention.
            kv = x.view(B, H_tgt, f, W_tgt, f, D_curr)
            kv = kv.permute(0, 1, 3, 2, 4, 5).contiguous()
            kv = kv.view(B * H_tgt * W_tgt, f * f, D_curr)

            coords_win = self._get_coords(H_curr, W_curr, device)
            coords_win = coords_win.view(1, H_tgt, f, W_tgt, f, 2).expand(B, -1, -1, -1, -1, -1).contiguous()
            coords_win = coords_win.permute(0, 1, 3, 2, 4, 5).contiguous()
            coords_win = coords_win.view(B * H_tgt * W_tgt, f * f, 2)

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
        skips_rev = list(reversed(skips))

        for i in range(n_stages):
            shortcut = skips_rev[i]  # (B, H_high, W_high, D_high)
            B_s, H_high, W_high, D_high = shortcut.shape

            # Upsample using the shortcut as the spatial guide.
            x = self.decoder_stages[i]["upscaler"](x, shortcut)

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

            # Explicit skip fusion.
            if self.skip_fusion == "concat":
                x = torch.cat([x, shortcut], dim=-1)
                x = self.decoder_stages[i]["skip_proj"](x)

        # ------------------------------------------------------------------
        # Head
        # ------------------------------------------------------------------
        x = self.head(x)
        return x.permute(0, 3, 1, 2).contiguous()
