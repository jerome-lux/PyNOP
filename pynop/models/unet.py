"""Classic (non-operator) 2D U-Net for PyNOP.

This module provides a traditional symmetric encoder/decoder U-Net, as opposed to
the neural-operator models elsewhere in PyNOP (FNO, Transolver, Galerkin, ...):

- **Downsampling** with a strided ``Conv2d`` (stride ``factor``, default 2).
- **Upsampling** with sub-pixel convolution (``PixelShuffleUpsample`` from
  ``pynop.core.blocks``), which introduces no bilinear low-pass filtering.
- **Pluggable stage blocks**: any callable block is accepted, in particular the
  three convolutional blocks of PyNOP -- ``ResBlock``, ``BottleneckBlock`` and
  ``ConvNextv2Block``.
- **Optional ViT-style ``PatchEmbed`` stem** (strided conv) for a token-based
  input; the head then un-patches with a matching pixel-shuffle.
- **Optional transformer bottleneck**: a stack of pre-LN ``TransformerBlock``s
  applied to the flattened deepest feature map. Positional information is
  either (a) an additive encoding computed by ``PEBlock`` (default, as
  required the PE is always built through the ``PEBlock`` class) or (b) the
  block's built-in rotary embedding (``rope=True``,
  :class:`~pynop.core.encoding.AdaptiveRoPE2D`). This mirrors modern U-Net
  variants (e.g. TransUNet).
- **Skip fusion** either by channel concatenation (default) or addition.

The model is *not* resolution invariant (strided conv + fixed pixel-shuffle
factors): the input spatial size must be divisible by
``patch_size * factor ** (n_stages - 1)`` (with ``patch_size = 1`` when the
patch stem is disabled) for the output to match the input size exactly.
"""

from __future__ import annotations

import inspect
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from pynop.core.blocks import PEBlock, PatchEmbed, PixelShuffleUpsample, ResBlock, TransformerBlock
from pynop.core.norm import LayerNorm2d
from pynop.core.ops import ConvLayer

__all__ = ["UNet", "TransformerBottleneck", "unet"]


def _block_accepts_out_channels(block: Callable) -> bool:
    """Return True if ``block`` can be called with an ``out_channels`` keyword.

    Some blocks (e.g. ``ConvNextv2Block``) only accept ``in_channels`` (their
    input and output channels are equal) and swallow extra keywords into
    ``**kwargs``. Inspecting the signature lets the U-Net call both kinds of
    blocks uniformly.

    Args:
        block: A callable (class or ``functools.partial``) instantiating a block.

    Returns:
        True if the ``out_channels`` keyword is part of the signature, False
        otherwise (in which case the block must handle ``in_channels ==
        out_channels`` itself).
    """
    try:
        params = inspect.signature(block).parameters
    except (TypeError, ValueError):
        # Builtins / exotic callables: assume the standard block signature.
        return True
    return "out_channels" in params


class TransformerBottleneck(nn.Module):
    """Transformer bottleneck: positional information + stacked ``TransformerBlock``s.

    The input feature map ``(B, C, H, W)`` is flattened to a token sequence
    ``(B, H*W, C)`` and passes through ``n_layers`` pre-LN transformer blocks
    (``Attention`` + MLP with residual connections). A final normalization
    (RMSNorm or LayerNorm) is applied before the tokens are reshaped back to a
    feature map ``(B, C, H, W)``.

    Positional information (required to break the permutation invariance of
    the self-attention) is provided in one of two ways:

    - **additive PE** (default): a 2D encoding computed from normalized
      coordinates in ``[-1, 1]²`` **through the ``PEBlock`` class** (so the
      encoding is resolution adaptive, e.g. ``method="adaptive"`` band-limits
      the frequencies to the Nyquist of the current grid), added to the tokens;
    - **rotary PE** (``rope=True``): each ``TransformerBlock`` owns an
      :class:`~pynop.core.encoding.AdaptiveRoPE2D` applied to the query/key
      projections from the grid coordinates -- no additive PE is built in this
      mode (the two are mutually exclusive).

    This is the "transformer bridge" of modern U-Net variants (e.g. TransUNet,
    arXiv:2102.04306): global self-attention at the lowest resolution, where it
    is cheapest and most useful, while the local conv stages keep the high
    frequencies.

    Args:
        dim: Feature dimension ``C`` of the bottleneck map (must be divisible
            by ``n_heads``).
        n_layers: Number of stacked ``TransformerBlock``s (>= 1).
        n_heads: Number of attention heads.
        mlp_factor: MLP expansion factor (``mlp_dim = dim * mlp_factor``).
        dropout: Dropout rate inside the transformer blocks.
        pe_method: Positional-encoding method passed to ``PEBlock`` (one of
            ``"random_fourier"``, ``"fourier"``, ``"ipe"``, ``"random_ipe"``,
            ``"siren"``, ``"mlp"``, ``"adaptive"``, or ``None``). Defaults to
            ``"adaptive"`` (Nyquist-adaptive Fourier features). Ignored when
            ``rope=True``.
        pe_kwargs: Extra keyword arguments forwarded to ``PEBlock`` (e.g.
            ``{"max_freq_val": 32.0}``).
        rope: If True, use the blocks' built-in rotary embedding
            (``AdaptiveRoPE2D``) instead of the additive ``PEBlock`` encoding.
            Requires ``dim // n_heads`` divisible by 4.
        rope_kwargs: Extra keyword arguments forwarded to ``AdaptiveRoPE2D``
            (e.g. ``{"domain_size": 1.0, "base": 8.0}``).
        rmsnorm: If True use ``nn.RMSNorm`` (default), else ``nn.LayerNorm``.
        std_ini: Std of the truncated-normal init of the block projections.

    Raises:
        ValueError: If ``dim`` is not divisible by ``n_heads``, ``n_layers < 1``,
            ``pe_method is None`` (without RoPE), or ``rope=True`` with a head
            dimension not divisible by 4.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_factor: int = 2,
        dropout: float = 0.1,
        pe_method: str = "adaptive",
        pe_kwargs: Optional[dict] = None,
        rope: bool = False,
        rope_kwargs: Optional[dict] = None,
        rmsnorm: bool = True,
        std_ini: float = 0.02,
    ):
        super().__init__()
        if dim < 1 or n_heads < 1 or dim % n_heads != 0:
            raise ValueError(
                f"TransformerBottleneck: dim ({dim}) must be a positive multiple of "
                f"n_heads ({n_heads})."
            )
        if n_layers < 1:
            raise ValueError(f"TransformerBottleneck: n_layers must be >= 1, got {n_layers}.")
        if not rope and pe_method is None:
            raise ValueError(
                "TransformerBottleneck: pe_method=None is not supported (a positional encoding is required "
                "to break permutation invariance of the self-attention); use one of 'adaptive', 'fourier', "
                "'random_fourier', 'ipe', 'random_ipe', 'siren' or 'mlp', or rope=True."
            )
        if rope and (dim // n_heads) % 4 != 0:
            raise ValueError(
                f"TransformerBottleneck: rope=True requires dim // n_heads == {dim // n_heads} to be "
                f"divisible by 4 for 2-D RoPE (dim={dim}, n_heads={n_heads})."
            )

        self.dim = dim
        self.n_layers = n_layers
        self.use_rope = rope

        # Positional information: either an additive PEBlock encoding (default)
        # or the blocks' own rotary embedding (rope=True) -- mutually exclusive.
        self.pe = None if rope else PEBlock(in_features=2, out_features=dim, method=pe_method, **(pe_kwargs or {}))

        # Stacked pre-LN transformer blocks; `scaling` enables the depth-scaled
        # output-projection init (1 / sqrt(2 * n_layers), Megatron-LM style).
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    mlp_dim=dim * mlp_factor,
                    dropout=dropout,
                    rmsnorm=rmsnorm,
                    std_ini=std_ini,
                    scaling=n_layers,
                    rope=rope,
                    rope_kwargs=rope_kwargs,
                )
                for _ in range(n_layers)
            ]
        )

        # Final normalization of the residual stream (standard ViT practice).
        self.final_norm = nn.RMSNorm(dim) if rmsnorm else nn.LayerNorm(dim)

    @staticmethod
    def _coords(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Normalized 2D coordinate grid.

        Args:
            H: Grid height.
            W: Grid width.
            device: Target device.

        Returns:
            Tensor of shape ``(1, H, W, 2)`` with coordinates in ``[-1, 1]²``.
        """
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a feature map through the transformer stack.

        Args:
            x: Feature map of shape ``(B, C, H, W)`` with ``C == dim``.

        Returns:
            Feature map of shape ``(B, C, H, W)``.
        """
        B, C, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"TransformerBottleneck expects C == {self.dim}, got {C}.")

        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if self.use_rope:
            # 4-D grid coordinates so AdaptiveRoPE2D infers the (possibly
            # non-square) grid resolution for its Nyquist-adaptive frequencies.
            coords = self._coords(H, W, x.device).expand(B, -1, -1, -1)  # (B, H, W, 2)
            for layer in self.layers:
                tokens = layer(tokens, coords=coords)
        else:
            pe = self.pe(self._coords(H, W, x.device)).expand(B, -1, -1, -1)  # (B, H, W, C)
            tokens = tokens + pe.reshape(B, H * W, C)
            for layer in self.layers:
                tokens = layer(tokens)

        tokens = self.final_norm(tokens)

        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


class UNet(nn.Module):
    """Classic 2D U-Net (encoder/decoder with skip connections), not a neural operator.

    Architecture (``n = len(filters)`` stages, ``f_i = filters[i]``)::

        input (B, Cin, H, W)
        -> stem          : PatchEmbed(patch_size)  [if patch_size set]
                           or ConvLayer(3x3, stride 1)
        -> stage 0        : repeats[0] blocks  (f0)          -> skip 0
        -> down 0         : Conv2d(f0 -> f1, k=down_kernel, stride=factor)
        -> stage 1        : repeats[1] blocks  (f1)          -> skip 1
        -> ...
        -> stage n-2      : repeats[n-2] blocks (f_{n-2})     -> skip n-2
        -> down n-2       : Conv2d(f_{n-2} -> f_{n-1}, stride=factor)
        -> bottleneck     : repeats[n-1] blocks (f_{n-1})
                           + [TransformerBottleneck if enabled]
        -> up 0           : PixelShuffleUpsample(f_{n-1} -> f_{n-2}, factor)
                           + skip (n-2) fused (concat or add)
        -> stage D0       : repeats[n-2] blocks (f_{n-2})
        -> ...
        -> up n-2         : PixelShuffleUpsample(f1 -> f0, factor) + skip 0
        -> stage D(n-2)   : repeats[0] blocks (f0)
        -> head           : PixelShuffleUpsample(f0 -> Cout, factor=patch_size)
                           [if patch_size set] or Conv2d(f0 -> Cout, 1x1)
        -> output (B, Cout, H, W)

    Notes:
        - Stage blocks are called with ``in_channels == out_channels`` (channel
          changes are done by the down/up operators), so blocks that only
          support identity channel mapping (``ConvNextv2Block``) work as-is.
        - The model is **not resolution invariant**: the input spatial size
          must be divisible by ``patch_size * factor ** (n - 1)`` for an
          exact-size output.
        - The default ``ResBlock``/``BottleneckBlock`` use a channel-agnostic
          ``LayerNorm2d`` (ConvNeXt-style), so any ``filters`` works out of the
          box; override via ``block_kwargs["norm"]`` if needed.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        block: Callable instantiating a stage block (a class or
            ``functools.partial``). Must accept an ``in_channels`` keyword;
            blocks without an ``out_channels`` keyword (``ConvNextv2Block``)
            are supported. Default ``ResBlock``.
        block_kwargs: Extra keyword arguments forwarded to every stage block
            (e.g. ``{"se": False}`` for ``BottleneckBlock``). Must match the
            block signature.
        filters: Channel widths per stage, e.g. ``(32, 64, 128, 256)``.
        repeats: Number of stage blocks per stage; must match ``len(filters)``.
        patch_size: If set (>= 2), use a ``PatchEmbed`` stem of this patch
            size instead of the plain conv stem; the head un-patches with a
            matching pixel-shuffle. If None, a ``3x3`` conv stem is used.
        stem_kernel_size: Stem conv kernel size (conv stem only).
        stem_norm: Normalization class for the conv stem (or applied after the
            patch stem). ``None`` disables it.
        stem_activation: Activation class for the conv stem. ``None`` disables
            it.
        down_kernel_size: Kernel size of the strided downsampling convs.
        factor: Downsampling stride **and** pixel-shuffle upsampling factor
            (>= 2).
        up_kernel_size: Kernel size of the depthwise refinement conv inside
            ``PixelShuffleUpsample`` (must be odd).
        fusion: Skip-connection fusion, ``"concat"`` (default) or ``"add"``.
        use_transformer_bottleneck: If True (default), insert a
            ``TransformerBottleneck`` after the deepest stage.
        n_transformer_layers: Number of transformer blocks in the bottleneck.
        n_heads: Number of attention heads in the bottleneck.
        mlp_factor: MLP expansion factor in the bottleneck.
        dropout: Dropout rate in the bottleneck.
        pe_method: Positional-encoding method (via ``PEBlock``) for the
            bottleneck. Default ``"adaptive"``. Ignored when ``rope=True``.
        pe_kwargs: Extra keyword arguments forwarded to ``PEBlock``.
        rope: If True, the bottleneck blocks use their built-in rotary
            embedding (``AdaptiveRoPE2D``) instead of the additive ``PEBlock``
            encoding (requires ``filters[-1] // n_heads`` divisible by 4).
        rope_kwargs: Extra keyword arguments forwarded to ``AdaptiveRoPE2D``.
        rmsnorm: Use ``nn.RMSNorm`` (True, default) or ``nn.LayerNorm`` in
            the bottleneck.
        std_ini: Std of the truncated-normal init in the bottleneck blocks.

    Raises:
        ValueError: On inconsistent ``filters``/``repeats``/``factor``/``fusion``
            or an input whose channels or spatial size are invalid (checked at
            forward time).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: Callable = ResBlock,
        block_kwargs: Optional[dict] = None,
        filters: Sequence[int] = (32, 64, 128, 256),
        repeats: Sequence[int] = (2, 2, 2, 2),
        patch_size: Optional[int] = None,
        stem_kernel_size: int = 3,
        stem_norm: Optional[Callable] = LayerNorm2d,
        stem_activation: Optional[Callable] = nn.GELU,
        down_kernel_size: int = 3,
        factor: int = 2,
        up_kernel_size: int = 3,
        fusion: str = "concat",
        use_transformer_bottleneck: bool = True,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        mlp_factor: int = 2,
        dropout: float = 0.1,
        pe_method: str = "adaptive",
        pe_kwargs: Optional[dict] = None,
        rope: bool = False,
        rope_kwargs: Optional[dict] = None,
        rmsnorm: bool = True,
        std_ini: float = 0.02,
    ):
        super().__init__()

        filters = tuple(int(f) for f in filters)
        repeats = tuple(int(r) for r in repeats)

        if len(filters) != len(repeats):
            raise ValueError(f"UNet: filters ({filters}) and repeats ({repeats}) must have the same length.")
        if any(f < 1 for f in filters) or any(r < 1 for r in repeats):
            raise ValueError("UNet: filters and repeats must be positive integers.")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("UNet: in_channels and out_channels must be >= 1.")
        if factor < 2:
            raise ValueError(f"UNet: factor must be >= 2, got {factor}.")
        if fusion not in ("concat", "add"):
            raise ValueError(f"UNet: fusion must be 'concat' or 'add', got {fusion!r}.")
        if patch_size is not None and patch_size < 2:
            raise ValueError(f"UNet: patch_size must be None or >= 2, got {patch_size}.")
        if use_transformer_bottleneck and filters[-1] % n_heads != 0:
            raise ValueError(
                f"UNet: deepest filter width ({filters[-1]}) must be divisible by n_heads ({n_heads})."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.repeats = repeats
        self.factor = factor
        self.fusion = fusion
        self.patch_size = patch_size if patch_size is not None else 1
        self.n_stages = len(filters)

        self.block = block
        self.block_kwargs = dict(block_kwargs or {})
        self._block_takes_out_channels = _block_accepts_out_channels(block)

        # ------------------------------------------------------------------ stem
        if patch_size is not None:
            self.use_patch_embed = True
            self.patch_embed = PatchEmbed(in_channels, filters[0], patch_size=patch_size)
            self.stem_norm = stem_norm(filters[0]) if stem_norm is not None else None
            self.stem_act = stem_activation() if stem_activation is not None else None
        else:
            self.use_patch_embed = False
            self.stem = ConvLayer(
                in_channels,
                filters[0],
                kernel_size=stem_kernel_size,
                stride=1,
                norm=stem_norm,
                activation=stem_activation,
            )

        # ---------------------------------------------------------------- encoder
        self.encoder = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(self.n_stages):
            stage = nn.ModuleList([self._make_block(filters[i], filters[i]) for _ in range(repeats[i])])
            self.encoder.append(stage)
            if i < self.n_stages - 1:
                # Classic Conv2D stride-2 downsampling (channel change included).
                # Padding keeps the spatial size exactly H // factor (for even H
                # and odd kernels), so the pixel-shuffle upsamples restore it.
                self.downsamples.append(
                    nn.Conv2d(
                        filters[i],
                        filters[i + 1],
                        kernel_size=down_kernel_size,
                        stride=factor,
                        padding=down_kernel_size // 2,
                    )
                )

        # ---------------------------------------------------------- bottleneck
        if use_transformer_bottleneck:
            self.bottleneck = TransformerBottleneck(
                filters[-1],
                n_layers=n_transformer_layers,
                n_heads=n_heads,
                mlp_factor=mlp_factor,
                dropout=dropout,
                pe_method=pe_method,
                pe_kwargs=pe_kwargs,
                rope=rope,
                rope_kwargs=rope_kwargs,
                rmsnorm=rmsnorm,
                std_ini=std_ini,
            )
        else:
            self.bottleneck = None

        # ---------------------------------------------------------------- decoder
        self.upblocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
        prev_ch = filters[-1]
        for d in range(self.n_stages - 1):
            target_ch = filters[self.n_stages - 2 - d]
            skip_ch = target_ch if fusion == "concat" else None
            self.upblocks.append(
                PixelShuffleUpsample(
                    prev_ch, target_ch, factor=factor, skip_channels=skip_ch, kernel_size=up_kernel_size
                )
            )
            stage = nn.ModuleList([self._make_block(target_ch, target_ch) for _ in range(repeats[self.n_stages - 2 - d])])
            self.decoder.append(stage)
            prev_ch = target_ch

        # ------------------------------------------------------------------- head
        if patch_size is not None:
            # Un-patch: pixel-shuffle back by the patch size to the input grid.
            self.head = PixelShuffleUpsample(filters[0], out_channels, factor=patch_size, kernel_size=up_kernel_size)
        else:
            self.head = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def _make_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Instantiate one stage block from the configured ``block`` callable.

        Args:
            in_ch: Block input channels (equal to ``out_ch`` by design).
            out_ch: Block output channels.

        Returns:
            The instantiated block.
        """
        if self._block_takes_out_channels:
            return self.block(in_channels=in_ch, out_channels=out_ch, **self.block_kwargs)
        return self.block(in_channels=in_ch, **self.block_kwargs)

    def _stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stem (patch embed or conv) to the input.

        Args:
            x: Input tensor ``(B, in_channels, H, W)``.

        Returns:
            Feature map ``(B, filters[0], H // patch_size, W // patch_size)``.
        """
        if self.use_patch_embed:
            x = self.patch_embed.forward_map(x)
            if self.stem_norm is not None:
                x = self.stem_norm(x)
            if self.stem_act is not None:
                x = self.stem_act(x)
        else:
            x = self.stem(x)
        return x

    def required_divisibility(self) -> int:
        """Minimum spatial divisibility for an exact-size output.

        Returns:
            ``patch_size * factor ** (n_stages - 1)``.
        """
        return self.patch_size * self.factor ** (self.n_stages - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the U-Net.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)`` with ``H`` and
                ``W`` divisible by :meth:`required_divisibility`.

        Returns:
            Tensor of shape ``(B, out_channels, H, W)``.

        Raises:
            ValueError: If the input is not 4D, has a wrong channel count, or a
                spatial size not divisible by :meth:`required_divisibility`.
        """
        if x.ndim != 4:
            raise ValueError(f"UNet expects an input of shape (B, C, H, W), got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"UNet expects {self.in_channels} input channels, got {x.shape[1]}."
            )

        required = self.required_divisibility()
        if x.shape[-2] % required != 0 or x.shape[-1] % required != 0:
            raise ValueError(
                f"UNet: input spatial size {tuple(x.shape[-2:])} must be divisible by {required} "
                f"(patch_size={self.patch_size}, factor={self.factor}, n_stages={self.n_stages}) "
                f"for an exact-size output."
            )

        x = self._stem_forward(x)

        # Encoder: collect the skip of each stage (before its downsampling).
        shortcuts: list[torch.Tensor] = []
        for i in range(self.n_stages):
            for blk in self.encoder[i]:
                x = blk(x)
            if i < self.n_stages - 1:
                shortcuts.append(x)
                x = self.downsamples[i](x)

        # Bottleneck (deepest stage already applied above).
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        # Decoder: mirror the encoder, fusing the corresponding skip.
        shortcuts = shortcuts[::-1]
        for d in range(len(self.upblocks)):
            if self.fusion == "concat":
                x = self.upblocks[d](x, skip=shortcuts[d])
            else:
                x = self.upblocks[d](x)
                x = x + shortcuts[d]
            for blk in self.decoder[d]:
                x = blk(x)

        return self.head(x)


# Backward-compatible name (the historical class was lowercase `unet`).
unet = UNet
