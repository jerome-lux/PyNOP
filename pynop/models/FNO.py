"""Fourier Neural Operator (FNO) model.

A stack of :class:`~pynop.core.blocks.FNOBlock` layers between a 1x1-conv *lifting*
(input -> hidden) and a 1x1-conv *projection* (hidden -> output). Optional positional
encodings (fixed Cartesian coordinates and/or a learnable spectral embedding) and an
optional conditioning field can be concatenated to the input channels before the lifting.

The model is **resolution-invariant**: each spectral convolution keeps a fixed number
of low-frequency modes and, when the input grid is smaller than the configured modes,
uses the low-frequency subspace of its learned weights (see
:class:`~pynop.core.ops.SpectralConv2d`).
"""

from typing import Sequence, Union, Optional, Callable

import collections.abc as abc

import numpy as np
import torch
import torch.nn as nn

from pynop.core.blocks import FNOBlock
from pynop.core.encoding import CartesianEmbedding
from pynop.core.norm import LayerNorm2d

#: Supported block types (registry) -> block class.
REGISTERED_FNO = {"FNO": FNOBlock}


class FNO(nn.Module):
    """Implementation of the Fourier Neural Operator (FNO).

    Architecture::

        x -> [fixed PE] -> [trainable PE] -> [cond] -> lifting (1x1 conv)
            -> block_1 -> block_2 -> ... -> block_n -> projection (1x1 conv)

    where each ``block_i`` is an :class:`~pynop.core.blocks.FNOBlock` (``"FNO"``).

    Parameters
    ----------
    in_channels: int
        Number of input channels (before positional encodings / conditioning).
    out_channels: int
        Number of output channels.
    modes: int or sequence of int
        Number of Fourier modes kept in each dimension. A single int is broadcast to
        both dimensions; a 2-sequence is ``(modes_x, modes_y)``.
    hidden_channels: int or sequence of int
        Hidden width of each block. A single int is used for every block; the number
        of blocks then equals the number of entries in ``blocks`` (or 4 by default when
        ``blocks`` is a single string).
    blocks: str or sequence of str
        Block type(s) per layer. Supported: ``"FNO"``. A single string
        is used for every block. An unknown name raises ``ValueError``.
    spectral_layer_type: str
        Spectral convolution used inside every block: ``"standard"`` (default),
        ``"separable"`` or ``"tucker"``.
    spectral_compression_factor: int, sequence, or None
        Compression factor for the Tucker ranks; only used when
        ``spectral_layer_type="tucker"``. ``None`` means no compression (factor 1).
        An int applies to all three rank dimensions; a 3-sequence applies per dimension.
    activation: callable or None
        Activation used inside the blocks (default ``nn.GELU``); ``None`` disables it.
    norm: callable or None
        Normalization used inside the blocks (default ``LayerNorm2d``); ``None``
        disables it.
    fixed_pos_encoding: bool
        If True (default), concatenate fixed Cartesian coordinates (2 channels) to the
        input. See :class:`~pynop.core.encoding.CartesianEmbedding`.
    trainable_pos_encoding: bool
        If True, concatenate a learnable spectral positional embedding
        (``trainable_pos_encoding_dims`` channels) to the input.
    trainable_pos_encoding_modes: tuple of int
        Base grid ``(mx, my)`` of the learnable positional embedding (default
        ``(16, 16)``); it is resampled to the input resolution. Only used when
        ``trainable_pos_encoding`` is True.
    trainable_pos_encoding_dims: int
        Number of channels of the learnable positional embedding (default ``8``).
        Only used when ``trainable_pos_encoding`` is True.
    local_conv: int
        Number of :class:`~pynop.core.ops.FiniteDifferenceLayer` (differential
        convolution) layers in the local high-frequency branch of every block
        (default ``0`` = disabled). This branch is added in parallel with the
        shortcut and captures local high frequencies that the global spectral
        branch (a structural low-pass) cannot represent, while remaining
        resolution-invariant via the ``1/h`` kernel rescaling.
    local_kernel_size: int
        Spatial kernel size for the local conv layers (odd integer, default ``3``).
    local_grid_width: float
        Physical width of the domain for the local conv layers (default ``2.0``
        for ``[-1, 1]``). Used to compute the grid spacing ``h = grid_width / (N - 1)``.
    block_kwargs: dict, optional
        Extra keyword arguments forwarded to each block constructor.
    dt: float
        Reserved (currently unused). Kept for API compatibility with derivative-output
        FNO variants.
    cond_dim: int, optional
        If set, a conditioning field of ``cond_dim`` channels is concatenated to the
        input (see :meth:`forward`).

    Notes
    -----
    * The number of blocks equals the length of ``hidden_channels`` / ``blocks`` after
      normalization.
    * ``spectral_compression_factor`` is ignored for the ``"standard"`` and
      ``"separable"`` spectral layers (it only parameterizes the Tucker ranks).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Union[int, Sequence[int]] = (64, 64, 64, 64),
        blocks: Union[str, Sequence[str]] = "FNO",
        spectral_layer_type: str = "standard",
        spectral_compression_factor: Union[None, int, Sequence[int]] = None,
        activation: Union[Callable, None] = nn.GELU,
        norm: Union[Callable, None] = LayerNorm2d,
        fixed_pos_encoding: bool = True,
        trainable_pos_encoding: bool = False,
        trainable_pos_encoding_modes: Sequence[int] = (16, 16),
        trainable_pos_encoding_dims: int = 8,
        local_conv: int = 0,
        local_kernel_size: int = 3,
        local_grid_width: float = 2.0,
        block_kwargs: Optional[dict] = None,
        dt: float = 1.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fixed_pos_encoding = fixed_pos_encoding
        self.trainable_pos_encoding = trainable_pos_encoding
        self.dt = dt

        # ------------------------------------------------------------------ input channels
        # Total input channels = in_channels + (2 if fixed PE) + (dims if trainable PE)
        #                         + (cond_dim if cond).
        total_in = in_channels
        if fixed_pos_encoding:
            total_in += 2
            self.grid_encoding = CartesianEmbedding()
        if trainable_pos_encoding:
            self.pos_embedding_weights = nn.Parameter(
                torch.randn(1, trainable_pos_encoding_dims, *trainable_pos_encoding_modes, dtype=torch.cfloat)
            )
            total_in += trainable_pos_encoding_dims
        if cond_dim is not None:
            total_in += cond_dim
        self._lifting_in_channels = total_in

        # ------------------------------------------------------------- block configuration
        # Normalize `modes` to a 2-tuple (used for the Tucker rank `prod(modes)`).
        modes = (modes, modes) if isinstance(modes, int) else tuple(modes)
        if len(modes) != 2:
            raise ValueError(f"modes must be an int or a 2-sequence, got {modes!r}.")
        self.modes = modes

        # Resolve `blocks` and `hidden_channels` to two equal-length sequences.
        hidden_channels, blocks = self._resolve_blocks(hidden_channels, blocks)

        # Validate block names (no silent fallback to a default block).
        for name in blocks:
            if name not in REGISTERED_FNO:
                raise ValueError(
                    f"Unknown block type {name!r}. Valid block types: {sorted(REGISTERED_FNO)}."
                )

        # ----------------------------------------------------------------------- lifting
        self.lifting = nn.Conv2d(total_in, hidden_channels[0], 1, bias=True)

        # --------------------------------------------------------------------------- blocks
        self.ops = nn.ModuleList()
        for i, out_ch in enumerate(hidden_channels):
            in_ch = hidden_channels[0] if i == 0 else hidden_channels[i - 1]
            block_cls = REGISTERED_FNO[blocks[i]]
            kwargs = dict(
                in_channels=in_ch,
                out_channels=out_ch,
                modes=modes,
                activation=activation,
                normalization=norm,
                spectral_layer_type=spectral_layer_type,
                local_conv=local_conv,
                local_kernel_size=local_kernel_size,
                local_grid_width=local_grid_width,
                **(block_kwargs or {}),
            )
            if spectral_layer_type == "tucker":
                factor = (1, 1, 1) if spectral_compression_factor is None else spectral_compression_factor
                ranks = np.ceil(
                    np.divide([in_ch, out_ch, np.prod(modes)], factor)
                ).astype(int)
                kwargs["ranks"] = tuple(ranks)
            self.ops.append(block_cls(**kwargs))

        # --------------------------------------------------------------------- projection
        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    @staticmethod
    def _resolve_blocks(
        hidden_channels: Union[int, Sequence[int]], blocks: Union[str, Sequence[str]]
    ) -> tuple:
        """Normalize ``hidden_channels`` and ``blocks`` to equal-length sequences.

        Rules:
        - a scalar is broadcast to the length of the other argument (if it is a
          sequence), or to the default depth (4) when both are scalars;
        - two sequences must have the same length.
        """
        hidden_is_seq = isinstance(hidden_channels, abc.Sequence) and not isinstance(hidden_channels, str)
        blocks_is_seq = isinstance(blocks, abc.Sequence) and not isinstance(blocks, str)

        if hidden_is_seq and blocks_is_seq:
            if len(hidden_channels) != len(blocks):
                raise ValueError(
                    "hidden_channels and blocks must have the same length "
                    f"({len(hidden_channels)} != {len(blocks)})."
                )
            return list(hidden_channels), list(blocks)

        if hidden_is_seq:  # blocks is a scalar -> repeat for every hidden channel
            return list(hidden_channels), [blocks] * len(hidden_channels)

        if blocks_is_seq:  # hidden is a scalar -> repeat for every block
            return [hidden_channels] * len(blocks), list(blocks)

        # Both scalars -> default depth of 4 blocks.
        default_depth = 4
        return [hidden_channels] * default_depth, [blocks] * default_depth

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input of shape ``(B, in_channels, H, W)``.
            cond: Optional conditioning field of shape
                ``(B, cond_dim, H, W)`` (requires ``cond_dim`` to be set).

        Returns:
            torch.Tensor: Output of shape ``(B, out_channels, H, W)``.
        """
        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)  # (B, C + 2, H, W)

        if self.trainable_pos_encoding:
            # Resample the learnable spectral embedding to the input resolution and
            # broadcast it over the batch.
            pos_embeddings = torch.fft.irfftn(self.pos_embedding_weights, s=x.shape[-2:])
            repeat_shape = [1] * len(x.shape)
            repeat_shape[0] = x.shape[0]
            x = torch.cat([x, pos_embeddings.repeat(*repeat_shape)], dim=1)

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        x = self.lifting(x)
        for op in self.ops:
            x = op(x)
        x = self.projection(x)
        return x
