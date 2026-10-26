import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable, Type

from ..core.blocks import (
    TransformerBlock,
    PEBlock,
    WindowAttention,
    WindowCrossAttention,
    AdaptiveSwinBlock,
    PaddedSwinBlock,
    BlockCrossAttentionUpsampler,
)


class SwinNO(nn.Module):
    """A Swin-based neural operator that maps input fields to output fields while preserving spatial structure."""

    def __init__(
        self,
        num_windows: int,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        num_blocks: int = 5,
        num_heads: int = 4,
        mlp_factor: int = 4,
        activation: Type[nn.Module] = nn.GELU,
        block_type: str = "swin",
        bc_mode: str = "circular",
        rmsnorm: bool = True,
        cond_dim: Optional[int] = None,
        ndim: int = 2,
    ):
        """Initialize the neural operator.

        Args:
            num_windows: Number of windows per axis. The window size adapts to the input resolution
                (``win = H // num_windows``), so the model works at any resolution divisible by it.
            in_channels: Number of input channels in the physical field.
            out_channels: Number of output channels in the predicted field.
            latent_dim: Hidden dimension of the latent representation.
            num_blocks: Number of Swin blocks in the encoder stack.
            num_heads: Number of attention heads in each block.
            mlp_factor: Expansion factor for the MLP hidden size.
            activation: Activation class used by the MLP layers.
            block_type: Type of Swin block to use: "swin" or "padded".
            bc_mode: Boundary padding mode for padded blocks.
            rmsnorm: Use RMSNorm if True, else LayerNorm.
            cond_dim: If set, a conditioning field with this many channels is concatenated to the input.
            ndim: Number of spatial coordinates used as positional features.
        """
        super().__init__()

        assert latent_dim % num_heads == 0, "Model dimension must be divisible by num_heads"
        assert block_type.lower() in [
            "swin",
            "padded",
        ], f"Block type {block_type} is not supported. Supported blocks: 'swin' or 'padded'"

        self.num_windows = num_windows  # Number of windows per axis
        self.latent_dim = latent_dim  # Global depth (D)
        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm
        in_ch = in_channels if cond_dim is None else in_channels + cond_dim
        # Lift the input features to the latent dimension.
        self.pos_encoder = PEBlock(in_features=2, out_features=latent_dim, method="adaptive")
        self.mixer = nn.Linear(2 * latent_dim, latent_dim)
        self.input_proj = nn.Linear(in_ch, latent_dim)
        self.input_norm = norm_layer(latent_dim)

        blocks = []
        for i in range(num_blocks):
            # Alternate shift (if num_block is even, start with shifted block)
            if num_blocks % 2 == 0:
                shift = i % 2 == 0
            else:
                shift = i % 2 == 1

            if block_type == "swin":
                blocks.append(
                    AdaptiveSwinBlock(
                        dim=latent_dim,
                        grid_windows=(num_windows, num_windows),
                        num_heads=num_heads,
                        shift=shift,
                        mlp_factor=mlp_factor,
                        activation=activation,
                    )
                )
            elif block_type == "padded":
                blocks.append(
                    PaddedSwinBlock(
                        dim=latent_dim,
                        grid_windows=(num_windows, num_windows),
                        num_heads=num_heads,
                        pad=shift,  # Pad on shifted layers
                        mlp_factor=mlp_factor,
                        activation=activation,
                        bc_mode=bc_mode,
                    )
                )

        self.swin_layers = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(latent_dim, out_channels)

    def _get_coords(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1).view(1, H, W, 2)
        cell_size = torch.tensor([2.0 / H, 2.0 / W], device=device).view(1, 1, 1, 2)
        return coords, cell_size

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input x: [B, in_channels, H, W] (Physical tensor)
        Output:  [B, M*M, latent_dim] (Flat sequence of tokens for temporal model)
        """

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        B, C, H_in, W_in = x.shape
        device = x.device

        in_coords, _ = self._get_coords(H_in, W_in, device)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.input_proj(x)
        x = self.input_norm(x)

        pe = self.pos_encoder(in_coords).expand(B, -1, -1, -1)
        x = self.mixer(torch.cat([x, pe], dim=-1))

        # Process through Swin Blocks
        x = self.swin_layers(x)
        x = self.out_proj(x)
        x = x.permute(0, 3, 1, 2)

        return x


class MultiscaleCAEncoder(nn.Module):
    """
    Multiscale Encoder combining Window Cross-Attention and Swin/Global Self-Attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_windows: List[Tuple[int, int]],
        model_dim: Union[int, List[int]] = 128,
        grid_resolutions: List[Tuple[int, int]] = [(32, 32), (16, 16)],
        num_heads: int = 4,
        mlp_factor: int = 4,
        repeats: Union[int, List[int]] = 1,
        rmsnorm: bool = True,
        activation: Callable = nn.GELU,
        ndim: int = 2,
    ):
        """
        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        out_channels : int
            Output feature dimension.
        num_windows : List[Tuple[int, int]]
            Number of windows per axis for each Swin stage (the window size adapts to the
            stage resolution). Use ``None`` for a stage to apply a global Transformer block.
        model_dim : Union[int, List[int]]
            Hidden dimensions per stage.
        grid_resolutions : List[Tuple[int, int]]
            Target resolutions per stage.
        num_heads : int
            Number of attention heads.
        mlp_factor : int
            MLP dimension multiplier.
        repeats : Union[int, List[int]]
            Number of repetitions for self-attention blocks per stage.
            Can be a single int or a list matching len(grid_resolutions).
        rmsnorm : bool
            Use RMSNorm if True, else LayerNorm.
        activation : Callable
            Activation function class.
        ndim : int
            Spatial dimensions.
        """
        super().__init__()
        self.grid_resolutions = grid_resolutions
        self.num_windows = num_windows
        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm
        num_stages = len(grid_resolutions)

        if isinstance(model_dim, int):
            self.model_dim = [model_dim] * (num_stages + 1)
        else:
            self.model_dim = model_dim

        # Adapt repeats to be a list with one value per stage
        if isinstance(repeats, int):
            self.repeats = [repeats] * num_stages
        else:
            if len(repeats) != num_stages:
                raise ValueError(f"Length of repeats ({len(repeats)}) must match num_stages ({num_stages})")
            self.repeats = repeats

        self.pos_encoder = PEBlock(in_features=2, out_features=self.model_dim[0], method="adaptive")
        self.mixer = nn.Linear(2 * self.model_dim[0], self.model_dim[0])
        self.input_proj = nn.Linear(in_channels, self.model_dim[0])
        self.input_norm = norm_layer(self.model_dim[0])

        self.stages = nn.ModuleList()

        for i, (res, win) in enumerate(zip(grid_resolutions, num_windows)):
            current_dim = self.model_dim[i]
            next_dim = self.model_dim[i + 1]
            stage_repeats = self.repeats[i]

            print(f"building encoder block {i+1} using {win} windows. Target resolution {res}")
            print(f"Current model dim: {current_dim}, next level {next_dim} | Repeats: {stage_repeats}")

            cross_attn = WindowCrossAttention(in_ch=current_dim, dim=next_dim, num_heads=num_heads, m=1)
            cross_norm = norm_layer(next_dim)

            self_attn_blocks = nn.ModuleList()
            if win is not None:
                # 1 repeat = 1 pair of Swin blocks (unshifted + shifted)
                for _ in range(stage_repeats):
                    self_attn_blocks.append(
                        AdaptiveSwinBlock(
                            dim=next_dim,
                            grid_windows=win,
                            num_heads=num_heads,
                            activation=activation,
                            mlp_factor=mlp_factor,
                            shift=False,
                        )
                    )
                    self_attn_blocks.append(
                        AdaptiveSwinBlock(
                            dim=next_dim,
                            grid_windows=win,
                            num_heads=num_heads,
                            activation=activation,
                            mlp_factor=mlp_factor,
                            shift=True,
                        )
                    )
            else:
                # 1 repeat = 1 standard Transformer block
                for _ in range(stage_repeats):
                    self_attn_blocks.append(
                        TransformerBlock(
                            dim=next_dim,
                            n_heads=num_heads,
                            activation=activation,
                            rmsnorm=rmsnorm,
                            mlp_dim=mlp_factor * next_dim,
                        )
                    )

            self.stages.append(
                nn.ModuleDict(
                    {
                        "cross_attn": cross_attn,
                        "cross_norm": cross_norm,
                        "self_attn": self_attn_blocks,
                    }
                )
            )

        self.output_proj = nn.Linear(self.model_dim[-1], out_channels)

    def _get_coords(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1).view(1, H, W, 2)
        cell_size = torch.tensor([2.0 / H, 2.0 / W], device=device).view(1, 1, 1, 2)
        return coords, cell_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode an input field through the multiscale encoder and return latent shortcuts.

        Args:
            x: Input tensor of shape [B, C, H_in, W_in].

        Returns:
            A tuple containing the final latent tensor and the list of stage shortcuts.
        """
        B, C, H_in, W_in = x.shape
        device = x.device

        in_coords, _ = self._get_coords(H_in, W_in, device)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.input_proj(x)
        x = self.input_norm(x)

        pe = self.pos_encoder(in_coords).expand(B, -1, -1, -1)
        x = self.mixer(torch.cat([x, pe], dim=-1))

        shortcuts = [x.view(B, H_in, W_in, -1)]
        kv_context = x  # [B, H*W, D]
        coords_win = in_coords
        H_curr, W_curr = H_in, W_in

        for i, stage in enumerate(self.stages):
            H_tgt, W_tgt = self.grid_resolutions[i]
            D_curr = self.model_dim[i]
            D_next = self.model_dim[i + 1]

            win_h, win_w = H_curr // H_tgt, W_curr // W_tgt
            assert H_curr % H_tgt == 0 and W_curr % W_tgt == 0, "Resolution mismatch"

            kv_context = kv_context.view(B, H_tgt, win_h, W_tgt, win_w, D_curr)
            kv_context = kv_context.permute(0, 1, 3, 2, 4, 5).contiguous()
            kv_context = kv_context.view(B * H_tgt * W_tgt, win_h * win_w, D_curr)

            coords_win = coords_win.view(1, H_tgt, win_h, W_tgt, win_w, 2).expand(B, -1, -1, -1, -1, -1).contiguous()
            coords_win = coords_win.permute(0, 1, 3, 2, 4, 5).contiguous()
            coords_win = coords_win.view(B * H_tgt * W_tgt, win_h * win_w, 2)

            x_latent = stage["cross_attn"](kv_context, coords_win)
            x_latent = stage["cross_norm"](x_latent)

            x_latent = x_latent.view(B, H_tgt, W_tgt, D_next)

            if self.num_windows[i] is not None:
                for swin_block in stage["self_attn"]:
                    x_latent = swin_block(x_latent)
                kv_context = x_latent
            else:
                x_latent = x_latent.view(B, H_tgt * W_tgt, D_next)
                for trans_block in stage["self_attn"]:
                    x_latent = trans_block(x_latent)
                kv_context = x_latent.view(B, H_tgt, W_tgt, D_next)

            shortcuts.append(kv_context)
            H_curr, W_curr = H_tgt, W_tgt
            coords_win, _ = self._get_coords(H_curr, W_curr, device)

        out = self.output_proj(kv_context)
        return out, shortcuts


class MultiscaleCADecoder(nn.Module):
    """
    Multiscale Decoder handling progressive upsampling via cross-attention and self-attention refinement.
    """

    def __init__(
        self,
        latent_dim: List[int],
        out_channels: int,
        num_windows: List[Union[Tuple[int, int], None]],
        n_heads: int = 4,
        mlp_ratio: int = 4,
        rmsnorm: bool = True,
        repeats: Union[List[int], int] = 1,
        activation: Callable = nn.GELU,
    ):
        """
        Args:
            latent_dim: List of hidden dimensions from lowest to highest resolution.
            out_channels: Final output feature dimension.
            num_windows: Number of windows per axis for each stage (Tuple for Swin, None for Global).
            n_heads: Number of attention heads per stage.
            mlp_ratio: Multiplier for MLP hidden dimension.
            rmsnorm: Use RMSNorm if True, else LayerNorm.
            repeats: Number of self-attention block pairs per stage (int or list matching the number of stages).
            activation: Activation function class.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_windows = num_windows
        num_stages = len(num_windows)

        if isinstance(repeats, int):
            self.repeats = [repeats] * num_stages
        else:
            if len(repeats) != num_stages:
                raise ValueError(f"Length of repeats ({len(repeats)}) must match num_stages ({num_stages})")
            self.repeats = repeats

        self.stages = nn.ModuleList()

        for i, windows in enumerate(num_windows):
            current_dim = self.latent_dim[i]
            next_dim = self.latent_dim[i + 1]
            stage_repeats = self.repeats[i]
            print(f"building decoder block {i+1} using {windows} windows")
            print(
                f"Current model dim: {self.latent_dim[i]}, next :level {self.latent_dim[i + 1]} | repeats: {stage_repeats}"
            )

            upscaler = BlockCrossAttentionUpsampler(
                latent_dim=current_dim,
                shortcut_dim=next_dim,
                num_heads=n_heads,
                activation=activation,
                mlp_factor=mlp_ratio,
            )
            self_attn_blocks = nn.ModuleList()
            if windows is not None:
                for _ in range(stage_repeats):
                    self_attn_blocks.append(
                        AdaptiveSwinBlock(
                            dim=next_dim,
                            grid_windows=windows,
                            num_heads=n_heads,
                            activation=activation,
                            mlp_factor=mlp_ratio,
                            shift=False,
                        )
                    )

                    self_attn_blocks.append(
                        AdaptiveSwinBlock(
                            dim=next_dim,
                            grid_windows=windows,
                            num_heads=n_heads,
                            activation=activation,
                            mlp_factor=mlp_ratio,
                            shift=True,
                        )
                    )

            else:
                for _ in range(stage_repeats):
                    self_attn_blocks.append(
                        TransformerBlock(
                            dim=next_dim,
                            n_heads=n_heads,
                            activation=activation,
                            rmsnorm=rmsnorm,
                            mlp_dim=mlp_ratio * next_dim,
                        )
                    )

            self.stages.append(
                nn.ModuleDict(
                    {
                        "upscaler": upscaler,
                        "self_attn": self_attn_blocks,
                    }
                )
            )

        self.head = nn.Sequential(
            nn.RMSNorm(latent_dim[-1]),
            nn.Linear(latent_dim[-1], latent_dim[-1]),
            activation(),
            nn.Linear(latent_dim[-1], out_channels),
        )

    def forward(self, z: torch.Tensor, shortcuts: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z: Initial structural latent tensor [B, H_in, W_in, D_lowest].
            shortcuts: List of encoder shortcut tensors from highest to lowest resolution.
        Returns:
            Output tensor of shape [B, C_out, H_highest, W_highest].
        """
        # Align shortcuts from low-res to high-res
        shortcuts_sorted = list(reversed(shortcuts))

        for stage_idx, shortcut in enumerate(shortcuts_sorted):
            # print("stage", stage_idx, shortcut.shape)
            B, H, W, _ = shortcut.shape
            windows = self.num_windows[stage_idx]

            # Upsample the low-resolution latent grid using the high-resolution shortcut.
            z = self.stages[stage_idx]["upscaler"](z, shortcut)  # Output: [B, H, W, D_next]

            # Refine the upsampled features with self-attention.
            if windows is not None:
                # Swin blocks process [B, H, W, D] directly
                for blk in self.stages[stage_idx]["self_attn"]:
                    z = blk(z)
            else:
                # Global transformer expects sequential input [B, H*W, D]
                D_current = z.shape[-1]
                z = z.view(B, H * W, D_current)
                for blk in self.stages[stage_idx]["self_attn"]:
                    z = blk(z)
                # Reshape back to 2D spatial grid for the next upsampling stage
                z = z.view(B, H, W, D_current)

        # Project the final features to the output channels.
        out = self.head(z)  # [B, H_highest, W_highest, out_channels]
        return out.permute(0, 3, 1, 2).contiguous()


class MultiscaleSWINNO(nn.Module):
    """
    Multiscale SWIN Neural Operator
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        enc_num_windows: List[Tuple[int, int]],
        dec_num_windows: List[Union[Tuple[int, int], None]],
        model_dim: Union[int, List[int]] = 128,
        grid_resolutions: List[Tuple[int, int]] = [(32, 32), (16, 16)],
        enc_repeats: Union[List[int], int] = 1,
        dec_repeats: Union[List[int], int] = 1,
        num_heads: int = 4,
        mlp_factor: int = 4,
        rmsnorm: bool = True,
        activation: Callable = nn.GELU,
        cond_dim: Optional[int] = None,
    ):
        """
        Args:
            in_channels:       Number of input channels.
            out_channels:      Number of output channels.
            enc_num_windows: Number of windows per axis for the encoder stages.
            dec_num_windows: Number of windows per axis for the decoder stages (None for a global stage).
            model_dim:         Hidden dimensions (int or list mapping resolutions).
            grid_resolutions:  Target resolutions for downsampling stages.
            enc_repeats:       Number of self-attention block pairs per encoder stage (int or list matching the number of stages).
            dec_repeats:       Number of self-attention block pairs per decoder stage (int or list matching the number of stages).
            num_heads:         Number of attention heads.
            mlp_factor:        Multiplier for MLP hidden dimension.
            rmsnorm:           Use RMSNorm if True, else LayerNorm.
            activation:        Activation function class.
            cond_dim:          If set, a conditioning field with this many channels is concatenated to the input.
        """
        super().__init__()

        # Set up the encoder and decoder dimensions.
        if isinstance(model_dim, int):
            self.enc_dims = [model_dim] * (len(grid_resolutions) + 1)
        else:
            self.enc_dims = model_dim

        # Mirror the encoder dimensions in reverse order for the decoder.
        self.dec_dims = list(reversed(self.enc_dims))

        in_ch = in_channels if cond_dim is None else in_channels + cond_dim

        # Instantiate sub-modules
        self.encoder = MultiscaleCAEncoder(
            in_channels=in_ch,
            out_channels=self.enc_dims[-1],
            num_windows=enc_num_windows,
            model_dim=self.enc_dims,
            grid_resolutions=grid_resolutions,
            num_heads=num_heads,
            mlp_factor=mlp_factor,
            rmsnorm=rmsnorm,
            activation=activation,
            repeats=enc_repeats,
        )

        self.decoder = MultiscaleCADecoder(
            latent_dim=self.dec_dims,
            out_channels=out_channels,
            num_windows=dec_num_windows,
            n_heads=num_heads,
            mlp_ratio=mlp_factor,
            rmsnorm=rmsnorm,
            activation=activation,
            repeats=dec_repeats,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Maps input tensor to latent space and returns shortcuts.

        Args:
            x: Input tensor [B, in_channels, H, W].
        Returns:
            Tuple containing:
                - Latent tensor [B, H_last * W_last, D_latent]
                - List of shortcut tensors from each encoder stage.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor, shortcuts: List[torch.Tensor]) -> torch.Tensor:
        """
        Decodes latent tensor back to target spatial resolution using shortcuts.

        Args:
            z: Latent tensor [B, H_last * W_last, D_latent] or [B, H_last, W_last, D_latent].
            shortcuts: List of shortcut tensors from the encoder.
        Returns:
            Reconstructed output tensor [B, out_channels, H_orig, W_orig].
        """
        # Reconstruct spatial structure if latent comes flat from the global stage
        if len(z.shape) == 3:
            H_last, W_last = self.encoder.grid_resolutions[-1]
            B, _, D = z.shape
            z = z.view(B, H_last, W_last, D)

        return self.decoder(z, shortcuts)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the full multi-scale cross-attention autoencoder.

        Args:
            x: Input tensor [B, in_channels, H, W].
            cond: constant tensor field concatenated to the input
        Returns:
            Output reconstruction [B, out_channels, H, W].
        """

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        # Compress the input with progressive cross-attentions.
        z, shortcuts = self.encode(x)

        # Decompress the latent representation with block-grouped cross-attentions.
        out = self.decode(z, shortcuts[:-1])

        return out
