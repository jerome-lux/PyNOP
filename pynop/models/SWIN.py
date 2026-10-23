import math
from typing import Callable, Type, Union, Tuple, List
from git import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from ..core.norm import AdaRMSNorm, AdaptiveLayerNorm
from ..core.blocks import TransformerBlock, MLPBlock, PEBlock
from ..core.ops import Attention
from ..core.encoding import AdaptiveRoPE2D, IntegratedPositionalEncoding, AdaptiveFourierEmbedding


class WindowAttention(nn.Module):
    """Window-based self-attention layer with rotary positional encoding."""

    def __init__(self, dim: int, num_heads: int):
        """Initialize the attention layer.

        Args:
            dim: Size of each token embedding.
            num_heads: Number of attention heads. The embedding size must be divisible by this value.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.scale = self.d_head**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.rope = AdaptiveRoPE2D(self.d_head)

    def forward(
        self,
        x: torch.Tensor,  # [B*nW, N, dim]
        coords: torch.Tensor,  # [B*nW, win_h, win_w, 2]  absolute coordinates(y,x) in [-1, 1]
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply window attention to a batch of local tokens.

        Args:
            x: Input tokens with shape [B*nW, N, dim].
            coords: Absolute coordinates (of original grid) reshaped as [num_windows, win_h*win_w, win_h*win_w, 2]
            mask: Optional attention mask with shape [num_windows, win_h*win_w, win_h*win_w].

        Returns:
            A tensor of shape [B*nW, N, dim] containing the attended features.
        """

        b_win, n, c = x.shape
        h = self.num_heads
        # [3, b*nwin, heads, n, d_head] where n is the number of token in a window
        qkv = self.qkv(x).reshape(b_win, n, 3, h, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*nW, Heads, N, d_head]

        # Apply RoPE to queries and keys only.
        q = self.rope.rotate(q, coords)
        k = self.rope.rotate(k, coords)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_win // nw, nw, h, n, n) + mask.unsqueeze(1)
            attn = attn.view(b_win, h, n, n)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_win, n, c)
        return self.proj(x)


class WindowCrossAttention(nn.Module):
    """Local cross-attention module that compresses tokens into a small learnable query grid."""

    def __init__(self, in_ch: int, dim: int, num_heads: int, m: int = 1):
        """Initialize the cross-attention module.

        Args:
            in_ch: Input feature dimension for the keys and values.
            dim: Hidden dimension used for the queries, keys, values, and output projection.
            num_heads: Number of attention heads. The dimension must be divisible by this value.
            m: Side length of the learnable query grid, yielding $m^2$ queries per window.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.m = m
        self.scale = self.d_head**-0.5

        # Learnable queries defined in the target latent space.
        self.latent_queries = nn.Parameter(torch.randn(1, m * m, dim) * 0.02)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(in_ch, dim, bias=True)
        self.v_proj = nn.Linear(in_ch, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compress a set of local tokens into a learned query grid.

        Args:
            x: Input tokens with shape [B_win, N_in, in_ch].
            coords: Absolute coordinates for the input tokens with shape [B_win, N_in, 2].

        Returns:
            Aggregated tokens with shape [B_win, m*m, dim].
        """
        b_win, n, _ = x.shape
        h = self.num_heads

        # Project queries
        q = self.latent_queries.expand(b_win, -1, -1)
        q = self.q_proj(q).reshape(b_win, self.m * self.m, h, self.d_head).transpose(1, 2)

        # Project keys and values
        k = self.k_proj(x).reshape(b_win, n, h, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(b_win, n, h, self.d_head).transpose(1, 2)

        # Position encoding on keys can be added here if needed.

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b_win, self.m * self.m, self.dim)
        return self.out_proj(out)


class AdaptiveSwinBlock(nn.Module):
    """A Swin-style transformer block with local window attention and optional shifted windows."""

    def __init__(
        self,
        dim: int,
        grid_windows: tuple[int, int],
        num_heads: int,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
        shift: bool = False,
    ):
        """Initialize the Swin block.

        Args:
            dim: Feature dimension of each token.
            grid_windows: Number of windows along the height and width axes, provided as (nw_h, nw_w).
            num_heads: Number of attention heads in the local window attention module.
            activation: Activation class used by the MLP subnetwork.
            mlp_factor: Expansion factor for the hidden size of the MLP.
            shift: If True, use shifted windows in the attention operation.
        """
        super().__init__()
        self.dim = dim
        self.nw_h, self.nw_w = grid_windows
        self.num_heads = num_heads
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_factor * dim),
            activation(),
            nn.Linear(mlp_factor * dim, dim),
        )

    def _create_mask(self, h, w, win_h, win_w, shift_h, shift_w, device):
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (slice(0, -win_h), slice(-win_h, -shift_h), slice(-shift_h, None))
        w_slices = (slice(0, -win_w), slice(-win_w, -shift_w), slice(-shift_w, None))
        cnt = 0
        for h_sl in h_slices:
            for w_sl in w_slices:
                img_mask[:, h_sl, w_sl, :] = cnt
                cnt += 1
        mask_windows = img_mask.view(1, self.nw_h, win_h, self.nw_w, win_w, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _build_coords(self, h, w, win_h, win_w, shift_h, shift_w, b, device):
        """Build absolute coordinates for tokens after window partitioning.

        Returns a tensor of shape [B*nW, win_h*win_w, 2] in the range [-1, 1].
        """
        # Global grid before shifting.
        gy = (torch.arange(h, device=device).float() / h) * 2 - 1  # [-1, 1)
        gx = (torch.arange(w, device=device).float() / w) * 2 - 1
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [H, W]
        coords_img = torch.stack([cy, cx], dim=-1)  # [H, W, 2]

        # Apply the same shift to the coordinates as to the feature map.
        if shift_h > 0 or shift_w > 0:
            coords_img = torch.roll(coords_img, shifts=(-shift_h, -shift_w), dims=(0, 1))

        # Partition the coordinates exactly like the feature tokens.
        coords_win = coords_img.view(self.nw_h, win_h, self.nw_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(self.nw_h * self.nw_w, win_h, win_w, 2)

        # Repeat over the batch dimension.
        coords_win = coords_win.unsqueeze(0).expand(b, -1, -1, -1, -1)  # [B, nW, win_h, win_w, 2]
        return coords_win.reshape(b * self.nw_h * self.nw_w, win_h, win_w, 2)

    def forward(self, x: torch.Tensor, use_mask: bool = True) -> torch.Tensor:
        """Apply the Swin block to a 4D feature map.

        Args:
            x: Input tensor of shape [B, H, W, C].
            use_mask: If True, generate an attention mask when shifted windows are enabled.

        Returns:
            A tensor with the same shape as the input, containing the updated features.
        """
        b, h, w, d = x.shape
        shortcut = x
        x = self.norm1(x)

        win_h, win_w = h // self.nw_h, w // self.nw_w
        shift_h, shift_w = (win_h // 2, win_w // 2) if self.shift else (0, 0)

        if self.shift and use_mask:
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
            mask = self._create_mask(h, w, win_h, win_w, shift_h, shift_w, x.device)
        else:
            mask = None

        # Absolute coordinates with shape [B * num_win, win_h, win_w, 2].
        coords = self._build_coords(h, w, win_h, win_w, shift_h, shift_w, b, x.device)

        x_win = x.view(b, self.nw_h, win_h, self.nw_w, win_w, d)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, d)

        attn_win = self.attn(x_win, coords=coords, mask=mask)

        x = attn_win.view(b, self.nw_h, self.nw_w, win_h, win_w, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, d)

        if self.shift:
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

        return shortcut + x + self.mlp(self.norm2(x))


class PaddedSwinBlock(nn.Module):
    """A Swin-style block with optional padding and boundary-aware window attention."""

    def __init__(
        self,
        dim: int,
        grid_windows: tuple[int, int],
        num_heads: int,
        activation: Callable = nn.GELU,
        pad: bool = False,
        mlp_factor: int = 4,
        bc_mode: str = "circular",
        rope_max_freq: float = 8.0,
    ):
        """Initialize the padded Swin block.

        Args:
            dim: Feature dimension of each token.
            grid_windows: Number of windows along height and width, provided as (nw_h, nw_w).
            num_heads: Number of attention heads in the local window attention module.
            activation: Activation class used by the MLP subnetwork.
            pad: If True, pad the input tensor before computing window attention.
            mlp_factor: Expansion factor for the hidden size of the MLP.
            bc_mode: Boundary padding mode used for the padded feature map.
            rope_max_freq: Maximum frequency used by the positional encoding component.
        """
        super().__init__()
        self.dim = dim
        self.nw_h, self.nw_w = grid_windows
        self.num_heads = num_heads
        self.pad = pad
        self.bc_mode = bc_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_factor * dim),
            activation(),
            nn.Linear(mlp_factor * dim, dim),  # corrig� : �tait 4*dim
        )

    def _build_coords(
        self,
        h: int,
        w: int,
        h_pad: int,
        w_pad: int,
        pad_h: int,
        pad_w: int,
        win_h: int,
        win_w: int,
        current_nw_h: int,
        current_nw_w: int,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build absolute coordinates for padded tokens in the range [-1, 1].

        Returns a tensor of shape [B * current_nw_h * current_nw_w, win_h * win_w, 2].
        """
        # Uniform step based on the original grid.
        step_y = 2.0 / h
        step_x = 2.0 / w

        # Coordinates of the padded tensor: original tokens lie in [-1, 1), while padded tokens lie outside that range.
        gy = (torch.arange(h_pad, device=device).float() - pad_h) * step_y - 1.0 + step_y / 2
        gx = (torch.arange(w_pad, device=device).float() - pad_w) * step_x - 1.0 + step_x / 2

        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [H_pad, W_pad]
        coords_img = torch.stack([cy, cx], dim=-1)  # [H_pad, W_pad, 2]

        # Partition the coordinates like the flattened window tokens.
        coords_win = coords_img.view(current_nw_h, win_h, current_nw_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(current_nw_h * current_nw_w, win_h, win_w, 2)

        return coords_win.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * current_nw_h * current_nw_w, win_h, win_w, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the padded Swin block to a 4D feature map.

        Args:
            x: Input tensor of shape [B, H, W, C].

        Returns:
            A tensor with the same shape as the input, containing the updated features.
        """
        b, h, w, d = x.shape
        shortcut = x
        x = self.norm1(x)

        win_h, win_w = h // self.nw_h, w // self.nw_w

        if self.pad:
            pad_h, pad_w = win_h // 2, win_w // 2

            x_padded = F.pad(
                x.permute(0, 3, 1, 2),
                (pad_w, pad_w, pad_h, pad_h),
                mode=self.bc_mode,
            ).permute(
                0, 2, 3, 1
            )  # [B, H_pad, W_pad, D]

            h_pad, w_pad = x_padded.shape[1], x_padded.shape[2]
            current_nw_h = h_pad // win_h
            current_nw_w = w_pad // win_w

            coords = self._build_coords(
                h,
                w,
                h_pad,
                w_pad,
                pad_h,
                pad_w,
                win_h,
                win_w,
                current_nw_h,
                current_nw_w,
                b,
                x.device,
            )

            x_win = x_padded.view(b, current_nw_h, win_h, current_nw_w, win_w, d)
            x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, d)

            attn_win = self.attn(x_win, coords=coords)

            x_merged = attn_win.view(b, current_nw_h, current_nw_w, win_h, win_w, d)
            x_merged = x_merged.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h_pad, w_pad, d)
            x = x_merged[:, pad_h:-pad_h, pad_w:-pad_w, :]

        else:
            coords = self._build_coords(h, w, h, w, 0, 0, win_h, win_w, self.nw_h, self.nw_w, b, x.device)

            x_win = x.view(b, self.nw_h, win_h, self.nw_w, win_w, d)
            x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, d)

            attn_win = self.attn(x_win, coords=coords)

            x = attn_win.view(b, self.nw_h, self.nw_w, win_h, win_w, d)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, d)

        return shortcut + x + self.mlp(self.norm2(x))


class CondTransformerBlock(nn.Module):
    """Transformer block with adaptive normalization layers conditioned on an external context vector."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        activation: Type[nn.Module] = nn.GELU,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        rmsnorm: bool = True,
    ):
        """Initialize the conditioned transformer block.

        Args:
            dim: Token embedding size.
            n_heads: Number of attention heads.
            activation: Activation class used in the MLP.
            mlp_dim: Hidden size of the feed-forward network.
            dropout: Dropout probability applied in the MLP.
            rmsnorm: If True, use RMS normalization; otherwise use adaptive layer normalization.
        """
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.num_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5

        # Projections for multi-head self-attention
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        # Time-dependent adaptive normalization layers
        if rmsnorm:
            self.norm1 = AdaRMSNorm(dim)
            self.norm2 = AdaRMSNorm(dim)
        else:
            self.norm1 = AdaptiveLayerNorm(dim)
            self.norm2 = AdaptiveLayerNorm(dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the conditioned transformer block to a sequence of tokens.

        Args:
            x: Input token tensor of shape [B, N, C].
            cond: Conditioning vector of shape [B, C] used by the adaptive normalization layers.

        Returns:
            The updated token tensor with the same shape as the input.
        """
        B, N, C = x.shape

        # Attention block (pre-LN).
        res = x
        x = self.norm1(x, cond)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # Added projection to mix head outputs (missing in your code)

        x = x + res

        # MLP block (pre-LN).
        res = x
        x = self.norm2(x, cond)
        x = self.mlp(x)
        x = x + res

        return x


class SwinEncoder(nn.Module):
    """Encoder built from adaptive Swin blocks and a local cross-attention projection to extract compact latents."""

    def __init__(
        self,
        m: int,
        in_channels: int,
        latent_dim: int,
        num_blocks: int = 5,
        num_heads: int = 4,
        mlp_factor: int = 4,
        activation: Type[nn.Module] = nn.GELU,
        block_type: str = "swin",
        bc_mode: str = "circular",
        cross_attention: str = "vanilla",
        ndim: int = 2,
    ):
        """Initialize the encoder.

        Args:
            m: Side length of the target latent grid, producing $m \times m$ latent tokens.
            in_channels: Number of input channels in the physical field.
            latent_dim: Hidden dimension of the latent features.
            num_blocks: Number of Swin blocks stacked in the encoder.
            num_heads: Number of attention heads used in each block.
            mlp_factor: Expansion factor used in the MLP subnetwork.
            activation: Activation class used by the block MLPs.
            block_type: Type of Swin block to use: "swin" or "padded".
            bc_mode: Boundary padding mode for padded blocks.
            cross_attention: Type of cross-attention used for latent extraction.
            ndim: Number of extra spatial coordinates concatenated to the input features.
        """
        super().__init__()

        assert latent_dim % num_heads == 0, "Model dimension must be divisible by num_heads"
        assert block_type.lower() in [
            "swin",
            "padded",
        ], f"Block type {block_type} is not supported. Supported blocks: 'swin' or 'padded'"
        assert cross_attention.lower() in [
            "vanilla",
            "linear",
        ], f"Cross attention block {block_type} is not supported. Supported blocks: 'vanilla' or 'galerkin'"

        self.m = m  # Target grid size (M x M)
        self.latent_dim = latent_dim  # Global depth (D)

        # Lift physical channels into the latent space.
        self.lifting = nn.Linear(in_channels + ndim, latent_dim)
        # self.pe = MultiScaleFourierEmbedding(2, latent_dim, [10, 20])

        # Stack Swin blocks, ensuring the last one is not shifted so the latent extractor sees a clean grid.
        blocks = []
        for i in range(num_blocks):
            # Alternate shift if num_blocks is even, start with shifted windows
            if num_blocks % 2 == 0:
                shift = i % 2 == 0
            else:
                shift = i % 2 == 1

            if block_type == "swin":
                blocks.append(
                    AdaptiveSwinBlock(
                        dim=latent_dim,
                        grid_windows=(m, m),
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
                        grid_windows=(m, m),
                        num_heads=num_heads,
                        pad=shift,  # Pad on shifted layers
                        mlp_factor=mlp_factor,
                        activation=activation,
                        bc_mode=bc_mode,
                    )
                )

        self.swin_layers = nn.Sequential(*blocks)

        # Extract compact latent tokens with local cross-attention.
        self.latent_extractor = WindowCrossAttention(in_ch=latent_dim, dim=latent_dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, in_channels, H, W] (Physical tensor)
        Output:  [B, M*M, latent_dim] (Flat sequence of tokens for temporal model)
        """

        b, _, h, w = x.shape

        # Add coordinates
        grid_h = torch.linspace(-1.0, 1.0, h, device=x.device)
        grid_w = torch.linspace(-1.0, 1.0, w, device=x.device)
        coords_h, coords_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        global_coords = torch.stack([coords_h, coords_w], dim=0).unsqueeze(0).expand(b, -1, -1, -1)

        x = torch.cat([x, global_coords], dim=1)

        x = x.permute(0, 2, 3, 1)

        x = self.lifting(x)

        # Process through Swin Blocks
        x = self.swin_layers(x)

        # Extract the compact latent tokens using local cross-attention.
        latent = self.latent_extractor(x)

        return latent


class SwinNO(nn.Module):
    """A Swin-based neural operator that maps input fields to output fields while preserving spatial structure."""

    def __init__(
        self,
        winsize: int,
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
            winsize: Window size used by the Swin blocks.
            in_channels: Number of input channels in the physical field.
            out_channels: Number of output channels in the predicted field.
            latent_dim: Hidden dimension of the latent representation.
            num_blocks: Number of Swin blocks in the encoder stack.
            num_heads: Number of attention heads in each block.
            mlp_factor: Expansion factor for the MLP hidden size.
            activation: Activation class used by the MLP layers.
            block_type: Type of Swin block to use: "swin" or "padded".
            bc_mode: Boundary padding mode for padded blocks.
            ndim: Number of spatial coordinates used as positional features.
        """
        super().__init__()

        assert latent_dim % num_heads == 0, "Model dimension must be divisible by num_heads"
        assert block_type.lower() in [
            "swin",
            "padded",
        ], f"Block type {block_type} is not supported. Supported blocks: 'swin' or 'padded'"

        self.winsize = winsize  # Window size
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
                        grid_windows=(winsize, winsize),
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
                        grid_windows=(winsize, winsize),
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


class ResidualMLPBlock(nn.Module):
    """A residual MLP block composed of two linear layers with a non-linearity."""

    def __init__(self, dim, activation=nn.GELU):
        """Initialize the residual MLP block.

        Args:
            dim: Feature dimension of the input and output tensor.
            activation: Activation class used between the two linear layers.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """Apply the residual block to an input tensor.

        Args:
            x: Input tensor of shape [..., dim].

        Returns:
            The updated tensor with the same shape as the input.
        """
        return x + self.net(x)

class CrossAttentionUpsampler(nn.Module):
    """Full cross-attention for upscaling"""

    def __init__(
        self,
        latent_dim: int,
        shortcut_dim: int,
        num_heads: int = 4,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
    ):
        """
        Args:
            latent_dim:   Dimension of the latent grid features (D).
            shortcut_dim: Dimension of the high-res shortcut features (C).
            num_heads:    Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (latent_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(shortcut_dim, latent_dim)
        self.kv_proj = nn.Linear(latent_dim, latent_dim * 2)
        self.out_proj = nn.Linear(latent_dim, shortcut_dim)
        self.mlp = nn.Sequential(
            nn.Linear(shortcut_dim, mlp_factor * shortcut_dim),
            activation(),
            nn.Linear(mlp_factor * shortcut_dim, shortcut_dim),
        )
        self.q_norm = nn.RMSNorm(shortcut_dim)
        self.latent_norm = nn.RMSNorm(latent_dim)
        self.out_norm = nn.RMSNorm(shortcut_dim)

    def forward(self, z_grid: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_grid:   [B, M1, M2, latent_dim]   Low-res latent grid.
            shortcut: [B, H, W, shortcut_dim]   High-res encoder features.

        Returns:
            [B, H, W, shortcut_dim]             Upsampled features.
        """
        B, M1, M2, D = z_grid.shape
        _, H, W, C = shortcut.shape
        h = self.num_heads
        d_head = D // h

        # Normalize the shortcut features.
        shortcut = self.q_norm(shortcut).reshape
        z_grid = self.latent_norm(z_grid)

        # Project queries to latent dim
        q = self.q_proj(shortcut).view(B, H * W, h, d_head).permute(0, 2, 1, 3)

        k, v = self.kv_proj(z_grid).view(B, M1 * M2, 2, h, d_head).unbind(2)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # attn [B, heads, HW, M1M2]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # out [B, heads, HW, latent_dim]
        attn = torch.matmul(attn, v).permute(0, 2, 1, 3).view(B, H, W, D)
        # Prject to [B, HW, shortcut_dim]
        out = self.out_proj(attn)

        out = out + shortcut

        return out + self.mlp(self.out_norm(out))


class BlockCrossAttentionUpsampler(nn.Module):
    """Upsampler doing local cross-attention by grouping high-res queries into blocks."""

    def __init__(
        self,
        latent_dim: int,
        shortcut_dim: int,
        num_heads: int = 4,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
    ):
        """
        Args:
            latent_dim:   Dimension of the latent grid features (D).
            shortcut_dim: Dimension of the high-res shortcut features (C).
            num_heads:    Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (latent_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(shortcut_dim, latent_dim)
        self.kv_proj = nn.Linear(latent_dim, latent_dim * 2)
        self.out_proj = nn.Linear(latent_dim, shortcut_dim)
        self.mlp = nn.Sequential(
            nn.Linear(shortcut_dim, mlp_factor * shortcut_dim),
            activation(),
            nn.Linear(mlp_factor * shortcut_dim, shortcut_dim),
        )
        self.q_norm = nn.RMSNorm(shortcut_dim)
        self.latent_norm = nn.RMSNorm(latent_dim)
        self.out_norm = nn.RMSNorm(shortcut_dim)

    def forward(self, z_grid: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_grid:   [B, M1, M2, latent_dim]   Low-res latent grid.
            shortcut: [B, H, W, shortcut_dim]   High-res encoder features.

        Returns:
            [B, H, W, shortcut_dim]             Upsampled features.
        """
        B, M1, M2, D = z_grid.shape
        _, H, W, C = shortcut.shape
        h = self.num_heads
        d_head = D // h

        # Compute upsampling ratios per axis
        rh, rw = H // M1, W // M2
        n_queries = rh * rw  # Number of high-res pixels per latent token

        # Normalize the shortcut features.
        shortcut = self.q_norm(shortcut)
        z_grid = self.latent_norm(z_grid)

        # Reshape and permute shortcut into local blocks
        # [B, H, W, C] -> [B, M1, rh, M2, rw, C] -> [B, M1, M2, rh*rw, C]
        q_local = (
            shortcut.view(B, M1, rh, M2, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, M1, M2, n_queries, C)
        )

        # Project Queries, Keys, and Values
        # Q: [B, M1, M2, h, n_queries, d_head]
        q = self.q_proj(q_local).view(B, M1, M2, n_queries, h, d_head).permute(0, 1, 2, 4, 3, 5)

        # KV: [B, M1, M2, 1, 2*D] -> K, V: [B, M1, M2, h, 1, d_head]
        k, v = self.kv_proj(z_grid).view(B, M1, M2, 1, 2, h, d_head).unbind(dim=4)
        k = k.permute(0, 1, 2, 4, 3, 5)
        v = v.permute(0, 1, 2, 4, 3, 5)

        # Compute Attention over the block: [B, M1, M2, h, n_queries, 1]
        # Each of the n_queries attends to the single local latent token
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Context aggregation and output projection
        # out_local: [B, M1, M2, n_queries, shortcut_dim]
        out_local = torch.matmul(attn, v).permute(0, 1, 2, 4, 3, 5).reshape(B, M1, M2, n_queries, D)
        out_local = self.out_proj(out_local)

        # Reconstruct original high-res grid layout [B, H, W, shortcut_dim]
        out = out_local.view(B, M1, M2, rh, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        out = out + shortcut

        return out + self.mlp(self.out_norm(out))


class BidirectionalUpsampler(nn.Module):
    """Two-way cross-attention block inspired by the SAM mask decoder.

    It alternates between updating the high-res queries using the low-res latent,
    and updating the low-res latent using the high-res queries, using local blocks
    to avoid quadratic memory footprint.
    """

    def __init__(self, latent_dim: int, shortcut_dim: int, num_heads: int = 4):
        """
        Args:
            latent_dim:   Dimension of the latent grid features (D).
            shortcut_dim: Dimension of the high-res shortcut features (C).
            num_heads:    Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads

        # High-resolution queries attend to the low-resolution latent grid.
        self.q_proj = nn.Linear(shortcut_dim, latent_dim)
        self.kv_proj = nn.Linear(latent_dim, latent_dim * 2)
        self.out_proj_hr = nn.Linear(latent_dim, shortcut_dim)
        self.norm_hr = nn.LayerNorm(shortcut_dim)

        # The low-resolution latent grid attends back to the high-resolution queries.
        self.latent_q_proj = nn.Linear(latent_dim, latent_dim)
        self.shortcut_kv_proj = nn.Linear(shortcut_dim, latent_dim * 2)
        self.out_proj_lr = nn.Linear(latent_dim, latent_dim)
        self.norm_lr = nn.LayerNorm(latent_dim)

    def forward(
        self, z_grid: torch.Tensor, shortcut: torch.Tensor, rh: int, rw: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_grid:   [B, M1, M2, latent_dim]      Low-res latent grid.
            shortcut: [B, H, W, shortcut_dim]      High-res features.
            rh, rw:   Upsampling ratios (H // M1, W // M2).

        Returns:
            Updated (z_grid, shortcut) with identical shapes.
        """
        B, M1, M2, D = z_grid.shape
        _, H, W, C = shortcut.shape
        h = self.num_heads
        d_head = D // h
        n_queries = rh * rw

        # Structuring high-res features into local blocks: [B, M1, M2, n_queries, C]
        q_local = (
            shortcut.view(B, M1, rh, M2, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, M1, M2, n_queries, C)
        )

        # High-resolution features attend to the low-resolution latent grid.
        q = self.q_proj(q_local).view(B, M1, M2, n_queries, h, d_head).permute(0, 1, 2, 4, 3, 5)
        k, v = self.kv_proj(z_grid).view(B, M1, M2, 1, 2, h, d_head).unbind(dim=4)
        k, v = k.permute(0, 1, 2, 4, 3, 5), v.permute(0, 1, 2, 4, 3, 5)

        attn_hr = torch.matmul(q, k.transpose(-1, -2)) * (d_head**-0.5)
        attn_hr = F.softmax(attn_hr, dim=-1)

        out_hr = torch.matmul(attn_hr, v).permute(0, 1, 2, 4, 3, 5).view(B, M1, M2, n_queries, D)
        shortcut_update = self.out_proj_hr(out_hr)

        # Reshape back to [B, H, W, C] and apply residual connection
        shortcut_update = (
            shortcut_update.view(B, M1, M2, rh, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        )
        shortcut = self.norm_hr(shortcut + shortcut_update)

        # Re-extract q_local from the updated shortcut for step 2
        q_local = (
            shortcut.view(B, M1, rh, M2, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, M1, M2, n_queries, C)
        )

        # The low-resolution latent grid attends back to the high-resolution features.
        # Latent acts as Query: [B, M1, M2, h, 1, d_head]
        q_lr = self.latent_q_proj(z_grid).view(B, M1, M2, 1, h, d_head).permute(0, 1, 2, 4, 3, 5)

        # High-res block acts as Key/Value: [B, M1, M2, h, n_queries, d_head]
        k_lr, v_lr = self.shortcut_kv_proj(q_local).view(B, M1, M2, n_queries, 2, h, d_head).unbind(dim=4)
        k_lr, v_lr = k_lr.permute(0, 1, 2, 4, 3, 5), v_lr.permute(0, 1, 2, 4, 3, 5)

        attn_lr = torch.matmul(q_lr, k_lr.transpose(-1, -2)) * (d_head**-0.5)
        attn_lr = F.softmax(attn_lr, dim=-1)  # Attention over the n_queries dimension

        out_lr = torch.matmul(attn_lr, v_lr).permute(0, 1, 2, 4, 3, 5).view(B, M1, M2, D)
        z_grid = self.norm_lr(z_grid + self.out_proj_lr(out_lr))

        return z_grid, shortcut


class MultiscaleCAEncoder(nn.Module):
    """
    Multiscale Encoder combining Window Cross-Attention and Swin/Global Self-Attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        windows: List[Tuple[int, int]],
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
        windows : List[Tuple[int, int]]
            Window sizes for Swin stages.
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
        self.windows = windows
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

        for i, (res, win) in enumerate(zip(grid_resolutions, windows)):
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

            if self.windows[i] is not None:
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


# class MultiscaleCAEncoder(nn.Module):
#     """
#     Multiscale Encoder combining Window Cross-Attention and Swin/Global Self-Attention.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         windows: List[Tuple[int, int]],
#         model_dim: Union[int, List[int]] = 128,
#         grid_resolutions: List[Tuple[int, int]] = [(32, 32), (16, 16)],
#         num_heads: int = 4,
#         mlp_factor: int = 4,
#         rmsnorm: bool = True,
#         activation: Callable = nn.GELU,
#         ndim: int = 2,
#     ):
#         """
#         Args:
#             in_channels: Input feature dimension.
#             out_channels: Output feature dimension.
#             windows: Window sizes for Swin stages.
#             model_dim: Hidden dimensions per stage.
#             grid_resolutions: Target resolutions per stage.
#             num_heads: Number of attention heads.
#             mlp_factor: MLP dimension multiplier.
#             rmsnorm: Use RMSNorm if True, else LayerNorm.
#             activation: Activation function class.
#             ndim: Spatial dimensions.
#             pe: Positional encoding type.
#             pe_param: Hyperparameter for PE.
#         """
#         super().__init__()
#         self.grid_resolutions = grid_resolutions
#         self.windows = windows
#         norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm

#         if isinstance(model_dim, int):
#             self.model_dim = [model_dim] * (len(grid_resolutions) + 1)
#         else:
#             self.model_dim = model_dim

#         self.pos_encoder = PEBlock(in_features=2, out_features=self.model_dim[0], method="adaptive")
#         self.mixer = nn.Linear(2 * self.model_dim[0], self.model_dim[0])
#         self.input_proj = nn.Linear(in_channels, self.model_dim[0])
#         self.input_norm = norm_layer(self.model_dim[0])

#         self.stages = nn.ModuleList()
#         num_stages = len(grid_resolutions)

#         for i, (res, win) in enumerate(zip(grid_resolutions, windows)):
#             is_last_stage = i == num_stages - 1
#             current_dim = self.model_dim[i]
#             next_dim = self.model_dim[i + 1]
#             print(f"building encoder block {i+1} using {win} windows. Target resolution {res}")
#             print(f"Current model dim: {current_dim}, next :level {next_dim}")

#             cross_attn = WindowCrossAttention(in_ch=current_dim, dim=next_dim, num_heads=num_heads, m=1)
#             cross_norm = norm_layer(next_dim)

#             if win is not None:
#                 self_attn_1 = AdaptiveSwinBlock(
#                     dim=next_dim,
#                     grid_windows=win,
#                     num_heads=num_heads,
#                     activation=activation,
#                     mlp_factor=mlp_factor,
#                     shift=False,
#                 )
#                 self_attn_2 = AdaptiveSwinBlock(
#                     dim=next_dim,
#                     grid_windows=win,
#                     num_heads=num_heads,
#                     activation=activation,
#                     mlp_factor=mlp_factor,
#                     shift=True,
#                 )
#                 self_attn_blocks = nn.ModuleList([self_attn_1, self_attn_2])
#             else:
#                 self_attn_blocks = TransformerBlock(
#                     dim=next_dim,
#                     n_heads=num_heads,
#                     activation=activation,
#                     rmsnorm=rmsnorm,
#                     mlp_dim=mlp_factor * next_dim,
#                 )

#             self.stages.append(
#                 nn.ModuleDict(
#                     {
#                         "cross_attn": cross_attn,
#                         "cross_norm": cross_norm,
#                         "self_attn": self_attn_blocks,
#                     }
#                 )
#             )

#         self.output_proj = nn.Linear(self.model_dim[-1], out_channels)

#     def _get_coords(self, H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#         y = torch.linspace(-1, 1, H, device=device)
#         x = torch.linspace(-1, 1, W, device=device)
#         grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
#         coords = torch.stack([grid_y, grid_x], dim=-1).view(1, H, W, 2)
#         cell_size = torch.tensor([2.0 / H, 2.0 / W], device=device).view(1, 1, 1, 2)
#         return coords, cell_size

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
#         """Encode an input field through the multiscale encoder and return latent shortcuts.

#         Args:
#             x: Input tensor of shape [B, C, H_in, W_in].

#         Returns:
#             A tuple containing the final latent tensor of shape [B, H_last * W_last, out_channels] and the list of stage shortcuts.
#         """
#         B, C, H_in, W_in = x.shape
#         device = x.device

#         in_coords, _ = self._get_coords(H_in, W_in, device)
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = self.input_proj(x)
#         x = self.input_norm(x)

#         pe = self.pos_encoder(in_coords).expand(B, -1, -1, -1)
#         x = self.mixer(torch.cat([x, pe], dim=-1))

#         shortcuts = [x.view(B, H_in, W_in, -1)]
#         kv_context = x  # [B, H*W, D]
#         coords_win = in_coords
#         H_curr, W_curr = H_in, W_in

#         for i, stage in enumerate(self.stages):
#             is_last_stage = i == len(self.stages) - 1
#             H_tgt, W_tgt = self.grid_resolutions[i]
#             D_curr = self.model_dim[i]
#             D_next = self.model_dim[i + 1]

#             # Compute window downsampling sizes
#             win_h, win_w = H_curr // H_tgt, W_curr // W_tgt
#             assert H_curr % H_tgt == 0 and W_curr % W_tgt == 0, "Resolution mismatch"
#             # Reshape the context and coordinates into window batches.
#             kv_context = kv_context.view(B, H_tgt, win_h, W_tgt, win_w, D_curr)
#             kv_context = kv_context.permute(0, 1, 3, 2, 4, 5).contiguous()
#             kv_context = kv_context.view(B * H_tgt * W_tgt, win_h * win_w, D_curr)

#             coords_win = coords_win.view(1, H_tgt, win_h, W_tgt, win_w, 2).expand(B, -1, -1, -1, -1, -1).contiguous()
#             coords_win = coords_win.permute(0, 1, 3, 2, 4, 5).contiguous()
#             coords_win = coords_win.view(B * H_tgt * W_tgt, win_h * win_w, 2)

#             # Compress the window tokens into a compact latent representation.
#             x_latent = stage["cross_attn"](kv_context, coords_win)  # [B * H_tgt * W_tgt, 1, D_next]
#             x_latent = stage["cross_norm"](x_latent)

#             # Reconstruct the target spatial grid with shape [B, H_tgt, W_tgt, D_next].
#             x_latent = x_latent.view(B, H_tgt, W_tgt, D_next)

#             # Apply the self-attention refinement step.
#             if self.windows[i] is not None:
#                 # Swin expects [B, H, W, D]
#                 x_latent = stage["self_attn"][0](x_latent)
#                 x_latent = stage["self_attn"][1](x_latent)
#                 kv_context = x_latent
#             else:
#                 # Global transformer expects sequential format [B, N, D]
#                 x_latent = x_latent.view(B, H_tgt * W_tgt, D_next)
#                 x_latent = stage["self_attn"](x_latent)
#                 kv_context = x_latent.view(B, H_tgt, W_tgt, D_next)

#             shortcuts.append(kv_context)
#             # Update dimensions for the next stage
#             H_curr, W_curr = H_tgt, W_tgt
#             coords_win, _ = self._get_coords(H_curr, W_curr, device)

#         # Prepare the output tensor for the final projection.
#         out = self.output_proj(kv_context)  # [B, H_last * W_last, out_channels]
#         return out, shortcuts


class MultiscaleCADecoder(nn.Module):
    """
    Multiscale Decoder handling progressive upsampling via cross-attention and self-attention refinement.
    """

    def __init__(
        self,
        latent_dim: List[int],
        out_channels: int,
        grid_windows: List[Union[Tuple[int, int], None]],
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
            grid_windows: Window configuration per stage (Tuple for Swin, None for Global).
            n_heads: Number of attention heads per stage.
            mlp_ratio: Multiplier for MLP hidden dimension.
            rmsnorm: Use RMSNorm if True, else LayerNorm.
            activation: Activation function class.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_windows = grid_windows
        num_stages = len(grid_windows)

        if isinstance(repeats, int):
            self.repeats = [repeats] * num_stages
        else:
            if len(repeats) != num_stages:
                raise ValueError(f"Length of repeats ({len(repeats)}) must match num_stages ({num_stages})")
            self.repeats = repeats

        self.stages = nn.ModuleList()

        for i, windows in enumerate(grid_windows):
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
            windows = self.grid_windows[stage_idx]

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
        enc_windows: List[Tuple[int, int]],
        dec_windows: List[Union[Tuple[int, int], None]],
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
            enc_windows:       Window sizes for the encoder stages.
            dec_windows:       Window configurations for the decoder stages.
            model_dim:         Hidden dimensions (int or list mapping resolutions).
            grid_resolutions:  Target resolutions for downsampling stages.
            num_heads:         Number of attention heads.
            mlp_factor:        Multiplier for MLP hidden dimension.
            rmsnorm:           Use RMSNorm if True, else LayerNorm.
            activation:        Activation function class.
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
            windows=enc_windows,
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
            grid_windows=dec_windows,
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
