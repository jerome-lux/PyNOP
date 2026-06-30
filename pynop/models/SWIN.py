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

    def __init__(self, dim: int, num_heads: int):
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
        b_win, n, c = x.shape
        h = self.num_heads

        qkv = self.qkv(x).reshape(b_win, n, 3, h, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*nW, H, N, d_head]

        # RoPE appliqu� sur q et k uniquement
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
    """
    Local cross-attention using learnable queries per window.
    """

    def __init__(self, in_ch: int, dim: int, num_heads: int, m: int = 1):
        """
        Args:
            in_ch: Input dimension for key/value.
            dim: Output dimension for queries/keys/values.
            num_heads: Number of attention heads.
            m: Grid size of the learnable queries (m*m queries).
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.m = m
        self.scale = self.d_head**-0.5

        # Learnable queries defined in target dim
        self.latent_queries = nn.Parameter(torch.randn(1, m * m, dim) * 0.02)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(in_ch, dim, bias=True)
        self.v_proj = nn.Linear(in_ch, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)

        # self.rope = AdaptiveRoPE2D(self.d_head) # Assumed defined elsewhere

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape [B_win, N_in, in_ch].
            coords: Absolute coordinates of shape [B_win, N_in, 2].
        Returns:
            Aggregated tokens of shape [B_win, m*m, dim].
        """
        b_win, n, _ = x.shape
        h = self.num_heads

        # Project queries
        q = self.latent_queries.expand(b_win, -1, -1)
        q = self.q_proj(q).reshape(b_win, self.m * self.m, h, self.d_head).transpose(1, 2)

        # Project keys and values
        k = self.k_proj(x).reshape(b_win, n, h, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(b_win, n, h, self.d_head).transpose(1, 2)

        # Apply position encoding on keys (placeholder for your RoPE logic)
        # k = self.rope.rotate(k, coords)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b_win, self.m * self.m, self.dim)
        return self.out_proj(out)


class AdaptiveSwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_windows: tuple[int, int],
        num_heads: int,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
        shift: bool = False,
    ):
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
        """
        Coordonnees absolues de chaque token apres partitionnement en fenetres.
        Retourne [B*nW, win_h*win_w, 2] en espace [-1, 1].
        Le pas inter-pixel est 2/H et 2/W (coordonnees image globales).
        """
        # Grille globale AVANT shift (positions des tokens dans l'image)
        gy = (torch.arange(h, device=device).float() / h) * 2 - 1  # [-1, 1)
        gx = (torch.arange(w, device=device).float() / w) * 2 - 1
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [H, W]
        coords_img = torch.stack([cy, cx], dim=-1)  # [H, W, 2]

        # Applique le m�me roll que sur x
        if shift_h > 0 or shift_w > 0:
            coords_img = torch.roll(coords_img, shifts=(-shift_h, -shift_w), dims=(0, 1))

        # Partitionne exactement comme x_win
        coords_win = coords_img.view(self.nw_h, win_h, self.nw_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(self.nw_h * self.nw_w, win_h, win_w, 2)

        # Repete sur le batch
        coords_win = coords_win.unsqueeze(0).expand(b, -1, -1, -1, -1)  # [B, nW, win_h, win_w, 2]
        return coords_win.reshape(b * self.nw_h * self.nw_w, win_h, win_w, 2)

    def forward(self, x: torch.Tensor, use_mask: bool = True) -> torch.Tensor:
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

        # Coordonnees absolues [B * num_win, win_h, win_w, 2]
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
        """
        Coordonnees absolues en espace [-1, 1] pour les tokens paddes.
        Les tokens hors grille originale re�oivent des coords extrapol�es �
        coherent avec la s�mantique de bc_mode (circulaire, r�fl�chi, etc.).
        Retourne [B * current_nw_h * current_nw_w, win_h * win_w, 2].
        """
        # Pas uniforme bas� sur la grille originale
        step_y = 2.0 / h
        step_x = 2.0 / w

        # Coordonn�es du tenseur padd� : tokens originaux dans [-1, 1),
        # tokens de padding extrapol�s naturellement hors de cet intervalle
        gy = (torch.arange(h_pad, device=device).float() - pad_h) * step_y - 1.0 + step_y / 2
        gx = (torch.arange(w_pad, device=device).float() - pad_w) * step_x - 1.0 + step_x / 2

        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [H_pad, W_pad]
        coords_img = torch.stack([cy, cx], dim=-1)  # [H_pad, W_pad, 2]

        # Partitionne comme x_win
        coords_win = coords_img.view(current_nw_h, win_h, current_nw_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(current_nw_h * current_nw_w, win_h, win_w, 2)

        return coords_win.unsqueeze(0).expand(b, -1, -1, -1).reshape(b * current_nw_h * current_nw_w, win_h, win_w, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """Transformer block with adaptive normalization layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        activation: Type[nn.Module] = nn.GELU,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        rmsnorm: bool = True,
    ):
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
        """
        x:     [B, N, C] tokens
        t_emb: [B, C] global conditionning tensor
        """
        B, N, C = x.shape

        # 1. Attention chunk (Pre-LN)
        res = x
        x = self.norm1(x, cond)

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # Added projection to mix head outputs (missing in your code)

        x = x + res

        # 2. MLP chunk (Pre-LN)
        res = x
        x = self.norm2(x, cond)
        x = self.mlp(x)
        x = x + res

        return x


class SwinEncoder(nn.Module):
    """
    Encoder combining Adaptive Swin Transformer blocks with a
    Local Galerkin Cross-Attention projection to extract resolution-invariant latents.
    """

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

        # 1. Input Lifting: Map physical channels to latent space depth
        self.lifting = nn.Linear(in_channels + ndim, latent_dim)
        # self.pe = MultiScaleFourierEmbedding(2, latent_dim, [10, 20])

        # 2. Stack of Swin Blocks (Alternating between Standard and Shifted)
        # Note: According to your constraint, we ensure the last block is NOT shifted
        # so that the Local Galerkin Extractor receives a clean M x M topology.
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

        # 3. Final Latent Extraction via Local Cross-Attention
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

        # pe = self.pe(global_coords.permute(0, 2, 3, 1).view(b, h * w, 2)).view(b, h, w, -1)
        # x = torch.cat([x, pe], dim=-1)

        x = self.lifting(x)

        # Process through Swin Blocks
        x = self.swin_layers(x)

        # Extract the compact M*M latent tokens using Local Cross-Attention
        latent = self.latent_extractor(x)

        return latent


class SwinNO(nn.Module):

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
        ndim: int = 2,
    ):
        super().__init__()

        assert latent_dim % num_heads == 0, "Model dimension must be divisible by num_heads"
        assert block_type.lower() in [
            "swin",
            "padded",
        ], f"Block type {block_type} is not supported. Supported blocks: 'swin' or 'padded'"

        self.winsize = winsize  # Window size
        self.latent_dim = latent_dim  # Global depth (D)

        # Input Lifting
        self.lifting = nn.Linear(in_channels, latent_dim)
        self.pos_encoder = nn.Linear(ndim, latent_dim)
        self.pre_norm = nn.RMSNorm(latent_dim)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, in_channels, H, W] (Physical tensor)
        Output:  [B, M*M, latent_dim] (Flat sequence of tokens for temporal model)
        """

        b, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1)

        # Add coordinates
        grid_h = torch.linspace(-1.0, 1.0, h, device=x.device)
        grid_w = torch.linspace(-1.0, 1.0, w, device=x.device)
        coords_h, coords_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
        global_coords = torch.stack([coords_h, coords_w], dim=-1).unsqueeze(0)
        pe = self.pos_encoder(global_coords).expand(b, -1, -1, -1)

        x = self.lifting(x)
        x = self.pre_norm(x)
        x = x + pe

        # Process through Swin Blocks
        x = self.swin_layers(x)

        x = self.out_proj(x)
        x = x.permute(0, 3, 1, 2)

        return x


class TimeEmbedding(nn.Module):
    """Maps a scalar time t to a continuous vector embedding using Fourier features."""

    def __init__(self, dim: int, scale: float = 10.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("frequencies", torch.randn(dim // 2) * scale)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [B, 1]
        phases = t @ self.frequencies.unsqueeze(0)  # [B, dim // 2]
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)


class ContinuousTimePropagator(nn.Module):
    """Propagates the latent initial state g_0 to g(t) using time modulation."""

    def __init__(
        self,
        latent_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        activation=nn.GELU,
        mlp_factor: int = 4,
        rmsnorm: bool = True,
    ):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Each block uses self-attention on the M*M tokens, conditioned by the time embedding
        self.layers = nn.ModuleList(
            [
                CondTransformerBlock(
                    dim=latent_dim,
                    n_heads=num_heads,
                    mlp_dim=latent_dim * mlp_factor,
                    activation=activation,
                    rmsnorm=rmsnorm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, z_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        g_0: [B, M*M, latent_dim] (Latent at t=0)
        t:   [B, 1] (Target time, continuous)
        """
        # 1. Compute global time embedding context
        t_emb = self.time_embedding(t)  # [B, latent_dim]

        # 2. Propagate latent through the modulated transformer layers
        z_t = z_0
        for layer in self.layers:
            # En pratique, vous passez t_emb à vos blocs pour moduler les AdaNorm :
            # g_t = layer(g_t, t_emb)
            z_t = layer(z_t, t_emb)  # Version standard simplifiée pour le script

        return z_t  # [B, M*M, latent_dim]


class LocalTextureEstimatorDecoder(nn.Module):
    """Implicit Neural Representation Decoder using Local Texture Estimation (LTE).

    Dynamically estimates local Fourier frequencies and amplitudes from latent codes.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        grid_size: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # LTE Estimator functions: maps latent feature to frequencies and amplitudes
        # For 2D coordinates, we estimate frequencies for both X and Y axes
        self.freq_estimator = nn.Linear(latent_dim, hidden_dim * 2)
        self.amp_estimator = nn.Linear(latent_dim, hidden_dim)

        # Main MLP processing the modulated Fourier features
        self.im_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_channels),
        )

    def forward(self, z: torch.Tensor, out_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, M*M, latent_dim] Latent grid tokens.
            out_coords: [1, H_out, W_out, 2] Target evaluation coordinates in [-1, 1].
        Returns:
            [B, out_channels, H_out, W_out] Continuous field prediction.
        """
        b, _, d = z.shape
        _, h, w, _ = out_coords.shape
        out_coords = out_coords.expand(b, -1, -1, -1)

        # 1. Reshape sequence to spatial grid for neighborhood sampling
        z_grid = z.view(b, self.grid_size, self.grid_size, d).permute(0, 3, 1, 2)  # [B, D, M, M]

        # 2. Locate the 4 nearest cells (LIIF/LTE grid-unfolding technique)
        # Transform coords to pixel index space [0, M]
        grid_coords = (out_coords + 1.0) * (self.grid_size / 2.0) - 0.5
        tl_coords = torch.floor(grid_coords)

        offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        outputs = []
        areas = []

        # 3. Predict field for each of the 4 local neighbor cells
        for dy, dx in offsets:
            # Absolute cell index
            n_coords = tl_coords + torch.tensor([dx, dy], device=out_coords.device)
            n_coords = torch.clamp(n_coords, 0, self.grid_size - 1)

            # Center of the current neighbor cell in [-1, 1] space
            n_centers = (n_coords + 0.5) / (self.grid_size / 2.0) - 1.0

            # Relative coordinate vector from cell center to target coordinate
            rel_coords = out_coords - n_centers  # [B, H, W, 2]

            # Sample latent code at this neighbor cell location
            n_coords_norm = n_centers  # Coordinate used for nearest sampling
            z_sample = F.grid_sample(z_grid, n_coords_norm, mode="nearest", padding_mode="border", align_corners=False)
            z_sample = z_sample.permute(0, 2, 3, 1)  # [B, H, W, D]

            # 4. LTE Core: Estimate dynamic frequencies and amplitudes from z
            # Frequencies split into X and Y components: [B, H, W, hidden_dim]
            freqs = self.freq_estimator(z_sample).view(b, h, w, self.hidden_dim, 2)
            amps = self.amp_estimator(z_sample)  # [B, H, W, hidden_dim]

            # Mathematical formulation: Amp * cos(pi * (Freq_x * rel_x + Freq_y * rel_y))
            # rel_coords project onto estimated local coordinate frequency grid
            rel_projected = (rel_coords.unsqueeze(-2) * freqs).sum(dim=-1)  # [B, H, W, hidden_dim]

            # Dynamic Fourier feature expansion
            lte_features = amps * torch.cos(math.pi * rel_projected)

            # Pass through localized MLP projection
            cell_pred = self.im_net(lte_features)
            outputs.append(cell_pred)

            # 5. Compute bilinear area weight for late fusion (barycentric vote)
            # Area is inversely proportional to distance to the opposite corner
            area = torch.prod(1.0 - torch.abs(grid_coords - n_coords), dim=-1, keepdim=True)
            areas.append(area + 1e-6)

        # 6. Complementary area weighted ensemble (Ensemble vote from 4 neighbors)
        total_weight = sum(areas)
        final_output = sum(out * (weight / total_weight) for out, weight in zip(outputs, areas))

        return final_output.permute(0, 3, 1, 2)


class ImplicitDecoder(nn.Module):
    """Continuous Implicit Representation Decoder using grid features."""

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        grid_size: int,
        activation=nn.GELU,
        ndim: int = 2,
        rmsnorm: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        # self.pe = MultiScaleFourierEmbedding(ndim, latent_dim, scales=[10, 20])
        # self.pe = IntegratedPositionalEncoding(ndim, latent_dim)

        self.norm = nn.RMSNorm(latent_dim) if rmsnorm else nn.LayerNorm(latent_dim)

        self.im_net = nn.Sequential(
            nn.Linear(latent_dim + 2 * ndim, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            activation(),
            nn.Linear(latent_dim, out_channels),
        )

    def forward(self, z_t: torch.Tensor, out_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: [B, M*M, D] Latent grid sequence.
            out_coords: [1, H_out, W_out, 2] Target coordinates scaled in [-1, 1].
        Returns:
            [B, out_channels, H_out, W_out] Reconstructed image.
        """
        b, _, d = z_t.shape
        out_coords = out_coords.expand(b, -1, -1, -1)
        _, h_out, w_out, _ = out_coords.shape

        # 1. Feature extraction via bilinear sampling
        z_grid = z_t.view(b, self.grid_size, self.grid_size, self.latent_dim).permute(0, 3, 1, 2)

        # out_coords expected in [-1, 1] for grid_sample
        local_features = F.grid_sample(z_grid, out_coords, mode="bilinear", padding_mode="border", align_corners=False)
        local_features = local_features.permute(0, 2, 3, 1)
        local_features = self.norm(local_features)

        # 2. Compute relative coordinates to the nearest latent cell center
        # Scale coords from [-1, 1] to [0, M] to easily find cell centers
        grid_coords = (out_coords + 1.0) * (self.grid_size / 2.0)

        # Find nearest center by rounding, then shift back by 0.5 for pixel center
        nearest_centers = torch.floor(grid_coords) + 0.5

        # Relative distance within the cell, normalized back to [-1, 1] range
        rel_coords = (grid_coords - nearest_centers) * 2.0
        rel_coords = rel_coords.view(b, h_out, w_out, -1)

        # 3. Apply Positional Encoding on local relative coordinates
        # cell_size = torch.tensor([2.0 / h_out, 2.0 / w_out], device=z_t.device)
        # cell_size = cell_size.view(1, 1, 2).view(1, 1, 2)
        # global_pe = self.pe(out_coords.view(b, -1, 2), cell_size).view(b, h_out, w_out, d)

        # 4. Forward MLP
        inr_input = torch.cat([local_features, rel_coords, out_coords], dim=-1)
        output = self.im_net(inr_input)

        return output.permute(0, 3, 1, 2)


class LIIFDecoder(nn.Module):
    """Local Implicit Image Function (LIIF) Decoder with local ensemble."""

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        grid_size: int,
        activation=nn.GELU,
        rmsnorm: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size  # Latent grid size M (e.g., 16)
        self.latent_dim = latent_dim

        # Coordinate dimension is 2 (dx, dy)
        in_features = latent_dim + 2

        if rmsnorm:
            self.norm = nn.RMSNorm(latent_dim)
        else:
            self.norm = nn.LayerNorm(latent_dim)

        # MLP evaluating the local implicit function
        self.im_net = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim // 2),
            activation(),
            nn.Linear(latent_dim // 2, out_channels),
        )

    def forward(self, z_t: torch.Tensor, out_coords: torch.Tensor) -> torch.Tensor:
        """
        z_t:          [B, M*M, latent_dim] (Latent grid sequence)
        out_coords:   [B1 H_out, W_out, 2] (Target continuous coords in [-1, 1] or [0, 1])
        """
        b, h_out, w_out, _ = out_coords.shape
        M = self.grid_size

        # Reshape latent to spatial grid: [B, D, M, M]
        z_grid = z_t.view(b, M, M, self.latent_dim).permute(0, 3, 1, 2)

        # 1. Generate cell centers coordinates of the MxM grid
        # Generating coordinates in range [-1, 1] to match F.grid_sample convention
        rx = torch.linspace(-1 + 1 / M, 1 - 1 / M, M, device=out_coords.device)
        ry = torch.linspace(-1 + 1 / M, 1 - 1 / M, M, device=out_coords.device)
        grid_y, grid_x = torch.meshgrid(ry, rx, indexing="ij")
        # [M, M, 2] -> Centers of each latent token
        latent_centers = torch.stack([grid_x, grid_y], dim=-1)

        # Map out_coords to [-1, 1] if they are in [0, 1]
        if out_coords.min() >= 0.0 and out_coords.max() <= 1.0:
            coords = out_coords * 2.0 - 1.0
        else:
            coords = out_coords

        # 2. Find the 4 nearest neighbors on the MxM grid for each output coordinate
        # Scale coordinates to grid index space [0, M-1]
        grid_coords = (coords + 1.0) * (M / 2.0) - 0.5

        # Top-left neighbor indices
        x0 = torch.floor(grid_coords[..., 0]).long().clamp(0, M - 2)
        y0 = torch.floor(grid_coords[..., 1]).long().clamp(0, M - 2)

        # 4 Neighbor combinations: (top-left, top-right, bottom-left, bottom-right)
        neighbor_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]

        preds = []
        areas = []

        # 3. Local Ensemble: Loop over the 4 nearest neighbors
        for dx_idx, dy_idx in neighbor_offsets:
            # Query indices for this specific neighbor
            xi = x0 + dx_idx
            yi = y0 + dy_idx

            # Extract corresponding latent centers: [B, H_out, W_out, 2]
            # (Gathering from the grid centers for relative distance computation)
            centers_i = latent_centers[yi, xi]

            # Compute relative coordinate (dx, dy) scaled by grid cell size
            rel_coords = (coords - centers_i) * M  # [B, H_out, W_out, 2]

            # Extract latent features using grid_sample with nearest mode for the specific neighbor grid
            # To be efficient, we prepare custom grid sampling for this specific neighbor quadrant
            # Transform indices back to [-1, 1] coordinate space for grid_sample
            norm_xi = (xi.float() + 0.5) / (M / 2.0) - 1.0
            norm_yi = (yi.float() + 0.5) / (M / 2.0) - 1.0
            grid_i = torch.stack([norm_xi, norm_yi], dim=-1)  # [B, H_out, W_out, 2]

            # Sample the exact token feature for this neighbor
            feat_i = F.grid_sample(z_grid, grid_i, mode="nearest", padding_mode="border", align_corners=False)
            feat_i = feat_i.permute(0, 2, 3, 1)  # [B, H_out, W_out, D]
            feat_i = self.norm(feat_i)

            # Concatenate token feature and local relative coordinate
            inr_input = torch.cat([feat_i, rel_coords], dim=-1)  # [B, H_out, W_out, D + 2]

            # Forward through MLP
            pred_i = self.im_net(inr_input.view(-1, self.latent_dim + 2))
            preds.append(pred_i.view(b, h_out, w_out, -1))

            # Compute areas for bilinear interpolation weight (distance to the opposite neighbor)
            area_i = torch.abs(grid_coords[..., 0] - xi.float()) * torch.abs(grid_coords[..., 1] - yi.float())
            areas.append(area_i)

        # 4. Complement areas for interpolation weights
        # The weight is proportional to the area of the rectangle opposite to the neighbor
        total_area = sum(areas)
        weights = [areas[3], areas[2], areas[1], areas[0]]  # Swap corners for bilinear weight

        # 5. Weighted ensemble sum
        output = sum(p * (w.unsqueeze(-1) / (total_area.unsqueeze(-1) + 1e-6)) for p, w in zip(preds, weights))

        # [B, out_channels, H_out, W_out]
        return output.permute(0, 3, 1, 2)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, activation=nn.GELU):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class LIIFDecoder(nn.Module):
    """LIIF-inspired implicit decoder with optional shortcut from the encoder.

    For each query coordinate the decoder:
      1. Locates the 4 surrounding latent cells and performs bilinear interpolation.
      2. Encodes the sub-cell displacement with an Integrated Positional Encoding.
      3. Concatenates [z_interp | PE | bilinear_weights | optional_shortcut] and
         passes the result through a residual MLP.

    Convention: out_coords[..., 0] = y (row / H axis),
                out_coords[..., 1] = x (col / W axis).
    z is stored row-major: z_grid[b, row, col] = z_grid[b, y, x].
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int,
        grid_size: int,
        shortcut_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_blocks: int = 4,
        activation: Type[nn.Module] = nn.GELU,
        pe_max_freq: int = 256,
    ):
        """
        Args:
            latent_dim:   Dimension of each latent token (D).
            out_channels: Output feature channels.
            grid_size:    Side length M of the square latent grid.
            shortcut_dim: Channel dim of the optional encoder shortcut tensor.
                          Pass None (default) to disable.
            hidden_dim:   Hidden width of the residual MLP. Defaults to latent_dim.
            n_blocks:     Number of ResidualMLPBlocks.
            activation:   Activation *class* (not instance), e.g. nn.GELU.
            pe_max_freq:  Maximum frequency for IntegratedPositionalEncoding.

        """

        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        hidden_dim = hidden_dim if hidden_dim is not None else latent_dim
        _shortcut_dim = shortcut_dim if shortcut_dim is not None else 0

        # Input layout: z_interp [D] | PE [D] | bilinear_weights [4] | shortcut [shortcut_dim]
        in_dim = 2 * latent_dim + 4 + _shortcut_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Integrated Positional Encoding over 2D local displacement
        self.pe = IntegratedPositionalEncoding(2, latent_dim, max_freq=pe_max_freq)

        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, activation=activation) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_channels),
        )

        self._expects_shortcut = shortcut_dim is not None

    def forward(
        self,
        z: torch.Tensor,
        out_coords: torch.Tensor,
        shortcut: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:          [B, M*M, D]               Latent grid tokens (row-major).
            out_coords: [B, H, W, 2] or [1, H, W, 2]
                        Continuous query coordinates in [-1, 1].
                        Axis order: [..., 0] = y (row / H), [..., 1] = x (col / W).
            shortcut:   [B, H, W, shortcut_dim]   Optional high-res encoder features.

        Returns:
            [B, out_channels, H, W]  Reconstructed continuous field.
        """
        B, _, D = z.shape
        H, W = out_coords.shape[-3], out_coords.shape[-2]
        M = self.grid_size

        # Broadcast batch dim if a single coordinate grid is shared across the batch
        if out_coords.shape[0] == 1 and B > 1:
            out_coords = out_coords.expand(B, H, W, 2).contiguous()

        # ── 1. Reshape latent sequence to spatial grid ────────────────────────────
        # [B, M*M, D] → [B, M, M, D]  (z_grid[b, row, col] = z_grid[b, y, x])
        z_grid = z.view(B, M, M, D)

        # ── 2. Map query coordinates [-1, 1] → continuous pixel indices [0, M-1] ─
        # out_coords[..., 0] = y → row index
        # out_coords[..., 1] = x → col index
        gy = (out_coords[..., 0] + 1.0) * 0.5 * (M - 1)  # [B, H, W]
        gx = (out_coords[..., 1] + 1.0) * 0.5 * (M - 1)  # [B, H, W]

        # Top-left corner indices, clamped so iy1 = iy0+1 and ix1 = ix0+1 stay in [0, M-1]
        iy0 = torch.floor(gy).long().clamp(0, M - 2)  # row index, [B, H, W]
        ix0 = torch.floor(gx).long().clamp(0, M - 2)  # col index, [B, H, W]
        iy1 = iy0 + 1
        ix1 = ix0 + 1

        # Sub-cell fractional offsets in [0, 1]
        dy = (gy - iy0.float()).unsqueeze(-1)  # [B, H, W, 1]  vertical
        dx = (gx - ix0.float()).unsqueeze(-1)  # [B, H, W, 1]  horizontal

        # ── 3. Bilinear weights ───────────────────────────────────────────────────
        # w_{row_offset, col_offset}: weight for the corner at (iy0+row_off, ix0+col_off)
        w00 = (1.0 - dy) * (1.0 - dx)  # top-left
        w01 = (1.0 - dy) * dx  # top-right
        w10 = dy * (1.0 - dx)  # bottom-left
        w11 = dy * dx  # bottom-right
        # [B, H, W, 4] — order matches z gather below
        weights = torch.cat([w00, w01, w10, w11], dim=-1)

        # ── 4. Gather the 4 neighbouring latent tokens ────────────────────────────
        # z_grid is indexed [B, row, col] = [B, y, x]
        batch_idx = torch.arange(B, device=z.device).view(B, 1, 1)
        z00 = z_grid[batch_idx, iy0, ix0]  # [B, H, W, D]  top-left
        z01 = z_grid[batch_idx, iy0, ix1]  # top-right
        z10 = z_grid[batch_idx, iy1, ix0]  # bottom-left
        z11 = z_grid[batch_idx, iy1, ix1]  # bottom-right

        # Bilinear interpolation of latent codes
        z_interp = w00 * z00 + w01 * z01 + w10 * z10 + w11 * z11  # [B, H, W, D]

        # ── 5. Integrated Positional Encoding of sub-cell displacement ────────────
        # Displacement is centred: 0.0 at cell centre, ±0.5 at cell edges.
        # Convention mirrors out_coords: local_coords[..., 0] = dy, [..., 1] = dx
        local_coords = torch.cat([dy - 0.5, dx - 0.5], dim=-1)  # [B, H, W, 2]

        # Taille du pixel de sortie mesurée en unités de cellule latente
        # Si H = M, cell_size = 1.0 (le pixel occupe toute la cellule)
        # Si H = 2M, cell_size = 0.5 (le pixel occupe la moitié de la cellule)
        cell_size_y = M / H
        cell_size_x = M / W

        # Tensor de forme [B, H, W, 2] ou [1, 1, 1, 2] pour le broadcasting
        cell_size = torch.tensor([cell_size_y, cell_size_x], device=z.device, dtype=z.dtype).view(1, 1, 1, 2)

        # With Adaptive PE:
        # res_y = H / M
        # res_x = W / M

        # pe_out = self.pe(
        #     local_coords.view(B, H * W, 2),
        #     resolution=(res_y, res_x)
        # ).view(B, H, W, -1)

        pe_out = self.pe(local_coords.view(B, H * W, 2), cell_size=cell_size).view(B, H, W, -1)  # [B, H, W, latent_dim]

        parts = [z_interp, pe_out, weights]
        if shortcut is not None:
            parts.append(shortcut)

        x = torch.cat(parts, dim=-1)  # [B, H, W, in_dim]
        x = self.input_proj(x)  # [B, H, W, hidden_dim]
        x = self.blocks(x)
        x = self.head(x)  # [B, H, W, out_channels]

        return x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]


class LightLIIFDecoder(nn.Module):
    """Lightweight and efficient final upsampling local coordinate probing"""

    def __init__(self, dim: int, out_channels: int, activation: nn.Module = nn.GELU):
        """
        Args:
            dim: Internal model dimension (model_dim).
            out_channels: Final output channels (e.g., 3 for RGB).
        """
        super().__init__()
        # Small MLP applied pixel-wise
        self.fusion_mlp = nn.Sequential(nn.Linear(2 * dim, dim), activation(), nn.Linear(dim, out_channels))

    def forward(self, shortcut: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shortcut: [B, H_in, W_in, dim] - high-res shortcut.
            latent: [B, H_low, W_low, dim] - Current decoder latent state.
        """
        B, H_in, W_in, D = shortcut.shape
        _, H_low, W_low, _ = latent.shape
        # 1. Reshape low-res context to spatial grid for grid_sample: [B, D, H_low, W_low]
        low_res_grid = latent.permute(0, 3, 1, 2).contiguous()

        # 2. Upsample the low-res context features to target resolution via bilinear interpolation
        # This acts as a fast, geometric shortcut replacing cross-attention
        context_upsampled = F.interpolate(low_res_grid, size=(H_in, W_in), mode="bilinear", align_corners=False)
        context_upsampled = context_upsampled.permute(0, 2, 3, 1).contiguous().view(B, H_in, W_in, D)

        # 3. Concatenate high-res shortcut (fine details) and upsampled context (global semantics)
        x = torch.cat([shortcut, context_upsampled], dim=-1)

        # 4. Pixel-wise projection
        return self.fusion_mlp(x)


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

        # normalization
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

        # normalization
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

        # 1. Cross-attention: High-Res Queries attend to Low-Res Latent
        self.q_proj = nn.Linear(shortcut_dim, latent_dim)
        self.kv_proj = nn.Linear(latent_dim, latent_dim * 2)
        self.out_proj_hr = nn.Linear(latent_dim, shortcut_dim)
        self.norm_hr = nn.LayerNorm(shortcut_dim)

        # 2. Inverse Cross-attention: Low-Res Latent attends to High-Res Queries
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

        # --- STEP 1: HIGH-RES ATTENDS TO LOW-RES (Standard local cross-attention) ---
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

        # --- STEP 2: LOW-RES ATTENDS TO HIGH-RES (Inverse cross-attention) ---
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
        rmsnorm: bool = True,
        activation: Callable = nn.GELU,
        ndim: int = 2,
        pe: str = "adaptive",
        pe_param: int = 128,
    ):
        """
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            windows: Window sizes for Swin stages.
            model_dim: Hidden dimensions per stage.
            grid_resolutions: Target resolutions per stage.
            num_heads: Number of attention heads.
            mlp_factor: MLP dimension multiplier.
            rmsnorm: Use RMSNorm if True, else LayerNorm.
            activation: Activation function class.
            ndim: Spatial dimensions.
            pe: Positional encoding type.
            pe_param: Hyperparameter for PE.
        """
        super().__init__()
        self.grid_resolutions = grid_resolutions
        self.windows = windows
        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm

        if isinstance(model_dim, int):
            self.model_dim = [model_dim] * (len(grid_resolutions) + 1)
        else:
            self.model_dim = model_dim

        # Placeholder for PEBlock
        self.pos_encoder = PEBlock(in_features=2, out_features=self.model_dim[0], method=pe)
        self.mixer = nn.Linear(2 * self.model_dim[0], self.model_dim[0])
        self.input_proj = nn.Linear(in_channels, self.model_dim[0])
        self.input_norm = norm_layer(self.model_dim[0])

        self.stages = nn.ModuleList()
        num_stages = len(grid_resolutions)

        for i, (res, win) in enumerate(zip(grid_resolutions, windows)):
            is_last_stage = i == num_stages - 1
            current_dim = self.model_dim[i]
            next_dim = self.model_dim[i + 1]
            print(f"building encoder block {i+1} using {win} windows. Target resolution {res}")
            print(f"Current model dim: {current_dim}, next :level {next_dim}")

            cross_attn = WindowCrossAttention(in_ch=current_dim, dim=next_dim, num_heads=num_heads, m=1)
            cross_norm = norm_layer(next_dim)

            if win is not None:
                self_attn_1 = AdaptiveSwinBlock(
                    dim=next_dim,
                    grid_windows=win,
                    num_heads=num_heads,
                    activation=activation,
                    mlp_factor=mlp_factor,
                    shift=False,
                )
                self_attn_2 = AdaptiveSwinBlock(
                    dim=next_dim,
                    grid_windows=win,
                    num_heads=num_heads,
                    activation=activation,
                    mlp_factor=mlp_factor,
                    shift=True,
                )
                self_attn_blocks = nn.ModuleList([self_attn_1, self_attn_2])
            else:
                self_attn_blocks = TransformerBlock(
                    dim=next_dim,
                    n_heads=num_heads,
                    activation=activation,
                    rmsnorm=rmsnorm,
                    mlp_dim=mlp_factor * next_dim,
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
            is_last_stage = i == len(self.stages) - 1
            H_tgt, W_tgt = self.grid_resolutions[i]
            D_curr = self.model_dim[i]
            D_next = self.model_dim[i + 1]

            # Compute window downsampling sizes
            win_h, win_w = H_curr // H_tgt, W_curr // W_tgt
            assert H_curr % H_tgt == 0 and W_curr % W_tgt == 0, "Resolution mismatch"
            # Reshape context and coordinates to window batches
            kv_context = kv_context.view(B, H_tgt, win_h, W_tgt, win_w, D_curr)
            kv_context = kv_context.permute(0, 1, 3, 2, 4, 5).contiguous()
            kv_context = kv_context.view(B * H_tgt * W_tgt, win_h * win_w, D_curr)

            coords_win = coords_win.view(1, H_tgt, win_h, W_tgt, win_w, 2).expand(B, -1, -1, -1, -1, -1).contiguous()
            coords_win = coords_win.permute(0, 1, 3, 2, 4, 5).contiguous()
            coords_win = coords_win.view(B * H_tgt * W_tgt, win_h * win_w, 2)

            # Cross attention: compresses win_h*win_w to m*m tokens (m=1 here)
            x_latent = stage["cross_attn"](kv_context, coords_win)  # [B * H_tgt * W_tgt, 1, D_next]
            x_latent = stage["cross_norm"](x_latent)

            # Reconstruct the target spatial grid [B, H_tgt, W_tgt, D_next]
            x_latent = x_latent.view(B, H_tgt, W_tgt, D_next)

            # Self-Attention processing step
            if self.windows[i] is not None:
                # Swin expects [B, H, W, D]
                x_latent = stage["self_attn"][0](x_latent)
                x_latent = stage["self_attn"][1](x_latent)
                kv_context = x_latent
            else:
                # Global transformer expects sequential format [B, N, D]
                x_latent = x_latent.view(B, H_tgt * W_tgt, D_next)
                x_latent = stage["self_attn"](x_latent)
                kv_context = x_latent.view(B, H_tgt, W_tgt, D_next)

            shortcuts.append(kv_context)
            # Update dimensions for the next stage
            H_curr, W_curr = H_tgt, W_tgt
            coords_win, _ = self._get_coords(H_curr, W_curr, device)

        # Output format alignment
        out = self.output_proj(kv_context)  # [B, H_last * W_last, out_channels]
        return out, shortcuts


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

        self.stages = nn.ModuleList()

        for i, windows in enumerate(grid_windows):
            print(f"building decoder block {i+1} using {windows} windows")
            print(f"Current model dim: {self.latent_dim[i]}, next :level {self.latent_dim[i + 1]}")
            current_dim = self.latent_dim[i]
            next_dim = self.latent_dim[i + 1]

            upscaler = BlockCrossAttentionUpsampler(
                latent_dim=current_dim,
                shortcut_dim=next_dim,
                num_heads=n_heads,
                activation=activation,
                mlp_factor=mlp_ratio,
            )

            if windows is not None:
                self_attn_blocks = nn.ModuleList(
                    [
                        AdaptiveSwinBlock(
                            dim=next_dim,
                            grid_windows=windows,
                            num_heads=n_heads,
                            activation=activation,
                            mlp_factor=mlp_ratio,
                            shift=False,
                        ),
                        AdaptiveSwinBlock(
                            dim=next_dim,
                            grid_windows=windows,
                            num_heads=n_heads,
                            activation=activation,
                            mlp_factor=mlp_ratio,
                            shift=True,
                        ),
                    ]
                )
            else:
                self_attn_blocks = TransformerBlock(
                    dim=next_dim,
                    n_heads=n_heads,
                    activation=activation,
                    rmsnorm=rmsnorm,
                    mlp_dim=mlp_ratio * next_dim,
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
            B, H, W, _ = shortcut.shape
            windows = self.grid_windows[stage_idx]

            # 1. Upsample low-res z using the high-res shortcut spatial framework
            z = self.stages[stage_idx]["upscaler"](z, shortcut)  # Output: [B, H, W, D_next]

            # 2. Refinement via Self-Attention
            if windows is not None:
                # Swin blocks process [B, H, W, D] directly
                z = self.stages[stage_idx]["self_attn"][0](z)
                z = self.stages[stage_idx]["self_attn"][1](z)
            else:
                # Global transformer expects sequential input [B, H*W, D]
                D_current = z.shape[-1]
                z = z.view(B, H * W, D_current)
                z = self.stages[stage_idx]["self_attn"](z)
                # Reshape back to 2D spatial grid for the next upsampling stage
                z = z.view(B, H, W, D_current)

        # 3. Final projection to output channels
        out = self.head(z)  # [B, H_highest, W_highest, out_channels]
        return out.permute(0, 3, 1, 2).contiguous()


class MultiscaleSWINNO(nn.Module):
    """
    Complete Multiscale SWIN Neura Operator
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        enc_windows: List[Tuple[int, int]],
        dec_windows: List[Union[Tuple[int, int], None]],
        model_dim: Union[int, List[int]] = 128,
        grid_resolutions: List[Tuple[int, int]] = [(32, 32), (16, 16)],
        num_heads: int = 4,
        mlp_factor: int = 4,
        rmsnorm: bool = True,
        activation: Callable = nn.GELU,
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

        # Process model dimensions
        if isinstance(model_dim, int):
            self.enc_dims = [model_dim] * (len(grid_resolutions) + 1)
        else:
            self.enc_dims = model_dim

        # Decoder mirrors the encoder dimensions in reverse order
        self.dec_dims = list(reversed(self.enc_dims))

        # Instantiate sub-modules
        self.encoder = MultiscaleCAEncoder(
            in_channels=in_channels,
            out_channels=self.enc_dims[-1],
            windows=enc_windows,
            model_dim=self.enc_dims,
            grid_resolutions=grid_resolutions,
            num_heads=num_heads,
            mlp_factor=mlp_factor,
            rmsnorm=rmsnorm,
            activation=activation,
        )

        self.decoder = MultiscaleCADecoder(
            latent_dim=self.dec_dims,
            out_channels=out_channels,
            grid_windows=dec_windows,
            n_heads=num_heads,
            mlp_ratio=mlp_factor,
            rmsnorm=rmsnorm,
            activation=activation,
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
            z: Latent tensor [B, H_last * W_last, D_latent].
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full multi-scale cross-attention autoencoder.

        Args:
            x: Input tensor [B, in_channels, H, W].
        Returns:
            Output reconstruction [B, out_channels, H, W].
        """
        # 1. Compress via progressive cross-attentions
        z, shortcuts = self.encode(x)

        # 2. Decompress via block-grouped cross-attentions
        out = self.decode(z, shortcuts[:-1])

        return out
