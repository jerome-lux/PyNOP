import math
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """1D RoPE"""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        # Precompute inverse frequencies
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, coords):
        # x:      [B, n_h, L, head_dim]
        # coords: [B, L, 1]

        angles = coords.unsqueeze(1) * self.inv_freq.view(1, 1, 1, -1)
        # [B, L, 1] → unsqueeze(1) → [B, 1, L, 1]
        # inv_freq: [1, 1, 1, dim//2]
        # angles:   [B, 1, L, dim//2] ✓ broadcast sur n_h

        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [B, 1, L, head_dim]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)

        def rotate_half(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape(x.shape)

        return (x * cos) + (rotate_half(x) * sin)


class CartesianEmbedding(nn.Module):
    """
    Generates and concatenates normalized Cartesian coordinates (x, y) as additional channels.
    Coordinates are normalized to the range [-1, 1] using torch.meshgrid.
    """

    def __init__(self, minval=-1, maxval=1):
        super().__init__()
        # No learnable parameters needed for this embedding layer
        self.minval = minval
        self.maxval = maxval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates normalized (x, y) coordinates using meshgrid and concatenates them to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Input tensor concatenated with normalized x and y coordinates.
                          Shape (B, C + 2, H, W).
        """
        batch_size, _, height, width = x.shape
        device = x.device

        x_lin = torch.linspace(self.minval, self.maxval, steps=width, device=device)  # Shape (W,)
        y_lin = torch.linspace(self.minval, self.maxval, steps=height, device=device)  # Shape (H,)

        grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)

        coords = coords.permute(2, 0, 1).unsqueeze(0)  # Shape (1, 2, H, W)

        # Expand for the batch size to match the input tensor batch size
        coords = coords.expand(batch_size, -1, -1, -1)  # Shape (B, 2, H, W)

        # Output shape (B, C_in + 2, H, W)
        output = torch.cat([x, coords], dim=1)

        return output


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    From https://github.com/facebookresearch/tuna-2/blob/main/tuna/models/misc.py
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):

        super().__init__()
        self.dim = dim
        self.max_period = max_period

        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, timesteps):
        """
        :param timesteps: Tensor 1-D de taille [N]
        :return: Tensor de taille [N, dim]
        """
        # timesteps[:, None] transforme [N] en [N, 1]
        args = timesteps[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2:
            zero_pad = torch.zeros(timesteps.shape[0], 1, device=timesteps.device)
            embedding = torch.cat([embedding, zero_pad], dim=-1)

        return embedding


class AdaptivePE2d(nn.Module):
    def __init__(self, dim):
        
        super().__init__()
        assert dim % 4 == 0
        self.num_frequencies = dim // 4
        self.register_buffer("indices", torch.arange(self.num_frequencies).float())

    def forward(self, coords):
        """
        Args:
            coords: [H, W, 2]
        """
        H, W, _ = coords.shape
        device = coords.device

        max_freq_h = float(H) / 2.0
        max_freq_w = float(W) / 2.0

        log_h = math.log(max_freq_h) if max_freq_h > 1 else 0.0
        log_w = math.log(max_freq_w) if max_freq_w > 1 else 0.0

        freq_h = torch.exp(self.indices * (log_h / max(1, self.num_frequencies - 1))).to(device)
        freq_w = torch.exp(self.indices * (log_w / max(1, self.num_frequencies - 1))).to(device)

        angles_h = coords[:, :, 0:1] * freq_h * math.pi
        angles_w = coords[:, :, 1:2] * freq_w * math.pi

        pe_h = torch.stack([torch.sin(angles_h), torch.cos(angles_h)], dim=-1)
        pe_w = torch.stack([torch.sin(angles_w), torch.cos(angles_w)], dim=-1)

        return torch.cat([pe_h.flatten(2), pe_w.flatten(2)], dim=-1)


def positional_encoding_2d(d_model, H=512, W=512):

    # implementation of the classical PE adpated to 2D cartesian grids

    # d_model must be divisible by 4 (sin/cos for H and sin/cos for W)
    assert d_model % 4 == 0, "d_model must be a multiple of 4 for 2D PE"

    pe = torch.zeros(d_model, H, W)

    # Split d_model in two: one half for H, one half for W
    d_half = d_model // 2

    # Compute div_term for the half dimension
    # Using the classic 10000 base from Vaswani et al.
    div_term = torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half))

    # Create position grids
    pos_h = torch.arange(0, H).unsqueeze(1)
    pos_w = torch.arange(0, W).unsqueeze(1)

    # Compute encodings for H (rows)
    pe_h = torch.zeros(H, d_half)
    pe_h[:, 0::2] = torch.sin(pos_h * div_term)
    pe_h[:, 1::2] = torch.cos(pos_h * div_term)

    # Compute encodings for W (columns)
    pe_w = torch.zeros(W, d_half)
    pe_w[:, 0::2] = torch.sin(pos_w * div_term)
    pe_w[:, 1::2] = torch.cos(pos_w * div_term)

    # Expand and concatenate to create the 2D grid [d_model, H, W]
    # [H, 1, d_half] + [1, W, d_half] -> [H, W, d_model]
    pe[0:d_half, :, :] = pe_h.transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    pe[d_half:, :, :] = pe_w.transpose(0, 1).unsqueeze(1).repeat(1, H, 1)

    return pe

def sin_positional_encoding_2d(coords, d_model, max_freq=32.0):
    """
    Args:
        coords: [H, W, 2] - Coordonnées normalisées entre -1 et 1
        d_model: Dimension totale de l'encodage (doit être multiple de 4)
        max_freq: Fréquence maximale (Nyquist)
    """
    H, W, _ = coords.shape
    device = coords.device

    # Nombre de fréquences par axe
    d_freq = d_model // 4

    freq_bands = torch.exp(torch.linspace(0, math.log(max_freq), d_freq, device=device)) * math.pi

    y_coords = coords[:, :, 0:1]  # [H, W, 1]
    x_coords = coords[:, :, 1:2]  # [H, W, 1]

    # Calcul des angles : [H, W, d_freq]
    angles_y = y_coords * freq_bands
    angles_x = x_coords * freq_bands

    # Encodage sin/cos pour chaque axe
    pe_y = torch.cat([torch.sin(angles_y), torch.cos(angles_y)], dim=-1)
    pe_x = torch.cat([torch.sin(angles_x), torch.cos(angles_x)], dim=-1)

    # Concaténation finale : [H, W, d_model]
    pe = torch.cat([pe_y, pe_x], dim=-1)

    return pe
