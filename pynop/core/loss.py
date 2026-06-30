import torch
from torch import nn
import torch.nn.functional as F


def ZeroCenteredGradientPenalty(data, critics):
    (Gradient,) = torch.autograd.grad(outputs=critics.sum(), inputs=data, create_graph=True)
    return Gradient.square().sum([1, 2, 3])


def D_GANLoss(discriminator, fake_data, real_data, gan_type="GAN", gp_weight=1, R1=False, R2=False, softlabel=0.0):
    """Compute GAN Loss
    discriminator: discriminator network
    generator: generator network
    real_data: batch of real data
    l: weight for gradient pe
    R1: Apply R1 gradient penalty
    R2: Apply R2 gradient penalty
    """

    # detach to avoid training the generator and require_grad because we need it to compute the gradient penalties
    fake_data = fake_data.detach().requires_grad_(True)
    real_data = real_data.detach().requires_grad_(True)

    real_critics = discriminator(real_data)
    fake_critics = discriminator(fake_data)

    valid = torch.ones_like(real_critics) - softlabel
    fake = torch.zeros_like(real_critics) + softlabel

    R1p = 0.0
    R2p = 0.0
    if R1:
        R1p = ZeroCenteredGradientPenalty(real_data, real_critics).mean()
        # print("R1", R1, end=",")
    if R2:
        R2p = ZeroCenteredGradientPenalty(fake_data, fake_critics).mean()
        # print("R2", R2, end=" ")

    if gan_type == "WGAN":
        loss = -torch.mean(real_critics) + torch.mean(fake_critics)
    elif gan_type == "R3GAN":
        loss = torch.mean(F.softplus(fake_critics - real_critics))
    elif gan_type == "RpGAN":
        valid_loss = nn.BCEWithLogitsLoss()(real_critics - fake_critics, valid)
        fake_loss = nn.BCEWithLogitsLoss()(fake_critics - real_critics, fake)
        loss = valid_loss + fake_loss
    elif gan_type == "GAN":
        valid_loss = nn.BCEWithLogitsLoss()(real_critics, valid)
        fake_loss = nn.BCEWithLogitsLoss()(fake_critics, fake)
        loss = valid_loss + fake_loss
    else:
        raise NotImplementedError(f"GAN type {gan_type} not implemented.")

    return loss + gp_weight * (R1p + R2p), R1p, R2p


def G_GANLoss(discriminator, fake_data, real_data, gan_type="GAN"):

    fake_critics = discriminator(fake_data)
    if gan_type == "GAN":
        valid = torch.ones_like(fake_critics)
        return nn.BCEWithLogitsLoss()(fake_critics, valid)
    elif gan_type == "WGAN":
        return -torch.mean(fake_critics)
    elif gan_type == "SoftplusWGAN":
        return torch.mean(F.softplus(-fake_critics))
    elif gan_type == "R3GAN":
        real_critics = discriminator(real_data)
        return torch.mean(F.softplus(-(fake_critics - real_critics)))
    elif gan_type == "RpGAN":
        real_critics = discriminator(real_data)
        valid = torch.ones_like(real_critics)
        return nn.BCEWithLogitsLoss()(fake_critics - real_critics, valid)
    else:
        raise NotImplementedError(f"GAN type {gan_type} not implemented.")


def diffusion_loss(c_pred, ct, dt, diffusivity, x_coords, y_coords, time_derivative="fd"):
    """
    Compute the diffusion loss for a given prediction and initial condition.
    Parameters
    ----------
    c_pred : torch.Tensor
        The predicted concentration field at time t+dt.
    c0 : torch.Tensor
        The initial concentration field at time t=t.
    dt : float
        The time step size.
    diffusivity : float
        The diffusivity constant.
    x_coords : torch.Tensor
        The x-coordinates of the grid points.
    y_coords : torch.Tensor
        The y-coordinates of the grid points.
    time_derivative : str
        Either 'fd' (finite differnce) or 'auto' (autograd). If auto, dt is not the timestep, bu the actual time point.
    Returns
    -------
    torch.Tensor
        The diffusion loss.
    """

    if not x_coords.requires_grad:
        x_coords.requires_grad_(True)
    if not y_coords.requires_grad:
        y_coords.requires_grad_(True)

    if time_derivative == "fd":
        dc_dt_approx = (c_pred - ct) / dt
    elif time_derivative == "auto":
        dc_dt_approx = torch.autograd.grad(
            outputs=c_pred, inputs=dt, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True
        )[0]
    else:
        raise ValueError("time_derivative must be either 'fd' or 'auto'")

    grad_c_x = torch.autograd.grad(
        outputs=c_pred, inputs=x_coords, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True
    )[0]

    grad_c_y = torch.autograd.grad(
        outputs=c_pred, inputs=y_coords, grad_outputs=torch.ones_like(c_pred), create_graph=True, retain_graph=True
    )[0]

    D_grad_c_x = diffusivity * grad_c_x
    D_grad_c_y = diffusivity * grad_c_y

    div_D_grad_c_x = torch.autograd.grad(
        outputs=D_grad_c_x,
        inputs=x_coords,
        grad_outputs=torch.ones_like(D_grad_c_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    div_D_grad_c_y = torch.autograd.grad(
        outputs=D_grad_c_y,
        inputs=y_coords,
        grad_outputs=torch.ones_like(D_grad_c_y),
        create_graph=True,
        retain_graph=True,
    )[0]

    div_D_grad_c = div_D_grad_c_x + div_D_grad_c_y

    residual = dc_dt_approx - div_D_grad_c

    return torch.mean(residual**2)


class DWMSELoss(nn.Module):
    """
    Compute the MSE Loss per channel
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        spatial_dims = tuple(range(2, pred.ndim))

        # per channel - MSE
        mse_per_channel = torch.mean((pred - target) ** 2, dim=spatial_dims)

        return torch.mean(mse_per_channel)


class NormalizedTimeDerivativeMSE(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, input, dt):
        # input MUSt be B, C, H, W, ...
        # pred is the time derivative prediction

        spatial_dims = tuple(range(2, pred.ndim))
        gt_derivative = (target - input) / dt

        # per channel - MSE
        mse_per_channel = torch.mean((pred - gt_derivative) ** 2, dim=spatial_dims)
        norm = torch.mean(gt_derivative**2, dim=spatial_dims)
        rel_mse = mse_per_channel / (norm + self.eps)

        return torch.mean(rel_mse)


class nDWMSELoss(nn.Module):
    """
    Compute the normalized MSE Loss per channel
    """

    def __init__(self, shift=0.0, scale=1.0, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.shift = shift
        self.scale = scale

    def forward(self, pred, target):

        # pred = torch.sign(pred) * torch.log(1 + torch.abs(pred) / self.eps)
        # target = torch.sign(target) * torch.log(1 + torch.abs(target) / self.eps)

        spatial_dims = tuple(range(2, pred.ndim))

        # per channel - MSE
        mse_per_channel = torch.mean((pred - target) ** 2, dim=spatial_dims)
        norm = torch.mean(((target + self.shift) * self.scale) ** 2, dim=spatial_dims)
        rel_mse = mse_per_channel / (norm + self.eps)

        return torch.mean(rel_mse)


class WeightedMSE(nn.Module):
    """
    Penalize errors more heavily on channels with lower L2 norms.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Identify spatial dimensions (B, C, H, W -> dims 2, 3)
        spatial_dims = tuple(range(2, pred.ndim))

        # MSE per channel: (B, C)
        mse_per_channel = torch.mean((pred - target) ** 2, dim=spatial_dims)

        # L2 Norm per channel: (B, C)
        channel_norm = torch.sqrt(torch.mean(target**2, dim=spatial_dims) + self.eps)

        # Scale based on the maximum energy in the batch/channels
        # Channels with max energy get weight 1.0
        # Channels with lower energy get weight > 1.0
        max_norm, _ = torch.max(channel_norm, dim=1, keepdim=True)
        weights = max_norm / (channel_norm + self.eps)

        # Apply weights to focus on low-energy channels
        weighted_mse = mse_per_channel * weights

        return torch.mean(weighted_mse)


class MaskedSpectralLoss(nn.Module):
    def __init__(self, k_min=0.0, k_max=0.1, norm="ortho"):
        """
        Spectral loss focused on a radial frequency band.
        - k_min=0.0, k_max=0.1: Low frequencies
        - k_min=0.3, k_max=0.7: High frequencies
        """
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.norm = norm
        self.register_buffer("_k_mag", None)

    def _get_k_mag(self, h, w, device):
        # Cache the radial grid to avoid recomputing every step
        fy = torch.fft.fftfreq(h, device=device).abs().reshape(-1, 1)
        fx = torch.fft.rfftfreq(w, device=device).reshape(1, -1)
        return torch.sqrt(fx**2 + fy**2)

    def forward(self, pred, target):
        p_fft = torch.fft.rfftn(pred, dim=(-2, -1), norm=self.norm)
        t_fft = torch.fft.rfftn(target, dim=(-2, -1), norm=self.norm)

        h, w = pred.shape[-2:]
        if self._k_mag is None or self._k_mag.shape != p_fft.shape[-2:]:
            self._k_mag = self._get_k_mag(h, w, pred.device)

        # Create the radial mask
        mask = (self._k_mag >= self.k_min) & (self._k_mag <= self.k_max)

        # Apply mask and compute MSE on complex coefficients
        diff = torch.abs(p_fft - t_fft)  # / (torch.abs(t_fft) + 1e-6)
        masked_diff = (diff**2) * mask.float()

        # We divide by the number of active points in the mask
        # to have a consistent mean regardless of mask size.
        loss = masked_diff.sum() / (mask.sum() + 1e-8)

        return loss


class SpectralLoss(nn.Module):
    def __init__(self, beta=1.0, alpha=2.0):
        super().__init__()
        self.beta = beta  # if beta > 1, it penalizes high frequencies
        self.alpha = alpha  # 1-> MAE, 2-> MSE

    def forward(self, pred, target):
        # input shape: B, C, H, W

        # Fast Fourier Transform (Real 2D)
        # Shape: [batch, channel, height, width_freq]
        pred_fft = torch.fft.rfftn(pred, dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.rfftn(target, dim=(-2, -1), norm="ortho")

        # Compute frequency coordinates
        h, w = pred.shape[-2], pred.shape[-1]
        freq_y = torch.fft.fftfreq(h, device=pred.device).abs().reshape(-1, 1)
        freq_x = torch.fft.rfftfreq(w, device=pred.device).reshape(1, -1)

        # Magnitude of frequency vector (distance from origin)
        k_mag = torch.sqrt(freq_x**2 + freq_y**2)
        weight = 1.0 + (k_mag * 2) ** self.beta

        # Weighted error in frequency domain
        diff_fft = torch.abs(pred_fft - target_fft)  # / (torch.abs(target_fft) + self.eps)

        spec_loss = torch.mean(weight * (diff_fft**self.alpha))

        return spec_loss


class StructureLoss(nn.Module):
    def __init__(self, k=3, loss_type="mse", alpha=1.0, eps=1e-8):
        super().__init__()
        self.k = k
        self.eps = eps
        self.alpha = alpha
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=k)
        self.loss_fn = F.mse_loss if loss_type == "mse" else F.l1_loss

    def forward(self, pred, target):

        error = self.loss_fn(pred, target, reduction="none")

        patch_error = self.avg_pool(error)
        mean_target = self.avg_pool(target)
        mean_sq_target = self.avg_pool(target**2)
        patch_std = torch.sqrt(F.relu(mean_sq_target - mean_target**2) + self.eps)

        loss = patch_error * (patch_std + self.alpha)
        return loss.mean()


class SobolevLoss(nn.Module):

    def __init__(self, l1: float = 1, l2: float = 1, eps: float = 1e-6):
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.eps = eps

    def forward(self, pred, target, norm=False):
        # Gradient
        grad_pred_x = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_pred_y = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_target_y = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Normalisation
        norm_grad = torch.mean(grad_target_x[:, :, :, :-1] ** 2 + grad_target_y[:, :, :-1, :] ** 2) + self.eps
        grad_loss = F.mse_loss(grad_pred_x, grad_target_x) + F.mse_loss(grad_pred_y, grad_target_y)
        if norm:
            grad_loss = grad_loss / norm_grad

        # Laplacian
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=pred.dtype, device=pred.device)
        lap_kernel = lap_kernel.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)

        # On utilise circular pour des domaines périodiques ou replicate sinon
        pred_lap = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode="replicate"), lap_kernel, groups=pred.shape[1])
        target_lap = F.conv2d(F.pad(target, (1, 1, 1, 1), mode="replicate"), lap_kernel, groups=target.shape[1])

        norm_lap = torch.mean(target_lap**2) + self.eps
        lap_loss = F.mse_loss(pred_lap, target_lap)
        if norm:
            lap_loss = lap_loss / norm_lap

        return self.l1 * grad_loss + self.l2 * lap_loss


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.register_buffer("bandwidth_multipliers", mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))
        self.bandwidth = bandwidth

    def forward(self, X, Y):
        """
        X: [B, D]
        Y: [B, D]
        """
        X_flat = X.view(-1, X.shape[-1])  # [B*?, D]
        Y_flat = Y.view(-1, Y.shape[-1])  # [B*?, D]

        L2_distances = torch.cdist(X_flat, Y_flat) ** 2  # [N, M]

        if self.bandwidth is None:
            n_samples = X_flat.shape[0]
            X_dist = torch.cdist(X_flat, X_flat) ** 2
            bandwidth = X_dist.data.sum() / (n_samples**2 - n_samples)
        else:
            bandwidth = self.bandwidth

        # [n_kernels, N, M]
        kernels = torch.exp(-L2_distances[None, ...] / (bandwidth * self.bandwidth_multipliers.view(-1, 1, 1)))

        return kernels.mean(dim=0)  # [N, M]


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, z, prior=None):
        """
        z: [batch, modes, hidden_channels] ou [batch, latent_dim]
        prior: échantillons N(0,1) de même taille que z, ou None pour génération auto
        """
        batch_size = z.shape[0]
        z_flat = z.view(batch_size, -1)  # [batch, latent_dim]

        if prior is None:
            prior = torch.randn_like(z_flat)

        # Calcul MMD
        K_zz = self.kernel(z_flat, z_flat)
        K_zp = self.kernel(z_flat, prior)
        K_pp = self.kernel(prior, prior)

        mmd = K_zz.mean() - 2 * K_zp.mean() + K_pp.mean()
        return mmd


def preprocess_to_sigreg(latent):
    """Preprocess channels-last token tensors [B, M, C] for SIGReg.

    Standardizes the features across the batch and token dimensions
    to mimic the behavior of BatchNorm1d, then
    permutes to [M, B, C] to align with SIGReg's expectation.
    """
    if latent.ndim != 3:
        raise ValueError(f"Expected latent with shape [batch, num_tokens, channels], got {tuple(latent.shape)}")

    batch_size, num_tokens, channels = latent.shape

    if num_tokens == 1:
        # Standardize across batch dimension only if there is only 1 token
        eps = 1e-5
        mean = latent.mean(dim=0, keepdim=True)
        var = latent.var(dim=0, keepdim=True, unbiased=False)
        latent_norm = (latent - mean) / torch.sqrt(var + eps)
        return latent_norm.view(batch_size, channels)

    # Standardize across Batch (0) and Tokens (1) for each channel (2)
    # This matches the exact reduction behavior of BatchNorm
    eps = 1e-5
    mean = latent.mean(dim=(0, 1), keepdim=True)
    var = latent.var(dim=(0, 1), keepdim=True, unbiased=False)
    latent_norm = (latent - mean) / torch.sqrt(var + eps)

    # Permute from [B, M, C] to [M, B, C]
    # Axe -3 inside SIGReg will point to B
    return latent_norm.permute(1, 0, 2)


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Accepts latents of shape (B, N, D) with two modes:
      - 'flat'     : reshape to (B*N, D) — recommanded for homogeneous spatial tokens
      - 'per_pos'  : SIGREG per position over (B, D), averaged over N — more rigorous
                     but requires a large B to estimate covariance reliably

    Args:
        n_sketches  : number of random projection directions (default: 256)
        knots       : number of quadrature points for the integral (default: 17)
        mode        : 'flat' or 'per_pos' (default: 'flat')
    """

    def __init__(
        self,
        n_sketches: int = 256,
        knots: int = 17,
        mode: str = "flat",
    ):
        super().__init__()

        if mode not in ("flat", "per_pos"):
            raise ValueError(f"mode must be 'flat' or 'per_pos', got '{mode}'")
        self.mode = mode
        self.n_sketches = n_sketches

        # Quadrature grid on [0, 3] — approximates ∫ error(t) · φ(t) dt
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoidal weights × Gaussian window φ(t) = exp(-t²/2)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)  # (knots,)
        self.register_buffer("phi", window)  # (knots,)
        self.register_buffer("weights", weights * window)  # (knots,)

    def _sigreg_2d(self, z: torch.Tensor) -> torch.Tensor:
        """
        Core SIGREG on a 2D tensor of shape (S, D).
        S is the number of i.i.d. samples (either B*N or B for one position).

        The statistic estimates:
            ∫ [ |E[cos(t·<z,a>)] - exp(-t²/2)|² + |E[sin(t·<z,a>)]|² ] φ(t) dt
        averaged over random unit directions a ~ Uniform(S^{D-1}).

        This is zero iff z ~ N(0, I).
        """
        S, D = z.shape

        # Random sketch matrix — unit-norm columns : (D, n_sketches)
        A = torch.randn(D, self.n_sketches, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        # Projections : (S, n_sketches)
        proj = z @ A

        # Characteristic function evaluations : (S, n_sketches, knots)
        x_t = proj.unsqueeze(-1) * self.t  # broadcast over knots

        # Empirical characteristic function, averaged over S samples
        # E[cos], E[sin] : (n_sketches, knots)
        cf_cos = x_t.cos().mean(dim=0)
        cf_sin = x_t.sin().mean(dim=0)

        # Target characteristic function of N(0,1): φ(t) = exp(-t²/2)
        # Error : (n_sketches, knots)
        err = (cf_cos - self.phi).square() + cf_sin.square()

        # Quadrature integral + scale by S (makes the statistic sample-size-independent)
        # (n_sketches,)
        statistic = (err @ self.weights) * S

        return statistic.mean()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : tensor of shape (B, N, D)

        Returns:
            scalar loss
        """
        if z.ndim != 3:
            raise ValueError(f"Expected tensor of shape (B, N, D), got {tuple(z.shape)}")
        B, N, D = z.shape

        if self.mode == "flat":
            # Treat all B*N tokens as i.i.d. samples
            # Valid when tokens are spatially homogeneous (regular grid, symmetric slots…)
            z_flat = z.reshape(B * N, D)  # (B*N, D)
            return self._sigreg_2d(z_flat)

        else:  # per_pos
            # Estimate the distribution independently at each of the N positions.
            # Requires B large enough for a reliable covariance estimate.
            # z.permute(1, 0, 2) : (N, B, D)
            loss = torch.stack([self._sigreg_2d(z[:, n, :]) for n in range(N)]).mean()
            return loss
