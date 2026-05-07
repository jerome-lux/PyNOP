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
