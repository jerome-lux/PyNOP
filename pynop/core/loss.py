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


class nMSELoss(nn.Module):
    """
    Compute the normalized MSE Loss per channel
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):

        spatial_dims = tuple(range(2, pred.ndim))

        # per channel - MSE
        mse_per_channel = torch.mean((pred - target) ** 2, dim=spatial_dims)
        norm = torch.mean(target**2, dim=spatial_dims)
        rel_mse = mse_per_channel / (norm + self.eps)

        return torch.mean(rel_mse)


class nL2Loss(torch.nn.Module):
    """
    Compute the normalized L2 Loss per channel
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):

        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)

        diff_norms = torch.norm(pred_flat - target_flat, p=2, dim=2)  # [B, C]
        norm = torch.norm(target_flat, p=2, dim=2)  # [B, C]

        # Per channel relative error
        rel_errors = diff_norms / (norm + self.eps)  # [B, C]

        loss_per_sample = torch.mean(rel_errors, dim=1)  # [B]
        return torch.mean(loss_per_sample)


def ortho_loss(basis, n_samples=2048, normalize=True, mode="MSE"):
    """
    compute
    - A loss based on the QR decomposition if mode=="QR"
    - a sampled loss to force orthogonality of the kernels
    if mode ="fro" compute the Frobenius norm gram - identity (seems to be numerically bad)
    basis: (B, H, W, m1, m2) - note that basis should be already normalized
    n_samples: number of samples
    """
    B, H, W, m1, m2 = basis.shape
    M = m1 * m2
    basis_flat = basis.contiguous().view(B, H * W, M)
    n_samples = max(min(n_samples, H * W), 1)

    # Spatial sampling
    indices = torch.randint(0, H * W, (n_samples,), device=basis.device)
    sampled_basis = basis_flat[:, indices, :]  # (B, n_samples, M)

    # Gram matrix over space
    # (B, M, n_samples) @ (B, n_samples, M) -> (B, M, M)
    if basis.is_complex():
        gram = torch.matmul(sampled_basis.transpose(-2, -1).conj(), sampled_basis)
    else:
        gram = torch.matmul(sampled_basis.transpose(-2, -1), sampled_basis)

    if normalize:
        gram = gram * ((H * W) / n_samples)
    else:
        gram = gram / n_samples

    identity = torch.eye(M, device=basis.device).unsqueeze(0).to(basis.dtype)
    if mode == "fro":
        ortho_loss = torch.linalg.matrix_norm(gram - identity, ord="fro") / M
        ortho_loss = ortho_loss.mean()
    else:
        ortho_loss = torch.mean(torch.abs(gram - identity) ** 2)

    off_diag = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))

    return off_diag.abs().mean()
