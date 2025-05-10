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
