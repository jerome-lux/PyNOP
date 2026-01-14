import os
from typing import Callable, Sequence, Union
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from .metrics import AverageMeter
from .loss import G_GANLoss, D_GANLoss, ZeroCenteredGradientPenalty, diffusion_loss
import math
from torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train_step(model, inputs, targets, loss_fn, optimizer, lossweights=1.0):

    preds = model(inputs)
    # residual = targets - preds
    if preds.dim() == 5 and preds.shape[1] == 1:
        preds = preds.squeeze(1)

    loss = 0.0
    if not (isinstance(lossweights, tuple) or isinstance(lossweights, list)):
        lossweights = [lossweights] * len(loss_fn)
    for i, loss_function in enumerate(loss_fn):
        # loss += loss_function(residual, torch.zeros_like(residual)) * lossweights[i]
        loss += loss_function(preds, targets) * lossweights[i]
    # loss = loss / len(loss_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def test_step(model, inputs, targets, loss_fn, lossweights=1.0):

    preds = model(inputs)
    if preds.dim() == 5 and preds.shape[1] == 1:
        preds = preds.squeeze(1)

    loss = 0.0

    if isinstance(loss_fn, (list, tuple)):
        if not isinstance(lossweights, (list, tuple)):
            lossweights = [lossweights] * len(loss_fn)

        for i, loss_function in enumerate(loss_fn):
            loss += loss_function(preds, targets) * lossweights[i]
    else:
        loss = loss_fn(preds, targets) * lossweights

    return loss


# 3 TODO: implemmmment one stage GAN https://github.com/zju-vipa/OSGAN


def train(
    model,
    dataloader,
    epochs,
    optimizer,
    loss_fn,
    savepath,
    test_loader=None,
    device=None,
    lossweights=1.0,
    scheduler=None,
    iterations=None,
):

    os.makedirs(savepath, exist_ok=True)

    writer = SummaryWriter(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)
    # print("Model device:", next(model.parameters()).device)
    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):

        model.train()
        loss_meter = AverageMeter()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            loss = train_step(model, inputs, targets, loss_fn, optimizer, lossweights=lossweights)

            loss_meter.update(loss.item(), inputs.size(0))

            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None and hasattr(scheduler, "get_last_lr")
                else optimizer.param_groups[0]["lr"]
            )

            progress_bar.set_postfix(loss=loss_meter.avg, lr=current_lr)

            if i >= iterations:
                break

        avg_loss = loss_meter.avg

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / Path("best_model.pth"))

        log_lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None and hasattr(scheduler, "get_last_lr")
            else optimizer.param_groups[0]["lr"]
        )

        writer.add_scalar("loss", avg_loss, epoch + 1)

        writer.add_scalar("lr", log_lr, epoch + 1)

        if test_loader is not None:

            with torch.no_grad():
                valoss_meter = AverageMeter()
                model.eval()
                progress_bar = tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=True, total=len(test_loader)
                )
                for i, (inputs, targets) in enumerate(progress_bar):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    valoss = test_step(model, inputs, targets, loss_fn, lossweights=lossweights)

                    valoss_meter.update(valoss.item(), inputs.size(0))

            progress_bar.set_postfix(loss=valoss_meter.avg)

            avg_valoss = valoss_meter.avg

            if avg_valoss < best_val_loss or epoch == 0:
                best_val_loss = avg_valoss
                torch.save(model.state_dict(), Path(savepath) / Path("best_val_model.pth"))

            writer.add_scalar("val_loss", avg_valoss, epoch + 1)


def adversarial_train(
    generator,
    discriminator,
    dataloader,
    epochs,
    gen_optimizer,
    discr_optimizer,
    test_loader=None,
    gen_loss_func=G_GANLoss,
    discr_loss_func=D_GANLoss,
    rec_loss_func=None,
    rec_loss_weight=0.5,
    savepath=os.path.join(os.getcwd(), "logs"),
    d_train_steps_ratio=1,
    gen_scheduler=None,
    discr_scheduler=None,
    iterations=None,
    device=None,
):
    """TODO:
    Maybe implement update rule in Accelerated WGAN update strategy with loss change rate balancing abs()
    update D if (Ld(i) - Ld(i-1)) / Ld(i-1) > lambda * (Lg(i) - Lg(i-1)) / Lg(i-1)
    """

    writer = SummaryWriter(savepath)

    os.makedirs(savepath, exist_ok=True)

    device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    gen_best_loss = float("inf")
    rec_best_loss = float("inf")
    discr_best_loss = float("inf")

    test_gen_best_loss = float("inf")
    test_rec_best_loss = float("inf")
    test_discr_best_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)
    else:
        iterations = min(len(dataloader), iterations)

    for epoch in range(epochs):

        generator.train()
        discriminator.train()

        gen_loss_meter = AverageMeter()
        discr_loss_meter = AverageMeter()
        adv_loss_meter = AverageMeter()
        rec_loss_val_meter = AverageMeter()
        rec_loss_val_meter.update(0.0, 1)
        gp1_meter = AverageMeter()
        gp2_meter = AverageMeter()

        # Training loop
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (real_data, _) in enumerate(progress_bar):
            real_data = real_data.to(device)
            fake_data = generator(real_data)

            # discriminator loss
            discr_optimizer.zero_grad()
            discr_loss, gp1, gp2 = discr_loss_func(discriminator, fake_data, real_data)

            discr_loss.backward()
            discr_optimizer.step()
            discr_loss_meter.update(discr_loss.item(), real_data.size(0))
            gp1_meter.update(gp1.item(), real_data.size(0))
            gp2_meter.update(gp2.item(), real_data.size(0))

            # generator loss
            if i % d_train_steps_ratio == 0:

                gen_optimizer.zero_grad()

                adv_loss = gen_loss_func(discriminator, fake_data, real_data)

                rec_loss_value = 0.0
                if rec_loss_func is not None:
                    if not isinstance(rec_loss_func, (list, tuple)):
                        rec_loss_func = [rec_loss_func]
                    for rec_loss in rec_loss_func:
                        rec_loss_value += rec_loss(fake_data, real_data)

                    gen_loss = (1.0 - rec_loss_weight) * adv_loss + rec_loss_weight * rec_loss_value

                else:
                    gen_loss = adv_loss

                gen_loss.backward()
                gen_optimizer.step()

                # Update loss meter
                gen_loss_meter.update(gen_loss.item(), real_data.size(0))
                adv_loss_meter.update(adv_loss.item(), real_data.size(0))
                if rec_loss_func is not None:
                    rec_loss_val_meter.update(rec_loss_value.item(), real_data.size(0))

            progress_bar.set_postfix(
                ordered_dict={
                    "gen_loss": gen_loss_meter.avg,
                    "adv_loss": adv_loss_meter.avg,
                    "rec_loss": rec_loss_val_meter.avg,
                    "discr_loss": discr_loss_meter.avg,
                    "gp1": gp1_meter.avg,
                    "gp2": gp2_meter.avg,
                    "gen_lr": gen_optimizer.param_groups[0]["lr"],
                    "discr_lr": discr_optimizer.param_groups[0]["lr"],
                }
            )

            # Schedulers qui doivent être updatés à chaque batch
            if gen_scheduler is not None and not isinstance(gen_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                gen_scheduler.step()

            if discr_scheduler is not None and not isinstance(
                discr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                discr_scheduler.step()

            if i >= iterations:
                break

        gen_avg_loss = gen_loss_meter.avg
        discr_avg_loss = discr_loss_meter.avg

        # Schedulers qui doivent être updatés à chaque epoch
        if gen_scheduler is not None and isinstance(gen_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            gen_scheduler.step(gen_avg_loss)

        if discr_scheduler is not None and isinstance(discr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            discr_scheduler.step(discr_avg_loss)

        if gen_avg_loss < gen_best_loss or epoch == 0:
            gen_best_loss = gen_avg_loss
            torch.save(generator.state_dict(), Path(savepath) / Path("gen_best_model.pth"))

        if rec_loss_val_meter.avg < rec_best_loss or epoch == 0:
            rec_best_loss = rec_loss_val_meter.avg
            torch.save(generator.state_dict(), Path(savepath) / Path("rec_best_model.pth"))

        if discr_avg_loss < discr_best_loss or epoch == 0:
            discr_best_loss = discr_avg_loss
            torch.save(discriminator.state_dict(), Path(savepath) / Path("discriminator_best_model.pth"))

        writer.add_scalar("Gen_loss", gen_avg_loss, epoch + 1)
        writer.add_scalar("Gen GAN loss", adv_loss_meter.avg, epoch + 1)
        writer.add_scalar("Gen rec_loss", rec_loss_val_meter.avg, epoch + 1)
        writer.add_scalar("Discriminator_loss", discr_loss_meter.avg, epoch + 1)
        writer.add_scalar("Gen lr", gen_optimizer.param_groups[0]["lr"], epoch + 1)
        writer.add_scalar("Discr lr", discr_optimizer.param_groups[0]["lr"], epoch + 1)

        if test_loader is not None:

            test_gen_loss_meter = AverageMeter()
            test_discr_loss_meter = AverageMeter()
            test_adv_loss_meter = AverageMeter()
            test_rec_loss_val_meter = AverageMeter()
            test_rec_loss_val_meter.update(0, 1)

            # Test step: set the model to inference mode
            generator.eval()
            discriminator.eval()
            progress_bar = tqdm(
                test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=True, total=len(test_loader)
            )

            for i, (real_data, _) in enumerate(progress_bar):
                real_data = real_data.to(device)
                fake_data = generator(real_data)

                discr_loss, _, _ = discr_loss_func(discriminator, fake_data, real_data)

                adv_loss = gen_loss_func(discriminator, fake_data, real_data)

                rec_loss_value = 0.0
                if rec_loss_func is not None:
                    if not isinstance(rec_loss_func, (list, tuple)):
                        rec_loss_func = [rec_loss_func]
                    for rec_loss in rec_loss_func:
                        rec_loss_value += rec_loss(fake_data, real_data)

                    gen_loss = (1.0 - rec_loss_weight) * adv_loss + rec_loss_weight * rec_loss_value

                else:
                    gen_loss = adv_loss
                test_discr_loss_meter.update(discr_loss.item(), real_data.size(0))
                test_discr_loss_meter.update(gen_loss.item(), real_data.size(0))
                test_adv_loss_meter.update(adv_loss.item(), real_data.size(0))
                if rec_loss_func is not None:
                    test_rec_loss_val_meter.update(rec_loss_value.item(), real_data.size(0))
                test_gen_loss_meter.update(gen_loss.item(), real_data.size(0))

                progress_bar.set_postfix(
                    ordered_dict={
                        "gen_loss": test_gen_loss_meter.avg,
                        "adv_loss": test_adv_loss_meter.avg,
                        "rec_loss": test_rec_loss_val_meter.avg,
                        "discr_loss": test_discr_loss_meter.avg,
                    }
                )

            if test_gen_loss_meter.avg < test_gen_best_loss or epoch == 0:
                test_gen_best_loss = test_gen_loss_meter.avg
                torch.save(generator.state_dict(), Path(savepath) / Path("test_gen_best_model.pth"))

            if test_rec_loss_val_meter.avg < test_rec_best_loss or epoch == 0:
                rec_best_loss = test_rec_loss_val_meter.avg
                torch.save(generator.state_dict(), Path(savepath) / Path("test_rec_best_model.pth"))

            if test_discr_loss_meter.avg < test_discr_best_loss or epoch == 0:
                discr_best_loss = test_discr_loss_meter.avg
                torch.save(discriminator.state_dict(), Path(savepath) / Path("test_discriminator_best_model.pth"))

            writer.add_scalar("Gen_test_loss", test_gen_loss_meter.avg, epoch + 1)
            writer.add_scalar("Gen GAN test loss", test_adv_loss_meter.avg, epoch + 1)
            writer.add_scalar("Gen rec test loss", test_rec_loss_val_meter.avg, epoch + 1)
            writer.add_scalar("Discriminator test loss", test_discr_loss_meter.avg, epoch + 1)


def train_unrolled(
    model,
    dataloader,
    epochs,
    optimizer,
    savepath,
    loss_fn: Callable = F.mse_loss,
    detach_every_k: int = 4,
    test_loader=None,
    device=None,
    lossweights=1.0,
    scheduler=None,
    iterations=None,
):
    """The dataloader must return a sequence of tensors at different successive times: t_0, ..., t_n
    -> shape b t_n c h w
    """

    os.makedirs(savepath, exist_ok=True)

    writer = SummaryWriter(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)

    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):

        model.train()
        loss_meter = AverageMeter()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            T_unroll = inputs.shape[1]

            loss = 0.0
            preds = inputs[:, 0, ...]
            loss += loss_fn(preds, targets[:, 0, ...])

            # threshold = math.around(T_unroll * epoch/epochs)
            threshold = int(T_unroll * epoch / epochs)

            for t in range(1, T_unroll):

                if t > threshold:
                    preds = model(inputs[:, t, ...])
                    loss += loss_fn(preds, targets[:, t, ...])
                else:
                    preds = model(preds)
                    if preds.dim() == 5 and preds.shape[1] == 1:
                        preds = preds.squeeze(1)

                    targets_t = targets[:, t, ...]
                    # targets_t = targets[:, step, ...]

                    loss += loss_fn(preds, targets_t)

                    # Truncated gradient: detach every k steps
                    if (t % detach_every_k) == 0:
                        preds = preds.detach()

            loss = loss / T_unroll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), inputs.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

            if i >= iterations:
                break

        avg_loss = loss_meter.avg

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / Path("best_model.pth"))

        writer.add_scalar("loss", avg_loss, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if test_loader is not None:

            with torch.no_grad():
                valoss_meter = AverageMeter()
                # Test step: set the model to inference mode
                model.eval()
                progress_bar = tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=True, total=len(test_loader)
                )

                for i, (inputs, targets) in enumerate(progress_bar):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    T_unroll = inputs.shape[1]
                    valoss = 0.0
                    preds = inputs[:, 0, ...]

                    for t in range(1, T_unroll):
                        preds = model(preds)  # Predict u_{t}

                        if preds.dim() == 5 and preds.shape[1] == 1:
                            preds = preds.squeeze(1)

                        targets_t = targets[:, t, ...]

                        valoss += loss_fn(preds, targets_t)

                    valoss = valoss / T_unroll

                    valoss_meter.update(valoss.item(), inputs.size(0))

            progress_bar.set_postfix(loss=valoss_meter.avg)

            avg_valoss = valoss_meter.avg

            if avg_valoss < best_val_loss or epoch == 0:
                best_val_loss = avg_valoss
                torch.save(model.state_dict(), Path(savepath) / Path("best_val_model.pth"))

            writer.add_scalar("val_loss", avg_valoss, epoch + 1)


def train_new_unrolled(
    model,
    dataloader,
    epochs,
    optimizer,
    savepath,
    loss_fn: Callable = F.mse_loss,
    detach_every_k: int = 4,
    test_loader=None,
    device=None,
    lossweights=1.0,
    scheduler=None,
    iterations=None,
):
    """
    Dataloader must return (inputs, targets) with shapes:
        inputs:  [B, T, C, H, W]
        targets: [B, T, C, H, W]
    where typically targets[:, t] ~ u_{t+1} and inputs[:, t] ~ u_{t}.
    """

    os.makedirs(savepath, exist_ok=True)
    writer = SummaryWriter(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)

    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):

        model.train()
        loss_meter = AverageMeter()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            B, T_unroll, C, H, W = inputs.shape
            threshold = int(T_unroll * epoch / epochs)

            loss = 0.0

            preds = model(inputs[:, 0, ...])
            if preds.dim() == 5 and preds.shape[1] == 1:
                preds = preds.squeeze(1)

            loss += loss_fn(preds, targets[:, 0, ...])

            for t in range(1, T_unroll):

                if t > threshold:
                    # TEACHER FORCING:
                    preds = model(inputs[:, t, ...])

                else:
                    # AUTOREGRESSIVE:
                    preds = model(preds)

                if preds.dim() == 5 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)

                targets_t = targets[:, t, ...]
                loss += loss_fn(preds, targets_t)

                if (t % detach_every_k) == 0:
                    preds = preds.detach()

            loss = loss / T_unroll

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), inputs.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

            if i >= iterations:
                break

        avg_loss = loss_meter.avg

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / "best_model.pth")

        writer.add_scalar("loss", avg_loss, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if test_loader is not None:
            with torch.no_grad():
                valoss_meter = AverageMeter()
                model.eval()

                progress_bar = tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=True, total=len(test_loader)
                )

                for i, (inputs, targets) in enumerate(progress_bar):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    B, T_unroll, C, H, W = inputs.shape

                    valoss = 0.0

                    preds = model(inputs[:, 0, ...])
                    if preds.dim() == 5 and preds.shape[1] == 1:
                        preds = preds.squeeze(1)
                    valoss += loss_fn(preds, targets[:, 0, ...])

                    for t in range(1, T_unroll):
                        preds = model(preds)

                        if preds.dim() == 5 and preds.shape[1] == 1:
                            preds = preds.squeeze(1)

                        targets_t = targets[:, t, ...]
                        valoss += loss_fn(preds, targets_t)

                    valoss = valoss / T_unroll
                    valoss_meter.update(valoss.item(), inputs.size(0))

                progress_bar.set_postfix(loss=valoss_meter.avg)
                avg_valoss = valoss_meter.avg

                if avg_valoss < best_val_loss or epoch == 0:
                    best_val_loss = avg_valoss
                    torch.save(model.state_dict(), Path(savepath) / "best_val_model.pth")

                writer.add_scalar("val_loss", avg_valoss, epoch + 1)


def train_unrolled_lin_tf(
    model,
    dataloader,
    epochs,
    optimizer,
    savepath,
    loss_fn=F.mse_loss,
    detach_every_k=4,
    test_loader=None,
    device=None,
    scheduler=None,
    iterations=None,
):
    os.makedirs(savepath, exist_ok=True)
    writer = SummaryWriter(savepath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)

    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):
        model.train()

        # tf_prob = 1.0 - epoch / max(1, (epochs - 1))
        tf_prob = max(0.1, 1.0 - epoch / (epochs - 1))

        loss_meter = AverageMeter()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            B, T, C, H, W = inputs.shape

            x = inputs[:, 0]
            loss = 0.0

            for t in range(T):
                pred = model(x)
                if pred.dim() == 5 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)

                loss = loss + loss_fn(pred, targets[:, t])

                if t < T - 1:
                    use_teacher = torch.rand((), device=device) < tf_prob
                    x = inputs[:, t + 1] if use_teacher else pred

                    if detach_every_k and ((t + 1) % detach_every_k == 0):
                        x = x.detach()

            loss = loss / T

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_meter.update(loss.item(), B)
            progress_bar.set_postfix(
                loss=f"{loss_meter.avg:.3e}", tf_prob=f"{tf_prob:.3f}", lr=optimizer.param_groups[0]["lr"]
            )

            if i >= iterations:
                break

        avg_loss = loss_meter.avg

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / "best_model.pth")

        writer.add_scalar("loss", avg_loss, epoch + 1)
        writer.add_scalar("tf_prob", tf_prob, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if test_loader is not None:
            with torch.no_grad():
                valoss_meter = AverageMeter()
                model.eval()

                progress_bar = tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=True, total=len(test_loader)
                )

                for i, (inputs, targets) in enumerate(progress_bar):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    B, T_unroll, C, H, W = inputs.shape

                    valoss = 0.0

                    preds = model(inputs[:, 0, ...])
                    if preds.dim() == 5 and preds.shape[1] == 1:
                        preds = preds.squeeze(1)
                    valoss += loss_fn(preds, targets[:, 0, ...])

                    for t in range(1, T_unroll):
                        preds = model(preds)

                        if preds.dim() == 5 and preds.shape[1] == 1:
                            preds = preds.squeeze(1)

                        targets_t = targets[:, t, ...]
                        valoss += loss_fn(preds, targets_t)

                    valoss = valoss / T_unroll
                    valoss_meter.update(valoss.item(), inputs.size(0))

                progress_bar.set_postfix(loss=valoss_meter.avg)
                avg_valoss = valoss_meter.avg

                if avg_valoss < best_val_loss or epoch == 0:
                    best_val_loss = avg_valoss
                    torch.save(model.state_dict(), Path(savepath) / "best_val_model.pth")

                writer.add_scalar("val_loss", avg_valoss, epoch + 1)


def CoDANO_training(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    savepath: Union[str, Path],
    loss_fn: Callable = F.mse_loss,
    loss_method: str = "physical",
    dt: Union[float, None] = None,
    detach_every_k: int = 4,
    test_loader: Union[None, torch.utils.data.DataLoader] = None,
    device: Union[torch.device, None] = None,
    lossweights: Union[Sequence[float], None] = None,
    scheduler: Union[None,] = None,
    iterations: Union[None, int] = None,
):
    """Train a model using the CoDANO method.
    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training. It must return a tuple of (inputs, static_field).
        The inputs are the sequence of tensors at different successive times: t_0, ..., t_n
        -> shape b t_n c h w, where c is the number of variables (e.g. velocity, pressure, etc.) and h, w are the spatial dimensions.
        The static_field is the static field to use for training. It must be the same spatial shape as the inputs.
    epochs : int
        The number of epochs to train for.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    savepath : str
        The path to save the model.
    loss_fn : Callable
        The loss function to use for training.
    loss_method : str
        The method to use for loss calculation. "physical", "data" or "both".
    dt : float
        The time step size. Must be set if loss_method == 'physical' or 'both'.
    detach_every_k : int
        The number of steps to detach the gradient.
    test_loader : torch.utils.data.DataLoader
        The dataloader to use for testing.
    device : torch.device
        The device to use for training. If None, will use the first available GPU or CPU.
    lossweights : float
        The weights to use for physical and data loss. If None, will use 1.0 for both.
    scheduler : torch.optim.lr_scheduler
        The scheduler to use for training.
    iterations : int
        The number of iterations to train for. If None, will use the length of the dataloader.
    """

    os.makedirs(savepath, exist_ok=True)

    writer = SummaryWriter(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)

    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    if lossweights is None:
        lossweights = [1.0, 1.0]

    for epoch in range(epochs):

        model.train()
        loss_meter = AverageMeter()
        physical_loss_meter = AverageMeter()
        data_loss_meter = AverageMeter()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (static_field, inputs) in enumerate(progress_bar):
            inputs = inputs.to(device)
            static_field = static_field.to(device)
            T_unroll = inputs.shape[1]
            loss = 0.0
            physical_loss = 0.0
            data_loss = 0.0
            previous_preds = inputs[:, 0, ...]
            for t in range(1, T_unroll):

                preds, coords = model(
                    previous_preds, static_channel=static_field, return_coords=True
                )  # Predict u_{t} from u{t-1}
                # Ground truth
                targets = inputs[:, t, ...]

                # Accumulate loss
                if loss_method == "physical" or loss_method == "both":
                    physical_loss += diffusion_loss(
                        preds,
                        previous_preds,
                        dt=dt,
                        diffusivity=static_field,
                        x_coords=coords[0],
                        y_coords=coords[1],
                        time_derivative="fd",
                    )
                if loss_method == "data" or loss_method == "both":
                    data_loss += loss_fn(preds, targets)

                loss += lossweights[0] * data_loss + lossweights[1] * physical_loss

                # Truncated gradient: detach every k steps
                if (t % detach_every_k) == 0:
                    preds = preds.detach()

                previous_preds = preds

            loss = loss / (T_unroll - 1)
            physical_loss = physical_loss / (T_unroll - 1)
            data_loss = data_loss / (T_unroll - 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            loss_meter.update(loss.item(), inputs.size(0))
            physical_loss_meter.update(physical_loss.item(), inputs.size(0))
            data_loss_meter.update(data_loss.item(), inputs.size(0))

            progress_bar.set_postfix(
                loss=loss_meter.avg,
                data_loss=data_loss_meter.avg,
                physical_loss=physical_loss_meter.avg,
                lr=optimizer.param_groups[0]["lr"],
            )

            if i >= iterations:
                break

            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        avg_loss = loss_meter.avg

        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / Path("best_model.pth"))

        writer.add_scalar("loss", avg_loss, epoch + 1)
        writer.add_scalar("data_loss", data_loss_meter.avg, epoch + 1)
        writer.add_scalar("physical_loss", physical_loss_meter.avg, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if test_loader is not None:

            with torch.no_grad():
                valoss_meter = AverageMeter()
                physical_loss_meter = AverageMeter()
                data_loss_meter = AverageMeter()
                # Test step: set the model to inference mode
                model.eval()
                progress_bar = tqdm(
                    test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=True, total=len(test_loader)
                )

                for i, (static_field, inputs) in enumerate(progress_bar):
                    inputs = inputs.to(device)
                    static_field = static_field.to(device)
                    T_unroll = inputs.shape[1]
                    valoss = 0.0
                    physical_loss = 0.0
                    data_loss = 0.0
                    previous_preds = inputs[:, 0, ...]
                    for t in range(1, T_unroll):
                        preds, coords = model(previous_preds, static_channel=static_field, return_coords=True)
                        targets = inputs[:, t, ...]
                        if loss_method == "physical" or loss_method == "both":
                            physical_loss += diffusion_loss(
                                preds,
                                previous_preds,
                                dt=dt,
                                diffusivity=static_field,
                                x_coords=coords[0],
                                y_coords=coords[1],
                                time_derivative="fd",
                            )
                        if loss_method == "data" or loss_method == "both":
                            data_loss += loss_fn(preds, targets)
                        valoss += lossweights[0] * data_loss + lossweights[1] * physical_loss
                        previous_preds = preds

                    valoss = valoss / (T_unroll - 1)
                    physical_loss = physical_loss / (T_unroll - 1)
                    data_loss = data_loss / (T_unroll - 1)

                    valoss_meter.update(valoss.item(), inputs.size(0))
                    physical_loss_meter.update(physical_loss.item(), inputs.size(0))
                    data_loss_meter.update(data_loss.item(), inputs.size(0))

                progress_bar.set_postfix(
                    loss=valoss_meter.avg,
                    data_loss=data_loss_meter.avg,
                    physical_loss=physical_loss_meter.avg,
                )

                avg_valoss = valoss_meter.avg

                if avg_valoss < best_val_loss or epoch == 0:
                    best_val_loss = avg_valoss
                    torch.save(model.state_dict(), Path(savepath) / Path("best_val_model.pth"))

                writer.add_scalar("val_loss", avg_valoss, epoch + 1)
                writer.add_scalar("val_data_loss", data_loss_meter.avg, epoch + 1)
                writer.add_scalar("val_physical_loss", physical_loss_meter.avg, epoch + 1)

    writer.close()


def train_deeponet(
    model,
    dataloader,
    epochs,
    optimizer,
    loss_fn,
    savepath,
    test_loader=None,
    device=None,
    lossweights=1.0,
    scheduler=None,
    iterations=None,
):
    os.makedirs(savepath, exist_ok=True)
    writer = SummaryWriter(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)

    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", total=iterations)

        for i, (inputs, timestep, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            timestep = timestep.to(device)

            loss = train_step_deeponet(model, inputs, timestep, targets, loss_fn, optimizer, lossweights)

            loss_meter.update(loss.item(), inputs.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

            if i >= iterations:
                break

        avg_loss = loss_meter.avg

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / "best_model.pth")

        writer.add_scalar("loss", avg_loss, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # Validation
        if test_loader is not None:
            model.eval()
            val_loss_meter = AverageMeter()
            val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", total=len(test_loader))

            with torch.no_grad():
                for inputs, timestep, targets in val_bar:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    timestep = timestep.to(device)

                    val_loss = test_step_deeponet(model, inputs, timestep, targets, loss_fn, lossweights)
                    val_loss_meter.update(val_loss.item(), inputs.size(0))

            avg_val_loss = val_loss_meter.avg
            val_bar.set_postfix(val_loss=avg_val_loss)

            if avg_val_loss < best_val_loss or epoch == 0:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), Path(savepath) / "best_val_model.pth")

            writer.add_scalar("val_loss", avg_val_loss, epoch + 1)

    writer.close()


def train_step_deeponet(
    model,
    inputs,
    times,
    targets,
    loss_fn,
    optimizer,
    lossweights=1.0,
    gamma=5.0,
    early_t_max=1.0,
    time_weighted=True,
):

    device = inputs.device
    inputs = inputs.to(device)
    times = times.to(device)
    targets = targets.to(device)

    preds = model(inputs, times)

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

    if not isinstance(loss_fn, (tuple, list)):
        loss_fn = [loss_fn]

    if not isinstance(lossweights, (tuple, list)):
        lossweights = [lossweights] * len(loss_fn)

    t_flat = times.view(-1)
    time_w = torch.ones_like(t_flat)
    time_w[t_flat <= early_t_max] = gamma
    time_w = time_w.view(-1, 1, 1, 1)

    loss = 0.0
    for i, loss_function in enumerate(loss_fn):
        loss += loss_function(preds, targets) * lossweights[i]

    # for lf, lw in zip(loss_fn, lossweights):
    #     base = lf(preds, targets)

    #     if time_weighted and isinstance(base, torch.Tensor) and base.shape == preds.shape:
    #         loss_term = (base * time_w).mean()
    #     else:
    #         loss_term = base

    #     loss += loss_term * lw

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test_step_deeponet(
    model, inputs, times, targets, loss_fn, lossweights=1.0, gamma=5.0, early_t_max=1.0, time_weighted=True
):

    device = inputs.device
    inputs = inputs.to(device)
    times = times.to(device)
    targets = targets.to(device)

    preds = model(inputs, times)

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, targets {targets.shape}")

    # if not isinstance(loss_fn, (tuple, list)):
    #     loss_fn = [loss_fn]

    # if not isinstance(lossweights, (tuple, list)):
    #     lossweights = [lossweights] * len(loss_fn)

    if isinstance(loss_fn, (list, tuple)):
        if not isinstance(lossweights, (list, tuple)):
            lossweights = [lossweights] * len(loss_fn)
        loss = sum(loss_fn[i](preds, targets) * lossweights[i] for i in range(len(loss_fn)))
    else:
        loss = loss_fn(preds, targets) * lossweights

    return loss

    # t_flat = times.view(-1)
    # time_w = torch.ones_like(t_flat)
    # time_w[t_flat <= early_t_max] = gamma
    # time_w = time_w.view(-1, 1, 1, 1)

    # total_loss = 0.0
    # for lf, lw in zip(loss_fn, lossweights):
    #     base = lf(preds, targets)

    #     if time_weighted and isinstance(base, torch.Tensor) and base.shape == preds.shape:
    #         loss_term = (base * time_w).mean()
    #     else:
    #         loss_term = base

    #     total_loss += loss_term * lw

    # return total_loss


def train_warmup(
    model,
    dataloader,
    epochs,
    optimizer,
    loss_fn,
    savepath,
    test_loader=None,
    device=None,
    lossweights=1.0,
    scheduler=None,
    iterations=None,
):
    """
    Training loop variant for AE using Cosine_WarmUp.
    Stepped smoothly each batch with fractional epoch increments.
    """

    os.makedirs(savepath, exist_ok=True)
    writer = SummaryWriter(savepath)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
    model.to(device)

    best_loss = float("inf")
    best_val_loss = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):
        model.train()
        loss_meter = AverageMeter()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=True, total=iterations)

        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = train_step(model, inputs, targets, loss_fn, optimizer, lossweights=lossweights)

            if scheduler is not None:
                scheduler.step()

            loss_meter.update(loss.item(), inputs.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

            if i >= iterations:
                break

        avg_loss = loss_meter.avg

        if avg_loss < best_loss or epoch == 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), Path(savepath) / "best_model.pth")

        writer.add_scalar("loss", avg_loss, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        if test_loader is not None:
            with torch.no_grad():
                model.eval()
                val_meter = AverageMeter()
                for vinputs, vtargets in tqdm(test_loader, desc=f"[Val epoch {epoch+1}]"):
                    vinputs, vtargets = vinputs.to(device), vtargets.to(device)
                    vloss = test_step(model, vinputs, vtargets, loss_fn, lossweights=lossweights)
                    val_meter.update(vloss.item(), vinputs.size(0))

            avg_val = val_meter.avg
            if avg_val < best_val_loss or epoch == 0:
                best_val_loss = avg_val
                torch.save(model.state_dict(), Path(savepath) / "best_val_model.pth")
            writer.add_scalar("val_loss", avg_val, epoch + 1)


def train_time_unrolled(
    model,
    dataloader,
    epochs: int,
    optimizer,
    savepath,
    loss_fn: Callable = F.mse_loss,
    detach_every_k: int = 4,
    test_loader=None,
    device = None,
    scheduler = None,
    iterations = None,
    teacher_forcing: str = "linear",  
    grad_clip = 1.0,
):
    """
    Expects dataloader batch from UnrolledH5DatasetWithTime:
      (x_in, x_out, t_in, t_out)   OR   (x_in, x_out, t_in, t_out, dt)

    Shapes:
      x_in :  (B, T, C, H, W)   where T = T_unroll-1
      x_out:  (B, T, C, H, W)
      t_in :  (B, T) or (B, T, 1)  (time associated with x_in steps)
    Model API:
      model(x, t) where x=(B,C,H,W), t=(B,1) (or (B,))
    """

    os.makedirs(savepath, exist_ok=True)
    writer = SummaryWriter(savepath)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_train = float("inf")
    best_val = float("inf")

    if iterations is None:
        iterations = len(dataloader)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [train]", total=iterations, leave=True)

        for it, batch in enumerate(pbar):
            if len(batch) == 5:
                x_in, x_out, t_in, t_out, dt = batch  
            else:
                x_in, x_out, t_in, t_out = batch

            x_in = x_in.to(device)    
            x_out = x_out.to(device)   
            t_in = t_in.to(device)     

            B, T, C, H, W = x_in.shape

            if t_in.dim() == 2:
                t_in = t_in.unsqueeze(-1)

            if teacher_forcing == "linear":
                tf_steps = int(T * (1.0 - epoch / max(1, epochs - 1)))
            elif teacher_forcing == "none":
                tf_steps = 0
            else:
                raise ValueError(f"teacher_forcing must be 'linear' or 'none', got {teacher_forcing}")

            optimizer.zero_grad(set_to_none=True)

            loss = 0.0

            # step 0 always uses ground truth input
            pred = model(x_in[:, 0], t_in[:, 0])  
            loss = loss + loss_fn(pred, x_out[:, 0])

            # rollout
            for k in range(1, T):
                if k < tf_steps:
                    # teacher forcing: use true state at this step
                    pred = model(x_in[:, k], t_in[:, k])
                else:
                    # autoregressive: feed previous prediction, but advance time
                    pred = model(pred, t_in[:, k])

                loss = loss + loss_fn(pred, x_out[:, k])

                if detach_every_k is not None and detach_every_k > 0 and (k % detach_every_k) == 0:
                    pred = pred.detach()

            loss = loss / T

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = x_in.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            avg = running_loss / max(1, n_samples)
            pbar.set_postfix(loss=avg, tf_steps=tf_steps, lr=optimizer.param_groups[0]["lr"])

            if it + 1 >= iterations:
                break

        train_loss = running_loss / max(1, n_samples)
        writer.add_scalar("loss/train", train_loss, epoch + 1)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step()

        # save best train
        if train_loss < best_train:
            best_train = train_loss
            torch.save(model.state_dict(), Path(savepath) / "best_train_model.pth")

        # validation (pure autoregressive rollout)
        if test_loader is not None:
            model.eval()
            val_running = 0.0
            val_n = 0

            with torch.no_grad():
                vpbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=True)

                for batch in vpbar:
                    if len(batch) == 5:
                        x_in, x_out, t_in, t_out, dt = batch
                    else:
                        x_in, x_out, t_in, t_out = batch

                    x_in = x_in.to(device)
                    x_out = x_out.to(device)
                    t_in = t_in.to(device)

                    B, T, C, H, W = x_in.shape
                    if t_in.dim() == 2:
                        t_in = t_in.unsqueeze(-1)

                    vloss = 0.0

                    pred = model(x_in[:, 0], t_in[:, 0])
                    vloss = vloss + loss_fn(pred, x_out[:, 0])

                    for k in range(1, T):
                        pred = model(pred, t_in[:, k])  # AR
                        vloss = vloss + loss_fn(pred, x_out[:, k])

                    vloss = vloss / T

                    bs = x_in.size(0)
                    val_running += vloss.item() * bs
                    val_n += bs
                    vpbar.set_postfix(val_loss=val_running / max(1, val_n))

            val_loss = val_running / max(1, val_n)
            writer.add_scalar("loss/val", val_loss, epoch + 1)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), Path(savepath) / "best_val_model.pth")

    writer.close()


#PdeBench

# def train_one_epoch(model, loader, optimizer, device, initial_step=10, t_train=101):
#     model.train()
#     loss_fn = nn.MSELoss(reduction="mean")

#     step_loss_sum = 0.0
#     full_loss_sum = 0.0
#     n_batches = 0

#     # pbar = tqdm(loader, desc="Train", leave=False)

#     for xx, yy, grid in loader:
#         xx = xx.to(device)      # [B,X,Y,initial_step,C]
#         yy = yy.to(device)      # [B,X,Y,T,C]
#         grid = grid.to(device)  # [B,X,Y,2]

#         B, X, Y, t_init, C = xx.shape
#         T = yy.shape[-2]
#         t_train_eff = min(t_train, T)

#         loss = 0.0
#         pred = yy[..., :initial_step, :]  

#         # autoregressive rollout loss
#         for t in range(initial_step, t_train_eff):
#             inp = xx.reshape(B, X, Y, t_init * C)     # [B,X,Y,initial_step*C]
#             target = yy[..., t:t+1, :]                # [B,X,Y,1,C]
#             im = model(inp, grid)                     # [B,X,Y,1,C]

#             loss = loss + loss_fn(im.reshape(B, -1), target.reshape(B, -1))

#             pred = torch.cat((pred, im), dim=-2)      # append along time axis
#             # xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # slide window
#             xx = torch.cat((xx[..., 1:, :], im.detach()), dim=-2)

#         optimizer.zero_grad(set_to_none=True)
#         # loss.backward()
#         # optimizer.step()
#         from torch.cuda.amp import autocast, GradScaler
#         scaler = GradScaler()
#         # optimizer.zero_grad(set_to_none=True)

#         with autocast():
#             im = model(inp, grid)
#             loss = loss + loss_fn(im.reshape(B, -1), target.reshape(B, -1))

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # "full" loss over [0:t_train_eff]
#         with torch.no_grad():
#             yy_cut = yy[..., :t_train_eff, :]
#             full = loss_fn(pred.reshape(B, -1), yy_cut.reshape(B, -1))

#         step_loss_sum += float(loss.detach())
#         full_loss_sum += float(full.detach())
#         # pbar.set_postfix(loss=float(loss.detach()))

#     n_batches = len(loader)

#     return step_loss_sum / max(n_batches, 1), full_loss_sum / max(n_batches, 1)

def train_one_epoch(model, loader, optimizer, device, initial_step=10, t_train=101):
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")

    step_loss_sum = 0.0
    full_loss_sum = 0.0

    for xx, yy, grid in loader:
        xx = xx.to(device)     
        yy = yy.to(device)      
        grid = grid.to(device)  

        B, X, Y, t_init, C = xx.shape
        T = yy.shape[-2]
        t_train_eff = min(t_train, T)

        loss = 0.0
        pred = yy[..., :initial_step, :]  

        for t in range(initial_step, t_train_eff):
            inp = xx.reshape(B, X, Y, t_init * C)
            target = yy[..., t:t+1, :]

            im = model(inp, grid)
            loss = loss + loss_fn(im.reshape(B, -1), target.reshape(B, -1))

            pred = torch.cat((pred, im), dim=-2)
            xx = torch.cat((xx[..., 1:, :], im), dim=-2)  

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            yy_cut = yy[..., :t_train_eff, :]
            full = loss_fn(pred.reshape(B, -1), yy_cut.reshape(B, -1))

        step_loss_sum += float(loss.detach())
        full_loss_sum += float(full.detach())

    n_batches = len(loader)
    return step_loss_sum / max(n_batches, 1), full_loss_sum / max(n_batches, 1)


from contextlib import nullcontext

def train_one_epoch_tbptt(
    model, loader, optimizer, device,
    initial_step=10, t_train=101,
    tbptt_k=5,                 
    amp=True,                
    scaler=None,               
):
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")

    step_loss_sum = 0.0
    full_loss_sum = 0.0
    n_batches = 0

    autocast = torch.cuda.amp.autocast if (amp and device.type == "cuda") else nullcontext

    for xx, yy, grid in loader:
        xx = xx.to(device)      
        yy = yy.to(device)      
        grid = grid.to(device)  

        B, X, Y, t_init, C = xx.shape
        T = yy.shape[-2]
        t_train_eff = min(t_train, T)

        pred = yy[..., :initial_step, :] 
        full_loss = None

        optimizer.zero_grad(set_to_none=True)
        chunk_loss = 0.0
        chunk_count = 0

        for t in range(initial_step, t_train_eff):
            inp = xx.reshape(B, X, Y, t_init * C)
            target = yy[..., t:t+1, :]
            with autocast():
                with torch.cuda.amp.autocast(enabled=False):
                    im = model(inp.float(), grid.float())  
                loss_t = loss_fn(im.reshape(B, -1), target.reshape(B, -1))
            pred = torch.cat((pred, im), dim=-2)
            xx = torch.cat((xx[..., 1:, :], im), dim=-2)

            chunk_loss = chunk_loss + loss_t
            chunk_count += 1

            if chunk_count == tbptt_k or t == (t_train_eff - 1):
                if scaler is None:
                    chunk_loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(chunk_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                optimizer.zero_grad(set_to_none=True)

                xx = xx.detach()
                pred = pred.detach()

                chunk_loss = 0.0
                chunk_count = 0

        # full loss for reporting (no grad)
        with torch.no_grad():
            yy_cut = yy[..., :t_train_eff, :]
            full_loss = loss_fn(pred.reshape(B, -1), yy_cut.reshape(B, -1))

        step_loss_sum += float(full_loss) 
        full_loss_sum += float(full_loss)
        n_batches += 1

    return step_loss_sum / max(n_batches, 1), full_loss_sum / max(n_batches, 1)

@torch.no_grad()
def validate(model, loader, device, initial_step=10, t_train=101):
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")

    step_loss_sum = 0.0
    full_loss_sum = 0.0
    n_batches = 0

    for xx, yy, grid in loader:
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)

        B, X, Y, t_init, C = xx.shape
        T = yy.shape[-2]
        t_train_eff = min(t_train, T)

        loss = 0.0
        pred = yy[..., :initial_step, :]

        for t in range(initial_step, T):
            inp = xx.reshape(B, X, Y, t_init * C)
            target = yy[..., t:t+1, :]
            
            im = model(inp, grid)

            loss = loss + loss_fn(im.reshape(B, -1), target.reshape(B, -1))
            pred = torch.cat((pred, im), dim=-2)
            xx = torch.cat((xx[..., 1:, :], im), dim=-2)

        pred_cut = pred[..., initial_step:t_train_eff, :]
        yy_cut = yy[..., initial_step:t_train_eff, :]
        full = loss_fn(pred_cut.reshape(B, -1), yy_cut.reshape(B, -1))

        step_loss_sum += float(loss)
        full_loss_sum += float(full)
    n_batches = len(loader)

    return step_loss_sum / max(n_batches, 1), full_loss_sum / max(n_batches, 1)

from tqdm import trange

def fit_like_pdebench_tbptt(
    model, train_loader, val_loader, device,
    epochs=500, lr=1e-3, weight_decay=1e-4,
    scheduler_step=100, scheduler_gamma=0.5,
    model_update=10, initial_step=10, t_train=101,
    tbptt_k=5, amp=True,
    ckpt_path="ckpt.pt", save_history_path=None,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type=="cuda"))

    history = {"epoch": [], "train_full": [], "val_full": [], "lr": []}
    best_val, best_epoch = float("inf"), -1

    for ep in range(epochs):
        tr_step, tr_full = train_one_epoch_tbptt(
            model, train_loader, opt, device,
            initial_step=initial_step, t_train=t_train,
            tbptt_k=tbptt_k, amp=amp, scaler=scaler,
        )

        if ep % model_update == 0:
            va_step, va_full = validate(model, val_loader, device, initial_step=initial_step, t_train=t_train)

            history["epoch"].append(ep)
            history["train_full"].append(float(tr_full))
            history["val_full"].append(float(va_full))
            history["lr"].append(float(opt.param_groups[0]["lr"]))

            if va_full < best_val:
                best_val, best_epoch = va_full, ep
                torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(), "loss": best_val}, ckpt_path)

            if save_history_path is not None:
                import numpy as np
                np.savez(save_history_path, **{k: np.array(v) for k, v in history.items()})

        sch.step()

    return best_val, best_epoch, history

def fit_fno2d_pdebench(
    model,
    train_loader,
    val_loader,
    device,
    epochs=500,
    lr=1e-3,
    weight_decay=1e-4,
    scheduler_step=100,
    scheduler_gamma=0.5,
    model_update=10,
    initial_step=10,
    t_train=101,
    ckpt_path="2D_diff-react_FNO.pt",
    save_history_path=None,
):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    best_val = float("inf")
    best_epoch = -1

    history = {
        "epoch": [],
        "lr": [],
        "train_step": [],
        "train_full": [],
        "val_step": [],
        "val_full": [],
    }

    total_steps = epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    global_step = 0
    for ep in range(epochs):
        tr_step, tr_full = train_one_epoch(
            model, train_loader, optimizer, device,
            initial_step=initial_step, t_train=t_train
        )

        pbar.update(len(train_loader))
        global_step += len(train_loader)

        if ep % model_update == 0:
            va_step, va_full = validate(
                model, val_loader, device,
                initial_step=initial_step, t_train=t_train
            )

            cur_lr = optimizer.param_groups[0]["lr"]

            history["epoch"].append(ep)
            history["lr"].append(cur_lr)
            history["train_step"].append(tr_step)
            history["train_full"].append(tr_full)
            history["val_step"].append(va_step)
            history["val_full"].append(va_full)

            if va_full < best_val:
                best_val = va_full
                best_epoch = ep
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val,
                    },
                    ckpt_path,
                )

            pbar.set_postfix(
                ep=ep,
                lr=f"{cur_lr:.1e}",
                tr=f"{tr_full:.2e}",
                va=f"{va_full:.2e}",
                best=f"{best_val:.2e}",
            )

            if save_history_path is not None:
                np.savez(
                    str(save_history_path),
                    epoch=np.array(history["epoch"]),
                    lr=np.array(history["lr"]),
                    train_step=np.array(history["train_step"]),
                    train_full=np.array(history["train_full"]),
                    val_step=np.array(history["val_step"]),
                    val_full=np.array(history["val_full"]),
                )

        scheduler.step()

    pbar.close()
    return best_val, best_epoch, history

from typing import Callable, Optional, Dict
from tqdm.auto import tqdm


def _rollout_loss_runA(
    model,
    xx_init,    
    yy_full,    
    grid,       
    *,
    T_unroll: int,
    initial_step: int,
    loss_fn: Callable = F.mse_loss,
    use_nrmse: bool = False,
    eps: float = 1e-12,
    teacher_forcing_prob: float = 0.0,
    detach_every_k: int = 0,
):
    B, X, Y, init, C = xx_init.shape
    T = yy_full.shape[-2]
    assert init == initial_step, f"xx_init has init={init}, but initial_step={initial_step}"

    t0 = initial_step
    t1 = min(T, t0 + T_unroll)
    steps = t1 - t0
    if steps <= 0:
        raise ValueError(f"No steps to unroll: initial_step={initial_step}, T={T}, T_unroll={T_unroll}")

    state = xx_init 

    preds = []
    targets = []

    for k, t in enumerate(range(t0, t1)):
        inp = state.reshape(B, X, Y, init * C)   
        im = model(inp, grid)                    

        if im.ndim == 4:                          
            im = im.unsqueeze(-2)
        if im.shape[-2] != 1:
            raise RuntimeError(f"Model must output [B,X,Y,1,C], got {im.shape}")

        preds.append(im)                         
        targets.append(yy_full[..., t:t+1, :])   

        # scheduled teacher forcing
        if teacher_forcing_prob > 0.0 and torch.rand(()) < teacher_forcing_prob:
            state = yy_full[..., (t - init + 1):(t + 1), :].detach()  
        else:
            state = torch.cat([state[..., 1:, :], im], dim=-2)

        if detach_every_k and ((k + 1) % detach_every_k == 0):
            state = state.detach()

    pred = torch.cat(preds, dim=-2)    
    targ = torch.cat(targets, dim=-2)  

    if not use_nrmse:
        return loss_fn(pred, targ)

    num = torch.mean((pred - targ) ** 2)
    den = torch.mean(targ ** 2)
    return torch.sqrt(num / (den + eps))


@torch.no_grad()
def validate_runA(
    model,
    loader,
    device,
    *,
    T_unroll: int,
    initial_step: int,
    loss_fn: Callable = F.mse_loss,
    use_nrmse: bool = False,
):
    model.eval()
    losses = []
    for xx_init, yy_full, grid in loader:
        xx_init = xx_init.to(device)
        yy_full = yy_full.to(device)
        grid = grid.to(device)

        loss = _rollout_loss_runA(
            model, xx_init, yy_full, grid,
            T_unroll=T_unroll,
            initial_step=initial_step,
            loss_fn=loss_fn,
            use_nrmse=use_nrmse,
            teacher_forcing_prob=0.0,  
            detach_every_k=0,
        )
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")

def tf_prob_sigmoid(ep, epochs, p_start, p_end, mid=0.35, sharp=12.0):
    if epochs <= 1:
        return p_end
    t = ep / (epochs - 1)             
    s = 1.0 / (1.0 + math.exp(sharp * (t - mid)))  
    return p_end + (p_start - p_end) * s

def train_runA_with_history(
    model,
    train_loader,
    epochs: int,
    optimizer,
    savepath,
    *,
    device=None,
    test_loader=None,
    scheduler=None,
    loss_fn: Callable = F.mse_loss,
    T_unroll: int = 20,
    initial_step: int = 1,                
    use_nrmse: bool = False,              
    detach_every_k: int = 0,              
    tf_prob_start: float = 0.0,
    tf_prob_end: float = 0.0,
    validate_every: int = 1,
    save_history_path: Optional[str | Path] = None,
):
    """
    Run-A style training:
      - dataloader yields (xx_init [B,X,Y,init,C], yy_full [B,X,Y,T,C], grid [B,X,Y,2])
      - optimize rollout loss over T_unroll steps starting at t=initial_step

    Returns:
      history dict with epoch/train/val/lr
    """
    savepath = Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history: Dict[str, list] = {"epoch": [], "train": [], "val": [], "lr": []}
    best_val = float("inf")

    epoch_bar = tqdm(range(epochs), desc="Epochs", dynamic_ncols=True)

    for ep in epoch_bar:
        model.train()

        # # linear schedule for teacher forcing probability
        # if epochs > 1:
        #     tf_prob = tf_prob_start + (tf_prob_end - tf_prob_start) * (ep / (epochs - 1))
        # else:
        #     tf_prob = tf_prob_end
        tf_prob = tf_prob_sigmoid(
            ep, epochs,
            tf_prob_start, tf_prob_end,
            mid=0.3, sharp=14.0
        )

        train_pbar = tqdm(train_loader, desc=f"Train {ep+1}/{epochs}", leave=False, dynamic_ncols=True)

        running = []
        for xx_init, yy_full, grid in train_pbar:
            xx_init = xx_init.to(device)
            yy_full = yy_full.to(device)
            grid = grid.to(device)

            loss = _rollout_loss_runA(
                model, xx_init, yy_full, grid,
                T_unroll=T_unroll,
                initial_step=initial_step,
                loss_fn=loss_fn,
                use_nrmse=use_nrmse,
                teacher_forcing_prob=tf_prob,
                detach_every_k=detach_every_k,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running.append(float(loss))
            train_pbar.set_postfix(
                loss=float(np.mean(running)),
                lr=float(optimizer.param_groups[0]["lr"]),
                tf=float(tf_prob),
            )

        train_loss = float(np.mean(running)) if running else float("nan")

        # # scheduler step
        # if scheduler is not None:
        #     # ReduceLROnPlateau expects a metric
        #     if hasattr(scheduler, "step") and scheduler.__class__.__name__ == "ReduceLROnPlateau":
        #         scheduler.step(train_loss)
        #     else:
        #         scheduler.step()

        # validation
        val_loss = None
        if test_loader is not None and ((ep + 1) % validate_every == 0):
            val_loss = validate_runA(
                model, test_loader, device,
                T_unroll=T_unroll,
                initial_step=initial_step,
                loss_fn=loss_fn,
                use_nrmse=use_nrmse,
            )
            if scheduler is not None and scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), savepath / "best_val_model.pth")

        torch.save(model.state_dict(), savepath / "last_model.pth")

        history["epoch"].append(ep + 1)
        history["train"].append(train_loss)
        history["val"].append(val_loss if val_loss is not None else np.nan)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if save_history_path is not None:
            save_history_path = Path(save_history_path)
            np.savez(
                save_history_path,
                epoch=np.array(history["epoch"]),
                train=np.array(history["train"]),
                val=np.array(history["val"]),
                lr=np.array(history["lr"]),
            )

        epoch_bar.set_postfix(
            train=train_loss,
            val=(val_loss if val_loss is not None else np.nan),
            lr=float(optimizer.param_groups[0]["lr"]),
        )

    return history


def train_one_epoch_ar_tbptt(
    model, loader, optimizer, device,
    initial_step=10, t_train=101, tbptt_k=5,
):
    model.train()
    loss_fn = nn.MSELoss(reduction="mean")

    epoch_loss_sum = 0.0
    epoch_steps = 0

    for xx, yy, grid in loader:
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)

        B, X, Y, t_init, C = xx.shape
        T = yy.shape[-2]
        t_train_eff = min(t_train, T)

        optimizer.zero_grad(set_to_none=True)

        chunk_loss = 0.0
        chunk_count = 0

        for t in range(initial_step, t_train_eff):
            inp = xx.reshape(B, X, Y, t_init * C)
            target = yy[..., t:t+1, :]

            im = model(inp, grid)
            if im.ndim == 4:
                im = im.unsqueeze(-2)

            loss_t = loss_fn(im.reshape(B, -1), target.reshape(B, -1))
            chunk_loss = chunk_loss + loss_t
            chunk_count += 1

            xx = torch.cat((xx[..., 1:, :], im), dim=-2)

            if (chunk_count == tbptt_k) or (t == t_train_eff - 1):
                (chunk_loss / chunk_count).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                xx = xx.detach()

                epoch_loss_sum += float(chunk_loss.detach())
                epoch_steps += chunk_count

                chunk_loss = 0.0
                chunk_count = 0

    return epoch_loss_sum / max(epoch_steps, 1)


def fit_with_warmup_tbptt(
    model, train_loader, val_loader, device,
    *,
    epochs=200,
    warmup_epochs=10,
    warmup_start_factor=0.1,
    initial_step=10,
    t_train=101,
    tbptt_k=5,
    lr=1e-3,
    weight_decay=1e-4,
    model_update=10,
    ckpt_path="best.pt",
    save_history_path=None,
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"epoch": [], "lr": [], "train": [], "val": []}
    best_val = float("inf")

    base_lr = lr
    start_lr = base_lr * warmup_start_factor

    scaler = None

    for ep in tqdm(range(epochs), desc="Epochs", dynamic_ncols=True):
        if warmup_epochs > 0 and ep < warmup_epochs:
            alpha = (ep + 1) / warmup_epochs
            lr_now = start_lr + alpha * (base_lr - start_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        tr = train_one_epoch_ar_tbptt(
            model, train_loader, optimizer, device,
            initial_step=initial_step, t_train=t_train, tbptt_k=tbptt_k
        )

        if ep % model_update == 0:
            va_step, va_full = validate(
                model, val_loader, device,
                initial_step=initial_step, t_train=t_train
            )
            va = va_full  

            cur_lr = optimizer.param_groups[0]["lr"]
            history["epoch"].append(ep)
            history["lr"].append(cur_lr)
            history["train"].append(tr)
            history["val"].append(va)

            if not (warmup_epochs > 0 and ep < warmup_epochs):
                plateau.step(va)

            if va < best_val:
                best_val = va
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_val,
                    },
                    ckpt_path,
                )

            if save_history_path is not None:
                np.savez(
                    str(save_history_path),
                    epoch=np.array(history["epoch"]),
                    lr=np.array(history["lr"]),
                    train=np.array(history["train"]),
                    val=np.array(history["val"]),
                )

    return history
