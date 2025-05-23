import os
from typing import Callable, Sequence, Union
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from .metrics import AverageMeter
from .loss import G_GANLoss, D_GANLoss, ZeroCenteredGradientPenalty, diffusion_loss

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F


def train_step(model, inputs, targets, loss_fn, optimizer, lossweights=1.0):
    preds = model(inputs)
    # residual = targets - preds
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
    # residual = targets - preds
    loss = 0.0
    if not (isinstance(lossweights, tuple) or isinstance(lossweights, list)):
        lossweights = [lossweights] * len(loss_fn)
    for i, loss_function in enumerate(loss_fn):
        # loss += loss_function(residual, torch.zeros_like(residual)) * lossweights[i]
        loss += loss_function(preds, targets) * lossweights[i]
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

            loss = train_step(model, inputs, targets, loss_fn, optimizer, lossweights=lossweights)

            # Update loss and accuracy
            loss_meter.update(loss.item(), inputs.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

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

                    valoss = test_step(model, inputs, targets, loss_fn, lossweights=lossweights)

                    # Update loss and accuracy
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

        for i, inputs in enumerate(progress_bar):
            inputs = inputs.to(device)
            T_unroll = inputs.shape[1]
            loss = 0.0
            preds = inputs[:, 0, ...]
            for t in range(1, T_unroll):
                preds = model(preds)  # Predict u_{t} from u{t-1}

                # Ground truth
                targets = inputs[:, t, ...]

                # Accumulate loss
                loss += loss_fn(preds, targets)

                # Truncated gradient: detach every k steps
                if (t % detach_every_k) == 0:
                    preds = preds.detach()

            loss = loss / T_unroll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            loss_meter.update(loss.item(), inputs.size(0))

            progress_bar.set_postfix(loss=loss_meter.avg, lr=optimizer.param_groups[0]["lr"])

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
                    T_unroll = inputs.shape[1]
                    valoss = 0.0
                    preds = inputs[:, 0, ...]
                    for t in range(1, T_unroll):
                        preds = model(preds)  # Predict u_{t}
                        # Ground truth
                        targets = inputs[:, t, ...]

                        # Accumulate loss
                        valoss += loss_fn(preds, targets)

                    valoss = test_step(model, inputs, targets, loss_fn, lossweights=lossweights)

                    # Update loss and accuracy
                    valoss_meter.update(valoss.item(), inputs.size(0))

            progress_bar.set_postfix(loss=valoss_meter.avg)

            avg_valoss = valoss_meter.avg

            if avg_valoss < best_val_loss or epoch == 0:
                best_val_loss = avg_valoss
                torch.save(model.state_dict(), Path(savepath) / Path("best_val_model.pth"))

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

        for i, (inputs, static_field) in enumerate(progress_bar):
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

                for i, (inputs, static_field) in enumerate(progress_bar):
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
