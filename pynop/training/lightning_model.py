import math
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os
import copy
import json
from typing import Any, Iterable
from torchmetrics import MeanMetric
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import dataclass, asdict
from ..core import add_noise
from ..core.loss import DWMSELoss, NormalizedTimeDerivativeMSE, SIGReg, preprocess_to_sigreg, SpectralLoss, SobolevLoss
from ..core.utils import sample_square_crop_boxes, apply_square_crop, make_inpainted_input

default_scheduler_config = [
    {
        "name": "ReduceLROnPlateau",
        "mode": "min",
        "patience": 10,
        "factor": 0.5,
        "monitor": "train_loss",
        "interval": "epoch",
        "frequency": 1,
    },
]

default_opt_config = {"optimizer": "AdamW", "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-08, "weight_decay": 0.01}


@dataclass
class TrainingConfig:
    dt: float = 1
    start_autoregressive: Any = None  # start o fthe autoregressive training
    final_autoregressive: Any = None  # epoch at which the number of autoregressive steps is maximum
    min_autoregressive_steps: int = 0  # minimum number of autoregressive steps (after start_autoregressive epochs)
    max_autoregressive_steps: int = 0  # maximum number of autoregressive steps
    detach_grad_steps: int = 4  # number of steps before detaching the gradient in autoregressive mode
    loss_fn: Any = torch.nn.MSELoss()
    derivative_loss_weight: float = 1.0
    loss_weights: Any = 1.0
    noise_level: float = 0  # no noise if 0
    time_normalization: float = 1
    n_slices: int = 2  # Number of slices in a single sample
    temporal_weighting: float = (
        1.2  # Increase loss with the timestep during autoregressive training: temporal_weighting**timestep
    )

    def to_json_dict(self):
        def serialize_value(v):
            # Handle lists or tuples recursively
            if isinstance(v, (list, tuple)):
                return [serialize_value(i) for i in v]

            # Handle classes or types
            if callable(v) or isinstance(v, type):
                return v.__name__ if hasattr(v, "__name__") else v.__class__.__name__

            # Handle custom class instances
            if hasattr(v, "__class__") and type(v).__module__ != "builtins":
                return v.__class__.__name__

            return v

        return {k: serialize_value(v) for k, v in asdict(self).items()}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=4)


# custom callback to begin the monitoring only after a certain number of epochs
class CurriculumCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)


class MultiStepNOModel(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        train_config=TrainingConfig(),
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.dt = train_config.dt
        self.scheduler_config = scheduler_config
        self.loss_fn = self.train_config.loss_fn if self.train_config.loss_fn is not None else []
        self.derivative_loss_weight = self.train_config.derivative_loss_weight

        assert self.train_config.n_slices > 0, "n_slices must be > 0"

        if not isinstance(self.loss_fn, Iterable):
            self.loss_fn = [self.loss_fn]

        self.loss_weights = (
            [self.train_config.loss_weights]
            if not isinstance(self.train_config.loss_weights, Iterable)
            else self.train_config.loss_weights
        )

        self.detach_every_k = self.train_config.detach_grad_steps
        self.train_loss_avg = MeanMetric()

        if len(self.loss_weights) != len(self.loss_fn):
            self.loss_weights = [self.loss_weights[0]] * len(self.loss_fn)

        self.derivative_loss_fn = NormalizedTimeDerivativeMSE()

        print(f"Loss functions: {self.loss_fn}, with weights: {self.loss_weights}")
        print(f"Derivative loss  {self.derivative_loss_fn}, with weight: {self.derivative_loss_weight}")

    def forward(self, x, training=True, **kwargs):
        return self.model(x, training=training, **kwargs)

    def configure_optimizers(self):

        # Configure the schedulers if given self.scheduler_config
        # Else return optimizers (can be a list of optimizers)

        # Schedulers
        schedulers = []
        if hasattr(self, "scheduler_config") and self.scheduler_config:
            for sch_cfg in self.scheduler_config:
                sch_cfg = dict(sch_cfg)  # copy
                sch_class = sch_cfg.pop("scheduler")
                # Get scheduler params
                lightning_keys = ["interval", "monitor", "frequency", "strict"]
                lightning_params = {k: sch_cfg.pop(k) for k in lightning_keys if k in sch_cfg}
                scheduler = sch_class(self.optimizer, **sch_cfg)

                # Lightning scheduler dict
                sch_dict = {"scheduler": scheduler}
                # Set scheduler params
                sch_dict["interval"] = lightning_params.get("interval", "epoch")
                if "monitor" in lightning_params:
                    sch_dict["monitor"] = lightning_params["monitor"]
                sch_dict["frequency"] = lightning_params.get("frequency", 1)
                sch_dict["strict"] = lightning_params.get("strict", True)
                schedulers.append(sch_dict)

        if schedulers:
            # returns a list of dict in the "lr_scheduler" key
            return [self.optimizer], schedulers
        else:
            return self.optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()

    def training_step(self, batch, batch_idx):

        inputs, time_idx, cond_field = batch

        B, T_unroll, C, H, W = inputs.shape
        n = self.train_config.n_slices

        epoch = self.current_epoch
        max_AR_steps = int(min(T_unroll - n - 1, self.train_config.max_autoregressive_steps))
        min_AR_steps = max(self.train_config.min_autoregressive_steps, 0)
        min_AR_steps = int(min(min_AR_steps, T_unroll - n - 1))

        start_ep = self.train_config.start_autoregressive
        final_ep = self.train_config.final_autoregressive

        if start_ep is not None and final_ep is not None:
            if epoch < start_ep:
                AR_steps = min_AR_steps
            elif epoch >= final_ep:
                AR_steps = max_AR_steps
            else:
                progress = (epoch - start_ep) / (final_ep - start_ep)
                AR_steps = int(min_AR_steps + progress * (max_AR_steps - min_AR_steps))
        else:
            AR_steps = min_AR_steps
        # ----------------------------------

        loss = 0.0
        RMSE = 0.0
        preds = None
        num_predictions = 0
        AR_preds = 0
        gamma = self.train_config.temporal_weighting
        time_weighting_factor = 1
        ARmode = False

        indiv_losses = [0] * (len(self.loss_fn))
        derivative_loss = 0
        step_losses = []

        # On commence à t = n-1 pour prédire le slice t+1. n doit être > 0
        for t in range(n - 1, T_unroll - 1):
            # Auto regressive predictions
            ARmode = num_predictions >= (T_unroll - 1 - (n - 1) - AR_steps) and preds is not None
            # if num_predictions < AR_steps and preds is not None:  # AR first
            if ARmode:  # AR last
                # if self.train_config.noise_level > 0:
                #     preds = add_noise(preds, self.train_config.noise_level, max_amplitude=1e-2, positive=False)
                current_input = torch.cat([current_input[:, 1:, ...], preds.unsqueeze(1)], dim=1)
                AR_preds += 1
                time_weighting_factor = gamma**AR_preds
            else:
                # TEACHER FORCING
                real_window = inputs[:, t - (n - 1) : t + 1, ...]
                if self.train_config.noise_level > 0:
                    current_input = add_noise(real_window, self.train_config.noise_level, max_val=0.1, positive=False)
                else:
                    current_input = real_window
                time_weighting_factor = 1.0

            # Forward pass
            model_input = current_input.reshape(B, -1, H, W)

            derivative = self.model(model_input, cond=cond_field)

            # compute the function at t+dt
            preds = current_input[:, -1, ...] + self.dt * derivative

            targets_t = inputs[:, t + 1, ...]

            # Loss between pred and targets
            for i, loss_fn in enumerate(self.loss_fn):
                l = time_weighting_factor * self.loss_weights[i] * loss_fn(preds, targets_t)
                # l = time_weighting_factor * self.loss_weights[i] * loss_fn(preds, delta)
                indiv_losses[i] = indiv_losses[i] + l.item()
                loss = loss + l

            # # loss over the derivative
            l = (
                time_weighting_factor
                * self.derivative_loss_weight
                * self.derivative_loss_fn(derivative, targets_t, current_input[:, -1, ...], self.dt)
            )
            derivative_loss = derivative_loss + l.item()
            loss = loss + l

            with torch.no_grad():
                temp_RMSE = torch.sqrt(torch.mean((preds - targets_t) ** 2))
                RMSE += temp_RMSE
                if ARmode:
                    step_losses.append(temp_RMSE.item())

            num_predictions += 1

            # Troncature des gradients (BPTT)
            if AR_preds > 0 and AR_preds % self.detach_every_k == 0:
                preds = preds.detach()

        if num_predictions > 0:
            loss = loss / num_predictions
            derivative_loss /= num_predictions
            with torch.no_grad():
                RMSE = RMSE / num_predictions
                for i, _ in enumerate(indiv_losses):
                    indiv_losses[i] = indiv_losses[i] / num_predictions

        self.train_loss_avg.update(loss)
        # Logging
        # self.log("LFE", low_freq_err, on_epoch=True, prog_bar=True)
        # self.log("HFE", high_freq_err, on_epoch=True, prog_bar=True)

        if ARmode:
            for i, step_loss in enumerate(step_losses):
                self.log(f"step_{i+1}", step_loss, on_step=False, on_epoch=True, prog_bar=True)
        for i, loss_fn in enumerate(self.loss_fn):
            self.log(f"loss_{i}", indiv_losses[i], on_step=True, on_epoch=True, prog_bar=True)
        self.log("nMSEdt", derivative_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("AR_steps", int(AR_steps), prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log("RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, time_idx, cond_field = batch
        B, T_unroll, C, H, W = inputs.shape
        n = self.train_config.n_slices
        t_norm = self.train_config.time_normalization

        loss = 0.0
        RMSE = 0.0

        preds = None
        num_predictions = 0

        for t in range(n - 1, T_unroll - 1):

            if t == n - 1:
                current_input = inputs[:, :n, ...]
            else:
                current_input = torch.cat([current_input[:, 1:, ...], preds.unsqueeze(1)], dim=1)

            # 2. Forward pass.
            preds = self.model(current_input.view(B, -1, H, W), cond=cond_field) * self.dt + current_input[:, -1, ...]

            targets_t = inputs[:, t + 1, ...]
            for i, loss_fn in enumerate(self.loss_fn):
                # l = loss_fn(preds, targets_t)
                # print(t, i, l)
                loss += self.loss_weights[i] * loss_fn(preds, targets_t)

            mse_per_sample = torch.mean((preds - targets_t) ** 2, dim=[1, 2, 3])
            rmse_per_sample = torch.sqrt(mse_per_sample)
            RMSE += torch.mean(rmse_per_sample)

            num_predictions += 1

            if (num_predictions) % self.detach_every_k == 0:
                preds = preds.detach()

        if num_predictions > 0:
            loss = loss / num_predictions
            RMSE = RMSE / num_predictions

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


class CausalNOModel(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        train_config=TrainingConfig(),
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.loss_fn = self.train_config.loss_fn if self.train_config.loss_fn is not None else []
        self.train_mode = train_config.train_mode
        self.derivative_loss_weight = self.train_config.derivative_loss_weight
        self.beta_factor = 1
        self.beta_max_epoch = 1

        if self.train_mode == "predict":
            self.autoencoder = False
            self.time_attention = True

        else:
            self.time_attention = False
            self.autoencoder = True

        if not isinstance(self.loss_fn, Iterable):
            self.loss_fn = [self.loss_fn]

        self.warmup = max(0, min(train_config.n_slices, self.model.memory - 1))

        self.loss_weights = (
            [self.train_config.loss_weights]
            if not isinstance(self.train_config.loss_weights, Iterable)
            else self.train_config.loss_weights
        )

        self.detach_every_k = self.train_config.detach_grad_steps
        self.train_loss_avg = MeanMetric()
        self.derivative_loss_fn = NormalizedTimeDerivativeMSE()

        if len(self.loss_weights) != len(self.loss_fn):
            self.loss_weights = [self.loss_weights[0]] * len(self.loss_fn)

        print(f"Loss functions: {self.loss_fn}, with weights: {self.loss_weights}")
        print(f"Training mode:{self.train_mode}")
        print("Time-attention:", self.time_attention)
        if self.autoencoder:
            print("VAE loss weight", self.train_config.VAELoss_weight)

    def forward(self, x, training=True, **kwargs):
        return self.model(x, training=training, **kwargs)

    def configure_optimizers(self):

        # Configure the schedulers if given self.scheduler_config
        # Else return optimizers (can be a list of optimizers)

        # Schedulers
        schedulers = []
        if hasattr(self, "scheduler_config") and self.scheduler_config:
            for sch_cfg in self.scheduler_config:
                sch_cfg = dict(sch_cfg)  # copy
                sch_class = sch_cfg.pop("scheduler")
                # Get scheduler params
                lightning_keys = ["interval", "monitor", "frequency", "strict"]
                lightning_params = {k: sch_cfg.pop(k) for k in lightning_keys if k in sch_cfg}
                scheduler = sch_class(self.optimizer, **sch_cfg)

                # Lightning scheduler dict
                sch_dict = {"scheduler": scheduler}
                # Set scheduler params
                sch_dict["interval"] = lightning_params.get("interval", "epoch")
                if "monitor" in lightning_params:
                    sch_dict["monitor"] = lightning_params["monitor"]
                sch_dict["frequency"] = lightning_params.get("frequency", 1)
                sch_dict["strict"] = lightning_params.get("strict", True)
                schedulers.append(sch_dict)

        if schedulers:
            # returns a list of dict in the "lr_scheduler" key
            return [self.optimizer], schedulers
        else:
            return self.optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()

    def training_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape
        t_norm = self.train_config.time_normalization

        loss = 0.0
        RMSE = 0.0
        preds = None
        history = None
        num_predictions = 0
        AR_preds = 0
        gamma = self.train_config.temporal_weighting
        time_weighting_factor = 1
        indiv_losses = [0] * (len(self.loss_fn))
        derivative_loss = 0
        VAELoss = 0

        if self.autoencoder:
            flat_inputs = inputs.reshape(-1, C, H, W)
            flat_time = time_idx + torch.arange(T_unroll, device=time_idx.device).view(1, T_unroll)
            flat_time = flat_time.reshape(-1, 1)
            preds, _, VAELoss = self.model(
                flat_inputs,
                time=flat_time,
                history=history,
                timeattention=False,
                sampling=True,
                # latent_noise=self.train_config.noise_level,
            )
            # linear increase
            beta_factor = min((self.current_epoch + 1) / self.beta_max_epoch, 1) * self.beta_factor
            VAELoss = self.train_config.VAELoss_weight * beta_factor * VAELoss
            loss += VAELoss

            for i, loss_fn in enumerate(self.loss_fn):
                l = time_weighting_factor * self.loss_weights[i] * loss_fn(preds, flat_inputs)
                indiv_losses[i] += l.item()
                loss += l

            with torch.no_grad():
                RMSE = torch.sqrt(torch.mean((preds - flat_inputs) ** 2))

        else:
            epoch = self.current_epoch
            max_AR_steps = int(min(T_unroll - self.warmup - 1, self.train_config.max_autoregressive_steps))
            min_AR_steps = max(self.train_config.min_autoregressive_steps, 0)
            min_AR_steps = int(min(min_AR_steps, T_unroll - self.warmup - 1))

            start_ep = self.train_config.start_autoregressive
            final_ep = self.train_config.final_autoregressive

            if start_ep is not None and final_ep is not None:
                if epoch < start_ep:
                    AR_steps = min_AR_steps
                elif epoch >= final_ep:
                    AR_steps = max_AR_steps
                else:
                    progress = (epoch - start_ep) / (final_ep - start_ep)
                    AR_steps = int(min_AR_steps + progress * (max_AR_steps - min_AR_steps))
            else:
                AR_steps = min_AR_steps

            # Fill the latent history
            for t in range(self.warmup):
                current_time = (time_idx + t) / t_norm
                # We only care about the updated history here
                # with torch.no_grad():  # Optional:
                _, history, _ = self.model(
                    inputs[:, t, ...],
                    time=current_time,
                    history=history,
                    timeattention=True,
                    sampling=False,
                    latent_noise=self.train_config.noise_level,
                )

            for t in range(self.warmup, T_unroll - 1):

                current_time = (time_idx + t) / t_norm

                # Auto regressive predictions
                ARmode = (
                    (AR_steps > 0)
                    and (num_predictions >= T_unroll - 1 - self.warmup - AR_steps)
                    and (preds is not None)
                )

                if ARmode:
                    # Auto-Regressive
                    current_input = preds
                    AR_preds += 1
                    time_weighting_factor = gamma**AR_preds
                else:
                    # Teacher Forcing
                    current_input = inputs[:, t]
                    time_weighting_factor = 1.0

                preds, history, _ = self.model(
                    current_input,
                    time=current_time,
                    history=history,
                    timeattention=True,
                    sampling=False,
                    latent_noise=self.train_config.noise_level,
                )

                target = inputs[:, t + 1, ...]

                # Loss between pred and targets
                for i, loss_fn in enumerate(self.loss_fn):
                    l = time_weighting_factor * self.loss_weights[i] * loss_fn(preds, target)
                    indiv_losses[i] += l.item()
                    loss += l

                # derivative loss
                derivative = (preds - current_input) / self.dt
                l = (
                    time_weighting_factor
                    * self.derivative_loss_weight
                    * self.derivative_loss_fn(derivative, target, current_input, self.dt)
                )
                derivative_loss += l.item()
                loss += l

                with torch.no_grad():
                    RMSE += torch.sqrt(torch.mean((preds - target) ** 2))

                num_predictions += 1

                # Troncature des gradients (BPTT)
                if num_predictions % self.detach_every_k == 0:
                    preds = preds.detach()
                    if history is not None:
                        history = history.detach()

            if num_predictions > 0:
                loss = loss / num_predictions
                with torch.no_grad():
                    RMSE = RMSE / num_predictions
                    for i, _ in enumerate(indiv_losses):
                        indiv_losses[i] = indiv_losses[i] / num_predictions

        self.train_loss_avg.update(loss)
        # Logging
        # self.log("LFE", low_freq_err, on_epoch=True, prog_bar=True)
        # self.log("HFE", high_freq_err, on_epoch=True, prog_bar=True)
        for i, loss_fn in enumerate(self.loss_fn):
            self.log(f"loss_{i}", indiv_losses[i], on_step=True, on_epoch=True, prog_bar=True)
        if self.autoencoder:
            self.log("VAELoss", VAELoss, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log("nMSEdt", derivative_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("AR_steps", int(AR_steps), prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log("RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape
        t_norm = self.train_config.time_normalization

        loss = 0.0
        RMSE = 0.0

        preds = None
        history = None
        num_predictions = 0
        indiv_losses = [0.0] * len(self.loss_fn)

        if self.autoencoder:
            flat_inputs = inputs.view(-1, C, H, W)
            flat_time = time_idx / t_norm

            preds, _, _ = self.model(
                flat_inputs,
                time=flat_time,
                history=history,
                timeattention=self.time_attention,
                # latent_noise=self.train_config.noise_level,
            )

            RMSE = torch.sqrt(torch.mean((preds - flat_inputs) ** 2))

        else:
            # Build the latent history
            for t in range(self.warmup):
                current_time = (time_idx + t) / t_norm
                _, history, _ = self.model(
                    inputs[:, t, ...], time=current_time, history=history, timeattention=self.time_attention
                )

            for t in range(self.warmup, T_unroll - 1):
                # fully autoregressive rollout
                if preds is not None:
                    current_input = preds
                else:
                    current_input = inputs[:, t]

                current_time = (time_idx + t) / t_norm

                preds, history, _ = self.model(
                    current_input, time=current_time, history=history, timeattention=self.time_attention, sampling=False
                )

                targets_t = inputs[:, t + 1, ...]

                for i, loss_fn in enumerate(self.loss_fn):
                    l = self.loss_weights[i] * loss_fn(preds, targets_t)
                    indiv_losses[i] += l.item()
                    loss = loss + l

                RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

                num_predictions += 1

            if num_predictions > 0:
                loss = loss / num_predictions
                RMSE = RMSE / num_predictions

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


@dataclass
class AETrainingConfig:

    sigreg_weight: float = 0.3
    mse_weight: float = 1
    mae_weight: float = 1
    crop_ratio: float = 0.15
    inpaint_mask_ratio: float = 0.35
    sobolev_weight: float = 0
    spectral_weight: float = 0

    def to_json_dict(self):
        def serialize_value(v):
            # Handle lists or tuples recursively
            if isinstance(v, (list, tuple)):
                return [serialize_value(i) for i in v]

            # Handle classes or types
            if callable(v) or isinstance(v, type):
                return v.__name__ if hasattr(v, "__name__") else v.__class__.__name__

            # Handle custom class instances
            if hasattr(v, "__class__") and type(v).__module__ != "builtins":
                return v.__class__.__name__

            return v

        return {k: serialize_value(v) for k, v in asdict(self).items()}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=4)


class AutoEncoderModel(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        train_config=AETrainingConfig(),
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.inpaint_mask_ratio = train_config.inpaint_mask_ratio
        self.sigreg_weight = train_config.sigreg_weight
        self.mse_weight = train_config.mse_weight
        self.mae_weight = train_config.mae_weight
        self.sigreg_loss_func = SIGReg()
        self.crop_ratio = train_config.crop_ratio

        self.spectral_loss_fun = SpectralLoss(beta=2)
        self.sobolev_loss_fun = SobolevLoss(l1=1.0, l2=1.0)
        self.sobolev_weigth = train_config.sobolev_weight
        self.spectral_weigth = train_config.spectral_weight

        self.frozen_model = copy.deepcopy(model)

        print(f"Loss weight: \nMAE {self.mae_weight:.2e}\nMSE {self.mse_weight:.2e}\nSIGReg {self.sigreg_weight:.2e}\n")

        for param in self.frozen_model.parameters():
            param.requires_grad_(False)
        self.frozen_model.eval()

        self.train_loss_avg = MeanMetric()

    def forward(self, x, training=True, **kwargs):
        return self.model(x, training=training, **kwargs)

    def crop_resize_views(self, images, clean_recon, masked_recon):
        top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=self.crop_ratio)
        crop_images = apply_square_crop(images, top, left, crop_size)
        crop_clean_recon = apply_square_crop(clean_recon, top, left, crop_size)
        crop_masked_recon = apply_square_crop(masked_recon, top, left, crop_size)
        return crop_images, crop_clean_recon, crop_masked_recon

    def configure_optimizers(self):

        # Configure the schedulers if given self.scheduler_config
        # Else return optimizers (can be a list of optimizers)

        # Schedulers
        schedulers = []
        if hasattr(self, "scheduler_config") and self.scheduler_config:
            for sch_cfg in self.scheduler_config:
                sch_cfg = dict(sch_cfg)  # copy
                sch_class = sch_cfg.pop("scheduler")
                # Get scheduler params
                lightning_keys = ["interval", "monitor", "frequency", "strict"]
                lightning_params = {k: sch_cfg.pop(k) for k in lightning_keys if k in sch_cfg}
                scheduler = sch_class(self.optimizer, **sch_cfg)

                # Lightning scheduler dict
                sch_dict = {"scheduler": scheduler}
                # Set scheduler params
                sch_dict["interval"] = lightning_params.get("interval", "epoch")
                if "monitor" in lightning_params:
                    sch_dict["monitor"] = lightning_params["monitor"]
                sch_dict["frequency"] = lightning_params.get("frequency", 1)
                sch_dict["strict"] = lightning_params.get("strict", True)
                schedulers.append(sch_dict)

        if schedulers:
            # returns a list of dict in the "lr_scheduler" key
            return [self.optimizer], schedulers
        else:
            return self.optimizer

    def judge_encode(self, x):
        return self.frozen_model.encode(x)

    def sync_frozen_model(self):
        self.frozen_model.load_state_dict(self.model.state_dict())
        self.frozen_model.eval()

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()

    def training_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape

        loss = 0.0
        RMSE = 0.0

        images = inputs.reshape(-1, C, H, W)
        masked_images = make_inpainted_input(images, mask_ratio=self.inpaint_mask_ratio)

        z_clean, w = self.model.encode(images)
        clean_recon = self.model.decode(z_clean, w, H, W)
        z_masked, w = self.model.encode(masked_images)
        masked_recon = self.model.decode(z_masked, w, H, W)

        if self.mse_weight > 0:
            clean_recon_z = self.judge_encode(clean_recon)[0]
            masked_recon_z = self.judge_encode(masked_recon)[0]
            crop_images, crop_clean_recon, crop_masked_recon = self.crop_resize_views(images, clean_recon, masked_recon)
            with torch.no_grad():
                target_z = self.judge_encode(crop_images)[0]

            consistency_loss = F.mse_loss(clean_recon_z, masked_recon_z)
            clean_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_clean_recon)[0])
            masked_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_masked_recon)[0])
            mse_loss = self.mse_weight * (consistency_loss + clean_crop_loss + masked_crop_loss) / 3
        else:
            mse_loss = 0

        mae = self.mae_weight * 0.5 * (F.l1_loss(images, clean_recon) + F.l1_loss(images, masked_recon))
        t_norm = torch.norm(images, p=2, dim=(-3, -2, -1))
        mae = (
            self.mae_weight
            # * 0.5
            * (
                torch.mean(torch.norm(clean_recon - images, p=2, dim=(-3, -2, -1)) / t_norm)
                # + torch.mean(torch.norm(masked_recon - images, p=2, dim=(-3, -2, -1)) / t_norm)
            )
        )
        sigreg_loss = 0
        if self.sigreg_weight > 0:
            sigreg_loss = (
                self.sigreg_weight
                * 0.5
                * (
                    self.sigreg_loss_func(preprocess_to_sigreg(z_clean))
                    + self.sigreg_loss_func(preprocess_to_sigreg(z_masked))
                )
            )

        sobolev_loss = (
            self.sobolev_weigth * self.sobolev_loss_fun(images, clean_recon) if self.sobolev_weigth > 0 else 0
        )
        spectral_loss = (
            self.spectral_weigth * self.spectral_loss_fun(images, clean_recon) if self.spectral_weigth > 0 else 0
        )

        loss = mse_loss + sigreg_loss + mae + sobolev_loss + spectral_loss

        self.train_loss_avg.update(loss)

        with torch.no_grad():
            RMSE = torch.sqrt(torch.mean((clean_recon - images) ** 2)) / images.shape[0]

        if self.mae_weight > 0:
            self.log(f"mae", mae.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.mse_weight > 0:
            self.log(f"consistency_loss", consistency_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"clean_crop_loss", clean_crop_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"masked_crop_loss", masked_crop_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"total_mse_loss", mse_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.sigreg_weight > 0:
            self.log(f"sigreg_loss", sigreg_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.sobolev_weigth > 0:
            self.log("sobolev_loss", sobolev_loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.spectral_weigth > 0:
            self.log("spectral_loss", spectral_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log(f"RMSE", RMSE.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Triggered strictly after backward and optimizer.step()
        self.sync_frozen_model()

    def validation_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape

        loss = 0.0

        images = inputs.reshape(-1, C, H, W)
        masked_images = make_inpainted_input(images, mask_ratio=self.inpaint_mask_ratio)

        z_clean, w = self.model.encode(images)
        clean_recon = self.model.decode(z_clean, w, H, W)
        z_masked, w = self.model.encode(masked_images)
        masked_recon = self.model.decode(z_masked, w, H, W)

        if self.mse_weight > 0:
            clean_recon_z = self.judge_encode(clean_recon)[0]
            masked_recon_z = self.judge_encode(masked_recon)[0]
            crop_images, crop_clean_recon, crop_masked_recon = self.crop_resize_views(images, clean_recon, masked_recon)
            target_z = self.judge_encode(crop_images)[0]
            consistency_loss = F.mse_loss(clean_recon_z, masked_recon_z)
            clean_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_clean_recon)[0])
            masked_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_masked_recon)[0])
            mse_loss = self.mse_weight * (consistency_loss + clean_crop_loss + masked_crop_loss) / 3
        else:
            mse_loss = 0
        sigreg_loss = 0
        if self.sigreg_weight > 0:
            sigreg_loss = (
                self.sigreg_weight
                * 0.5
                * (
                    self.sigreg_loss_func(preprocess_to_sigreg(z_clean))
                    + self.sigreg_loss_func(preprocess_to_sigreg(z_masked))
                )
            )
        mae = self.mae_weight * 0.5 * (F.l1_loss(images, clean_recon) + F.l1_loss(images, masked_recon))
        sobolev_loss = (
            self.sobolev_weigth * self.sobolev_loss_fun(images, clean_recon) if self.sobolev_weigth > 0 else 0
        )
        spectral_loss = (
            self.spectral_weigth * self.spectral_loss_fun(images, clean_recon) if self.spectral_weigth > 0 else 0
        )

        loss = mse_loss + sigreg_loss + mae + sobolev_loss + spectral_loss

        with torch.no_grad():
            RMSE = torch.sqrt(torch.mean((clean_recon - images) ** 2)) / images.shape[0]

        if self.mse_weight > 0:
            self.log(f"val_consistency_loss", consistency_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_clean_crop_loss", clean_crop_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_masked_crop_loss", masked_crop_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_total_mse_loss", mse_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        if self.sigreg_weight > 0:
            self.log(f"val_sigreg_loss", sigreg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        if self.mae_weight > 0:
            self.log(f"val_mae", mae.item(), on_step=False, on_epoch=True, prog_bar=True)
        if self.sobolev_weigth > 0:
            self.log("val_sobolev_loss", sobolev_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.spectral_weigth > 0:
            self.log("val_spectral_loss", spectral_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val_RMSE", RMSE.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


class AutoEncoderModelv2(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        train_config=AETrainingConfig(),
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.inpaint_mask_ratio = train_config.inpaint_mask_ratio
        self.sigreg_weight = train_config.sigreg_weight
        self.mse_weight = train_config.mse_weight
        self.mae_weight = train_config.mae_weight
        self.sigreg_loss_func = SIGReg()
        self.crop_ratio = train_config.crop_ratio

        self.spectral_loss_fun = SpectralLoss(beta=2)
        self.sobolev_loss_fun = SobolevLoss(l1=1.0, l2=1.0)
        self.sobolev_weigth = train_config.sobolev_weight
        self.spectral_weigth = train_config.spectral_weight

        self.frozen_model = copy.deepcopy(model)

        print(f"Loss weight: \nMAE {self.mae_weight:.2e}\nMSE {self.mse_weight:.2e}\nSIGReg {self.sigreg_weight:.2e}\n")

        for param in self.frozen_model.parameters():
            param.requires_grad_(False)
        self.frozen_model.eval()

        self.train_loss_avg = MeanMetric()

    def forward(self, x, training=True, **kwargs):
        return self.model(x, training=training, **kwargs)

    def crop_resize_views(self, images, clean_recon, masked_recon):
        top, left, crop_size = sample_square_crop_boxes(images, crop_ratio=self.crop_ratio)
        crop_images = apply_square_crop(images, top, left, crop_size)
        crop_clean_recon = apply_square_crop(clean_recon, top, left, crop_size)
        crop_masked_recon = apply_square_crop(masked_recon, top, left, crop_size)
        return crop_images, crop_clean_recon, crop_masked_recon

    def configure_optimizers(self):

        # Configure the schedulers if given self.scheduler_config
        # Else return optimizers (can be a list of optimizers)

        # Schedulers
        schedulers = []
        if hasattr(self, "scheduler_config") and self.scheduler_config:
            for sch_cfg in self.scheduler_config:
                sch_cfg = dict(sch_cfg)  # copy
                sch_class = sch_cfg.pop("scheduler")
                # Get scheduler params
                lightning_keys = ["interval", "monitor", "frequency", "strict"]
                lightning_params = {k: sch_cfg.pop(k) for k in lightning_keys if k in sch_cfg}
                scheduler = sch_class(self.optimizer, **sch_cfg)

                # Lightning scheduler dict
                sch_dict = {"scheduler": scheduler}
                # Set scheduler params
                sch_dict["interval"] = lightning_params.get("interval", "epoch")
                if "monitor" in lightning_params:
                    sch_dict["monitor"] = lightning_params["monitor"]
                sch_dict["frequency"] = lightning_params.get("frequency", 1)
                sch_dict["strict"] = lightning_params.get("strict", True)
                schedulers.append(sch_dict)

        if schedulers:
            # returns a list of dict in the "lr_scheduler" key
            return [self.optimizer], schedulers
        else:
            return self.optimizer

    def judge_encode(self, x):
        return self.frozen_model.encode(x)

    def sync_frozen_model(self):
        self.frozen_model.load_state_dict(self.model.state_dict())
        self.frozen_model.eval()

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()

    def training_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape

        loss = 0.0
        RMSE = 0.0

        h_coords = torch.linspace(-1, 1, H, device=inputs.device)
        w_coords = torch.linspace(-1, 1, W, device=inputs.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        out_coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)

        images = inputs.reshape(-1, C, H, W)
        masked_images = make_inpainted_input(images, mask_ratio=self.inpaint_mask_ratio)

        z_clean = self.model.encode(images)
        clean_recon = self.model.decode(z_clean, out_coords)
        z_masked = self.model.encode(masked_images)
        masked_recon = self.model.decode(z_masked, out_coords)

        if self.mse_weight > 0:
            clean_recon_z = self.judge_encode(clean_recon)[0]
            masked_recon_z = self.judge_encode(masked_recon)[0]
            crop_images, crop_clean_recon, crop_masked_recon = self.crop_resize_views(images, clean_recon, masked_recon)
            with torch.no_grad():
                target_z = self.judge_encode(crop_images)[0]

            consistency_loss = F.mse_loss(clean_recon_z, masked_recon_z)
            clean_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_clean_recon)[0])
            masked_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_masked_recon)[0])
            mse_loss = self.mse_weight * (consistency_loss + clean_crop_loss + masked_crop_loss) / 3
        else:
            mse_loss = 0

        mae = self.mae_weight * 0.5 * (F.l1_loss(images, clean_recon) + F.l1_loss(images, masked_recon))
        t_norm = torch.norm(images, p=2, dim=(-3, -2, -1))
        mae = (
            self.mae_weight
            # * 0.5
            * (
                torch.mean(torch.norm(clean_recon - images, p=2, dim=(-3, -2, -1)) / t_norm)
                # + torch.mean(torch.norm(masked_recon - images, p=2, dim=(-3, -2, -1)) / t_norm)
            )
        )
        sigreg_loss = 0
        if self.sigreg_weight > 0:
            sigreg_loss = (
                self.sigreg_weight
                * 0.5
                * (
                    self.sigreg_loss_func(preprocess_to_sigreg(z_clean))
                    + self.sigreg_loss_func(preprocess_to_sigreg(z_masked))
                )
            )

        sobolev_loss = (
            self.sobolev_weigth * self.sobolev_loss_fun(images, clean_recon) if self.sobolev_weigth > 0 else 0
        )
        spectral_loss = (
            self.spectral_weigth * self.spectral_loss_fun(images, clean_recon) if self.spectral_weigth > 0 else 0
        )

        loss = mse_loss + sigreg_loss + mae + sobolev_loss + spectral_loss

        self.train_loss_avg.update(loss)

        with torch.no_grad():
            RMSE = torch.sqrt(torch.mean((clean_recon - images) ** 2)) / images.shape[0]

        if self.mae_weight > 0:
            self.log(f"mae", mae.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.mse_weight > 0:
            self.log(f"consistency_loss", consistency_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"clean_crop_loss", clean_crop_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"masked_crop_loss", masked_crop_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"total_mse_loss", mse_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.sigreg_weight > 0:
            self.log(f"sigreg_loss", sigreg_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        if self.sobolev_weigth > 0:
            self.log("sobolev_loss", sobolev_loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.spectral_weigth > 0:
            self.log("spectral_loss", spectral_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log(f"RMSE", RMSE.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Triggered strictly after backward and optimizer.step()
        self.sync_frozen_model()

    def validation_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape

        loss = 0.0

        h_coords = torch.linspace(-1, 1, H, device=inputs.device)
        w_coords = torch.linspace(-1, 1, W, device=inputs.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        out_coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)

        images = inputs.reshape(-1, C, H, W)
        masked_images = make_inpainted_input(images, mask_ratio=self.inpaint_mask_ratio)

        z_clean = self.model.encode(images)
        clean_recon = self.model.decode(z_clean, out_coords)
        z_masked = self.model.encode(masked_images)
        masked_recon = self.model.decode(z_masked, out_coords)

        if self.mse_weight > 0:
            clean_recon_z = self.judge_encode(clean_recon)[0]
            masked_recon_z = self.judge_encode(masked_recon)[0]
            crop_images, crop_clean_recon, crop_masked_recon = self.crop_resize_views(images, clean_recon, masked_recon)
            target_z = self.judge_encode(crop_images)[0]
            consistency_loss = F.mse_loss(clean_recon_z, masked_recon_z)
            clean_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_clean_recon)[0])
            masked_crop_loss = F.mse_loss(target_z, self.judge_encode(crop_masked_recon)[0])
            mse_loss = self.mse_weight * (consistency_loss + clean_crop_loss + masked_crop_loss) / 3
        else:
            mse_loss = 0
        sigreg_loss = 0
        if self.sigreg_weight > 0:
            sigreg_loss = (
                self.sigreg_weight
                * 0.5
                * (
                    self.sigreg_loss_func(preprocess_to_sigreg(z_clean))
                    + self.sigreg_loss_func(preprocess_to_sigreg(z_masked))
                )
            )
        mae = self.mae_weight * 0.5 * (F.l1_loss(images, clean_recon) + F.l1_loss(images, masked_recon))
        sobolev_loss = (
            self.sobolev_weigth * self.sobolev_loss_fun(images, clean_recon) if self.sobolev_weigth > 0 else 0
        )
        spectral_loss = (
            self.spectral_weigth * self.spectral_loss_fun(images, clean_recon) if self.spectral_weigth > 0 else 0
        )

        loss = mse_loss + sigreg_loss + mae + sobolev_loss + spectral_loss

        with torch.no_grad():
            RMSE = torch.sqrt(torch.mean((clean_recon - images) ** 2)) / images.shape[0]

        if self.mse_weight > 0:
            self.log(f"val_consistency_loss", consistency_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_clean_crop_loss", clean_crop_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_masked_crop_loss", masked_crop_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"val_total_mse_loss", mse_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        if self.sigreg_weight > 0:
            self.log(f"val_sigreg_loss", sigreg_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        if self.mae_weight > 0:
            self.log(f"val_mae", mae.item(), on_step=False, on_epoch=True, prog_bar=True)
        if self.sobolev_weigth > 0:
            self.log("val_sobolev_loss", sobolev_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.spectral_weigth > 0:
            self.log("val_spectral_loss", spectral_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"val_RMSE", RMSE.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
