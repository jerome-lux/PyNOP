import math
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os
import json
from typing import Any, Iterable
from torchmetrics import MeanMetric
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import dataclass, asdict
from ..core import add_noise
from ..core.loss import DWMSELoss, NormalizedTimeDerivativeMSE

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
    n_slices: int = 2  # Only for MultiStepModel
    temporal_weighting: float = (
        1.2  # Increase loss with the timestep during autoregressive training: temporal_weighting**timestep
    )
    train_mode: str = "prediction"  # if "autoencoder", train only the enocder/decoer part in CausalLNO
    VAELoss_weight: float = 1e-4  # Only for VAE training (if train_mode="autoencoder")

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


def jacobian_regularization(model_input, derivative, n_projections=1):
    reg = 0.0
    for _ in range(n_projections):
        v = torch.randn_like(derivative)

        # JVP: Calcule le gradient de (derivative * v) par rapport a model_input
        jvp = torch.autograd.grad(
            outputs=derivative,
            inputs=model_input,
            grad_outputs=v,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        reg += torch.mean(jvp**2)

    return reg / n_projections


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
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape
        n = self.train_config.n_slices
        t_norm = self.train_config.time_normalization

        # epoch = self.current_epoch
        # max_AR_steps = int(min(T_unroll - n, self.train_config.max_autoregressive_steps))
        # min_AR_steps = max(self.train_config.min_autoregressive_steps, 0)
        # min_AR_steps = int(min(min_AR_steps, T_unroll - n))

        # if self.train_config.start_autoregressive is not None and self.train_config.final_autoregressive is not None:
        #     nint = max_AR_steps - min_AR_steps + 1
        #     delta = (self.train_config.final_autoregressive - self.train_config.start_autoregressive) // nint
        #     if epoch < self.train_config.start_autoregressive:
        #         AR_steps = min_AR_steps
        #     elif delta > 0:
        #         AR_steps = int(
        #             min(max(min_AR_steps + (epoch - self.train_config.start_autoregressive) // delta, 0), max_AR_steps)
        #         )
        #     else:
        #         AR_steps = max_AR_steps
        # else:
        #     AR_steps = min_AR_steps

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
            current_time = (time_idx + (t + 1)) / t_norm

            model_input = current_input.reshape(B, -1, H, W)

            derivative = self.model(model_input, time=current_time, return_derivative=True)

            # compute the function at t+dt
            preds = current_input[:, -1, ...] + self.model.dt * derivative

            targets_t = inputs[:, t + 1, ...]

            # Loss between pred and targets
            for i, loss_fn in enumerate(self.loss_fn):
                l = time_weighting_factor * self.loss_weights[i] * loss_fn(preds, targets_t)
                # l = time_weighting_factor * self.loss_weights[i] * loss_fn(preds, delta)
                indiv_losses[i] += l.item()
                loss += l

            # # loss over the derivative
            l = (
                time_weighting_factor
                * self.derivative_loss_weight
                * self.derivative_loss_fn(derivative, targets_t, current_input[:, -1, ...], self.model.dt)
            )
            derivative_loss += l.item()
            loss += l

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
        inputs, time_idx = batch
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
            current_time = (time_idx + t + 1) / t_norm

            preds = self.model(current_input.view(B, -1, H, W), time=current_time, return_derivative=False)

            targets_t = inputs[:, t + 1, ...]
            for i, loss_fn in enumerate(self.loss_fn):
                # l = loss_fn(preds, targets_t)
                # print(t, i, l)
                loss += self.loss_weights[i] * loss_fn(preds, targets_t)

            mse_per_sample = torch.mean((preds - targets_t) ** 2, dim=[1, 2, 3])
            rmse_per_sample = torch.sqrt(mse_per_sample)
            RMSE += torch.mean(rmse_per_sample)
            # RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

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
                derivative = (preds - current_input) / self.model.dt
                l = (
                    time_weighting_factor
                    * self.derivative_loss_weight
                    * self.derivative_loss_fn(derivative, target, current_input, self.model.dt)
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


class CausalNOModel_old(pl.LightningModule):

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
        self.loss_fn = self.train_config.loss_fn
        self.loss_weights = self.train_config.loss_weights
        self.detach_every_k = self.train_config.detach_grad_steps
        self.train_loss_avg = MeanMetric()
        self.warmup = min(train_config.n_slices, self.model.max_history)
        self.train_mode = train_config.train_mode
        self.derivative_loss_weight = self.train_config.derivative_loss_weight

    def forward(self, x, training=True, **kwargs):
        return self.model(x, training=training, residual=False, **kwargs)

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
        n = self.train_config.n_slices
        t_norm = self.train_config.time_normalization

        indiv_losses = [0] * len(self.loss_fn)
        loss = 0.0
        derivative_loss = 0.0

        # --- MODE AUTO-ENCODEUR  ---
        if getattr(self, "train_mode", "standard") == "autoencoder":
            # Flattening the sequence
            # [B, T, C, H, W] -> [B*T, C, H, W]
            x_flat = inputs.view(-1, C, H, W)

            # We need a time tensor, even if it's not used by the autoencoder model
            dummy_time = torch.zeros((B * T_unroll, 1), device=inputs.device)

            # Forward pass without any history and without residual
            preds, _ = self.model(x_flat, time=dummy_time, history=None, autoencoder=True, residual=False)

            with torch.no_grad():
                RMSE = torch.sqrt(torch.mean((preds - x_flat) ** 2))

            for i, loss_fn in enumerate(self.loss_fn):
                l = self.loss_weights[i] * loss_fn(preds, x_flat)
                indiv_losses[i] += l.item()
                loss += l

        else:
            epoch = self.current_epoch
            max_AR_steps = int(min(T_unroll - n, self.train_config.max_autoregressive_steps))
            min_AR_steps = max(self.train_config.min_autoregressive_steps, 0)
            min_AR_steps = int(min(min_AR_steps, T_unroll - n))

            if (
                self.train_config.start_autoregressive is not None
                and self.train_config.final_autoregressive is not None
            ):
                nint = max_AR_steps - min_AR_steps + 1
                delta = (self.train_config.final_autoregressive - self.train_config.start_autoregressive) // nint
                if epoch < self.train_config.start_autoregressive:
                    AR_steps = min_AR_steps
                elif delta > 0:
                    AR_steps = int(
                        min(
                            max(min_AR_steps + (epoch - self.train_config.start_autoregressive) // delta, 0),
                            max_AR_steps,
                        )
                    )
                else:
                    AR_steps = max_AR_steps
            else:
                AR_steps = min_AR_steps

            # MODE PREDICTION ----------------------------------

            RMSE = 0.0
            preds = None
            history = None  # <--- Initialize Latent History
            num_predictions = 0
            AR_count = 0
            gamma = self.train_config.temporal_weighting

            # --- PHASE 1: WARM-UP (No Loss) ---
            # Encode ground truth frames into latent history

            for t in range(self.warmup):
                current_time = (time_idx + t) / t_norm
                # We only care about the updated history here
                with torch.no_grad():  # Optional: disable grads for warm-up to save memory
                    _, history = self.model(
                        inputs[:, t, ...], time=current_time, history=history, residual=self.train_config.residual
                    )

            # --- PHASE 2: PREDICTION LOOP ---
            # Start predicting from the first frame after warm-up
            for t in range(self.warmup, T_unroll - 1):

                # Decide between Teacher Forcing and AR
                is_ar = num_predictions >= (T_unroll - 1 - self.warmup - AR_steps) and preds is not None

                if is_ar:
                    current_input = preds
                    AR_count += 1
                    time_weight = gamma**AR_count
                else:
                    current_input = inputs[:, t, ...]
                    if self.train_config.noise_level > 0:
                        current_input = add_noise(current_input, self.train_config.noise_level)
                    time_weight = 1.0

                target_time = (time_idx + t) / t_norm

                # Forward pass
                derivative, history = self.model(
                    current_input,
                    time=target_time,
                    history=history,
                    return_derivative=True,
                )

                # print(t, batch_idx, torch.isnan(preds).any(), torch.isnan(history).any())
                targets_t = inputs[:, t + 1, ...]
                for i, loss_fn in enumerate(self.loss_fn):
                    l = time_weight * self.loss_weights[i] * loss_fn(preds, targets_t)
                    indiv_losses[i] += l.item()
                    loss = loss + l
                # loss += self.loss_fn(preds - model_input[:, -2:, ...], targets_t - model_input[:, -2:, ...])

                with torch.no_grad():
                    RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

                num_predictions += 1

                # Troncature des gradients (BPTT)
                if AR_count % self.detach_every_k == 0:
                    preds = preds.detach()

            if num_predictions > 0:
                loss = loss / num_predictions
                with torch.no_grad():
                    RMSE = RMSE / num_predictions
                    for i, _ in enumerate(indiv_losses):
                        indiv_losses[i] = indiv_losses[i] / num_predictions

            self.log("AR_steps", int(AR_steps), prog_bar=True)

        self.train_loss_avg.update(loss)

        for i, loss_fn in enumerate(self.loss_fn):
            self.log(f"loss_{i}", indiv_losses[i], on_step=True, on_epoch=True, prog_bar=True)

        self.log("RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape
        t_norm = self.train_config.time_normalization

        indiv_losses = [0] * len(self.loss_fn)
        loss = 0.0

        # --- MODE AUTO-ENCODEUR (Phase 1) ---
        if getattr(self, "train_mode", "standard") == "autoencoder":
            # On traite toutes les frames du batch comme des exemples indépendants
            # [B, T, C, H, W] -> [B*T, C, H, W]
            x_flat = inputs.view(-1, C, H, W)

            # We need a time tensor, even if it's not used by the autoencoder model
            dummy_time = torch.zeros((B * T_unroll, 1), device=inputs.device)

            # Forward pass without any history
            preds, _ = self.model(x_flat, time=dummy_time, history=None, autoencoder=True, residual=False)
            RMSE = torch.sqrt(torch.mean((preds - x_flat) ** 2))

            loss = 0.0
            for i, loss_fn in enumerate(self.loss_fn):
                l = self.loss_weights[i] * loss_fn(preds, x_flat)
                indiv_losses[i] += l.item()
                loss += l

        else:

            RMSE = 0.0
            preds = None
            history = None
            num_predictions = 0
            # fil the history with the latent representation of past timesteps
            for t in range(self.warmup):
                current_time = (time_idx + t) / t_norm
                _, history = self.model(
                    inputs[:, t, ...], time=current_time, history=history, residual=self.train_config.residual
                )

            for t in range(self.warmup, T_unroll - 1):

                if preds is None:
                    current_input = inputs[:, 0, ...]
                else:
                    current_input = preds

                # 2. Forward pass.
                current_time = (time_idx + t + 1) / t_norm

                preds, history = self.model(
                    current_input,
                    time=current_time,
                    history=history,  # <--- Pass the history
                    residual=self.train_config.residual,
                )
                # print(t, batch_idx, torch.isnan(preds).any(), torch.isnan(history).any())
                targets_t = inputs[:, t + 1, ...]
                for i, loss_fn in enumerate(self.loss_fn):
                    l = self.loss_weights[i] * loss_fn(preds, targets_t)
                    indiv_losses[i] += l.item()
                    loss = loss + l

                RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

                num_predictions += 1

                if (num_predictions) % self.detach_every_k == 0:
                    preds = preds.detach()

            if num_predictions > 0:
                loss = loss / num_predictions
                RMSE = RMSE / num_predictions

        self.log("val_RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i, loss_fn in enumerate(self.loss_fn):
            self.log(f"val_loss_{i}", indiv_losses[i], on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
