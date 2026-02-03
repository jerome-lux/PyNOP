import math
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os
import json
from typing import Any
from torchmetrics import MeanMetric
from pytorch_lightning.callbacks import ModelCheckpoint
from dataclasses import dataclass, asdict
from ..core import add_noise

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
class TrainingSchedule:
    start_autoregressive: Any = None  # start o fthe autoregressive training
    final_autoregressive: Any = None  # epoch at which the number of autoregressive steps is maximum
    min_autoregressive_steps: int = 0  # minimum number of autoregressive steps (after start_autoregressive epochs)
    max_autoregressive_steps: int = 0  # maximum number of autoregressive steps
    detach_grad_steps: int = 4  # number of steps before detaching the gradient in autoregressive mode
    loss_fn: Any = torch.nn.MSELoss()
    noise_level: float = 0  # no noise if 0
    time_normalization: float = 1
    residual: bool = True  # learns f(x, t + dt) - f(x, t)
    n_slices: int = 2  # Only for NOModelAR

    def to_json_dict(self):
        def serialize_value(v):
            # 1. Si c'est une fonction ou une classe (ex: torch.nn.MSELoss)
            if callable(v) or isinstance(v, type):
                if hasattr(v, "__name__"):
                    return v.__name__
                return v.__class__.__name__

            # 2. Si c'est un objet complexe non-natif (mais pas une fonction)
            if hasattr(v, "__class__") and type(v).__module__ != "builtins":
                return v.__class__.__name__

            # 3. Retourner la valeur telle quelle pour les types natifs (int, float, bool, str, None)
            # JSON gère ces types parfaitement.
            return v

        # Utilisation de asdict(self) pour récupérer les données de la dataclass
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


class OneStepNOModel(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        train_config=TrainingSchedule(),
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.loss_fn = self.train_config.loss_fn
        self.detach_every_k = self.train_config.detach_grad_steps
        self.train_loss_avg = MeanMetric()

    def forward(self, x, training=True):
        return self.model(x, training=training, residual=self.train_config.residual)

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
        time_idx = (1 + time_idx) / t_norm
        dt = 1.0 / t_norm

        epoch = self.current_epoch
        max_AR_steps = int(min(T_unroll - 1, self.train_config.max_autoregressive_steps))
        min_AR_steps = max(self.train_config.min_autoregressive_steps, 0)

        if self.train_config.start_autoregressive is not None and self.train_config.final_autoregressive is not None:
            nint = max_AR_steps - min_AR_steps + 1
            delta = (self.train_config.final_autoregressive - self.train_config.start_autoregressive) // nint
            if epoch < self.train_config.start_autoregressive:
                AR_steps = min_AR_steps
            if delta > 0:
                AR_steps = int(
                    min(max(min_AR_steps + (epoch - self.train_config.start_autoregressive) // delta, 0), max_AR_steps)
                )
            else:
                AR_steps = max_AR_steps

        else:
            AR_steps = min_AR_steps

        threshold = AR_steps

        loss = 0.0
        RMSE = 0.0

        preds = None
        AR_counter = 0
        # Time unrolling
        for t in range(0, T_unroll - 1):

            if t < threshold:
                # AUTOREGRESSIVE:
                AR_counter += 1
                if preds is None:
                    preds = self.model(inputs[:, t, ...], time=time_idx, residual=self.train_config.residual)
                else:
                    preds = self.model(preds, time=time_idx + t * dt, residual=self.train_config.residual)
                if preds.dim() == 5 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)
            else:
                # TEACHER FORCING:
                preds = self.model(
                    add_noise(inputs[:, t, ...], self.train_config.noise_level, positive=False),
                    time=time_idx + t * dt,
                    residual=self.train_config.residual,
                )

            targets_t = inputs[:, t + 1, ...]
            loss += self.loss_fn(preds, targets_t)
            with torch.no_grad():
                RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))
            # limit the gradient backpropagation to detach_every_k time steps
            if AR_counter % self.detach_every_k == 0:
                preds = preds.detach()

        if T_unroll > 1:
            loss = loss / (T_unroll - 1)
            with torch.no_grad():
                RMSE = RMSE / (T_unroll - 1)

        self.train_loss_avg.update(loss)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log("RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("AR_steps", int(AR_steps), on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape
        t_norm = self.train_config.time_normalization
        time_idx = (1 + time_idx) / t_norm
        dt = 1.0 / t_norm

        loss = 0.0
        RMSE = 0.0

        preds = None
        # AR_counter = 0
        # Time unrolling
        for t in range(0, T_unroll - 1):

            if preds is None:
                preds = self.model(inputs[:, t, ...], time=time_idx, residual=self.train_config.residual)
            else:
                preds = self.model(preds, time=time_idx + t * dt, residual=self.train_config.residual)

            targets_t = inputs[:, t + 1, ...]
            loss += self.loss_fn(preds, targets_t)
            RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

        if T_unroll > 1:
            loss = loss / (T_unroll - 1)
            with torch.no_grad():
                RMSE = RMSE / (T_unroll - 1)

        if T_unroll > 1:
            loss = loss / (T_unroll - 1)
            RMSE = RMSE / (T_unroll - 1)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


class MultiStepNOModel(pl.LightningModule):

    def __init__(
        self,
        model,
        optimizer,
        train_config=TrainingSchedule(),
        scheduler_config=None,
    ):
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.loss_fn = self.train_config.loss_fn
        self.detach_every_k = self.train_config.detach_grad_steps
        self.train_loss_avg = MeanMetric()

    def forward(self, x, training=True):
        return self.model(x, training=training, residual=self.train_config.residual)

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

    # def training_step(self, batch, batch_idx):

    #     inputs, time_idx = batch
    #     B, T_unroll, C, H, W = inputs.shape
    #     n = self.train_config.n_slices
    #     t_norm = self.train_config.time_normalization

    #     loss = 0.0
    #     RMSE = 0.0

    #     preds = None
    #     num_predictions = 0

    #     for t in range(n - 1, T_unroll - 1):

    #         if t == n - 1:
    #             current_input = inputs[:, :n, ...]
    #         else:
    #             current_input = torch.cat([current_input[:, 1:, ...], preds.unsqueeze(1)], dim=1)

    #         # 2. Forward pass.
    #         current_time = (time_idx + t) / t_norm

    #         preds = self.model(current_input.view(B, -1, H, W), time=current_time, residual=self.train_config.residual)

    #         targets_t = inputs[:, t + 1, ...]
    #         loss += self.loss_fn(preds, targets_t)

    #         with torch.no_grad():
    #             RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

    #         num_predictions += 1

    #         if (num_predictions) % self.detach_every_k == 0:
    #             preds = preds.detach()

    #     if num_predictions > 0:
    #         loss = loss / num_predictions
    #         with torch.no_grad():
    #             RMSE = RMSE / num_predictions

    #     self.train_loss_avg.update(loss)

    #     self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
    #     self.log("RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)

    #     return loss

    def training_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape
        n = self.train_config.n_slices
        t_norm = self.train_config.time_normalization

        epoch = self.current_epoch
        max_AR_steps = int(min(T_unroll - n, self.train_config.max_autoregressive_steps))
        min_AR_steps = max(self.train_config.min_autoregressive_steps, 0)
        min_AR_steps = int(min(min_AR_steps, T_unroll - n))

        if self.train_config.start_autoregressive is not None and self.train_config.final_autoregressive is not None:
            nint = max_AR_steps - min_AR_steps + 1
            delta = (self.train_config.final_autoregressive - self.train_config.start_autoregressive) // nint
            if epoch < self.train_config.start_autoregressive:
                AR_steps = min_AR_steps
            elif delta > 0:

                AR_steps = int(
                    min(max(min_AR_steps + (epoch - self.train_config.start_autoregressive) // delta, 0), max_AR_steps)
                )
            else:
                AR_steps = max_AR_steps
        else:
            AR_steps = min_AR_steps
        # ----------------------------------

        loss = 0.0
        RMSE = 0.0
        preds = None
        num_predictions = 0
        AR_preds = 0

        # On commence à t = n-1 pour prédire le slice n
        for t in range(n - 1, T_unroll - 1):

            if num_predictions < AR_steps and preds is not None:
                current_input = torch.cat([current_input[:, 1:, ...], preds.unsqueeze(1)], dim=1)
                AR_preds += 1
            else:
                # Mode TEACHER FORCING : On utilise les données réelles (avec bruit éventuel)
                real_window = inputs[:, t - (n - 1) : t + 1, ...]
                if self.train_config.noise_level > 0:
                    current_input = add_noise(real_window, self.train_config.noise_level, positive=False)
                else:
                    current_input = real_window

            # Forward pass
            current_time = (time_idx + (t + 1)) / t_norm

            model_input = current_input.reshape(B, -1, H, W)
            preds = self.model(model_input, time=current_time, residual=self.train_config.residual)

            targets_t = inputs[:, t + 1, ...]
            loss += self.loss_fn(preds, targets_t)

            with torch.no_grad():
                RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

            num_predictions += 1

            # 4. Troncature des gradients (BPTT)
            if AR_preds % self.detach_every_k == 0:
                preds = preds.detach()

        # Normalisation finale
        if num_predictions > 0:
            loss = loss / num_predictions
            with torch.no_grad():
                RMSE = RMSE / num_predictions

        self.train_loss_avg.update(loss)
        # Logging
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
            current_time = (time_idx + t) / t_norm

            preds = self.model(current_input.view(B, -1, H, W), time=current_time, residual=self.train_config.residual)

            targets_t = inputs[:, t + 1, ...]
            loss += self.loss_fn(preds, targets_t)

            RMSE += torch.sqrt(torch.mean((preds - targets_t) ** 2))

            num_predictions += 1

            if (num_predictions) % self.detach_every_k == 0:
                preds = preds.detach()

        if num_predictions > 0:
            loss = loss / num_predictions
            RMSE = RMSE / num_predictions

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_RMSE", RMSE, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
