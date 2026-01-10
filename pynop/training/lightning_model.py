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
from attr import dataclass

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


class Model(pl.LightningModule):

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
        return self.model(x, training=training)

    def configure_optimizers(self):

        # Configure the schedulers if given  self.scheduler_config
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

    # def on_before_optimizer_step(self, optimizer):
    #     # Calcule la norme L2 globale de tous les poids du modèle
    #     norms = [p.norm(2) for p in self.model.parameters() if p.grad is not None]
    #     total_norm = torch.norm(torch.stack(norms))
    #     self.log("weight_norm", total_norm, prog_bar=True, on_step=True)

    def training_step(self, batch, batch_idx):

        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape

        epoch = self.current_epoch
        max_AR_steps = min(T_unroll - 1, self.train_config.max_autoregressive_steps)
        min_AR_steps = max(self.train_config.min_autoregressive_steps, 1)

        if self.train_config.start_autoregressive is not None and self.train_config.final_autoregressive is not None:
            nint = max_AR_steps - min_AR_steps + 1
            delta = (self.train_config.final_autoregressive - self.train_config.start_autoregressive) // nint
            if delta > 0:
                AR_steps = min(max(min_AR_steps + (epoch // delta), 0), max_AR_steps)
            else:
                AR_steps = max_AR_steps

        else:
            AR_steps = 0

        threshold = (T_unroll - 1) - AR_steps

        loss = 0.0

        # First prediction - always teacher forcing
        preds = self.model(inputs[:, 0, ...])  # predict next time step -> should be B, C, H, W
        if preds.dim() == 5 and preds.shape[1] == 1:
            preds = preds.squeeze(1)

        loss += self.loss_fn(preds, inputs[:, 1, ...])

        # Time unrolling
        for t in range(1, T_unroll - 1):

            if t < threshold:
                # TEACHER FORCING:
                preds = self.model(inputs[:, t, ...])
            else:
                # AUTOREGRESSIVE:
                preds = self.model(preds)
                if preds.dim() == 5 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)

            targets_t = inputs[:, t + 1, ...]
            loss += self.loss_fn(preds, targets_t)

            # The real batch size is B * self.detach_every_k
            if (t % self.detach_every_k) == 0:
                preds = preds.detach()

        loss = loss / (T_unroll - 1)
        self.train_loss_avg.update(loss)

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_loss", self.train_loss_avg.compute(), prog_bar=True)
        self.log("AR_steps", int(AR_steps), on_step=True, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, time_idx = batch
        B, T_unroll, C, H, W = inputs.shape

        epoch = self.current_epoch
        max_AR_steps = min(T_unroll - 1, self.train_config.max_autoregressive_steps)
        min_AR_steps = max(self.train_config.min_autoregressive_steps, 1)

        if self.train_config.start_autoregressive is not None and self.train_config.final_autoregressive is not None:
            nint = max_AR_steps - min_AR_steps + 1
            delta = (self.train_config.final_autoregressive - self.train_config.start_autoregressive) // nint
            if delta > 0:
                AR_steps = min(max(min_AR_steps + (epoch // delta), 0), max_AR_steps)
            else:
                AR_steps = max_AR_steps

        else:
            AR_steps = 0

        threshold = (T_unroll - 1) - AR_steps

        loss = 0.0

        # First prediction - always teacher forcing
        preds = self.model(inputs[:, 0, ...])  # predict next time step -> should be B, C, H, W
        if preds.dim() == 5 and preds.shape[1] == 1:
            preds = preds.squeeze(1)

        loss += self.loss_fn(preds, inputs[:, 1, ...])

        # Time unrolling - The real batch size is B * T_unroll
        for t in range(1, T_unroll - 1):

            if t < threshold:
                # TEACHER FORCING:
                preds = self.model(inputs[:, t, ...])
            else:
                # AUTOREGRESSIVE:
                preds = self.model(preds)

                if preds.dim() == 5 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)

            targets_t = inputs[:, t + 1, ...]
            loss += self.loss_fn(preds, targets_t)

            # needed in autoregressive mode to prevent the gradient going back to much.
            if (t % self.detach_every_k) == 0:
                preds = preds.detach()

        loss = loss / (T_unroll - 1)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
