import torch
from pytorch_lightning.callbacks import Callback

class ITLNOGradientLogger(Callback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        writer = trainer.logger.experiment  # SummaryWriter
        step = trainer.global_step

        # 
        groups = {
            "basis/coord_generator": pl_module.model.coord_generator,
            "basis/signal_generator": pl_module.model.signal_generator,
            "transform/lifting":      pl_module.model.lifting,
            "transform/projection":   pl_module.model.projection,
            "conditioning/time_scaling": pl_module.model.time_scaling,
        }

        for group_name, module in groups.items():
            grads = [p.grad.flatten()
                     for p in module.parameters()
                     if p.grad is not None]
            if not grads:
                continue

            g = torch.cat(grads)
            writer.add_histogram(f"grad_hist/{group_name}", g, step)
            writer.add_scalar(f"grad_norm/{group_name}", g.norm(), step)

            params = torch.cat([p.flatten()
                                 for p in module.parameters()])
            writer.add_scalar(
                f"grad_param_ratio/{group_name}",
                g.norm() / (params.norm() + 1e-8),
                step
            )