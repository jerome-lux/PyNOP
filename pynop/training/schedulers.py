from torch.optim.lr_scheduler import LambdaLR
import math


def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7, num_cycles=0.5):

    base_lr = optimizer.param_groups[0]["lr"]

    min_ratio = min_lr / base_lr

    def lr_lambda(current_step):
        # 1. Phase de Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 2. Phase de Cosine Decay
        # On calcule la progression apr�s le warmup (de 0 � 1)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        lr_mult = min_ratio + (1.0 - min_ratio) * cosine_decay

        return max(min_ratio, lr_mult)

    return LambdaLR(optimizer, lr_lambda)
