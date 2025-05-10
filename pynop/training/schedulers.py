from torch.optim.lr_scheduler import LambdaLR
import math


def cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7, num_cycles=0.5, last_epoch=-1
):
    """
    Crée un scheduler de taux d'apprentissage qui augmente linéairement le taux d'apprentissage
    de 0.0 à lr pendant num_warmup_steps, puis diminue jusqu'à 0.0 selon un calendrier cosinus
    sur les num_training_steps-num_warmup_steps restants (en supposant num_cycles = 0.5).

    Basé sur l'implémentation de Hugging Face :
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): L'optimiseur pour lequel planifier le taux d'apprentissage.
        num_warmup_steps (:obj:`int`): Le nombre d'étapes pour la phase d'échauffement.
        num_training_steps (:obj:`int`): Le nombre total d'étapes d'entrainement.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5): Le nombre de cycles dans le calendrier cosinus.
            Par défaut, 0.5 (diminution de la valeur maximale à 0 suivant un demi-cosinus).
        last_epoch (:obj:`int`, `optional`, defaults to -1): L'index de la dernière èpoque lors de la reprise de l'entrainement.

    Returns:
        :obj:`torch.optim.lr_scheduler.LambdaLR` avec le scheduler approprié.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * progress)))
        return max(min_lr / optimizer.param_groups[0]["lr"], cosine_decay)

    return LambdaLR(optimizer, lr_lambda)
