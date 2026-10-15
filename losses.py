"""Per-sample spatial reconstruction loss."""
from torch.nn import functional as F


def reconstruction_loss(logits, target, sample_weights, ual_lambda=0.5):
    boundary = 1 + 5 * (F.avg_pool2d(target, 31, stride=1, padding=15) - target).abs()
    reduce_dims = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (boundary * bce).sum(reduce_dims) / boundary.sum(reduce_dims)
    probability = logits.sigmoid()
    intersection = (boundary * probability * target).sum(reduce_dims)
    union = (boundary * (probability + target)).sum(reduce_dims) - intersection
    iou = 1 - (intersection + 1) / (union + 1)
    ual = (1 - (2 * probability - 1).square()).mean(reduce_dims)
    per_sample = bce + iou + ual_lambda * ual
    if sample_weights.shape != per_sample.shape:
        raise ValueError("Expected one scalar weight per sample.")
    loss = (per_sample * sample_weights).mean()
    components = {"bce": bce.detach().mean().item(),
                  "iou": iou.detach().mean().item(), "ual": ual.detach().mean().item()}
    return loss, components
