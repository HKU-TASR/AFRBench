"""Losses of AdvCloak's two-stage GAN training.

Reference Code: https://github.com/liuxuannan/AdvCloak (CC BY-SA 4.0, adapted; see LICENSE)
"""

import math

import torch
import torch.nn.functional as F


def soft_labels(n: int, real: bool, device) -> torch.Tensor:
    """Soft GAN labels for real (mostly in [0.9, 1]) or fake (mostly in [0, 0.1]) samples,
    with 5% flipped.
    """
    true_num = math.floor(n * 0.95)
    false_num = n - true_num
    if real:
        label_true = 0.1 * torch.rand(true_num, device=device) + 0.9
        label_false = 0.1 * torch.rand(false_num, device=device)
    else:
        label_true = 0.1 * torch.rand(true_num, device=device)
        label_false = 0.1 * torch.rand(false_num, device=device) + 0.9
    return torch.cat((label_true, label_false))


def discriminator_loss(pred: torch.Tensor, real: bool) -> torch.Tensor:
    """BCE-with-logits of ``pred`` against soft (flipped) real/fake labels."""
    labels = soft_labels(pred.shape[0], real, pred.device)
    return F.binary_cross_entropy_with_logits(pred, labels)


def generator_gan_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """Generator's GAN objective: make the discriminator predict real (1)."""
    return F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))


def norm_constraint_loss(perturbation: torch.Tensor, c: float) -> torch.Tensor:
    """``mean(max(||p||_2, C))`` over the batch; penalizes only norms above ``C``."""
    norm = torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1)
    c_tensor = torch.full_like(norm, c)
    return torch.mean(torch.max(norm, c_tensor))


def feature_similarity_loss(clean_feats: torch.Tensor, adv_feats: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between clean and protected embeddings (minimized)."""
    return torch.mean(torch.cosine_similarity(clean_feats, adv_feats, dim=1))
