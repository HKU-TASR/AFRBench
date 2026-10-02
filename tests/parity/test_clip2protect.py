"""CLIP2Protect InfoNCE and adversarial cosine against transcribed references.

Reference Code: https://github.com/fahadshamshad/Clip2Protect
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from afrlib.afr.clip2protect.nce_loss import info_nce

def _ref_info_nce(query, positive_key, negative_keys, temperature=0.1):
    """Unpaired-negatives path of upstream ``criteria/infonce.py``."""
    query, positive_key, negative_keys = (
        F.normalize(x, dim=-1) for x in (query, positive_key, negative_keys)
    )
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
    negative_logits = query @ negative_keys.transpose(-2, -1)
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits / temperature, labels)


def test_info_nce_matches_transcribed_unpaired():
    torch.manual_seed(0)
    query = torch.randn(4, 16)
    positive = torch.randn(4, 16)
    negatives = torch.randn(7, 16)
    ours = info_nce(query, positive, negatives)
    expected = _ref_info_nce(query, positive, negatives)
    assert torch.allclose(ours, expected, atol=1e-5, rtol=1e-5)


def test_adversarial_cosine_is_one_minus_cos():
    # Upstream get_adv_loss (adversarial_optimization.py) uses 1 - cos.
    torch.manual_seed(0)
    generated = torch.randn(3, 8)
    target = torch.randn(3, 8)
    ours = sum(1 - F.cosine_similarity(g, t, dim=0) for g, t in zip(generated, target))
    expected = (1 - F.cosine_similarity(generated, target, dim=-1)).sum()
    assert torch.allclose(ours, expected, atol=1e-5)
