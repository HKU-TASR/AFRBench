"""DPA networks: the ArcFace training head, the hard-model conv wrapper, and the trainable surrogate.

Reference Code: https://github.com/fengfanzhou/DPA
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from afrlib.contracts import ChannelType


class ArcMarginProduct(nn.Module):
    """ArcFace head mapping embeddings to margin-adjusted logits; a fresh one is trained with
    each backbone and then discarded, as the victim's head is unavailable.
    """

    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        s: float = 32.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.head_name = "ArcFace"

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # Rounding can push |cosine| to 1 or beyond, where sqrt(1 - cosine**2)
        # has an infinite gradient or a negative radicand; clamp to stay finite.
        eps = torch.finfo(cosine.dtype).eps
        cosine = cosine.clamp(min=-1.0 + eps, max=1.0 - eps)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp_min(eps))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s


class SurrogateForward(nn.Module):
    """A trainable copy of one recognizer's backbone behind that recognizer's own preprocessing;
    embeds ``[0, 255]`` RGB crops as ``[B, D]``.

    Args:
        backbone: A deep copy of the recognizer's backbone.
        input_size: The recognizer's input size.
        mean: Per-channel normalization mean.
        std: Per-channel normalization standard deviation.
        channel: The recognizer's channel order.
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: Tuple[int, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        channel: ChannelType,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = tuple(int(v) for v in input_size)
        self.channel = channel
        self.register_buffer("_mean", mean.reshape(1, -1, 1, 1).float().clone())
        self.register_buffer("_std", std.reshape(1, -1, 1, 1).float().clone())

    def forward(self, imgs255: torch.Tensor) -> torch.Tensor:
        x = imgs255.contiguous().to(self._mean.device, dtype=torch.float32) / 255.0
        if tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(
                x, size=self.input_size, mode="bilinear", align_corners=False, antialias=True
            )
        if self.channel is ChannelType.BGR:
            x = x[:, [2, 1, 0]]
        x = (x - self._mean) / self._std
        out = self.backbone(x)
        return out.flatten(1) if out.dim() > 2 else out


class HardModelConv(nn.Module):
    """Wrap one ``Conv2d`` so that later attack iterations add ``eta * sign(grad_prev)`` to its
    output (paper Eqs. 14-15); the first iteration is a plain convolution.
    """

    def __init__(self, conv: nn.Conv2d, eta: float) -> None:
        super().__init__()
        self.conv = conv
        self.eta = float(eta)
        self.grad_prev: torch.Tensor | None = None

    def reset(self) -> None:
        self.grad_prev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if out.requires_grad:
            out.register_hook(self._capture_grad)
        if self.grad_prev is not None:
            self.grad_prev = self.grad_prev.to(device=out.device, dtype=out.dtype)
            if self.grad_prev.shape == out.shape:
                out = out + self.eta * torch.sign(self.grad_prev)
        return out

    def _capture_grad(self, grad: torch.Tensor) -> None:
        self.grad_prev = grad.detach()


def reinit_backbone(backbone: nn.Module) -> nn.Module:
    """Randomly re-initialize a copied backbone in place, by calling ``reset_parameters()`` on
    every submodule that defines it.
    """

    for module in backbone.modules():
        if module is backbone:
            continue
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    return backbone


def wrap_conv_layers(backbone: nn.Module, eta: float) -> List[HardModelConv]:
    """Replace every ``Conv2d`` in ``backbone`` with a :class:`HardModelConv` and return the
    wrappers, so the caller can reset their stored gradients before each attack.
    """
    wrappers: List[HardModelConv] = []
    for name, module in list(backbone.named_modules()):
        if isinstance(module, nn.Conv2d):
            wrapper = HardModelConv(module, eta)
            parent, leaf = _split_module_path(backbone, name)
            setattr(parent, leaf, wrapper)
            wrappers.append(wrapper)
    return wrappers


def unwrap_conv_layers(backbone: nn.Module) -> None:
    """Replace every :class:`HardModelConv` in ``backbone`` with its inner ``Conv2d``, to export
    plain weights.
    """
    for name, module in list(backbone.named_modules()):
        if isinstance(module, HardModelConv):
            parent, leaf = _split_module_path(backbone, name)
            setattr(parent, leaf, module.conv)


def _split_module_path(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Return (parent module, leaf attribute name) for a dotted module path."""
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
