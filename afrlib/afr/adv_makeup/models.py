"""Adv-Makeup generator and discriminator networks.

Reference Code: https://github.com/TencentYoutuResearch/Adv-Makeup (commit 359af7e; academic research only, see LICENSE)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


_ENCODER_CONFIG = (
    ("conv2d", (64, 3, 5, 5, 2, 2)),
    ("lrelu", (0.2, True)),
    ("conv2d", (128, 64, 5, 5, 2, 2)),
    ("bn", (128,)),
)

_DECODER_CONFIG = (
    ("relu", (True,)),
    ("conv2d", (128, 128, 3, 3, 1, 1)),
    ("bn", (128,)),
    ("cat1", ()),
    ("relu", (True,)),
    ("up", (2, "bilinear", True)),
    ("conv2d", (64, 256, 3, 3, 1, 1)),
    ("bn", (64,)),
    ("cat2", ()),
    ("relu", (True,)),
    ("up", (2, "bilinear", True)),
    ("conv2d", (64, 128, 3, 3, 1, 1)),
    ("bn", (64,)),
    ("relu", (True,)),
    ("conv2d", (3, 64, 3, 3, 1, 1)),
    ("tanh", ()),
)


class Encoder(nn.Module):
    """Two-level encoder with explicit parameters for first-order fast weights."""

    def __init__(self, input_dim: int = 3) -> None:
        super().__init__()
        if input_dim != 3:
            raise ValueError("The released Adv-Makeup encoder requires RGB input")

        self.vars_enc = nn.ParameterList()
        self.vars_bn_enc = nn.ParameterList()
        for name, values in _ENCODER_CONFIG:
            if name == "conv2d":
                weight = nn.Parameter(torch.ones(*values[:4]))
                init.kaiming_normal_(weight)
                self.vars_enc.extend(
                    (weight, nn.Parameter(torch.zeros(values[0])))
                )
            elif name == "bn":
                channels = values[0]
                self.vars_enc.extend(
                    (
                        nn.Parameter(torch.ones(channels)),
                        nn.Parameter(torch.zeros(channels)),
                    )
                )
                self.vars_bn_enc.extend(
                    (
                        nn.Parameter(
                            torch.zeros(channels), requires_grad=False
                        ),
                        nn.Parameter(
                            torch.ones(channels), requires_grad=False
                        ),
                    )
                )

    def fast_parameters(self) -> list[nn.Parameter]:
        """Parameters adapted by the released first-order meta step."""

        return list(self.vars_enc)

    def forward(
        self,
        inputs: torch.Tensor,
        parameters: Optional[Sequence[torch.Tensor]] = None,
        *,
        bn_training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.vars_enc if parameters is None else parameters
        parameter_index = 0
        bn_index = 0
        first_feature: Optional[torch.Tensor] = None

        for layer_index, (name, config) in enumerate(_ENCODER_CONFIG):
            if name == "conv2d":
                weight, bias = (
                    values[parameter_index],
                    values[parameter_index + 1],
                )
                inputs = F.conv2d(
                    inputs,
                    weight,
                    bias,
                    stride=config[4],
                    padding=config[5],
                )
                parameter_index += 2
            elif name == "bn":
                weight, bias = (
                    values[parameter_index],
                    values[parameter_index + 1],
                )
                running_mean, running_var = (
                    self.vars_bn_enc[bn_index],
                    self.vars_bn_enc[bn_index + 1],
                )
                inputs = F.batch_norm(
                    inputs,
                    running_mean,
                    running_var,
                    weight=weight,
                    bias=bias,
                    training=bn_training,
                )
                parameter_index += 2
                bn_index += 2
            elif name == "lrelu":
                # Upstream configures "lrelu" but dispatches on "leakyrelu", so
                # this activation is skipped there; it is applied here.
                inputs = F.leaky_relu(
                    inputs,
                    negative_slope=config[0],
                    inplace=config[1],
                )
            if layer_index == 0:
                first_feature = inputs

        assert first_feature is not None
        return first_feature, inputs


class Decoder(nn.Module):
    """U-Net-like decoder accepting one-step fast weights."""

    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        if output_dim != 3:
            raise ValueError("The released Adv-Makeup decoder requires RGB output")

        self.vars_dec = nn.ParameterList()
        self.vars_bn_dec = nn.ParameterList()
        for name, values in _DECODER_CONFIG:
            if name == "conv2d":
                weight = nn.Parameter(torch.ones(*values[:4]))
                init.kaiming_normal_(weight)
                self.vars_dec.extend(
                    (weight, nn.Parameter(torch.zeros(values[0])))
                )
            elif name == "bn":
                channels = values[0]
                self.vars_dec.extend(
                    (
                        nn.Parameter(torch.ones(channels)),
                        nn.Parameter(torch.zeros(channels)),
                    )
                )
                self.vars_bn_dec.extend(
                    (
                        nn.Parameter(
                            torch.zeros(channels), requires_grad=False
                        ),
                        nn.Parameter(
                            torch.ones(channels), requires_grad=False
                        ),
                    )
                )

    def fast_parameters(self) -> list[nn.Parameter]:
        return list(self.vars_dec)

    def forward(
        self,
        first_feature: torch.Tensor,
        second_feature: torch.Tensor,
        parameters: Optional[Sequence[torch.Tensor]] = None,
        *,
        bn_training: bool = True,
    ) -> torch.Tensor:
        values = self.vars_dec if parameters is None else parameters
        inputs = second_feature
        parameter_index = 0
        bn_index = 0
        decoder_first: Optional[torch.Tensor] = None
        decoder_second: Optional[torch.Tensor] = None

        for layer_index, (name, config) in enumerate(_DECODER_CONFIG):
            if name == "conv2d":
                weight, bias = (
                    values[parameter_index],
                    values[parameter_index + 1],
                )
                inputs = F.conv2d(
                    inputs,
                    weight,
                    bias,
                    stride=config[4],
                    padding=config[5],
                )
                parameter_index += 2
            elif name == "bn":
                weight, bias = (
                    values[parameter_index],
                    values[parameter_index + 1],
                )
                running_mean, running_var = (
                    self.vars_bn_dec[bn_index],
                    self.vars_bn_dec[bn_index + 1],
                )
                inputs = F.batch_norm(
                    inputs,
                    running_mean,
                    running_var,
                    weight=weight,
                    bias=bias,
                    training=bn_training,
                )
                parameter_index += 2
                bn_index += 2
            elif name == "relu":
                inputs = F.relu(inputs, inplace=config[0])
            elif name == "cat1":
                assert decoder_first is not None
                inputs = torch.cat((decoder_first, second_feature), dim=1)
            elif name == "cat2":
                assert decoder_second is not None
                inputs = torch.cat((decoder_second, first_feature), dim=1)
            elif name == "up":
                inputs = F.interpolate(
                    inputs,
                    scale_factor=config[0],
                    mode=config[1],
                    align_corners=config[2],
                )
            elif name == "tanh":
                inputs = torch.tanh(inputs)

            if layer_index == 2:
                decoder_first = inputs
            elif layer_index == 7:
                decoder_second = inputs

        return inputs


class _LeakyReLUConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=True,
            ),
        ]
        if instance_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        layers.append(nn.LeakyReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


class Discriminator(nn.Module):
    """Released patch discriminator for 160x420 eye strips."""

    def __init__(self, input_dim: int = 3, width: int = 64) -> None:
        super().__init__()
        channels = width * 2
        self.model = nn.Sequential(
            _LeakyReLUConv2d(
                input_dim,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                instance_norm=True,
            ),
            _LeakyReLUConv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                instance_norm=True,
            ),
            _LeakyReLUConv2d(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                instance_norm=True,
            ),
            _LeakyReLUConv2d(
                channels,
                channels,
                kernel_size=1,
                stride=1,
            ),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).reshape(-1)


def initialize_discriminator(
    discriminator: Discriminator, gain: float = 0.02
) -> Discriminator:
    """Apply the released initialization: normal weights with std ``gain``, zero biases."""

    for module in discriminator.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init.normal_(module.weight, 0.0, gain)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            init.normal_(module.weight, 1.0, gain)
            init.constant_(module.bias, 0.0)
    return discriminator
