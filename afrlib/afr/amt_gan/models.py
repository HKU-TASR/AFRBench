# Adapted from AMT-GAN (https://github.com/CGCL-codes/AMT-GAN).
# Copyright (c) 2022 CGCL-codes
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Checkpoint-compatible neural networks from the official AMT-GAN release."""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "Discriminator",
    "Generator",
    "GetMatrix",
    "H_RRDB",
    "NONLocalBlock2D",
    "Regularizer",
    "ResidualBlock",
    "ResidualDenseBlock_5C",
    "RRDB",
    "SpectralNorm",
]


def _make_layer(block: Callable[[], nn.Module], n_layers: int) -> nn.Sequential:
    return nn.Sequential(*(block() for _ in range(n_layers)))


class ResidualBlock(nn.Module):
    """Shape-preserving residual block used by AMT-GAN's PNet and TNet."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        net_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if net_mode in ("p", None):
            use_affine = True
        elif net_mode == "t":
            use_affine = False
        else:
            raise ValueError(f"Unsupported net_mode: {net_mode!r}")

        self.main = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
        )

    def forward(self, x: Tensor) -> Tensor:
        # (N, dim_in, H, W) -> (N, dim_out, H, W); AMT-GAN uses dim_in == dim_out.
        return x + self.main(x)


class GetMatrix(nn.Module):
    """Predict the one-channel affine makeup matrices gamma and beta."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.get_gamma = nn.Conv2d(
            dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.get_beta = nn.Conv2d(
            dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # x: (N, dim_in, H, W); gamma/beta: (N, dim_out, H, W).
        return x, self.get_gamma(x), self.get_beta(x)


class NONLocalBlock2D(nn.Module):
    """Apply a precomputed spatial correspondence matrix to a feature map."""

    def __init__(self) -> None:
        super().__init__()
        # Unused in forward, but its weight and bias are in the released G.pth.
        self.g = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, source: Tensor, weight: Tensor) -> Tensor:
        # source: (N, 1, H, W), weight: sparse/dense (N, HW, HW).
        batch_size = source.size(0)
        g_source = source.view(batch_size, 1, -1).permute(0, 2, 1)
        dense_weight = weight.to_dense() if weight.is_sparse else weight
        y = torch.bmm(dense_weight, g_source)
        y = y.permute(0, 2, 1).contiguous()
        return y.view(batch_size, 1, *source.size()[2:])


class ResidualDenseBlock_5C(nn.Module):
    """Five-convolution residual dense block from ESRGAN."""

    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # (N, nf, H, W) -> (N, nf, H, W).
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, nf: int, gc: int = 32) -> None:
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: Tensor) -> Tensor:
        # (N, nf, H, W) -> (N, nf, H, W).
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class Generator(nn.Module):
    """AMT-GAN's PNet makeup encoder and TNet transfer decoder."""

    _ATTENTION_SIZE = 64
    _MAKEUP_PARTS = 3

    def __init__(self) -> None:
        super().__init__()

        self.pnet_in = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        curr_dim = 64
        for i in range(2):
            layers = nn.Sequential(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            )
            setattr(self, f"pnet_down_{i + 1}", layers)
            curr_dim *= 2

        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetMatrix(curr_dim, 1)
        for i in range(3):
            setattr(
                self,
                f"pnet_bottleneck_{i + 1}",
                ResidualBlock(curr_dim, curr_dim, net_mode="p"),
            )

        self.tnet_in_conv = nn.Conv2d(
            3, 64, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        curr_dim = 64
        for i in range(2):
            setattr(
                self,
                f"tnet_down_conv_{i + 1}",
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            setattr(
                self,
                f"tnet_down_spade_{i + 1}",
                nn.InstanceNorm2d(curr_dim * 2, affine=False),
            )
            setattr(self, f"tnet_down_relu_{i + 1}", nn.ReLU(inplace=True))
            curr_dim *= 2

        for i in range(6):
            setattr(
                self,
                f"tnet_bottleneck_{i + 1}",
                ResidualBlock(curr_dim, curr_dim, net_mode="t"),
            )

        for i in range(2):
            setattr(
                self,
                f"tnet_up_conv_{i + 1}",
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            setattr(
                self,
                f"tnet_up_spade_{i + 1}",
                nn.InstanceNorm2d(curr_dim // 2, affine=False),
            )
            setattr(self, f"tnet_up_relu_{i + 1}", nn.ReLU(inplace=True))
            curr_dim //= 2

        self.tnet_out = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def atten_feature(
        mask_s: Tensor,
        weight: Tensor,
        gamma_s: Tensor,
        beta_s: Tensor,
        atten_module_g: NONLocalBlock2D,
        atten_module_b: NONLocalBlock2D,
    ) -> Tuple[Tensor, Tensor]:
        """Transport three masked makeup regions into content-image coordinates."""
        channel_num = gamma_s.shape[1]
        # mask_s: (3, 1, Hm, Wm); gamma_s/beta_s: (1, C, H, W).
        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(
            1, channel_num, 1, 1
        )
        gamma_s = gamma_s.repeat(3, 1, 1, 1) * mask_s_re
        beta_s = beta_s.repeat(3, 1, 1, 1) * mask_s_re

        gamma = atten_module_g(gamma_s, weight)
        beta = atten_module_b(beta_s, weight)
        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(
        self,
        mask_c: Tensor,
        mask_s: Tensor,
        fea_c: Tensor,
        fea_s: Tensor,
        diff_c: Tensor,
        diff_s: Tensor,
    ) -> Tensor:
        """Build sparse (3, 4096, 4096) makeup-region correspondence weights."""
        spatial_size = self._ATTENTION_SIZE
        hw = spatial_size * spatial_size
        channel_num = fea_s.shape[1]

        # Masks have one batch entry per lip/skin/eye region.
        mask_c_re = F.interpolate(mask_c, size=spatial_size).repeat(
            1, channel_num, 1, 1
        )
        fea_c = fea_c.repeat(self._MAKEUP_PARTS, 1, 1, 1) * mask_c_re
        mask_s_re = F.interpolate(mask_s, size=spatial_size).repeat(
            1, channel_num, 1, 1
        )
        fea_s = fea_s.repeat(self._MAKEUP_PARTS, 1, 1, 1) * mask_s_re

        theta_input = torch.cat((fea_c * 0.01, diff_c), dim=1)
        phi_input = torch.cat((fea_s * 0.01, diff_s), dim=1)
        theta_target = theta_input.view(self._MAKEUP_PARTS, -1, hw).permute(
            0, 2, 1
        )
        phi_source = phi_input.view(self._MAKEUP_PARTS, -1, hw)
        weight = torch.bmm(theta_target, phi_source)

        with torch.no_grad():
            # Clone to match the released implementation's CUDA-memory lifetime.
            weight_ind = weight.detach().nonzero().long().permute(1, 0).clone()
            # The CUDA cache is not emptied here, so freed blocks stay with this
            # process for the next memory peak.

        weight = F.softmax(weight * 200, dim=-1)
        values = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        # Indices from nonzero() satisfy the sparse invariants by construction,
        # so the costly global check is disabled.
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            return torch.sparse_coo_tensor(
                weight_ind,
                values,
                size=(self._MAKEUP_PARTS, hw, hw),
                device=weight.device,
                dtype=weight.dtype,
                check_invariants=False,
            )

    def forward(
        self,
        c: Tensor,
        s: Tensor,
        mask_c: Tensor,
        mask_s: Tensor,
        diff_c: Tensor,
        diff_s: Tensor,
        gamma: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
        ret: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        """Transfer makeup from ``s`` to ``c``: ``(1, 3, 256, 256)`` images, three part masks, and
        three 136-channel 64x64 distance maps each (a leading loader dimension is accepted).
        """
        c, s, mask_c, mask_s, diff_c, diff_s = [
            x.squeeze(0) if x.ndim == 5 else x
            for x in (c, s, mask_c, mask_s, diff_c, diff_s)
        ]

        c_tnet = self.tnet_in_relu(self.tnet_in_spade(self.tnet_in_conv(c)))
        s = self.pnet_in(s)

        for i in range(2):
            if gamma is None:
                s = getattr(self, f"pnet_down_{i + 1}")(s)
            c_tnet = getattr(self, f"tnet_down_conv_{i + 1}")(c_tnet)
            c_tnet = getattr(self, f"tnet_down_spade_{i + 1}")(c_tnet)
            c_tnet = getattr(self, f"tnet_down_relu_{i + 1}")(c_tnet)

        for i in range(6):
            if i == 3:
                if gamma is None:
                    s, gamma, beta = self.simple_spade(s)
                    weight = self.get_weight(
                        mask_c, mask_s, c_tnet, s, diff_c, diff_s
                    )
                    gamma, beta = self.atten_feature(
                        mask_s,
                        weight,
                        gamma,
                        beta,
                        self.atten_bottleneck_g,
                        self.atten_bottleneck_b,
                    )
                    if ret:
                        return [gamma, beta]
                if gamma is None or beta is None:
                    raise ValueError("gamma and beta must be provided together")
                c_tnet = c_tnet * (1 + gamma) + beta

            if gamma is None and i <= 2:
                s = getattr(self, f"pnet_bottleneck_{i + 1}")(s)
            c_tnet = getattr(self, f"tnet_bottleneck_{i + 1}")(c_tnet)

        for i in range(2):
            c_tnet = getattr(self, f"tnet_up_conv_{i + 1}")(c_tnet)
            c_tnet = getattr(self, f"tnet_up_spade_{i + 1}")(c_tnet)
            c_tnet = getattr(self, f"tnet_up_relu_{i + 1}")(c_tnet)

        return self.tnet_out(c_tnet)


def _l2normalize(vector: Tensor, eps: float = 1e-12) -> Tensor:
    return vector / (vector.norm() + eps)


class _SpectralNorm:
    """AMT-GAN's spectral-normalization pre-hook, which keeps ``weight_u``, ``weight_v`` and
    ``weight_bar`` as checkpointed parameters.
    """

    name = "weight"

    def __init__(self) -> None:
        self.power_iterations = 1

    def compute_weight(self, module: nn.Module) -> Tensor:
        u = getattr(module, f"{self.name}_u")
        v = getattr(module, f"{self.name}_v")
        weight = getattr(module, f"{self.name}_bar")
        height = weight.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = _l2normalize(
                torch.mv(weight.view(height, -1).data.t(), u.data)
            )
            u.data = _l2normalize(torch.mv(weight.view(height, -1).data, v.data))
        sigma = u.dot(weight.view(height, -1).mv(v))
        return weight / sigma.expand_as(weight)

    @staticmethod
    def apply(module: nn.Module) -> "_SpectralNorm":
        hook = _SpectralNorm()
        name = hook.name
        try:
            getattr(module, f"{name}_u")
            getattr(module, f"{name}_v")
            getattr(module, f"{name}_bar")
        except AttributeError:
            weight = getattr(module, name)
            height = weight.data.shape[0]
            width = weight.view(height, -1).data.shape[1]
            u = nn.Parameter(
                weight.data.new(height).normal_(0, 1), requires_grad=False
            )
            v = nn.Parameter(
                weight.data.new(width).normal_(0, 1), requires_grad=False
            )
            weight_bar = nn.Parameter(weight.data)
            module.register_parameter(f"{name}_u", u)
            module.register_parameter(f"{name}_v", v)
            module.register_parameter(f"{name}_bar", weight_bar)

        del module._parameters[name]
        setattr(module, name, hook.compute_weight(module))
        module.register_forward_pre_hook(hook)
        return hook

    def remove(self, module: nn.Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[f"{self.name}_u"]
        del module._parameters[f"{self.name}_v"]
        del module._parameters[f"{self.name}_bar"]
        module.register_parameter(self.name, nn.Parameter(weight.data))

    def __call__(self, module: nn.Module, inputs: Tuple[Tensor, ...]) -> None:
        del inputs
        setattr(module, self.name, self.compute_weight(module))


def SpectralNorm(module: nn.Module) -> nn.Module:
    """Apply AMT-GAN's custom spectral normalization to ``module``."""
    _SpectralNorm.apply(module)
    return module


class Discriminator(nn.Module):
    """AMT-GAN PatchGAN discriminator."""

    def __init__(
        self,
        image_size: int = 128,
        conv_dim: int = 64,
        repeat_num: int = 3,
        norm: str = "SN",
    ) -> None:
        super().__init__()
        del image_size  # Kept in the public signature for upstream compatibility.
        use_spectral_norm = norm == "SN"

        def convolution(*args: int, **kwargs: object) -> nn.Module:
            layer = nn.Conv2d(*args, **kwargs)
            return SpectralNorm(layer) if use_spectral_norm else layer

        layers: List[nn.Module] = [
            convolution(3, conv_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.extend(
                [
                    convolution(
                        curr_dim,
                        curr_dim * 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.01, inplace=True),
                ]
            )
            curr_dim *= 2

        layers.extend(
            [
                convolution(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                ),
                nn.LeakyReLU(0.01, inplace=True),
            ]
        )
        curr_dim *= 2
        self.main = nn.Sequential(*layers)
        self.conv1 = convolution(
            curr_dim,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        # (N, 3, H, W), optionally with one leading loader dimension, -> patch map.
        if x.ndim == 5:
            x = x.squeeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D image tensor, got {x.ndim} dimensions")
        return self.conv1(self.main(x))


class Regularizer(nn.Module):
    """Four-RRDB image regularizer used by the released AMT-GAN solver."""

    def __init__(
        self,
        in_nc: int = 3,
        nf: int = 64,
        nb: int = 4,
        gc: int = 32,
    ) -> None:
        super().__init__()
        if nb != 4:
            raise ValueError(f"AMT-GAN Regularizer requires exactly 4 RRDBs, got {nb}")
        rrdb_factory = partial(RRDB, nf=nf, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = _make_layer(rrdb_factory, nb)
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # (N, in_nc, H, W) -> (N, 3, H, W).
        features = self.conv_first(x)
        return self.trunk_conv(self.RRDB_trunk(features))


# Name used by the upstream solver and the H.pth documentation.
H_RRDB = Regularizer
