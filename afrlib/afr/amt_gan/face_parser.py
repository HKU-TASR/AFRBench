# Adapted from AMT-GAN (https://github.com/CGCL-codes/AMT-GAN),
# Copyright (c) 2022 CGCL-codes, MIT License. See LICENSE in this package.

"""BiSeNet face parser used by the released AMT-GAN preprocessing code."""

from __future__ import annotations

from os import PathLike
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


Device = Union[str, torch.device]
Pathish = Union[str, PathLike[str]]


def _conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def _create_layer_basic(
    in_chan: int,
    out_chan: int,
    block_count: int,
    stride: int = 1,
) -> nn.Sequential:
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    layers.extend(
        BasicBlock(out_chan, out_chan, stride=1)
        for _ in range(block_count - 1)
    )
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    """ResNet-18 feature extractor with the released checkpoint key layout."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _create_layer_basic(64, 64, block_count=2)
        self.layer2 = _create_layer_basic(64, 128, block_count=2, stride=2)
        self.layer3 = _create_layer_basic(128, 256, block_count=2, stride=2)
        self.layer4 = _create_layer_basic(256, 512, block_count=2, stride=2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def get_params(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        wd_params: list[torch.Tensor] = []
        nowd_params: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.extend(module.parameters())
        return wd_params, nowd_params


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))

    def _init_weight(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight, a=1)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, n_classes: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan)
        self.conv_out = nn.Conv2d(
            mid_chan,
            n_classes,
            kernel_size=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv_out.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))

    def get_params(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        wd_params: list[torch.Tensor] = []
        nowd_params: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.extend(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan)
        self.conv_atten = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=1,
            bias=False,
        )
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv_atten.weight, a=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)
        self.conv_avg = ConvBNReLU(
            512,
            128,
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat8, feat16, feat32 = self.resnet(x)
        size8 = feat8.size()[2:]
        size16 = feat16.size()[2:]
        size32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size32, mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = F.interpolate(feat32_sum, size16, mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = F.interpolate(feat16_sum, size8, mode="nearest")
        feat16_up = self.conv_head16(feat16_up)
        return feat8, feat16_up, feat32_up

    def get_params(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        wd_params: list[torch.Tensor] = []
        nowd_params: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.extend(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(
            in_chan,
            out_chan,
            kernel_size=1,
            padding=0,
        )
        self.conv1 = nn.Conv2d(
            out_chan,
            out_chan // 4,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_chan // 4,
            out_chan,
            kernel_size=1,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv1.weight, a=1)
        nn.init.kaiming_normal_(self.conv2.weight, a=1)

    def forward(
        self,
        spatial: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.convblk(torch.cat([spatial, context], dim=1))
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        return torch.mul(feat, atten) + feat

    def get_params(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        wd_params: list[torch.Tensor] = []
        nowd_params: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.extend(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        output = self.conv_out(feat_fuse)
        output16 = self.conv_out16(feat_cp8)
        output32 = self.conv_out32(feat_cp16)
        size = (height, width)
        output = F.interpolate(
            output,
            size,
            mode="bilinear",
            align_corners=True,
        )
        output16 = F.interpolate(
            output16,
            size,
            mode="bilinear",
            align_corners=True,
        )
        output32 = F.interpolate(
            output32,
            size,
            mode="bilinear",
            align_corners=True,
        )
        return output, output16, output32

    def get_params(
        self,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        wd_params: list[torch.Tensor] = []
        nowd_params: list[torch.Tensor] = []
        lr_mul_wd_params: list[torch.Tensor] = []
        lr_mul_nowd_params: list[torch.Tensor] = []
        for child in self.children():
            child_wd, child_nowd = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params.extend(child_wd)
                lr_mul_nowd_params.extend(child_nowd)
            else:
                wd_params.extend(child_wd)
                nowd_params.extend(child_nowd)
        return (
            wd_params,
            nowd_params,
            lr_mul_wd_params,
            lr_mul_nowd_params,
        )


class FaceParser:
    """Load the released 19-class parser and remap it to AMT-GAN classes."""

    _CLASS_REMAP = (0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0)

    def __init__(self, checkpoint_path: Pathish, device: Device) -> None:
        self.device = torch.device(device)
        self.dic = torch.tensor(
            self._CLASS_REMAP,
            device=self.device,
        ).unsqueeze(1)

        net = BiSeNet(n_classes=19)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        # The released checkpoint is a plain state_dict.
        net.load_state_dict(state_dict)
        self.net = net.to(self.device).eval()

    def parse(
        self,
        image: Image.Image | np.ndarray,
    ) -> torch.Tensor:
        """Return a remapped float mask of shape ``[512, 512, 1]``."""

        array = np.array(image, copy=True)
        assert array.shape[:2] == (512, 512)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("FaceParser expects a 512x512 RGB image")

        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255.0)
        else:
            tensor = tensor.float()
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        tensor = tensor.sub(mean).div(std).unsqueeze(0).to(self.device)

        with torch.no_grad():
            parsing = self.net(tensor)[0].squeeze(0).argmax(0)
        return F.embedding(parsing, self.dic).float()


__all__ = ["FaceParser"]
