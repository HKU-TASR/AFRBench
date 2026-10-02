"""AdvCloak generator and patch discriminator for 112x112 faces in ``[-1, 1]``.

Reference Code: https://github.com/liuxuannan/AdvCloak (CC BY-SA 4.0, adapted; see LICENSE)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Encoder / bottleneck / decoder mapping a face to a perturbation in [-1, 1]."""

    def __init__(self, input_nc: int = 3, image_nc: int = 3) -> None:
        super(Generator, self).__init__()
        encoder_lis = [
            # 3 x 112 x 112
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64 x 112 x 112
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128 x 56 x 56
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 256 x 28 x 28
        ]

        bottle_neck_lis = [ResnetBlock(256), ResnetBlock(256), ResnetBlock(256)]

        decoder_lis = [
            # 256 x 28 x 28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([128, 56, 56]),
            nn.ReLU(),
            # 128 x 56 x 56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LayerNorm([64, 112, 112]),
            nn.ReLU(),
            # 64 x 112 x 112
            nn.ConvTranspose2d(64, image_nc, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh(),
            # image_nc x 112 x 112
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    """Patch discriminator on 112x112 faces; returns the flattened 4x4 logit maps (16 per image)."""

    def __init__(self, image_nc: int = 3) -> None:
        super(Discriminator, self).__init__()
        model = [
            # 3 x 112 x 112
            nn.Conv2d(image_nc, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.995),
            # 32 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.995),
            # 64 x 28 x 28
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.995),
            nn.LeakyReLU(0.2),
            # 128 x 14 x 14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.995),
            nn.LeakyReLU(0.2),
            # 256 x 7 x 7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512, eps=0.001, momentum=0.995),
            nn.LeakyReLU(0.2),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, 1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return torch.reshape(output, [-1, 1]).squeeze()


class ResnetBlock(nn.Module):
    """Residual block with reflection padding and a norm layer (default BatchNorm)."""

    def __init__(
        self,
        dim: int,
        padding_type: str = "reflect",
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        use_bias: bool = False,
    ) -> None:
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)
