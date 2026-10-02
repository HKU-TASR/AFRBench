"""Vendored (verbatim) from Adv-Inversion ``models/Encoders.py``.

Only the classes on the Adv-Inversion-IDF inference path are kept:
``FeatureiResnet`` and ``PostProcessModel_First`` (the enhanced FSE). The unused
siblings from the upstream file (``PostProcessModel``, ``PostProcessModel_attension``,
``FeatureEncoderMult2``, ``ClipModel``) and their heavy imports (``clip``,
``models.attension``) are dropped. The kept classes are unchanged except that
``PostProcessModel_First`` now takes the two pretrained-weight paths as arguments
(they were hardcoded relative paths upstream) so the encoder can be built from any
working directory.
"""
import argparse

import torch
import torch.nn as nn

from .Net import FeatureEncoderMult, IBasicBlock, conv1x1
from .stylegan2.model import PixelNorm


class FeatureiResnet(nn.Module):
    def __init__(self, blocks, inplanes=1024):
        super().__init__()

        self.res_blocks = {}

        for n, block in enumerate(blocks, start=1):
            planes, num_blocks = block

            for k in range(1, num_blocks + 1):
                downsample = None
                if inplanes != planes:
                    downsample = nn.Sequential(conv1x1(inplanes, planes, 1), nn.BatchNorm2d(planes, eps=1e-05, ), )

                self.res_blocks[f'res_block_{n}_{k}'] = IBasicBlock(inplanes, planes, 1, downsample, 1, 64, 1)
                inplanes = planes

        self.res_blocks = nn.ModuleDict(self.res_blocks)

    def forward(self, x):
        for module in self.res_blocks.values():
            x = module(x)
        return x


class PostProcessModel_First(nn.Module):
    def __init__(self, arcface_model_path="pretrained_models/ArcFace/backbone_ir50.pth",
                 latent_avg_path="pretrained_models/latent_avg.pt"):
        super().__init__()
        self.encoder_face = FeatureEncoderMult(fs_layers=[9], opts=argparse.Namespace(
            **{'arcface_model_path': arcface_model_path}))

        self.latent_avg = torch.load(latent_avg_path, map_location='cpu')
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.to_latent = nn.Sequential(nn.Linear(1024, 1024), nn.LayerNorm([1024]), nn.LeakyReLU(),
                                       nn.Linear(1024, 512))  # 初始化像素归一化
        self.pixelnorm = PixelNorm()

    def forward(self, source, target):
        s_face, [f_face] = self.encoder_face(source)
        return self.latent_avg + s_face, f_face   # 返回潜在平均值和特征
