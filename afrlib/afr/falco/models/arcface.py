"""ArcFace IR-SE50 feature extractor (reference FALCO ``lib/arcface.py``).

Used only for the optional ``nn_feature="arcface"`` pairing features. The
backbone blocks are shared with the e4e encoder (``encoders/helpers.py``);
the ``Backbone`` here differs from the encoder trunk in its output head
(512-d embedding, L2-normalized).
"""
import torch
from torch import nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, PReLU, Sequential, Module

from .encoders.helpers import Flatten, bottleneck_IR, bottleneck_IR_SE, get_blocks, l2_norm


class ArcFace(nn.Module):
    def __init__(self, pretrained_path: str):
        super(ArcFace, self).__init__()
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for param in self.facenet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        return self.facenet(x)


class Backbone(Module):
    """IR/IR-SE backbone from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)."""

    def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        if input_size == 112:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512),
                                           BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(drop_ratio),
                                           Flatten(),
                                           Linear(512 * 14 * 14, 512),
                                           BatchNorm1d(512, affine=affine))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)
