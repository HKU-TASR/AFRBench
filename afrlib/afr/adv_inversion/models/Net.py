import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms
#
# from .CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor  #导入外部模块用于面部解析
from .stylegan2.model import Generator #导入stylegan2生成器模型
from ..utils.drive import download_weight #导入工具函数，用于下载模型权重

transform_to_256 = transforms.Compose([
    transforms.Resize((256, 256)),
])

transform_to_512 = transforms.Compose([
    transforms.Resize((512, 512)),
])

transform_to_1024 = transforms.Compose([
    transforms.Resize((1024, 1024)),
])

__all__ = ['Net', 'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200', 'FeatureEncoderMult',
           'IBasicBlock', 'conv1x1', 'get_segmentation']# 定义模块的导出内容，表示哪些类和函数将会被导出


class Net(nn.Module):
    """
    包含生成器模型、PCA 模型和面部解析的核心网络类
    """
    def __init__(self, opts):
        super(Net, self).__init__()#
        self.opts = opts
        self.generator = Generator(opts.size, opts.latent, opts.n_mlp, channel_multiplier=opts.channel_multiplier) #初始化stylegan2生成器
        self.cal_layer_num()#计算生成器的层数
        self.load_weights()#加载预训练权重
        self.load_PCA_model()#加载PCA模型
        # FaceParsing_tensor.parsing_img()#预加载面部解析模型

    def load_weights(self):
        """
               加载生成器的预训练权重
               """
        if not os.path.exists(self.opts.ckpt):
            print('Downloading StyleGAN2 checkpoint: {}'.format(self.opts.ckpt))
            download_weight(self.opts.ckpt)

        print('Loading StyleGAN2 from checkpoint: {}'.format(self.opts.ckpt))
        checkpoint = torch.load(self.opts.ckpt, map_location=self.opts.device)
        device = self.opts.device
        self.generator.load_state_dict(checkpoint['g_ema']) # 加载生成器的权重
        self.latent_avg = checkpoint['latent_avg']# 获取平均潜在向量
        self.generator.to(device)
        self.latent_avg = self.latent_avg.to(device) # 将平均潜在向量移到指定设备

        for param in self.generator.parameters():
            param.requires_grad = False  # 冻结生成器的所有参数
        self.generator.eval()  # 将生成器设置为评估模式

    def build_PCA_model(self, PCA_path):
        """
               构建 PCA 模型，用于生成潜在空间的主成分分析
               """
        with torch.no_grad():
            latent = torch.randn((1000000, 512), dtype=torch.float32)# 生成100w个潜在向量
            # latent = torch.randn((10000, 512), dtype=torch.float32)
            self.generator.style.cpu() # 将生成器的样式网络移到 CPU 上
            pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()# 通过样式网络生成脉冲空间
            self.generator.style.to(self.opts.device)#将样式网络移回指定设备

        from ..utils.PCA_utils import IPCAEstimator #导入PCA工具类

        transformer = IPCAEstimator(512) #初始化PCA估计器
        X_mean = pulse_space.mean(0)#计算脉冲空间的均值
        transformer.fit(pulse_space - X_mean)#拟合PCA模型
        X_comp, X_stdev, X_var_ratio = transformer.get_components()# 获取PCA组件
        np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)# 保存 PCA 模型到文件

    def load_PCA_model(self):
        """
        加载或构建 PCA 模型
        """
        device = self.opts.device

        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz' # 构建 PCA 模型文件的路径

        if not os.path.isfile(PCA_path): # 如果 PCA 模型文件不存在，则构建 PCA 模型
            self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().to(device)
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().to(device)
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().to(device)

    # def make_noise(self):
    #     noises_single = self.generator.make_noise()
    #     noises = []
    #     for noise in noises_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())
    #
    #     return noises

    def cal_layer_num(self):
        """
        根据图像尺寸计算生成器的层数
        """
        if self.opts.size == 1024:
            self.layer_num = 18
        elif self.opts.size == 512:
            self.layer_num = 16
        elif self.opts.size == 256:
            self.layer_num = 14

        self.S_index = self.layer_num - 11 # 7、5、3

        return

    def cal_p_norm_loss(self, latent_in):
        """
        计算潜在向量的 p 范数损失
        """
        latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
            self.X_comp.T.unsqueeze(0)) / self.X_stdev# 计算 p 范数
        p_norm_loss = self.opts.p_norm_lambda * (latent_p_norm.pow(2).mean())# 计算 p 范数损失
        return p_norm_loss

    def cal_l_F(self, latent_F, F_init):
        return self.opts.l_F_lambda * (latent_F - F_init).pow(2).mean() # 计算潜在特征的均方误差损失


def get_segmentation(img_rgb, resize=True):
    """
    对输入的 RGB 图像进行面部解析，生成分割掩码
    """
    parsing, _ = FaceParsing_tensor.parsing_img(img_rgb) #进行面部解析
    parsing = FaceParsing_tensor.swap_parsing_label_to_celeba_mask(parsing)# 将解析结果转换为 CelebA 的标签
    mask_img = parsing.long()[None, None, ...]# 转换为掩码图像
    if resize:
        mask_img = transforms.functional.resize(mask_img, (256, 256),
                                                interpolation=transforms.InterpolationMode.NEAREST)# 如果需要，调整掩码大小
    return mask_img# 返回掩码图像

# 定义用于特征编码器的卷积核大小和步幅
fs_kernals = {
    0: (12, 12),
    1: (12, 12),
    2: (6, 6),
    3: (6, 6),
    4: (3, 3),
    5: (3, 3),
    6: (3, 3),
    7: (3, 3),
}

fs_strides = {
    0: (7, 7),
    1: (7, 7),
    2: (4, 4),
    3: (4, 4),
    4: (2, 2),
    5: (2, 2),
    6: (1, 1),
    7: (1, 1),
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05, )
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05, )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = []
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            out.append(x)
            x = self.layer2(x)
            out.append(x)
            x = self.layer3(x)
            out.append(x)
            x = self.layer4(x)
            out.append(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)

        if return_features:
            out.append(x)
            return out
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)


class FeatureEncoder(nn.Module):
    def __init__(self, n_styles=18, opts=None, residual=False,
                 use_coeff=False, resnet_layer=None,
                 video_input=False, f_maps=512, stride=(1, 1)):
        super(FeatureEncoder, self).__init__()

        resnet50 = iresnet50()
        resnet50.load_state_dict(torch.load(opts.arcface_model_path, map_location='cpu'))

        # input conv layer
        if video_input:
            self.conv = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                *list(resnet50.children())[1:3]
            )
        else:
            self.conv = nn.Sequential(*list(resnet50.children())[:3])

        # define layers
        self.block_1 = list(resnet50.children())[3]  # 15-18
        self.block_2 = list(resnet50.children())[4]  # 10-14
        self.block_3 = list(resnet50.children())[5]  # 5-9
        self.block_4 = list(resnet50.children())[6]  # 1-4
        self.content_layer = nn.Sequential(
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.PReLU(num_parameters=512),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.styles = nn.ModuleList()
        for i in range(n_styles):
            self.styles.append(nn.Linear(960 * 9, 512))

    def apply_head(self, x):
        latents = []
        for i in range(len(self.styles)):
            latents.append(self.styles[i](x))#生成每一层的风格向量
        out = torch.stack(latents, dim=1)# # 将所有风格向量堆叠在一起
        return out

    def forward(self, x):
        latents = []
        features = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        content = self.content_layer(x)
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        return self.apply_head(x), content


class FeatureEncoderMult(FeatureEncoder):
    def __init__(self, fs_layers=(5,), ranks=None, **kwargs):
        super().__init__(**kwargs)

        self.fs_layers = fs_layers
        self.content_layer = nn.ModuleList()
        self.ranks = ranks
        shift = 0 if max(fs_layers) <= 7 else 2
        scale = 1 if max(fs_layers) <= 7 else 2
        for i in range(len(fs_layers)):
            if ranks is not None:
                stride, kern = ranks_data[ranks[i] - shift]
                layer1 = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(fs_kernals[fs_layers[i] - shift][0], kern),
                              stride=(fs_strides[fs_layers[i] - shift][0], stride),
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                layer2 = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                              bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(kern, fs_kernals[fs_layers[i] - shift][1]),
                              stride=(stride, fs_strides[fs_layers[i] - shift][1]),
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True)
                )
                layer = nn.ModuleList([layer1, layer2])
            else:
                layer = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=fs_kernals[fs_layers[i] - shift],
                              stride=fs_strides[fs_layers[i] - shift],
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
            self.content_layer.append(layer)

    def forward(self, x):
        x = transform_to_256(x)
        features = []
        content = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        if max(self.fs_layers) > 7:
            for layer in self.content_layer:
                if self.ranks is not None:
                    mat1 = layer[0](x)
                    mat2 = layer[1](x)
                    content.append(torch.matmul(mat1, mat2))
                else:
                    content.append(layer(x))
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        if len(content) == 0:
            for layer in self.content_layer:
                if self.ranks is not None:
                    mat1 = layer[0](x)
                    mat2 = layer[1](x)
                    content.append(torch.matmul(mat1, mat2))
                else:
                    content.append(layer(x))
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        return self.apply_head(x), content

class FeatureEncoderMult2(FeatureEncoder):
    def __init__(self, fs_layers=(5,), ranks=None, **kwargs):
        super().__init__(**kwargs)

        self.fs_layers = fs_layers
        self.content_layer = nn.ModuleList()
        self.ranks = ranks
        shift = 0 if max(fs_layers) <= 7 else 2
        scale = 1 if max(fs_layers) <= 7 else 2
        for i in range(len(fs_layers)):
            if ranks is not None:
                stride, kern = ranks_data[ranks[i] - shift]
                layer1 = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(fs_kernals[fs_layers[i] - shift][0], kern),
                              stride=(fs_strides[fs_layers[i] - shift][0], stride),
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
                layer2 = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                              bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    nn.Conv2d(512, 512, kernel_size=(kern, fs_kernals[fs_layers[i] - shift][1]),
                              stride=(stride, fs_strides[fs_layers[i] - shift][1]),
                              padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                   track_running_stats=True)
                )
                layer = nn.ModuleList([layer1, layer2])
            else:
                layer = nn.Sequential(
                    nn.BatchNorm2d(256 // scale, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Conv2d(256 // scale, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.PReLU(num_parameters=512),
                    # nn.Conv2d(512, 512, kernel_size=fs_kernals[fs_layers[i] - shift],
                    nn.Conv2d(512, 256, kernel_size=fs_kernals[fs_layers[i] - shift],
                              stride=fs_strides[fs_layers[i] - shift],
                              padding=(1, 1), bias=False),
                    # nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
            self.content_layer.append(layer)

    def forward(self, x):
        x = transform_to_512(x)
        # x = transform_to_1024(x)
        features = []
        content = []
        x = self.conv(x)
        x = self.block_1(x)
        features.append(self.avg_pool(x))
        x = self.block_2(x)
        if max(self.fs_layers) > 7:
            for layer in self.content_layer:
                if self.ranks is not None:
                    mat1 = layer[0](x)
                    mat2 = layer[1](x)
                    content.append(torch.matmul(mat1, mat2))
                else:
                    content.append(layer(x))
        features.append(self.avg_pool(x))
        x = self.block_3(x)
        if len(content) == 0:
            for layer in self.content_layer:
                if self.ranks is not None:
                    mat1 = layer[0](x)
                    mat2 = layer[1](x)
                    content.append(torch.matmul(mat1, mat2))
                else:
                    content.append(layer(x))
        features.append(self.avg_pool(x))
        x = self.block_4(x)
        features.append(self.avg_pool(x))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        return self.apply_head(x), content

def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[: len(name) + 1] == name + '.'}
    return d_filt
