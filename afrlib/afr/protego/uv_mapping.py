"""UV mapping: project facial images onto the canonical FLAME UV space.

Reference Code: https://github.com/HKU-TASR/Protego
"""
import os
from typing import Tuple

import torch
import torch.nn.functional as F

from afrlib.utils.assets import get_asset
from .smirk.src.smirk_encoder import SmirkEncoder
from .smirk.src.FLAME.FLAME import FLAME
from .flame_renderer import Renderer

SMIRK_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smirk")


def init_smirk(smirk_ckpts_path: str, device: torch.device) -> SmirkEncoder:
    """Load the SMIRK encoder from a full SMIRK checkpoint (encoder + generator)."""
    smirk_encoder = SmirkEncoder().to(device)
    ckpts = torch.load(smirk_ckpts_path, map_location=device, weights_only=False)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in ckpts.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()
    return smirk_encoder


class UVGenerator(object):
    """SMIRK encoder + FLAME + renderer pipeline that maps images to UV grids."""

    def __init__(self, device: torch.device):
        self.device = device
        self.img_sz = 224
        self.smirk_encoder = init_smirk(get_asset("protego/SMIRK_em1.pt"), device)
        self.flame = FLAME(
            flame_model_path=get_asset("protego/generic_model.pkl"), base_path=SMIRK_BASE_PATH
        ).to(device)
        self.renderer = Renderer(smirk_path=SMIRK_BASE_PATH).to(device)

    def forward(self, imgs: torch.Tensor, batch_size: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate UV coordinates for the input images.

        Args:
            imgs (torch.Tensor): Input images in the shape of (B, 3, H, H), RGB, range [0, 1], dtype torch.float32.
            batch_size (int): The batch size for processing images. Default is -1, which means all images are processed in one batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - uv_grid (torch.Tensor): UV coordinates in the shape of (B, 224, 224, 2), zero where no mesh is rasterized.
                - visibility_mask (torch.Tensor): Binary visibility mask in the shape of (B, 1, 224, 224).
        """
        if batch_size <= 0:
            return self.run(imgs)
        total_num = imgs.shape[0]
        uvs, bin_masks = [], []
        for start_idx in range(0, total_num, batch_size):
            end_idx = min(start_idx + batch_size, total_num)
            _uvs, _bin_masks = self.run(imgs[start_idx:end_idx])
            uvs.append(_uvs.cpu())
            bin_masks.append(_bin_masks.cpu())
        return torch.cat(uvs, dim=0).to(self.device), torch.cat(bin_masks, dim=0).to(self.device)

    def run(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-batch UV generation; see :meth:`forward` for shapes."""
        _, _, H, _ = imgs.shape
        if H != self.img_sz:
            resz_imgs = F.interpolate(imgs, size=(self.img_sz, self.img_sz), mode='bilinear', align_corners=False)
        else:
            resz_imgs = imgs.clone()
        resz_imgs = resz_imgs.to(self.device)
        smirk_outputs = self.smirk_encoder(resz_imgs)
        flame_outputs = self.flame.forward(smirk_outputs)
        renderer_output = self.renderer.forward(flame_outputs['vertices'], smirk_outputs['cam'],
                                                landmarks_fan=flame_outputs['landmarks_fan'],
                                                landmarks_mp=flame_outputs['landmarks_mp'])
        uv_grid = renderer_output['uv_grid']                    # [N, H, W, 2]
        visibility_mask = renderer_output['visibility_mask']    # [N, H, W, 1]
        return uv_grid, visibility_mask.permute(0, 3, 1, 2)
