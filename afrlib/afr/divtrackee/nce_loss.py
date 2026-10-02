"""Text-guided makeup loss for DivTrackee: the reference InfoNCE loss and the paper's directional loss.

Reference Code: https://github.com/fiora6/divtrackee
"""
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms

from . import clip

# The 79 ImageNet prompt templates used to average text embeddings.
imagenet_templates = [
    'a bad photo of a {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def info_nce(query, positive_key, negative_keys, temperature=0.1, reduction='mean'):
    """InfoNCE with explicit unpaired negative keys over L2-normalized embeddings, as in the
    reference code.
    """
    query, positive_key, negative_keys = (F.normalize(x, dim=-1) for x in
                                          (query, positive_key, negative_keys))
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
    negative_logits = query @ negative_keys.transpose(-2, -1)
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


class NCELoss(nn.Module):
    """CLIP makeup losses: the reference code's InfoNCE (:meth:`forward`) and the paper's
    directional loss (:meth:`directional`).
    """

    def __init__(self, device, clip_ckpt: str):
        super(NCELoss, self).__init__()
        self.device = device
        self.model, clip_preprocess = clip.load(clip_ckpt, device=device)
        if device.type != "cuda":  # clip.load only auto-floats on "cpu"
            self.model.float()

        # Released InfoNCE path: the un-normalize assumes [-1, 1] inputs but is
        # fed [0, 1] images, mapping them to [0.5, 1]; kept as released.
        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] +
            clip_preprocess.transforms[:2] +      # Resize(224, bicubic) + CenterCrop
            clip_preprocess.transforms[4:])       # CLIP mean/std normalization

        # Paper Eq. 13 path: standard CLIP preprocessing for [0, 1] RGB tensors.
        self.preprocess_01 = transforms.Compose(
            clip_preprocess.transforms[:2] + clip_preprocess.transforms[4:])

    def get_text_features(self, class_str: str, norm: bool = True) -> torch.Tensor:
        """Embed a class string with all 79 ImageNet templates; (79, 512)."""
        template_text = [template.format(class_str) for template in imagenet_templates]
        tokens = clip.tokenize(template_text).to(self.device)
        text_features = self.model.encode_text(tokens).detach()
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True,
                           input_range: str = "release") -> torch.Tensor:
        """Embed images; ``input_range='01'`` uses standard CLIP preprocessing."""
        prep = self.preprocess_01 if input_range == "01" else self.preprocess
        image_features = self.model.encode_image(prep(img).to(self.device))
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)
        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def forward(self, src_img: torch.Tensor, source_class: str,
                target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        """The reference ``infonce_loss``: the image edit direction against the text directions
        (positives) and the source-text direction (negative), in CLIP space.
        """
        source_text = self.get_text_features(source_class)
        target_text = self.get_text_features(target_class).mean(axis=0, keepdim=True)

        src_features = self.get_image_features(src_img)
        target_features = self.get_image_features(target_img)
        target_text = target_text.repeat(target_features.shape[0], 1)

        query = target_features - src_features
        query /= query.clone().norm(dim=-1, keepdim=True)

        pos1_direction = self.compute_text_direction(source_class, target_class)
        pos1_direction = pos1_direction.repeat(target_features.shape[0], 1)

        pos2_direction = target_text - src_features
        pos2_direction /= pos2_direction.clone().norm(dim=-1, keepdim=True)

        neg_direction = source_text - src_features[:1]
        neg_direction /= neg_direction.clone().norm(dim=-1, keepdim=True)

        return info_nce(query.repeat(2, 1),
                        torch.cat([pos1_direction, pos2_direction], 0),
                        neg_direction)

    def directional(self, src_img: torch.Tensor, source_class: str,
                    target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        """Directional CLIP loss ``1 - cos(dI, dT)`` (paper Eq. 13) between the image and text edit
        directions, for ``[0, 1]`` images; a near-zero image direction gives a detached zero.
        """
        image_direction = (
            self.get_image_features(target_img, input_range="01").float()
            - self.get_image_features(src_img, input_range="01").float()
        )
        if float(image_direction.norm(dim=-1).mean()) < 1e-6:
            return image_direction.new_zeros(()).detach()
        image_direction = F.normalize(image_direction, dim=-1, eps=1e-8)
        text_direction = self.compute_text_direction(source_class, target_class).float()
        return (1.0 - F.cosine_similarity(image_direction, text_direction, dim=-1)).mean()
