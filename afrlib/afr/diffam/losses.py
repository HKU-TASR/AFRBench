"""DiffAM training losses: CLIP makeup direction, regional histograms, targeted FR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class CLIPMakeupDirectionLoss(nn.Module):
    """CLIP ViT-B/16 loss aligning the source-to-generated direction with the reference's
    non-makeup-to-makeup direction, on ``[-1, 1]`` RGB images.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        model_name: str = "ViT-B/16",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        import clip

        model, clip_preprocess = clip.load(model_name, device=self.device)
        self.model = model.eval().requires_grad_(False)
        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)
                ),
                *clip_preprocess.transforms[:2],
                *clip_preprocess.transforms[4:],
            ]
        )

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(self.preprocess(images))
        return F.normalize(features.float(), dim=-1)

    def forward(
        self,
        source: torch.Tensor,
        non_makeup: torch.Tensor,
        generated: torch.Tensor,
        makeup: torch.Tensor,
    ) -> torch.Tensor:
        reference_direction = self._encode(makeup) - self._encode(non_makeup)
        reference_direction = F.normalize(reference_direction, dim=-1, eps=1e-7)
        edit_direction = self._encode(generated) - self._encode(source)
        edit_direction = F.normalize(edit_direction, dim=-1, eps=1e-7)
        cosine = F.cosine_similarity(edit_direction, reference_direction, dim=-1)
        return -torch.log(((1.0 + cosine) / 2.0).clamp_min(1e-12)).mean()


def _cdf(channel_values: np.ndarray) -> list[float]:
    histogram = torch.histc(
        torch.from_numpy(channel_values), bins=256, min=0, max=256
    ).numpy()
    total = histogram.sum()
    if total == 0:
        raise ValueError("Histogram region is empty")
    values = [float(value / total) for value in histogram]
    for index in range(1, 256):
        values[index] += values[index - 1]
    return values


def _transfer_table(source: list[float], reference: list[float]) -> np.ndarray:
    table = np.arange(256, dtype=np.int64)
    for source_value in range(1, 256):
        for reference_value in range(1, 256):
            if (
                source[source_value] >= reference[reference_value - 1]
                and source[source_value] <= reference[reference_value]
            ):
                table[source_value] = reference_value
                break
    table[255] = 255
    return table


def _mask_indices(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    squeezed = mask.squeeze()
    if squeezed.ndim != 2:
        raise ValueError("DiffAM histogram masks require batch size one")
    indices = squeezed.nonzero(as_tuple=True)
    if indices[0].numel() == 0:
        raise ValueError("DiffAM histogram region is empty")
    return indices


def expand_eye_region(eye: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
    """Fill the eye's bounding box, padded by ten pixels, from the face mask."""

    rows, columns = _mask_indices(eye)
    row_start = max(0, int(rows.min()) - 10)
    row_end = min(eye.shape[-2], int(rows.max()) + 11)
    col_start = max(0, int(columns.min()) - 10)
    col_end = min(eye.shape[-1], int(columns.max()) + 11)
    expanded = eye.clone()
    expanded[..., row_start:row_end, col_start:col_end] = face[
        ..., row_start:row_end, col_start:col_end
    ]
    return expanded


def histogram_region_loss(
    generated: torch.Tensor,
    reference: torch.Tensor,
    source_mask: torch.Tensor,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    """L1, in ``[0, 255]``, between a generated region and its detached match to the reference
    region's histogram.
    """

    if generated.shape[0] != 1 or reference.shape[0] != 1:
        raise ValueError("DiffAM histogram loss supports batch size one")
    source_rows, source_columns = _mask_indices(source_mask)
    ref_rows, ref_columns = _mask_indices(reference_mask)
    source255 = generated.add(1.0).div(2.0).clamp(0, 1).mul(255).squeeze(0)
    reference255 = reference.add(1.0).div(2.0).clamp(0, 1).mul(255).squeeze(0)
    source_mask3 = source_mask.squeeze(0).expand(3, -1, -1)
    reference_mask3 = reference_mask.squeeze(0).expand(3, -1, -1)
    source_masked = source255 * source_mask3
    reference_masked = reference255 * reference_mask3

    detached_source = source_masked.detach().cpu().numpy().copy()
    detached_reference = reference_masked.detach().cpu().numpy()
    source_pixels = detached_source[:, source_rows.cpu(), source_columns.cpu()]
    reference_pixels = detached_reference[:, ref_rows.cpu(), ref_columns.cpu()]
    for channel in range(3):
        table = _transfer_table(
            _cdf(source_pixels[channel]), _cdf(reference_pixels[channel])
        )
        original = source_pixels[channel].copy()
        source_pixels[channel] = table[original.astype(np.int64)]
        detached_source[channel, source_rows.cpu(), source_columns.cpu()] = (
            source_pixels[channel]
        )
    matched = torch.from_numpy(detached_source).to(
        device=generated.device, dtype=generated.dtype
    )
    return F.l1_loss(source_masked, matched)


def reference_regions(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Lip, skin, and padded eye masks from a reference face-parsing map."""
    if labels.shape != (1, 256, 256):
        raise ValueError(
            f"Reference labels must have shape [1,256,256], got {tuple(labels.shape)}"
        )
    labels = labels.unsqueeze(0)
    face = ((labels == 1) | (labels == 6)).float()
    return {
        "lip": ((labels == 7) | (labels == 9)).float(),
        "skin": ((labels == 1) | (labels == 6) | (labels == 13)).float(),
        "left_eye": expand_eye_region((labels == 4).float(), face),
        "right_eye": expand_eye_region((labels == 5).float(), face),
    }


def source_regions(masks: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Lip, skin, and padded eye masks from a training sample's five masks."""
    if len(masks) != 5:
        raise ValueError("DiffAM source samples require five masks")
    left_eye, right_eye, lip, skin, face = masks
    return {
        "lip": lip,
        "skin": skin,
        "left_eye": expand_eye_region(left_eye, face),
        "right_eye": expand_eye_region(right_eye, face),
    }


def regional_makeup_loss(
    generated: torch.Tensor,
    makeup: torch.Tensor,
    source: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Sum of the regional histogram losses, with skin weighted by 0.15."""
    return (
        histogram_region_loss(
            generated, makeup, source["left_eye"], reference["left_eye"]
        )
        + histogram_region_loss(
            generated, makeup, source["right_eye"], reference["right_eye"]
        )
        + 0.15
        * histogram_region_loss(
            generated, makeup, source["skin"], reference["skin"]
        )
        + histogram_region_loss(
            generated, makeup, source["lip"], reference["lip"]
        )
    )


def targeted_ensemble_loss(
    source_embeddings: Sequence[torch.Tensor],
    target_embeddings: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean over surrogates of ``1 - cos(source, target)``."""
    if len(source_embeddings) == 0 or len(source_embeddings) != len(
        target_embeddings
    ):
        raise ValueError(
            "DiffAM targeted loss requires matching non-empty embedding pairs"
        )
    losses = [
        1.0 - F.cosine_similarity(source, target, dim=1).mean()
        for source, target in zip(source_embeddings, target_embeddings)
    ]
    return torch.stack(losses).mean()


__all__ = [
    "CLIPMakeupDirectionLoss",
    "expand_eye_region",
    "histogram_region_loss",
    "reference_regions",
    "regional_makeup_loss",
    "source_regions",
    "targeted_ensemble_loss",
]
