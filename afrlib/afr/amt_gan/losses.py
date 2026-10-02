# Adapted from AMT-GAN (https://github.com/CGCL-codes/AMT-GAN), crtiterions/.
# Copyright (c) 2022 CGCL-codes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Loss helpers for AMT-GAN training."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Device = Union[str, torch.device]


def lsgan_loss(
    prediction: torch.Tensor,
    target_is_real: bool,
    *,
    real_label: float = 1.0,
    fake_label: float = 0.0,
) -> torch.Tensor:
    """Return the released least-squares GAN objective."""
    label = real_label if target_is_real else fake_label
    return F.mse_loss(prediction, torch.full_like(prediction, label))


class GANLoss(nn.Module):
    """Released default LSGAN objective, with device-safe targets."""

    def __init__(
        self,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label

    def forward(
        self, prediction: torch.Tensor, target_is_real: bool
    ) -> torch.Tensor:
        return lsgan_loss(
            prediction,
            target_is_real,
            real_label=self.real_label,
            fake_label=self.fake_label,
        )


def _cumulative_histogram(channel: torch.Tensor) -> torch.Tensor:
    """Compute the released 256-bin cumulative histogram."""
    hist = torch.histc(channel.float(), bins=256, min=0, max=256)
    return torch.cumsum(hist / hist.sum(), dim=0)


def _transfer_table(
    source_cdf: torch.Tensor, reference_cdf: torch.Tensor
) -> torch.Tensor:
    """AMT-GAN's histogram transfer table: the released nested loops, vectorized; unmatched
    source levels map to themselves.
    """
    table = torch.arange(256, dtype=torch.long, device=source_cdf.device)
    source_levels = source_cdf[1:255].unsqueeze(1)
    reference_low = reference_cdf[:-1].unsqueeze(0)
    reference_high = reference_cdf[1:].unsqueeze(0)
    matches = (source_levels >= reference_low) & (source_levels <= reference_high)
    has_match = matches.any(dim=1)
    first_match = matches.to(torch.long).argmax(dim=1) + 1
    table[1:255] = torch.where(has_match, first_match, table[1:255])
    table[255] = 255
    return table


@torch.no_grad()
def histogram_matching(
    destination: torch.Tensor,
    reference: torch.Tensor,
    indices: Sequence[torch.Tensor],
    device: Optional[Device] = None,
) -> torch.Tensor:
    """Match the ``indices = (dst_y, dst_x, ref_y, ref_x)`` pixels of a CHW image to the
    reference histogram, quantized to integer levels as in the reference code.
    """
    if destination.ndim != 3 or reference.ndim != 3:
        raise ValueError("destination and reference must be CHW tensors")
    if destination.shape[0] != 3 or reference.shape[0] != 3:
        raise ValueError("histogram matching expects three-channel images")
    if len(indices) != 4:
        raise ValueError("indices must contain dst_y, dst_x, ref_y, ref_x")

    output_device = torch.device(device) if device is not None else destination.device
    dst_y, dst_x, ref_y, ref_x = (
        index.detach().to(device=output_device, dtype=torch.long) for index in indices
    )
    destination_values = destination.detach().to(device=output_device).clone()
    reference_values = reference.detach().to(device=output_device)
    matched = destination_values.clone()

    for channel in range(3):
        destination_channel = destination_values[channel, dst_y, dst_x]
        reference_channel = reference_values[channel, ref_y, ref_x]
        destination_cdf = _cumulative_histogram(destination_channel)
        reference_cdf = _cumulative_histogram(reference_channel)
        table = _transfer_table(destination_cdf, reference_cdf)
        levels = destination_channel.to(torch.long).clamp_(0, 255)
        matched[channel, dst_y, dst_x] = table[levels].to(matched.dtype)

    return matched.to(dtype=destination.dtype).detach()


def _as_single_channel_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    name: str,
) -> torch.Tensor:
    if mask.ndim == 2 and batch_size == 1:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W]")
    if mask.shape != (batch_size, 1, height, width):
        raise ValueError(
            f"{name} has shape {tuple(mask.shape)}, expected "
            f"{(batch_size, 1, height, width)}"
        )
    return mask


class HistogramLoss(nn.Module):
    """AMT-GAN histogram-matching L1 loss, with masks indexed by true (y, x) positions."""

    @staticmethod
    def de_norm(value: torch.Tensor) -> torch.Tensor:
        return ((value + 1) / 2).clamp(0, 1)

    def forward(
        self,
        input_data: torch.Tensor,
        target_data: torch.Tensor,
        mask_src: torch.Tensor,
        mask_tar: torch.Tensor,
        device: Optional[Device] = None,
    ) -> torch.Tensor:
        if input_data.ndim != 4 or target_data.ndim != 4:
            raise ValueError("input_data and target_data must be BCHW tensors")
        if input_data.shape[0] != target_data.shape[0]:
            raise ValueError("input_data and target_data batch sizes must match")
        if input_data.shape[1] != 3 or target_data.shape[1] != 3:
            raise ValueError("HistogramLoss expects three-channel images")

        batch_size = input_data.shape[0]
        mask_src = _as_single_channel_mask(
            mask_src,
            batch_size=batch_size,
            height=input_data.shape[2],
            width=input_data.shape[3],
            name="mask_src",
        ).to(device=input_data.device)
        mask_tar = _as_single_channel_mask(
            mask_tar,
            batch_size=batch_size,
            height=target_data.shape[2],
            width=target_data.shape[3],
            name="mask_tar",
        ).to(device=target_data.device)

        input_255 = self.de_norm(input_data) * 255
        target_255 = self.de_norm(target_data) * 255
        input_masked = input_255 * mask_src.to(dtype=input_255.dtype)
        target_masked = target_255 * mask_tar.to(dtype=target_255.dtype)

        matched_batch = []
        for batch_index in range(batch_size):
            # nonzero() on the [H, W] mask gives true (y, x) columns; in the
            # release, unsqueeze() made the channel index act as y.
            src_yx = mask_src[batch_index, 0].nonzero(as_tuple=False)
            tar_yx = mask_tar[batch_index, 0].nonzero(as_tuple=False)
            if src_yx.numel() == 0 or tar_yx.numel() == 0:
                # An empty region has no histogram; it contributes zero loss
                # instead of a NaN CDF.
                matched_batch.append(input_masked[batch_index].detach().clone())
                continue
            matched_batch.append(
                histogram_matching(
                    input_masked[batch_index],
                    target_masked[batch_index],
                    (
                        src_yx[:, 0],
                        src_yx[:, 1],
                        tar_yx[:, 0],
                        tar_yx[:, 1],
                    ),
                    device=input_data.device if device is None else device,
                )
            )

        # The matched target is detached, as in the release.
        input_match = torch.stack(matched_batch).detach()
        return F.l1_loss(input_masked, input_match)


def cosine_similarity_mean(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean cosine similarity used by AMT-GAN's face-recognition loss."""
    return F.cosine_similarity(
        embedding_a, embedding_b, dim=dim, eps=eps
    ).mean()


def cosine_distance(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return ``1 - cosine_similarity_mean`` for targeted attacks."""
    return 1 - cosine_similarity_mean(embedding_a, embedding_b, dim, eps)


def make_lpips(
    device: Optional[Device] = None, *, freeze: bool = True
) -> nn.Module:
    """The AlexNet LPIPS model, imported on first use."""
    import lpips

    model = lpips.LPIPS(net="alex")
    if device is not None:
        model = model.to(device)
    model.eval()
    if freeze:
        model.requires_grad_(False)
    return model


def _self_check_histogram_spatial_axes() -> dict[str, float]:
    """Regression check that a masked pixel below row zero receives gradient."""
    source = torch.full((1, 3, 4, 5), -1.0, requires_grad=True)
    reference = torch.full((1, 3, 4, 5), -1.0)
    with torch.no_grad():
        source[:, :, 2, 3] = 10 / 127.5 - 1
        reference[:, :, 1, 4] = 200 / 127.5 - 1

    source_mask = torch.zeros(1, 1, 4, 5)
    reference_mask = torch.zeros(1, 1, 4, 5)
    source_mask[:, :, 2, 3] = 1
    reference_mask[:, :, 1, 4] = 1

    loss = HistogramLoss()(
        source, reference, source_mask, reference_mask, device="cpu"
    )
    loss.backward()
    selected_gradient = source.grad[0, :, 2, 3].abs().sum().item()
    wrong_row_gradient = source.grad[0, :, 0, 2].abs().sum().item()
    if not loss.item() > 0 or not selected_gradient > 0:
        raise AssertionError("non-top-row masked pixel was not histogram-matched")
    if wrong_row_gradient != 0:
        raise AssertionError("histogram matching used the wrong spatial row")
    return {
        "loss": loss.item(),
        "selected_gradient": selected_gradient,
        "wrong_row_gradient": wrong_row_gradient,
    }


if __name__ == "__main__":
    print(_self_check_histogram_spatial_axes())
