"""In-memory five-point alignment and Deep3DFaceRecon coefficient inference."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50

from afrlib.utils.log import get_logger


logger = get_logger(__name__)

LANDMARK_ORDER = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")

# The five points Deep3DFaceRecon's load_lm3d() derives from similarity_Lm3D_all.mat, embedded
# so that alignment does not need the licensed full BFM.
DEEP3D_LANDMARK_TEMPLATE = np.asarray(
    [
        [-0.31148657, 0.29036078, 0.13377953],
        [0.30979887, 0.28972036, 0.13179526],
        [0.00325350, -0.04617932, 0.55244243],
        [-0.25216928, -0.38133916, 0.22405732],
        [0.24846619, -0.38128236, 0.22235769],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class FaceCoefficients:
    """The six coefficient groups emitted by the 257-D reconstruction model."""

    identity: torch.Tensor  # (B, 80)
    expression: torch.Tensor  # (B, 64)
    texture: torch.Tensor  # (B, 80)
    angle: torch.Tensor  # (B, 3)
    lighting: torch.Tensor  # (B, 27)
    translation: torch.Tensor  # (B, 3)

    @classmethod
    def from_vector(cls, coefficients: torch.Tensor) -> "FaceCoefficients":
        if coefficients.ndim != 2 or coefficients.shape[1] != 257:
            raise ValueError(f"Expected reconstruction coefficients (B,257), got {tuple(coefficients.shape)}")
        return cls(
            identity=coefficients[:, :80],
            expression=coefficients[:, 80:144],
            texture=coefficients[:, 144:224],
            angle=coefficients[:, 224:227],
            lighting=coefficients[:, 227:254],
            translation=coefficients[:, 254:257],
        )

    def expand(self, batch_size: int) -> "FaceCoefficients":
        """Broadcast a single coefficient set to ``batch_size`` without copying."""
        if self.identity.shape[0] == batch_size:
            return self
        if self.identity.shape[0] != 1:
            raise ValueError(
                f"Cannot broadcast coefficient batch {self.identity.shape[0]} to {batch_size}"
            )
        return FaceCoefficients(
            identity=self.identity.expand(batch_size, -1),
            expression=self.expression.expand(batch_size, -1),
            texture=self.texture.expand(batch_size, -1),
            angle=self.angle.expand(batch_size, -1),
            lighting=self.lighting.expand(batch_size, -1),
            translation=self.translation.expand(batch_size, -1),
        )


@dataclass(frozen=True)
class Deep3DTransform:
    """POS scale and crop that map an original canvas to the 224 crop.

    Attributes:
        original_size: ``(height, width)`` of the original.
        scale: POS scale factor.
        resized_size: ``(height, width)`` after scaling.
        crop_box: PIL-style ``(left, top, right, bottom)`` window on the resized image.
        target_size: Side of the crop.
    """

    original_size: Tuple[int, int]
    scale: float
    resized_size: Tuple[int, int]
    crop_box: Tuple[int, int, int, int]
    target_size: int = 224


@dataclass(frozen=True)
class PreparedFaces:
    """Canonical crops, FR landmarks on those crops, and 3DMM coefficients."""

    images: torch.Tensor  # RGB (B, 3, 224, 224), [0, 255]
    landmarks: torch.Tensor  # (B, 5, 2), re-detected on the 224 crop for FR
    coefficients: FaceCoefficients
    transforms: Tuple[Deep3DTransform, ...]


class FaceDetectionError(RuntimeError):
    """Raised when AT3D's required MTCNN landmarks are unavailable."""


class Deep3DRegressor(nn.Module):
    """Deep3DFaceRecon's torchvision ResNet-50 with seven 1x1 coefficient heads on its
    pre-flatten ``(B, 2048, 1, 1)`` features.
    """

    HEAD_DIMS = (80, 64, 80, 3, 27, 2, 1)

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList(nn.Conv2d(2048, dim, kernel_size=1) for dim in self.HEAD_DIMS)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state: Dict[str, torch.Tensor] = checkpoint["net_recon"]
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state, strict=True)
        for index, head in enumerate(self.heads):
            head.weight.data.copy_(state[f"final_layers.{index}.weight"])
            head.bias.data.copy_(state[f"final_layers.{index}.bias"])

        self.eval().requires_grad_(False)

    def forward(self, images01: torch.Tensor) -> torch.Tensor:
        features = self.backbone.conv1(images01)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        features = self.backbone.avgpool(features)
        return torch.cat([head(features) for head in self.heads], dim=1).flatten(1)


class FaceReconstructor:
    """Detect, align, and regress 3DMM coefficients without intermediate files."""

    def __init__(self, checkpoint_path: str, device: torch.device) -> None:
        # MTCNN imports TensorFlow; hide its GPUs before any TF op runs (CUDA_VISIBLE_DEVICES
        # no longer helps once PyTorch has initialized CUDA).
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        from mtcnn import MTCNN

        try:
            self.detector = MTCNN(device="CPU:0")
        except TypeError:  # compatibility with mtcnn 0.1.x
            self.detector = MTCNN()

        self.device = device
        self.regressor = Deep3DRegressor(checkpoint_path).to(device)

    @torch.no_grad()
    def prepare(
        self,
        images: torch.Tensor,
        fallback_landmarks: Optional[Sequence[Any]] = None,
        image_ids: Optional[Sequence[Any]] = None,
    ) -> PreparedFaces:
        """Canonical 224 crops (Deep3D POS), landmarks re-detected on them for FR alignment, and
        3DMM coefficients of RGB ``[0, 255]`` images, as the reference preprocessing makes them.

        Args:
            images: The photos.
            fallback_landmarks: Five-point landmarks for photos where MTCNN finds no face.
            image_ids: Labels used in log and error messages.
        """
        _validate_images(images)
        if fallback_landmarks is not None and len(fallback_landmarks) != images.shape[0]:
            raise ValueError(
                "fallback_landmarks length "
                f"{len(fallback_landmarks)} != image batch {images.shape[0]}"
            )
        if image_ids is not None and len(image_ids) != images.shape[0]:
            raise ValueError(
                f"image_ids length {len(image_ids)} != image batch {images.shape[0]}"
            )
        crops, fr_landmarks, transforms = [], [], []
        for index, image in enumerate(images):
            image_np = (
                image.detach()
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            label = image_ids[index] if image_ids is not None else index
            fallback = (
                fallback_landmarks[index] if fallback_landmarks is not None else None
            )
            landmarks = self._first_stage_landmarks(image_np, label, fallback)
            crop, pos_crop_landmarks, transform = align_deep3d(image_np, landmarks)
            # Second-stage detection on the canonical crop (released process_input).
            try:
                crop_landmarks = self._detect_landmarks(crop, label)
            except FaceDetectionError:
                logger.warning(
                    "AT3D second-stage MTCNN found no face on eval image %s; "
                    "using POS-mapped landmarks for FR alignment",
                    label,
                )
                crop_landmarks = pos_crop_landmarks
            crops.append(torch.from_numpy(crop).permute(2, 0, 1))
            fr_landmarks.append(torch.from_numpy(crop_landmarks))
            transforms.append(transform)

        crops255 = torch.stack(crops).to(self.device, dtype=torch.float32)
        landmarks224 = torch.stack(fr_landmarks).to(self.device, dtype=torch.float32)
        coefficient_vector = self.regressor(crops255 / 255.0)
        return PreparedFaces(
            crops255,
            landmarks224,
            FaceCoefficients.from_vector(coefficient_vector),
            tuple(transforms),
        )

    @torch.no_grad()
    def align_to_canonical(self, images: torch.Tensor) -> torch.Tensor:
        """Return Deep3D POS 224 crops for clean/target comparison tensors."""
        _validate_images(images)
        crops = []
        for index, image in enumerate(images):
            image_np = (
                image.detach()
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            landmarks = self._detect_landmarks(image_np, index)
            crop, _, _ = align_deep3d(image_np, landmarks)
            crops.append(torch.from_numpy(crop).permute(2, 0, 1))
        return torch.stack(crops).to(device=images.device, dtype=torch.float32)

    @torch.no_grad()
    def detectable_indices(self, images: torch.Tensor) -> list[int]:
        """Return samples that pass both released MTCNN detection stages."""
        _validate_images(images)
        valid = []
        for index, image in enumerate(images):
            image_np = (
                image.detach()
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            try:
                landmarks = self._detect_landmarks(image_np, index)
                crop, _, _ = align_deep3d(image_np, landmarks)
                self._detect_landmarks(crop, index)
            except FaceDetectionError:
                continue
            valid.append(index)
        return valid

    def _first_stage_landmarks(
        self,
        image: np.ndarray,
        index: Any,
        fallback: Any = None,
    ) -> np.ndarray:
        """Detect 5 points, or seed POS from stored geometry when MTCNN misses."""
        try:
            return self._detect_landmarks(image, index)
        except FaceDetectionError:
            if fallback is None:
                raise
            landmarks = _as_five_landmarks(fallback)
            logger.warning(
                "AT3D first-stage MTCNN found no face on eval image %s; "
                "using stored eval_geometry landmarks for Deep3D POS alignment",
                index,
            )
            return landmarks

    def _detect_landmarks(self, image: np.ndarray, index: Any) -> np.ndarray:
        """Return the MTCNN five points of the face horizontally closest to the image center."""
        detections = self.detector.detect_faces(image)
        if not detections:
            raise FaceDetectionError(
                f"MTCNN found no face in batch item {index}"
            )
        width = image.shape[1]
        detections.sort(
            key=lambda result: abs(result["box"][0] + result["box"][2] / 2.0 - width / 2.0)
        )
        keypoints = detections[0]["keypoints"]
        return np.asarray([keypoints[name] for name in LANDMARK_ORDER], dtype=np.float32)


def _as_five_landmarks(landmarks: Any) -> np.ndarray:
    """Validate the Deep3D / AT3D five-point convention ``(5, 2)``."""
    array = np.asarray(landmarks, dtype=np.float32)
    if array.shape != (5, 2):
        raise ValueError(f"Expected five 2D landmarks, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Expected finite five-point landmarks")
    return array


def align_deep3d(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: int = 224,
    rescale_factor: float = 102.0,
) -> Tuple[np.ndarray, np.ndarray, Deep3DTransform]:
    """Apply the released Deep3DFaceRecon POS alignment and crop."""
    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected five 2D landmarks, got {landmarks.shape}")

    height, width = image.shape[:2]
    landmarks_y_up = landmarks.copy()
    landmarks_y_up[:, 1] = height - 1 - landmarks_y_up[:, 1]
    translation, scale = _position_and_scale(
        landmarks_y_up.T, DEEP3D_LANDMARK_TEMPLATE.T
    )
    scale = float(rescale_factor / scale)

    resized_width, resized_height = int(width * scale), int(height * scale)
    left = int(
        resized_width / 2
        - target_size / 2
        + (translation[0] - width / 2) * scale
    )
    top = int(
        resized_height / 2
        - target_size / 2
        + (height / 2 - translation[1]) * scale
    )
    box = (left, top, left + target_size, top + target_size)
    transform = Deep3DTransform(
        original_size=(height, width),
        scale=scale,
        resized_size=(resized_height, resized_width),
        crop_box=box,
        target_size=target_size,
    )

    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), resample=Image.Resampling.BICUBIC
    )
    crop = np.asarray(resized.crop(box), dtype=np.uint8).copy()
    crop_landmarks = landmarks * scale - np.asarray([left, top], dtype=np.float32)
    return crop, crop_landmarks.astype(np.float32), transform


def _resize_float(arr: np.ndarray, out_w: int, out_h: int, is_mask: bool) -> np.ndarray:
    """Bicubic-resample a float array without a uint8 round trip (no quantization or mask
    aliasing).
    """
    if is_mask:
        img = Image.fromarray(arr.astype(np.float32)).resize(
            (out_w, out_h), resample=Image.Resampling.BICUBIC
        )
        return np.asarray(img, dtype=np.float32)
    channels = [
        np.asarray(
            Image.fromarray(arr[..., c].astype(np.float32)).resize(
                (out_w, out_h), resample=Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        for c in range(arr.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def reverse_warp_to_original(
    contribution224: torch.Tensor,
    mask224: torch.Tensor,
    originals: torch.Tensor,
    transforms: Tuple[Deep3DTransform, ...],
) -> torch.Tensor:
    """Composite the premultiplied 224 px patch ``contribution224`` back onto the originals as
    ``original * (1 - m) + patch * m``, applying the mask once; pixels outside the crop stay.
    """
    if contribution224.shape[0] != originals.shape[0]:
        raise ValueError(
            "contribution/original batch mismatch: "
            f"{contribution224.shape[0]} vs {originals.shape[0]}"
        )
    if mask224.shape[0] != contribution224.shape[0]:
        raise ValueError(
            f"mask/contribution batch mismatch: {mask224.shape[0]} vs {contribution224.shape[0]}"
        )
    if len(transforms) != contribution224.shape[0]:
        raise ValueError(
            f"Expected {contribution224.shape[0]} transforms, got {len(transforms)}"
        )

    outputs = []
    for index in range(contribution224.shape[0]):
        transform = transforms[index]
        left, top, right, bottom = transform.crop_box
        resized_h, resized_w = transform.resized_size
        target = transform.target_size
        if (bottom - top, right - left) != (target, target):
            raise ValueError(f"Unexpected POS crop box: {transform.crop_box}")

        contrib_np = (
            contribution224[index]
            .detach()
            .float()
            .clamp(0, 255)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        mask_np = mask224[index].detach().float().cpu().numpy()
        if mask_np.ndim == 3:
            mask_np = mask_np.mean(axis=0)
        mask_np = np.clip(mask_np, 0.0, 1.0).astype(np.float32)

        # Un-premultiply where the mask is non-zero; the mask is applied once,
        # at the final composite.
        patch_np = np.where(
            mask_np[..., None] > 1e-6,
            contrib_np / np.clip(mask_np[..., None], 1e-6, None),
            contrib_np,
        ).astype(np.float32)

        patch_resized = np.zeros((resized_h, resized_w, 3), dtype=np.float32)
        mask_resized = np.zeros((resized_h, resized_w), dtype=np.float32)
        # Mirror PIL.Image.crop: out-of-bounds regions are ignored / zero-filled.
        src_y0 = max(0, -top)
        src_x0 = max(0, -left)
        dst_y0 = max(0, top)
        dst_x0 = max(0, left)
        dst_y1 = min(resized_h, bottom)
        dst_x1 = min(resized_w, right)
        copy_h = dst_y1 - dst_y0
        copy_w = dst_x1 - dst_x0
        if copy_h > 0 and copy_w > 0:
            patch_resized[dst_y0:dst_y1, dst_x0:dst_x1] = patch_np[
                src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
            ]
            mask_resized[dst_y0:dst_y1, dst_x0:dst_x1] = mask_np[
                src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
            ]

        original_h, original_w = transform.original_size
        patch_orig = _resize_float(patch_resized, original_w, original_h, is_mask=False)
        mask_orig = _resize_float(mask_resized, original_w, original_h, is_mask=True)
        original = (
            originals[index]
            .detach()
            .float()
            .clamp(0, 255)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        if original.shape[:2] != (original_h, original_w):
            raise ValueError(
                "Original image size does not match retained POS transform: "
                f"{original.shape[:2]} vs {(original_h, original_w)}"
            )
        composed = original * (1.0 - mask_orig[..., None]) + patch_orig * mask_orig[..., None]
        outputs.append(torch.from_numpy(np.clip(composed, 0, 255)).permute(2, 0, 1))
    return torch.stack(outputs, dim=0).to(device=originals.device, dtype=torch.float32)


def _position_and_scale(points_2d: np.ndarray, points_3d: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve the weak-perspective POS system used by Deep3DFaceRecon."""
    count = points_2d.shape[1]
    matrix = np.zeros((2 * count, 8), dtype=np.float64)
    matrix[0 : 2 * count - 1 : 2, 0:3] = points_3d.T
    matrix[0 : 2 * count - 1 : 2, 3] = 1
    matrix[1 : 2 * count : 2, 4:7] = points_3d.T
    matrix[1 : 2 * count : 2, 7] = 1
    rhs = points_2d.T.reshape(2 * count, 1)
    solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0].reshape(-1)
    scale = (np.linalg.norm(solution[0:3]) + np.linalg.norm(solution[4:7])) / 2.0
    return np.asarray([solution[3], solution[7]]), float(scale)


def _validate_images(images: torch.Tensor) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected RGB images (B,3,H,W), got {tuple(images.shape)}")
    if not torch.is_floating_point(images):
        raise TypeError(f"Expected floating-point images, got {images.dtype}")
    if images.shape[0] == 0:
        raise ValueError("AT3D requires at least one source image")
    minimum, maximum = float(images.amin()), float(images.amax())
    if minimum < 0 or maximum > 255:
        raise ValueError(f"Expected image values in [0,255], got [{minimum:.3f}, {maximum:.3f}]")
