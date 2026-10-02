"""AT3D-P: impersonation with a differentiably rendered 3D eye-and-nose patch.

Reference Paper: Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition (CVPR 2023)
Reference Code: https://github.com/thu-ml/AT3D
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    ImageSpec,
    StageInputs,
)
from afrlib.afr.base import AFRMethod
from afrlib.fr import FaceRecognizer
from afrlib.utils.assets import get_asset
from afrlib.utils.log import get_logger, tqdm
from frbench import align

from .reconstruction import (
    FaceReconstructor,
    PreparedFaces,
    _as_five_landmarks,
    reverse_warp_to_original,
)
from .renderer import AT3DRenderer


logger = get_logger(__name__)

# The reference code's seed.
_DEFAULT_SEED = 1234


def _set_seed(seed: int) -> None:
    """Seed torch on the CPU and all CUDA devices, as the released code does."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AT3D(AFRMethod):
    """An eye-and-nose patch whose 3D face coefficients are optimized toward the target and
    rendered onto each face (the coefficient-space variant AT3D-P; requires CUDA).

    Args:
        iterations: Adam iterations.
        eta: The paper's budget; as in the reference code it only sets the learning rate
            ``1.5 * eta / iterations``.
        learning_rate: Adam learning rate; overrides the one derived from ``eta``.
        use_source_lighting: Light the patch as the source (Algorithm 1) rather than as the
            target (the reference data loader).
        seed: Torch seed set at the start of each :meth:`protect` call.
        bfm_model_path: BFM front-patch model; the registered asset if ``None``.
        recon_checkpoint_path: Deep3DFaceRecon checkpoint; the registered asset if ``None``.
    """

    name = "AT3D-P"
    eval_inputs = StageInputs(
        required=("eval_imgs", "target_img"),
    )
    def __init__(
        self,
        device: torch.device,
        fr_models: Sequence[FaceRecognizer] = (),
        *,
        iterations: int = 300,
        eta: float = 3.0,
        learning_rate: Optional[float] = None,
        use_source_lighting: bool = True,
        seed: int = _DEFAULT_SEED,
        bfm_model_path: Optional[str] = None,
        recon_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__(device, fr_models)
        if device.type != "cuda":
            raise RuntimeError("AT3D requires CUDA for nvdiffrast")

        self.iterations = int(iterations)
        self.eta = float(eta)
        self.learning_rate = (
            float(learning_rate)
            if learning_rate is not None
            else (1.5 * self.eta / self.iterations if self.iterations else 0.0)
        )
        self.use_source_lighting = bool(use_source_lighting)
        self.seed = int(seed)
        self.bfm_model_path = bfm_model_path
        self.recon_checkpoint_path = recon_checkpoint_path

        bfm_model_path = bfm_model_path or get_asset("at3d/BFM_model_front_patch.mat")
        recon_checkpoint_path = recon_checkpoint_path or get_asset("at3d/epoch_20.pth")

        # Build the CPU MTCNN (TensorFlow) before the CUDA models so TensorFlow
        # cannot reserve the GPU.
        self.reconstructor = FaceReconstructor(recon_checkpoint_path, device)
        self.renderer = AT3DRenderer(bfm_model_path).to(device).eval()
        self.renderer.requires_grad_(False)

    def attack_spec(self) -> ImageSpec:
        # AT3D detects and POS-aligns internally on the whole photo.
        return ImageSpec(
            channel=ChannelType.RGB,
            crop_type=CropType.Original,
            align_type=AlignType.NoAlign,
        )

    def spec_for(self, role: str) -> ImageSpec:
        if role == "eval_imgs":
            return super().spec_for(role)
        # The impersonation target arrives as an FFHQ-framed 1024 crop.
        return ImageSpec(
            size=1024,
            channel=ChannelType.RGB,
            crop_type=CropType.FFHQ,
            align_type=AlignType.FFHQQuad,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "iterations": self.iterations,
            "eta": self.eta,
            "learning_rate": self.learning_rate,
            "use_source_lighting": self.use_source_lighting,
            "seed": self.seed,
            "bfm_model_path": self.bfm_model_path,
            "recon_checkpoint_path": self.recon_checkpoint_path,
        })
        return config

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Fit one target-initialized patch per photo on its canonical 224 px crop and warp the
        render back; faces are found with MTCNN, falling back to ``eval_geometry``.

        Args:
            data.target_img: The impersonation target, shared by all photos.
        """
        self._check_inputs("protect", data)
        self._require_fr()
        _set_seed(self.seed)
        photos = list(data.eval_imgs)
        geometry = data.eval_geometry
        if geometry is not None and len(geometry) != len(photos):
            raise ValueError(
                f"eval_geometry length {len(geometry)} != eval_imgs length {len(photos)}"
            )
        target_img = data.target_img.to(self.device)
        return [
            self._protect_one(
                photo.to(self.device).unsqueeze(0),
                target_img,
                fallback_landmarks=_fallback_landmarks(geometry, index),
                image_index=index,
            )[0]
            for index, photo in enumerate(tqdm(photos, desc="AT3D-P"))
        ]

    def _protect_one(
        self,
        unprotected_imgs: torch.Tensor,
        target_img: torch.Tensor,
        fallback_landmarks: Optional[np.ndarray] = None,
        image_index: int = 0,
    ) -> torch.Tensor:
        """Render and fit the patch for one already-batched original image."""
        source = self.reconstructor.prepare(
            unprotected_imgs,
            fallback_landmarks=(
                None if fallback_landmarks is None else (fallback_landmarks,)
            ),
            image_ids=(image_index,),
        )
        target = _expand_prepared(
            self.reconstructor.prepare(target_img), source.images.shape[0]
        )

        with torch.no_grad():
            target_embeddings = self._embed(target.images, target.landmarks)

        target_coefficients = target.coefficients
        identity = target_coefficients.identity.detach().clone().requires_grad_(True)
        expression = target_coefficients.expression.detach().clone().requires_grad_(True)
        texture = target_coefficients.texture.detach().clone().requires_grad_(True)
        lighting = (
            source.coefficients.lighting
            if self.use_source_lighting
            else target_coefficients.lighting
        ).detach()
        angle = source.coefficients.angle.detach()
        translation = source.coefficients.translation.detach()

        if self.iterations:
            # Algorithm 1 updates identity, expression, and texture alternately.
            optimizers = [
                torch.optim.Adam([identity], lr=self.learning_rate),
                torch.optim.Adam([expression], lr=self.learning_rate),
                torch.optim.Adam([texture], lr=self.learning_rate),
            ]
            progress = tqdm(range(self.iterations), desc="AT3D-P optimizing")
            for _ in progress:
                loss_value = 0.0
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                    composite = self._render_composite(
                        source.images,
                        identity,
                        expression,
                        texture,
                        angle,
                        translation,
                        lighting,
                    )
                    current_embeddings = self._embed(composite, source.landmarks)
                    loss = torch.stack(
                        [
                            -F.cosine_similarity(current, target_feature, dim=1).mean()
                            for current, target_feature in zip(
                                current_embeddings, target_embeddings
                            )
                        ]
                    ).mean()
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.item()
                progress.set_postfix(loss=f"{loss_value:.4f}")

        # Final render from the optimized coefficients; only this result is
        # reverse-warped to the original frame.
        with torch.no_grad():
            mask, patch, _, _ = self.renderer(
                identity, expression, texture, angle, translation, lighting
            )
            composite = (255.0 * patch * mask + source.images * (1.0 - mask)).clamp(
                0.0, 255.0
            )
            contribution = (255.0 * patch * mask).clamp(0.0, 255.0)
            return reverse_warp_to_original(
                contribution,
                mask,
                unprotected_imgs,
                source.transforms,
            )

    def _render_composite(
        self,
        source_images: torch.Tensor,
        identity: torch.Tensor,
        expression: torch.Tensor,
        texture: torch.Tensor,
        angle: torch.Tensor,
        translation: torch.Tensor,
        lighting: torch.Tensor,
    ) -> torch.Tensor:
        mask, patch, _, _ = self.renderer(
            identity, expression, texture, angle, translation, lighting
        )
        # Unclamped during optimization, as in the released code; the returned
        # image is clamped separately.
        return 255.0 * patch * mask + source_images * (1.0 - mask)

    def _embed(
        self, images: torch.Tensor, landmarks: torch.Tensor
    ) -> Sequence[torch.Tensor]:
        """Embed canonical crops with every FR model, aligned by landmarks found once on the clean
        crop; renders are clamped to ``[0, 255]``.
        """
        if images.shape[0] != landmarks.shape[0]:
            raise ValueError(
                f"Image/landmark batch mismatch: {images.shape[0]} vs {landmarks.shape[0]}")
        images = images.clamp(0.0, 255.0)
        return [
            model.forward(
                torch.cat(
                    [
                        align(
                            image,
                            image_landmarks.unsqueeze(0),
                            output_size=model.input_size,
                            max_supersample=4,
                        )
                        for image, image_landmarks in zip(images, landmarks)
                    ],
                    dim=0,
                ),
                need_crop=False,
                tta=(),
                l2_normalize=True,
            ).embeddings
            for model in self.fr_models
        ]


def _fallback_landmarks(
    geometry: Optional[Sequence[Any]], index: int
) -> Optional[np.ndarray]:
    """Return the stored 5-point landmarks of one eval image, if provided."""
    if geometry is None:
        return None
    entry = geometry[index]
    landmarks = entry.get("landmarks") if isinstance(entry, dict) else None
    if landmarks is None:
        return None
    return _as_five_landmarks(landmarks)


def _expand_prepared(prepared: PreparedFaces, batch_size: int) -> PreparedFaces:
    if prepared.images.shape[0] == batch_size:
        return prepared
    if prepared.images.shape[0] != 1:
        raise ValueError(
            f"Cannot broadcast prepared batch {prepared.images.shape[0]} to {batch_size}"
        )
    return PreparedFaces(
        images=prepared.images.expand(batch_size, -1, -1, -1),
        landmarks=prepared.landmarks.expand(batch_size, -1, -1),
        coefficients=prepared.coefficients.expand(batch_size),
        transforms=tuple(prepared.transforms[0] for _ in range(batch_size)),
    )
