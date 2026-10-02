"""Data contracts between AFR methods and their callers: image formats and per-stage inputs."""
from __future__ import annotations

import dataclasses
import enum
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from afrlib.data import TrainingCorpus


class ChannelType(enum.Enum):
    """Channel order of an image."""

    RGB = "rgb"
    BGR = "bgr"


class CropType(enum.Enum):
    """How the face is framed in the image (independent of any alignment warp)."""

    Original = "original"  # The original image, uncropped.
    FaceBox = "facebox"    # Square crop around the detected face box.
    FFHQ = "ffhq"          # Loose FFHQ/CelebA-HQ head crop.
    LADN600 = "ladn600"    # LADN / Adv-Makeup frontal makeup framing.


class AlignType(enum.Enum):
    """The geometric warp applied to a crop, if any."""

    NoAlign = "noalign"      # Axis-aligned crop and resize only.
    ArcFace5 = "arcface5"    # 5-point similarity warp to the ArcFace template.
    FFHQQuad = "ffhqquad"    # FFHQ quad warp (StyleGAN aligner) from eye/mouth landmarks.
    LADN600 = "ladn600"      # 600px LADN canvas; face_crop uses the FFHQ quad warp.


@dataclasses.dataclass(frozen=True)
class ImageSpec:
    """Format of an image: size, channel order, framing, warp, and value range.

    Attributes:
        size: Side of the square image in pixels, or ``None`` when the image
            keeps its own size (e.g. :data:`ORIGINAL_SPEC`).
        channel: Channel order.
        crop_type: How the face is framed.
        align_type: The warp applied to the crop; ``None`` means unset.
        range_min: Lowest pixel value.
        range_max: Highest pixel value.
    """

    size: Optional[int] = None
    channel: ChannelType = ChannelType.RGB
    crop_type: CropType = CropType.FFHQ
    align_type: Optional[AlignType] = None
    range_min: float = 0.0
    range_max: float = 255.0


#: Format of ``eval_imgs``: the original image, uncropped and unwarped.
#: ``AFRMethod.spec_for("eval_imgs")`` must return this.
ORIGINAL_SPEC = ImageSpec(
    channel=ChannelType.RGB,
    crop_type=CropType.Original,
    align_type=AlignType.NoAlign,
)


@dataclasses.dataclass(frozen=True)
class StageInputs:
    """The roles one stage reads from its data object; a required role that is missing is an
    error, and a role the stage does not read is ignored with a warning.

    Args:
        required: Roles the stage needs; declaring ``eval_imgs`` also makes ``eval_geometry``
            optional.
        optional: Roles the stage uses if given.

    Example:
        ::

            eval_inputs = StageInputs(required=("eval_imgs", "target_img"))
    """

    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for attribute in ("required", "optional"):
            names = getattr(self, attribute)
            if isinstance(names, str):
                # frozenset("prompt") would split the name into single letters
                # that match no field, silently disabling the check.
                raise TypeError(
                    f"StageInputs {attribute} must be a collection of field "
                    f'names, not the string {names!r} (write ("{names}",)).'
                )
            object.__setattr__(self, attribute, frozenset(names))
        if "eval_imgs" in self.known and "eval_geometry" not in self.known:
            object.__setattr__(self, "optional", self.optional | {"eval_geometry"})
        unknown = sorted(self.known - EVAL_FIELDS - TRAIN_FIELDS - FIT_FIELDS)
        if unknown:
            raise ValueError(
                f"StageInputs names {unknown}, which no stage dataclass defines"
            )
        overlap = self.required & self.optional
        if overlap:
            raise ValueError(
                f"StageInputs fields {sorted(overlap)} are both required and optional"
            )

    @property
    def known(self) -> frozenset[str]:
        """Every field this stage reads."""
        return self.required | self.optional


@dataclasses.dataclass(frozen=True)
class TrainData:
    """Input of ``AFRMethod.train``: data baked in for the whole population.

    Args:
        corpus: The training corpus, from :func:`afrlib.data.get_corpus`. The
            method builds its own dataset and loader from it.
        target_img: One impersonation target, baked into the weights.
        makeup_reference_name: Name of a registered makeup-reference asset
            baked into the weights (e.g. DiffAM's reference triplet).
    """

    corpus: Optional[TrainingCorpus] = None
    target_img: Optional[torch.Tensor] = None
    makeup_reference_name: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class FitData:
    """Input of ``AFRMethod.fit``: per-protectee personalization data.

    Args:
        protectee_album: The protectee's own images, e.g. the album a
            personalized mask is optimized over, in the method's
            ``spec_for("protectee_album")`` format.
    """

    protectee_album: Optional[torch.Tensor] = None


@dataclasses.dataclass(frozen=True)
class EvalData:
    """Input of ``AFRMethod.protect``: data that may change per call.

    Args:
        eval_imgs: The original images to protect: RGB float tensors in
            ``[0, 255]``, as a batch or a ragged list of ``[3, Hi, Wi]``.
        eval_geometry: Per-image ``{"landmarks", "face_box"}`` (5-point
            landmarks and an ``(x1, y1, x2, y2)`` box) in that image's pixel
            coordinates, parallel to ``eval_imgs``.
        target_img: One impersonation target, applied to the whole batch.
        target_pool: Candidate targets the method selects from itself (e.g.
            TIP-IM's greedy insertion, DivTrackee's random auxiliary draw).
        makeup_reference_name: Name of a released makeup-reference image for
            this call (e.g. AMT-GAN's ``ref01.png`` ... ``ref10.png``).
        protected_imgs: Earlier protected images of the protectee, for methods
            whose output depends on their own history.
        reference_imgs: Held-out images of the protectee's identity, one per
            source image (the verification counterpart a dodging attack moves
            away from).
        prompt: A text prompt steering the edit.
        message: A watermark payload to embed.
    """

    eval_imgs: Optional[torch.Tensor | Sequence[torch.Tensor]] = None
    eval_geometry: Optional[Sequence[dict]] = None
    target_img: Optional[torch.Tensor] = None
    target_pool: Optional[torch.Tensor] = None
    makeup_reference_name: Optional[str] = None
    protected_imgs: Optional[torch.Tensor] = None
    reference_imgs: Optional[torch.Tensor] = None
    prompt: Optional[str] = None
    message: Optional[torch.Tensor] = None


#: Any of the three stage data objects.
StageData = TrainData | FitData | EvalData

#: Field names of each stage's data object, for validating declarations.
TRAIN_FIELDS = frozenset(f.name for f in dataclasses.fields(TrainData))
FIT_FIELDS = frozenset(f.name for f in dataclasses.fields(FitData))
EVAL_FIELDS = frozenset(f.name for f in dataclasses.fields(EvalData))


def provided_fields(data: StageData) -> frozenset[str]:
    """Names of the fields of ``data`` that are not ``None``."""
    return frozenset(
        field.name
        for field in dataclasses.fields(data)
        if getattr(data, field.name) is not None
    )


def assemble_stage(spec: StageInputs, data_class: type, pool: dict) -> StageData:
    """Build a stage's data object from the fields it declares in ``spec``.

    Args:
        spec: The stage's declared inputs.
        data_class: :class:`TrainData`, :class:`FitData` or :class:`EvalData`.
        pool: Every role the caller can supply; ``None`` values count as absent.

    Raises:
        ValueError: If ``pool`` lacks a role that ``spec`` requires.
    """
    missing = sorted(name for name in spec.required if pool.get(name) is None)
    if missing:
        raise ValueError(
            f"Missing required input {', '.join(missing)}: "
            f"{data_class.__name__} declares it as required."
        )
    return data_class(
        **{name: pool[name] for name in spec.known if pool.get(name) is not None}
    )
