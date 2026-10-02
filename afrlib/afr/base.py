"""Base class of AFRLib's AFR methods, and helpers to read saved run folders."""
from __future__ import annotations

import inspect
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import torch

from afrlib.contracts import (
    ORIGINAL_SPEC,
    CropType,
    EvalData,
    FitData,
    ImageSpec,
    StageData,
    StageInputs,
    TrainData,
    provided_fields,
)
from afrlib.utils.log import get_logger

if TYPE_CHECKING:
    from afrlib.fr import FaceRecognizer

logger = get_logger(__name__)

# Layout of a run folder written by :meth:`AFRMethod.save_run`.
CONFIG_FILENAME = "config.json"
INPUTS_DIRNAME = "inputs"

# stage name -> (the attribute declaring its inputs, the data class it takes).
_STAGES = {
    "train": ("train_inputs", TrainData),
    "fit": ("fit_inputs", FitData),
    "protect": ("eval_inputs", EvalData),
}

# Image roles the caller prepares to ``spec_for(role)``; reported by
# :meth:`AFRMethod.input_specs`. ``eval_imgs`` is always the original image.
_IMAGE_ROLES = frozenset(
    {
        "target_img",
        "target_pool",
        "protectee_album",
        "protected_imgs",
        "reference_imgs",
    }
)


class AFRMethod(ABC):
    """Base class of AFRLib's AFR methods: up to three stages (:meth:`train`, :meth:`fit`,
    :meth:`protect`) over RGB float32 images in ``[0, 255]``, against injected FR surrogates.

    Example:
        ::

            frs = build_fr_ensemble(specs, device)
            method = Fawkes(device, fr_models=frs)
            protected = method.protect(EvalData(eval_imgs=images))
    """

    # Display name, e.g. "Fawkes". Set by every subclass.
    name: str

    # Which fields of TrainData/FitData/EvalData each stage reads.
    train_inputs: StageInputs = StageInputs()
    fit_inputs: StageInputs = StageInputs()
    eval_inputs: StageInputs = StageInputs(
        required=("eval_imgs",), optional=("eval_geometry",)
    )

    # Minimum number of FR surrogates, checked by _require_fr. A subclass may
    # override it with a property when it depends on the configuration.
    min_surrogates: int = 1

    # Name of the corpus in DATASET_REGISTRY this method trains on, if any.
    training_corpus: Optional[str] = None

    def __init__(
        self,
        device: "torch.device | str",
        fr_models: Sequence["FaceRecognizer"] = (),
        *,
        assets_path: "str | os.PathLike[str] | None" = None,
    ) -> None:
        self.device = torch.device(device)
        self.fr_models = [
            model.to(self.device).eval().requires_grad_(False)
            for model in fr_models
        ]
        # FR specs the artifacts are bound to (snapshotted by a stage or read
        # from a loaded config); _require_fr warns when the injected ones differ.
        self._recorded_fr_specs: Optional[List[Dict[str, Any]]] = None
        self._fr_mismatch_warned = False

        self.assets_path = (
            Path(assets_path).expanduser().resolve()
            if assets_path is not None
            else None
        )
        # Provenance of the filtered corpus given to train(), for config.json.
        self._corpus_config: Optional[Dict[str, Any]] = None
        self._trained = False
        self._fitted = False
        # A method that implements neither stage has nothing to prepare.
        self._inference_ready = not (
            self.implements_train() or self.implements_fit()
        )

    # Stages

    @abstractmethod
    def attack_spec(self) -> ImageSpec:
        """The image format this method works in internally, typically the one :meth:`protect`
        crops to; callers ask :meth:`spec_for` instead.
        """
        raise NotImplementedError

    def spec_for(self, role: str) -> ImageSpec:
        """The format the caller must deliver ``role`` in: the original photo for ``eval_imgs``,
        otherwise :meth:`attack_spec` unless the method overrides it.
        """
        if role == "eval_imgs":
            return ORIGINAL_SPEC
        return self.attack_spec()

    def input_specs(self, stage: str) -> Dict[str, ImageSpec]:
        """Map each image role ``stage`` reads to the spec the caller must deliver, excluding
        ``eval_imgs`` and the non-image roles.

        Args:
            stage: ``"train"``, ``"fit"`` or ``"protect"``.
        """
        attribute, _ = _STAGES[stage]
        declared: StageInputs = getattr(self, attribute)
        return {
            role: self.spec_for(role)
            for role in sorted(declared.known)
            if role in _IMAGE_ROLES
        }

    def train(self, data: TrainData) -> None:
        """Train the method on a corpus, for the general population; a no-op unless overridden."""

    def fit(self, data: FitData) -> None:
        """Personalize the method for one protectee; a no-op unless overridden."""

    @abstractmethod
    def protect(self, data: EvalData) -> "list[torch.Tensor]":
        """Protect original images and return the protected originals; how is up to the method,
        with :mod:`afrlib.utils.face_crop` as optional crop and paste helpers.

        Args:
            data: ``data.eval_imgs`` is a ragged list of original images
                (``[3, Hi, Wi]`` RGB float in ``[0, 255]``);
                ``data.eval_geometry`` gives each image's
                ``{"landmarks", "face_box"}`` in its own pixel coordinates.

        Returns:
            One protected image per input, at the input's size.
        """
        raise NotImplementedError

    @classmethod
    def implements_train(cls) -> bool:
        """Whether this method has a corpus-training stage."""
        return cls.train is not AFRMethod.train

    @classmethod
    def implements_fit(cls) -> bool:
        """Whether this method has a per-protectee fitting stage."""
        return cls.fit is not AFRMethod.fit

    @property
    def trained(self) -> bool:
        """Whether this instance has corpus-trained weights (own or loaded)."""
        return self._trained

    @property
    def fitted(self) -> bool:
        """Whether this instance is personalized to a protectee."""
        return self._fitted

    @property
    def ready(self) -> bool:
        """Whether every stage this method implements has run or been loaded; corpus-trained
        weights alone still need :meth:`fit`.
        """
        return self._inference_ready

    def _check_inputs(self, stage: str, data: StageData) -> None:
        """Validate a stage's data against its declared inputs and record any corpus provenance.

        Raises:
            TypeError: On the wrong data class for the stage.
            ValueError: When a required role is missing.
        """
        attribute, expected = _STAGES[stage]
        if not isinstance(data, expected):
            raise TypeError(
                f"{type(self).__name__}.{stage}() takes a {expected.__name__}, "
                f"got {type(data).__name__}."
            )
        spec: StageInputs = getattr(self, attribute)
        provided = provided_fields(data)
        missing = sorted(spec.required - provided)
        if missing:
            raise ValueError(
                f"{type(self).__name__}.{stage}() needs "
                f"{expected.__name__}({', '.join(f'{f}=...' for f in missing)}); "
                f"got {sorted(provided) or 'nothing'}."
            )
        ignored = sorted(provided - spec.known)
        if ignored:
            logger.warning(
                "%s.%s() does not read %s; those inputs are ignored.",
                type(self).__name__,
                stage,
                ", ".join(ignored),
            )
        corpus = getattr(data, "corpus", None)
        if corpus is not None and "corpus" in spec.known:
            self._corpus_config = corpus.get_config()

    # Lifecycle

    def _begin_training(self) -> None:
        """Claim this instance's single training run and record its surrogates.

        Raises:
            RuntimeError: If the instance was already trained.
        """
        if self._trained:
            raise RuntimeError(
                f"{type(self).__name__} has already been trained; construct a "
                "fresh instance for another training run (pass "
                "assets_path=<run folder> to continue from saved assets)."
            )
        self._trained = True
        self._record_surrogates()

    def _record_surrogates(self) -> None:
        """Snapshot the surrogates a stage is about to bind the artifact to."""
        if self.fr_models:
            self._recorded_fr_specs = [model.get_spec() for model in self.fr_models]

    def _begin_fitting(self) -> None:
        """Claim this instance's single fit run and record its surrogates.

        Raises:
            RuntimeError: If the instance was already fitted, or the method trains but has no
                trained weights yet.
        """
        if self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} has already been fitted to a protectee; "
                "construct a fresh instance to fit another one (pass "
                "assets_path=<run folder> to reuse the corpus-trained weights)."
            )
        if self.implements_train() and not self._trained:
            raise RuntimeError(
                f"{type(self).__name__}.fit() needs corpus-trained weights; "
                "call train() first, or construct with assets_path=<run folder>."
            )
        self._fitted = True
        self._record_surrogates()

    def _mark_ready(self) -> None:
        """Declare the method's artifacts complete, i.e. ready for ``protect``."""
        self._inference_ready = True

    def _require_savable(self) -> None:
        """Raise unless there is something to write to a run folder; unlike :meth:`_require_ready`,
        a trained but unfitted method qualifies.
        """
        if self._inference_ready or self._trained or self._fitted:
            return
        raise RuntimeError(
            f"{type(self).__name__}.save_run() has nothing to save; run "
            f"{'train()' if self.implements_train() else 'fit()'} first."
        )

    def _require_ready(self, action: str = "protect()") -> None:
        """Raise unless every stage the method implements has run or been loaded, and
        ``spec_for("eval_imgs")`` is the original-image spec that :meth:`protect` requires.
        """
        if self.spec_for("eval_imgs").crop_type is not CropType.Original:
            raise RuntimeError(
                f"{type(self).__name__}.spec_for('eval_imgs') must return the "
                "original-image spec; protect() takes original photos, so a "
                "caller must never be handed the method's attack domain. "
                "Delegate that role to super().spec_for()."
            )
        if self._inference_ready:
            return
        pending = [
            stage
            for stage, implemented, done in (
                ("train()", self.implements_train(), self._trained),
                ("fit()", self.implements_fit(), self._fitted),
            )
            if implemented and not done
        ]
        raise RuntimeError(
            f"{type(self).__name__}.{action} requires prepared assets; call "
            f"{' then '.join(pending) or 'train()'} first, or construct with "
            "assets_path=<run folder>."
        )

    # Construction

    def get_config(self) -> Dict[str, Any]:
        """Flat, JSON-able config and provenance; the keys naming an ``__init__`` parameter are
        what :meth:`from_config` rebuilds from.
        """
        return {
            "afr": self.name,
            "fr_models": [model.describe() for model in self.fr_models],
            "stages": {"trained": self._trained, "fitted": self._fitted},
            "trained_against_surrogate": self._recorded_fr_specs,
            "trained_on_corpus": self._corpus_config,
        }

    @classmethod
    def from_config(
        cls,
        device: "torch.device | str",
        fr_models: Sequence["FaceRecognizer"] = (),
        config: Optional[Mapping[str, Any]] = None,
        *,
        assets_path: "str | os.PathLike[str] | None" = None,
        **overrides: Any,
    ) -> "AFRMethod":
        """Construct from a config dict plus ``overrides``, using only the keys that name an
        ``__init__`` parameter and keeping the recorded FR specs for the surrogate check.
        """
        merged = {**dict(config or {}), **overrides}
        selected = cls._select_config(merged)
        if "assets_path" in inspect.signature(cls.__init__).parameters:
            selected["assets_path"] = assets_path
        method = cls(device, fr_models, **selected)
        method._recorded_fr_specs = fr_specs_from_config(merged) or None
        return method

    @classmethod
    def from_run(
        cls,
        run_dir: "str | os.PathLike[str]",
        device: "torch.device | str",
        fr_models: Optional[Sequence["FaceRecognizer"]] = None,
        *,
        restore_fr: bool = False,
        **overrides: Any,
    ) -> "AFRMethod":
        """Rebuild a saved run: read ``config.json``, construct the method, load its assets.

        Args:
            run_dir: A folder written by :meth:`save_run`.
            device: Device for the rebuilt method.
            fr_models: Surrogates to inject; most methods need none at inference.
            restore_fr: Rebuild the ensemble recorded in the config when ``fr_models`` is
                ``None``.
            **overrides: Constructor arguments overriding the saved config.
        """
        root = Path(run_dir).expanduser().resolve()
        config = read_run_config(root)
        if fr_models is None:
            fr_models = build_recorded_fr(config, device) if restore_fr else ()
        return cls.from_config(
            device, fr_models, config, assets_path=root, **overrides
        )

    @classmethod
    def _select_config(cls, config: Mapping[str, Any]) -> Dict[str, Any]:
        """Keep only the config keys this class's ``__init__`` accepts."""
        params = inspect.signature(cls.__init__).parameters
        reserved = {"self", "device", "fr_models", "assets_path"}
        return {
            key: value
            for key, value in config.items()
            if key in params and key not in reserved
        }

    # FR access

    def _require_fr(self) -> List["FaceRecognizer"]:
        """Return the injected FR ensemble, checked here rather than in ``__init__`` so that a
        method loaded only for inference needs none.

        Raises:
            ValueError: With fewer than ``min_surrogates`` models.
        """
        needed = int(self.min_surrogates)
        if len(self.fr_models) < needed:
            raise ValueError(
                f"{type(self).__name__} needs at least {needed} FR "
                f"surrogate(s), got {len(self.fr_models)}. Inject them at "
                "construction, e.g. "
                f"{type(self).__name__}(device, fr_models=build_fr_ensemble(specs, device))."
            )
        self._warn_on_surrogate_mismatch()
        return self.fr_models

    def _warn_on_surrogate_mismatch(self) -> None:
        """Warn once when injected surrogates differ from the recorded ones."""
        if self._fr_mismatch_warned or not self._recorded_fr_specs:
            return
        current = [model.get_spec() for model in self.fr_models]
        if current != self._recorded_fr_specs:
            self._fr_mismatch_warned = True
            logger.warning(
                "%s was trained against %s but is being used with %s; results "
                "for a surrogate-specific method are only reproducible against "
                "the recorded surrogates.",
                type(self).__name__,
                self._recorded_fr_specs,
                current,
            )

    # Run folder

    def save_run(self, save_path: "str | os.PathLike[str]", **provenance: Any) -> Path:
        """Write ``config.json`` into ``save_path`` and return the run folder; subclasses call this
        last, through ``super()``, with any extra ``provenance`` keys.
        """
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._write_config(root, **provenance)
        return root

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Restore the stage flags from ``config.json`` and return the run folder, which may also be
        given as a file inside it; subclasses call this first and then load their weights.
        """
        root = Path(load_path).expanduser().resolve()
        if root.is_file():
            root = root.parent
        config_path = root / CONFIG_FILENAME
        if config_path.is_file():
            stages = json.loads(config_path.read_text()).get("stages") or {}
            self._trained = bool(stages.get("trained", self._trained))
            self._fitted = bool(stages.get("fitted", self._fitted))
        return root

    def _maybe_load_run(self) -> None:
        """Load ``assets_path`` if set; call as the subclass ``__init__``'s last step."""
        if self.assets_path is None:
            return
        self.load_run(self.assets_path)

    def _write_config(self, root: "str | os.PathLike[str]", **provenance: Any) -> Path:
        """Atomically write ``config.json`` (``get_config()`` plus extras)."""
        config = self.get_config()
        config.update(provenance)
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / CONFIG_FILENAME
        tmp = root / f".{CONFIG_FILENAME}.tmp"
        tmp.write_text(json.dumps(config, indent=2, default=str))
        os.replace(tmp, path)
        return path

    def _write_input_images(
        self,
        root: "str | os.PathLike[str]",
        name: str,
        imgs: torch.Tensor,
    ) -> List[str]:
        """Write a baked input image (or stack) under ``<root>/inputs`` as PNGs.

        Args:
            root: The run folder.
            name: File stem; a stack is written as ``<name>_000.png``, ...
            imgs: ``[3, H, W]`` or ``[N, 3, H, W]`` RGB in ``[0, 255]``.

        Returns:
            The run-relative paths, for recording in ``config.json``.
        """
        from torchvision.transforms.functional import to_pil_image

        batch = imgs.detach().float().cpu()
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        if batch.dim() != 4 or batch.shape[1] != 3:
            raise ValueError(
                f"Expected baked input {name!r} as (3,H,W) or (N,3,H,W), "
                f"got {tuple(imgs.shape)}"
            )
        inputs_dir = Path(root) / INPUTS_DIRNAME
        inputs_dir.mkdir(parents=True, exist_ok=True)
        single = batch.shape[0] == 1
        written = []
        for index, image in enumerate(batch):
            filename = f"{name}.png" if single else f"{name}_{index:03d}.png"
            to_pil_image(image.div(255.0).clamp(0, 1)).save(inputs_dir / filename)
            written.append(f"{INPUTS_DIRNAME}/{filename}")
        return written


# Run-folder helpers


def read_run_config(run_dir: "str | os.PathLike[str]") -> Dict[str, Any]:
    """Read ``config.json`` from a run folder (or an explicit file path)."""
    path = Path(run_dir).expanduser().resolve()
    if path.is_dir():
        path = path / CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"No {CONFIG_FILENAME} under {run_dir}")
    return json.loads(path.read_text())


def fr_specs_from_config(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The round-trippable FR specs of a saved config: those recorded when a stage ran, else the
    ``spec`` entries of ``fr_models``.
    """
    recorded = config.get("trained_against_surrogate")
    if recorded:
        return [dict(spec) for spec in recorded]
    specs = []
    for entry in config.get("fr_models") or []:
        spec = entry.get("spec") if isinstance(entry, Mapping) else None
        if spec:
            specs.append(dict(spec))
    return specs


def build_recorded_fr(
    config: Mapping[str, Any],
    device: "torch.device | str | None" = None,
) -> List["FaceRecognizer"]:
    """Rebuild the FR ensemble a saved config was trained against."""
    from afrlib.fr import build_fr_ensemble

    specs = fr_specs_from_config(config)
    if not specs:
        raise ValueError(
            "This config records no rebuildable FR specs; inject fr_models "
            "explicitly."
        )
    return build_fr_ensemble(specs, device)
