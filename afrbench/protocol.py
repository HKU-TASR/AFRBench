"""The AFRBench protocol, the same on AFRBench-basic and AFRBench-DF: method roster, surrogates, FR systems, and eval suite."""
from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: FR model of the default evaluation system.
DEFAULT_FR = {"backend": "frbench", "backbone": "irse-100", "loss": "arcface", "dataset": "ms1m"}

#: Attack surrogates for train/fit/protect, in load order.
SURROGATES = [
    {"backend": "frbench", "backbone": "swinv2-b", "loss": "arcface", "dataset": "glint360k"},
    {"backend": "frbench", "backbone": "swinv2-b", "loss": "centerloss", "dataset": "ms1m"},
    {"backend": "frbench", "backbone": "convnextv2-b", "loss": "arcface", "dataset": "ms1m"},
]

FIT_ALBUM_N = 50
EVAL_DETECTOR = "retinaface_resnet50"
SEED = 0
EXPERIMENT = "basic"

#: Implemented in afrlib but not executed by the benchmark.
HELD = frozenset({"efficient_decision", "dpa"})

PIPELINE_PRESETS = (
    "minimal", "quantized", "jpeg", "tta", "full", "median", "gfpgan", "codeformer", "diffpure",
)

#: The 17 evaluation jobs of the protocol, in suite order.
EVAL_SUITE = (
    "minimal",
    "whitebox_swinv2b_arcface_glint360k",
    "whitebox_swinv2b_centerloss_ms1m",
    "whitebox_convnextv2b_arcface_ms1m",
    "tta",
    "jpeg",
    "quantize",
    "tta_jpeg_quantize",
    "median",
    "gfpgan",
    "codeformer",
    "mc_no_targets",
    "mc_drop_2005",
    "mc_ids_500000",
    "mc_ids_100000",
    "mc_ids_20000",
    "mc_ids_4000",
)

#: Suite jobs that score the intruder pipeline, and so accept another FR model.
PIPELINE_EVALS = {
    "minimal": "minimal",
    "tta": "tta",
    "quantize": "quantized",
    "tta_jpeg_quantize": "full",
    "median": "median",
    "gfpgan": "gfpgan",
    "codeformer": "codeformer",
}


def roster() -> List[str]:
    """Built-in methods the benchmark executes."""
    from afrlib import AFR_REGISTRY

    return sorted(name for name in AFR_REGISTRY if name not in HELD)


# FR systems


def fr_slug(fr: Dict[str, Any]) -> str:
    """Short name of an FR spec, e.g. ``irse100_arcface_ms1m``."""
    if fr.get("backend") == "frbench":
        return f"{fr['backbone']}_{fr['loss']}_{fr['dataset']}".replace("-", "")
    return str(fr["backend"]).rsplit(":", 1)[-1].rsplit(".", 1)[-1].lower()


def fr_system(preset: str, fr: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The FR-system block of one pipeline preset, inlined into eval configs."""
    fr = dict(fr or DEFAULT_FR)
    pipeline = {
        "quantize": {"enabled": False, "precision": "uint8"},
        "jpeg": {"enabled": False, "quality": 75, "impl": "diffjpeg"},
        "resize": {"enabled": False, "factor": 0.4, "mode": "bicubic"},
        "purify": {"enabled": False, "method": "median", "kernel_size": 3},
        "detect": {"detector": EVAL_DETECTOR, "keep": "largest", "on_fail": "count_as_evaded"},
        "align": {"template": "arcface_112"},
        "tta": {"enabled": False, "ops": ["flip_horizontal"]},
    }
    if preset == "minimal":
        pass
    elif preset == "quantized":
        pipeline["quantize"]["enabled"] = True
    elif preset == "jpeg":
        pipeline["quantize"]["enabled"] = True
        pipeline["jpeg"]["enabled"] = True
    elif preset == "tta":
        pipeline["tta"]["enabled"] = True
    elif preset == "full":
        pipeline["quantize"]["enabled"] = True
        pipeline["jpeg"]["enabled"] = True
        pipeline["tta"]["enabled"] = True
    elif preset == "median":
        pipeline["purify"]["enabled"] = True
        pipeline["purify"]["method"] = "median"
    elif preset == "gfpgan":
        pipeline["purify"] = {"enabled": True, "method": "gfpgan"}
    elif preset == "codeformer":
        pipeline["purify"] = {"enabled": True, "method": "codeformer", "w": 0.5, "adain": True}
    elif preset == "diffpure":
        pipeline["purify"] = {
            "enabled": True, "method": "diffpure", "t": 500, "sample_step": 1, "seed": 0,
        }
    else:
        raise ValueError(f"Unknown FR-system preset {preset!r}")
    return {"name": f"{fr_slug(fr)}@{preset}", "fr": fr, "pipeline": pipeline}


def jpeg_only_system(fr: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """DiffJPEG q=75 with quantize off (the suite's ``jpeg`` job)."""
    spec = fr_system("minimal", fr)
    spec["name"] = f"{fr_slug(spec['fr'])}@jpeg_only"
    spec["pipeline"]["jpeg"]["enabled"] = True
    spec["pipeline"]["quantize"]["enabled"] = False
    return spec


# Method roles


def method_roles(cls) -> Dict[str, bool]:
    """Whether a method trains, fits, bakes a target, reads one at protect time, or uses a pool."""
    train = getattr(cls, "train_inputs", None)
    eval_inputs = getattr(cls, "eval_inputs", None)
    train_known = set(getattr(train, "known", ())) if train is not None else set()
    eval_known = set(getattr(eval_inputs, "known", ())) if eval_inputs is not None else set()
    return {
        "train": bool(cls.implements_train()),
        "fit": bool(cls.implements_fit()),
        "target_baked": "target_img" in train_known,
        "runtime_target": "target_img" in eval_known,
        "target_pool": "target_pool" in eval_known,
    }


def _roles_for(name: str) -> Dict[str, bool]:
    from afrlib import AFR_REGISTRY

    return method_roles(AFR_REGISTRY[name])


def bakes_target(name: str) -> bool:
    """True when ``target_img`` is a train input (baked into weights)."""
    return _roles_for(name)["target_baked"]


def protects_per_target(name: str) -> bool:
    """True when protect runs once per impersonation target."""
    roles = _roles_for(name)
    return roles["target_baked"] or roles["runtime_target"]


def uses_target_pool(name: str) -> bool:
    """True when protect reads ``target_pool``."""
    return _roles_for(name)["target_pool"]


def resolved_method_defaults(name: str) -> Dict[str, Any]:
    """Constructor defaults, written into each config as provenance."""
    from afrlib import AFR_REGISTRY

    cls = AFR_REGISTRY[name]
    skip = {"self", "device", "fr_models", "assets_path"}
    defaults: Dict[str, Any] = {}
    for key, param in inspect.signature(cls.__init__).parameters.items():
        if key in skip or param.default is inspect.Parameter.empty:
            continue
        value = param.default
        if isinstance(value, (str, int, float, bool)) or value is None:
            defaults[key] = value
        elif isinstance(value, (list, tuple)):
            defaults[key] = list(value)
        else:
            defaults[key] = str(value)
    return defaults


# Dataset-side inputs


def select_fit_album(
    album_paths: Sequence[str], *, seed: int, identity: str, n: int = FIT_ALBUM_N
) -> List[str]:
    """Deterministically keep ``n`` album images for a fit/protect job."""
    paths = sorted(album_paths)
    if len(paths) <= n:
        return paths
    rng = random.Random(f"fit-album:{seed}:{identity}")
    return sorted(rng.sample(paths, n))


def purge_lists(dataset_root: "str | Path") -> Dict[str, Dict[str, Any]]:
    """The dataset's ``purge/<corpus>.json`` lists as corpus-block fragments."""
    lists: Dict[str, Dict[str, Any]] = {}
    for path in sorted((Path(dataset_root) / "purge").glob("*.json")):
        payload = json.loads(path.read_text())
        key = payload.get("key")
        if key not in ("identity", "name"):
            raise ValueError(f"{path} has key {key!r}; expected 'identity' or 'name'.")
        field = "exclude_identities" if key == "identity" else "exclude_images"
        lists[path.stem] = {field: sorted(payload.get("entries", []))}
    return lists


def corpus_block(corpus: str, purges: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """A train config's ``corpus:`` block with the dataset's purge applied."""
    if corpus not in purges:
        raise ValueError(f"no purge list for training corpus {corpus!r}")
    return {"name": corpus, **purges[corpus]}


def protectee_targets(dataset_root: "str | Path") -> Dict[str, str]:
    """Protectee -> assigned HQ target, from ``protectee_groups.json``."""
    payload = json.loads((Path(dataset_root) / "protectee_groups.json").read_text())
    return {str(name): str(target) for name, target in payload["membership"].items()}


# Eval suite


def target_identity_ids(dataset) -> List[str]:
    return sorted(set(dataset.hq_target_identities()) | set(dataset.target_identities()))


def drop_non_target_identities(dataset, n: Optional[int] = None, *, seed: int = 0) -> List[str]:
    """Random purged-pack identities outside the HQ and pool targets (``mc_drop_2005``)."""
    if n is None:
        n = len(target_identity_ids(dataset))
    protected = set(target_identity_ids(dataset))
    pool = [identity for identity in dataset.noise_identity_ids() if identity not in protected]
    if n > len(pool):
        raise ValueError(f"cannot drop {n} non-target identities; only {len(pool)} available")
    rng = random.Random(f"noise-identities-drop:{seed}")
    return sorted(rng.sample(pool, n))


def eval_job(name: str, dataset) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """``(fr_system, visual_quality, distractors)`` of one suite job."""
    full = {"mode": "full"}
    if name == "minimal":
        return fr_system("minimal", DEFAULT_FR), True, full
    if name.startswith("whitebox_"):
        for spec in SURROGATES:
            if name == f"whitebox_{fr_slug(spec)}":
                return fr_system("minimal", spec), False, full
    if name == "jpeg":
        return jpeg_only_system(), False, full
    if name in PIPELINE_EVALS:
        return fr_system(PIPELINE_EVALS[name]), False, full
    if name == "mc_no_targets":
        return fr_system("minimal", DEFAULT_FR), False, {
            "mode": "full", "exclude_identities": target_identity_ids(dataset),
        }
    if name == "mc_drop_2005":
        return fr_system("minimal", DEFAULT_FR), False, {
            "mode": "full", "exclude_identities": drop_non_target_identities(dataset),
        }
    if name.startswith("mc_ids_"):
        count = int(name.rsplit("_", 1)[1])
        return fr_system("minimal", DEFAULT_FR), False, {
            "mode": "identities", "identity_count": count, "seed": 0,
        }
    raise ValueError(f"unknown eval job {name!r}; choose from {EVAL_SUITE}")
