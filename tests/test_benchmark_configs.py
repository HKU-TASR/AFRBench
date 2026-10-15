"""Job configs written by ``afrbench.benchmark`` for each method lifecycle."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from afrbench import benchmark
from afrlib.afr.base import AFRMethod
from afrlib.contracts import ImageSpec, StageInputs

DATASET = Path(__file__).resolve().parents[1] / "data" / "afrbench-basic"


class _Stub(AFRMethod):
    name = "stub"

    def __init__(self, device="cpu", fr_models=(), strength: float = 1.0, *, assets_path=None):
        super().__init__(device, fr_models, assets_path=assets_path)
        self.strength = strength

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=112)

    def get_config(self):
        return {**super().get_config(), "strength": self.strength}

    def protect(self, data):
        return list(data.eval_imgs)


class Untargeted(_Stub):
    eval_inputs = StageInputs(required=("eval_imgs",))


class RuntimeTarget(_Stub):
    eval_inputs = StageInputs(required=("eval_imgs", "target_img"))


class TargetPool(_Stub):
    eval_inputs = StageInputs(required=("eval_imgs", "target_pool"))


class Fitted(_Stub):
    fit_inputs = StageInputs(required=("protectee_album",))
    eval_inputs = StageInputs(required=("eval_imgs",))

    def fit(self, data):
        pass


class Baked(_Stub):
    training_corpus = "ms1m"
    train_inputs = StageInputs(required=("corpus", "target_img"))
    eval_inputs = StageInputs(required=("eval_imgs",))

    def train(self, data):
        pass


@pytest.fixture(autouse=True)
def _stubs_importable(monkeypatch):
    monkeypatch.setitem(sys.modules, "bench_stubs", sys.modules[__name__])


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_untargeted_method(tmp_path):
    paths = benchmark.generate("bench_stubs:Untargeted", DATASET, tmp_path)
    assert {k: len(v) for k, v in paths.items()} == {"train": 0, "fit": 0, "protect": 50, "eval": 1}
    protect = _load(paths["protect"][0])
    assert protect["lifecycle"] == "untargeted" and protect["from_run"] is None
    assert len(protect["protectee"]["images"]) == 50 and protect["protectee"]["album_n"] == 50
    assert protect["afr"] == {"name": "bench_stubs:Untargeted", "set": {"strength": 1.0}}
    evaluation = _load(paths["eval"][0])
    assert list(evaluation["runs"]) == ["untargeted"]
    assert evaluation["gallery"] == {"distractors": {"mode": "full"}}


def test_runtime_target_uses_assigned_targets(tmp_path):
    groups = yaml.safe_load((DATASET / "protectee_groups.json").read_text())["membership"]
    paths = benchmark.generate("bench_stubs:RuntimeTarget", DATASET, tmp_path)
    assert len(paths["protect"]) == 50
    for path in paths["protect"]:
        config = _load(path)
        assert config["target"] == groups[config["protectee"]["identity"]]
        assert config["lifecycle"] == "runtime_fixed_target"
    runs = _load(paths["eval"][0])["runs"]
    assert sorted(runs) == sorted(f"runtimetarget@{t}" for t in set(groups.values()))


def test_pool_fit_and_baked_chaining(tmp_path):
    pool = benchmark.generate("bench_stubs:TargetPool", DATASET, tmp_path / "pool")
    config = _load(pool["protect"][0])
    assert config["lifecycle"] == "target_pool" and config["target_pool_size"] == 32

    fitted = benchmark.generate("bench_stubs:Fitted", DATASET, tmp_path / "fit")
    assert len(fitted["fit"]) == 50
    fit = _load(fitted["fit"][0])
    protect = _load(fitted["protect"][0])
    assert protect["from_run"] == fit["save_path"]

    baked = benchmark.generate("bench_stubs:Baked", DATASET, tmp_path / "baked")
    assert len(baked["train"]) == 5
    train = _load(baked["train"][0])
    assert train["corpus"]["name"] == "ms1m" and train["corpus"]["exclude_identities"]
    protect = _load(baked["protect"][0])
    assert protect["from_run"].endswith(f"/train/baked/{protect['target']}")
    assert protect["lifecycle"] == "target_baked"


def test_held_methods_are_rejected(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.generate("dpa", DATASET, tmp_path)


def test_existing_configs_are_not_overwritten(tmp_path):
    paths = benchmark.generate("bench_stubs:Untargeted", DATASET, tmp_path)
    paths["protect"][0].write_text(paths["protect"][0].read_text() + "# edited\n")
    with pytest.raises(SystemExit):
        benchmark.generate("bench_stubs:Untargeted", DATASET, tmp_path)
    benchmark.generate("bench_stubs:Untargeted", DATASET, tmp_path, overwrite=True)


def test_overrides_and_extra_eval_model(tmp_path):
    paths = benchmark.generate(
        "bench_stubs:Untargeted", DATASET, tmp_path,
        overrides={"strength": 2.0}, evals=("minimal", "jpeg"),
        eval_fr=benchmark.parse_fr("irse-50/arcface/ms1m"),
    )
    assert _load(paths["protect"][0])["afr"]["set"]["strength"] == 2.0
    names = sorted(path.stem for path in paths["eval"])
    assert names == ["jpeg", "jpeg@irse50_arcface_ms1m", "minimal", "minimal@irse50_arcface_ms1m"]


def test_experiment_label_follows_the_dataset_name(tmp_path):
    basic = benchmark.generate("bench_stubs:Untargeted", DATASET, tmp_path / "basic")
    assert _load(basic["protect"][0])["experiment"] == "basic"
    renamed = tmp_path / "renamed"
    shutil.copytree(DATASET, renamed)
    info = json.loads((renamed / "dataset.json").read_text())
    info["name"] = "afrbench-df"
    (renamed / "dataset.json").write_text(json.dumps(info))
    other = benchmark.generate("bench_stubs:Untargeted", renamed, tmp_path / "other")
    assert _load(other["protect"][0])["experiment"] == "df"
