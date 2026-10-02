"""Out-of-tree methods and FR models resolved by ``"package.module:Class"``."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from afrlib import AFR_REGISTRY, FR_REGISTRY
from afrlib.contracts import EvalData
from afrlib.fr.base import FaceRecognizer

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "custom_method"


class TinyRecognizer(FaceRecognizer):
    """A random, weight-free FR model for tests."""

    backend_name = "tests.test_plugins:TinyRecognizer"
    channel = "rgb"
    embedding_dim = 16
    input_size = (112, 112)

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 5, stride=4), torch.nn.ReLU(), torch.nn.Flatten(),
            torch.nn.LazyLinear(self.embedding_dim),
        )
        self.backbone(torch.zeros(1, 3, 112, 112))
        self.register_buffer("_mean", torch.full((1, 3, 1, 1), 0.5), persistent=False)
        self.register_buffer("_std", torch.full((1, 3, 1, 1), 0.5), persistent=False)

    @property
    def name(self) -> str:
        return "tiny"

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def std(self) -> torch.Tensor:
        return self._std

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def _write_module(root: Path, name: str, source: str) -> None:
    (root / f"{name}.py").write_text(source)


def test_registries_resolve_import_paths(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_module(tmp_path, "plug_ok", "from afrlib.afr.fawkes import Fawkes as Method\nclass NotAMethod:\n    pass\n")
    assert AFR_REGISTRY["plug_ok:Method"].__name__ == "Fawkes"
    with pytest.raises(TypeError):
        AFR_REGISTRY["plug_ok:NotAMethod"]
    with pytest.raises(KeyError):
        AFR_REGISTRY["no_such_method"]
    assert len(AFR_REGISTRY) == 26


def test_fr_plugin_must_name_itself(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_module(tmp_path, "plug_fr", (
        "from tests.test_plugins import TinyRecognizer\n"
        "class Good(TinyRecognizer):\n    backend_name = 'plug_fr:Good'\n"
        "class Bad(TinyRecognizer):\n    pass\n"
    ))
    assert FR_REGISTRY["plug_fr:Good"].__name__ == "Good"
    with pytest.raises(ValueError, match="backend_name"):
        FR_REGISTRY["plug_fr:Bad"]
    assert sorted(FR_REGISTRY) == ["facenet_pytorch", "frbench"]


def test_method_digest_tracks_plugin_source(tmp_path, monkeypatch):
    from afrbench.configs import JobConfig, _method_code_digest

    monkeypatch.syspath_prepend(str(tmp_path))
    package = tmp_path / "plug_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "method.py").write_text("X = 1\n")
    config = JobConfig.from_dict({
        "kind": "protect", "experiment": "t", "save_path": str(tmp_path / "out"),
        "afr": "plug_pkg.method:M", "surrogates": [],
    })
    before = _method_code_digest(config)
    (package / "method.py").write_text("X = 2\n")
    assert _method_code_digest(config) != before


def test_example_method_protects_within_budget(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    method = AFR_REGISTRY["pgd_cloak:PGDCloak"]("cpu", [TinyRecognizer()], epsilon=8.0, steps=2)
    torch.manual_seed(0)
    photo = torch.rand(3, 240, 200) * 255
    geometry = {
        "landmarks": [[80, 100], [120, 100], [100, 125], [85, 150], [115, 150]],
        "face_box": [60, 70, 140, 175],
    }
    out = method.protect(EvalData(eval_imgs=[photo], eval_geometry=[geometry]))[0]
    assert out.shape == photo.shape
    assert 0 < float((out - photo).abs().max()) <= 8.0 + 1e-3
    assert "pgd_cloak" in sys.modules
