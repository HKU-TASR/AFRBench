"""AMT-GAN fallbacks: default face box on a HOG miss, source crop on a per-image failure."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("dlib")

from afrlib.afr.amt_gan.amt_gan import AMT_GAN  # noqa: E402
from afrlib.afr.amt_gan.preprocess import PreparedFace, default_face_rect  # noqa: E402
from afrlib.contracts import EvalData  # noqa: E402


def test_default_face_rect_insets_aligned_crop():
    box = default_face_rect(256, 256, inset=0.12)
    assert box.left() == 31
    assert box.top() == 31
    assert box.right() == 225
    assert box.bottom() == 225


def test_prepare_inference_uses_default_box_when_hog_misses(monkeypatch):
    from afrlib.afr.amt_gan import preprocess as prep

    captured = {}

    class _Stub:
        image_size = 256
        device = torch.device("cpu")

        def _detect(self, _crop):
            return prep.dlib.rectangles()

        def _landmarks(self, crop, face):
            captured["face"] = face
            return np.stack(
                [np.linspace(40, 200, 68), np.linspace(40, 200, 68)],
                axis=1,
            )

        def _process_regions(self, _mask, landmarks, _device):
            return torch.zeros(3, 1, 256, 256), torch.zeros(3, 136, 64, 64)

    class _Parser:
        def parse(self, _array):
            return torch.zeros(512, 512)

    stub = _Stub()
    stub.face_parser = _Parser()
    crop = Image.new("RGB", (256, 256), (80, 80, 80))
    prepared = prep.AMTGANPreprocessor.prepare_inference(stub, crop)
    assert prepared is not None
    assert captured["face"].left() == default_face_rect(256, 256).left()
    assert captured["face"].right() == default_face_rect(256, 256).right()


def test_prepare_inference_returns_none_when_predictor_fails(monkeypatch):
    from afrlib.afr.amt_gan import preprocess as prep

    class _Stub:
        image_size = 256
        device = torch.device("cpu")

        def _detect(self, _crop):
            return prep.dlib.rectangles()

        def _landmarks(self, _crop, _face):
            raise RuntimeError("predictor failed")

    class _Parser:
        def parse(self, _array):
            return torch.zeros(512, 512)

    stub = _Stub()
    stub.face_parser = _Parser()
    crop = Image.new("RGB", (256, 256), (80, 80, 80))
    assert prep.AMTGANPreprocessor.prepare_inference(stub, crop) is None


def test_protect_returns_source_crop_when_prepare_inference_fails(monkeypatch):
    method = AMT_GAN.__new__(AMT_GAN)
    method.device = torch.device("cpu")
    method.G = torch.nn.Identity()
    crop = torch.linspace(0, 255, 3 * 8 * 8).reshape(3, 8, 8)

    class _Preprocessor:
        def prepare_inference(self, image):
            if image.size == (32, 32):
                return PreparedFace(
                    torch.zeros(1, 3, 256, 256),
                    torch.zeros(3, 1, 256, 256),
                    torch.zeros(3, 136, 64, 64),
                )
            return None

    monkeypatch.setattr(method, "_check_inputs", lambda *_args, **_kw: None)
    monkeypatch.setattr(method, "_require_ready", lambda: None)
    monkeypatch.setattr(method, "_validate_image_batch", lambda photo, **_kw: [photo])
    monkeypatch.setattr(method, "_get_preprocessor", lambda: _Preprocessor())
    monkeypatch.setattr(method, "_load_makeup_reference", lambda _name: Image.new("RGB", (32, 32)))
    monkeypatch.setattr(
        method,
        "_prepared_to_device",
        lambda prepared: (prepared.image, prepared.masks, prepared.distances),
    )
    monkeypatch.setattr(
        "afrlib.afr.amt_gan.amt_gan.prepare_inputs",
        lambda photos, geometry, spec: ([crop], [None]),
    )
    monkeypatch.setattr(
        "afrlib.afr.amt_gan.amt_gan.paste_faces",
        lambda photos, faces, plans: list(faces),
    )
    photo = torch.zeros(3, 8, 8)
    out = method.protect(
        EvalData(
            eval_imgs=[photo],
            eval_geometry=[{"landmarks": [], "face_box": [0, 0, 1, 1]}],
            makeup_reference_name="ref01.png",
        )
    )
    assert len(out) == 1
    assert torch.equal(out[0], crop)
    assert torch.isfinite(out[0]).all()
