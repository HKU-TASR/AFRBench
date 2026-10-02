"""FR-system ops, pipeline order, gallery digest, and digest pins."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from afrbench.dataset import BenchDataset, decode_image
from afrbench.evaluator.embed import LazyImages
from afrbench.fr_system import FRSystem
from afrbench.fr_system.ops import jpeg, median, quantize, resize
from afrbench.fr_system.system import ALIGNED_CROP_SIDE, _host_materialize
from afrbench.runs import ProtectedImage
from afrlib.utils.face_crop import extract_frame
from afrbench.protocol import fr_system

MINIMAL_DIGEST = "3a9efb8b68aa9518"
QUANTIZED_DIGEST = "820b39bd3fcb6f74"
TTA_DIGEST = "b3993cfae63412a2"


def _rgb(side=16, value=12.3):
    return torch.full((3, side, side), float(value))


def test_quantize_idempotent_and_integer_valued():
    image = _rgb()
    once = quantize(image)
    twice = quantize(once)
    assert torch.equal(once, twice)
    assert torch.equal(once, once.round())
    assert once.min() >= 0 and once.max() <= 255


def test_jpeg_shape_range_and_changes_at_q50():
    torch.manual_seed(0)
    image = torch.rand(3, 32, 32) * 255
    out = jpeg(image, quality=50)
    assert out.shape == image.shape
    assert out.min() >= 0 and out.max() <= 255
    assert not torch.equal(out, quantize(image))


def test_jpeg_pads_non_multiple_of_16():
    torch.manual_seed(0)
    image = torch.rand(3, 17, 25) * 255
    out = jpeg(image, quality=75)
    assert out.shape == image.shape


def test_jpeg_output_need_not_be_integer():
    torch.manual_seed(0)
    image = torch.rand(3, 16, 16) * 255
    out = jpeg(image, quality=75)
    # DiffJPEG may return non-integers; uint8-grid is quantize's job.
    assert out.dtype == torch.float32


def test_resize_preserves_shape():
    image = _rgb(33)
    assert resize(image, factor=0.4).shape == image.shape
    tiny = resize(image, factor=0.01)
    assert tiny.shape == image.shape


def test_median_default_k_is_3():
    image = torch.rand(3, 33, 33, generator=torch.Generator().manual_seed(0)) * 255
    default = median(image)
    assert default.shape == image.shape
    assert torch.equal(default, median(image, kernel_size=3))
    assert torch.equal(median(image, kernel_size=4), median(image, kernel_size=5))
    assert not torch.equal(default, median(image, kernel_size=5))


def test_pipeline_order_via_stub_ops(monkeypatch):
    order = []

    def _record(name):
        def op(image, **_kwargs):
            order.append(name)
            return image

        return op

    monkeypatch.setattr(
        "afrbench.fr_system.system.OPS",
        {name: _record(name) for name in ("quantize", "jpeg", "resize", "median")},
    )
    system = FRSystem.from_dict({
        "name": "order",
        "fr": {"backend": "frbench", "backbone": "irse-100", "loss": "arcface", "dataset": "ms1m"},
        "pipeline": {
            "quantize": {"enabled": True, "precision": "uint8"},
            "jpeg": {"enabled": True, "quality": 75, "impl": "diffjpeg"},
            "resize": {"enabled": True, "factor": 0.5, "mode": "bicubic"},
            "purify": {"enabled": True, "method": "median", "kernel_size": 3},
            "detect": {"detector": "retinaface_resnet50"},
            "align": {"template": "arcface_112"},
            "tta": {"enabled": False},
        },
    })
    system._apply_pre_detect(_rgb())
    assert order == ["quantize", "jpeg", "resize"]


class _StubFR:
    input_size = (ALIGNED_CROP_SIDE, ALIGNED_CROP_SIDE)
    embedding_dim = 4

    def embed_crops(self, crops, **_kwargs):
        return torch.ones(crops.shape[0], self.embedding_dim)


def test_embed_aligned_skips_model_restorer(monkeypatch):
    called = {"n": 0}

    def fake_gfpgan(image, **_kwargs):
        called["n"] += 1
        return image

    monkeypatch.setattr("afrbench.fr_system.ops.gfpgan", fake_gfpgan)
    monkeypatch.setattr("afrbench.fr_system.system.OPS", {**__import__("afrbench.fr_system.ops", fromlist=["OPS"]).OPS, "gfpgan": fake_gfpgan})
    monkeypatch.setattr(FRSystem, "build_fr", lambda self, device: _StubFR())
    system = FRSystem.from_dict(fr_system("gfpgan"))
    crop = torch.rand(3, 112, 112) * 255
    system.embed_aligned([("a", crop)], device="cpu", batch_size=1)
    assert called["n"] == 0


def test_embed_aligned_stays_at_112(monkeypatch):
    seen = []

    def fake_apply(self, image):
        seen.append(tuple(image.shape[-2:]))
        return image

    monkeypatch.setattr(FRSystem, "_apply_pre_detect", fake_apply)
    monkeypatch.setattr(FRSystem, "build_fr", lambda self, device: _StubFR())
    system = FRSystem.from_dict(fr_system("jpeg"))
    crop = torch.rand(3, 112, 112) * 255
    system.embed_aligned([("a", crop)], device="cpu", batch_size=1)
    assert seen == [(112, 112)]


def test_from_dict_rejects_position_and_ignores_emulation():
    raw = fr_system("minimal")
    raw["pipeline"]["purify"]["position"] = "face"
    with pytest.raises(ValueError, match="position"):
        FRSystem.from_dict(raw)
    leftover = fr_system("minimal")
    leftover["pipeline"]["noise_emulation"] = {"enabled": True, "face_side": 224}
    system = FRSystem.from_dict(leftover)
    assert "noise_emulation" not in system.pipeline
    assert "noise_emulation" not in system.to_dict()["pipeline"]


def test_from_dict_rejects_enabled_gaussian_and_drops_leftover():
    raw = fr_system("minimal")
    raw["pipeline"]["gaussian"] = {"enabled": True, "kernel_size": 9, "sigma": 2.0}
    with pytest.raises(ValueError, match="gaussian"):
        FRSystem.from_dict(raw)
    leftover = fr_system("minimal")
    leftover["pipeline"]["gaussian"] = {"enabled": False, "kernel_size": 9, "sigma": 2.0}
    system = FRSystem.from_dict(leftover)
    assert "gaussian" not in system.pipeline
    assert "gaussian" not in system.to_dict()["pipeline"]
    assert system.to_dict()["pipeline"]["purify"]["kernel_size"] == 3


def test_for_gallery_digest_equals_purify_off():
    off = FRSystem.from_dict(fr_system("minimal"))
    assert off.for_gallery() is off
    for preset in ("median", "gfpgan", "codeformer", "diffpure"):
        system = FRSystem.from_dict(fr_system(preset))
        gallery = system.for_gallery()
        assert gallery is not system
        assert gallery.digest == off.digest
        assert gallery.pipeline["purify"]["enabled"] is False


def test_for_gallery_strips_quantize_only_when_jpeg_and_tta_off():
    minimal = FRSystem.from_dict(fr_system("minimal"))
    quantized = FRSystem.from_dict(fr_system("quantized"))
    gallery = quantized.for_gallery()
    assert gallery is not quantized
    assert gallery.digest == minimal.digest
    assert gallery.pipeline["quantize"]["enabled"] is False
    assert quantized.digest == QUANTIZED_DIGEST
    full = FRSystem.from_dict(fr_system("full"))
    assert full.for_gallery() is full
    assert full.pipeline["quantize"]["enabled"] is True
    tta = FRSystem.from_dict(fr_system("tta"))
    assert tta.for_gallery() is tta


def test_minimal_and_quantized_digests_are_pinned():
    global MINIMAL_DIGEST, QUANTIZED_DIGEST, TTA_DIGEST
    minimal = FRSystem.from_dict(fr_system("minimal"))
    quantized = FRSystem.from_dict(fr_system("quantized"))
    tta = FRSystem.from_dict(fr_system("tta"))
    # On a mismatch, the assertion message prints the digest to re-pin.
    assert minimal.digest == MINIMAL_DIGEST, f"re-pin MINIMAL_DIGEST = {minimal.digest!r}"
    assert quantized.digest == QUANTIZED_DIGEST, f"re-pin QUANTIZED_DIGEST = {quantized.digest!r}"
    assert tta.digest == TTA_DIGEST, f"re-pin TTA_DIGEST = {tta.digest!r}"
    assert "noise_emulation" not in minimal.to_dict()["pipeline"]
    assert "gaussian" not in minimal.to_dict()["pipeline"]
    assert "position" not in minimal.to_dict()["pipeline"]["purify"]
    assert minimal.to_dict()["pipeline"]["jpeg"]["impl"] == "diffjpeg"
    assert minimal.to_dict()["pipeline"]["purify"]["kernel_size"] == 3


def test_jpeg_batch_matches_per_image():
    torch.manual_seed(0)
    batch = torch.rand(4, 3, 24, 24) * 255
    batched = jpeg(batch, quality=60)
    looped = torch.stack([jpeg(batch[index], quality=60) for index in range(4)])
    assert batched.shape == batch.shape
    assert torch.allclose(batched, looped, atol=1e-5)


def test_embed_aligned_reuses_resident_fr(monkeypatch):
    from afrbench.fr_system import system as system_mod

    system_mod._FR_CACHE.clear()
    calls = {"n": 0}

    def fake_build(spec, device):
        calls["n"] += 1
        return _StubFR()

    monkeypatch.setattr(system_mod, "_construct_fr", fake_build)
    system = FRSystem.from_dict(fr_system("minimal"))
    crop = torch.rand(3, 112, 112) * 255
    system.embed_aligned([("a", crop), ("b", crop)], device="cpu", batch_size=1)
    assert calls["n"] == 1
    system.embed_aligned([("c", crop)], device="cpu", batch_size=1)
    assert calls["n"] == 1


def test_jpeg_without_quantize_stays_off():
    raw = fr_system("minimal")
    raw["pipeline"]["jpeg"]["enabled"] = True
    raw["pipeline"]["quantize"]["enabled"] = False
    system = FRSystem.from_dict(raw)
    assert system.pipeline["jpeg"]["enabled"] is True
    assert system.pipeline["quantize"]["enabled"] is False
    preset = fr_system("jpeg")
    assert preset["pipeline"]["quantize"]["enabled"] is True


def _asset_cached(key: str) -> bool:
    try:
        from afrlib.utils.assets import _lookup
        import afrbench  # noqa: F401

        return _lookup(key).is_cached()
    except Exception:
        return False


@pytest.mark.skipif(not _asset_cached("bench/gfpgan_v1.3.pth"), reason="GFPGAN weights not cached")
def test_gfpgan_op_on_512_canvas():
    from afrbench.fr_system.ops import gfpgan

    torch.manual_seed(0)
    canvas = torch.rand(3, 512, 512) * 255
    out = gfpgan(canvas)
    assert out.shape == canvas.shape
    assert out.min() >= -1 and out.max() <= 256


@pytest.mark.skipif(not _asset_cached("bench/gfpgan_v1.3.pth"), reason="GFPGAN weights not cached")
def test_gfpgan_rejects_non_canvas():
    from afrbench.fr_system.ops import gfpgan

    with pytest.raises(ValueError, match="512"):
        gfpgan(torch.rand(3, 112, 112) * 255)


@pytest.mark.skipif(not _asset_cached("bench/codeformer.pth"), reason="CodeFormer weights not cached")
def test_codeformer_rejects_non_canvas():
    from afrbench.fr_system.ops import codeformer

    with pytest.raises(ValueError, match="512"):
        codeformer(torch.rand(3, 112, 112) * 255)


@pytest.mark.skipif(not _asset_cached("bench/diffpure_celeba_hq.ckpt"), reason="DiffPure weights not cached")
def test_diffpure_rejects_non_canvas():
    from afrbench.fr_system.ops import diffpure

    with pytest.raises(ValueError, match="256"):
        diffpure(torch.rand(3, 112, 112) * 255)


def test_embed_aligned_batch_matches_list(monkeypatch):
    monkeypatch.setattr(FRSystem, "build_fr", lambda self, device: _StubFR())
    system = FRSystem.from_dict(fr_system("minimal"))
    torch.manual_seed(1)
    crops = [torch.rand(3, 112, 112) * 255 for _ in range(5)]
    items = [(f"k{i}", crops[i]) for i in range(5)]
    ids, listed, failed = system.embed_aligned(items, device="cpu", batch_size=2)
    batched = torch.cat(
        [
            system.embed_aligned_batch(torch.stack(crops[start : start + 2]), "cpu")
            for start in range(0, 5, 2)
        ],
        dim=0,
    )
    assert failed == []
    assert ids == [f"k{i}" for i in range(5)]
    assert torch.equal(listed, batched)


class _AlwaysAlign:
    def detect(self, images):
        return [object()] * len(images)

    def align(self, images, detections, size=None, **_kwargs):
        height, width = size if size is not None else (ALIGNED_CROP_SIDE, ALIGNED_CROP_SIDE)
        out = []
        for image in images:
            crop = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            )[0]
            out.append(crop.unsqueeze(0))
        return out


def _write_jpeg(path: Path, seed: int, size: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    array = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95)


def _path_a_dataset(root: Path) -> BenchDataset:
    images = []
    for index in range(4):
        rel = f"protectees/P00/{index}.jpg"
        _write_jpeg(root / rel, seed=10 + index, size=48)
        images.append({
            "path": rel,
            "identity": "P00",
            "split": "protectee",
            "role": "query" if index % 2 else "gallery",
            "image_id": f"P00-{index}",
            "sha256": f"p00{index}",
            "size": [48, 48],
            "session": f"P00-{index}",
            "original_box": [4, 4, 32],
            "face_box": [8, 8, 40, 40],
            "face_square": [8, 8, 32],
            "landmarks": [[12, 12], [28, 12], [20, 20], [14, 28], [26, 28]],
        })
    (root / "annotations.json").write_text(json.dumps({
        "schema": "afrbench-dataset-v4",
        "images": images,
        "meta": {},
    }))
    (root / "identities.json").write_text(json.dumps({
        "schema": "afrbench-identities-v4",
        "identities": {"P00": {}},
    }))
    return BenchDataset(root)


def test_path_a_prefetch_matches_serial_jpeg_and_npy(tmp_path, monkeypatch):
    dataset = _path_a_dataset(tmp_path)
    records = [dataset.record(record.path) for record in dataset.records]
    jpeg_ids = [record.path for record in records]
    jpeg_loader_threads: list[int] = []
    main_thread = threading.get_ident()

    def load_jpeg(path: str) -> torch.Tensor:
        jpeg_loader_threads.append(threading.get_ident())
        record = dataset.record(path)
        if record.original_box is not None:
            return dataset.original(record)
        return dataset.image(record)

    npy_ids = []
    npy_loader_threads: list[int] = []
    npy_items = []
    for index, record in enumerate(records):
        array = (torch.rand(3, 40, 40) * 255).numpy().astype(np.float32)
        out = tmp_path / f"protected_{index}.npy"
        np.save(out, array)
        image = ProtectedImage(
            source_path=record.path, output=out, dtype="float32"
        )
        npy_ids.append(record.path)
        npy_items.append(image)

    def load_npy(path: str) -> torch.Tensor:
        npy_loader_threads.append(threading.get_ident())
        for image in npy_items:
            if image.source_path == path:
                return image.load()
        raise KeyError(path)

    monkeypatch.setattr(FRSystem, "build_fr", lambda self, device: _StubFR())
    monkeypatch.setattr(
        "afrbench.fr_system.detector.get_detector",
        lambda *args, **kwargs: _AlwaysAlign(),
    )
    system = FRSystem.from_dict(fr_system("minimal"))
    to_threads: list[int] = []
    original_to = torch.Tensor.to

    def tracked_to(self, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if isinstance(target, (str, torch.device)):
            to_threads.append(threading.get_ident())
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)

    jpeg_lazy = LazyImages(jpeg_ids, load_jpeg)
    npy_lazy = LazyImages(npy_ids, load_npy)
    jpeg_serial = system.embed(jpeg_lazy, device="cpu", batch_size=2, prefetch=False)
    jpeg_pref = system.embed(jpeg_lazy, device="cpu", batch_size=2, prefetch=True)
    npy_serial = system.embed(npy_lazy, device="cpu", batch_size=2, prefetch=False)
    npy_pref = system.embed(npy_lazy, device="cpu", batch_size=2, prefetch=True)

    assert jpeg_serial[0] == jpeg_pref[0] == jpeg_ids
    assert npy_serial[0] == npy_pref[0] == npy_ids
    assert torch.equal(jpeg_serial[1], jpeg_pref[1])
    assert torch.equal(npy_serial[1], npy_pref[1])
    assert jpeg_serial[2] == jpeg_pref[2] == []
    assert npy_serial[2] == npy_pref[2] == []
    assert any(thread != main_thread for thread in jpeg_loader_threads)
    assert any(thread != main_thread for thread in npy_loader_threads)
    assert to_threads
    assert all(thread == main_thread for thread in to_threads)

    already = [(image_id, load_jpeg(image_id)) for image_id in jpeg_ids]
    listed = system.embed(already, device="cpu", batch_size=2, prefetch=True)
    assert torch.equal(listed[1], jpeg_serial[1])

    noise_threads: list[int] = []

    def load_noise(path: str) -> torch.Tensor:
        noise_threads.append(threading.get_ident())
        return load_jpeg(path)

    noise = LazyImages(jpeg_ids, load_noise, host_prefetch=False)
    system.embed(noise, device="cpu", batch_size=2, prefetch=True)
    assert noise_threads
    assert all(thread == main_thread for thread in noise_threads)

    chunk = _host_materialize(jpeg_lazy, 0, 2)
    assert [item[0] for item in chunk] == jpeg_ids[:2]
    # Materialized frames equal extract_frame applied to decode_image of the JPEG.
    first = dataset.original(records[0])
    assert torch.equal(chunk[0][1], first)
    raw = decode_image((tmp_path / records[0].path).read_bytes())
    assert torch.equal(dataset.image(records[0]), raw)
    assert torch.equal(
        first,
        extract_frame(raw, tuple(records[0].original_box), tuple(records[0].size[:2])),
    )


def test_purify_resize_is_rejected():
    from afrbench.fr_system import FRSystem

    raw = {
        "name": "x",
        "fr": {"backend": "frbench", "backbone": "irse-100", "loss": "arcface", "dataset": "ms1m"},
        "pipeline": {"purify": {"enabled": True, "method": "resize"}},
    }
    with pytest.raises(ValueError, match="purify.method"):
        FRSystem.from_dict(raw)
