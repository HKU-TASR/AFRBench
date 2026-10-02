"""Visual-quality metrics: per-image values, caching, eval-digest exclusion, CUDA/CPU agreement."""
from __future__ import annotations

import pytest
import torch

from afrbench.configs import JobConfig, _eval_code_digest
from afrbench.evaluator.metrics import quality


def test_eval_code_digest_skips_quality_py(monkeypatch):
    captured: list = []

    def fake_hash(paths, root):
        captured.extend(paths)
        return "digest"

    monkeypatch.setattr("afrbench.configs._hash_files", fake_hash)
    config = JobConfig(
        kind="eval", experiment="t", seed=0, save_path="x", body={},
    )
    assert _eval_code_digest(config) == "digest"
    names = {path.name for path in captured}
    assert "quality.py" not in names
    assert "report.py" not in names
    assert "job.py" in names


@pytest.mark.slow
def test_per_image_keys_and_perturbation():
    clean = torch.rand(3, 176, 176) * 255
    protected = (clean + 3.0).clamp(0, 255)
    values = quality.per_image(protected, clean, "cpu")
    assert values["psnr"] < 99.0
    assert 0.0 < values["changed_fraction"] <= 1.0
    assert set(values) >= {
        "psnr", "ssim", "ms_ssim", "lpips",
        "brisque", "niqe",
        "changed_fraction", "mae", "rmse", "max_abs",
    }


def test_cached_visual_quality_skips_recompute(tmp_path, monkeypatch):
    from afrbench.evaluator import job

    calls = {"n": 0}

    def fake(run, dataset, device):
        calls["n"] += 1
        return {"n_images": 1, "psnr": 30.0}, [{"source_path": "a", "psnr": 30.0}]

    monkeypatch.setattr(job, "_visual_quality", fake)
    first = job._cached_visual_quality(tmp_path, "adv_diffusion@1", None, None, "cpu")
    second = job._cached_visual_quality(tmp_path, "adv_diffusion@1", None, None, "cpu")
    assert first == second
    assert calls["n"] == 1
    assert (tmp_path / "vq_partial" / "adv_diffusion@1.json").is_file()


def test_no_reference_cuda_matches_cpu():
    if not torch.cuda.is_available():
        return
    image = torch.rand(3, 192, 192) * 255
    cpu = quality.no_reference(image, "niqe", "cpu")
    gpu = quality.no_reference(image, "niqe", "cuda")
    assert abs(cpu - gpu) < 5e-3
