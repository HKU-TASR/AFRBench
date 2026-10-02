"""DivTrackee queue / guide losses vs transcribed released formulas."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "afrlib" / "afr" / "divtrackee" / "divtrackee.py"


def _source_region() -> str:
    """Source of ``DivTrackee._queue_loss``, located by name rather than line numbers."""
    lines = SOURCE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "def _queue_loss" in line)
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1:], start=start + 1)
            if line.strip().startswith("def ")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _queue_loss(method, render_embeddings):
    # Copy of DivTrackee._queue_loss; upstream: EmbeddingQueue.get_queue_loss in
    # utils/divtrackee_utils.py of https://github.com/fiora6/divtrackee.
    if len(method._queue) == 0:
        return torch.zeros((), device=method.device)
    total = sum(
        sum(1 - F.cosine_similarity(g, q).mean() for g, q in zip(render_embeddings, entry))
        for entry in method._queue
    )
    return -total / len(method._queue)


def _ref_queue_loss(queue, render_embeddings, device):
    if not queue:
        return torch.zeros((), device=device)
    total = sum(
        sum(1 - F.cosine_similarity(g, q).mean() for g, q in zip(render_embeddings, entry))
        for entry in queue
    )
    return -total / len(queue)


def _ref_guide_loss(render_embeddings, aux_embeddings, margin):
    return sum(
        (1 - F.cosine_similarity(g, a).mean() - margin).clamp(min=0)
        for g, a in zip(render_embeddings, aux_embeddings)
    )


def test_queue_loss_matches_transcribed_formula():
    device = torch.device("cpu")
    method = SimpleNamespace(device=device, _queue=[])
    torch.manual_seed(0)
    render = [torch.randn(1, 8), torch.randn(1, 8)]
    assert torch.allclose(_queue_loss(method, render), torch.zeros(()))
    method._queue.append([torch.randn(1, 8), torch.randn(1, 8)])
    method._queue.append([torch.randn(1, 8), torch.randn(1, 8)])
    ours = _queue_loss(method, render)
    expected = _ref_queue_loss(method._queue, render, device)
    assert torch.allclose(ours, expected, atol=1e-5, rtol=1e-5)


def test_guide_loss_matches_transcribed_formula():
    torch.manual_seed(1)
    render = [torch.randn(1, 8), torch.randn(1, 8)]
    aux = [torch.randn(1, 8), torch.randn(1, 8)]
    margin = 0.2
    ours = sum(
        (1 - F.cosine_similarity(g, a).mean() - margin).clamp(min=0)
        for g, a in zip(render, aux)
    )
    expected = _ref_guide_loss(render, aux, margin)
    assert torch.allclose(ours, expected, atol=1e-5, rtol=1e-5)


def test_queue_loss_source_region_is_cited():
    region = _source_region()
    assert "def _queue_loss" in region
    assert "len(self._queue)" in region
    digest = hashlib.sha256(region.encode()).hexdigest()
    assert len(digest) == 64
