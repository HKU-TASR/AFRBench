"""Evaluator ranking, metrics, report rows, visual-quality reuse, and eval-lock currentness."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import torch

from afrbench import configs
from afrbench.configs import JobConfig, is_current, write_lock
from afrbench.evaluator.job import _load_visual_quality, embed_eval_sources
from afrbench.evaluator.metrics.ranking import Ranking, compute_ranking
from afrbench.evaluator.metrics.retrieval import average_precision_per_query
from afrbench.evaluator.report import _query_mix_rows


def _unit_vector(cosine: float) -> list[float]:
    return [cosine, math.sqrt(1.0 - cosine**2)]


def test_compute_ranking_is_global_and_chunk_invariant():
    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([
        _unit_vector(0.6),  # relevant
        _unit_vector(0.5),
        _unit_vector(0.9),
        _unit_vector(0.8),
        _unit_vector(0.7),  # relevant
        _unit_vector(0.4),
    ])
    gallery_labels = ["user", "noise", "noise", "noise", "user", "noise"]

    chunked = compute_ranking(
        queries,
        gallery,
        ["user"],
        gallery_labels,
        top_k=4,
        chunk=2,
    )
    unchunked = compute_ranking(
        queries,
        gallery,
        ["user"],
        gallery_labels,
        top_k=4,
        chunk=len(gallery),
    )

    assert chunked.ranks[0].tolist() == [3, 4]
    assert torch.equal(chunked.ranks[0], unchunked.ranks[0])
    assert chunked.top_labels == unchunked.top_labels
    assert torch.allclose(chunked.top_sims, unchunked.top_sims)


def test_compute_ranking_matches_brute_force_for_multiple_labels():
    generator = torch.Generator().manual_seed(7)
    queries = torch.nn.functional.normalize(
        torch.randn(4, 8, generator=generator), dim=1
    )
    gallery = torch.nn.functional.normalize(
        torch.randn(31, 8, generator=generator), dim=1
    )
    query_labels = ["a", "b", "c", "missing"]
    gallery_labels = ["a", "b", "c", "noise"] * 7 + ["a", "b", "noise"]
    similarities = queries @ gallery.T

    ranking = compute_ranking(
        queries,
        gallery,
        query_labels,
        gallery_labels,
        top_k=10,
        chunk=5,
    )

    for index, label in enumerate(query_labels):
        relevant = torch.tensor([value == label for value in gallery_labels])
        own = similarities[index, relevant]
        expected = torch.sort(
            1 + (similarities[index].unsqueeze(1) > own.unsqueeze(0)).sum(dim=0)
        ).values
        assert torch.equal(ranking.ranks[index], expected)
    assert torch.allclose(
        ranking.top_sims,
        torch.topk(similarities, 10, dim=1).values,
    )


def test_primary_without_key_keeps_top2_alignment():
    """A primary-gallery hit without a key still occupies its top-k slot (key ``None``)."""
    from afrbench.evaluator.metrics.identification import per_query

    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([_unit_vector(0.99)])
    extra = (
        torch.tensor([_unit_vector(0.5)]),
        ["noise/other"],
        ["other__0.jpg"],
    )
    ranking = compute_ranking(
        queries,
        gallery,
        ["user"],
        ["user"],
        top_k=2,
        extra_blocks=[extra],
    )
    assert ranking.top_keys[0][0] is None
    assert ranking.top_keys[0][1] == "other__0.jpg"
    scenario = type("S", (), {"query_labels": ["user"]})()
    metrics = per_query(ranking, scenario, ks=(1, 2))
    assert all(not name.startswith("target_") for name in metrics)
    assert metrics["rank1_appearance"] == [1.0]


def test_ap_at_k_uses_the_standard_relevant_count_denominator():
    ranking = Ranking(
        ranks=[torch.tensor([1, 200, 300])],
        top_labels=[[]],
        top_sims=torch.empty((1, 0)),
        gallery_size=300,
    )

    assert average_precision_per_query(ranking, k=100) == pytest.approx([1.0 / 3.0])
    assert average_precision_per_query(ranking, k=2) == pytest.approx([0.5])
    assert average_precision_per_query(ranking) == pytest.approx([
        (1.0 + 2.0 / 200.0 + 3.0 / 300.0) / 3.0
    ])


def test_query_mix_assignments_are_paired_across_runs():
    rows = []
    for run in ("method-a", "method-b"):
        for query in range(20):
            for scenario, protected in (("1A", 0.0), ("2A", 1.0)):
                rows.append({
                    "run": run,
                    "fr_system": "fr",
                    "scenario": scenario,
                    "query_id": f"query-{query}",
                    "protected": protected,
                })

    mixed = _query_mix_rows(pd.DataFrame(rows))
    half = mixed[mixed["scenario"] == "mix-query-r0.5"]
    assignments = half.pivot(index="query_id", columns="run", values="protected")

    assert assignments["method-a"].tolist() == assignments["method-b"].tolist()


_FR_BLOCK = {
    "name": "fr",
    "fr": {"backend": "frbench", "backbone": "irse-100", "loss": "arcface", "dataset": "ms1m"},
    "pipeline": {
        "quantize": {"enabled": False, "precision": "uint8"},
        "jpeg": {"enabled": False, "quality": 75, "impl": "diffjpeg"},
        "resize": {"enabled": False, "factor": 0.4, "mode": "bicubic"},
        "purify": {"enabled": False, "method": "median", "kernel_size": 3},
        "detect": {"detector": "retinaface_resnet50", "keep": "largest", "on_fail": "count_as_evaded"},
        "align": {"template": "arcface_112"},
        "tta": {"enabled": False, "ops": ["flip_horizontal"]},
    },
}


def test_reuse_visual_quality_cli_accepts_metrics_without_a_lock(tmp_path, monkeypatch):
    from afrbench.__main__ import _assert_reusable_visual_quality, main

    (tmp_path / "metrics.json").write_text(json.dumps({
        "fr_system": "fr",
        "visual_quality": {"method": {"psnr": 42.0, "n_images": 1}},
    }))
    (tmp_path / "per_image.jsonl").write_text(json.dumps({
        "run": "method",
        "fr_system": "fr",
        "scenario": "visual_quality",
        "psnr": 42.0,
    }) + "\n")
    yaml_path = tmp_path / "eval.yaml"
    yaml_path.write_text(
        "kind: eval\nexperiment: test\nseed: 0\n"
        f"save_path: {tmp_path}\nvisual_quality: true\n"
        "runs:\n  method: /tmp/run\n"
        "fr_system:\n  name: fr\n"
        "  fr: {backend: frbench, backbone: irse-100, loss: arcface, dataset: ms1m}\n"
        "  pipeline:\n"
        "    quantize: {enabled: false, precision: uint8}\n"
        "    jpeg: {enabled: false, quality: 75, impl: diffjpeg}\n"
        "    resize: {enabled: false, factor: 0.4, mode: bicubic}\n"
        "    purify: {enabled: false, method: median, kernel_size: 3}\n"
        "    detect: {detector: retinaface_resnet50, keep: largest, on_fail: count_as_evaded}\n"
        "    align: {template: arcface_112}\n"
        "    tta: {enabled: false, ops: [flip_horizontal]}\n"
    )
    config = JobConfig.load(yaml_path)
    _assert_reusable_visual_quality(config)
    assert not (tmp_path / "config.lock.yaml").is_file()

    called = {}

    def fake_run_job(cfg, device, *, reuse_visual_quality=False, force=False):
        called["reuse"] = reuse_visual_quality
        called["force"] = force

    import afrbench.__main__ as entrance
    monkeypatch.setattr(entrance, "run_job", fake_run_job)
    assert main([str(yaml_path), "--force", "--reuse-visual-quality"]) == 0
    assert called == {"reuse": True, "force": True}


def test_reuse_visual_quality_cli_rejects_missing_vq_section(tmp_path):
    from afrbench.__main__ import _assert_reusable_visual_quality

    (tmp_path / "metrics.json").write_text(json.dumps({"fr_system": "fr"}))
    (tmp_path / "per_image.jsonl").write_text("{}\n")
    config = JobConfig(
        kind="eval",
        experiment="test",
        seed=0,
        save_path=str(tmp_path),
        body={"visual_quality": True, "runs": {"method": "/tmp/run"}, "fr_system": _FR_BLOCK},
    )
    with pytest.raises(ValueError, match="visual-quality runs"):
        _assert_reusable_visual_quality(config)


def test_visual_quality_can_be_reused_for_metric_only_rescoring(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({
        "fr_system": "fr",
        "visual_quality": {"method": {"psnr": 42.0, "n_images": 1}},
    }))
    quality_row = {
        "run": "method",
        "fr_system": "fr",
        "scenario": "visual_quality",
        "query_id": "protectees/person/image.jpg",
        "psnr": 42.0,
    }
    (tmp_path / "per_image.jsonl").write_text(
        json.dumps({"scenario": "1A", "run": "method"}) + "\n"
        + json.dumps(quality_row) + "\n"
    )

    reused = _load_visual_quality(tmp_path, {"method": object()}, "fr")

    assert reused == {
        "method": ({"psnr": 42.0, "n_images": 1}, [quality_row])
    }


def test_eval_lock_invalidates_when_evaluator_code_changes(tmp_path, monkeypatch):
    config = JobConfig(
        kind="eval",
        experiment="test",
        seed=0,
        save_path=str(tmp_path),
    )
    (tmp_path / "metrics.json").write_text("{}")
    write_lock(config)

    assert is_current(config)
    monkeypatch.setattr(configs, "_eval_code_digest", lambda _: "changed")
    assert not is_current(config)
    assert is_current(config, include_runtime=False)



def test_extra_stream_is_shared_across_own_sims():
    from afrbench.evaluator.metrics.ranking import extra_rank_states

    queries = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    extras = [(
        torch.nn.functional.normalize(torch.tensor([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.5, 0.5],
        ]), dim=1),
        ["noise/a", "noise/b", "noise/c"],
        ["ka", "kb", "kc"],
    )]
    gallery_a = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.2, 0.8]]), dim=1)
    gallery_b = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0], [0.8, 0.2]]), dim=1)
    labels_q = ["u", "v"]
    labels_a = ["u", "v"]
    labels_b = ["u", "v"]
    first = compute_ranking(
        queries, gallery_a, labels_q, labels_a, top_k=2, extra_blocks=list(extras)
    )
    second = compute_ranking(
        queries, gallery_b, labels_q, labels_b, top_k=2, extra_blocks=list(extras)
    )
    from afrbench.evaluator.metrics.ranking import _prepare_primary

    owns = [
        _prepare_primary(queries, gallery_a, labels_q, labels_a, top_k=2, device=torch.device("cpu")).own_sims,
        _prepare_primary(queries, gallery_b, labels_q, labels_b, top_k=2, device=torch.device("cpu")).own_sims,
    ]
    states = extra_rank_states(queries, extras, owns, top_k=2)
    shared_a = compute_ranking(
        queries, gallery_a, labels_q, labels_a, top_k=2, extra_state=states[0]
    )
    shared_b = compute_ranking(
        queries, gallery_b, labels_q, labels_b, top_k=2, extra_state=states[1]
    )
    assert shared_a.ranks[0].tolist() == first.ranks[0].tolist()
    assert shared_b.ranks[0].tolist() == second.ranks[0].tolist()
    assert shared_a.top_keys == first.top_keys
    assert shared_b.top_labels == second.top_labels
    assert shared_a.gallery_size == first.gallery_size


def test_add_greater_from_block_matches_loop():
    from afrbench.evaluator.metrics.ranking import _add_greater_from_block, _count_greater

    generator = torch.Generator().manual_seed(3)
    sims = torch.randn(12, 5000, generator=generator)
    owns = [
        torch.randn(12, 5, generator=generator),
        torch.randn(12, 3, generator=generator),
    ]
    got = [torch.zeros(own.shape, dtype=torch.long) for own in owns]
    ref = [torch.zeros(own.shape, dtype=torch.long) for own in owns]
    _add_greater_from_block(sims, owns, got)
    for own, greater in zip(owns, ref):
        _count_greater(sims, own, greater)
    for left, right in zip(got, ref):
        assert torch.equal(left, right)


def test_keep_mask_extra_block_matches_compact_block():
    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([_unit_vector(0.1)])
    embeddings = torch.tensor([
        _unit_vector(0.95),
        _unit_vector(0.4),
        _unit_vector(0.2),
    ])
    compact = [(
        embeddings[[0, 2]],
        torch.tensor([0, 2], dtype=torch.int64),
        torch.tensor([0, 2], dtype=torch.int64),
    )]
    masked = [(
        embeddings,
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([True, False, True]),
    )]

    class _Stream:
        def __init__(self, blocks):
            self._blocks = blocks

        def __iter__(self):
            return iter(self._blocks)

        def finish_topk(self, codes, index):
            labels = []
            keys = []
            for row_c, row_i in zip(codes.tolist(), index.tolist()):
                labels.append([f"noise/{c}" for c, i in zip(row_c, row_i) if i >= 0])
                keys.append([f"k{i}" for i in row_i if i >= 0])
            return labels, keys

    compact_rank = compute_ranking(
        queries, gallery, ["user"], ["user"], top_k=2, extra_blocks=_Stream(compact)
    )
    masked_rank = compute_ranking(
        queries, gallery, ["user"], ["user"], top_k=2, extra_blocks=_Stream(masked)
    )
    assert masked_rank.ranks[0].tolist() == compact_rank.ranks[0].tolist()
    assert masked_rank.gallery_size == compact_rank.gallery_size
    assert masked_rank.top_keys == compact_rank.top_keys


def test_integer_extra_blocks_match_string_extra_blocks():
    """Integer code/index extra blocks rank the same as string-labelled extra blocks."""
    from afrbench.evaluator.gallery import NoiseRankStream

    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([_unit_vector(0.1)])
    embeddings = torch.tensor([_unit_vector(0.9), _unit_vector(0.2)])
    string_blocks = [(
        embeddings,
        ["noise/245237", "noise/other"],
        ["245237__0.jpg", "other__0.jpg"],
    )]
    integer_blocks = [(
        embeddings,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
    )]

    class _TinyStream(NoiseRankStream):
        def __init__(self, blocks):
            self._blocks = blocks
            self.reader = type("R", (), {
                "vocab": ["245237", "other"],
                "lookup_keys": staticmethod(lambda idx: [
                    ["245237__0.jpg", "other__0.jpg"][i] for i in idx
                ]),
            })()
            self.excluded = []
            self.chunk = 2

        def __iter__(self):
            return iter(self._blocks)

    string_rank = compute_ranking(
        queries, gallery, ["user"], ["user"], top_k=2, extra_blocks=string_blocks
    )
    integer_rank = compute_ranking(
        queries, gallery, ["user"], ["user"], top_k=2,
        extra_blocks=_TinyStream(integer_blocks),
    )
    assert integer_rank.ranks[0].tolist() == string_rank.ranks[0].tolist()
    assert integer_rank.top_labels == string_rank.top_labels
    assert all(key is None for key in integer_rank.top_keys[0])
    assert torch.allclose(integer_rank.top_sims, string_rank.top_sims)


def test_compute_ranking_streams_extra_blocks_and_keeps_keys():
    queries = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([_unit_vector(0.1)])
    extras = (
        (
            torch.tensor([_unit_vector(0.9), _unit_vector(0.2)]),
            ["noise/245237", "noise/other"],
            ["245237__0.jpg", "other__0.jpg"],
        )
        for _ in range(1)
    )
    ranking = compute_ranking(
        queries, gallery, ["user"], ["user"], top_k=2, extra_blocks=extras
    )
    assert ranking.gallery_size == 3
    assert ranking.top_keys[0][0] == "245237__0.jpg"
    assert ranking.top_labels[0][0] == "noise/245237"


def test_identification_ignores_foreign_top_labels():
    from types import SimpleNamespace

    from afrbench.evaluator.metrics.identification import per_query

    ranking = Ranking(
        ranks=[torch.tensor([3])],
        top_labels=[["noise/245237", "user"]],
        top_sims=torch.tensor([[0.9, 0.1]]),
        gallery_size=10,
        top_keys=[["245237__0.jpg", "user__1.jpg"]],
    )
    scenario = SimpleNamespace(query_labels=["user"], n_query_failed=0)
    values = per_query(ranking, scenario, ks=(1, 5))
    assert all(not name.startswith("target_") for name in values)
    assert values["rank1_appearance"] == [0.0]
    assert values["rank5_appearance"] == [1.0]


def test_embed_eval_sources_does_path_a_before_noise(monkeypatch):
    order = []

    def fake_embed(source, *args, **kwargs):
        order.append(source)

    def fake_noise(**kwargs):
        order.append("millioncelebs")

    monkeypatch.setattr("afrbench.evaluator.job.embed_source", fake_embed)
    monkeypatch.setattr(
        "afrbench.evaluator.noise_cache.build_full_noise_cache", fake_noise
    )
    embed_eval_sources(
        object(),
        object(),
        object(),
        "cpu",
        eval_lists={},
        loaded={"a": object(), "b": object()},
        run_sources={"a": "protected__a@aaa", "b": "protected__b@bbb"},
        full_noise=True,
        distractors=-1,
        noise_seed=0,
        exclude_identities=None,
        batch_size=32,
    )
    assert order == [
        "originals",
        "targets",
        "protected__a@aaa",
        "protected__b@bbb",
        "millioncelebs",
    ]
