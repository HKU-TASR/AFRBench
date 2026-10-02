"""Scenario source membership and failure accounting."""
from __future__ import annotations

from tests.test_frozen_roles import _FakeStore, _set, _write_v4

import numpy as np

from afrbench.evaluator.gallery import build_scenario


def _ids():
    return (
        "protectees/A/c.jpg",
        "protectees/A/d.jpg",
        "protectees/B/g.jpg",
        "protectees/B/h.jpg",
    )


def _store(failed_protected=()):
    q_a, g_a, q_b, g_b = _ids()
    originals = _set(
        [q_a, g_a, q_b, g_b],
        [np.eye(4, dtype=np.float32)[i] for i in range(4)],
    )
    prot_vecs = [np.eye(4, dtype=np.float32)[(i + 1) % 4] for i in range(4)]
    protected = _set([q_a, g_a, q_b, g_b], prot_vecs, failed=failed_protected)
    noise = _set(
        ["T__0.jpg", "U__0.jpg"],
        [np.array([0.5, 0.5, 0.0, 0.0], np.float32), np.array([0.0, 0.0, 0.5, 0.5], np.float32)],
    )
    return _FakeStore({
        "originals": originals,
        "protected__m@abc": protected,
        "noise": noise,
    })


def _eval_lists():
    q_a, g_a, q_b, g_b = _ids()
    return {"A": [q_a, g_a], "B": [q_b, g_b]}


def _build(tmp_path, kind, **kwargs):
    ds = _write_v4(tmp_path)
    store = kwargs.pop("store", None) or _store()
    return build_scenario(
        kind,
        dataset=ds,
        store=store,
        system_digest="x",
        protected_source=kwargs.pop("protected_source", "protected__m@abc"),
        eval_lists=_eval_lists(),
        distractors=kwargs.pop("distractors", 0),
        **kwargs,
    )


def test_base_scenario_sources(tmp_path):
    q_a, g_a, q_b, g_b = _ids()
    cases = {
        "1A": (False, True),
        "1B": (False, False),
        "2A": (True, True),
        "2B": (True, False),
    }
    store = _store()
    orig = store.get("originals", "x")
    prot = store.get("protected__m@abc", "x")
    for kind, (query_prot, gallery_prot) in cases.items():
        built = _build(tmp_path, kind, store=store)
        q_set = prot if query_prot else orig
        g_set = prot if gallery_prot else orig
        for image_id, row in zip(built.query_ids, built.query_embeddings.numpy()):
            assert np.allclose(row, q_set.embeddings[q_set.row_index()[image_id]])
        for image_id, row in zip(built.gallery_ids, built.gallery_embeddings.numpy()):
            if image_id in (q_a, g_a, q_b, g_b):
                assert np.allclose(row, g_set.embeddings[g_set.row_index()[image_id]])


def test_mix_gallery_endpoints_match_2a_2b(tmp_path):
    store = _store()
    two_b = _build(tmp_path, "2B", store=store)
    two_a = _build(tmp_path, "2A", store=store)
    mix0 = _build(tmp_path, "mix-gallery", store=store, mix_r=0.0)
    mix1 = _build(tmp_path, "mix-gallery", store=store, mix_r=1.0)
    assert mix0.gallery_ids == two_b.gallery_ids
    assert np.allclose(mix0.gallery_embeddings.numpy(), two_b.gallery_embeddings.numpy())
    assert mix1.gallery_ids == two_a.gallery_ids
    assert np.allclose(mix1.gallery_embeddings.numpy(), two_a.gallery_embeddings.numpy())


def test_failed_protected_query_stays_in_denominator(tmp_path):
    q_a, g_a, q_b, g_b = _ids()
    store = _store(failed_protected=[q_a])
    built = _build(tmp_path, "2A", store=store)
    assert q_a in built.failed_query_ids
    assert built.n_query_failed == 1
    assert q_a in built.intended_query_ids
    assert q_a not in built.query_ids


def test_failed_gallery_drops_row_keeps_intended_count(tmp_path):
    q_a, g_a, q_b, g_b = _ids()
    store = _store(failed_protected=[g_a])
    built = _build(tmp_path, "2A", store=store)
    assert g_a not in built.gallery_ids
    assert built.n_gallery_failed == 1
    assert built.intended_relevant["A"] == 1


def test_exclude_identities_drops_target_distractors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "afrbench.dataset.BenchDataset.noise_view_id_for",
        lambda self, exclude_identities=None: "view",
    )
    q_a, g_a, q_b, g_b = _ids()
    originals = _set(
        [q_a, g_a, q_b, g_b],
        [np.eye(4, dtype=np.float32)[i] for i in range(4)],
    )
    noise = _set(
        ["U__0.jpg", "V__0.jpg"],
        [
            np.array([0.0, 0.0, 0.5, 0.5], np.float32),
            np.array([0.5, 0.5, 0.0, 0.0], np.float32),
        ],
    )
    store = _FakeStore({"originals": originals, "noise": noise})
    built = _build(
        tmp_path, "1B", store=store, distractors=2,
        exclude_identities=["T"], seed=0, protected_source=None,
    )
    assert all(not label.endswith("/T") for label in built.gallery_labels)
    assert "noise/U" in built.gallery_labels
    assert "noise/V" in built.gallery_labels


def test_baseline_uses_originals_on_both_sides(tmp_path):
    store = _store()
    built = _build(tmp_path, "2A", store=store, protected_source=None)
    orig = store.get("originals", "x")
    for image_id, row in zip(built.query_ids, built.query_embeddings.numpy()):
        assert np.allclose(row, orig.embeddings[orig.row_index()[image_id]])
    for image_id, row in zip(built.gallery_ids, built.gallery_embeddings.numpy()):
        assert np.allclose(row, orig.embeddings[orig.row_index()[image_id]])
