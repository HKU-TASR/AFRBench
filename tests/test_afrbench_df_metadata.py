"""The AFRBench-DF metadata the repository ships: its shape, its labels, and its agreement with AFRBench-basic."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from afrbench import deeperforensics as df
from afrbench import protocol as P
from afrbench.dataset import BenchDataset

ROOT = Path(__file__).resolve().parents[1] / "data"
DF, BASIC = ROOT / "afrbench-df", ROOT / "afrbench-basic"


@pytest.fixture(scope="module")
def dataset():
    return BenchDataset(DF)


def test_shipped_metadata_has_the_documented_shape(dataset):
    records = [r for r in dataset.records if r.split == "protectee"]
    assert len(dataset.protectee_identities()) == 50 and len(records) == 5000
    assert collections.Counter(r.role for r in records) == {"album": 2500, "query": 1250, "gallery": 1250}
    for actor in dataset.protectee_identities():
        rows = dataset.images_of(actor)
        assert collections.Counter(r.role for r in rows) == {"album": 50, "query": 25, "gallery": 25}
        assert len({r.session for r in rows}) == 100                       # a frame from each of 100 clips
    assert collections.Counter(a[0] for a in dataset.protectee_identities()) == {"M": 25, "W": 25}
    assert {tuple(r.size) for r in records} == {(1920, 1080)}
    assert all(r.face_box and r.landmarks and r.original_box and r.valid_rect for r in records)
    assert all(r.det_score >= df.MIN_DET_SCORE for r in records)


def test_shipped_metadata_passes_the_metadata_checks(dataset):
    problems, counts = df.check_metadata(dataset)
    assert problems == []
    assert sorted(counts) == ["camera", "emotion"]
    assert sorted(counts["camera"]) == sorted(df.CAMERAS) and sorted(counts["emotion"]) == sorted(df.EMOTIONS)


def test_every_role_sees_every_camera_emotion_and_light_about_equally(dataset):
    for role, size in (("album", 2500), ("query", 1250), ("gallery", 1250)):
        rows = [r for r in dataset.records if r.split == "protectee" and r.role == role]
        for axis, values in (("camera", df.CAMERAS), ("emotion", df.EMOTIONS), ("light", df.LIGHTS)):
            counts = collections.Counter(
                r.labels[axis] if axis in r.labels else df.parse_clip(r.source["video"]).light for r in rows)
            assert set(counts) == set(values)
            assert max(counts.values()) - min(counts.values()) <= 0.1 * size / len(values) + 5, (role, axis)


def test_frames_are_taken_from_the_middle_half_of_clips_and_never_from_blendshapes(dataset):
    for record in (r for r in dataset.records if r.split == "protectee"):
        assert df.parse_clip(record.source["video"]).kind == "expression"
        assert record.source["dataset"] == df.SOURCE_NAME and record.source["frame"] >= 2


def test_groups_give_every_hq_target_five_men_and_five_women(dataset):
    groups = json.loads((DF / "protectee_groups.json").read_text())
    basic_targets = json.loads((BASIC / "protectee_groups.json").read_text())["targets"]
    assert groups["targets"] == basic_targets and sorted(dataset.hq_target_identities()) == sorted(basic_targets)
    for target, members in groups["groups"].items():
        assert len(members) == 10 and sum(m[0] == "M" for m in members) == 5, target
    assigned = P.protectee_targets(DF)
    assert sorted(assigned) == sorted(dataset.protectee_identities()) and set(assigned.values()) == set(basic_targets)


def test_actors_were_drawn_in_proportion_to_the_asian_marker_and_spread_over_the_groups(dataset):
    """The pool of 95 has 24 of 52 men and 12 of 43 women with the marker; 25 of each sex keep those shares."""
    actors = dataset.protectee_identities()
    men, women = [a for a in actors if a[0] == "M"], [a for a in actors if a[0] == "W"]
    assert sum(map(df.asian_marked, men)) == 12 and sum(map(df.asian_marked, women)) == 7
    groups = json.loads((DF / "protectee_groups.json").read_text())["groups"]
    assert sorted(sum(map(df.asian_marked, members)) for members in groups.values()) == [3, 4, 4, 4, 4]
    selection = json.loads((DF / "dataset.json").read_text())["selection"]["actors"]
    assert "Asian marker" in selection


def test_targets_are_basics_so_the_gallery_and_the_pool_can_be_shared(dataset):
    basic = BenchDataset(BASIC)
    mine = {r.path: (r.identity, r.face_box, r.landmarks) for r in dataset.records if r.split == "target"}
    theirs = {r.path: (r.identity, r.face_box, r.landmarks) for r in basic.records if r.split == "target"}
    assert mine == theirs and len(mine) == 5
    info = json.loads((DF / "dataset.json").read_text())
    assert info["name"] == "afrbench-df" and info["counts"]["protectee_images"] == 5000
    assert info["selection"]["min_det_score"] == df.MIN_DET_SCORE
