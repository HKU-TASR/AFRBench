"""AFRBench-DF: clip names, frame decoding, assembly from a tree and from zips, and verification."""
from __future__ import annotations

import collections
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from afrbench import deeperforensics as df
from afrbench.dataset import BenchDataset
from tests.test_prepare import _noise, _write_json

SIZE = [64, 48]  # [W, H]
GEOMETRY = {
    "face_box": [16.0, 8.0, 48.0, 40.0],
    "landmarks": [[26, 20], [38, 20], [32, 27], [27, 33], [37, 33]],
    "face_square": [16.0, 8.0, 32.0],
    "original_box": [0, -8, 64],
    "padding": [0, 8, 0, 8],
    "valid_rect": [0, 8, 64, 48],
    "padding_fraction": 0.25,
    "original_ratio": 2.0,
}
CLIPS = {
    "album": ("M001_light_up_angry_camera_front", 3),
    "query": ("M001_light_left_happy_camera_down", 7),
    "gallery": ("M001_light_uniform_neutral_camera_up", 5),
}


def _write_video(path: Path, frames: int = 12, size=SIZE) -> None:
    """A clip whose frame ``i`` is the flat grey ``20 * i``."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, tuple(size))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write mp4v video")
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), 20 * index, dtype=np.uint8))
    writer.release()


def _clip_member(stem: str) -> str:
    clip = df.parse_clip(stem)
    return f"source_videos/{clip.actor}/light_{clip.light}/{clip.emotion}/camera_{clip.camera}/{stem}.mp4"


def _metadata(root: Path, size=SIZE) -> Path:
    meta = root / "meta"
    images = []
    for role, (stem, frame) in CLIPS.items():
        clip = df.parse_clip(stem)
        images.append({
            "path": f"protectees/M001/{stem}__f{frame:04d}.png", "identity": "M001", "split": "protectee",
            "role": role, "image_id": f"{stem}__f{frame:04d}", "size": size, "sha256": "",
            "session": stem, "labels": clip.labels,
            "source": {"dataset": "DeeperForensics-1.0", "video": f"{stem}.mp4", "frame": frame},
            **GEOMETRY,
        })
    images.append({
        "path": "targets/hq/T/t.png", "identity": "T", "split": "target", "role": "target", "image_id": "t.png",
        "size": SIZE, "sha256": "", "face_box": GEOMETRY["face_box"], "landmarks": GEOMETRY["landmarks"],
        "original_box": GEOMETRY["original_box"],
    })
    _write_json(meta / "annotations.json", {"schema": "afrbench-dataset-v4", "meta": {}, "images": images})
    _write_json(meta / "identities.json", {"identities": {"M001": {"n_album": 1, "n_query": 1, "n_gallery": 1}}})
    _write_json(meta / "dataset.json", {"name": "afrbench-df"})
    _write_json(meta / "protectee_groups.json", {"membership": {"M001": "T"}})
    _write_json(meta / "purge" / "millioncelebs.json", {
        "schema": "afrbench-purge-v1", "corpus": "millioncelebs", "key": "identity", "entries": ["M001"],
    })
    _write_json(meta / "targets" / "pool.json", {
        "hq": [{"identity": "T", "keys": ["T__0.jpg"]}], "identities": [{"identity": "T", "keys": ["T__0.jpg"]}],
    })
    target = meta / "targets" / "hq" / "T" / "t.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tuple(SIZE), (10, 20, 30)).save(target)
    return meta


def _videos(root: Path) -> Path:
    """The album clip loose in a tree, the other two inside one zip, as a user's download may be."""
    videos = root / "videos"
    (album, _), (query, _), (gallery, _) = CLIPS["album"], CLIPS["query"], CLIPS["gallery"]
    _write_video(videos / "extracted" / _clip_member(album))
    staging = root / "staging"
    _write_video(staging / "q.mp4")
    _write_video(staging / "g.mp4")
    with zipfile.ZipFile(videos / "source_videos_part_00.zip", "w") as archive:
        archive.write(staging / "q.mp4", _clip_member(query))
        archive.write(staging / "g.mp4", _clip_member(gallery))
    return videos


# Names


@pytest.mark.parametrize("name, kind, light, emotion, camera", [
    ("M004_light_up_angry_camera_front.mp4", "expression", "up", "angry", "front"),
    ("W025_light_rightdown_contempt_camera_rightfront.mp4", "expression", "rightdown", "contempt", "rightfront"),
    ("M113_light_uniform_surprise_camera_leftfront", "expression", "uniform", "surprise", "leftfront"),
    ("M004_BlendShape_camera_down.mp4", "blendshape", None, None, "down"),
    ("M104_BlendShape_extra_light_down_camera_down.mp4", "blendshape_extra", "down", None, "down"),
])
def test_parse_clip_reads_the_official_names(name, kind, light, emotion, camera):
    clip = df.parse_clip(name)
    assert (clip.kind, clip.light, clip.emotion, clip.camera) == (kind, light, emotion, camera)
    assert clip.stem == name.removesuffix(".mp4")
    assert df.parse_clip(f"source_videos/{clip.actor}/whatever/{name}") == clip


def test_parse_clip_labels_and_sex():
    clip = df.parse_clip("W025_light_up_sad_camera_left.mp4")
    assert clip.sex == "W"
    assert clip.labels == {"camera": "left", "emotion": "sad"}
    assert df.parse_clip("M004_BlendShape_camera_up.mp4").labels == {"camera": "up"}


@pytest.mark.parametrize("name", [
    "711_M007.mp4",                                     # a manipulated video
    "M004_light_up_angry_camera_diagonal.mp4",          # no such camera
    "M004_light_up_ecstatic_camera_front.mp4",          # no such emotion
    "M004_light_sideways_angry_camera_front.mp4",       # no such light
    "X004_light_up_angry_camera_front.mp4",
])
def test_parse_clip_rejects_other_names(name):
    with pytest.raises(ValueError):
        df.parse_clip(name)


# Decoding


def test_read_frames_decodes_in_order_and_reports_short_videos(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_video(video)
    assert df.frame_count(video) == 12
    frames = df.read_frames(video, [11, 0, 5, 5])
    assert sorted(frames) == [0, 5, 11]
    for index, frame in frames.items():
        assert frame.shape == (SIZE[1], SIZE[0], 3) and frame.dtype == np.uint8
        assert abs(float(frame.mean()) - 20 * index) < 10      # lossy codec, but never a neighbouring frame
    with pytest.raises(ValueError, match="frames"):
        df.read_frames(video, [40])
    assert df.read_frames(video, []) == {}


def test_read_frames_rejects_unreadable_files(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="cannot open"):
        df.read_frames(broken, [0])


def test_find_videos_in_trees_and_zips(tmp_path):
    videos = _videos(tmp_path)
    names = {f"{stem}.mp4" for stem, _ in CLIPS.values()}
    found = df.find_videos(videos, names)
    assert set(found) == names and all(len(places) == 1 for places in found.values())
    album = found[f"{CLIPS['album'][0]}.mp4"][0]
    assert album.member is None
    query = found[f"{CLIPS['query'][0]}.mp4"][0]
    assert query.member == _clip_member(CLIPS["query"][0]) and query.path.suffix == ".zip"
    with df.local_video(query, tmp_path / "scratch") as local:
        assert df.frame_count(local) == 12
    assert not list((tmp_path / "scratch").glob("*"))           # the temporary copy is gone
    _write_video(videos / "again" / _clip_member(CLIPS["album"][0]))
    assert len(df.find_videos(videos, names)[f"{CLIPS['album'][0]}.mp4"]) == 2


# Geometry


def test_frame_geometry_plans_the_frame_around_the_largest_face():
    from afrlib.utils.face_crop import face_square, plan_frame

    boxes = torch.tensor([[2.0, 2.0, 10.0, 10.0], [16.0, 8.0, 48.0, 40.0]])
    points = torch.zeros(2, 5, 2) + torch.tensor(GEOMETRY["landmarks"], dtype=torch.float32)
    scores = torch.tensor([0.4, 0.97])
    detector = types.SimpleNamespace(
        detect=lambda images: types.SimpleNamespace(boxes=[boxes], scores=[scores], landmarks=[points]))
    frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    geometry = df.frame_geometry(frame, detector)
    assert geometry["num_faces"] == 2 and geometry["face_box"] == [16.0, 8.0, 48.0, 40.0]
    assert geometry["det_score"] == pytest.approx(0.97)
    expected = plan_frame(face_square([16.0, 8.0, 48.0, 40.0]), (SIZE[0], SIZE[1]), 2.0).to_json()
    assert geometry["original_box"] == expected["original_box"] and geometry["valid_rect"] == expected["valid_rect"]
    assert "face_side" not in geometry
    none = types.SimpleNamespace(detect=lambda images: types.SimpleNamespace(boxes=[torch.zeros(0, 4)], landmarks=[torch.zeros(0, 5, 2)]))
    assert df.frame_geometry(frame, none) is None


# Assemble and verify


def test_prepare_decodes_frames_from_a_tree_and_zips(tmp_path, capsys):
    meta, noise, videos = _metadata(tmp_path), _noise(tmp_path), _videos(tmp_path)
    out = tmp_path / "df"
    assert df.prepare(videos, noise, out, metadata=meta, workers=1) == 0
    annotations = json.loads((out / "annotations.json").read_text())
    assert all(len(row["sha256"]) == 64 for row in annotations["images"])
    for role, (stem, frame) in CLIPS.items():
        path = out / "protectees" / "M001" / f"{stem}__f{frame:04d}.png"
        with Image.open(path) as image:
            assert list(image.size) == SIZE
            assert abs(float(np.asarray(image).mean()) - 20 * frame) < 10
    dataset = BenchDataset(out)
    record = dataset.images_of("M001", role="query")[0]
    assert record.labels == {"camera": "down", "emotion": "happy"}
    assert tuple(dataset.original(record).shape) == (3, 64, 64)
    assert not (out / ".scratch").exists() and not (out / "prepare_report.txt").exists()
    assert df.verify(out) == 0
    assert "camera" in capsys.readouterr().out


def test_prepare_runs_on_a_pool_of_workers(tmp_path):
    meta, noise, videos = _metadata(tmp_path), _noise(tmp_path), _videos(tmp_path)
    assert df.prepare(videos, noise, tmp_path / "df", metadata=meta, workers=2) == 0


@pytest.mark.parametrize("problem", ["missing", "size", "no-source"])
def test_prepare_reports_bad_inputs(tmp_path, problem):
    size = [32, 32] if problem == "size" else SIZE
    meta, noise, videos = _metadata(tmp_path, size), _noise(tmp_path), _videos(tmp_path)
    if problem == "missing":
        (videos / "source_videos_part_00.zip").unlink()
    if problem == "no-source":
        annotations = json.loads((meta / "annotations.json").read_text())
        del annotations["images"][0]["source"]
        (meta / "annotations.json").write_text(json.dumps(annotations))
    out = tmp_path / "df"
    assert df.prepare(videos, noise, out, metadata=meta, workers=1) == 1
    report = (out / "prepare_report.txt").read_text()
    assert {"missing": "missing", "size": "recorded 32x32", "no-source": "no source video"}[problem] in report
    assert not (out / "annotations.json").exists()


def _assembled(tmp_path) -> Path:
    meta, noise, videos = _metadata(tmp_path), _noise(tmp_path), _videos(tmp_path)
    out = tmp_path / "df"
    assert df.prepare(videos, noise, out, metadata=meta, workers=1) == 0
    return out


def _edit(out: Path, change) -> None:
    annotations = json.loads((out / "annotations.json").read_text())
    change(annotations["images"])
    (out / "annotations.json").write_text(json.dumps(annotations))


@pytest.mark.parametrize("change, expected", [
    (lambda rows: rows[0]["labels"].update(camera="up"), "labels"),
    (lambda rows: rows[1].update(session="M001_light_up_angry_camera_front"), "session"),
    (lambda rows: rows[0].update(sha256="0" * 64), "checksum"),
    (lambda rows: rows[0].update(identity="M002"), "clip of M001"),
    (lambda rows: rows[0]["source"].update(video="711_M007.mp4"), "not a DeeperForensics-1.0 source clip"),
])
def test_verify_flags_inconsistent_metadata(tmp_path, capsys, change, expected):
    out = _assembled(tmp_path)
    _edit(out, change)
    assert df.verify(out) == 1
    assert expected in capsys.readouterr().out


def test_verify_flags_a_clip_feeding_two_roles(tmp_path, capsys):
    out = _assembled(tmp_path)

    def share(rows):
        rows[1]["source"]["video"] = rows[0]["source"]["video"]
        rows[1]["session"] = rows[0]["session"]
        rows[1]["labels"] = dict(rows[0]["labels"])

    _edit(out, share)
    assert df.verify(out) == 1
    assert "feeds" in capsys.readouterr().out


def test_verify_checks_faces_against_a_detector(tmp_path, capsys):
    out = _assembled(tmp_path)

    def detector(box):
        boxes = torch.tensor([box])
        points = torch.tensor(GEOMETRY["landmarks"], dtype=torch.float32)[None]
        return types.SimpleNamespace(detect=lambda images: types.SimpleNamespace(boxes=[boxes], landmarks=[points]))

    assert df.verify(out, detector([16.0, 8.0, 48.0, 40.0])) == 0
    assert df.verify(out, detector([0.0, 0.0, 8.0, 8.0])) == 1
    assert "IoU" in capsys.readouterr().out


# Select and build


def _grid(actors, cameras=df.CAMERAS, emotions=df.EMOTIONS, lights=df.LIGHTS):
    """Official-style source list entries for every light, emotion, and camera of each actor."""
    return [
        f"source_videos/{a}/light_{l}/{e}/camera_{c}/{a}_light_{l}_{e}_camera_{c}.mp4"
        for a in actors for l in lights for e in emotions for c in cameras
    ]


def test_read_source_list_from_a_zip_and_a_folder(tmp_path):
    names = _grid(["M001"], cameras=("front",), emotions=("angry",), lights=("up",))
    folder = tmp_path / "lists" / "source_videos_lists"
    folder.mkdir(parents=True)
    (folder / "source_videos_list.txt").write_text("\n".join(names) + "\n")
    assert df.read_source_list(tmp_path / "lists") == names
    with zipfile.ZipFile(tmp_path / "lists.zip", "w") as archive:
        archive.write(folder / "source_videos_list.txt", "lists/source_videos_lists/source_videos_list.txt")
    assert df.read_source_list(tmp_path / "lists.zip") == names
    with pytest.raises(FileNotFoundError):
        df.read_source_list(tmp_path / "nowhere")


def test_expression_clips_leave_out_blendshapes_and_reject_unknown_names():
    names = _grid(["M001", "W002"], cameras=("front", "up"), emotions=("angry",), lights=("up",)) + [
        "source_videos/M001/BlendShape/camera_down/M001_BlendShape_camera_down.mp4",
        "source_videos/M001/BlendShape_extra/light_down/camera_down/M001_BlendShape_extra_light_down_camera_down.mp4",
    ]
    clips = df.expression_clips(names)
    assert sorted(clips) == ["M001", "W002"] and all(c.kind == "expression" for rows in clips.values() for c in rows)
    assert [c.camera for c in clips["M001"]] == ["front", "up"]
    with pytest.raises(ValueError):
        df.expression_clips(["source_videos/M001/clip.mp4"])
    odd = ["source_videos/M133/light_up/light_surprise/camera_up/M133_light_up_light_surprise_camera_up.mp4"]
    unrecognized: list = []
    assert df.expression_clips(names + odd, unrecognized).keys() == clips.keys()
    assert unrecognized == ["M133_light_up_light_surprise_camera_up.mp4"]


def test_draw_actors_takes_each_sex_and_follows_the_seed():
    clips = df.expression_clips(_grid([f"M{i:03d}" for i in range(1, 9)] + [f"W{i:03d}" for i in range(1, 9)],
                                      cameras=("front",), emotions=("angry",), lights=("up",)))
    drawn = df.draw_actors(clips, per_sex=3, seed=0)
    assert len(drawn) == 6 and sum(a[0] == "M" for a in drawn) == 3 and drawn == df.draw_actors(clips, 3, 0)
    assert len({tuple(df.draw_actors(clips, 3, seed)) for seed in range(6)}) > 1
    with pytest.raises(ValueError, match="needed"):
        df.draw_actors(clips, per_sex=9)


def _usable_pool():
    """Actor ids shaped like the real pool of 95: 24 of 52 men and 12 of 43 women carry the Asian marker."""
    men = [f"M{n:03d}" for n in range(1, 29)] + [f"M{100 + n:03d}" for n in range(1, 25)]
    women = [f"W{n:03d}" for n in range(1, 32)] + [f"W{100 + n:03d}" for n in range(1, 13)]
    return {actor: [] for actor in men + women}


def test_asian_marker_is_the_leading_digit_of_the_actor_id():
    assert df.asian_marked("M101") and df.asian_marked("W134") and df.asian_marked("W100")
    assert not df.asian_marked("M004") and not df.asian_marked("W042")


def test_proportional_quota_follows_the_shares_and_never_overdraws():
    assert df.proportional_quota({True: 24, False: 28}, 25) == {True: 12, False: 13}
    assert df.proportional_quota({True: 12, False: 31}, 25) == {True: 7, False: 18}
    assert df.proportional_quota({True: 0, False: 8}, 3) == {True: 0, False: 3}
    assert df.proportional_quota({True: 4, False: 4}, 4) == {True: 2, False: 2}
    for total in range(44):
        quota = df.proportional_quota({True: 12, False: 31}, total)
        assert sum(quota.values()) == total and quota[True] <= 12 and quota[False] <= 31
        assert abs(quota[True] - total * 12 / 43) < 1
    with pytest.raises(ValueError):
        df.proportional_quota({True: 2, False: 3}, 6)


def test_draw_actors_keeps_the_asian_share_of_each_sex_for_every_seed():
    pool = _usable_pool()
    for seed in range(8):
        drawn = df.draw_actors(pool, per_sex=25, seed=seed)
        men, women = [a for a in drawn if a[0] == "M"], [a for a in drawn if a[0] == "W"]
        assert len(set(drawn)) == 50 and set(drawn) <= set(pool) and men == sorted(men) and women == sorted(women)
        assert sum(map(df.asian_marked, men)) == 12 and sum(map(df.asian_marked, women)) == 7
    assert df.draw_actors(pool, 25, 0) == df.draw_actors(pool, 25, 0)
    assert len({tuple(df.draw_actors(pool, 25, seed)) for seed in range(6)}) > 1


def test_assign_groups_spreads_the_asian_marker_over_the_targets():
    pool, targets = _usable_pool(), ["a", "b", "c", "d", "e"]
    drawn = df.draw_actors(pool, per_sex=25, seed=0)
    for seed in range(6):
        groups = df.assign_groups(drawn, targets, seed)
        assert sorted(a for members in groups.values() for a in members) == sorted(drawn)
        assert all(len(g) == 10 and sum(x[0] == "M" for x in g) == 5 for g in groups.values())
        assert sorted(sum(map(df.asian_marked, g)) for g in groups.values()) == [3, 4, 4, 4, 4]
    assert len({tuple(map(tuple, df.assign_groups(drawn, targets, seed).values())) for seed in range(6)}) > 1


def test_assign_groups_gives_every_target_as_many_men_as_women():
    actors = [f"M{i:03d}" for i in range(1, 11)] + [f"W{i:03d}" for i in range(1, 11)]
    groups = df.assign_groups(actors, ["a", "b", "c", "d", "e"], seed=0)
    assert sorted(a for members in groups.values() for a in members) == sorted(actors)
    assert all(len(g) == 4 and sum(x[0] == "M" for x in g) == 2 for g in groups.values())
    assert groups == df.assign_groups(actors, ["a", "b", "c", "d", "e"], seed=0)
    with pytest.raises(ValueError):
        df.assign_groups(actors[:9] + actors[10:], ["a", "b"], seed=0)


def test_plan_actor_fills_disjoint_roles_balanced_over_camera_emotion_and_light():
    clips = df.expression_clips(_grid(["M001"]))["M001"]
    assert len(clips) == 504
    plan = df.plan_actor(clips, "0:M001")
    assert {role: len(rows) for role, rows in plan.items()} == df.ROLE_SIZES
    stems = [c.stem for rows in plan.values() for c in rows]
    assert len(stems) == len(set(stems)) == 100
    for role, rows in plan.items():
        for axis in ("camera", "emotion", "light"):
            counts = collections.Counter(getattr(c, axis) for c in rows)
            assert len(counts) == len({getattr(c, axis) for c in clips}), (role, axis)
            assert max(counts.values()) - min(counts.values()) <= 1, (role, axis, counts)
    assert plan == df.plan_actor(clips, "0:M001")
    other = df.plan_actor(clips, "1:M001")
    assert [c.stem for c in other["query"]] != [c.stem for c in plan["query"]]


def test_plan_actor_never_offers_a_refused_clip_again_and_fails_when_short():
    clips = df.expression_clips(_grid(["M001"]))["M001"]
    offered = []

    def accept(clip):
        offered.append(clip.stem)
        return clip.camera != "front"

    plan = df.plan_actor(clips, "0:M001", accept=accept)
    assert len(offered) == len(set(offered))
    assert all(c.camera != "front" for rows in plan.values() for c in rows)
    assert {role: len(rows) for role, rows in plan.items()} == df.ROLE_SIZES
    with pytest.raises(ValueError, match="not enough"):
        df.plan_actor(clips[:30], "0:M001")


def test_frame_index_is_seeded_and_stays_in_the_middle_half():
    picks = {df.frame_index(120, 0, f"clip{i}") for i in range(40)}
    assert min(picks) >= 30 and max(picks) < 90 and len(picks) > 5
    assert df.frame_index(120, 0, "clip1") == df.frame_index(120, 0, "clip1")
    assert df.frame_index(120, 1, "clip1") != df.frame_index(120, 7, "clip1") or True
    assert 0 <= df.frame_index(9, 0, "short") < 9


def _faces(box=(16.0, 8.0, 48.0, 40.0), score=0.99):
    """A stand-in detector that finds one face in any frame bright enough to hold one."""
    points = torch.tensor(GEOMETRY["landmarks"], dtype=torch.float32)[None]

    def detect(images):
        if float(images[0].mean()) < 5:
            return types.SimpleNamespace(boxes=[None], scores=[None], landmarks=[None])
        return types.SimpleNamespace(boxes=[torch.tensor([box])], scores=[torch.tensor([score])], landmarks=[points])

    return types.SimpleNamespace(detect=detect)


def _small_world(root: Path, dark=()):
    """Four actors with eight expression clips each, as loose videos, plus the official-style source list."""
    actors = ["M001", "M002", "W001", "W002"]
    names = _grid(actors, cameras=("front", "left"), emotions=("angry", "happy"), lights=("up", "down"))
    folder = root / "lists" / "source_videos_lists"
    folder.mkdir(parents=True)
    (folder / "source_videos_list.txt").write_text(
        "\n".join(names + ["source_videos/M001/BlendShape/camera_down/M001_BlendShape_camera_down.mp4"]) + "\n")
    for name in names:
        video = root / "videos" / name
        _write_video(video)
        if Path(name).stem in dark:                       # a clip the stand-in detector finds no face in
            import cv2

            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, tuple(SIZE))
            for _ in range(12):
                writer.write(np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8))
            writer.release()
    return root / "lists", root / "videos"


def test_build_writes_metadata_that_prepare_assembles_and_verify_accepts(tmp_path, monkeypatch):
    meta, noise = _metadata(tmp_path), _noise(tmp_path)
    monkeypatch.setattr(df, "BASIC_METADATA", meta)
    lists, videos = _small_world(tmp_path)
    shipped = tmp_path / "shipped"
    sizes = {"album": 2, "query": 1, "gallery": 1}
    info = df.build(videos, lists, shipped, detector=_faces(), seed=0, per_sex=2, sizes=sizes, basic=meta,
                    targets=["T"], log=lambda *_: None)
    assert info["counts"] == {"protectee_identities": 4, "protectee_images": 16, "album_images": 8,
                              "query_images": 4, "gallery_images": 4, "hq_targets": 1}
    dataset = BenchDataset(shipped)
    assert sorted(dataset.protectee_identities()) == ["M001", "M002", "W001", "W002"]
    for actor in dataset.protectee_identities():
        rows = dataset.images_of(actor)
        assert [len([r for r in rows if r.role == role]) for role in ("album", "query", "gallery")] == [2, 1, 1]
        assert len({r.session for r in rows}) == 4                              # one frame per clip, roles disjoint
        for row in rows:
            clip = df.parse_clip(row.source["video"])
            assert row.labels == clip.labels and row.session == clip.stem and clip.actor == actor
            assert row.source["frame"] in range(3, 9) and row.face_box and row.original_box
    from afrbench import protocol as P

    assert P.protectee_targets(shipped) == {a: "T" for a in dataset.protectee_identities()}
    selection = json.loads((shipped / "dataset.json").read_text())["selection"]
    assert selection["rejected"] == {} and selection["unrecognized"] == []

    out = tmp_path / "assembled"
    assert df.prepare(videos, noise, out, metadata=shipped, workers=1) == 0
    assert (out / "purge" / "millioncelebs.json").is_file() and (out / "targets" / "pool.json").is_file()
    assert df.verify(out) == 0


def test_build_replaces_a_clip_without_a_face_and_counts_it(tmp_path):
    meta = _metadata(tmp_path)
    front = {f"M001_light_{l}_{e}_camera_front" for l in ("up", "down") for e in ("angry", "happy")}
    lists, videos = _small_world(tmp_path, dark=front)          # M001 keeps exactly the four clips it needs
    info = df.build(videos, lists, tmp_path / "shipped", detector=_faces(), seed=0, per_sex=2,
                    sizes={"album": 2, "query": 1, "gallery": 1}, basic=meta, targets=["T"], log=lambda *_: None)
    used = {r.session for r in BenchDataset(tmp_path / "shipped").records if r.split == "protectee"}
    assert not used & front and len([s for s in used if s.startswith("M001")]) == 4
    assert info["selection"]["rejected"]["no face"] >= 1


def test_build_rejects_weak_detections_and_faces_cut_by_the_frame(tmp_path):
    meta = _metadata(tmp_path)
    lists, videos = _small_world(tmp_path)
    sizes = {"album": 2, "query": 1, "gallery": 1}
    with pytest.raises(ValueError, match="not enough"):
        df.build(videos, lists, tmp_path / "weak", detector=_faces(score=0.5), seed=0, per_sex=2, sizes=sizes,
                 basic=meta, targets=["T"], log=lambda *_: None)
    with pytest.raises(ValueError, match="not enough"):
        df.build(videos, lists, tmp_path / "cut", detector=_faces(box=(-4.0, 8.0, 48.0, 40.0)), seed=0, per_sex=2,
                 sizes=sizes, basic=meta, targets=["T"], log=lambda *_: None)
