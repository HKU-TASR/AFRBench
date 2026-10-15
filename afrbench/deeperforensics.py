"""AFRBench-DF: the AFRBench protocol on frames of the DeeperForensics-1.0 source videos.

DeeperForensics-1.0 (Jiang et al., CVPR 2020) records 100 actors under 9 lighting directions, 8 emotions
and 7 cameras. AFRBench-DF makes each actor a protectee and each decoded frame a photo, and keeps the
camera and emotion of the clip as labels, so ``python -m afrbench.groups`` can report protection per
viewpoint and expression. As AFRBench-basic keeps roles apart by day, AFRBench-DF keeps them apart by clip:
no clip contributes frames to two roles of one actor.

The repository ships the metadata under ``data/afrbench-df``; the videos come from the DeeperForensics
authors under their terms and are not redistributed. ``prepare`` reads them from the extracted
``source_videos`` tree or straight from the downloaded ``source_videos_part_*.zip`` files::

    python -m afrbench.deeperforensics prepare --videos DIR --noise PACK --out DIR
    python -m afrbench.deeperforensics verify DIR

``build`` regenerates the shipped metadata from the official lists and the videos: the actors, their groups, the
clips and frames are all seeded, so it is how the metadata can be audited or extended.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import dataclasses
import json
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

METADATA = Path(__file__).resolve().parents[1] / "data" / "afrbench-df"
BASIC_METADATA = Path(__file__).resolve().parents[1] / "data" / "afrbench-basic"

CAMERAS = ("down", "front", "left", "leftfront", "right", "rightfront", "up")
EMOTIONS = ("angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")
LIGHTS = ("down", "left", "leftdown", "leftup", "right", "rightdown", "rightup", "uniform", "up")

_ACTOR = r"(?P<actor>[MW]\d{3})"
_EXPRESSION = re.compile(rf"^{_ACTOR}_light_(?P<light>[a-z]+)_(?P<emotion>[a-z]+)_camera_(?P<camera>[a-z]+)$")
_BLENDSHAPE = re.compile(rf"^{_ACTOR}_BlendShape_camera_(?P<camera>[a-z]+)$")
_BLENDSHAPE_EXTRA = re.compile(rf"^{_ACTOR}_BlendShape_extra_light_(?P<light>[a-z]+)_camera_(?P<camera>[a-z]+)$")


# Clip names


@dataclasses.dataclass(frozen=True)
class SourceClip:
    """One source recording, from its file name.

    Attributes:
        actor: The actor id, ``M`` or ``W`` and three digits.
        kind: ``expression`` (a light, an emotion, a camera), ``blendshape`` (53 mouth and face shapes
            from one camera) or ``blendshape_extra`` (blendshapes under a light, a few actors only).
        camera: One of :data:`CAMERAS`.
        light: One of :data:`LIGHTS`, or ``None`` for ``blendshape``.
        emotion: One of :data:`EMOTIONS`, for ``expression`` only.
    """

    actor: str
    kind: str
    camera: str
    light: Optional[str] = None
    emotion: Optional[str] = None

    @property
    def sex(self) -> str:
        """``M`` or ``W``, the first letter of the actor id."""
        return self.actor[0]

    @property
    def stem(self) -> str:
        """The clip's file name without its extension."""
        if self.kind == "expression":
            return f"{self.actor}_light_{self.light}_{self.emotion}_camera_{self.camera}"
        if self.kind == "blendshape":
            return f"{self.actor}_BlendShape_camera_{self.camera}"
        return f"{self.actor}_BlendShape_extra_light_{self.light}_camera_{self.camera}"

    @property
    def labels(self) -> Dict[str, str]:
        """The attributes AFRBench-DF reports results by: ``camera``, and ``emotion`` where the clip has one.
        The light and the actor's sex stay in the clip's name.
        """
        labels = {"camera": self.camera}
        if self.emotion:
            labels["emotion"] = self.emotion
        return labels


def parse_clip(name: "str | Path") -> SourceClip:
    """Parse a source clip's file name (or path, or stem).

    Raises:
        ValueError: If ``name`` is not the name of a DeeperForensics-1.0 source clip, or names a camera,
            light, or emotion the dataset does not have.
    """
    stem = Path(str(name)).name
    stem = stem[: -len(".mp4")] if stem.endswith(".mp4") else stem
    for pattern, kind in ((_EXPRESSION, "expression"), (_BLENDSHAPE, "blendshape"), (_BLENDSHAPE_EXTRA, "blendshape_extra")):
        match = pattern.match(stem)
        if match:
            fields = match.groupdict()
            clip = SourceClip(
                actor=fields["actor"], kind=kind, camera=fields["camera"],
                light=fields.get("light"), emotion=fields.get("emotion"),
            )
            for value, allowed, what in (
                (clip.camera, CAMERAS, "camera"), (clip.light, LIGHTS, "light"), (clip.emotion, EMOTIONS, "emotion"),
            ):
                if value is not None and value not in allowed:
                    raise ValueError(f"{name!r} names the {what} {value!r}; expected one of {allowed}")
            return clip
    raise ValueError(f"{name!r} is not a DeeperForensics-1.0 source clip name")


# Decoding


def read_frames(video: "str | Path", indices: Iterable[int]) -> Dict[int, np.ndarray]:
    """Decode frames of a video as RGB ``uint8`` ``[H, W, 3]`` arrays.

    The video is read once from its start, so frame ``i`` is the ``i``-th decoded frame, whatever the
    container's seek behavior.

    Args:
        video: The video file.
        indices: Zero-based frame numbers.

    Raises:
        ValueError: If OpenCV cannot open the file or the video has fewer frames than asked for.
    """
    import cv2

    wanted = sorted({int(i) for i in indices})
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open {video}; OpenCV needs a build with ffmpeg (pip install opencv-python)")
    frames: Dict[int, np.ndarray] = {}
    pending = iter(wanted)
    target: Optional[int] = next(pending)
    try:
        for index in range(wanted[-1] + 1):
            if not capture.grab():
                raise ValueError(f"{video} has {index} frames; frame {target} was asked for")
            if index == target:
                ok, bgr = capture.retrieve()
                if not ok:
                    raise ValueError(f"{video}: frame {index} cannot be decoded")
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                target = next(pending, None)
    finally:
        capture.release()
    return frames


def frame_count(video: "str | Path") -> int:
    """The number of frames a video declares."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


@dataclasses.dataclass(frozen=True)
class VideoFile:
    """Where a clip is: a file, or a member of a zip."""

    path: Path
    member: Optional[str] = None


def find_videos(root: "str | Path", names: Set[str]) -> Dict[str, List[VideoFile]]:
    """Locate clips by file name under ``root``: loose ``.mp4`` files and the members of ``.zip`` files.

    Returns:
        Every location of each requested name that was found; a name with two locations is ambiguous.
    """
    root = Path(root)
    found: Dict[str, List[VideoFile]] = {}
    if root.is_file():
        files = [root]
    else:
        files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".mp4", ".zip"))
    for path in files:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.endswith(".mp4") and Path(member).name in names:
                        found.setdefault(Path(member).name, []).append(VideoFile(path, member))
        elif path.name in names:
            found.setdefault(path.name, []).append(VideoFile(path))
    return found


@contextlib.contextmanager
def local_video(video: VideoFile, scratch: Path) -> Iterator[Path]:
    """The clip as a file on disk: itself, or a temporary copy of its zip member."""
    if video.member is None:
        yield video.path
        return
    scratch.mkdir(parents=True, exist_ok=True)
    copy = scratch / f"{os.getpid()}_{Path(video.member).name}"
    try:
        with zipfile.ZipFile(video.path) as archive, archive.open(video.member) as source, open(copy, "wb") as sink:
            shutil.copyfileobj(source, sink)
        yield copy
    finally:
        copy.unlink(missing_ok=True)


# Geometry


def frame_geometry(frame: np.ndarray, detector, ratio: float = 2.0) -> Optional[Dict[str, Any]]:
    """The annotation fields of one decoded frame: its face and the frame window around it.

    Args:
        frame: RGB ``uint8`` ``[H, W, 3]``.
        detector: Any FR model with ``detect`` (for AFRBench the RetinaFace-ResNet50 of the FR system).
        ratio: Window side over face side; 2.0 as in AFRBench-basic.

    Returns:
        ``face_box``, ``landmarks``, ``face_square``, the detector's ``det_score`` and the frame plan
        (``original_box`` and its padding), taking the largest face; ``None`` if the detector finds none.
    """
    import torch
    from afrlib.utils.face_crop import face_square, plan_frame

    height, width = int(frame.shape[0]), int(frame.shape[1])
    result = detector.detect([torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float()])
    boxes, points = result.boxes[0], result.landmarks[0]
    if boxes is None or len(boxes) == 0:
        return None
    sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = int(torch.argmax(sizes))
    box = [float(v) for v in boxes[best][:4]]
    square = face_square(box)
    plan = plan_frame(square, (width, height), ratio).to_json()
    plan.pop("face_side")
    scores = getattr(result, "scores", [None])[0]
    return {
        "num_faces": int(len(boxes)),
        **({"det_score": float(scores[best])} if scores is not None else {}),
        "face_box": box,
        "landmarks": [[float(x), float(y)] for x, y in points[best].tolist()],
        "face_square": [float(v) for v in square],
        **plan,
    }


# Select and build


SOURCE_NAME = "DeeperForensics-1.0"
ROLE_SIZES = {"album": 50, "query": 25, "gallery": 25}
#: The order roles are filled in: the two scored roles are small, so they choose first and see every condition.
FILL_ORDER = ("query", "gallery", "album")
ACTORS_PER_SEX = 25
MIN_DET_SCORE = 0.9


def read_source_list(lists: "str | Path") -> List[str]:
    """The official ``source_videos_list.txt``, from the ``lists`` folder or from ``lists.zip``."""
    lists = Path(lists)
    inner = "source_videos_lists/source_videos_list.txt"
    if lists.is_file() and lists.suffix == ".zip":
        with zipfile.ZipFile(lists) as archive:
            return archive.read(f"lists/{inner}").decode().splitlines()
    for candidate in (lists / inner, lists / "lists" / inner):
        if candidate.is_file():
            return candidate.read_text().splitlines()
    raise FileNotFoundError(f"no {inner} under {lists}; pass the lists folder or lists.zip of DeeperForensics-1.0")


def expression_clips(names: Iterable[str], unrecognized: Optional[List[str]] = None) -> Dict[str, List[SourceClip]]:
    """The expression clips of every actor in a source list, by actor and sorted by name.

    Blendshape clips are left out: they are 53 exaggerated mouth and face shapes, mostly under uniform light,
    and not the emotions AFRBench-DF reports by.

    Args:
        names: Source clip names or paths.
        unrecognized: If given, names that are not DeeperForensics-1.0 source names are appended to it and
            skipped; otherwise they raise ``ValueError``. (The official list has seven, an extra take of
            actor M133 in a mistyped ``light_surprise`` folder.)
    """
    clips: Dict[str, List[SourceClip]] = {}
    for name in names:
        try:
            clip = parse_clip(name)
        except ValueError:
            if unrecognized is None:
                raise
            unrecognized.append(Path(name).name)
            continue
        if clip.kind == "expression":
            clips.setdefault(clip.actor, []).append(clip)
    return {actor: sorted(rows, key=lambda c: c.stem) for actor, rows in sorted(clips.items())}


def asian_marked(actor: str) -> bool:
    """Whether an actor id (``M101``, ``W025``) carries DeeperForensics' Asian marker, a leading 1 after the sex letter."""
    return actor[1] == "1"


def proportional_quota(sizes: Dict[Any, int], total: int) -> Dict[Any, int]:
    """Split ``total`` over strata in proportion to their sizes: the largest remainders take the leftover places.

    A tie goes to the larger stratum, so the split is deterministic. No stratum is asked for more than it holds.
    """
    population = sum(sizes.values())
    if not 0 <= total <= population:
        raise ValueError(f"cannot take {total} out of {population}")
    exact = {key: total * size / population for key, size in sizes.items()}
    quota = {key: int(value) for key, value in exact.items()}
    ranked = sorted(sizes, key=lambda key: (quota[key] - exact[key], -sizes[key], str(key)))
    for key in ranked[: total - sum(quota.values())]:
        quota[key] += 1
    return quota


def draw_actors(clips_by_actor: Dict[str, List[SourceClip]], per_sex: int = ACTORS_PER_SEX, seed: int = 0) -> List[str]:
    """``per_sex`` actors of each sex, drawn at random from those with expression clips.

    Within a sex the draw is stratified by the Asian marker (:func:`asian_marked`): the marked and the unmarked actors
    are drawn in proportion to their share of the pool, so the draw cannot over- or under-represent them by chance.
    """
    rng = random.Random(f"df-actors:{seed}")
    chosen: List[str] = []
    for sex in ("M", "W"):
        usable = sorted(actor for actor in clips_by_actor if actor[0] == sex)
        if len(usable) < per_sex:
            raise ValueError(f"{len(usable)} actors of sex {sex} have expression clips, {per_sex} are needed")
        strata = {marked: [a for a in usable if asian_marked(a) == marked] for marked in (True, False)}
        quota = proportional_quota({marked: len(members) for marked, members in strata.items()}, per_sex)
        picked: List[str] = []
        for marked in (True, False):
            picked += rng.sample(strata[marked], quota[marked])
        chosen += sorted(picked)
    return chosen


def assign_groups(actors: Sequence[str], targets: Sequence[str], seed: int = 0) -> Dict[str, List[str]]:
    """Deal the actors to the HQ targets, as many men as women to each, as AFRBench-basic does.

    The actors with the Asian marker are dealt round-robin across the targets (men first, women carrying on where the
    men stopped) and the others fill the places left, so no group is left without them and the counts differ by one
    at most.
    """
    men, women = sorted(a for a in actors if a[0] == "M"), sorted(a for a in actors if a[0] == "W")
    if len(men) != len(women) or len(men) % len(targets):
        raise ValueError(f"{len(men)} men and {len(women)} women do not split over {len(targets)} targets")
    rng = random.Random(f"df-protectee-groups:{seed}")
    step = len(men) // len(targets)
    groups: Dict[str, List[str]] = {target: [] for target in targets}
    turn = 0
    for members in (men, women):
        marked = [a for a in members if asian_marked(a)]
        others = [a for a in members if not asian_marked(a)]
        rng.shuffle(marked)
        rng.shuffle(others)
        room = {target: step for target in targets}
        for actor in marked:
            while room[targets[turn % len(targets)]] == 0:
                turn += 1
            target = targets[turn % len(targets)]
            groups[target].append(actor)
            room[target] -= 1
            turn += 1
        for actor in others:
            target = next(t for t in targets if room[t] > 0)
            groups[target].append(actor)
            room[target] -= 1
    return {target: sorted(members) for target, members in groups.items()}


def plan_actor(
    clips: Sequence[SourceClip],
    seed: str,
    sizes: Optional[Dict[str, int]] = None,
    accept=None,
) -> Dict[str, List[SourceClip]]:
    """Choose one actor's clips for each role: disjoint, and balanced within a role over camera, emotion, and light.

    A role is filled clip by clip. Each time the clip taken is one whose camera, emotion, and light are the least
    represented in that role so far (counted against the number of values each label has), and ties go to a seeded
    draw. So every role sees every camera, emotion, and light about equally often, whatever the actor's clips.

    Args:
        clips: The actor's expression clips.
        seed: Seeds the tie-breaks.
        sizes: Clips per role (default :data:`ROLE_SIZES`).
        accept: Optional ``accept(clip) -> bool`` called on each candidate; a clip it refuses (say, no usable face)
            is never offered again and the next best one is taken instead.

    Raises:
        ValueError: If the actor has too few clips (or too few accepted ones) for the roles.
    """
    sizes = dict(sizes or ROLE_SIZES)
    rng = random.Random(f"df-frames:{seed}")
    weights = {"camera": len(CAMERAS), "emotion": len(EMOTIONS), "light": len(LIGHTS)}
    left = sorted(clips, key=lambda c: c.stem)
    plan: Dict[str, List[SourceClip]] = {}
    for role in FILL_ORDER:
        seen = {axis: collections.Counter() for axis in weights}
        chosen: List[SourceClip] = []
        while len(chosen) < sizes[role]:
            if not left:
                raise ValueError(f"not enough usable clips for the {role} role ({len(chosen)} of {sizes[role]})")
            scores = [sum(seen[axis][getattr(clip, axis)] * weights[axis] for axis in weights) for clip in left]
            best = min(scores)
            pick = left.pop(rng.choice([i for i, score in enumerate(scores) if score == best]))
            if accept is not None and not accept(pick):
                continue
            chosen.append(pick)
            for axis in weights:
                seen[axis][getattr(pick, axis)] += 1
        plan[role] = chosen
    return plan


def frame_index(frames: int, seed: int, stem: str) -> int:
    """A seeded frame from the middle half of a clip, which starts and ends with the actor settling."""
    low, high = frames // 4, (3 * frames) // 4
    return low + random.Random(f"df-offset:{seed}:{stem}").randrange(max(1, high - low))


def frame_record(clip: SourceClip, role: str, index: int, size: Sequence[int], geometry: Dict[str, Any]) -> Dict[str, Any]:
    """The annotation of one chosen frame, in the layout ``prepare`` and ``BenchDataset`` read."""
    image_id = f"{clip.stem}__f{index:04d}"
    return {
        "path": f"protectees/{clip.actor}/{image_id}.png", "identity": clip.actor, "split": "protectee",
        "role": role, "image_id": image_id, "size": [int(size[0]), int(size[1])], **geometry,
        "session": clip.stem, "labels": clip.labels,
        "source": {"dataset": SOURCE_NAME, "video": f"{clip.stem}.mp4", "frame": int(index)}, "sha256": "",
    }


def build(
    videos: Path,
    lists: Path,
    out: Path,
    *,
    detector,
    seed: int = 0,
    per_sex: int = ACTORS_PER_SEX,
    sizes: Optional[Dict[str, int]] = None,
    basic: Path = BASIC_METADATA,
    targets: Optional[Sequence[str]] = None,
    scratch: Optional[Path] = None,
    log=print,
) -> Dict[str, Any]:
    """Build AFRBench-DF's metadata from the official lists and a copy of the source videos.

    The actors, their groups, the clips of each role, and a frame of each clip are all seeded. A clip is used only
    if its chosen frame decodes and the detector finds a face there, with a score of at least
    :data:`MIN_DET_SCORE` and the whole box inside the frame; otherwise the next best clip of the role is taken.
    The five HQ targets come from AFRBench-basic. Writes ``annotations.json``, ``identities.json``,
    ``dataset.json``, and ``protectee_groups.json`` into ``out``.

    Returns:
        The ``dataset.json`` content.
    """
    import torch

    sizes = dict(sizes or ROLE_SIZES)
    basic_annotations = json.loads((Path(basic) / "annotations.json").read_text())
    target_records = [r for r in basic_annotations["images"] if r["split"] == "target"]
    targets = list(targets or sorted({r["identity"] for r in target_records}))
    unrecognized: List[str] = []
    clips = expression_clips(read_source_list(lists), unrecognized)
    if unrecognized:
        log(f"skipping {len(unrecognized)} names the official list has but that are not source clip names: "
            f"{sorted(unrecognized)[:3]}...")
    actors = draw_actors(clips, per_sex, seed)
    groups = assign_groups(actors, targets, seed)
    found = find_videos(videos, {f"{c.stem}.mp4" for actor in actors for c in clips[actor]})
    scratch = Path(scratch or Path(out) / ".scratch")
    rejected: collections.Counter = collections.Counter()
    records: List[Dict[str, Any]] = []

    for number, actor in enumerate(actors, 1):
        chosen: Dict[str, Any] = {}

        def accept(clip: SourceClip) -> bool:
            places = found.get(f"{clip.stem}.mp4", [])
            if len(places) != 1:
                rejected["not found" if not places else "ambiguous"] += 1
                return False
            try:
                with local_video(places[0], scratch) as local:
                    count = frame_count(local)
                    if count < 8:
                        rejected["too short"] += 1
                        return False
                    index = frame_index(count, seed, clip.stem)
                    frame = read_frames(local, [index])[index]
            except (ValueError, OSError, zipfile.BadZipFile):
                rejected["undecodable"] += 1
                return False
            with torch.no_grad():
                geometry = frame_geometry(frame, detector)
            height, width = frame.shape[:2]
            if geometry is None:
                rejected["no face"] += 1
            elif geometry.get("det_score", 1.0) < MIN_DET_SCORE:
                rejected["weak detection"] += 1
            elif not (0 <= geometry["face_box"][0] and 0 <= geometry["face_box"][1]
                      and geometry["face_box"][2] <= width and geometry["face_box"][3] <= height):
                rejected["face cut by the frame"] += 1
            else:
                chosen[clip.stem] = (index, [width, height], geometry)
                return True
            return False

        plan = plan_actor(clips[actor], f"{seed}:{actor}", sizes, accept)
        for role in ("album", "query", "gallery"):
            for clip in plan[role]:
                index, size, geometry = chosen[clip.stem]
                records.append(frame_record(clip, role, index, size, geometry))
        log(f"[{number}/{len(actors)}] {actor}: {sum(len(v) for v in plan.values())} frames, "
            f"rejected so far {dict(rejected)}")
    shutil.rmtree(scratch, ignore_errors=True)

    counts = collections.Counter((r["identity"], r["role"]) for r in records)
    info = {
        "schema": "afrbench-dataset-v4", "name": "afrbench-df",
        "counts": {
            "protectee_identities": len(actors), "protectee_images": len(records),
            **{f"{role}_images": sum(v for (_, r), v in counts.items() if r == role) for role in ("album", "query", "gallery")},
            "hq_targets": len(targets),
        },
        "seeds": {"selection": seed, "allocate": seed, "targets": seed},
        "selection": {
            "source": SOURCE_NAME, "clips": "expression clips, one frame each, disjoint by role",
            "actors": (
                f"{per_sex} men and {per_sex} women drawn at random from the actors with expression clips, "
                "in proportion to the Asian marker (a leading 1 in the actor id) within each sex"
            ),
            "balance": "within a role over camera, emotion, and light", "frame": "seeded, middle half of the clip",
            "min_det_score": MIN_DET_SCORE, "rejected": dict(rejected), "unrecognized": sorted(unrecognized),
        },
        "protectees": actors,
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    role_rank = {"album": 0, "query": 1, "gallery": 2}
    ordered = sorted(records, key=lambda r: (r["identity"], role_rank[r["role"]], r["path"]))
    (out / "annotations.json").write_text(json.dumps({
        "schema": "afrbench-dataset-v4", "name": "afrbench-df",
        "meta": {"roles": ["album", "query", "gallery"], "original_ratio": 2.0},
        "images": ordered + target_records,
    }, indent=1) + "\n")
    (out / "identities.json").write_text(json.dumps({
        "schema": "afrbench-identities-v4",
        "identities": {a: {"role": "protectee", "n_images": sum(counts[(a, r)] for r in role_rank),
                           **{f"n_{r}": counts[(a, r)] for r in role_rank}} for a in actors},
    }, indent=1) + "\n")
    (out / "dataset.json").write_text(json.dumps(info, indent=1) + "\n")
    (out / "protectee_groups.json").write_text(json.dumps({
        "schema": "afrbench-df-protectee-groups-v1", "seed_key": f"df-protectee-groups:{seed}", "targets": targets,
        "per_target": len(actors) // len(targets), "per_stratum": per_sex // len(targets), "groups": groups,
        "membership": {actor: target for target, members in groups.items() for actor in members},
    }, indent=1) + "\n")
    return info


# Assemble


def _extract(job: Tuple[VideoFile, List[Tuple[int, str, List[int]]], str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Decode one video's frames into PNG files. Returns ``(destination, sha256, problem)`` per frame."""
    from afrbench.prepare import _sha256

    video, frames, scratch = job
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    try:
        with local_video(video, Path(scratch)) as local:
            decoded = read_frames(local, [index for index, _, _ in frames])
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        return [(dest, None, f"{video.path.name}: {error}") for _, dest, _ in frames]
    for index, dest, size in frames:
        frame = decoded[index]
        if [frame.shape[1], frame.shape[0]] != list(size):
            out.append((dest, None, f"{video.path.name}: frame is {frame.shape[1]}x{frame.shape[0]}, recorded {size[0]}x{size[1]}"))
            continue
        target = Path(dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        Image.fromarray(frame).save(temporary, format="PNG", compress_level=3)
        os.replace(temporary, target)
        out.append((dest, _sha256(target), None))
    return out


def prepare(
    videos: Path, noise: Path, out: Path, metadata: Path = METADATA, workers: int = 4,
) -> int:
    """Assemble AFRBench-DF from the shipped metadata, the source videos, and a noise pack.

    Each protectee record names its clip and frame in ``source``; the frame is decoded into
    ``protectees/<actor>/`` as a PNG, never resized, and must have the recorded size. The purge lists and the HQ
    targets are AFRBench-basic's unless the metadata brings its own, so the distractor gallery, the target pool, and
    the embedding caches are shared with it. The checks and the other files are those of
    :func:`afrbench.prepare.prepare`.

    Returns:
        0 when the dataset is ready, 1 after writing ``prepare_report.txt``.
    """
    from afrbench.prepare import _link, _sha256, check_pack

    if not (metadata / "annotations.json").is_file():
        raise SystemExit(
            f"no AFRBench-DF metadata at {metadata}; run from a clone of the repository "
            "(pip install -e .) or pass --metadata <clone>/data/afrbench-df"
        )
    annotations = json.loads((metadata / "annotations.json").read_text())
    records = annotations["images"]
    protectees = [r for r in records if r["split"] == "protectee"]
    problems: List[str] = []
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for record in protectees:
        source = record.get("source") or {}
        if not source.get("video") or not isinstance(source.get("frame"), int):
            problems.append(f"no source video and frame in the record of {record['path']}")
            continue
        by_video.setdefault(source["video"], []).append(record)
    found = find_videos(videos, set(by_video))
    for name in sorted(by_video):
        if name not in found:
            problems.append(f"missing  {name}")
        elif len(found[name]) > 1:
            problems.append(f"ambiguous {name}: {[str(v.path) for v in found[name]]}")

    out.mkdir(parents=True, exist_ok=True)
    report = out / "prepare_report.txt"
    digests: Dict[str, str] = {}
    if not problems:
        scratch = str(out / ".scratch")
        jobs = [
            (found[name][0], [(r["source"]["frame"], str(out / r["path"]), r["size"]) for r in rows], scratch)
            for name, rows in sorted(by_video.items())
        ]
        with contextlib.ExitStack() as stack:
            if workers > 1:
                pool = stack.enter_context(concurrent.futures.ProcessPoolExecutor(max_workers=workers))
                finished = pool.map(_extract, jobs)
            else:
                finished = map(_extract, jobs)
            for results in finished:
                for dest, digest, problem in results:
                    if problem:
                        problems.append(problem)
                    else:
                        digests[dest] = digest
        shutil.rmtree(scratch, ignore_errors=True)
    if not (noise / "data.mdb").is_file():
        problems.append(f"no LMDB at {noise} (run: python -m afrbench.millioncelebs pack)")
    if problems:
        report.write_text("\n".join(problems) + "\n")
        print(f"{len(problems)} problems; first ones:", *problems[:10], f"full list: {report}", sep="\n  ")
        return 1

    for name in ("identities.json", "dataset.json", "protectee_groups.json"):
        shutil.copyfile(metadata / name, out / name)
    for sub in ("purge", "targets"):    # shared with AFRBench-basic unless this metadata brings its own
        source = metadata / sub if (metadata / sub).is_dir() else BASIC_METADATA / sub
        shutil.copytree(source, out / sub, dirs_exist_ok=True)
    for record in protectees:
        record["sha256"] = digests[str(out / record["path"])]
    for record in records:
        if record["split"] == "target":
            record["sha256"] = _sha256(out / record["path"])
    (out / "annotations.json").write_text(json.dumps(annotations, indent=1) + "\n")
    _link(noise, out / "noise.lmdb", copy=False)

    from afrbench.dataset import BenchDataset

    dataset = BenchDataset(out)
    problem = check_pack(dataset)
    if problem:
        report.write_text(problem + "\n")
        print(problem)
        return 1
    if report.exists():
        report.unlink()
    print(f"AFRBench-DF ready at {out}: {len(protectees)} protectee frames from {len(by_video)} clips of "
          f"{len(dataset.protectee_identities())} actors, {len(dataset.hq_target_identities())} HQ targets, "
          f"{len(dataset.target_identities())} pool targets")
    return 0


# Verify


def check_metadata(dataset) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Check what the annotations alone can show, with no frame on disk.

    Every protectee record's clip name must parse and agree with its actor and labels; its session must be its
    clip, so that no clip feeds two roles of an actor; and each actor's roles must hold the counts that
    ``identities.json`` records.

    Returns:
        The problems found, and the number of frames for each value of each label.
    """
    problems: List[str] = []
    counts: Dict[str, Dict[str, int]] = {}
    roles_of_clip: Dict[Tuple[str, str], Set[str]] = {}
    for record in (r for r in dataset.records if r.split == "protectee"):
        video = (record.source or {}).get("video")
        try:
            clip = parse_clip(video or "")
        except ValueError as error:
            problems.append(f"{record.path}: {error}")
            continue
        if clip.actor != record.identity:
            problems.append(f"{record.path}: clip of {clip.actor}, record of {record.identity}")
        if dict(record.labels) != clip.labels:
            problems.append(f"{record.path}: labels {dict(record.labels)} differ from the clip's {clip.labels}")
        if record.session != clip.stem:
            problems.append(f"{record.path}: session {record.session!r} is not the clip {clip.stem!r}")
        roles_of_clip.setdefault((record.identity, clip.stem), set()).add(record.role)
        for key, value in record.labels.items():
            counts.setdefault(key, {})
            counts[key][value] = counts[key].get(value, 0) + 1
    for (identity, stem), roles in sorted(roles_of_clip.items()):
        if len(roles) > 1:
            problems.append(f"{identity}: clip {stem} feeds {sorted(roles)}")
    for identity, expected in dataset.identities.items():
        for role in ("album", "query", "gallery"):
            n = len(dataset.images_of(identity, role=role))
            if expected.get(f"n_{role}") is not None and n != expected[f"n_{role}"]:
                problems.append(f"{identity}: {n} {role} frames, identities.json says {expected[f'n_{role}']}")
    return problems, counts


def verify(root: Path, detector=None, min_iou: float = 0.5) -> int:
    """Check an assembled AFRBench-DF folder and print its counts per label.

    The checks are those of :func:`check_metadata`, and in addition every protectee record has a file of the
    recorded size and checksum. With a ``detector``, each frame's face is also detected again and must overlap
    the recorded box (IoU of at least ``min_iou``).

    Returns:
        0 when everything holds, 1 otherwise.
    """
    from afrbench.dataset import BenchDataset
    from afrbench.prepare import _sha256

    dataset = BenchDataset(root)
    problems, counts = check_metadata(dataset)
    for record in (r for r in dataset.records if r.split == "protectee"):
        path = Path(root) / record.path
        if not path.is_file():
            problems.append(f"missing {record.path}")
            continue
        with Image.open(path) as image:
            if list(image.size) != list(record.size[:2]):
                problems.append(f"{record.path}: size {list(image.size)} != recorded {record.size[:2]}")
        if record.sha256 and _sha256(path) != record.sha256:
            problems.append(f"{record.path}: checksum differs from the recorded one")
    if detector is not None:
        problems.extend(_check_faces(dataset, detector, min_iou))
    for key in sorted(counts):
        print(f"{key:<8}" + "  ".join(f"{value} {n}" for value, n in sorted(counts[key].items())))
    if problems:
        print(f"{len(problems)} problems; first ones:", *problems[:10], sep="\n  ")
        return 1
    print(f"AFRBench-DF at {root} verified: {len(dataset.protectee_identities())} actors")
    return 0


def _check_faces(dataset, detector, min_iou: float) -> List[str]:
    """Frames whose fresh detection does not match the recorded face box."""
    problems = []
    for record in (r for r in dataset.records if r.split == "protectee" and r.face_box):
        fresh = frame_geometry(dataset.image(record).permute(1, 2, 0).byte().numpy(), detector)
        if fresh is None:
            problems.append(f"{record.path}: no face detected again")
            continue
        ax1, ay1, ax2, ay2 = record.face_box
        bx1, by1, bx2, by2 = fresh["face_box"]
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        if union <= 0 or inter / union < min_iou:
            problems.append(f"{record.path}: detected box overlaps the recorded one with IoU {inter / max(union, 1e-9):.2f}")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m afrbench.deeperforensics", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cli = commands.add_parser("prepare", help="decode the frames and assemble the dataset")
    prepare_cli.add_argument("--videos", required=True, type=Path,
                             help="folder with the extracted source_videos tree or the source_videos_part_*.zip files")
    prepare_cli.add_argument("--noise", required=True, type=Path, help="packed MillionCelebs LMDB folder")
    prepare_cli.add_argument("--out", required=True, type=Path, help="where to assemble the dataset")
    prepare_cli.add_argument("--metadata", type=Path, default=METADATA, help="shipped metadata (data/afrbench-df)")
    prepare_cli.add_argument("--workers", type=int, default=4, help="videos decoded in parallel")
    make = commands.add_parser("build", help="build the metadata from the official lists and the videos")
    make.add_argument("--videos", required=True, type=Path, help="as for prepare")
    make.add_argument("--lists", required=True, type=Path, help="the official lists folder or lists.zip")
    make.add_argument("--out", required=True, type=Path, help="where to write the metadata")
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--basic", type=Path, default=BASIC_METADATA, help="AFRBench-basic metadata, for the HQ targets")
    make.add_argument("--device", default="auto")
    check = commands.add_parser("verify", help="check an assembled dataset")
    check.add_argument("root", type=Path, help="the assembled dataset")
    check.add_argument("--faces", action="store_true", help="also detect every face again (needs the FR weights)")
    check.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare(args.videos, args.noise, args.out, args.metadata, args.workers)
    if args.command == "build":
        info = build(args.videos, args.lists, args.out, detector=_detector(args.device), seed=args.seed, basic=args.basic,
                     log=lambda message: print(message, flush=True))
        print(f"wrote the metadata of {info['counts']['protectee_images']} frames to {args.out}; "
              f"rejected clips: {info['selection']['rejected']}")
        return 0
    return verify(args.root, _detector(args.device) if args.faces else None)


def _detector(device: str):
    """The default FR system's model, which carries the RetinaFace detector the evaluation uses."""
    import torch
    from afrlib import build_fr_ensemble
    from afrbench import protocol as P

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return build_fr_ensemble([P.DEFAULT_FR], torch.device(device))[0]


if __name__ == "__main__":
    sys.exit(main())
