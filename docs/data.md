# AFRBench data

AFRBench has two datasets. The repository ships the metadata of each; the photos and
videos are obtained separately, under their providers' terms, and assembled with a
`prepare` command (see [benchmark.md](benchmark.md)). Both are scored against the same
distractor gallery, MillionCelebs, with the same protocol ([protocol.md](protocol.md)).

| Dataset | Content | Assemble with |
|---|---|---|
| AFRBench-basic | 50 people, 100 high-resolution photos each | `python -m afrbench.prepare` |
| AFRBench-DF | frames of the DeeperForensics-1.0 source videos, labeled by camera and emotion | `python -m afrbench.deeperforensics prepare` |

## AFRBench-basic

The repository ships its metadata under [`data/afrbench-basic/`](../data/afrbench-basic).

### What the repository ships

| File | Content |
|---|---|
| `annotations.json` | 5,000 protectee records and 5 target records: identity, role (`album` / `query` / `gallery`), Getty asset id, image size, face box, five landmarks, and the face frame |
| `identities.json`, `dataset.json` | per-identity role counts; dataset counts and seeds |
| `protectee_groups.json` | the HQ target assigned to each protectee |
| `targets/hq/` | the five HQ target images (CelebA-HQ) |
| `targets/pool.json` | the 2,000 pool identities and 20 MillionCelebs keys each |
| `purge/*.json` | identities or images removed from each training corpus and from MillionCelebs because they are protectees |

Face geometry is given in the pixel coordinates of the 2048-pixel versions of the
photos. The annotations contain no image checksums.

### What you obtain yourself

| Data | Source | Size |
|---|---|---|
| 5,000 protectee photos | Getty Images, by the `getty_asset_id` of each record (2048-pixel versions) | 3.2 GB |
| MillionCelebs | the authors' release, <https://buptzyb.github.io/MillionCelebs/>; `python -m afrbench.millioncelebs download` fetches it | 445 GB download, 179 GiB packed |
| Training corpora | downloaded by `afrlib.data.get_corpus` when a train job needs them, or your own copy via `--corpus-root` | up to ~70 GB |

The protectee photos are editorial photos from Getty Images and cannot be
redistributed. Obtain them under terms that permit your use. `prepare` requires each
file to have exactly the recorded size and no EXIF rotation, so that the recorded
geometry applies; it does not modify the images.

## AFRBench-DF

AFRBench-DF is the video and pose counterpart of AFRBench-basic. It draws its photos from
the source recordings of [DeeperForensics-1.0](https://github.com/EndlessSora/DeeperForensics-1.0)
(Jiang et al., CVPR 2020): 100 actors recorded at 1920x1080 under 9 lighting directions
(`down`, `left`, `leftdown`, `leftup`, `right`, `rightdown`, `rightup`, `uniform`, `up`),
8 emotions (`angry`, `contempt`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`),
and 7 cameras that cover poses from -90 to +90 degrees (`down`, `front`, `left`,
`leftfront`, `right`, `rightfront`, `up`).

- **Protectees and photos.** 50 actors, 25 men and 25 women, drawn at random (seed 0) from
  the 95 who have expression clips. DeeperForensics tells its 40 Asian actors by a leading
  `1` in the actor id (`W101`; the others have a `0`, as in `M004`), and the draw keeps that
  share within each sex: 36 of the 95 carry the marker (24 of 52 men, 12 of 43 women), and
  19 of the 50 do (12 men, 7 women). Each has 100 frames, taken from 100 different clips and
  one frame per clip: 50 album, 25 query, and 25 gallery frames, as in AFRBench-basic. A
  frame is a seeded draw from the middle half of its clip. Five groups of ten actors, five
  men and five women each, are assigned to the five HQ targets, and the marked actors are
  spread over the groups: each group has three or four.
- **Roles are disjoint by clip.** No clip contributes to two roles of one actor, and a
  record's `session` is its clip. Within a role the clips are balanced over camera, emotion,
  and light: every role sees every value about equally often. An actor was recorded in a
  single session, so the roles share clothes, hairstyle, and background; they differ in
  viewpoint, expression, and lighting.
- **Labels.** A frame keeps the camera and emotion of its clip, so results can be reported
  per viewpoint and expression (`python -m afrbench.groups`). The light stays in the clip's
  name, `source.video`.
- **Which frames qualify.** A clip is used only if its chosen frame decodes and RetinaFace
  finds a face there with a score of at least 0.9 and the whole box inside the frame;
  otherwise the next best clip of the role is taken. Blendshape clips, the five actors who
  have none of the emotion clips, and seven extra clips of actor M133 in a mistyped
  `light_surprise` folder of the official release are not used.
- **Distractors, targets, and corpora are AFRBench-basic's.** The gallery is the same
  MillionCelebs pack, the HQ targets and the target pool are the same, and so are the purge
  lists. The actors are paid volunteers rather than public figures, and none is purged: the
  nearest of the 22.8M distractors to any of the 5,000 frames has a cosine of at most 0.66
  under the default FR model (exact duplicates of photos in the pack score 0.87 to 0.96),
  and by eye the closest ones are other people. Put the assembled dataset next to
  AFRBench-basic's: the two then share the distractor embedding cache, which is keyed by the
  pack and the FR model, not by the dataset.

The repository ships the metadata under [`data/afrbench-df/`](../data/afrbench-df).

### What the repository ships

| File | Content |
|---|---|
| `annotations.json` | 5,000 protectee frame records and the 5 HQ target records. A frame record has the actor, role, source clip and frame number (`source`), labels (`camera` and `emotion`), the clip as `session`, frame size, face box, five landmarks, detector score, and the face frame |
| `identities.json`, `dataset.json`, `protectee_groups.json` | role counts; counts, seeds, the selection rules, and the clips that were rejected; the HQ target assigned to each actor |

The purge lists, the HQ target images, and the target pool are AFRBench-basic's and are
copied from `data/afrbench-basic/` by `prepare`. Face geometry is given in the pixel
coordinates of the decoded 1920x1080 frame. The annotations contain no frames, no videos,
and no image checksums.

### What you obtain yourself

| Data | Source | Size |
|---|---|---|
| source videos | the DeeperForensics-1.0 authors: read their [terms of use](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master/dataset) and fill in their form; the `source_videos_part_*.zip` files are used as downloaded | about 164 GB |
| MillionCelebs, training corpora | as for AFRBench-basic | as above |

DeeperForensics-1.0 is available for non-commercial research under its own terms and
cannot be redistributed, so no frame is part of this repository. `prepare` decodes each
recorded frame from its clip with OpenCV, writes it as a PNG without resizing, and
requires the recorded size. The manipulated videos of DeeperForensics-1.0 are not used.

### Rebuilding the metadata

The shipped metadata is what `build` writes from the official `lists.zip` and the source
videos, with every choice above seeded, so it can be audited or extended (more actors, other
seeds) without trusting a frame list:

```bash
python -m afrbench.deeperforensics build --videos ~/df --lists ~/df/lists.zip --out my-df-metadata
```

A different decoder or detector build may judge a borderline frame differently and so take
another clip; the shipped metadata is the reference.

## Packed corpus layout

`afrlib.data` packs every training corpus into one LMDB with the same key layout, so a
corpus can be inspected or repacked without the packer:

```
meta/schema            b"afrlib-corpus-v1"
meta/count             number of samples, ASCII
meta/json              packer provenance (packer, source, options)
meta/sample/<i:08d>    JSON metadata for sample i: {"key", "identity", ...}
<sample key>/<field>   the payload, raw bytes
```

Image bytes are copied without re-encoding. The payload fields are whatever the dataset
ships — `image` always, plus for example `mask`, `landmarks` or `attributes` — and each
method's dataset decides which to read. Cropping, alignment, pairing, sampling and
train/test splits are the method's, not the packer's.

## Terms

The metadata is released for research use. The CelebA-HQ target images, the
MillionCelebs images, and the DeeperForensics-1.0 videos are subject to their providers'
terms; see [`THIRD_PARTY.md`](../THIRD_PARTY.md).
