# AFRBench-basic data

AFRBench-basic is the only dataset supported by `afrbench.benchmark`. The repository
ships its metadata under [`data/afrbench-basic/`](../data/afrbench-basic); the protectee
photos and the distractor gallery are obtained separately and assembled with
`python -m afrbench.prepare` (see [benchmark.md](benchmark.md)).

## What the repository ships

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

## What you obtain yourself

| Data | Source | Size |
|---|---|---|
| 5,000 protectee photos | Getty Images, by the `getty_asset_id` of each record (2048-pixel versions) | 3.2 GB |
| MillionCelebs | the authors' release, <https://buptzyb.github.io/MillionCelebs/>; `python -m afrbench.millioncelebs download` fetches it | 445 GB download, 179 GiB packed |
| Training corpora | downloaded by `afrlib.data.get_corpus` when a train job needs them, or your own copy via `--corpus-root` | up to ~70 GB |

The protectee photos are editorial photos from Getty Images and cannot be
redistributed. Obtain them under terms that permit your use. `prepare` requires each
file to have exactly the recorded size and no EXIF rotation, so that the recorded
geometry applies; it does not modify the images.

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

The metadata is released for research use. The CelebA-HQ target images and the
MillionCelebs images are subject to their providers' terms; see
[`THIRD_PARTY.md`](../THIRD_PARTY.md).
