"""MillionCelebs noise cache: lazy reader, resumable writer, crop loader, and cache build."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from afrbench.dataset import decode_image
from afrbench.evaluator.mc_crops import MillionCelebsCrops
from afrbench.evaluator.noise_cache import (
    CACHE_EMBED_BATCH,
    EMBEDDINGS_NAME,
    FAILED_NAME,
    FullNoiseCache,
    IDENTITIES_NAME,
    IncrementalWriter,
    KEYS_NAME,
    META_NAME,
    VOCAB_LINES_NAME,
    VOCAB_NAME,
    _cache_embed_batch,
    build_full_noise_cache,
    cache_identity,
)
from afrbench.fr_system import FRSystem
from afrlib.data.lmdb_io import CorpusWriter, lmdb_env
from afrbench.protocol import fr_system


def _write_cache(root, *, count=4, dim=2, failed=()):
    embeddings = np.memmap(root / EMBEDDINGS_NAME, mode="w+", dtype=np.float32, shape=(count, dim))
    embeddings[:] = np.eye(count, dim, dtype=np.float32)
    embeddings.flush()
    identities = np.memmap(root / IDENTITIES_NAME, mode="w+", dtype=np.int32, shape=(count,))
    identities[:] = np.array([0, 0, 1, 2], dtype=np.int32)
    identities.flush()
    (root / KEYS_NAME).write_text("".join(f"id{i}__{i}.jpg\n" for i in range(count)))
    (root / VOCAB_NAME).write_text(json.dumps(["245237", "55023", "other"]))
    (root / FAILED_NAME).write_text(json.dumps(list(failed)))
    (root / META_NAME).write_text(json.dumps({
        "complete": True,
        "count": count,
        "dim": dim,
        "pack_id": "pack",
        "fr_digest": "fr",
        "weight_identity": "w",
    }))


def test_iter_blocks_does_not_materialize_keys(tmp_path):
    _write_cache(tmp_path, failed=["id3__3.jpg"])
    reader = FullNoiseCache(tmp_path).reader()
    blocks = list(reader.iter_blocks(exclude_identities=["55023"], chunk=2))
    keys = [key for _emb, _labels, block_keys in blocks for key in block_keys]
    assert keys == ["id0__0.jpg", "id1__1.jpg"]
    assert reader._keys is None
    n_images, n_ids = reader.kept_counts(["55023"])
    assert n_images == 2
    assert n_ids == 1
    assert reader.kept_counts(["55023"]) == (2, 1)


def test_iter_rank_blocks_skips_keys_when_failed_empty(tmp_path):
    _write_cache(tmp_path)
    reader = FullNoiseCache(tmp_path).reader()
    blocks = list(reader.iter_rank_blocks(exclude_identities=["55023"], chunk=2))
    codes = []
    index = []
    for item in blocks:
        block_codes, block_idx = item[1], item[2]
        keep = item[3] if len(item) == 4 else None
        for code, row, kept in zip(
            block_codes.tolist(),
            block_idx.tolist(),
            keep.tolist() if keep is not None else [True] * len(block_codes),
        ):
            if kept:
                codes.append(int(code))
                index.append(int(row))
    assert codes == [0, 0, 2]
    assert index == [0, 1, 3]
    assert reader._keys is None
    assert reader.kept_counts(["55023"]) == (3, 2)
    assert reader.lookup_keys(index) == ["id0__0.jpg", "id1__1.jpg", "id3__3.jpg"]
    assert reader._keys is None


def test_incremental_writer_resumes_without_rewriting_vocab_json(tmp_path):
    root = tmp_path / "mc"
    writer = IncrementalWriter(root, dim=2, expected=4)
    writer.append(
        ["n__0", "n__1"],
        np.array([[1, 0], [0, 1]], dtype=np.float32),
        ["id0", "id1"],
    )
    assert json.loads((root / VOCAB_NAME).read_text()) == []
    lines = [
        json.loads(line)
        for line in (root / VOCAB_LINES_NAME).read_text().splitlines()
        if line
    ]
    assert lines == ["id0", "id1"]
    writer.close()

    resumed = IncrementalWriter(root, dim=2, expected=4)
    assert resumed.next_index == 2
    assert resumed.vocab == ["id0", "id1"]
    resumed.append(
        ["n__2", "n__3"],
        np.array([[1, 1], [0, 0]], dtype=np.float32),
        ["id2", "id0"],
    )
    resumed.finalize(pack_id="tiny", fr_digest="fr", weight_identity="w")
    assert json.loads((root / VOCAB_NAME).read_text()) == ["id0", "id1", "id2"]
    assert not (root / VOCAB_LINES_NAME).is_file()
    reader = FullNoiseCache(root).reader()
    assert reader.count == 4
    assert reader.identity_at(2) == "id2"
    assert reader.identity_at(3) == "id0"


class _StubFR:
    input_size = (112, 112)
    embedding_dim = 4

    def embed_crops(self, crops, **_kwargs):
        # Per-crop only: batch size must not change the row.
        base = crops.mean(dim=(1, 2, 3), keepdim=False)
        corner = crops[:, 0, 0, 0]
        return torch.stack([base, corner, base * 2, torch.ones_like(base)], dim=1)


def _jpeg_bytes(seed: int, size: int = 112) -> bytes:
    rng = np.random.RandomState(seed)
    array = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _write_tiny_lmdb(path, *, count: int = 4):
    with CorpusWriter(
        path,
        map_size=8 * 1024 * 1024,
        provenance={"aligned": True, "pack_id": "tiny-pack", "source": "test"},
    ) as writer:
        for index in range(count):
            key = f"id{index}__{index}.jpg"
            writer.add(key, {"image": _jpeg_bytes(index)}, identity=f"id{index}")
    return path


def _tiny_dataset(root, lmdb_path):
    return SimpleNamespace(
        root=root,
        noise_lmdb=lmdb_path,
        noise_provenance={"aligned": True, "pack_id": "tiny-pack", "source": "test"},
        noise_aligned=True,
    )


def test_millioncelebs_crops_match_decode_image_and_key_order(tmp_path):
    lmdb_path = _write_tiny_lmdb(tmp_path / "noise.lmdb", count=4)
    crops = MillionCelebsCrops(lmdb_path, start=0, scale_01=False)
    env = lmdb_env(lmdb_path)
    keys = []
    with env.begin() as txn:
        for index in range(4):
            record = json.loads(txn.get(f"meta/sample/{index:08d}".encode("ascii")))
            key = str(record["key"])
            payload = txn.get(f"{key}/image".encode("utf-8"))
            tensor, got_key, identity = crops[index]
            assert got_key == key
            assert identity == f"id{index}"
            assert torch.equal(tensor, decode_image(payload))
            keys.append(key)
    resumed = MillionCelebsCrops(lmdb_path, start=2, scale_01=False)
    assert len(resumed) == 2
    assert resumed[0][1] == keys[2]
    assert resumed[1][1] == keys[3]


def test_yaml_batch_32_does_not_shrink_mc_floor():
    assert CACHE_EMBED_BATCH == 256
    assert _cache_embed_batch(None, 32) == 256
    assert _cache_embed_batch(None, 1) == 256
    assert _cache_embed_batch(None, 256) == 256
    assert _cache_embed_batch(None, 512) == 512


def test_build_full_noise_cache_matches_list_embed_and_resumes(tmp_path, monkeypatch):
    lmdb_path = _write_tiny_lmdb(tmp_path / "noise.lmdb", count=4)
    dataset = _tiny_dataset(tmp_path / "basic", lmdb_path)
    monkeypatch.setattr(FRSystem, "build_fr", lambda self, device: _StubFR())
    system = FRSystem.from_dict(fr_system("minimal"))
    crops = MillionCelebsCrops(lmdb_path, start=0, scale_01=False)
    items = [(key, tensor) for tensor, key, _identity in (crops[i] for i in range(4))]
    ids, expected, failed = system.embed_aligned(items, device="cpu", batch_size=2)
    assert failed == []
    assert ids == [item[0] for item in items]

    cache_root = tmp_path / "cache"
    root = build_full_noise_cache(
        dataset=dataset,
        system=system,
        device="cpu",
        cache_root=cache_root,
        batch_size=32,
        num_workers=0,
    )
    cache = FullNoiseCache(root)
    meta = cache.meta()
    assert meta["aligned"] is True
    assert meta["source"] == str(lmdb_path)
    assert "amp" not in meta
    reader = cache.reader()
    assert list(reader.iter_keys()) == [item[0] for item in items]
    assert np.allclose(np.asarray(reader.embeddings), expected.numpy())

    weight_id = system.for_gallery().weight_identity
    digest = cache_identity(
        pack_id="tiny-pack",
        fr_digest=system.for_gallery().digest,
        weight_identity=weight_id,
    )
    resume_root = cache_root / "resume" / digest
    writer = IncrementalWriter(resume_root, dim=4, expected=4)
    writer.append(
        [items[0][0], items[1][0]],
        expected[:2].numpy(),
        ["id0", "id1"],
    )
    writer.embeddings.flush()
    writer.identities.flush()
    writer.close()
    resumed = build_full_noise_cache(
        dataset=dataset,
        system=system,
        device="cpu",
        cache_root=cache_root / "resume",
        batch_size=2,
        num_workers=0,
    )
    assert resumed == resume_root
    again = FullNoiseCache(resumed).reader()
    assert list(again.iter_keys()) == [item[0] for item in items]
    assert np.allclose(np.asarray(again.embeddings), expected.numpy())
