"""Registry, download, and cache of the model weights AFR methods depend on."""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import enum
import hashlib
import importlib.util
import os
import re
import shutil
import urllib.request

from .log import get_logger, tqdm

logger = get_logger(__name__)

# Override with AFRLIB_HOME or set_cache_dir().
_DEFAULT_HOME = "~/.afrlib"
_cache_dir_override: Optional[str] = None

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


# Cache location

def get_home_dir() -> str:
    """AFRLib's home directory, holding ``assets/`` and ``datasets/``."""
    home = _cache_dir_override or os.environ.get("AFRLIB_HOME", _DEFAULT_HOME)
    return os.path.expanduser(os.path.expandvars(home))


def get_cache_dir() -> str:
    """Root of the AFRLib asset cache (``<home>/assets``)."""
    return os.path.join(get_home_dir(), "assets")


def set_cache_dir(path: Optional[str]) -> None:
    """Override the AFRLib home directory (``None`` restores the default)."""
    global _cache_dir_override
    _cache_dir_override = path


def _offline() -> bool:
    return os.environ.get("AFRLIB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


# Asset spec and download sources

@dataclass(frozen=True)
class Url:
    """Plain HTTP(S) download; with ``member``, the file is an archive and only that member,
    or that directory prefix if it ends in a slash, is extracted.
    """

    url: str
    member: Optional[str] = None

    def describe(self) -> str:
        return self.url if self.member is None else f"{self.url}!{self.member}"


@dataclass(frozen=True)
class GitHubRelease:
    """Asset attached to a GitHub release (anonymous download URL)."""

    repo: str        # "owner/name"
    tag: str         # release tag
    asset_name: str  # file name attached to the release
    member: Optional[str] = None  # optional archive member to extract

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repo}/releases/download/{self.tag}/{self.asset_name}"

    def describe(self) -> str:
        base = f"github:{self.repo}@{self.tag}/{self.asset_name}"
        return base if self.member is None else f"{base}!{self.member}"


@dataclass(frozen=True)
class GDrive:
    """Google Drive file, fetched with ``gdown``; with ``member``, the file is an archive and
    only that member, or that directory prefix if it ends in a slash, is extracted.
    """

    file_id: str
    member: Optional[str] = None

    def describe(self) -> str:
        base = f"gdrive:{self.file_id}"
        return base if self.member is None else f"{base}!{self.member}"


@dataclass(frozen=True)
class HFHub:
    """File in a Hugging Face Hub repo, fetched through its own cache and then linked or copied
    into the AFRLib cache.
    """

    repo_id: str
    filename: str
    revision: Optional[str] = None

    def describe(self) -> str:
        return f"hf:{self.repo_id}/{self.filename}"


@dataclass(frozen=True)
class Kaggle:
    """Kaggle dataset, fetched with ``kagglehub`` (needs credentials); ``member`` is a path
    inside the extracted directory, not an archive member.
    """

    dataset: str  # "<owner>/<dataset-slug>"
    member: Optional[str] = None

    def describe(self) -> str:
        base = f"kaggle:{self.dataset}"
        return base if self.member is None else f"{base}!{self.member}"


@dataclass(frozen=True)
class Manual:
    """A file the user must obtain by hand; fetching it fails with ``note`` and the
    ``python -m afrlib add`` command that registers their copy.
    """

    url: str
    note: str

    def describe(self) -> str:
        return f"manual: {self.url}"


Source = Union[Url, GitHubRelease, GDrive, HFHub, Kaggle, Manual]


class AssetKind(enum.Enum):
    """Asset category: ``PRIOR`` for a frozen input a method consumes, ``RELEASED`` for an
    authors' checkpoint registered for download only.
    """

    PRIOR = "prior"
    RELEASED = "released"


@dataclass(frozen=True)
class Asset:
    """A downloadable file or extracted directory that a method needs.

    Attributes:
        key: ``"<namespace>/<filename>"``; the namespace (normally the method name)
            is the cache subfolder.
        sha256: Digest of the file, or of the downloaded archive for a directory
            extracted from one; :func:`hash_path` computes it for local files.
            ``""`` skips verification (with a warning).
        size: Size in bytes, for progress bars.
        sources: Download sources, tried in order.
        is_directory: The asset is a directory extracted from an archive; archive
            sources must then name a trailing-slash member prefix.
        rename: Cache file name when it must differ from the source file name.
        kind: :class:`AssetKind` (default ``PRIOR``).
        allow_symlink: Reuse an identical cached asset by symlink instead of a
            hard link or copy. Use only for read-only weights.
    """

    key: str
    sha256: str
    size: int                       # bytes, for progress bars and status display
    sources: Tuple[Source, ...]
    description: str = ""
    is_directory: bool = False
    rename: Optional[str] = None
    kind: AssetKind = AssetKind.PRIOR
    allow_symlink: bool = False

    def __post_init__(self):
        if "/" not in self.key:
            raise ValueError(f"Asset key {self.key!r} must be '<namespace>/<filename>'")
        if not self.sources:
            raise ValueError(f"Asset {self.key!r} declares no download sources")
        if not isinstance(self.kind, AssetKind):
            raise TypeError(
                f"Asset {self.key!r} kind must be an AssetKind, got {self.kind!r} "
                "(training corpora belong in afrlib.data.DATASET_REGISTRY)"
            )
        if self.sha256 and not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError(
                f"Asset {self.key!r} sha256 {self.sha256!r} is not 64 lowercase hex "
                "characters (leave it as '' if the digest has not been measured yet)"
            )
        if self.rename is not None:
            if self.is_directory:
                raise ValueError(f"Directory asset {self.key!r} cannot use rename")
            if not self.rename or self.rename != os.path.basename(self.rename):
                raise ValueError(
                    f"Asset {self.key!r} rename {self.rename!r} must be a plain filename"
                )
        # Only archive-extracted sources mark a directory by a trailing-slash member; Kaggle and
        # HFHub sources hand back a concrete path, so they are exempt.
        for source in self.sources:
            if not isinstance(source, (Url, GitHubRelease, GDrive)):
                continue
            member = source.member
            if self.is_directory and not (member and member.endswith("/")):
                raise ValueError(
                    f"Directory asset {self.key!r} requires a trailing-slash "
                    f"archive-member prefix for source {source.describe()!r}"
                )
            if not self.is_directory and member and member.endswith("/"):
                raise ValueError(
                    f"File asset {self.key!r} cannot use a directory-member "
                    f"prefix for source {source.describe()!r}"
                )

    @property
    def namespace(self) -> str:
        return self.key.split("/", 1)[0]

    @property
    def filename(self) -> str:
        return self.key.split("/", 1)[1]

    @property
    def path(self) -> str:
        return os.path.join(get_cache_dir(), self.namespace, self.rename or self.filename)

    def is_cached(self) -> bool:
        check = os.path.isdir if self.is_directory else os.path.isfile
        return check(self.path)


# Registry

_REGISTRY: Dict[str, Asset] = {}
_discovered = False


def register_assets(*assets: Asset) -> None:
    """Add assets to the global registry (idempotent for identical re-declarations)."""
    for asset in assets:
        existing = _REGISTRY.get(asset.key)
        if existing is not None and existing != asset:
            raise ValueError(f"Asset key {asset.key!r} registered twice with different specs")
        _REGISTRY[asset.key] = asset


def _discover() -> None:
    """Load every ``afrlib/{afr,fr}/<package>/assets.py`` once, by file path, so the method
    packages' torch-heavy ``__init__`` modules are not imported.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for group in ("afr", "fr"):
        group_root = os.path.join(pkg_root, group)
        if not os.path.isdir(group_root):
            continue
        for entry in sorted(os.listdir(group_root)):
            if entry == "__pycache__":
                continue
            decl = os.path.join(group_root, entry, "assets.py")
            if not os.path.isfile(decl):
                continue
            spec = importlib.util.spec_from_file_location(
                f"_afrlib_assets_{group}_{entry}", decl
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # runs register_assets(...) at module level


def list_assets() -> List[Asset]:
    """All registered assets, sorted by key."""
    _discover()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def _lookup(key: str) -> Asset:
    _discover()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown asset {key!r}. Registered assets: {sorted(_REGISTRY) or '(none)'}"
        )
    return _REGISTRY[key]


# Public API

def get_asset(key: str, force_download: bool = False) -> str:
    """Return the local path of asset ``key``, downloading it on cache miss."""
    asset = _lookup(key)
    _ensure_cached(asset, force=force_download)
    return asset.path


def _ensure_cached(asset: Asset, force: bool = False) -> None:
    """Materialize ``asset.path``: already cached, deduped, or freshly downloaded."""
    if not force:
        if asset.is_cached():
            return
        if _try_dedupe(asset):
            return
    if _offline():
        raise RuntimeError(
            f"Asset {asset.key!r} is not cached at {asset.path} and AFRLIB_OFFLINE "
            f"is set. Unset AFRLIB_OFFLINE, or seed the cache with "
            f"`python -m afrlib add {asset.key} <local-file>`."
        )
    _fetch(asset)


def download(
    keys: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    download_all: bool = False,
    force: bool = False,
    kinds: Optional[Sequence["AssetKind"]] = None,
) -> Dict[str, bool]:
    """Prefetch assets; returns ``{key: success}``.

    Args:
        keys: Explicit asset keys.
        methods: Method namespaces; all their assets are included.
        download_all: Fetch every registered asset.
        force: Re-download even when already cached.
        kinds: Keep only these :class:`AssetKind` values.
    """
    _discover()
    selected: Dict[str, Asset] = {}
    if download_all:
        selected = dict(_REGISTRY)
    else:
        for method in methods or ():
            found = {k: a for k, a in _REGISTRY.items() if a.namespace == method}
            if not found:
                logger.warning("No assets registered under namespace %r.", method)
            selected.update(found)
        for key in keys or ():
            selected[key] = _lookup(key)
    if kinds:
        wanted = set(kinds)
        selected = {k: a for k, a in selected.items() if a.kind in wanted}

    results: Dict[str, bool] = {}
    for key in sorted(selected):
        asset = selected[key]
        try:
            _ensure_cached(asset, force=force)
            results[key] = True
        except Exception as e:
            logger.error("Failed to download '%s': %s", key, e)
            results[key] = False
    return results


def add_local(key: str, path: str, move: bool = False) -> str:
    """Seed the cache from a local copy, verifying its checksum, so a file fetched elsewhere is
    not downloaded again.

    Args:
        key: The registered asset key.
        path: The file itself; for a directory asset extracted from an archive, that archive;
            for one with no archive (a ``Kaggle`` source), the extracted tree.
        move: Move the source into the cache instead of linking or copying it.
    """
    asset = _lookup(key)
    path = os.path.expanduser(path)
    member = _archive_member(asset) if asset.is_directory else None

    if member is not None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if asset.sha256:
            digest = _sha256_file(path)
            if digest != asset.sha256:
                raise ValueError(
                    f"Checksum mismatch for {key!r}: {path} has sha256 {digest}, "
                    f"expected {asset.sha256} (the archive's digest, not the "
                    "extracted tree's -- see the Asset docstring)."
                )
        os.makedirs(os.path.dirname(asset.path), exist_ok=True)
        _remove_path(asset.path)
        _extract_member(path, member, asset.path)
        if move:
            os.remove(path)
        logger.info("Extracted '%s' from local archive %s.", key, path)
        return asset.path

    expected = os.path.isdir if asset.is_directory else os.path.isfile
    if not expected(path):
        raise FileNotFoundError(path)
    if asset.sha256:
        digest = _sha256_path(path)
        if digest != asset.sha256:
            raise ValueError(
                f"Checksum mismatch for {key!r}: {path} has sha256 {digest}, "
                f"expected {asset.sha256}."
            )
    os.makedirs(os.path.dirname(asset.path), exist_ok=True)
    _remove_path(asset.path)
    if move:
        shutil.move(path, asset.path)
    elif asset.allow_symlink:
        _symlink_or_copy(path, asset.path, asset.is_directory)
    elif asset.is_directory:
        shutil.copytree(path, asset.path, copy_function=_link_or_copy)
    else:
        _link_or_copy(path, asset.path)
    logger.info("Cached '%s' from %s.", key, path)
    return asset.path


def hash_path(path: str) -> str:
    """The sha256 this registry expects for a local path: a plain file digest, or, for a
    directory, its relative paths and contents hashed in lexical order.
    """
    return _sha256_path(os.path.expanduser(path))


def purge(method: Optional[str] = None) -> List[str]:
    """Delete cached assets, all or one namespace's, and return the removed paths."""
    removed = []
    for asset in list_assets():
        if method is not None and asset.namespace != method:
            continue
        if asset.is_cached():
            _remove_path(asset.path)
            removed.append(asset.path)
    return removed


# Fetching

def _try_dedupe(asset: Asset) -> bool:
    """Create ``asset``'s cache entry from an already-cached asset with the same measured digest,
    by symlink where it opts in and otherwise a hard link or copy; ``True`` when one was found.
    """
    if not asset.sha256:
        return False
    for other in list_assets():
        if other.key == asset.key or other.sha256 != asset.sha256:
            continue
        if other.is_directory != asset.is_directory or not other.is_cached():
            continue
        os.makedirs(os.path.dirname(asset.path), exist_ok=True)
        _remove_path(asset.path)
        if asset.allow_symlink:
            _symlink_or_copy(other.path, asset.path, asset.is_directory)
            logger.info("Linked '%s' -> cached '%s' (same sha256, symlink).", asset.key, other.key)
        elif asset.is_directory:
            shutil.copytree(other.path, asset.path, copy_function=_link_or_copy)
            logger.info("Copied cached '%s' -> '%s' (same sha256).", other.key, asset.key)
        else:
            _link_or_copy(other.path, asset.path)
            logger.info("Linked cached '%s' -> '%s' (same sha256, hard-link/copy).", other.key, asset.key)
        return True
    return False


def _archive_member(asset: Asset) -> Optional[str]:
    """The trailing-slash member of a directory asset's archive source, or ``None`` when it has
    no archive to extract from.
    """
    for source in asset.sources:
        if isinstance(source, (Url, GitHubRelease, GDrive)):
            return source.member
    return None


def _fetch(asset: Asset) -> None:
    """Try each source in order; verify checksum; move into place atomically."""
    fetch(
        asset.path,
        asset.sources,
        sha256=asset.sha256,
        size=asset.size,
        label=asset.key,
        allow_symlink=asset.allow_symlink,
    )


def fetch(
    dest: str,
    sources: Sequence[Source],
    *,
    sha256: str = "",
    size: int = 0,
    label: str = "",
    allow_symlink: bool = False,
) -> str:
    """Materialize ``dest`` from the first source that works, streaming to a ``*.partial`` file,
    verifying any recorded digest, and renaming into place.

    Args:
        dest: Final path of the file or extracted directory.
        sources: Candidates tried in order; the first success wins.
        sha256: Expected digest, or ``""`` to skip verification. For a
            directory extracted from an archive this is the *archive's* digest
            (see :class:`Asset`).
        size: Expected byte count, for the progress bar.
        label: What to call this in logs (an asset key, a corpus part name).
        allow_symlink: Point at an existing local tree instead of copying it,
            where the source hands one back (currently :class:`Kaggle`).

    Returns:
        ``dest``.
    """
    label = label or os.path.basename(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    partial = dest + ".partial"
    errors: List[str] = []
    for source in sources:
        logger.info("Downloading '%s' from %s ...", label, source.describe())
        try:
            _remove_path(partial)
            verified = _download_from(
                source,
                partial,
                sha256=sha256,
                size=size,
                label=label,
                allow_symlink=allow_symlink,
            )
            if not verified:
                if sha256:
                    digest = _sha256_path(partial)
                    if digest != sha256:
                        raise IOError(f"checksum mismatch (got {digest}, expected {sha256})")
                else:
                    logger.warning(
                        "'%s' has no recorded sha256; caching without verification.",
                        label,
                    )
            _remove_path(dest)
            os.replace(partial, dest)
            logger.info("Cached '%s' at %s.", label, dest)
            return dest
        except Exception as e:
            errors.append(f"{source.describe()}: {e}")
            logger.warning("Source %s failed for '%s': %s", source.describe(), label, e)
        finally:
            _remove_path(partial)
    raise RuntimeError(
        f"All sources failed for {label!r}:\n  " + "\n  ".join(errors)
    )


def _shared_archive_path(source: Source) -> Optional[str]:
    """Stable cache path for an archive several assets extract members from, so it is downloaded
    once rather than per member.
    """
    root = os.path.join(get_cache_dir(), ".archives")
    if isinstance(source, GDrive):
        return os.path.join(root, f"gdrive-{source.file_id}")
    if isinstance(source, Url):
        digest = hashlib.sha256(source.url.encode("utf-8")).hexdigest()[:32]
        return os.path.join(root, f"url-{digest}")
    if isinstance(source, GitHubRelease):
        digest = hashlib.sha256(source.url.encode("utf-8")).hexdigest()[:32]
        return os.path.join(root, f"github-{digest}")
    return None


def _download_from(
    source: Source,
    dest: str,
    *,
    sha256: str,
    size: int,
    label: str,
    allow_symlink: bool,
) -> bool:
    """Materialize ``dest`` from ``source`` and report whether the checksum was verified here;
    an archive is checked before extraction, against its own digest.

    Raises:
        RuntimeError: For a :class:`Manual` source, with the note and the ``afrlib add`` command.
    """
    if isinstance(source, Manual):
        raise RuntimeError(
            f"{label} must be obtained manually from {source.url}: {source.note} "
            f"Then run: python -m afrlib add {label} <file>"
        )
    if isinstance(source, Kaggle):
        _download_kaggle(source, dest, allow_symlink=allow_symlink)
        return False
    member = getattr(source, "member", None)
    if member is not None:
        # The source publishes an archive; reuse one cached copy when several
        # assets extract different members from the same file.
        shared = _shared_archive_path(source)
        archive = shared if shared is not None else dest + ".archive"
        ephemeral = shared is None
        try:
            if not os.path.isfile(archive):
                os.makedirs(os.path.dirname(archive), exist_ok=True)
                partial_archive = archive + ".partial"
                if os.path.exists(partial_archive):
                    os.remove(partial_archive)
                _download_from_raw(
                    source, partial_archive, desc=os.path.basename(member)
                )
                os.replace(partial_archive, archive)
            verified = False
            if member.endswith("/") and sha256:
                digest = _sha256_file(archive)
                if digest != sha256:
                    raise IOError(
                        f"checksum mismatch for archive {source.describe()} "
                        f"(got {digest}, expected {sha256})"
                    )
                verified = True
            _extract_member(archive, member, dest)
            return verified
        finally:
            if ephemeral and os.path.exists(archive):
                os.remove(archive)
            partial_archive = archive + ".partial"
            if os.path.exists(partial_archive):
                os.remove(partial_archive)
    _download_from_raw(source, dest, desc=label, total=size)
    return False


def _download_from_raw(source: Source, dest: str, desc: str, total: Optional[int] = None) -> None:
    if isinstance(source, (Url, GitHubRelease)):
        _http_download(source.url, dest, desc=desc, total=total)
    elif isinstance(source, GDrive):
        import gdown

        from .log import are_progress_bars_enabled

        out = gdown.download(id=source.file_id, output=dest, quiet=not are_progress_bars_enabled())
        if out is None:
            raise IOError("gdown returned None (permission / quota / bad id?)")
    elif isinstance(source, HFHub):
        from huggingface_hub import hf_hub_download

        hf_path = hf_hub_download(
            repo_id=source.repo_id, filename=source.filename, revision=source.revision
        )
        _link_or_copy(os.path.realpath(hf_path), dest)
    else:
        raise TypeError(f"Unknown source type: {type(source)!r}")


def _download_kaggle(source: Kaggle, dest: str, *, allow_symlink: bool) -> None:
    """Resolve ``source`` via ``kagglehub`` and materialize it at ``dest``, symlinking into
    kagglehub's own cache where allowed rather than copying the dataset again.
    """
    import kagglehub

    root = os.path.realpath(kagglehub.dataset_download(source.dataset))
    target = os.path.join(root, source.member) if source.member else root
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"Kaggle dataset {source.dataset!r} has no member {source.member!r} "
            f"(downloaded root: {root})"
        )
    if allow_symlink:
        _symlink_or_copy(target, dest, os.path.isdir(target))
    elif os.path.isdir(target):
        shutil.copytree(target, dest, copy_function=_link_or_copy)
    else:
        _link_or_copy(target, dest)


def _extract_member(archive: str, member: str, dest: str) -> None:
    """Extract one file, or a trailing-slash directory prefix, from an archive; tar is preferred
    when both detectors match, since zip detection false-positives on tars of ``.pth`` files.
    """
    import tarfile
    import zipfile

    if member.endswith("/"):
        os.makedirs(dest, exist_ok=True)
        extracted = 0
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                for info in sorted(tf.getmembers(), key=lambda item: item.name):
                    if not info.isfile() or not info.name.startswith(member):
                        continue
                    relative = info.name[len(member):]
                    target = _archive_member_path(dest, relative)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    src = tf.extractfile(info)
                    if src is None:
                        raise IOError(f"cannot read member {info.name!r}")
                    with src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out, length=1 << 20)
                    extracted += 1
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith("/") or not name.startswith(member):
                        continue
                    relative = name[len(member):]
                    target = _archive_member_path(dest, relative)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out, length=1 << 20)
                    extracted += 1
        else:
            raise IOError(f"{archive} is neither a zip nor a tar archive")
        if extracted == 0:
            raise KeyError(f"directory prefix {member!r} not found in {archive}")
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            fobj = tf.extractfile(member)
            if fobj is None:
                raise KeyError(f"member {member!r} not found in {archive}")
            with fobj as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf, zf.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out, length=1 << 20)
    else:
        raise IOError(f"{archive} is neither a zip nor a tar archive")


def _archive_member_path(root: str, relative: str) -> str:
    """Resolve an archive-relative path while rejecting path traversal."""
    parts = relative.replace("\\", "/").split("/")
    if not relative or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe archive member path: {relative!r}")
    target = os.path.abspath(os.path.join(root, *parts))
    if os.path.commonpath((os.path.abspath(root), target)) != os.path.abspath(root):
        raise ValueError(f"archive member escapes destination: {relative!r}")
    return target


def _http_download(url: str, dest: str, desc: str, total: Optional[int] = None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "afrlib"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        if resp.headers.get("Content-Length"):
            total = int(resp.headers["Content-Length"])
        with tqdm(total=total, desc=desc, unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                bar.update(len(chunk))
    if os.path.getsize(dest) == 0:
        raise IOError("downloaded file is empty")


def _link_or_copy(src: str, dest: str) -> None:
    """Hard-link ``src`` to ``dest`` (no extra disk); fall back to a copy."""
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _symlink_or_copy(src: str, dest: str, is_directory: bool) -> None:
    """Symlink ``dest`` -> ``src``; fall back to a hard-link/copy if symlinks
    are not supported (e.g. some Windows setups without the privilege)."""
    try:
        os.symlink(src, dest, target_is_directory=is_directory)
    except OSError:
        if is_directory:
            shutil.copytree(src, dest, copy_function=_link_or_copy)
        else:
            _link_or_copy(src, dest)


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def _sha256_path(path: str) -> str:
    return _sha256_directory(path) if os.path.isdir(path) else _sha256_file(path)


def _sha256_directory(path: str) -> str:
    """Hash relative paths and contents in deterministic lexical order."""
    files = []
    for root, _dirs, names in os.walk(path):
        for name in names:
            full_path = os.path.join(root, name)
            if os.path.isfile(full_path):
                files.append(full_path)
    files.sort(key=lambda item: os.path.relpath(item, path).replace(os.sep, "/"))
    total = sum(os.path.getsize(item) for item in files)
    digest = hashlib.sha256()
    with tqdm(
        total=total,
        desc="verifying",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for full_path in files:
            relative = os.path.relpath(full_path, path).replace(os.sep, "/")
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with open(full_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
                    bar.update(len(chunk))
            digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f, tqdm(
        total=size, desc="verifying", unit="B", unit_scale=True, unit_divisor=1024, leave=False
    ) as bar:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()
