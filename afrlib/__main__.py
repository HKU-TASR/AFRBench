"""AFRLib asset-management CLI: ``python -m afrlib <command>`` (see ``--help``)."""
import argparse
import sys

from .utils import assets
from .utils.assets import AssetKind
from .utils.log import get_logger

logger = get_logger(__name__)

_KINDS = tuple(kind.value for kind in AssetKind)


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"


def _cmd_status(args) -> int:
    entries = assets.list_assets()
    kinds = _selected_kinds(args)
    if kinds:
        entries = [a for a in entries if a.kind in kinds]
    if not entries:
        print("No assets registered.")
        return 0
    key_w = max(len(a.key) for a in entries)
    print(f"Cache dir: {assets.get_cache_dir()}\n")
    for kind in sorted({a.kind for a in entries}, key=lambda k: k.value):
        group = [a for a in entries if a.kind is kind]
        cached = sum(1 for a in group if a.is_cached())
        left = sum(a.size for a in group if not a.is_cached())
        print(f"[{kind.value}]  {cached}/{len(group)} cached"
              + (f"; {_fmt_size(left)} left" if left else ""))
        print(f"  {'ASSET':<{key_w}}  {'STATUS':<8}  {'SIZE':>9}  SOURCES")
        for a in group:
            status = "cached" if a.is_cached() else "missing"
            sources = ", ".join(s.describe() for s in a.sources)
            print(f"  {a.key:<{key_w}}  {status:<8}  {_fmt_size(a.size):>9}  {sources}")
        print()
    missing = [a for a in entries if not a.is_cached()]
    total = sum(a.size for a in missing)
    print(f"{len(entries) - len(missing)}/{len(entries)} cached"
          + (f"; {_fmt_size(total)} left to download" if missing else ""))
    return 0


def _selected_kinds(args) -> list:
    """The ``--kind`` values as :class:`AssetKind` members."""
    return [AssetKind(value) for value in getattr(args, "kind", None) or ()]


def _cmd_download(args) -> int:
    if not args.names and not args.all:
        print("Nothing to download: pass method names / asset keys, or --all.", file=sys.stderr)
        return 2
    keys = [n for n in args.names if "/" in n]
    methods = [n for n in args.names if "/" not in n]
    results = assets.download(keys=keys, methods=methods,
                              download_all=args.all, force=args.force,
                              kinds=_selected_kinds(args))
    if not results:
        print("No matching assets found.", file=sys.stderr)
        return 1
    failed = sorted(k for k, ok in results.items() if not ok)
    print(f"\n{len(results) - len(failed)}/{len(results)} assets ready.")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


def _cmd_add(args) -> int:
    try:
        path = assets.add_local(args.key, args.path, move=args.move)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Cached at {path}")
    return 0


def _cmd_hash(args) -> int:
    try:
        print(assets.hash_path(args.path))
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_purge(args) -> int:
    removed = assets.purge(args.method)
    if removed:
        for path in removed:
            print(f"Removed {path}")
    else:
        print("Nothing to purge.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m afrlib",
                                     description="Manage AFRLib model assets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="show all registered assets and cache state")
    p.add_argument("--kind", action="append", choices=_KINDS,
                   help="only show this kind (repeatable)")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("download", help="download assets into the cache")
    p.add_argument("names", nargs="*",
                   help="method names (e.g. dadp) or asset keys (e.g. dadp/ffhq256_autoenc.ckpt)")
    p.add_argument("--all", action="store_true", help="download every registered asset")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.add_argument("--kind", action="append", choices=_KINDS,
                   help="only fetch this kind, e.g. --kind prior to skip the "
                        "authors' released artifacts (repeatable)")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("add", help="seed the cache from a local file")
    p.add_argument("key", help="asset key, e.g. dadp/ffhq256_autoenc.ckpt")
    p.add_argument("path", help="local file to import (checksum verified)")
    p.add_argument("--move", action="store_true", help="move instead of link/copy")
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("hash", help="print the sha256 a local file/directory would need")
    p.add_argument("path", help="local file or directory to hash")
    p.set_defaults(func=_cmd_hash)

    p = sub.add_parser("purge", help="delete cached assets")
    p.add_argument("method", nargs="?", default=None,
                   help="only purge this method's assets (default: all)")
    p.set_defaults(func=_cmd_purge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
