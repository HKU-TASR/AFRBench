#!/usr/bin/env python
"""Post-install check of the imports and, unless ``--cpu``, of CUDA availability."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _ok(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"OK   {name}{suffix}")


def _fail(name: str, exc: BaseException) -> None:
    print(f"FAIL {name}: {exc}")


def main() -> int:
    cpu = "--cpu" in sys.argv[1:]
    hard_failed = False

    try:
        import torch

        _ok("torch", f"{torch.__version__} cuda={torch.version.cuda}")
        if not cpu:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is False")
            _ok("torch.cuda", f"devices={torch.cuda.device_count()}")
    except Exception as exc:
        _fail("torch", exc)
        hard_failed = True

    try:
        import onnxruntime as ort

        _ok("onnxruntime", f"providers={ort.get_available_providers()}")
    except Exception as exc:
        _fail("onnxruntime", exc)
        hard_failed = True

    for name in ("yaml", "cv2", "lmdb", "frbench", "afrlib", "afrbench"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "")
            _ok(name, version)
        except Exception as exc:
            _fail(name, exc)
            hard_failed = True

    if hard_failed:
        return 1
    print("environment ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
