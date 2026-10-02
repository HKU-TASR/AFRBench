"""Keep one AFRBench job from occupying every host core."""
from __future__ import annotations

import os

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "ORT_NUM_THREADS",
)


def pin_cpu_threads(n: int = 1) -> None:
    """Cap OpenMP / MKL / OpenCV / Torch thread pools for one GPU job."""
    for key in THREAD_KEYS:
        os.environ.setdefault(key, str(n))
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(n)
    except Exception:
        pass
