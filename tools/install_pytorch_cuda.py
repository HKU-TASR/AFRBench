#!/usr/bin/env python
"""Install the pinned torch and torchvision wheels for the host CUDA version."""
import os
import re
import subprocess
import sys

TORCH = "torch==2.11.0"
TORCHVISION = "torchvision==0.26.0"

def get_cuda_version():
    """Return the host CUDA ``(major, minor)``, or None, from the CUDA version file, then nvcc,
    then nvidia-smi.
    """
    try:
        if os.path.exists("/usr/local/cuda/version.json"):
            import json
            with open("/usr/local/cuda/version.json", "r") as f:
                data = json.load(f)
                v = data.get("cuda", {}).get("version")
                if v:
                    m = re.match(r"^(\d+)\.(\d+)", v)
                    if m:
                        return int(m.group(1)), int(m.group(2))
    except Exception:
        pass

    for nvcc_cmd in ["/usr/local/cuda/bin/nvcc", "nvcc"]:
        try:
            out = subprocess.check_output([nvcc_cmd, "--version"], stderr=subprocess.STDOUT).decode()
            m = re.search(r"release (\d+)\.(\d+)", out)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    try:
        out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
        m = re.search(r"CUDA Version:\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass

    return None

def main():
    print("Detecting system CUDA version...")
    cuda_ver = get_cuda_version()
    if not cuda_ver:
        print("Warning: Could not detect CUDA version. Installing standard PyTorch (CPU/Default)...")
        cmd = [sys.executable, "-m", "pip", "install", TORCH, TORCHVISION]
        subprocess.check_call(cmd)
        return

    major, minor = cuda_ver
    print(f"Detected CUDA version: {major}.{minor}")

    # PyTorch wheel indexes, newest first; the newest one not above the host CUDA is used,
    # cu126 for older CUDA, and plain PyPI when no CUDA is detected.
    indices = [
        (13, 0, "cu130"),
        (12, 8, "cu128"),
        (12, 6, "cu126"),
    ]

    selected_tag = None
    for index_major, index_minor, tag in indices:
        if (index_major < major) or (index_major == major and index_minor <= minor):
            selected_tag = tag
            break

    if not selected_tag:
        # Host CUDA is older than every index.
        selected_tag = "cu126"

    index_url = f"https://download.pytorch.org/whl/{selected_tag}"
    print(f"Compatible PyTorch CUDA tag selected: {selected_tag}")
    print(f"Installing PyTorch and torchvision from {index_url}...")

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        TORCH,
        TORCHVISION,
        "--index-url",
        index_url,
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("PyTorch and torchvision installed successfully!")

if __name__ == "__main__":
    main()
