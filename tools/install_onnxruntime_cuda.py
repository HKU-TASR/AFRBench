#!/usr/bin/env python
"""Install onnxruntime-gpu matched to the host CUDA version, or the CPU build."""
import os
import re
import subprocess
import sys


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
            out = subprocess.check_output(
                [nvcc_cmd, "--version"], stderr=subprocess.STDOUT
            ).decode()
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


def uninstall_existing_ort():
    """onnxruntime and onnxruntime-gpu share the same import path; remove both."""
    print("Removing any existing onnxruntime / onnxruntime-gpu packages...")
    subprocess.call(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            "onnxruntime",
            "onnxruntime-gpu",
        ]
    )


def select_ort_spec(cuda_ver):
    """Return (pip_specs, description) for the best ORT build for this CUDA."""
    if not cuda_ver:
        return (["onnxruntime"], "CPU onnxruntime (CUDA not detected)")

    major, minor = cuda_ver
    # The [cuda] extra pulls the matching nvidia-*-cu{12,13} runtime wheels; a mismatch fails
    # at import with "libcudart.so.<N>: cannot open shared object file".
    if major >= 13:
        return (
            ["onnxruntime-gpu[cuda]==1.27.0"],
            f"onnxruntime-gpu 1.27.0 + CUDA 13 extras (host CUDA {major}.{minor})",
        )
    if major >= 12:
        return (
            ["onnxruntime-gpu[cuda]==1.26.0"],
            f"onnxruntime-gpu 1.26.0 + CUDA 12 extras (host CUDA {major}.{minor})",
        )

    # CUDA 11.x: a legacy wheel that may not support the newest Python; main()
    # falls back to CPU if it fails to install.
    return (
        ["onnxruntime-gpu==1.19.2"],
        f"onnxruntime-gpu 1.19.2 (legacy CUDA 11.x wheel; host CUDA {major}.{minor})",
    )


def verify_install():
    """Check in a subprocess that onnxruntime imports with CUDAExecutionProvider."""
    code = (
        "import onnxruntime as ort; "
        "print(ort.__version__); "
        "print(ort.get_available_providers())"
    )
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", code], stderr=subprocess.STDOUT
        ).decode()
        print(out.strip())
        if "CUDAExecutionProvider" in out:
            print("Success: CUDAExecutionProvider is available.")
            return True
        print("Warning: ORT imported, but CUDAExecutionProvider is not available.")
        return False
    except subprocess.CalledProcessError as e:
        print("Verification failed:")
        print(e.output.decode() if e.output else e)
        return False


def main():
    print("Detecting system CUDA version for ONNX Runtime...")
    cuda_ver = get_cuda_version()
    if cuda_ver:
        print(f"Detected CUDA version: {cuda_ver[0]}.{cuda_ver[1]}")
    else:
        print("Warning: Could not detect CUDA version.")

    specs, description = select_ort_spec(cuda_ver)
    print(f"Selected: {description}")

    uninstall_existing_ort()

    cmd = [sys.executable, "-m", "pip", "install", *specs]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        if specs[0].startswith("onnxruntime-gpu"):
            print(
                "GPU ORT install failed; falling back to CPU onnxruntime..."
            )
            uninstall_existing_ort()
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "onnxruntime"]
            )
        else:
            raise

    print("Verifying ONNX Runtime import...")
    verify_install()
    print("ONNX Runtime installation finished.")


if __name__ == "__main__":
    main()
