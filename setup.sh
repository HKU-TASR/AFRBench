#!/usr/bin/env bash
# Create (or reuse) the AFRBench conda environment and install AFRLib/AFRBench into it.
#
#   bash setup.sh                 # CUDA machine: conda env "afrbench"
#   bash setup.sh --cpu           # CPU only (tests and light methods)
#   bash setup.sh --prefix DIR    # conda env at DIR instead of a named env
#   bash setup.sh --fresh         # recreate the env
set -euo pipefail

ENV_NAME="afrbench"
PREFIX=""
PYTHON_VERSION="3.12"
CPU=0
FRESH=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) CPU=1 ;;
        --fresh) FRESH=1 ;;
        --prefix) PREFIX="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
CHUMPY="git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
PYTORCH3D="git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
NVDIFFRAST="git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"

command -v conda >/dev/null 2>&1 || { echo "Error: conda is not on PATH." >&2; exit 1; }
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "$PREFIX" ]]; then TARGET=(-p "$PREFIX"); else TARGET=(-n "$ENV_NAME"); fi
exists=0
if [[ -n "$PREFIX" ]]; then [[ -d "$PREFIX/conda-meta" ]] && exists=1
else conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" && exists=1; fi
if [[ "$exists" -eq 1 && "$FRESH" -eq 1 ]]; then
    conda env remove "${TARGET[@]}" -y
    exists=0
fi
if [[ "$exists" -eq 0 ]]; then
    conda create "${TARGET[@]}" "python=$PYTHON_VERSION" -y
fi
conda activate "${PREFIX:-$ENV_NAME}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OS_TYPE="$(uname)"

if [[ "$CPU" -eq 1 ]]; then
    if [[ "$OS_TYPE" == "Linux" ]]; then
        python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
            --index-url https://download.pytorch.org/whl/cpu
    else
        python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
    fi
else
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null; then
        echo "Error: no working NVIDIA driver (nvidia-smi). Use --cpu for a CPU-only install." >&2
        exit 1
    fi
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export PATH="$CUDA_HOME/bin:$PATH"
    python tools/install_pytorch_cuda.py
fi

python -m pip install -r requirements.txt
python -m pip install "$CHUMPY" --no-build-isolation

if [[ "$CPU" -eq 1 ]]; then
    python -m pip install onnxruntime==1.26.0
    echo "CPU install: skipped pytorch3d, nvdiffrast and TensorFlow (Protego and AT3D need them)."
else
    python -m pip install xformers==0.0.35 --no-deps
    if command -v nvcc >/dev/null 2>&1 && { command -v g++ >/dev/null 2>&1 || command -v c++ >/dev/null 2>&1; }; then
        python -m pip install "$PYTORCH3D" --no-build-isolation
        python -m pip install "$NVDIFFRAST" --no-build-isolation
    else
        echo "WARNING: nvcc or a C++ compiler is missing; skipped pytorch3d and nvdiffrast (Protego and AT3D need them)."
    fi
    python -m pip install "tensorflow[and-cuda]==2.21.0"
    python tools/install_onnxruntime_cuda.py
fi

# Several dependencies pull in GUI or contrib OpenCV builds; keep only the headless one.
python -m pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
python -m pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

python -m pip install -e . --no-deps
if [[ "$CPU" -eq 1 ]]; then python tools/verify_env.py --cpu; else python tools/verify_env.py; fi
echo "Done. Activate with: conda activate ${PREFIX:-$ENV_NAME}"
