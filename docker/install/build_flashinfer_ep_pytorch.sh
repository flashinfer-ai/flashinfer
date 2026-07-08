#!/usr/bin/env bash
# Build the FlashInfer NCCL-EP environment INSIDE the NVIDIA PyTorch base image
# (nvcr.io/nvidia/pytorch:26.05-py3), mirroring docker/Dockerfile.flashinfer-ep-pytorch
# but as a script so it can run under `srun --container-save=<out>.sqsh` on
# SLURM/pyxis (the same pattern as ep_bench/scripts/setup_container.sh).
#
# Uses the base image's own python + torch (no venv) so the whole CUDA / NCCL /
# torch / IB-GDAKI stack stays self-consistent — this is what makes cross-node
# NCCL-EP HIGH_THROUGHPUT work (vs the CUDA-13.0 devel image, which 2884s).
#
# Env:
#   FI_SRC        FlashInfer checkout (default /host/flashinfer)
#   NCCL_VERSION  nvidia-nccl-cu13 pin (default 2.30.7)
#   NCCL4PY_SPEC  nccl4py pin (default nccl4py[cu13]==0.3.1)
set -euo pipefail

FI_SRC="${FI_SRC:-/host/flashinfer}"
NCCL_VERSION="${NCCL_VERSION:-2.30.7}"
NCCL4PY_SPEC="${NCCL4PY_SPEC:-nccl4py[cu13]==0.3.1}"
CUDA_CORE_VERSION="${CUDA_CORE_VERSION:-1.0.1}"
CUDA_BINDINGS_VERSION="${CUDA_BINDINGS_VERSION:-13.2.0}"

echo "== base python / torch / cuda =="
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version | grep release || true

echo "== pin NCCL-EP runtime wheels to ep_bench's verified set =="
# PIP_CONSTRAINT= overrides the NVIDIA base image's constraint file (which pins
# nvidia-nccl-cu13 to torch's 2.30.4) so we install the 2.30.7 that nccl4py 0.3.1's
# libnccl_ep.so expects. --no-deps on the NCCL wheels keeps the base torch intact;
# nccl.ep additionally imports cuda.core / cuda.bindings, installed explicitly at
# ep_bench's exact versions (cuda-core 1.0.1, cuda-bindings 13.2.0).
PIP_CONSTRAINT="" pip install --no-cache-dir --no-deps \
    "nvidia-nccl-cu13==${NCCL_VERSION}" \
    "${NCCL4PY_SPEC}"
PIP_CONSTRAINT="" pip install --no-cache-dir \
    "cuda-core==${CUDA_CORE_VERSION}" \
    "cuda-bindings==${CUDA_BINDINGS_VERSION}"
python -c "import nccl.ep; from nccl.core import Communicator; print('nccl.ep + nccl4py import OK')"

echo "== build & install FlashInfer (NCCL-EP only) =="
cd "${FI_SRC}"
BUILD_NCCL_EP=1 BUILD_NIXL_EP=0 \
    pip install --no-cache-dir --no-build-isolation -e .

echo "== smoke probe =="
python -c "\
from flashinfer.moe_ep import available_backends; \
b = available_backends(); print('moe_ep backends:', b); \
assert 'nccl_ep' in b, 'nccl_ep backend missing'"
echo "BUILD OK"
