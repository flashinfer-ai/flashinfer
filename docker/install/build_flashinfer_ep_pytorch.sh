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
#   CUDA_MAJOR    CUDA major for the cuXX wheel suffix (default: auto from torch.version.cuda)
#   NCCL_VERSION  nvidia-nccl-<cuXX> pin (default 2.30.7)
#   NCCL4PY_SPEC  nccl4py pin (default nccl4py[<cuXX>]==0.3.1)
set -euo pipefail

# Derive the CUDA major (cu12 / cu13 / ...) from the base image's torch so the
# wheel suffixes below track the base image instead of being hardcoded. Override
# with CUDA_MAJOR=<n> if torch can't be imported for some reason.
CUDA_MAJOR="${CUDA_MAJOR:-$(python -c 'import torch; v = torch.version.cuda or ""; print(v.split(".")[0])' 2>/dev/null || true)}"
: "${CUDA_MAJOR:?could not detect CUDA major from torch.version.cuda; set CUDA_MAJOR explicitly}"
CU="cu${CUDA_MAJOR}"

FI_SRC="${FI_SRC:-/host/flashinfer}"
NCCL_VERSION="${NCCL_VERSION:-2.30.7}"
NCCL4PY_SPEC="${NCCL4PY_SPEC:-nccl4py[${CU}]==0.3.1}"
CUDA_CORE_VERSION="${CUDA_CORE_VERSION:-1.0.1}"
CUDA_BINDINGS_VERSION="${CUDA_BINDINGS_VERSION:-13.2.0}"
DEEPGEMM_SRC="${DEEPGEMM_SRC:-/tmp/DeepGEMM}"
DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-891d57b4db1071624b5c8fa0d1e51cb317fa709f}"

echo "== base python / torch / cuda =="
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version | grep release || true

echo "== pin NCCL-EP runtime wheels to ep_bench's verified set =="
# PIP_CONSTRAINT= overrides the NVIDIA base image's constraint file (which pins
# nvidia-nccl-<cuXX> to torch's 2.30.4) so we install the 2.30.7 that nccl4py 0.3.1's
# libnccl_ep.so expects. --no-deps on the NCCL wheels keeps the base torch intact;
# nccl.ep additionally imports cuda.core / cuda.bindings, installed explicitly at
# ep_bench's exact versions (cuda-core 1.0.1, cuda-bindings 13.2.0).
PIP_CONSTRAINT="" pip install --no-cache-dir --no-deps \
    "nvidia-nccl-${CU}==${NCCL_VERSION}" \
    "${NCCL4PY_SPEC}"
PIP_CONSTRAINT="" pip install --no-cache-dir \
    "cuda-core==${CUDA_CORE_VERSION}" \
    "cuda-bindings==${CUDA_BINDINGS_VERSION}"
python -c "import nccl.ep; from nccl.core import Communicator; print('nccl.ep + nccl4py import OK')"

echo "== install DeepGEMM + NVSHMEM / CUTLASS DSL deps =="
PIP_CONSTRAINT="" python -m pip install --no-cache-dir \
    "nvshmem4py-${CU}" \
    pytest \
    "nvidia-nvshmem-${CU}" \
    filelock \
    "nvidia-cutlass-dsl[${CU}]==4.5.0"
(
    if [ ! -d "${DEEPGEMM_SRC}/.git" ]; then
        git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "${DEEPGEMM_SRC}"
    fi
    cd "${DEEPGEMM_SRC}"
    git checkout "${DEEPGEMM_COMMIT}"
    git submodule update --init --recursive
    ./install.sh
)

echo "== build & install FlashInfer (NCCL-EP + Mega path) =="
# The EP backends are ON by default now: NCCL-EP needs no build step (nccl4py
# is a base dependency of flashinfer-python), so only NIXL-EP is opted out.
# PIP_CONSTRAINT= so the build hook's --no-deps NCCL floor upgrade
# (_ensure_nccl_floor, nvidia-nccl-cu13>=2.30.7) isn't blocked by the base
# image's constraint file — a no-op here since 2.30.7 is already pinned above.
cd "${FI_SRC}"
PIP_CONSTRAINT="" BUILD_NIXL_EP=0 \
    pip install --no-cache-dir --no-build-isolation -e .

echo "== pre-warm FlashInfer JIT cache (trtllm fused-MoE reference kernels) =="
# First-use JIT of fused_moe_trtllm_sm100 costs ~25 min of nvcc. Compiled
# lazily under torchrun, that outlives torch's 10-min NCCL watchdog and
# SIGABRTs the job (rank 0 compiles while the others wait in a collective).
# Bake the compiled modules into the image instead: they land in
# ~/.cache/flashinfer, which --container-save captures.
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0a}" python - <<'PYEOF'
from flashinfer.jit.fp4_quantization import gen_fp4_quantization_sm100_module
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

for gen in (gen_trtllm_gen_fused_moe_sm100_module, gen_fp4_quantization_sm100_module):
    spec = gen()
    print(f"[prewarm] building {spec.name} ...", flush=True)
    spec.build_and_load()
    print(f"[prewarm] {spec.name} OK", flush=True)
PYEOF

echo "== smoke probe =="
python -c "\
from flashinfer.moe_ep import available_backends; \
b = available_backends(); print('moe_ep backends:', b); \
assert 'nccl_ep' in b, 'nccl_ep backend missing'"
echo "BUILD OK"
