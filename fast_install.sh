# #!/usr/bin/env bash
# # Fast dev install for FlashInfer: moe_ep NCCL-EP (nccl4py wheel) + NIXL-EP + mega runtime deps.
# #
# # Usage (from repo root):
# #   bash fast_install.sh
# #
# # NCCL-EP (default): provided by the nccl4py wheel (>=0.3.1), which ships the
# #   nccl.ep API and bundled libnccl_ep.so. No in-tree make / submodule build.
# #
# # Mega path: installs nvshmem + CuTe DSL so nvfp4/mxfp8/deep_gemm mega kernels can
# #   bootstrap at runtime (deep_gemm itself remains a separate install).
# #
# # NIXL-EP (default): installs nixl-cu13, inits 3rdparty/nixl, builds nixl_ep_cpp.so.
# #   Needs meson, libibverbs. Skip with FI_ENABLE_NIXL_EP=0 or FI_BUILD_NIXL_EP=0.
# #
# # Override NCCL-EP wheel pins (match ep_bench / docker defaults):
# #   FI_NCCL_VERSION=2.30.7 FI_NCCL4PY_SPEC='nccl4py==0.3.1' bash fast_install.sh
# #
# # Skip NCCL-EP wheel install entirely:
# #   FI_ENABLE_NCCL_EP=0 bash fast_install.sh

# set -euo pipefail

# cd "$(dirname "${BASH_SOURCE[0]}")"

# # NVIDIA containers export NCCL_VERSION=2.28.3-1 (Git tag, not a PyPI wheel version).
# # Do NOT inherit NCCL_VERSION here — use FI_NCCL_VERSION to override our PyPI pin.
# NCCL_PYPI_VERSION="${FI_NCCL_VERSION:-2.30.7}"
# # Do not use nccl4py[cu13]: its metadata can pull nvidia-nccl-cu13==2.28.3-1.
# NCCL4PY_SPEC="${FI_NCCL4PY_SPEC:-nccl4py==0.3.1}"
# CUDA_CORE_VERSION="${FI_CUDA_CORE_VERSION:-1.0.1}"
# CUDA_BINDINGS_VERSION="${FI_CUDA_BINDINGS_VERSION:-13.2.0}"
# CUTLASS_DSL_SPEC="${FI_CUTLASS_DSL_SPEC:-nvidia-cutlass-dsl[cu13]>=4.5.0}"

# # NCCL-EP runtime wheels (nccl4py); no BUILD_NCCL_EP compile during pip install.
# ENABLE_NCCL_EP="${FI_ENABLE_NCCL_EP:-${FI_BUILD_NCCL_EP:-1}}"
# BUILD_NIXL_EP="${FI_BUILD_NIXL_EP:-${FI_ENABLE_NIXL_EP:-1}}"
# BUILD_NVEP=0

# # Ignore NVIDIA pip constraint/config files (pins like nvidia-nccl-cu13==2.28.3-1).
# pip_install() {
#   env -u PIP_CONSTRAINT \
#     PIP_CONSTRAINT= \
#     PIP_CONFIG_FILE=/dev/null \
#     python -m pip install --no-cache-dir "$@"
# }

# echo "=== FlashInfer fast install (NCCL-EP=${ENABLE_NCCL_EP}, NIXL-EP=${BUILD_NIXL_EP}) ==="

# if [[ "${ENABLE_NCCL_EP}" == "1" ]]; then
#   echo "Installing NCCL-EP runtime (nvidia-nccl-cu13==${NCCL_PYPI_VERSION}, ${NCCL4PY_SPEC})"
#   pip_install --no-deps "nvidia-nccl-cu13==${NCCL_PYPI_VERSION}"
#   pip_install --no-deps "${NCCL4PY_SPEC}"
#   pip_install --no-deps \
#     "cuda-core==${CUDA_CORE_VERSION}" \
#     "cuda-bindings==${CUDA_BINDINGS_VERSION}"
#   python -c "import nccl.ep; from nccl.core import Communicator; print('nccl.ep + nccl4py import OK')"
# fi

# if [[ "${BUILD_NIXL_EP}" == "1" ]]; then
#   echo "Installing nixl-cu13 (NIXL-EP runtime base lib)"
#   pip_install --no-deps "nixl-cu13>=1.0.1"
# fi

# # Mega kernels: symmetric memory + CuTe DSL (imported by flashinfer + mega backends).
# echo "Installing mega runtime deps (nvshmem, CuTe DSL, filelock)"
# pip_install --no-deps nvshmem4py-cu13 nvidia-nvshmem-cu13
# pip_install filelock
# pip_install "${CUTLASS_DSL_SPEC}"

# if [[ "${BUILD_NIXL_EP}" == "1" ]]; then
#   echo "Initializing git submodules (NIXL-EP only)"
#   git submodule update --init --recursive 3rdparty/nixl
# fi

# # Editable FlashInfer. BUILD_NCCL_EP=0: NCCL-EP is the nccl4py wheel above, not an
# # in-tree compile. --no-deps: wheels above are already in the venv; --no-build-isolation:
# # build_backend runs in this env (meson needs torch on PATH for NIXL-EP).
# echo "Installing FlashInfer editable (NIXL-EP build=${BUILD_NIXL_EP})"
# BUILD_NVEP="${BUILD_NVEP}" \
# BUILD_NCCL_EP=0 \
# BUILD_NIXL_EP="${BUILD_NIXL_EP}" \
#   pip_install --no-build-isolation --no-deps -e ".[nvep]"

# if [[ "${ENABLE_NCCL_EP}" == "1" ]]; then
#   # Guard against any later package pulling torch's older NCCL pin.
#   pip_install --no-deps --force-reinstall "nvidia-nccl-cu13==${NCCL_PYPI_VERSION}"
# fi

# export FLASHINFER_DISABLE_VERSION_CHECK=1
# export BUILD_NIXL_EP

# echo "=== sanity checks ==="
# python -c "
# import importlib.util
# import os

# if importlib.util.find_spec('nccl.ep') is not None:
#     import nccl.ep  # noqa: F401
#     from nccl.core import Communicator  # noqa: F401
#     print('nccl.ep importable: True')
# else:
#     print('nccl.ep importable: False (FI_ENABLE_NCCL_EP=0?)')

# from flashinfer.moe_ep import (
#     MegaConfig,
#     MoEEpLayer,
#     NcclEpConfig,
#     SplitConfig,
#     available_backends,
#     have_nccl_ep,
#     have_nixl_ep,
# )
# print('flashinfer.moe_ep backends:', available_backends())
# print('have_nccl_ep():', have_nccl_ep())
# print('have_nixl_ep():', have_nixl_ep())
# if importlib.util.find_spec('nccl.ep') is not None:
#     assert have_nccl_ep(), 'nccl_ep backend not available (need nccl4py>=0.3.1)'
# if os.environ.get('BUILD_NIXL_EP') == '1':
#     assert have_nixl_ep(), 'nixl_ep backend not available (need nixl-cu13 + BUILD_NIXL_EP=1 build)'

# # Mega API imports cleanly; runtime still needs nvshmem for symmetric-memory kernels.
# try:
#     import nvshmem.core  # noqa: F401
#     print('nvshmem.core importable: True')
# except ImportError as e:
#     print('nvshmem.core importable: False (%s)' % e)

# print('MegaConfig / SplitConfig import OK')
# "

python -m pip install --no-cache-dir \
  nvshmem4py-cu13 \
  pytest \
  nvidia-nvshmem-cu13 \
  filelock \
  "nvidia-cutlass-dsl[cu13]==4.5.0"


git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
git checkout 891d57b4db1071624b5c8fa0d1e51cb317fa709f
git submodule update --init --recursive
./install.sh

BUILD_NVEP=0 BUILD_NCCL_EP=1 BUILD_NIXL_EP=0 \
    pip install --no-cache-dir --no-build-isolation -e ".[nvep]"

