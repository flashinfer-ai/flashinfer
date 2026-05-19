#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Bring up the moe_ep dev environment inside an enroot container on Lyris
# (or any aarch64 Grace-Blackwell host). Run this inside a `srun
# --container-image=nvcr.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
# --container-writable` session; it installs system deps, builds UCX
# v1.21.x + GDRCopy v2.5.1 from source, creates a venv with FlashInfer
# pinned, and finally runs `BUILD_NCCL_EP=1 BUILD_NIXL_EP=1 pip install
# -e ".[nvep]"`.
#
# Env knobs:
#   REPO_ROOT          path to the flashinfer checkout (defaults to PWD)
#   DOCA_VERSION       DOCA version tag (default: 3.2.0-125000-25.10)
#   UCX_VERSION        UCX branch/tag (default: v1.21.x)
#   GDRCOPY_VERSION    GDRCopy tag (default: v2.5.1)
#   VENV               venv path (default: /opt/flashinfer-venv)
#
# Time budget on Lyris aarch64: ~30-40 min (NCCL EP template compile
# dominates; UCX ~5 min; meson NIXL EP ~2 min; FlashInfer kernels ~5 min).

set -eo pipefail

: "${REPO_ROOT:=${PWD}}"
: "${DOCA_VERSION:=3.2.0-125000-25.10}"
: "${UCX_VERSION:=v1.21.x}"
: "${UCX_PREFIX:=/opt/ucx}"
: "${GDRCOPY_VERSION:=v2.5.1}"
: "${VENV:=/opt/flashinfer-venv}"

# Detect Debian multiarch dir at runtime (aarch64-linux-gnu on Lyris).
MULTIARCH="$(dpkg-architecture -q DEB_HOST_MULTIARCH 2>/dev/null || echo "$(uname -m)-linux-gnu")"

export DEBIAN_FRONTEND=noninteractive

echo "=== build_in_container.sh on $(uname -m) [multiarch=${MULTIARCH}] ==="

# ---- 1. apt system deps ----------------------------------------------------
apt-get update
apt-get install -y --no-install-recommends \
    build-essential autoconf automake libtool m4 \
    cmake meson ninja-build patchelf pkg-config \
    git git-lfs ca-certificates curl wget \
    python3 python3-pip python3-venv python3-dev \
    pybind11-dev \
    rdma-core libibverbs-dev libibverbs1 libibumad3 \
    librdmacm-dev libnl-3-dev libnl-route-3-dev \
    ibverbs-providers infiniband-diags \
    libopenmpi-dev openmpi-bin
rm -rf /var/lib/apt/lists/*

# ---- 2. DOCA SDK -----------------------------------------------------------
# Mellanox publishes per-arch .debs at this naming convention; we resolve the
# arch suffix from `dpkg --print-architecture` (arm64 on aarch64, amd64 on x86).
DOCA_ARCH="$(dpkg --print-architecture)"
DOCA_DEB="/tmp/doca-host.deb"
DOCA_URL="https://www.mellanox.com/downloads/DOCA/DOCA_v3.2.0/host/doca-host_${DOCA_VERSION}-ubuntu2404_${DOCA_ARCH}.deb"
echo "=== fetching DOCA from ${DOCA_URL} ==="
wget --tries=3 --waitretry=5 --no-verbose "${DOCA_URL}" -O "${DOCA_DEB}"
dpkg -i "${DOCA_DEB}" || true  # repo registration; install runs below
apt-get update
apt-get install -y --no-install-recommends \
    doca-sdk-gpunetio libdoca-sdk-gpunetio-dev libdoca-sdk-verbs-dev
rm -f "${DOCA_DEB}"
rm -rf /var/lib/apt/lists/*

# Refresh IB verbs from DOCA's apt repo so mlx5dv direct-verbs symbols are
# available (the distro libibverbs-dev predates them).
apt-get update
apt-get install -y --reinstall --no-install-recommends \
    libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
    libnuma-dev librdmacm-dev ibverbs-providers
rm -rf /var/lib/apt/lists/*

# ---- 3. UCX v1.21.x from source -------------------------------------------
echo "=== building UCX ${UCX_VERSION} into ${UCX_PREFIX} ==="
git clone --depth=1 --branch "${UCX_VERSION}" https://github.com/openucx/ucx.git /tmp/ucx
(cd /tmp/ucx \
    && ./autogen.sh \
    && ./configure --prefix="${UCX_PREFIX}" --enable-experimental-api \
        --with-cuda=/usr/local/cuda \
        --with-verbs --with-dm \
        --enable-shared --disable-static \
        --disable-doxygen-doc \
    && make -j"$(nproc)" install)
rm -rf /tmp/ucx
export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:/opt/mellanox/doca/lib/${MULTIARCH}/pkgconfig"
export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:/opt/mellanox/doca/lib/${MULTIARCH}:${LD_LIBRARY_PATH}"
export PATH="${UCX_PREFIX}/bin:${PATH}"
# DOCA gpunetio headers transitively included by UCX's gdaki.cuh; CPATH
# is respected by nvcc just like gcc.
export CPATH="/opt/mellanox/doca/include${CPATH:+:${CPATH}}"

# ---- 4. GDRCopy v2.5.1 -----------------------------------------------------
echo "=== building GDRCopy ${GDRCOPY_VERSION} ==="
git clone --depth=1 --branch "${GDRCOPY_VERSION}" https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
(cd /tmp/gdrcopy && make -j"$(nproc)" lib lib_install)
rm -rf /tmp/gdrcopy

# ---- 5. uv + venv ----------------------------------------------------------
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:${PATH}"

uv venv --python 3.12 --clear "${VENV}"
export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"

uv pip install --python "${VENV}/bin/python" \
    torch setuptools packaging \
    "apache-tvm-ffi>=0.1.6,<0.2,!=0.1.8,!=0.1.8.post0" \
    cython pybind11

# moe_ep runtime base libraries — supplied by these pip wheels. The EP
# plugins ctypes-load libnccl.so.2 / libnixl.so from these wheels at first
# use.
uv pip install --python "${VENV}/bin/python" --no-deps \
    "nixl-cu13>=1.0.1" \
    "nvidia-nccl-cu13>=2.30.4"

# ---- 6. FlashInfer + both EP backends -------------------------------------
# Wipe any half-populated meson subproject extracts from prior aborted runs.
# Meson refuses to setup a subproject whose dir exists but lacks a
# meson.build (happens when the tarball extract from packagecache was
# interrupted). Wraps + packagecache are kept so re-extract is fast.
for sp in /tmp /var/tmp "${REPO_ROOT}/3rdparty/nixl/subprojects"; do : ; done
NIXL_SP="${REPO_ROOT}/3rdparty/nixl/subprojects"
if [[ -d "${NIXL_SP}" ]]; then
    for d in "${NIXL_SP}"/*/ ; do
        name="$(basename "${d}")"
        # packagecache + packagefiles are meson-managed sidecars; keep them.
        if [[ "${name}" == "packagecache" || "${name}" == "packagefiles" ]]; then
            continue
        fi
        if [[ ! -f "${d}/meson.build" ]]; then
            echo "=== cleaning partial meson subproject: ${d} ==="
            rm -rf "${d}"
        fi
    done
fi
# Wipe any stale build_nvep dir from prior aborted runs (we always
# re-setup with --reconfigure in _build_nixl_ep / _synthesize_nccl_builddir
# but a totally-empty dir confuses meson and a previously-completed setup
# may have referenced a removed subproject).
rm -rf "${REPO_ROOT}/build_nvep/nixl" "${REPO_ROOT}/build_nvep/nccl"

echo "=== building flashinfer + moe_ep backends (this is the long step) ==="
cd "${REPO_ROOT}"
BUILD_NCCL_EP=1 BUILD_NIXL_EP=1 \
    uv pip install --python "${VENV}/bin/python" \
        --no-build-isolation -e ".[nvep]"

# Post-build editable installs of the NCCL Python wrappers (sys.executable
# inside the build hook points at uv's isolated env which has no pip).
uv pip install --python "${VENV}/bin/python" \
    -e 3rdparty/nccl/contrib/nccl_ep/python
CUDA_HOME=/usr/local/cuda uv pip install --python "${VENV}/bin/python" \
    -e "3rdparty/nccl/bindings/nccl4py[cu13]"

# ---- 7. persist env for subsequent srun on the same container --------------
cat > /etc/profile.d/flashinfer.sh <<EOF
export PATH=${VENV}/bin:${UCX_PREFIX}/bin:/root/.local/bin:\${PATH}
export LD_LIBRARY_PATH=${UCX_PREFIX}/lib:/opt/mellanox/doca/lib/${MULTIARCH}:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
export PKG_CONFIG_PATH=${UCX_PREFIX}/lib/pkgconfig:/opt/mellanox/doca/lib/${MULTIARCH}/pkgconfig:\${PKG_CONFIG_PATH:-}
export CPATH=/opt/mellanox/doca/include:\${CPATH:-}
export VIRTUAL_ENV=${VENV}
EOF
chmod +x /etc/profile.d/flashinfer.sh

# ---- 8. final smoke probe (in-container) -----------------------------------
"${VENV}/bin/python" -c "
from flashinfer.moe_ep import available_backends
b = available_backends()
print('available_backends:', b)
assert 'nccl_ep' in b, 'nccl_ep backend missing'
assert 'nixl_ep' in b, 'nixl_ep backend missing'
print('build_in_container.sh: OK')
"
