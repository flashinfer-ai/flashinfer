# cd flashinfer
git submodule update --init 3rdparty/nccl   # required for make -C 3rdparty/nccl/...

pip install --no-build-isolation -e .

pip install --no-deps 'nvidia-nccl-cu13>=2.30.4'

python3 -c "
from build_backend import _synthesize_nccl_builddir
from pathlib import Path
_synthesize_nccl_builddir(Path('build_nvep/nccl'))
"

# GB200/B200 = sm_100. Override for H100: NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
: "${NVCC_GENCODE:=-gencode=arch=compute_100,code=sm_100}"

make -C 3rdparty/nccl/contrib/nccl_ep \
  BUILDDIR="$(pwd)/build_nvep/nccl" \
  "NVCC_GENCODE=${NVCC_GENCODE}" \
  _NCCL_EP_LSA_TEAM_SIZE_MIN=8 \
  _NCCL_EP_LSA_TEAM_SIZE_MAX=8 \
  _NCCL_EP_NUM_LSA_TEAMS_LIST="1" \
  lib -j"$(nproc)"

mkdir -p flashinfer/moe_ep/nccl_ep/_libs
cp build_nvep/nccl/lib/libnccl_ep.so flashinfer/moe_ep/nccl_ep/_libs/

pip install -e 3rdparty/nccl/contrib/nccl_ep/python
CUDA_HOME=/usr/local/cuda pip install -e '3rdparty/nccl/bindings/nccl4py[cu13]'

export FLASHINFER_DISABLE_VERSION_CHECK=1

python -c "from flashinfer.moe_ep import available_backends; print(available_backends())"