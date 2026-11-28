#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PARTITION="${PARTITION:-36x2-a01r}"
ACCOUNT="${ACCOUNT:-coreai_libraries_nccl}"
TIME_LIMIT="${TIME_LIMIT:-20}"
WORLD_SIZE="${WORLD_SIZE:-2}"
CUDA_VISIBLE="${CUDA_VISIBLE_DEVICES:-0,1}"
ATOL="${ATOL:-3.0}"
RTOL="${RTOL:-0.05}"
STRESS_LOOPS="${STRESS_LOOPS:-0}"
STRESS_POINTER_POOL="${STRESS_POINTER_POOL:-1}"
B_LAYOUT="${B_LAYOUT:-nocopy}"

if [[ "${WORLD_SIZE}" == "4" && "${CUDA_VISIBLE}" == "0,1" ]]; then
  CUDA_VISIBLE="0,1,2,3"
fi

srun \
  -p "${PARTITION}" \
  -A "${ACCOUNT}" \
  -J "${ACCOUNT}-flashinfer.blackwell-smoke" \
  -t "${TIME_LIMIT}" \
  --nodes=1 \
  --ntasks=1 \
  --overlap \
  --mpi=pmix \
  bash -lc "
    set -euo pipefail
    cd '$PWD'
    hostname
    export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE}'
    export NVSHMEM_BOOTSTRAP_PMI=PMIX
    export NVSHMEM_REMOTE_TRANSPORT=IBRC
    export NVSHMEM_ENABLE_NIC_PE_MAPPING=1
    if [[ -n \"\${NVSHMEM_HCA_LIST:-}\" ]]; then
      export NVSHMEM_HCA_LIST
    fi
    echo CUDA_VISIBLE_DEVICES=\"\${CUDA_VISIBLE_DEVICES}\"
    for d in /sys/class/infiniband/*; do
      echo \"== \$(basename \${d}) ==\"
      for p in \"\${d}\"/ports/*; do
        echo \"port \$(basename \${p}): \$(cat \${p}/state)\"
      done
    done
    env | egrep '^(PMI|PMIX|SLURM|NCCL|NVSHMEM|CUDA_VISIBLE_DEVICES)=' | sort
    nvidia-smi topo -m
    .venv/bin/torchrun --standalone --nproc-per-node='${WORLD_SIZE}' -- \
      benchmarks/gemm_reduce_scatter/cutlass_cute/bench_flashinfer_blackwell_compare.py \
      --m-values 2048 \
      --k-total-values 4096 \
      --n 1024 \
      --dtype bfloat16 \
      --b-layout '${B_LAYOUT}' \
      --atol '${ATOL}' \
      --rtol '${RTOL}' \
      --stress-loops '${STRESS_LOOPS}' \
      --stress-pointer-pool '${STRESS_POINTER_POOL}' \
      --warmup 1 \
      --iterations 2
  "
