#!/bin/bash
# Shared helpers for building flashinfer-jit-cache wheels.
# Sourced by:
#   - scripts/build_flashinfer_jit_cache_whl.sh         (release/nightly)
#   - scripts/task_test_jit_cache_package_build_import.sh (PR tests)

# Compute MAX_JOBS and FLASHINFER_NVCC_THREADS from system memory/CPU,
# clamping FLASHINFER_NVCC_THREADS to a sane range and budgeting per-job
# memory to avoid OOMs on multi-arch builds.
#
# Optional inputs (read from environment):
#   FLASHINFER_NVCC_THREADS - desired nvcc threads per compile (default: 1)
#   AOT_MAX_JOBS_MEMORY_GB  - per-job memory budget (default: max(8, NVCC_THREADS*2))
#   AOT_MAX_JOBS_CAP        - hard upper bound on MAX_JOBS (default: 0 = no cap)
#
# Exports: MAX_JOBS, FLASHINFER_NVCC_THREADS, MEM_PER_JOB
compute_jit_cache_parallelism() {
  local mem_available_gb nproc nvcc_threads
  mem_available_gb=$(free -g | awk '/^Mem:/ {print $7}')
  nproc=$(nproc)

  nvcc_threads=${FLASHINFER_NVCC_THREADS:-1}
  if ! [[ "$nvcc_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid FLASHINFER_NVCC_THREADS=${nvcc_threads}; using 1"
    nvcc_threads=1
  fi
  if (( nvcc_threads > 8 )); then nvcc_threads=8; fi
  if (( nvcc_threads > nproc )); then nvcc_threads=${nproc}; fi
  if (( nvcc_threads < 1 )); then nvcc_threads=1; fi

  # Default to the larger of the historical 8GB/job baseline and ~2GB per nvcc
  # thread when callers explicitly opt into higher nvcc threading.
  local arch_budget=8
  local thread_budget=$(( nvcc_threads * 2 ))
  local default_mem_per_job
  if (( thread_budget > arch_budget )); then
    default_mem_per_job=${thread_budget}
  else
    default_mem_per_job=${arch_budget}
  fi

  local mem_per_job=${AOT_MAX_JOBS_MEMORY_GB:-${default_mem_per_job}}
  local max_jobs_cap=${AOT_MAX_JOBS_CAP:-0}
  if ! [[ "$mem_per_job" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid AOT_MAX_JOBS_MEMORY_GB=${mem_per_job}; using ${default_mem_per_job}"
    mem_per_job=${default_mem_per_job}
  fi
  if ! [[ "$max_jobs_cap" =~ ^[0-9]+$ ]]; then
    echo "Invalid AOT_MAX_JOBS_CAP=${max_jobs_cap}; disabling cap"
    max_jobs_cap=0
  fi

  local max_jobs=$(( mem_available_gb / mem_per_job ))
  if (( max_jobs < 1 )); then
    max_jobs=1
  elif (( nproc < max_jobs )); then
    max_jobs=$nproc
  fi
  if (( max_jobs_cap > 0 && max_jobs > max_jobs_cap )); then
    max_jobs=$max_jobs_cap
  fi

  # Cap total threads at available CPUs.
  local total_threads=$(( max_jobs * nvcc_threads ))
  if (( total_threads > nproc )); then
    max_jobs=$(( nproc / nvcc_threads ))
    if (( max_jobs < 1 )); then max_jobs=1; fi
  fi

  export MAX_JOBS=$max_jobs
  export FLASHINFER_NVCC_THREADS=$nvcc_threads
  export MEM_PER_JOB=$mem_per_job
}

# Download and install sccache to /usr/local/bin/sccache, verifying the
# upstream sha256 checksum.
install_sccache() {
  local sccache_version=$1
  local sccache_arch=$2
  local sccache_package="sccache-v${sccache_version}-${sccache_arch}-unknown-linux-musl"
  local sccache_archive="${sccache_package}.tar.gz"
  local sccache_url="https://github.com/mozilla/sccache/releases/download/v${sccache_version}/${sccache_archive}"
  local sccache_tmpdir
  local sccache_sha256

  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required to verify sccache downloads"
    exit 1
  fi

  sccache_tmpdir=$(mktemp -d)
  curl -fsSL "${sccache_url}" -o "${sccache_tmpdir}/${sccache_archive}"
  curl -fsSL "${sccache_url}.sha256" -o "${sccache_tmpdir}/${sccache_archive}.sha256"
  sccache_sha256=$(awk '{print $1}' "${sccache_tmpdir}/${sccache_archive}.sha256")
  if [ -z "${sccache_sha256}" ]; then
    echo "ERROR: Missing checksum for ${sccache_archive}"
    exit 1
  fi

  printf '%s  %s\n' "${sccache_sha256}" "${sccache_tmpdir}/${sccache_archive}" | sha256sum -c -
  tar xzf "${sccache_tmpdir}/${sccache_archive}" -C "${sccache_tmpdir}"
  mv "${sccache_tmpdir}/${sccache_package}/sccache" /usr/local/bin/
  rm -rf "${sccache_tmpdir}"
  chmod +x /usr/local/bin/sccache
}

# Install sccache (if missing), configure environment, and start the server.
# Caller must have exported SCCACHE_BUCKET before calling.
#
# Args:
#   $1 - Cache key prefix tag (e.g. "cuda128-x86_64").
#   $2 - Source root directory for SCCACHE_BASEDIRS.
#
# Reads (optional): SCCACHE_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
# Sets/exports: SCCACHE_*, FLASHINFER_NVCC_LAUNCHER, FLASHINFER_CXX_LAUNCHER.
setup_sccache() {
  local key_prefix=$1
  local source_root=$2
  local sccache_arch
  sccache_arch=$(uname -m)
  install_sccache "0.9.1" "${sccache_arch}"

  export SCCACHE_REGION="${SCCACHE_REGION:-us-west-2}"
  export SCCACHE_BASEDIRS="${source_root}${SCCACHE_BASEDIRS:+:${SCCACHE_BASEDIRS}}"
  export SCCACHE_S3_KEY_PREFIX="${key_prefix}"
  export SCCACHE_IDLE_TIMEOUT=0
  export FLASHINFER_NVCC_LAUNCHER="sccache"
  export FLASHINFER_CXX_LAUNCHER="sccache"

  # Avoid leaking AWS credentials under set -x.
  local _sccache_xtrace=0
  case $- in *x*) _sccache_xtrace=1; set +x ;; esac
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    unset SCCACHE_S3_NO_CREDENTIALS
    echo "sccache mode: read-write"
  else
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
    export SCCACHE_S3_NO_CREDENTIALS=true
    echo "sccache mode: read-only (public bucket, no credentials)"
  fi
  (( _sccache_xtrace )) && set -x

  sccache --start-server
  echo "sccache version: $(sccache --version)"
  echo "sccache bucket: ${SCCACHE_BUCKET}"
  echo "sccache region: ${SCCACHE_REGION}"
  echo "sccache prefix: ${SCCACHE_S3_KEY_PREFIX}"
  echo "sccache basedirs: ${SCCACHE_BASEDIRS}"
}
