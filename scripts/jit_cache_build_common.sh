#!/bin/bash
# Shared helpers for building flashinfer-jit-cache wheels.
# Sourced by:
#   - scripts/build_flashinfer_jit_cache_whl.sh         (release/nightly)
#   - scripts/task_test_jit_cache_package_build_import.sh (PR tests)

SCCACHE_VERSION="0.17.0"
SCCACHE_CUDA_134_REVISION="e9b15a35f7240a7edd1b9644583edb388c6cb5f9"
SCCACHE_CUDA_134_SOURCE_SHA256="9e444cc5097a839f03c81c59c5cadebc20090aad1fc699e398d5c1c18c44118c"
# The v0.17 client-side architecture remains opt-in while this change isolates
# the version upgrade from a separate execution-mode change.

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

# Download and install a released sccache binary to /usr/local/bin/sccache,
# verifying the upstream sha256 checksum.
install_released_sccache() {
  local sccache_version=$1
  local sccache_arch=$2
  local sccache_package="sccache-v${sccache_version}-${sccache_arch}-unknown-linux-musl"
  local sccache_archive="${sccache_package}.tar.gz"
  local sccache_url="https://github.com/mozilla/sccache/releases/download/v${sccache_version}/${sccache_archive}"
  local sccache_tmpdir
  local sccache_sha256
  local sccache_install_started_at=${SECONDS}

  echo "::group::Install sccache v${sccache_version} (${sccache_arch})"

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
  echo "::endgroup::"
  echo "sccache install duration: $((SECONDS - sccache_install_started_at)) seconds"
}

# Build a pinned sccache revision natively. Building inside each manylinux
# builder produces the matching x86_64 or aarch64 binary without relying on an
# unpublished binary artifact.
build_patched_sccache() {
  local sccache_revision=$1
  local expected_sha256=$2
  local output_path=$3
  local sccache_package="sccache-${sccache_revision}"
  local sccache_archive="${sccache_package}.tar.gz"
  local sccache_url="https://github.com/mozilla/sccache/archive/${sccache_revision}.tar.gz"
  local sccache_tmpdir
  local required_command
  local sccache_build_started_at=${SECONDS}

  for required_command in cargo curl env install make perl sha256sum tar; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
      echo "ERROR: ${required_command} is required to build patched sccache"
      exit 1
    fi
  done

  sccache_tmpdir=$(mktemp -d)
  curl -fsSL "${sccache_url}" -o "${sccache_tmpdir}/${sccache_archive}"
  printf '%s  %s\n' "${expected_sha256}" "${sccache_tmpdir}/${sccache_archive}" | sha256sum -c -
  tar xzf "${sccache_tmpdir}/${sccache_archive}" -C "${sccache_tmpdir}"

  echo "Building sccache revision ${sccache_revision} for $(uname -m)"
  # Do not expose cache credentials to third-party Cargo build scripts.
  env \
    -u AWS_ACCESS_KEY_ID \
    -u AWS_SECRET_ACCESS_KEY \
    -u AWS_SESSION_TOKEN \
    CARGO_INCREMENTAL=0 \
    cargo build \
    --locked \
    --release \
    --no-default-features \
    --features s3,vendored-openssl \
    --bin sccache \
    --manifest-path "${sccache_tmpdir}/${sccache_package}/Cargo.toml"
  mkdir -p "$(dirname "${output_path}")"
  install -m 0755 \
    "${sccache_tmpdir}/${sccache_package}/target/release/sccache" \
    "${output_path}"
  rm -rf "${sccache_tmpdir}"
  echo "patched sccache build duration: $((SECONDS - sccache_build_started_at)) seconds"
}

# Install the pinned source build, reusing a binary produced by a dedicated
# workflow step when available. Other call sites retain a self-contained
# fallback that builds directly into /usr/local/bin.
install_patched_sccache() {
  local sccache_revision=$1
  local expected_sha256=$2

  if [ -n "${SCCACHE_PATCHED_BINARY_PATH:-}" ]; then
    if [ ! -x "${SCCACHE_PATCHED_BINARY_PATH}" ]; then
      echo "ERROR: Prebuilt patched sccache not found: ${SCCACHE_PATCHED_BINARY_PATH}"
      exit 1
    fi
    echo "Installing prebuilt patched sccache from ${SCCACHE_PATCHED_BINARY_PATH}"
    install -m 0755 "${SCCACHE_PATCHED_BINARY_PATH}" /usr/local/bin/sccache
  else
    build_patched_sccache \
      "${sccache_revision}" \
      "${expected_sha256}" \
      /usr/local/bin/sccache
  fi
}

# Install the official release by default. CUDA 13.4 uses the first upstream
# revision containing the CUDA 13.3+ dry-run parser fix while that fix remains
# unreleased: https://github.com/mozilla/sccache/pull/2722
install_sccache() {
  local sccache_version=$1
  local sccache_arch=$2

  case "${sccache_arch}" in
    x86_64|aarch64)
      ;;
    *)
      echo "ERROR: Unsupported sccache build architecture: ${sccache_arch}"
      exit 1
      ;;
  esac

  case "${CUDA_VERSION:-}" in
    13.4|134)
      install_patched_sccache \
        "${SCCACHE_CUDA_134_REVISION}" \
        "${SCCACHE_CUDA_134_SOURCE_SHA256}"
      export FLASHINFER_SCCACHE_INSTALL_SOURCE="github-source"
      export FLASHINFER_SCCACHE_REVISION="${SCCACHE_CUDA_134_REVISION}"
      ;;
    *)
      install_released_sccache "${sccache_version}" "${sccache_arch}"
      export FLASHINFER_SCCACHE_INSTALL_SOURCE="github-release"
      export FLASHINFER_SCCACHE_REVISION="v${sccache_version}"
      ;;
  esac
}

# Install sccache (if missing), configure environment, and start the server.
# Caller must have exported SCCACHE_BUCKET before calling.
#
# Args:
#   $1 - Cache key prefix tag (e.g. "cuda128-x86_64").
#   $2 - Source root directory for SCCACHE_BASEDIRS.
#
# Reads (optional): SCCACHE_REGION, SCCACHE_STATS_DIR,
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
# Sets/exports: SCCACHE_*, FLASHINFER_NVCC_LAUNCHER, FLASHINFER_CXX_LAUNCHER.
setup_sccache() {
  local key_prefix=$1
  local source_root=$2
  local sccache_arch
  sccache_arch=$(uname -m)
  install_sccache "${SCCACHE_VERSION}" "${sccache_arch}"

  export SCCACHE_REGION="${SCCACHE_REGION:-us-west-2}"
  export SCCACHE_BASEDIRS="${source_root}${SCCACHE_BASEDIRS:+:${SCCACHE_BASEDIRS}}"
  export SCCACHE_S3_KEY_PREFIX="${key_prefix}"
  export SCCACHE_IDLE_TIMEOUT=0
  export FLASHINFER_CXX_LAUNCHER="sccache"
  export FLASHINFER_NVCC_LAUNCHER="sccache"

  # Avoid leaking AWS credentials under set -x.
  local _sccache_xtrace=0
  case $- in *x*) _sccache_xtrace=1; set +x ;; esac
  if [ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    unset SCCACHE_S3_NO_CREDENTIALS
    export FLASHINFER_SCCACHE_CACHE_MODE="read-write"
    echo "sccache mode: read-write"
  else
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
    export SCCACHE_S3_NO_CREDENTIALS=true
    export FLASHINFER_SCCACHE_CACHE_MODE="read-only"
    echo "sccache mode: read-only (public bucket, no credentials)"
  fi
  (( _sccache_xtrace )) && set -x

  sccache --start-server
  sccache --zero-stats
  export FLASHINFER_SCCACHE_ACTIVE=true
  echo "sccache version: $(sccache --version)"
  echo "sccache install source: ${FLASHINFER_SCCACHE_INSTALL_SOURCE}"
  echo "sccache revision: ${FLASHINFER_SCCACHE_REVISION}"
  echo "sccache bucket: ${SCCACHE_BUCKET}"
  echo "sccache region: ${SCCACHE_REGION}"
  echo "sccache prefix: ${SCCACHE_S3_KEY_PREFIX}"
  echo "sccache basedirs: ${SCCACHE_BASEDIRS}"
  echo "sccache cxx launcher: ${FLASHINFER_CXX_LAUNCHER}"
  echo "sccache nvcc launcher: ${FLASHINFER_NVCC_LAUNCHER:-disabled}"

  if [ -n "${SCCACHE_STATS_DIR:-}" ]; then
    mkdir -p "${SCCACHE_STATS_DIR}"
    local git_commit
    git_commit=$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || true)
    {
      printf 'git_commit=%s\n' "${git_commit:-unknown}"
      printf 'sccache_version=%s\n' "$(sccache --version)"
      printf 'sccache_install_source=%s\n' "${FLASHINFER_SCCACHE_INSTALL_SOURCE}"
      printf 'sccache_revision=%s\n' "${FLASHINFER_SCCACHE_REVISION}"
      printf 'sccache_cache_mode=%s\n' "${FLASHINFER_SCCACHE_CACHE_MODE}"
      printf 'sccache_bucket=%s\n' "${SCCACHE_BUCKET}"
      printf 'sccache_region=%s\n' "${SCCACHE_REGION}"
      printf 'sccache_key_prefix=%s\n' "${SCCACHE_S3_KEY_PREFIX}"
      printf 'sccache_basedirs=%s\n' "${SCCACHE_BASEDIRS}"
      printf 'sccache_client_side=%s\n' "${SCCACHE_CLIENT_SIDE:-false}"
      printf 'machine_arch=%s\n' "${sccache_arch}"
      printf 'cuda_version=%s\n' "${CUDA_VERSION:-unknown}"
      printf 'cuda_arch_list=%s\n' "${FLASHINFER_CUDA_ARCH_LIST:-unknown}"
    } > "${SCCACHE_STATS_DIR}/sccache-metadata.txt"
  fi
}

# When SCCACHE_STATS_DIR is configured, retain text/JSON stats for a dedicated
# workflow log step and artifact upload. Otherwise, print the stats directly.
# This helper is intended for EXIT traps; callers should ignore failures so
# stats collection never masks the build result.
collect_sccache_stats() {
  if [ "${FLASHINFER_SCCACHE_ACTIVE:-false}" != "true" ] || \
     [ "${FLASHINFER_SCCACHE_STATS_COLLECTED:-false}" = "true" ] || \
     ! command -v sccache >/dev/null 2>&1; then
    return 0
  fi
  export FLASHINFER_SCCACHE_STATS_COLLECTED=true

  if [ -n "${SCCACHE_STATS_DIR:-}" ]; then
    mkdir -p "${SCCACHE_STATS_DIR}"
    if ! sccache --show-stats > "${SCCACHE_STATS_DIR}/sccache-stats.txt"; then
      echo "WARNING: Failed to collect text sccache stats" >&2
    fi
    if ! sccache --show-stats --stats-format=json \
        > "${SCCACHE_STATS_DIR}/sccache-stats.json"; then
      echo "WARNING: Failed to collect JSON sccache stats" >&2
    fi
    if ! sccache --show-adv-stats --stats-format=json \
        > "${SCCACHE_STATS_DIR}/sccache-advanced-stats.json"; then
      echo "WARNING: Failed to collect advanced JSON sccache stats" >&2
    fi
  else
    echo "::group::sccache stats"
    sccache --show-stats || true
    echo "::endgroup::"
  fi
  sccache --stop-server >/dev/null 2>&1 || true
  export FLASHINFER_SCCACHE_ACTIVE=false
}
