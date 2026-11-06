#!/bin/bash
# Trial driver: time the sm80 provider build under one sccache flavor and pass.
#
#   FLAVOR=upstream|rapids   PASS=seed|measure   [AOT_TRIAL_ARCHS=8.0]
#
# seed    : read-write (needs AWS_* creds), fills the flavor's S3 prefix.
# measure : read-only (no creds), the hit-path measurement.
# Run inside the CI image from the repo root. Writes results under
# ${TRIAL_OUT:-trial-out}/<flavor>-<pass>/ : timing.txt, sccache stats, wheel sha256s.
set -eo pipefail
FLAVOR=${FLAVOR:?upstream|rapids}
PASS=${PASS:?seed|measure}
OUT=${TRIAL_OUT:-trial-out}/${FLAVOR}-${PASS}
mkdir -p "${OUT}"
OUT=$(cd "${OUT}" && pwd)

[ "${FLAVOR}" = rapids ] && export FLASHINFER_SCCACHE_FLAVOR=rapids
[ "${PASS}" = measure ] && unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AOT_PROVIDER_ARCHS_OVERRIDE="${AOT_TRIAL_ARCHS:-8.0}"
export SCCACHE_STATS_DIR="${OUT}"
export SCCACHE_LOG=${SCCACHE_LOG:-warn}

start=$(date +%s.%N)
/usr/bin/time -v -o "${OUT}/time-v.txt" \
  scripts/task_test_jit_cache_package_build_import.sh 2>&1 | tee "${OUT}/build.log"
end=$(date +%s.%N)

{
  echo "flavor=${FLAVOR} pass=${PASS} archs=${AOT_PROVIDER_ARCHS_OVERRIDE}"
  echo "wall_s=$(echo "${end} - ${start}" | bc)"
  grep -E "Provider wheels built in|Maximum resident|User time|System time|Elapsed" \
    "${OUT}/build.log" "${OUT}/time-v.txt" || true
  echo "nproc=$(nproc) cpu=$(lscpu | sed -n 's/^Model name: *//p')"
} > "${OUT}/timing.txt"
( cd dist-jit-cache-provider-test && sha256sum *.whl ) > "${OUT}/wheels.sha256" || true
cat "${OUT}/timing.txt"
