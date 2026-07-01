#!/usr/bin/env bash
# Drive the FlashInfer EP comm matrix (28 cases) from the login node inside an
# existing salloc (-N 8), one `srun` step per config into the FlashInfer container.
# Mirrors contrib/nccl_ep/scripts/run_matrix.sh: same configs, COMMON args, fabric
# env (IB = NCCL_GIN_TYPE=3; MNNVL = + NCCL_MNNVL_ENABLE=1). Logs are named so the
# upstream scripts/parse_results.py parses them unchanged.
#
# Required env:
#   JOBID        SLURM job id of the salloc
#   REMOTE_WORK  work dir on the shared FS (holds flashinfer/ checkout + the sqsh)
# Optional:
#   IMAGE  container sqsh (default $REMOTE_WORK/flashinfer-nccl-b200.sqsh)
#   ONLY   regex; run only matching tags (default all)
set -euo pipefail
: "${JOBID:?set JOBID}"
: "${REMOTE_WORK:?set REMOTE_WORK}"
IMAGE="${IMAGE:-$REMOTE_WORK/flashinfer-nccl-b200.sqsh}"
ONLY="${ONLY:-.}"
MOUNTS="$REMOTE_WORK:/host"
LOGDIR="$REMOTE_WORK/logs_fi"
COMMON="--hidden 7168 --top-k 8 --experts 256 --warmup 20 --iters 100"
mkdir -p "$LOGDIR"

# run <N_nodes> <tag> <fabric_env-string> <bench args...>
run () {
  local N="$1" TAG="$2" FENV="$3"; shift 3
  [[ "$TAG" =~ $ONLY ]] || { echo "skip $TAG"; return 0; }
  rm -f "$REMOTE_WORK/sync_$TAG"   # stale file:// rendezvous would hang
  echo "=== [$TAG] N=$N ranks=$((N*8)) :: $FENV ==="
  # Don't let one failing case (e.g. the cross-node HT nccl_ep.cc:2884 crash) abort
  # the whole matrix under `set -e`: capture the srun status and keep going.
  local rc=0
  srun --jobid="$JOBID" -N "$N" --ntasks-per-node=8 \
    --container-image="$IMAGE" --container-mounts="$MOUNTS" \
    bash -lc "EP_SYNC=/host/sync_$TAG $FENV bash /host/flashinfer/benchmarks/${ONE_SCRIPT:-run_ep_matrix_one.sh} $*" \
    > "$LOGDIR/$TAG.log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then echo "  [$TAG] FAILED (rc=$rc) — see $TAG.log"; else echo "  [$TAG] done"; fi
}

SCALES=("1 8" "2 16" "4 32" "8 64")

# correctness gate (single node)
run 1 validate_ll "NCCL_GIN_TYPE=3" --algorithm ll --layout em --tokens 128 --validate
run 1 validate_ht "NCCL_GIN_TYPE=3" --algorithm ht --layout fl --tokens 4096 --validate

# LL: 128 tokens/rank, expert-major + rank-major
for L in em rm; do
  for s in "${SCALES[@]}"; do set -- $s; N=$1; G=$2
    run "$N" "ll_${L}_${G}g_ib" "NCCL_GIN_TYPE=3" \
      --algorithm ll --layout "$L" --tokens 128 $COMMON
    if [ "$N" -gt 1 ]; then
      run "$N" "ll_${L}_${G}g_mnnvl" "NCCL_GIN_TYPE=3 NCCL_MNNVL_ENABLE=1" \
        --algorithm ll --layout "$L" --tokens 128 $COMMON
    fi
  done
done

# HT: flat layout, 4096 + 8192 tokens/rank
for T in 4096 8192; do
  for s in "${SCALES[@]}"; do set -- $s; N=$1; G=$2
    run "$N" "ht_${T}_${G}g_ib" "NCCL_GIN_TYPE=3" \
      --algorithm ht --layout fl --tokens "$T" $COMMON
    if [ "$N" -gt 1 ]; then
      run "$N" "ht_${T}_${G}g_mnnvl" "NCCL_GIN_TYPE=3 NCCL_MNNVL_ENABLE=1" \
        --algorithm ht --layout fl --tokens "$T" $COMMON
    fi
  done
done

echo "All runs complete. Logs in $LOGDIR (parse with the upstream scripts/parse_results.py)"
