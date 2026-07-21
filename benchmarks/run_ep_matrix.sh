#!/usr/bin/env bash
# Drive the FlashInfer EP comm matrix from the login node inside an existing
# salloc, one `srun` step per config into the FlashInfer container.
# Mirrors contrib/nccl_ep/scripts/run_matrix.sh: same configs, COMMON args, fabric
# env (IB = NCCL_GIN_TYPE=3; MNNVL = + NCCL_MNNVL_ENABLE=1). Logs are named so the
# upstream scripts/parse_results.py parses them unchanged.
#
# Required env:
#   JOBID        SLURM job id of the salloc
#   REMOTE_WORK  work dir on the shared FS (holds flashinfer/ checkout + the sqsh)
# Optional:
#   IMAGE          container sqsh (default $REMOTE_WORK/flashinfer-nccl-b200.sqsh)
#   ONLY           regex; run only matching tags (default all)
#   GPUS_PER_NODE  GPUs (ranks) per node (default 4; GB200 = 4/node, B200 = 8/node)
#   NODE_COUNTS    space-separated node counts to sweep (default "1 2 4 8"). Each
#                  case runs on N nodes = N*GPUS_PER_NODE ranks; that product is
#                  the "<G>g" label in the log/tag names.
set -euo pipefail
: "${JOBID:?set JOBID}"
: "${REMOTE_WORK:?set REMOTE_WORK}"
IMAGE="${IMAGE:-$REMOTE_WORK/flashinfer-nccl-b200.sqsh}"
ONLY="${ONLY:-.}"
# Ranks (GPUs) per node. Drives srun --ntasks-per-node and the world size
# (WORLD_SIZE = N_nodes * GPUS_PER_NODE, resolved by SLURM in the wrapper).
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NODE_COUNTS="${NODE_COUNTS:-1 2 4 8}"
MOUNTS="$REMOTE_WORK:/host"
LOGDIR="$REMOTE_WORK/logs_fi"
COMMON="--hidden 7168 --top-k 8 --experts 256 --warmup 20 --iters 100"
mkdir -p "$LOGDIR"

# run <N_nodes> <tag> <fabric_env-string> <bench args...>
run () {
  local N="$1" TAG="$2" FENV="$3"; shift 3
  [[ "$TAG" =~ $ONLY ]] || { echo "skip $TAG"; return 0; }
  rm -f "$REMOTE_WORK/sync_$TAG"   # stale file:// rendezvous would hang
  echo "=== [$TAG] N=$N ranks=$((N*GPUS_PER_NODE)) :: $FENV ==="
  # Don't let one failing case (e.g. the cross-node HT nccl_ep.cc:2884 crash) abort
  # the whole matrix under `set -e`: capture the srun status and keep going.
  local rc=0
  srun --jobid="$JOBID" -N "$N" --ntasks-per-node="$GPUS_PER_NODE" \
    --container-image="$IMAGE" --container-mounts="$MOUNTS" \
    bash -lc "EP_SYNC=/host/sync_$TAG $FENV bash /host/flashinfer/benchmarks/${ONE_SCRIPT:-run_ep_matrix_one.sh} $*" \
    > "$LOGDIR/$TAG.log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then echo "  [$TAG] FAILED (rc=$rc) — see $TAG.log"; else echo "  [$TAG] done"; fi
}

# Build <N_nodes> <total_ranks> pairs from NODE_COUNTS x GPUS_PER_NODE. The
# second value (total ranks) is used only for the "<G>g" log/tag label; the
# actual world size comes from srun (-N * --ntasks-per-node).
SCALES=()
for _n in $NODE_COUNTS; do SCALES+=("$_n $((_n * GPUS_PER_NODE))"); done

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
