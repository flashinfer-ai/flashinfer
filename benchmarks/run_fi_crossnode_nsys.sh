#!/bin/bash
RW=/lustre/fsw/coreai_libraries_cudnn/agopal-moe-ep
IMG=$RW/flashinfer-ep-pt2605.sqsh
JOBID=2278799
COMMON="--hidden 7168 --top-k 8 --experts 256 --warmup 5 --iters 20"
run() { local N=$1 TAG=$2 FENV=$3; shift 3
  rm -f "$RW/sync_$TAG" "$RW/prof_fi/$TAG.nsys-rep"
  echo "=== [$TAG] N=$N ==="
  srun --jobid=$JOBID -N "$N" --ntasks-per-node=8 --container-image=$IMG --container-mounts=$RW:/host \
    bash -lc "EP_SYNC=/host/sync_$TAG PROF_TAG=$TAG $FENV bash /host/flashinfer/benchmarks/run_ep_matrix_one_pt_nsys.sh $*" \
    > "$RW/prof_fi/$TAG.runlog" 2>&1
  echo "  [$TAG] rc=$? rep=$(ls -la $RW/prof_fi/$TAG.nsys-rep 2>/dev/null | awk '{print $5}') bytes"
}
run 2 ll_em_16g_ib  "NCCL_GIN_TYPE=3" --algorithm ll --layout em --tokens 128  $COMMON
run 4 ll_em_32g_ib  "NCCL_GIN_TYPE=3" --algorithm ll --layout em --tokens 128  $COMMON
run 2 ht_4096_16g_ib "NCCL_GIN_TYPE=3" --algorithm ht --layout fl --tokens 4096 $COMMON
run 4 ht_4096_32g_ib "NCCL_GIN_TYPE=3" --algorithm ht --layout fl --tokens 4096 $COMMON
run 8 ht_4096_64g_ib "NCCL_GIN_TYPE=3" --algorithm ht --layout fl --tokens 4096 $COMMON
echo "ALL CROSSNODE DONE"
