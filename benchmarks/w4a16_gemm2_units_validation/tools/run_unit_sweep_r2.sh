#!/usr/bin/env bash
# Devbox: second FC2 unit-size sweep (cluster 2x1, route tile 64, repeat for noise).
set -uo pipefail
ROOT=/scratch/ziangli/flashinfer-w4a16-tp-low-concurrency
OUT=${OUT:-$ROOT/results/sweep/r2}
mkdir -p "$OUT"
cd "$ROOT/flashinfer"
source "$ROOT/venv/bin/activate"
export PYTHONPATH=$PWD FLASHINFER_CUDA_ARCH_LIST=10.3a
run() {  # gpu label hidden k experts topk arms tokens route_tiles clusters
  CUDA_VISIBLE_DEVICES=$1 timeout 5400 python ../tools/fc2_unit_sweep.py --label "$2" \
    --hidden "$3" --intermediate "$4" --experts "$5" --top-k "$6" --arms "$7" \
    --tokens "$8" --route-tiles "$9" --clusters "${10}" > "$OUT/$2.jsonl" 2> "$OUT/$2.log" &
}
T=1,2,4,8,16,32,64,128,256,512
run 0 v3-tp8-c21    7168 256 256 8  1,2,4 $T 8,16,32 2x1
run 1 glm5-tp8-c21  6144 256 256 8  1,2,4 $T 8,16,32 2x1
run 2 k3-tp8-c21    3584 384 896 16 1,2   $T 8,16,32 2x1
run 3 v3-tp4-c21    7168 512 256 8  1,2,4 $T 8,16,32 2x1
run 4 k3-tp8-rep    3584 384 896 16 1,2,4 $T 8,16,32 1x1
run 5 v3-tp8-rt64   7168 256 256 8  1,2,4 32,64,128,256,512 64 1x1
run 6 glm5-tp8-rt64 6144 256 256 8  1,2,4 32,64,128,256,512 64 1x1
run 7 v3-tp8-rep    7168 256 256 8  1,2,4 $T 8,16,32 1x1
wait
echo "done $OUT"
