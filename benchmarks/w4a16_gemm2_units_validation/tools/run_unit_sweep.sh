#!/usr/bin/env bash
# Devbox: one FC2 unit-size sweep per GPU (kernel-level, cold L2 CUPTI, CUDA graph).
set -uo pipefail
ROOT=/scratch/ziangli/flashinfer-w4a16-tp-low-concurrency
OUT=${OUT:-$ROOT/results/sweep/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
cd "$ROOT/flashinfer"
source "$ROOT/venv/bin/activate"
export PYTHONPATH=$PWD FLASHINFER_CUDA_ARCH_LIST=10.3a
TOK=${TOK:-1,2,4,8,16,32,64,128,256,512}
RT=${RT:-8,16,32}
CL=${CL:-1x1}
run() {  # gpu label hidden k experts topk arms
  CUDA_VISIBLE_DEVICES=$1 timeout 5400 python ../tools/fc2_unit_sweep.py --label "$2" \
    --hidden "$3" --intermediate "$4" --experts "$5" --top-k "$6" --arms "$7" \
    --tokens "$TOK" --route-tiles "$RT" --clusters "$CL" > "$OUT/$2.jsonl" 2> "$OUT/$2.log" &
}
run 0 v3-tp8      7168 256 256 8  1,auto,2,4,7,8
run 1 glm5-tp8    6144 256 256 8  1,auto,2,3,4,6,8
run 2 k3-tp8      3584 384 896 16 1,auto,2,4,7
run 3 v41f-tp4    5120 576 384 6  1,auto,2,4,5,8
run 4 n3u-tp8     2048 640 512 22 1,auto,2,4,8
run 5 v3-tp4      7168 512 256 8  1,auto,2,4,7,8
run 6 glm5-tp4    6144 512 256 8  1,auto,2,3,4,6
run 7 v3-k1024    7168 1024 256 8 1,auto,2,4
wait
echo "done $OUT"
