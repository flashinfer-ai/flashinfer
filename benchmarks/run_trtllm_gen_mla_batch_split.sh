#!/usr/bin/env bash
# Evaluate TRTLLM-GEN MLA decode with an optional batch-split LUT.
#
# Example:
#   LUT_CONFIG_ENV_VAL=/path/to/lut.json \
#   Q_LEN_PER_REQUEST=2 BATCH_SIZE=9 SEQ_LEN=20480 FIXED_SEQ_LEN=1 \
#   ./benchmarks/run_trtllm_gen_mla_batch_split.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BACKEND:-trtllm-gen}"
BATCH_SIZE="${BATCH_SIZE:-40}"
Q_LEN_PER_REQUEST="${Q_LEN_PER_REQUEST:-2}"
SEQ_LEN="${SEQ_LEN:-20480}"
PAGE_SIZE="${PAGE_SIZE:-32}"
NUM_QO_HEADS="${NUM_QO_HEADS:-64}"
HEAD_DIM_CKV="${HEAD_DIM_CKV:-512}"
HEAD_DIM_KPE="${HEAD_DIM_KPE:-64}"
QK_NOPE_HEAD_DIM="${QK_NOPE_HEAD_DIM:-128}"
DTYPE="${DTYPE:-fp8_e4m3}"
NUM_ITERS="${NUM_ITERS:-30}"
DRY_RUN_ITERS="${DRY_RUN_ITERS:-5}"
FIXED_SEQ_LEN="${FIXED_SEQ_LEN:-1}"
WORKSPACE_MIB="${WORKSPACE_MIB:-256}"
FLASHINFER_LOG_LEVEL="${FLASHINFER_LOG_LEVEL:-INFO}"

cd "${REPO_ROOT}"

export FLASHINFER_LOG_LEVEL
if [[ -n "${LUT_CONFIG_ENV_VAL:-}" ]]; then
  export FLASHINFER_TRTLLM_GEN_MLA_BATCH_SPLIT_LUT="${LUT_CONFIG_ENV_VAL}"
fi

EXTRA_ARGS=()
if [[ "${FIXED_SEQ_LEN}" == "1" ]]; then
  EXTRA_ARGS+=(--fixed_seq_len)
fi

echo "TRTLLM-GEN MLA batch-split evaluation config:"
echo "  BACKEND=${BACKEND}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  Q_LEN_PER_REQUEST=${Q_LEN_PER_REQUEST}"
echo "  SEQ_LEN=${SEQ_LEN}"
echo "  FIXED_SEQ_LEN=${FIXED_SEQ_LEN}"
echo "  PAGE_SIZE=${PAGE_SIZE}"
echo "  NUM_QO_HEADS=${NUM_QO_HEADS}"
echo "  HEAD_DIM_CKV=${HEAD_DIM_CKV}"
echo "  HEAD_DIM_KPE=${HEAD_DIM_KPE}"
echo "  QK_NOPE_HEAD_DIM=${QK_NOPE_HEAD_DIM}"
echo "  DTYPE=${DTYPE}"
echo "  WORKSPACE_MIB=${WORKSPACE_MIB}"
echo "  FLASHINFER_LOG_LEVEL=${FLASHINFER_LOG_LEVEL}"
echo "  FLASHINFER_TRTLLM_GEN_MLA_BATCH_SPLIT_LUT=${FLASHINFER_TRTLLM_GEN_MLA_BATCH_SPLIT_LUT:-}"

python benchmarks/evaluate_trtllm_gen_mla_batch_split.py \
  --backend "${BACKEND}" \
  --batch_size "${BATCH_SIZE}" \
  --q_len_per_request "${Q_LEN_PER_REQUEST}" \
  --seq_len "${SEQ_LEN}" \
  --page_size "${PAGE_SIZE}" \
  --num_qo_heads "${NUM_QO_HEADS}" \
  --head_dim_ckv "${HEAD_DIM_CKV}" \
  --head_dim_kpe "${HEAD_DIM_KPE}" \
  --qk_nope_head_dim "${QK_NOPE_HEAD_DIM}" \
  --dtype "${DTYPE}" \
  --num_iters "${NUM_ITERS}" \
  --dry_run_iters "${DRY_RUN_ITERS}" \
  --workspace_mib "${WORKSPACE_MIB}" \
  "${EXTRA_ARGS[@]}"
