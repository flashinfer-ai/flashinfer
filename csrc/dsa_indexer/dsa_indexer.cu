// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project


#include <cuda_runtime.h>

#include <flashinfer/dsa_indexer/dsa_indexer.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void dsa_indexer_seed_prep(TensorView slog, int64_t num_buckets, int64_t topk,
                           int64_t cand_cap, int64_t emit_limit, double headroom,
                           int64_t probe_stride_tok, int64_t hist_stride, TensorView origin,
                           TensorView inv_delta, TensorView th_bucket, TensorView bcount,
                           TensorView cand_val, TensorView cand_idx, TensorView cand_cnt) {
  CHECK_INPUT(slog);
  CHECK_DIM(2, slog);
  const int Q = static_cast<int>(slog.size(0));
  const int head = static_cast<int>(slog.size(1));
  const int NB = static_cast<int>(num_buckets);
  const int K = static_cast<int>(topk);
  const int cap = static_cast<int>(cand_cap);
  TVM_FFI_ICHECK(NB >= 2 && NB <= 4096) << "num_buckets out of range";
  TVM_FFI_ICHECK(K >= 1 && cap >= K) << "need cap >= topk >= 1";

  cudaSetDevice(slog.device().device_id);
  cudaStream_t stream = get_stream(slog.device());
  const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
  if (seed_smem > 48 * 1024) {
    static bool attr_set = false;
    if (!attr_set) {
      cudaFuncSetAttribute((void*)seed_prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           4 * 4096 * static_cast<int>(sizeof(int)));
      attr_set = true;
    }
  }
  const int emit_lim = emit_limit == 0 ? 0 : (emit_limit > 0 ? static_cast<int>(emit_limit) : head);
  const int pst = static_cast<int>(probe_stride_tok);
  const int hst = hist_stride > 1 ? static_cast<int>(hist_stride) : 1;

  seed_prep_kernel<<<Q, 1024, seed_smem, stream>>>(
      static_cast<float*>(slog.data_ptr()), static_cast<int>(slog.stride(0)), head, NB, K, cap,
      emit_lim, pst, hst, static_cast<float>(headroom),
      static_cast<float*>(origin.data_ptr()), static_cast<float*>(inv_delta.data_ptr()),
      static_cast<int32_t*>(th_bucket.data_ptr()),
      static_cast<int32_t*>(bcount.data_ptr()), static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_idx.data_ptr()), static_cast<int32_t*>(cand_cnt.data_ptr()));
}

void dsa_indexer_scan(TensorView q, TensorView kv, TensorView kv_scales, TensorView weights,
                      TensorView cu_start, TensorView cu_end, TensorView origin,
                      TensorView inv_delta, TensorView th_bucket, TensorView cand_val,
                      TensorView cand_idx, TensorView cand_cnt, TensorView bcount,
                      int64_t num_buckets, int64_t topk, int64_t refresh_every,
                      int64_t num_kv_splits_override, int64_t probe_group, int64_t probe_add_max) {
  CHECK_INPUT(q);
  CHECK_INPUT(kv);
  CHECK_INPUT(kv_scales);
  CHECK_INPUT(weights);
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  const int cand_cap = static_cast<int>(cand_val.size(1));
  const int num_buckets_i = static_cast<int>(num_buckets);
  const int topk_i = static_cast<int>(topk);
  TVM_FFI_ICHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM)
      << "only GLM DSA H=32 D=128 is supported";
  const bool external_refresh = (refresh_every < 0);
  const int refresh_every_i = external_refresh ? 0x7fffffff : static_cast<int>(refresh_every);

  cudaSetDevice(q.device().device_id);
  cudaStream_t stream = get_stream(q.device());
  const int esz_fp8 = 1, esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM,
                      seq_len * NUM_HEADS, HEAD_DIM, BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
  auto tm_kv = make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM, seq_len_kv,
                       HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
  auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32, ks_aligned, 1,
                       BLOCK_KV, 1, 0, 0);
  auto tm_w = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32, NUM_HEADS,
                      seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

  const int smem = compute_smem_bytes();
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV,
                                                    NUM_Q_STAGES, NUM_KV_STAGES, NUM_SMS,
                                                    SPEC_THREADS, MATH_THREADS>;
  cudaFuncSetAttribute((void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  const int total_kv_blocks = (seq_len_kv + BLOCK_KV - 1) / BLOCK_KV;
  int num_kv_splits;
  if (num_kv_splits_override > 0) {
    num_kv_splits = static_cast<int>(num_kv_splits_override);
  } else {
    constexpr int kWaves = 4;
    const int qb = num_q_blocks > 0 ? num_q_blocks : 1;
    num_kv_splits = (kWaves * NUM_SMS + qb - 1) / qb;
    const int max_useful_splits = total_kv_blocks > 0 ? (total_kv_blocks + 1) / 2 : 1;
    if (num_kv_splits > max_useful_splits) num_kv_splits = max_useful_splits;
  }
  if (num_kv_splits < 1) num_kv_splits = 1;
  if (num_kv_splits > total_kv_blocks) num_kv_splits = total_kv_blocks > 0 ? total_kv_blocks : 1;
  int grid_q = num_q_blocks;
  dim3 grid((unsigned)grid_q, (unsigned)num_kv_splits, 1);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      (uint32_t)seq_len, (uint32_t)seq_len_kv, static_cast<uint32_t*>(cu_start.data_ptr()),
      static_cast<uint32_t*>(cu_end.data_ptr()), static_cast<float*>(origin.data_ptr()),
      static_cast<float*>(inv_delta.data_ptr()), static_cast<int32_t*>(th_bucket.data_ptr()),
      static_cast<int32_t*>(bcount.data_ptr()), (uint32_t)num_buckets_i, (uint32_t)topk_i,
      (uint32_t)refresh_every_i, (uint32_t)num_kv_splits, (uint32_t)probe_group,
      probe_group > 0 ? (((1ULL << 42) + (uint64_t)probe_group - 1) / (uint64_t)probe_group) : 0ULL,
      (uint32_t)probe_add_max, static_cast<float*>(cand_val.data_ptr()),
      static_cast<int32_t*>(cand_idx.data_ptr()), static_cast<int32_t*>(cand_cnt.data_ptr()),
      (uint32_t)cand_cap, tm_q, tm_kv, tm_ks, tm_w);

  if (external_refresh) {
    int block = 128;
    int grid_r = (seq_len + block - 1) / block;
    refresh_threshold_from_bcount_kernel<<<grid_r, block, 0, stream>>>(
        static_cast<int32_t*>(th_bucket.data_ptr()), static_cast<int32_t*>(bcount.data_ptr()),
        seq_len, num_buckets_i, topk_i);
  }
}

void dsa_indexer_select(TensorView cand_val, TensorView cand_idx, TensorView cand_cnt,
                        TensorView origin, TensorView inv_delta, TensorView th_bucket,
                        int64_t num_buckets, int64_t topk, int64_t probe_group,
                        int64_t probe_add_max, Optional<TensorView> seed_base, TensorView out_val,
                        TensorView out_idx) {
  CHECK_INPUT(cand_val);
  CHECK_INPUT(cand_idx);
  CHECK_DIM(2, cand_val);
  const int R = static_cast<int>(cand_val.size(0));
  const int CAP = static_cast<int>(cand_val.size(1));
  const int K = static_cast<int>(topk);
  const int NB = static_cast<int>(num_buckets);
  TVM_FFI_ICHECK(K >= 1 && K <= CAP) << "K must be in [1, CAP]";
  TVM_FFI_ICHECK(NB >= 2 && NB <= 4096) << "num_buckets out of range";
  if (probe_group > 0) {
    TVM_FFI_ICHECK(seed_base.has_value()) << "seed_base [R] int32 required with probe_group";
  }
  cudaSetDevice(cand_val.device().device_id);
  cudaStream_t stream = get_stream(cand_val.device());
  compact_topk_min_thr_litetopk_kernel<<<R, 256, 0, stream>>>(
      static_cast<float*>(cand_val.data_ptr()), static_cast<int32_t*>(cand_idx.data_ptr()),
      static_cast<int32_t*>(cand_cnt.data_ptr()), static_cast<float*>(origin.data_ptr()),
      static_cast<float*>(inv_delta.data_ptr()), static_cast<int32_t*>(th_bucket.data_ptr()), R,
      CAP, K, NB, static_cast<float*>(out_val.data_ptr()), static_cast<int32_t*>(out_idx.data_ptr()),
      (uint32_t)probe_group,
      probe_group > 0 ? (((1ULL << 42) + (uint64_t)probe_group - 1) / (uint64_t)probe_group) : 0ULL,
      (uint32_t)probe_add_max,
      probe_group > 0 ? static_cast<int32_t*>(seed_base.value().data_ptr()) : nullptr);
}
