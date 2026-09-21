/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda.h>

#include "dsa_indexer_select.cuh"
#include "tvm_ffi_utils.h"

namespace {

constexpr int kNumHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kBlockQ = 4;
constexpr int kBlockKV = 256;
constexpr int kNumQStages = 1;
constexpr int kNumKVStages = 4;
constexpr int kSpecThreads = 128;
constexpr int kMathThreads = 256;
constexpr int kNumBuckets = 256;
constexpr int kSeedThreads = 256;

CUtensorMap make_2d(void* ptr, CUtensorMapDataType dtype, int elem_size, int gmem_inner,
                    int gmem_outer, int smem_inner, int smem_outer, int64_t gmem_outer_stride,
                    int swizzle_bytes) {
  if (swizzle_bytes != 0) smem_inner = swizzle_bytes / elem_size;
  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(gmem_inner),
                               static_cast<cuuint64_t>(gmem_outer)};
  const cuuint32_t sdims[2] = {static_cast<cuuint32_t>(smem_inner),
                               static_cast<cuuint32_t>(smem_outer)};
  const cuuint64_t gstrides[1] = {static_cast<cuuint64_t>(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  const CUtensorMapSwizzle swizzle = swizzle_bytes == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                                     : swizzle_bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                     : swizzle_bytes == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                                           : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMap map;
  const CUresult r = cuTensorMapEncodeTiled(
      &map, dtype, 2, ptr, gdims, gstrides, sdims, estrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK_EQ(r, CUDA_SUCCESS) << "cuTensorMapEncodeTiled failed";
  return map;
}

// Must match the shared-memory carve-up at the top of sm100_dsa_litetopk.
constexpr int scan_smem_bytes() {
  constexpr int kMathWarps = kMathThreads / 32;
  constexpr int q = kBlockQ * kNumHeads * kHeadDim;
  constexpr int weights = kBlockQ * kNumHeads * 4;
  constexpr int kv = kBlockKV * kHeadDim;
  constexpr int kv_scales = (kBlockKV * 4 + 511) / 512 * 512;
  constexpr int barriers =
      (kNumQStages * 2 + kNumKVStages * 2 + (kMathThreads / 128) * dsa_litetopk::kUmmaStages * 2) *
      8;
  constexpr int slots = 4 * 4;  // TMEM pointer and daemon mailbox words
  constexpr int emit = kMathWarps * kBlockQ * (4 + dsa_litetopk::kEmitLaneSlots * 32 * 4);
  constexpr int hist = kBlockQ * kNumBuckets * 4;
  constexpr int ring = 2 * kMathWarps * kBlockQ * 4 + 3 * kBlockQ * 4 + kMathWarps * kBlockQ * 32;
  return kNumQStages * (q + weights) + kNumKVStages * (kv + kv_scales) + barriers + slots + emit +
         hist + ring;
}

}  // namespace

// prefix_logits covers kv[:P]; the scan covers [cu_start[i], cu_end[i]) with
// cu_start[i] == P. origin..cand_cnt are per-call scratch; out and status are the outputs.
void dsa_indexer_topk(TensorView q, TensorView kv, TensorView kv_scales, TensorView weights,
                      TensorView prefix_logits, TensorView cu_start, TensorView cu_end,
                      TensorView origin, TensorView inv_delta, TensorView th_bucket,
                      TensorView bcount, TensorView cand_val, TensorView cand_idx,
                      TensorView cand_cnt, TensorView out, TensorView status) {
  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(kv, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(kv_scales, dl_float32);
  CHECK_INPUT_AND_TYPE(weights, dl_float32);
  CHECK_INPUT_AND_TYPE(cu_start, dl_int32);
  CHECK_INPUT_AND_TYPE(cu_end, dl_int32);
  CHECK_INPUT_AND_TYPE(out, dl_int32);
  CHECK_INPUT_AND_TYPE(status, dl_int32);
  CHECK_CUDA(prefix_logits);
  CHECK_INPUT_TYPE(prefix_logits, dl_float32);
  CHECK_INPUT(cand_val);
  CHECK_INPUT(cand_idx);
  CHECK_DIM(3, q);
  CHECK_DIM(2, kv);
  CHECK_DIM(2, weights);
  CHECK_DIM(2, prefix_logits);
  CHECK_DIM(2, cand_val);
  CHECK_DIM(2, out);
  CHECK_DIM(1, cu_start);
  CHECK_DIM(1, cu_end);
  CHECK_DIM(1, status);
  for (const TensorView* t :
       {&kv, &kv_scales, &weights, &prefix_logits, &cu_start, &cu_end, &origin, &inv_delta,
        &th_bucket, &bcount, &cand_val, &cand_idx, &cand_cnt, &out, &status}) {
    CHECK_DEVICE((*t), q);
  }
  const int num_q = static_cast<int>(q.size(0));
  const int seq_kv = static_cast<int>(kv.size(0));
  const int prefix_len = static_cast<int>(prefix_logits.size(1));
  const int64_t prefix_stride = prefix_logits.stride(0);
  const int cand_cap = static_cast<int>(cand_val.size(1));
  const int top_k = static_cast<int>(out.size(1));
  const int ks_aligned = (seq_kv + 3) / 4 * 4;
  TVM_FFI_ICHECK(q.size(1) == kNumHeads && q.size(2) == kHeadDim && kv.size(1) == kHeadDim &&
                 weights.size(1) == kNumHeads)
      << "only 32 heads of dimension 128 are supported";
  for (const TensorView* t :
       {&weights, &prefix_logits, &cu_start, &cu_end, &cand_val, &out, &status}) {
    TVM_FFI_ICHECK_EQ(t->size(0), num_q) << "every per-query tensor needs num_q rows";
  }
  TVM_FFI_ICHECK(seq_kv <= (1 << dsa_litetopk::kCandidateIndexBits)) << "seq_kv must be <= 2^20";
  TVM_FFI_ICHECK_GE(kv_scales.numel(), ks_aligned)
      << "kv_scales needs round_up(seq_kv, 4) elements";
  TVM_FFI_ICHECK(top_k == dsa_litetopk::kTopK) << "only top_k == 2048 is supported";
  TVM_FFI_ICHECK((prefix_len == 12288 || (prefix_len >= top_k && prefix_len <= 8192)) &&
                 prefix_len <= seq_kv)
      << "prefix length must be in [2048, min(8192, seq_kv)] or 12288";
  TVM_FFI_ICHECK(prefix_logits.stride(1) == 1 && prefix_stride % 4 == 0 &&
                 reinterpret_cast<uintptr_t>(prefix_logits.data_ptr()) % 16 == 0)
      << "prefix_logits rows must be contiguous and 16-byte aligned";
  TVM_FFI_ICHECK(cand_cap >= 49152 && cand_cap <= (1 << dsa_litetopk::kCandidateIndexBits))
      << "cand_cap must be in [49152, 2^20]";
  if (num_q == 0) return;

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());
  auto* values = static_cast<uint16_t*>(cand_val.data_ptr());
  auto* indices = static_cast<int32_t*>(cand_idx.data_ptr());
  auto* counts = static_cast<int32_t*>(cand_cnt.data_ptr());
  auto* th = static_cast<int32_t*>(th_bucket.data_ptr());
  auto* hist = static_cast<int32_t*>(bcount.data_ptr());

  // 1. Seed: bucket transform, initial gate and prefix candidates from prefix_logits.
  const auto seed = prefix_len == 12288 ? &dsa_litetopk::seed_prep_kernel<12288, kSeedThreads>
                                        : &dsa_litetopk::seed_prep_kernel<8192, kSeedThreads>;
  seed<<<num_q, kSeedThreads, 4 * kNumBuckets * sizeof(int), stream>>>(
      static_cast<const float*>(prefix_logits.data_ptr()), prefix_stride, prefix_len, kNumBuckets,
      top_k, static_cast<float*>(origin.data_ptr()), static_cast<float*>(inv_delta.data_ptr()), th,
      values, indices, counts, cand_cap, 0, hist);

  // 2. Scan: fp8 MQA scoring of [P, cu_end) with the online gate and candidate emit.
  const auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim,
                            num_q * kNumHeads, kHeadDim, kBlockQ * kNumHeads, kHeadDim, kHeadDim);
  const auto tm_kv = make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, kHeadDim, seq_kv,
                             kHeadDim, kBlockKV, kHeadDim, kHeadDim);
  const auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, ks_aligned,
                             1, kBlockKV, 1, 0, 0);
  const auto tm_w = make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, kNumHeads,
                            num_q, kNumHeads, kBlockQ, kNumHeads, 0);
  const auto scan =
      &dsa_litetopk::sm100_dsa_litetopk<kNumHeads, kHeadDim, kBlockQ, kBlockKV, kNumQStages,
                                        kNumKVStages, kSpecThreads, kMathThreads>;
  constexpr int smem = scan_smem_bytes();
  TVM_FFI_ICHECK_EQ(cudaFuncSetAttribute(scan, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                    cudaSuccess);
  scan<<<(num_q + kBlockQ - 1) / kBlockQ, kSpecThreads + kMathThreads, smem, stream>>>(
      num_q, seq_kv, static_cast<uint32_t*>(cu_start.data_ptr()),
      static_cast<uint32_t*>(cu_end.data_ptr()), static_cast<const float*>(origin.data_ptr()),
      static_cast<const float*>(inv_delta.data_ptr()), th, hist, kNumBuckets, top_k, values,
      indices, counts, cand_cap, tm_q, tm_kv, tm_ks, tm_w);

  // 3. Select: exact top-k over each row's candidates.
  dsa_litetopk::exact_topk_kernel<<<num_q, dsa_litetopk::kThreads, 0, stream>>>(
      values, indices, counts, static_cast<int32_t*>(out.data_ptr()),
      static_cast<int32_t*>(status.data_ptr()), num_q, cand_cap, seq_kv, top_k);
  const cudaError_t err = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsa_indexer_topk, dsa_indexer_topk);
