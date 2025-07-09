/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <flashinfer/exception.h>
#include <nvrtc.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <flashinfer/semaphore_utils.cuh>
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/trtllm/fmha/gen_kernel_launcher.cuh>
#include <iostream>

// NOTE(Yingyi):
// dummy sliding window attention
// quantization not supported
namespace flashinfer {
template <Data_type CACHE_T>
void trtllm_paged_attention_mla_launcher(
    at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache, at::Tensor& workspace_buffer,
    double scale, at::Tensor& block_tables, at::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, int64_t qk_nope_head_dim, int64_t kv_lora_rank, int64_t qk_rope_head_dim,
    std::optional<int64_t> acc_q_len, std::optional<int64_t> max_attention_window_size,
    std::optional<int64_t> cyclic_attention_window_size) {
  int const num_seqs = query.size(0);
  int const batch_size = num_seqs;
  int const num_q_heads = query.size(1);
  int const num_kv_heads = 1;
  int head_size = query.size(2);
  int const beam_width = 1;                        // NOTE: beam_width always 1
  int const batch_beam = beam_width * batch_size;  // NOTE: batch_beam = batch_size
  int const max_num_blocks_per_seq = block_tables.size(-1);

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  uint32_t tokens_per_page = block_size;

  // todo(Yingyi): use multi_block mode always true??
  bool use_multi_block = true;
  static auto fmha_runner = TllmGenFmhaRunner(CACHE_T, CACHE_T, DATA_TYPE_BF16);

  TllmGenFmhaRunnerParams runner_params;
  memset(&runner_params, 0, sizeof(runner_params));

  // Parameters to select kernels.
  runner_params.mMaskType = TrtllmGenAttentionMaskType::Dense;
  runner_params.mKernelType = FmhaKernelType::Generation;
  // Note that the tileScheduler and multiCtasKvMode will be automatically tuned when using
  // multi_block mode. Otherwise, always enable the persistent scheduler for better performance.
  runner_params.mTileScheduler =
      use_multi_block ? TileScheduler::Static : TileScheduler::Persistent;
  runner_params.mMultiCtasKvMode = use_multi_block;

  // Q buffer.
  // NOTE(Yingyi): no additional quantization input data field here
  runner_params.qPtr = query.data_ptr();

  // KV buffer
  // Paged KV
  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.kvPtr = key_value_cache.data_ptr();
  runner_params.kvPageIdxPtr = block_tables.data_ptr<KVCachePageIndex>();
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = tokens_per_page;

  // num_kv_heads should be enough, but num_heads for safty at long seq len.
  size_t num_semaphores = batch_size * num_q_heads;

  // The partial buffers' pointers when the multiCtasKv mode is enabled.
  runner_params.multiCtasKvScratchPtr = reinterpret_cast<void*>(
      static_cast<char*>(workspace_buffer.data_ptr()) + num_semaphores * sizeof(uint32_t));
  runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_buffer.data_ptr());

  // The sequence lengths for K/V.
  runner_params.seqLensKvPtr = reinterpret_cast<int const*>(seq_lens.data_ptr<int>());

  runner_params.oPtr = out.data_ptr();
  // NOTE(yingyi): quantization is not supported for now
  runner_params.oSfPtr = nullptr;

  // MLA uses different head dimensions for Qk and V.
  runner_params.mHeadDimQk = num_q_heads;
  runner_params.mHeadDimV = kv_lora_rank;

  // NOTE: MLA use kv_heads = 1
  runner_params.mNumHeadsQ = num_q_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_q_heads / num_kv_heads;

  // NOTE: beam_width = 1
  runner_params.mBatchSize = batch_size;

  // It is used to construct contiguous kv cache TMA descriptors.
  auto const max_attention_window_size_opt = max_attention_window_size.value_or(max_seq_len);
  runner_params.mMaxSeqLenCacheKv = max_attention_window_size_opt;

  // This should be set to numDraftTokens + 1.
  auto const acc_q_len_opt = acc_q_len.value_or(batch_beam);
  runner_params.mMaxSeqLenQ = acc_q_len_opt / batch_beam;  // should be 1 if acc_q_len not provided
  runner_params.mMaxSeqLenKv = max_seq_len;
  runner_params.mSumOfSeqLensQ = int(batch_beam * runner_params.mMaxSeqLenQ);
  // Not used in the generation kernels as contiguous_kv or paged_kv layouts are used.
  runner_params.mSumOfSeqLensKv = int(batch_beam * runner_params.mMaxSeqLenKv);

  // The attention window size.
  // NOTE(Yingyi): for sliding window attention, temp to the fixed INT_MAX
  runner_params.mAttentionWindowSize = INT_MAX;
  // The chunked attention size.
  runner_params.mChunkedAttentionSize = INT_MAX;

  // The scaleQ that will be applied to the BMM1 output.
  // NOTE(Yingyi): set scale to be a api param, default 1.0
  // Q_SCALE & KV_SCALE not supported for now
  runner_params.mScaleQ = scale * sqrt((float)(qk_nope_head_dim + qk_rope_head_dim)) /
                          sqrtf((float)(kv_lora_rank + qk_rope_head_dim));

  // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds
  // access.
  // todo(Yingyi): check the memory pool size, bounded current kv cache pool passed by params
  // kv_cache
  // runner_params.mNumPagesInMemPool = INT_MAX;
  auto const [free_memory, total_memory] = getDeviceMemoryInfo(false);
  int max_head_dim_kv = head_size;
  runner_params.mNumPagesInMemPool =
      total_memory / (runner_params.mNumHeadsKv * runner_params.mNumTokensPerPage *
                      max_head_dim_kv * get_size_in_bytes(CACHE_T));

  runner_params.mMultiProcessorCount = getMultiProcessorCount();
  runner_params.stream = stream;
  // NOTE (Yingyi): quantization, not supported for now
  runner_params.mSfStartTokenIdx = 0;

  zero_gmem_semaphore_launcher(runner_params.multiCtasKvCounterPtr, num_semaphores,
                               /*enable_pdl=*/true, stream);

  fmha_runner.run(runner_params);
}

#define CALL_GEN_LAUNCHER(CACHE_T_ENUM)                                                         \
  trtllm_paged_attention_mla_launcher<CACHE_T_ENUM>(                                            \
      out, query, key_value_cache, workspace_buffer, scale, block_tables, seq_lens, block_size, \
      max_seq_len, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, acc_q_len,                 \
      max_attention_window_size, cyclic_attention_window_size);

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t>
#define DISPATCH_BY_QKV_DTYPE(Q_DTYPE, KV_DTYPE, FN)                                               \
  FLASHINFER_CHECK(Q_DTYPE == KV_DTYPE,                                                            \
                   "Q_DTYPE must be the same as KV_DTYPE. Hybrid type is not supported for now."); \
  if (Q_DTYPE == at::ScalarType::Float8_e4m3fn) {                                                  \
    FN(Data_type::DATA_TYPE_E4M3);                                                                 \
  } else if (Q_DTYPE == at::ScalarType::BFloat16) {                                                \
    FN(Data_type::DATA_TYPE_BF16);                                                                 \
  } else {                                                                                         \
    TORCH_CHECK(false, "Unsupported input type of QKV type: ", Q_DTYPE);                           \
  }

void trtllm_paged_attention_mla(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                                at::Tensor& workspace_buffer, double scale,
                                at::Tensor& block_tables, at::Tensor& seq_lens, int64_t block_size,
                                int64_t max_seq_len, int64_t qk_nope_head_dim, int64_t kv_lora_rank,
                                int64_t qk_rope_head_dim, std::optional<int64_t> acc_q_len,
                                std::optional<int64_t> max_attention_window_size,
                                std::optional<int64_t> cyclic_attention_window_size) {
  DISPATCH_BY_QKV_DTYPE(query.dtype(), key_value_cache.dtype(),
                        CALL_GEN_LAUNCHER);  // hybrid attention is not supported for now
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_paged_attention_mla", trtllm_paged_attention_mla);
}

}  // namespace flashinfer
