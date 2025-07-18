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

#include <algorithm>
#include <cmath>
#include <flashinfer/semaphore_utils.cuh>
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/trtllm/fmha/gen_kernel_launcher.cuh>
#include <flashinfer/utils.cuh>
#include <iostream>
#include <optional>

namespace flashinfer {
template <typename T, Data_type CACHE_T>
void trtllm_paged_attention_decode_launcher(
    at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache, at::Tensor& workspace_buffer,
    int64_t num_kv_heads, at::Tensor& block_tables, at::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, double bmm1_scale, double bmm2_scale, int64_t window_left,
    int64_t sum_seq_q, int64_t sum_seq_kv) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(-1);

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  uint32_t tokens_per_page = block_size;

  uint32_t num_k_heads = num_kv_heads;
  uint32_t num_v_heads = num_k_heads;
  if (num_heads % num_k_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_heads must be a multiple of num_k_heads, got num_k_heads: " << num_k_heads
            << "and num_heads: " << num_heads;
    FLASHINFER_ERROR(err_msg.str());
  }
  auto batch_size = num_seqs;

  int const beam_width = num_seqs / batch_size;  // always 1

  auto q_heads = reinterpret_cast<T*>(query.data_ptr());
  auto output_ptr = reinterpret_cast<T*>(out.data_ptr());

  auto cache_heads = reinterpret_cast<void*>(key_value_cache.data_ptr());

  auto io_type = TypeToDataType<T>::value;

  bool use_multi_block = true;
  auto q_data_type =
      key_value_cache.dtype() == at::ScalarType::Float8_e4m3fn ? DATA_TYPE_E4M3 : io_type;
  auto output_dtype = io_type;
  static auto fmha_runner = TllmGenFmhaRunner(q_data_type, CACHE_T, io_type);

  TllmGenFmhaRunnerParams runner_params;
  memset(&runner_params, 0, sizeof(runner_params));

  runner_params.mMaskType = TrtllmGenAttentionMaskType::Dense;
  runner_params.mKernelType = FmhaKernelType::Generation;
  runner_params.mTileScheduler =
      use_multi_block ? TileScheduler::Static : TileScheduler::Persistent;
  runner_params.mMultiCtasKvMode = use_multi_block;

  runner_params.qPtr = q_heads;

  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.kvPtr = cache_heads;
  runner_params.kvPageIdxPtr = block_tables.data_ptr<KVCachePageIndex>();
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = tokens_per_page;

  runner_params.mSumOfSeqLensQ = 1;
  runner_params.mSumOfSeqLensKv = sum_seq_kv;
  // runner_params.cumSeqLensQPtr = cum_seq_lens_q.data_ptr<int>();
  // runner_params.cumSeqLensKvPtr = cum_seq_lens_kv.data_ptr<int>();

  // num_kv_heads should be enough, but num_heads for safty at long seq len.
  // Round up num_semaphores to a multiple of 8 to satisfy the 16B alignment requirement for
  // multiCtasKvScratchPtr.
  size_t num_semaphores = round_up(batch_size * num_heads, 8);

  runner_params.multiCtasKvScratchPtr = reinterpret_cast<void*>(
      static_cast<char*>(workspace_buffer.data_ptr()) + num_semaphores * sizeof(uint32_t));
  runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_buffer.data_ptr());

  zero_gmem_semaphore_launcher(runner_params.multiCtasKvCounterPtr, num_semaphores,
                               /*enable_pdl=*/true, stream);

  if (head_size != 64 && head_size != 128 && head_size != 192 && head_size != 256) {
    std::ostringstream err_msg;
    err_msg << "head_size " << head_size << " is not supported!";
    FLASHINFER_ERROR(err_msg.str());
  }

  runner_params.seqLensKvPtr = reinterpret_cast<int const*>(seq_lens.data_ptr<int>());

  runner_params.oPtr = output_ptr;
  runner_params.mHeadDimQk = head_size;
  runner_params.mHeadDimV = head_size;
  runner_params.mNumHeadsQ = num_heads;
  runner_params.mNumHeadsKv = num_k_heads;
  runner_params.mNumHeadsQPerKv = num_heads / num_k_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenQ = 1;
  runner_params.mMaxSeqLenKv = max_seq_len;
  runner_params.mSumOfSeqLensQ = int(batch_size * runner_params.mMaxSeqLenQ);
  runner_params.mScaleQ = 1.0;

  // Set the chunked attention size and sliding window size to INT_MAX to disable them when checking
  // if the kernel is supported.
  runner_params.mChunkedAttentionSize = INT_MAX;
  runner_params.mAttentionWindowSize = window_left == -1 ? INT_MAX : window_left + 1;
  auto [foundKernels, kinfo] = fmha_runner.isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    err_msg << "Missing TRTLLM-GEN decode kernel:" << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }
  // else {
  //   std::cout << "Found TRTLLM-GEN decode kernel" << kinfo << std::endl;
  // }

  runner_params.mMultiProcessorCount = getMultiProcessorCount();
  auto const [free_memory, total_memory] = getDeviceMemoryInfo(false);
  int max_head_dim_kv = head_size;

  runner_params.mNumPagesInMemPool =
      total_memory / (runner_params.mNumHeadsKv * runner_params.mNumTokensPerPage *
                      max_head_dim_kv * get_size_in_bytes(CACHE_T));

  runner_params.stream = stream;

  runner_params.outputScale = bmm2_scale;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;

  fmha_runner.run(runner_params);
}

void trtllm_paged_attention_decode(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                                   at::Tensor& workspace_buffer, int64_t num_kv_heads,
                                   at::Tensor& block_tables, at::Tensor& seq_lens,
                                   int64_t block_size, int64_t max_seq_len, double bmm1_scale,
                                   double bmm2_scale, int64_t window_left, int64_t sum_seq_q,
                                   int64_t sum_seq_kv) {
  if (query.dtype() == at::ScalarType::Half && key_value_cache.dtype() == at::ScalarType::Half) {
    trtllm_paged_attention_decode_launcher<half, Data_type::DATA_TYPE_FP16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv);
  } else if (query.dtype() == at::ScalarType::BFloat16 &&
             key_value_cache.dtype() == at::ScalarType::Half) {
    trtllm_paged_attention_decode_launcher<__nv_bfloat16, Data_type::DATA_TYPE_FP16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv);
  } else if (query.dtype() == at::ScalarType::Half &&
             key_value_cache.dtype() == at::ScalarType::BFloat16) {
    trtllm_paged_attention_decode_launcher<half, Data_type::DATA_TYPE_BF16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv);
  } else if (query.dtype() == at::ScalarType::BFloat16 &&
             key_value_cache.dtype() == at::ScalarType::BFloat16) {
    trtllm_paged_attention_decode_launcher<__nv_bfloat16, Data_type::DATA_TYPE_BF16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv);
  } else if (query.dtype() == at::ScalarType::Half &&
             key_value_cache.dtype() == at::ScalarType::Float8_e4m3fn) {
    trtllm_paged_attention_decode_launcher<half, Data_type::DATA_TYPE_E4M3>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv);
  } else if (query.dtype() == at::ScalarType::BFloat16 &&
             key_value_cache.dtype() == at::ScalarType::Float8_e4m3fn) {
    trtllm_paged_attention_decode_launcher<__nv_bfloat16, Data_type::DATA_TYPE_E4M3>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv);
  } else {
    TORCH_CHECK(false, "Unsupported data type combination of query and kv cache: ", query.dtype(),
                " and ", key_value_cache.dtype());
  }
}

template <typename T, Data_type CACHE_T>
void trtllm_paged_attention_context_launcher(
    at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache, at::Tensor& workspace_buffer,
    int64_t num_kv_heads, at::Tensor& block_tables, at::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, double bmm1_scale, double bmm2_scale, int64_t batch_size,
    int64_t window_left, int64_t sum_seq_q, int64_t sum_seq_kv, at::Tensor& cum_seq_lens_q,
    at::Tensor& cum_seq_lens_kv) {
  int num_seqs = query.size(0) / batch_size;
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(-1);

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  uint32_t tokens_per_page = block_size;

  uint32_t num_k_heads = num_kv_heads;
  uint32_t num_v_heads = num_k_heads;

  auto q_heads = reinterpret_cast<T*>(query.data_ptr());
  auto output_ptr = reinterpret_cast<T*>(out.data_ptr());

  auto cache_heads = reinterpret_cast<void*>(key_value_cache.data_ptr());

  auto io_type = TypeToDataType<T>::value;

  auto q_data_type =
      key_value_cache.dtype() == at::ScalarType::Float8_e4m3fn ? DATA_TYPE_E4M3 : io_type;
  auto output_dtype = io_type;

  static auto fmha_runner = TllmGenFmhaRunner(q_data_type, CACHE_T, io_type);

  TllmGenFmhaRunnerParams runner_params;
  memset(&runner_params, 0, sizeof(runner_params));

  runner_params.mMaskType = TrtllmGenAttentionMaskType::Causal;
  runner_params.mKernelType = FmhaKernelType::Context;

  // Always use persistent scheduler for better performance.
  runner_params.mTileScheduler = TileScheduler::Persistent;
  runner_params.mMultiCtasKvMode = false;

  runner_params.qPtr = q_heads;

  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.kvPtr = cache_heads;
  runner_params.kvPageIdxPtr = block_tables.data_ptr<KVCachePageIndex>();
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = tokens_per_page;

  runner_params.seqLensKvPtr = reinterpret_cast<int const*>(seq_lens.data_ptr<int>());

  runner_params.oPtr = output_ptr;
  runner_params.mHeadDimQk = head_size;
  runner_params.mHeadDimV = head_size;
  runner_params.mNumHeadsQ = num_heads;
  runner_params.mNumHeadsKv = num_k_heads;
  runner_params.mNumHeadsQPerKv = num_heads / num_k_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenQ = query.size(0);
  runner_params.mMaxSeqLenKv = max_seq_len;

  runner_params.mSumOfSeqLensQ = sum_seq_q;
  runner_params.mSumOfSeqLensKv = sum_seq_kv;
  runner_params.cumSeqLensQPtr = cum_seq_lens_q.data_ptr<int>();
  runner_params.cumSeqLensKvPtr = cum_seq_lens_kv.data_ptr<int>();

  runner_params.mScaleQ = 1.0;
  // Set the chunked attention size and sliding window size to INT_MAX to disable them when checking
  // if
  // the kernel is supported.
  runner_params.mChunkedAttentionSize = INT_MAX;
  runner_params.mAttentionWindowSize = window_left == -1 ? INT_MAX : window_left + 1;

  runner_params.mMultiProcessorCount = getMultiProcessorCount();
  auto const [free_memory, total_memory] = getDeviceMemoryInfo(false);
  auto [foundKernels, kinfo] = fmha_runner.isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    TllmGenSelectKernelParams select_kernel_params{runner_params};
    err_msg << "Missing TRTLLM-GEN context kernel:" << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }
  int max_head_dim_kv = head_size;

  runner_params.mNumPagesInMemPool =
      total_memory / (runner_params.mNumHeadsKv * runner_params.mNumTokensPerPage *
                      max_head_dim_kv * get_size_in_bytes(CACHE_T));

  runner_params.outputScale = bmm2_scale;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;

  runner_params.stream = stream;
  fmha_runner.run(runner_params);
}

void trtllm_paged_attention_context(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                                    at::Tensor& workspace_buffer, int64_t num_kv_heads,
                                    at::Tensor& block_tables, at::Tensor& seq_lens,
                                    int64_t block_size, int64_t max_seq_len, double bmm1_scale,
                                    double bmm2_scale, int64_t batch_size, int64_t window_left,
                                    int64_t sum_seq_q, int64_t sum_seq_kv,
                                    at::Tensor& cum_seq_lens_q, at::Tensor& cum_seq_lens_kv) {
  if (query.dtype() == at::ScalarType::Half && key_value_cache.dtype() == at::ScalarType::Half) {
    trtllm_paged_attention_context_launcher<half, Data_type::DATA_TYPE_FP16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, batch_size, window_left, sum_seq_q,
        sum_seq_kv, cum_seq_lens_q, cum_seq_lens_kv);
  } else if (query.dtype() == at::ScalarType::BFloat16 &&
             key_value_cache.dtype() == at::ScalarType::Half) {
    trtllm_paged_attention_context_launcher<__nv_bfloat16, Data_type::DATA_TYPE_FP16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, batch_size, window_left, sum_seq_q,
        sum_seq_kv, cum_seq_lens_q, cum_seq_lens_kv);
  } else if (query.dtype() == at::ScalarType::Half &&
             key_value_cache.dtype() == at::ScalarType::BFloat16) {
    trtllm_paged_attention_context_launcher<half, Data_type::DATA_TYPE_BF16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, batch_size, window_left, sum_seq_q,
        sum_seq_kv, cum_seq_lens_q, cum_seq_lens_kv);
  } else if (query.dtype() == at::ScalarType::BFloat16 &&
             key_value_cache.dtype() == at::ScalarType::BFloat16) {
    trtllm_paged_attention_context_launcher<__nv_bfloat16, Data_type::DATA_TYPE_BF16>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, batch_size, window_left, sum_seq_q,
        sum_seq_kv, cum_seq_lens_q, cum_seq_lens_kv);
  } else if (query.dtype() == at::ScalarType::Half &&
             key_value_cache.dtype() == at::ScalarType::Float8_e4m3fn) {
    trtllm_paged_attention_context_launcher<half, Data_type::DATA_TYPE_E4M3>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, batch_size, window_left, sum_seq_q,
        sum_seq_kv, cum_seq_lens_q, cum_seq_lens_kv);
  } else if (query.dtype() == at::ScalarType::BFloat16 &&
             key_value_cache.dtype() == at::ScalarType::Float8_e4m3fn) {
    trtllm_paged_attention_context_launcher<__nv_bfloat16, Data_type::DATA_TYPE_E4M3>(
        out, query, key_value_cache, workspace_buffer, num_kv_heads, block_tables, seq_lens,
        block_size, max_seq_len, bmm1_scale, bmm2_scale, batch_size, window_left, sum_seq_q,
        sum_seq_kv, cum_seq_lens_q, cum_seq_lens_kv);
  } else {
    TORCH_CHECK(false, "Unsupported data type combination of query and kv cache: ", query.dtype(),
                " and ", key_value_cache.dtype());
  }
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_paged_attention_decode", trtllm_paged_attention_decode);
  m.def("trtllm_paged_attention_context", trtllm_paged_attention_context);
}

}  // namespace flashinfer
