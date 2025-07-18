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

#include "pytorch_extension_utils.h"

namespace flashinfer {
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void trtllm_paged_attention_decode_launcher(
    DTypeO* out, DTypeQ* query, DTypeKV* key_value_cache, void* workspace_buffer,
    KVCachePageIndex* block_tables, int* seq_lens, int64_t batch_size, int64_t max_seq_len,
    int64_t num_pages, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t page_size, int64_t max_num_blocks_per_seq, double bmm1_scale, double bmm2_scale,
    int64_t window_left, int64_t sum_seq_q, int64_t sum_seq_kv, int64_t sm_count,
    cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: " << num_kv_heads
            << " and num_qo_heads: " << num_qo_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  auto q_data_type = TypeToDataType<DTypeQ>::value;
  auto kv_data_type = TypeToDataType<DTypeKV>::value;
  auto o_data_type = TypeToDataType<DTypeO>::value;

  bool use_multi_block = true;
  static auto fmha_runner = TllmGenFmhaRunner(q_data_type, kv_data_type, o_data_type);

  TllmGenFmhaRunnerParams runner_params;

  runner_params.mMaskType = TrtllmGenAttentionMaskType::Dense;
  runner_params.mKernelType = FmhaKernelType::Generation;
  runner_params.mTileScheduler =
      use_multi_block ? TileScheduler::Static : TileScheduler::Persistent;
  runner_params.mMultiCtasKvMode = use_multi_block;

  runner_params.qPtr = query;

  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.kvPtr = key_value_cache;
  runner_params.kvPageIdxPtr = block_tables;
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = page_size;

  runner_params.mSumOfSeqLensQ = 1;
  runner_params.mSumOfSeqLensKv = sum_seq_kv;
  // runner_params.cumSeqLensQPtr = cum_seq_lens_q.data_ptr<int>();
  // runner_params.cumSeqLensKvPtr = cum_seq_lens_kv.data_ptr<int>();

  // num_kv_heads should be enough, but num_heads for safty at long seq len.
  // Round up num_semaphores to a multiple of 8 to satisfy the 16B alignment requirement for
  // multiCtasKvScratchPtr.
  size_t num_semaphores = round_up(batch_size * num_qo_heads, 8);

  runner_params.multiCtasKvScratchPtr = reinterpret_cast<void*>(
      static_cast<char*>(workspace_buffer) + num_semaphores * sizeof(uint32_t));
  runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_buffer);

  zero_gmem_semaphore_launcher(runner_params.multiCtasKvCounterPtr, num_semaphores,
                               /*enable_pdl=*/true, stream);

  if (head_dim != 64 && head_dim != 128 && head_dim != 192 && head_dim != 256) {
    std::ostringstream err_msg;
    err_msg << "head_dim " << head_dim << " is not supported!";
    FLASHINFER_ERROR(err_msg.str());
  }

  runner_params.seqLensKvPtr = seq_lens;

  runner_params.oPtr = out;
  runner_params.mHeadDimQk = head_dim;
  runner_params.mHeadDimV = head_dim;
  runner_params.mNumHeadsQ = num_qo_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_qo_heads / num_kv_heads;
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

  runner_params.mMultiProcessorCount = sm_count;
  int max_head_dim_kv = head_dim;
  runner_params.mNumPagesInMemPool = num_pages * 2;
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
                                   int64_t sum_seq_kv, int64_t sm_count) {
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(query.dtype(), c_type_q, {
    using DTypeQ = c_type_q;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;

    int batch_size = query.size(0);
    int num_qo_heads = query.size(1);
    int head_dim = query.size(2);
    int page_size = block_size;
    int max_num_blocks_per_seq = block_tables.size(-1);

    auto device = query.device();
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());

    trtllm_paged_attention_decode_launcher<DTypeQ, DTypeKV, DTypeO>(
        out.data_ptr<DTypeO>(), query.data_ptr<DTypeQ>(), key_value_cache.data_ptr<DTypeKV>(),
        workspace_buffer.data_ptr<void>(),
        reinterpret_cast<KVCachePageIndex*>(block_tables.data_ptr()), seq_lens.data_ptr<int>(),
        batch_size, max_seq_len, num_pages, num_qo_heads, num_kv_heads, head_dim, page_size,
        max_num_blocks_per_seq, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv,
        sm_count, stream);
    // Remove return true; as the function is void
  });
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void trtllm_paged_attention_context_launcher(
    DTypeO* out, DTypeQ* query, DTypeKV* key_value_cache, void* workspace_buffer,
    KVCachePageIndex* block_tables, int* seq_lens, int64_t batch_size, int64_t max_seq_len,
    int64_t num_pages, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t page_size, int64_t max_num_blocks_per_seq, double bmm1_scale, double bmm2_scale,
    int64_t window_left, int64_t sum_seq_q, int64_t sum_seq_kv, int64_t sm_count,
    cudaStream_t stream, int* cum_seq_lens_q = nullptr, int* cum_seq_lens_kv = nullptr) {
  auto q_data_type = TypeToDataType<DTypeQ>::value;
  auto kv_data_type = TypeToDataType<DTypeKV>::value;
  auto o_data_type = TypeToDataType<DTypeO>::value;

  static auto fmha_runner = TllmGenFmhaRunner(q_data_type, kv_data_type, o_data_type);

  TllmGenFmhaRunnerParams runner_params;

  runner_params.mMaskType = TrtllmGenAttentionMaskType::Causal;
  runner_params.mKernelType = FmhaKernelType::Context;

  // Always use persistent scheduler for better performance.
  runner_params.mTileScheduler = TileScheduler::Persistent;
  runner_params.mMultiCtasKvMode = false;

  runner_params.qPtr = query;

  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.kvPtr = key_value_cache;
  runner_params.kvPageIdxPtr = block_tables;
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = page_size;

  runner_params.seqLensKvPtr = seq_lens;

  runner_params.oPtr = out;
  runner_params.mHeadDimQk = head_dim;
  runner_params.mHeadDimV = head_dim;
  runner_params.mNumHeadsQ = num_qo_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_qo_heads / num_kv_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenQ =
      batch_size;  // Fix: should be batch_size, not query.size(0) (query is pointer)
  runner_params.mMaxSeqLenKv = max_seq_len;

  runner_params.mSumOfSeqLensQ = sum_seq_q;
  runner_params.mSumOfSeqLensKv = sum_seq_kv;
  runner_params.cumSeqLensQPtr = cum_seq_lens_q;
  runner_params.cumSeqLensKvPtr = cum_seq_lens_kv;

  runner_params.mScaleQ = 1.0;
  // Set the chunked attention size and sliding window size to INT_MAX to disable them when checking
  // if
  // the kernel is supported.
  runner_params.mChunkedAttentionSize = INT_MAX;
  runner_params.mAttentionWindowSize = window_left == -1 ? INT_MAX : window_left + 1;

  runner_params.mMultiProcessorCount = sm_count;
  auto [foundKernels, kinfo] = fmha_runner.isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    TllmGenSelectKernelParams select_kernel_params{runner_params};
    err_msg << "Missing TRTLLM-GEN context kernel:" << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }
  int max_head_dim_kv = head_dim;

  runner_params.mNumPagesInMemPool = num_pages * 2;
  runner_params.stream = stream;
  runner_params.outputScale = bmm2_scale;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  fmha_runner.run(runner_params);
}

void trtllm_paged_attention_context(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                                    at::Tensor& workspace_buffer, int64_t num_kv_heads,
                                    at::Tensor& block_tables, at::Tensor& seq_lens,
                                    int64_t block_size, int64_t max_seq_len, double bmm1_scale,
                                    double bmm2_scale, int64_t batch_size, int64_t window_left,
                                    int64_t sum_seq_q, int64_t sum_seq_kv,
                                    at::Tensor& cum_seq_lens_q, at::Tensor& cum_seq_lens_kv,
                                    int64_t sm_count) {
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(query.dtype(), c_type_q, {
    using DTypeQ = c_type_q;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;

    int batch_size_ = query.size(0);
    int num_qo_heads = query.size(1);
    int head_dim = query.size(2);
    int page_size = block_size;
    int max_num_blocks_per_seq = block_tables.size(-1);

    auto device = query.device();
    const auto stream = at::cuda::getCurrentCUDAStream(device.index());

    trtllm_paged_attention_context_launcher<DTypeQ, DTypeKV, DTypeO>(
        out.data_ptr<DTypeO>(), query.data_ptr<DTypeQ>(), key_value_cache.data_ptr<DTypeKV>(),
        workspace_buffer.data_ptr<void>(),
        reinterpret_cast<KVCachePageIndex*>(block_tables.data_ptr()), seq_lens.data_ptr<int>(),
        batch_size_, max_seq_len, num_pages, num_qo_heads, num_kv_heads, head_dim, page_size,
        max_num_blocks_per_seq, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sum_seq_kv,
        sm_count, stream, cum_seq_lens_q.defined() ? cum_seq_lens_q.data_ptr<int>() : nullptr,
        cum_seq_lens_kv.defined() ? cum_seq_lens_kv.data_ptr<int>() : nullptr);
  });
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_paged_attention_decode", trtllm_paged_attention_decode);
  m.def("trtllm_paged_attention_context", trtllm_paged_attention_context);
}

}  // namespace flashinfer
