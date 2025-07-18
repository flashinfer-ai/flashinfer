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
#include <flashinfer/trtllm/common.h>
#include <flashinfer/trtllm/fmha/decoder_impl_common.h>
#include <flashinfer/trtllm/fmha/fmhaRunnerParams.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <flashinfer/semaphore_utils.cuh>
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/trtllm/fmha/gen_kernel_launcher.cuh>
#include <flashinfer/utils.cuh>
#include <iostream>
#include <optional>
#include <sstream>

#include "pytorch_extension_utils.h"

namespace flashinfer {

enum class TllmPagedAttentionMode {
  Context,
  ForGen,
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO, TllmPagedAttentionMode mode>
void trtllm_paged_attention_launcher(
    DTypeO* out, DTypeQ* query, DTypeKV* key_value_cache, void* workspace_buffer,
    KVCachePageIndex* block_tables, int* seq_lens, int64_t batch_size, int64_t max_q_len,
    int64_t max_kv_len, int64_t num_pages, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_vo, int64_t page_size, int64_t max_num_blocks_per_seq,
    double bmm1_scale, double bmm2_scale, int64_t window_left, int64_t sum_seq_q, int64_t sm_count,
    cudaStream_t stream, int* cum_seq_lens_q = nullptr, int* cum_seq_lens_kv = nullptr) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: " << num_kv_heads
            << " and num_qo_heads: " << num_qo_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  auto q_data_type = TypeToDataType<DTypeQ>::value;
  auto kv_data_type = TypeToDataType<DTypeKV>::value;
  auto o_data_type = TypeToDataType<DTypeO>::value;
  static auto fmha_runner = TllmGenFmhaRunner(q_data_type, kv_data_type, o_data_type);

  TllmGenFmhaRunnerParams runner_params;

  // Common params
  runner_params.qPtr = query;
  runner_params.kvPtr = key_value_cache;
  runner_params.kvPageIdxPtr = block_tables;
  runner_params.seqLensKvPtr = seq_lens;
  runner_params.oPtr = out;
  runner_params.mHeadDimQk = head_dim_qk;
  runner_params.mHeadDimV = head_dim_vo;
  runner_params.mNumHeadsQ = num_qo_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_qo_heads / num_kv_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenKv = max_kv_len;
  runner_params.mMaxNumPagesPerSeqKv = max_num_blocks_per_seq;
  runner_params.mNumTokensPerPage = page_size;
  runner_params.mQkvLayout = QkvLayout::PagedKv;
  runner_params.mMultiProcessorCount = sm_count;
  runner_params.mNumPagesInMemPool = num_pages * 2;
  runner_params.stream = stream;
  runner_params.outputScale = bmm2_scale;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  runner_params.mChunkedAttentionSize = INT_MAX;
  runner_params.mAttentionWindowSize = window_left == -1 ? INT_MAX : window_left + 1;
  runner_params.mMaxSeqLenQ = max_q_len;
  runner_params.mSumOfSeqLensQ = sum_seq_q;

  if constexpr (mode == TllmPagedAttentionMode::Context) {
    runner_params.mMaskType = TrtllmGenAttentionMaskType::Causal;
    runner_params.mKernelType = FmhaKernelType::Context;
    runner_params.mTileScheduler = TileScheduler::Persistent;
    runner_params.mMultiCtasKvMode = false;

    runner_params.cumSeqLensQPtr = cum_seq_lens_q;
    runner_params.cumSeqLensKvPtr = cum_seq_lens_kv;
  } else {
    // ForGen
    runner_params.mMaskType = TrtllmGenAttentionMaskType::Dense;
    runner_params.mKernelType = FmhaKernelType::Generation;
    bool use_multi_block = true;
    runner_params.mTileScheduler =
        use_multi_block ? TileScheduler::Static : TileScheduler::Persistent;
    runner_params.mMultiCtasKvMode = use_multi_block;

    size_t num_semaphores = round_up(batch_size * num_qo_heads, 8);
    runner_params.multiCtasKvScratchPtr = reinterpret_cast<void*>(
        static_cast<char*>(workspace_buffer) + num_semaphores * sizeof(uint32_t));
    runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_buffer);
    zero_gmem_semaphore_launcher(runner_params.multiCtasKvCounterPtr, num_semaphores,
                                 /*enable_pdl=*/true, stream);
  }

  auto [foundKernels, kinfo] = fmha_runner.isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    err_msg << "Missing TRTLLM-GEN kernel ("
            << (mode == TllmPagedAttentionMode::Context ? "context" : "decode") << "): " << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }

  fmha_runner.run(runner_params);
}

void trtllm_paged_attention_decode(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                                   at::Tensor& workspace_buffer, int64_t num_kv_heads,
                                   at::Tensor& block_tables, at::Tensor& seq_lens,
                                   int64_t block_size, int64_t max_kv_len, double bmm1_scale,
                                   double bmm2_scale, int64_t window_left, int64_t sm_count) {
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(query.scalar_type(), DTypeQ, [&]() {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(key_value_cache.scalar_type(), DTypeKV, [&]() {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(out.scalar_type(), DTypeO, [&]() {
        // NOTE(Zihao): query is [B, Q, H, D]
        // where Q is the number of query tokens per request, used in MTP
        // based on profiled results, always use decode mode for MTP (q_len is small)
        // example: when kv_len = 10000, q < 200, decode mode is faster
        int batch_size = query.size(0);
        int q_len_per_request = query.size(1);
        int sum_seq_q = batch_size * q_len_per_request;
        int num_qo_heads = query.size(2);
        int head_dim_qk = query.size(3);
        int head_dim_vo = out.size(out.dim() - 1);
        int page_size = block_size;
        int max_num_blocks_per_seq = block_tables.size(-1);
        int num_pages = key_value_cache.size(0);

        auto device = query.device();
        const auto stream = at::cuda::getCurrentCUDAStream(device.index());

        trtllm_paged_attention_launcher<DTypeQ, DTypeKV, DTypeO, TllmPagedAttentionMode::ForGen>(
            static_cast<DTypeO*>(out.data_ptr()), static_cast<DTypeQ*>(query.data_ptr()),
            static_cast<DTypeKV*>(key_value_cache.data_ptr()), workspace_buffer.data_ptr(),
            static_cast<KVCachePageIndex*>(block_tables.data_ptr()),
            static_cast<int*>(seq_lens.data_ptr()), batch_size, /*max_q_len=*/q_len_per_request,
            max_kv_len, num_pages, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
            max_num_blocks_per_seq, bmm1_scale, bmm2_scale, window_left, sum_seq_q, sm_count,
            stream);
        return true;
      });
    });
  });
}

void trtllm_paged_attention_context(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                                    at::Tensor& workspace_buffer, int64_t num_kv_heads,
                                    at::Tensor& block_tables, at::Tensor& seq_lens,
                                    int64_t block_size, int64_t max_q_len, int64_t max_kv_len,
                                    double bmm1_scale, double bmm2_scale, int64_t batch_size,
                                    int64_t window_left, at::Tensor& cum_seq_lens_q,
                                    at::Tensor& cum_seq_lens_kv, int64_t sm_count) {
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE(query.scalar_type(), DTypeQ, [&]() {
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(key_value_cache.scalar_type(), DTypeKV, [&]() {
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE(out.scalar_type(), DTypeO, [&]() {
        int num_qo_heads = query.size(1);
        int sum_seq_q = query.size(0);
        int head_dim_qk = query.size(2);
        int head_dim_vo = out.size(out.dim() - 1);
        int page_size = block_size;
        int max_num_blocks_per_seq = block_tables.size(-1);
        int num_pages = key_value_cache.size(0);

        auto device = query.device();
        const auto stream = at::cuda::getCurrentCUDAStream(device.index());

        trtllm_paged_attention_launcher<DTypeQ, DTypeKV, DTypeO, TllmPagedAttentionMode::Context>(
            static_cast<DTypeO*>(out.data_ptr()), static_cast<DTypeQ*>(query.data_ptr()),
            static_cast<DTypeKV*>(key_value_cache.data_ptr()), workspace_buffer.data_ptr(),
            static_cast<KVCachePageIndex*>(block_tables.data_ptr()),
            static_cast<int*>(seq_lens.data_ptr()), batch_size, max_q_len, max_kv_len, num_pages,
            num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size, max_num_blocks_per_seq,
            bmm1_scale, bmm2_scale, window_left, sum_seq_q, sm_count, stream,
            cum_seq_lens_q.defined() ? cum_seq_lens_q.data_ptr<int>() : nullptr,
            cum_seq_lens_kv.defined() ? cum_seq_lens_kv.data_ptr<int>() : nullptr);
        return true;
      });
    });
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
