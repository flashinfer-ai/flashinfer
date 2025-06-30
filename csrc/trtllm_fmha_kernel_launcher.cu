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

__constant__ float out_scale = 1.0f;
__constant__ float log2e_h64 = 0.18033688011112042;
__constant__ float log2e_h128 = 0.12751743082459868;
__constant__ float log2e_h192 = 0.10411754627697266;
__constant__ float log2e_h256 = 0.09016844005556021;

namespace flashinfer {
template <typename T, Data_type CACHE_T>
void trtllm_paged_attention_launcher(at::Tensor& out, at::Tensor& query,
                                     at::Tensor& key_value_cache, at::Tensor& workspace_buffer,
                                     int64_t num_heads_dummy, int64_t num_kv_heads, double scale,
                                     at::Tensor& block_tables, at::Tensor& seq_lens,
                                     int64_t block_size, int64_t max_seq_len,
                                     const std::string kv_cache_dtype, double k_scale,
                                     double v_scale) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(-1);

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  float const k_scale_f = k_scale;
  float const v_scale_f = v_scale;

  uint32_t tokens_per_page = block_size;

  uint32_t num_k_heads = num_kv_heads;
  uint32_t num_v_heads = num_k_heads;
  auto batch_size = num_seqs;

  int const beam_width = num_seqs / batch_size;  // always 1

  auto q_heads = reinterpret_cast<T*>(query.data_ptr());
  auto output_ptr = reinterpret_cast<T*>(out.data_ptr());

  auto cache_heads = reinterpret_cast<void*>(key_value_cache.data_ptr());

  auto io_type = TypeToDataType<T>::value;

  bool use_multi_block = true;
  auto q_data_type = (kv_cache_dtype == "fp8_e4m3") ? DATA_TYPE_E4M3 : io_type;
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

  // num_kv_heads should be enough, but num_heads for safty at long seq len.
  size_t num_semaphores = batch_size * num_heads;

  runner_params.multiCtasKvScratchPtr =
      reinterpret_cast<void*>(workspace_buffer.data_ptr() + num_semaphores * sizeof(uint32_t));
  runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_buffer.data_ptr());

  zero_gmem_semaphore_launcher(runner_params.multiCtasKvCounterPtr, num_semaphores,
                               /*enable_pdl=*/true, stream);

  float* log_scale_ptr;
  float* output_scale_ptr;

  if (head_size == 64) {
    cudaGetSymbolAddress((void**)&log_scale_ptr, log2e_h64);
  } else if (head_size == 128) {
    cudaGetSymbolAddress((void**)&log_scale_ptr, log2e_h128);
  } else if (head_size == 192) {
    cudaGetSymbolAddress((void**)&log_scale_ptr, log2e_h192);
  } else if (head_size == 256) {
    cudaGetSymbolAddress((void**)&log_scale_ptr, log2e_h256);
  } else {
    std::ostringstream err_msg;
    err_msg << "head_size " << head_size << " is not supported!";
    FLASHINFER_ERROR(err_msg.str());
  }

  cudaGetSymbolAddress((void**)&output_scale_ptr, out_scale);

  runner_params.scaleSoftmaxLog2Ptr = log_scale_ptr;
  runner_params.outputScalePtr = output_scale_ptr;

  runner_params.seqLensKvPtr = reinterpret_cast<int const*>(seq_lens.data_ptr<int>());

  runner_params.oPtr = output_ptr;
  runner_params.mHeadDimQk = head_size;
  runner_params.mHeadDimV = head_size;
  runner_params.mNumHeadsQ = num_heads;
  runner_params.mNumHeadsKv = num_k_heads;
  runner_params.mNumHeadsQPerKv = 8;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenQ = 1;
  runner_params.mMaxSeqLenKv = max_seq_len;
  runner_params.mSumOfSeqLensQ = int(batch_size * runner_params.mMaxSeqLenQ);
  runner_params.mScaleQ = 1.0;
  // Set the chunked attention size and sliding window size to INT_MAX to disable them when checking
  // if
  // the kernel is supported.
  runner_params.mChunkedAttentionSize = INT_MAX;
  runner_params.mAttentionWindowSize = INT_MAX;

  runner_params.mMultiProcessorCount = getMultiProcessorCount();
  auto const [free_memory, total_memory] = getDeviceMemoryInfo(false);
  int max_head_dim_kv = head_size;

  runner_params.mNumPagesInMemPool =
      total_memory / (runner_params.mNumHeadsKv * runner_params.mNumTokensPerPage *
                      max_head_dim_kv * get_size_in_bytes(CACHE_T));

  runner_params.stream = stream;
  fmha_runner.run(runner_params);
}

#define CALL_GEN_LAUNCHER(T, CACHE_T_ENUM)                                                         \
  trtllm_paged_attention_launcher<T, CACHE_T_ENUM>(                                                \
      out, query, key_value_cache, workspace_buffer, num_heads, num_kv_heads, scale, block_tables, \
      seq_lens, block_size, max_seq_len, kv_cache_dtype, k_scale, v_scale);

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t>
#define DISPATCH_BY_KV_CACHE_ELEM_ENUM(SRC_DTYPE, KV_DTYPE, FN)                \
  if (KV_DTYPE == "auto") {                                                    \
    if (SRC_DTYPE == at::ScalarType::Half) {                                   \
      FN(half, Data_type::DATA_TYPE_FP16);                                     \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                        \
      FN(__nv_bfloat16, Data_type::DATA_TYPE_BF16);                            \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);   \
    }                                                                          \
  } else {                                                                     \
    if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                         \
      if (SRC_DTYPE == at::ScalarType::Half) {                                 \
        FN(half, Data_type::DATA_TYPE_E4M3);                                   \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__nv_bfloat16, Data_type::DATA_TYPE_E4M3);                          \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);     \
    }                                                                          \
  }

void trtllm_paged_attention(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                            at::Tensor& workspace_buffer, int64_t num_heads, int64_t num_kv_heads,
                            double scale, at::Tensor& block_tables, at::Tensor& seq_lens,
                            int64_t block_size, int64_t max_seq_len,
                            const std::string kv_cache_dtype, double k_scale, double v_scale) {
  DISPATCH_BY_KV_CACHE_ELEM_ENUM(query.dtype(), kv_cache_dtype, CALL_GEN_LAUNCHER);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_paged_attention", trtllm_paged_attention);
}

}  // namespace flashinfer
