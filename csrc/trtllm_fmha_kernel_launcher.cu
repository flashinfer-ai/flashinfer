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

#include <flashinfer/semaphore_utils.cuh>
#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/trtllm/fmha/gen_kernel_launcher.cuh>
#include <flashinfer/utils.cuh>
#include <iostream>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "pytorch_extension_utils.h"

namespace flashinfer {

enum class TllmPagedAttentionMode {
  Context,
  ForGen,
};

#include <memory>
#include <mutex>

class TllmGenFmhaRunnerCache {
 public:
  using Key = std::tuple<Data_type, Data_type, Data_type>;

  static std::shared_ptr<TllmGenFmhaRunner> get(Data_type q_data_type, Data_type kv_data_type,
                                                Data_type o_data_type) {
    static std::unordered_map<Key, std::shared_ptr<TllmGenFmhaRunner>, KeyHash> cache;
    static std::mutex cache_mutex;
    Key key = std::make_tuple(q_data_type, kv_data_type, o_data_type);

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    } else {
      auto runner = std::make_shared<TllmGenFmhaRunner>(q_data_type, kv_data_type, o_data_type);
      cache.emplace(key, runner);
      return runner;
    }
  }

 private:
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      return std::hash<int>()(static_cast<int>(std::get<0>(k))) ^
             (std::hash<int>()(static_cast<int>(std::get<1>(k))) << 1) ^
             (std::hash<int>()(static_cast<int>(std::get<2>(k))) << 2);
    }
  };
};

void trtllm_paged_attention_launcher(
    void* out, void* out_scale_factor, void* query, void* key_cache, void* value_cache,
    void* workspace_buffer, int* block_tables, int* seq_lens, int* cum_seq_lens_q,
    int* cum_seq_lens_kv, Data_type q_data_type, Data_type kv_data_type, Data_type o_data_type,
    TllmPagedAttentionMode mode, int64_t batch_size, int64_t max_q_len, int64_t max_kv_len,
    int64_t num_pages_in_mem_pool, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim_qk,
    int64_t head_dim_vo, int64_t page_size, int64_t kv_stride_keys_values, int64_t kv_stride_heads,
    int64_t kv_stride_batch, int64_t max_num_blocks_per_seq, double bmm1_scale, double bmm2_scale,
    float* bmm1_scale_log2_ptr, float* bmm2_scale_ptr, double o_sf_scale, int64_t o_sf_vec_size,
    int64_t window_left, int64_t sum_seq_q, int64_t sm_count, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: " << num_kv_heads
            << " and num_qo_heads: " << num_qo_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  auto fmha_runner = TllmGenFmhaRunnerCache::get(q_data_type, kv_data_type, o_data_type);
  TllmGenFmhaRunnerParams runner_params;

  // Common params
  runner_params.qPtr = query;
  runner_params.kPtr = key_cache;
  runner_params.vPtr = value_cache;
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
  runner_params.kvStrideKeysValues = kv_stride_keys_values;
  runner_params.kvStrideHeads = kv_stride_heads;
  runner_params.kvStrideBatch = kv_stride_batch;
  runner_params.mNumPagesInMemPool = num_pages_in_mem_pool;
  runner_params.stream = stream;
  runner_params.outputScale = bmm2_scale;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  runner_params.outputScalePtr = bmm2_scale_ptr;
  runner_params.scaleSoftmaxLog2Ptr = bmm1_scale_log2_ptr;
  runner_params.oSfPtr = out_scale_factor;
  runner_params.mScaleSfO = o_sf_scale;
  TORCH_CHECK(o_sf_vec_size == 16 || o_sf_vec_size == -1,
              "Only support o_sf_vec_size == 16 or -1(not used)");
  runner_params.mChunkedAttentionSize = INT_MAX;  // disable chunked attention by INT_MAX
  runner_params.mAttentionWindowSize =
      window_left == -1 ? INT_MAX : window_left + 1;  // disable window attention by INT_MAX
  runner_params.mMaxSeqLenQ = max_q_len;
  runner_params.mSumOfSeqLensQ = sum_seq_q;

  if (mode == TllmPagedAttentionMode::Context) {
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

    size_t num_semaphores =
        round_up(batch_size * num_qo_heads, 8);  // align multiCtasKvScratchPtr to 16 bytes
    runner_params.multiCtasKvScratchPtr = reinterpret_cast<void*>(
        static_cast<char*>(workspace_buffer) + num_semaphores * sizeof(uint32_t));
    runner_params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(workspace_buffer);
    zero_gmem_semaphore_launcher(runner_params.multiCtasKvCounterPtr, num_semaphores,
                                 /*enable_pdl=*/true, stream);
  }

  auto [foundKernels, kinfo] = fmha_runner->isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    err_msg << "Missing TRTLLM-GEN kernel ("
            << (mode == TllmPagedAttentionMode::Context ? "context" : "decode") << "): " << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }

  fmha_runner->run(runner_params);
}

inline Data_type torch_dtype_to_tllm_data_type(at::ScalarType dtype) {
  if (dtype == at::ScalarType::Float) {
    return Data_type::DATA_TYPE_FP32;
  } else if (dtype == at::ScalarType::Half) {
    return Data_type::DATA_TYPE_FP16;
  } else if (dtype == at::ScalarType::BFloat16) {
    return Data_type::DATA_TYPE_BF16;
  } else if (dtype == at::ScalarType::Float8_e4m3fn) {
    return Data_type::DATA_TYPE_E4M3;
  } else if (dtype == at::ScalarType::Float8_e5m2) {
    return Data_type::DATA_TYPE_E5M2;
  } else if (dtype == at::ScalarType::Byte) {
    // fp4 tensor is not supported in torch and use uint8_t as container.
    return Data_type::DATA_TYPE_E2M1;
  }
  return Data_type::DATA_TYPE_UNKNOWN;
}

inline bool is_4bit(Data_type data_type) { return data_type == Data_type::DATA_TYPE_E2M1; }

void trtllm_paged_attention_decode(at::Tensor out, std::optional<at::Tensor> const out_scale_factor,
                                   at::Tensor query, at::Tensor key_value_cache,
                                   at::Tensor workspace_buffer, at::Tensor block_tables,
                                   at::Tensor seq_lens, int64_t max_kv_len, double bmm1_scale,
                                   double bmm2_scale, double o_sf_scale, int64_t o_sf_vec_size,
                                   int64_t window_left, int64_t sm_count,
                                   std::optional<at::Tensor> bmm1_scale_log2_tensor,
                                   std::optional<at::Tensor> bmm2_scale_tensor) {
  auto q_data_type = torch_dtype_to_tllm_data_type(query.scalar_type());
  auto kv_data_type = torch_dtype_to_tllm_data_type(key_value_cache.scalar_type());
  auto o_data_type = torch_dtype_to_tllm_data_type(out.scalar_type());
  // NOTE(Zihao): query is [B, Q, H, D]
  // where Q is the number of query tokens per request, used in MTP
  // based on profiled results, always use decode mode for MTP (q_len is small)
  // example: when kv_len = 10000, q < 200, decode mode is faster
  int batch_size = query.size(0);
  int q_len_per_request = query.size(1);
  int sum_seq_q = batch_size * q_len_per_request;
  int num_qo_heads = query.size(2);
  // Multiply by two for FP4 tensor as it is stored as UINT8 dtype. Assume the dim is even.
  int head_dim_kv = is_4bit(kv_data_type) ? key_value_cache.size(-1) * 2 : key_value_cache.size(-1);
  int head_dim_qk = is_4bit(q_data_type) ? query.size(-1) * 2 : query.size(-1);
  TORCH_CHECK(head_dim_kv == head_dim_qk, "head_dim_kv and head_dim_qk must be the same, got " +
                                              std::to_string(head_dim_kv) + " and " +
                                              std::to_string(head_dim_qk));
  int head_dim_vo = is_4bit(o_data_type) ? out.size(-1) * 2 : out.size(-1);
  TORCH_CHECK(head_dim_kv == head_dim_vo, "head_dim_kv and head_dim_vo must be the same, got " +
                                              std::to_string(head_dim_kv) + " and " +
                                              std::to_string(head_dim_vo));
  // NOTE(Zihao): key_value_cache is [num_pages, 1/2, num_kv_heads, page_size, head_dim]
  // For KV-Cache sharing (MLA), the second dimension is 1 (key/value cache are shared)
  // otherwise it is 2, one for key and one for value
  TORCH_CHECK(key_value_cache.size(1) == 1 || key_value_cache.size(1) == 2,
              "The second dimension of key_value_cache must be 1 or 2, got " +
                  std::to_string(key_value_cache.size(1)));
  bool share_kv_cache = key_value_cache.size(1) == 1;
  int page_size = key_value_cache.size(-2);
  int num_kv_heads = key_value_cache.size(-3);
  int max_num_blocks_per_seq = block_tables.size(-1);
  int num_pages_in_mem_pool = key_value_cache.size(0) * key_value_cache.size(1);

  int kv_stride_keys_values = key_value_cache.stride(-2);  // key/values
  int kv_stride_heads = key_value_cache.stride(-3);        // head
  int kv_stride_batch = key_value_cache.stride(0);         // batch

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  float* bmm1_scale_log2_ptr = nullptr;
  float* bmm2_scale_ptr = nullptr;
  if (bmm1_scale_log2_tensor.has_value()) {
    bmm1_scale_log2_ptr = static_cast<float*>(bmm1_scale_log2_tensor.value().data_ptr());
  }
  if (bmm2_scale_tensor.has_value()) {
    bmm2_scale_ptr = static_cast<float*>(bmm2_scale_tensor.value().data_ptr());
  }
  void* output_sf_ptr = out_scale_factor ? out_scale_factor.value().data_ptr() : nullptr;

  trtllm_paged_attention_launcher(
      out.data_ptr(), output_sf_ptr, query.data_ptr(), key_value_cache.data_ptr(),
      (char*)key_value_cache.data_ptr() +
          (share_kv_cache ? 0 : key_value_cache.stride(1) * key_value_cache.element_size()),
      workspace_buffer.data_ptr(), static_cast<int*>(block_tables.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()),
      /*cum_seq_lens_q=*/nullptr,
      /*cum_seq_lens_kv=*/nullptr, q_data_type, kv_data_type, o_data_type,
      TllmPagedAttentionMode::ForGen, batch_size, /*max_q_len=*/q_len_per_request, max_kv_len,
      num_pages_in_mem_pool, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      kv_stride_keys_values, kv_stride_heads, kv_stride_batch, max_num_blocks_per_seq, bmm1_scale,
      bmm2_scale, bmm1_scale_log2_ptr, bmm2_scale_ptr, o_sf_scale, o_sf_vec_size, window_left,
      sum_seq_q, sm_count, stream);
}

void trtllm_paged_attention_context(at::Tensor out, at::Tensor query, at::Tensor key_value_cache,
                                    at::Tensor workspace_buffer, at::Tensor block_tables,
                                    at::Tensor seq_lens, int64_t max_q_len, int64_t max_kv_len,
                                    double bmm1_scale, double bmm2_scale, int64_t batch_size,
                                    int64_t window_left, at::Tensor cum_seq_lens_q,
                                    at::Tensor cum_seq_lens_kv, int64_t sm_count,
                                    std::optional<at::Tensor> bmm1_scale_log2_tensor,
                                    std::optional<at::Tensor> bmm2_scale_tensor) {
  auto q_data_type = torch_dtype_to_tllm_data_type(query.scalar_type());
  auto kv_data_type = torch_dtype_to_tllm_data_type(key_value_cache.scalar_type());
  auto o_data_type = torch_dtype_to_tllm_data_type(out.scalar_type());
  int num_qo_heads = query.size(1);
  int sum_seq_q = query.size(0);
  // Multiply by two for FP4 tensor as it is stored as UINT8 dtype. Assume the dim is even.
  int head_dim_kv = is_4bit(kv_data_type) ? key_value_cache.size(-1) * 2 : key_value_cache.size(-1);
  int head_dim_qk = is_4bit(q_data_type) ? query.size(-1) * 2 : query.size(-1);
  TORCH_CHECK(head_dim_kv == head_dim_qk, "head_dim_kv and head_dim_qk must be the same, got " +
                                              std::to_string(head_dim_kv) + " and " +
                                              std::to_string(head_dim_qk));
  int head_dim_vo = is_4bit(o_data_type) ? out.size(-1) * 2 : out.size(-1);
  TORCH_CHECK(head_dim_kv == head_dim_vo, "head_dim_kv and head_dim_vo must be the same, got " +
                                              std::to_string(head_dim_kv) + " and " +
                                              std::to_string(head_dim_vo));
  int max_num_blocks_per_seq = block_tables.size(-1);
  int num_pages_in_mem_pool = key_value_cache.size(0) * key_value_cache.size(1);
  // NOTE(Zihao): key_value_cache is [num_pages, 1/2, num_kv_heads, page_size, head_dim]
  // For KV-Cache sharing (MLA), the second dimension is 1 (key/value cache are shared)
  // otherwise it is 2, one for key and one for value
  TORCH_CHECK(key_value_cache.size(1) == 1 || key_value_cache.size(1) == 2,
              "The second dimension of key_value_cache must be 1 or 2, got " +
                  std::to_string(key_value_cache.size(1)));
  bool share_kv_cache = key_value_cache.size(1) == 1;
  int page_size = key_value_cache.size(-2);
  int num_kv_heads = key_value_cache.size(-3);

  int kv_stride_keys_values = key_value_cache.stride(-2);  // key/values
  int kv_stride_heads = key_value_cache.stride(-3);        // head
  int kv_stride_batch = key_value_cache.stride(0);         // batch

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());
  float* bmm1_scale_log2_ptr = nullptr;
  float* bmm2_scale_ptr = nullptr;
  if (bmm1_scale_log2_tensor.has_value()) {
    bmm1_scale_log2_ptr = static_cast<float*>(bmm1_scale_log2_tensor.value().data_ptr());
  }
  if (bmm2_scale_tensor.has_value()) {
    bmm2_scale_ptr = static_cast<float*>(bmm2_scale_tensor.value().data_ptr());
  }

  trtllm_paged_attention_launcher(
      out.data_ptr(), /*out_scale_factor=*/nullptr, query.data_ptr(), key_value_cache.data_ptr(),
      (char*)key_value_cache.data_ptr() +
          (share_kv_cache ? 0 : key_value_cache.stride(1) * key_value_cache.element_size()),
      workspace_buffer.data_ptr(), static_cast<int*>(block_tables.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()),
      /*cum_seq_lens_q=*/static_cast<int*>(cum_seq_lens_q.data_ptr()),
      /*cum_seq_lens_kv=*/static_cast<int*>(cum_seq_lens_kv.data_ptr()), q_data_type, kv_data_type,
      o_data_type, TllmPagedAttentionMode::Context, batch_size, max_q_len, max_kv_len,
      num_pages_in_mem_pool, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size,
      kv_stride_keys_values, kv_stride_heads, kv_stride_batch, max_num_blocks_per_seq, bmm1_scale,
      bmm2_scale, bmm1_scale_log2_ptr, bmm2_scale_ptr, /* o_sf_scale =*/-1, /* o_sf_vec_size =*/-1,
      window_left, sum_seq_q, sm_count, stream);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("trtllm_paged_attention_decode", trtllm_paged_attention_decode);
  m.def("trtllm_paged_attention_context", trtllm_paged_attention_context);
}

}  // namespace flashinfer
