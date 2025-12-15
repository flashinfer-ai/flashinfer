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

#include <flashinfer/allocator.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common.h>
#include <flashinfer/trtllm/fmha/decoder_impl_common.h>
#include <flashinfer/trtllm/fmha/fmhaRunnerParams.h>
#include <nvrtc.h>
#include <tvm/ffi/container/variant.h>

#include <flashinfer/trtllm/fmha/fmhaRunner.cuh>
#include <flashinfer/utils.cuh>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "tvm/ffi/error.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using tvm::ffi::Variant;

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
    int* cum_seq_lens_kv, float* attention_sinks, Data_type q_data_type, Data_type kv_data_type,
    Data_type o_data_type, TllmPagedAttentionMode mode, int64_t batch_size, int64_t max_q_len,
    int64_t max_kv_len, int64_t num_pages_in_mem_pool, int64_t num_qo_heads, int64_t num_kv_heads,
    int64_t head_dim_qk, int64_t head_dim_vo, int64_t page_size, int64_t kv_stride_keys_values,
    int64_t kv_stride_heads, int64_t kv_stride_batch, int64_t max_num_blocks_per_seq,
    double bmm1_scale, double bmm2_scale, const float* bmm1_scale_log2_ptr,
    const float* bmm2_scale_ptr, double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index,
    int64_t window_left, int64_t sum_seq_q, int64_t sparse_mla_top_k, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size, cudaStream_t stream) {
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
  runner_params.kStrideKeysValues = kv_stride_keys_values;
  runner_params.kStrideHeads = kv_stride_heads;
  runner_params.kStrideBatch = kv_stride_batch;
  runner_params.vStrideKeysValues = kv_stride_keys_values;
  runner_params.vStrideHeads = kv_stride_heads;
  runner_params.vStrideBatch = kv_stride_batch;
  runner_params.mNumPagesInMemPool = num_pages_in_mem_pool;
  runner_params.stream = stream;
  // the scaleSoftmaxLog2Ptr and outputScalePtr have higher priority than the scaleSoftmaxLog2 and
  // outputScale. if they are not nullptr, then scaleSoftmaxLog2 and outputScale will be ignored
  runner_params.outputScale = bmm2_scale;
  runner_params.outputScalePtr = bmm2_scale_ptr;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  runner_params.scaleSoftmaxLog2Ptr = bmm1_scale_log2_ptr;
  runner_params.oSfPtr = out_scale_factor;
  runner_params.mSfStartTokenIdx = o_sf_start_index;
  runner_params.mScaleSfO = o_sf_scale;
  TVM_FFI_ICHECK(o_sf_vec_size == 16 || o_sf_vec_size == -1)
      << "Only support o_sf_vec_size == 16 or -1(not used)";
  runner_params.mChunkedAttentionSize = INT_MAX;  // disable chunked attention by INT_MAX
  runner_params.mAttentionWindowSize =
      window_left == -1 ? INT_MAX : window_left + 1;  // disable window attention by INT_MAX
  runner_params.mMaxSeqLenQ = max_q_len;
  runner_params.mSumOfSeqLensQ = sum_seq_q;
  runner_params.ptrAttentionSinks = attention_sinks;
  runner_params.enable_pdl = enable_pdl;

  // The sparse MLA parameters.
  runner_params.mSparseMla = sparse_mla_top_k > 0;
  runner_params.mSparseMlaTopK = sparse_mla_top_k;
  TVM_FFI_ICHECK((head_dim_qk == 576 && head_dim_vo == 512) || sparse_mla_top_k <= 0)
      << "Only decode MLA supports sparse MLA";

  AlignedAllocator float_allocator(workspace_buffer, workspace_size);
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

    runner_params.cumSeqLensQPtr = cum_seq_lens_q;
    runner_params.cumSeqLensKvPtr = nullptr;

    size_t max_batch_size = 8192;   // todo(Yingyi): get from dlfw
    size_t max_num_qo_heads = 256;  // todo(Yingyi): get from dlfw, in total 8MB
    size_t num_semaphores =
        round_up(max_batch_size * max_num_qo_heads, 8);  // max 8MB, should align to 16 bytes
    // semaphores be at the first 8MB of workspace buffer: counter | scratch
    // todo(Yingyi): add softmax buffer later for lse return
    runner_params.multiCtasKvCounterPtr = float_allocator.aligned_alloc<int32_t>(
        num_semaphores * sizeof(uint32_t), 16, "trtllm_gen_counter_workspace");
    // scratch takes the rest of the workspace buffer
    runner_params.multiCtasKvScratchPtr =
        float_allocator.aligned_alloc<void>(0, 16, "trtllm_gen_scratch_workspace");
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

inline Data_type dl_dtype_to_tllm_data_type(const DLDataType dtype) {
  if (dtype == dl_float32) {
    return Data_type::DATA_TYPE_FP32;
  } else if (dtype == dl_float16) {
    return Data_type::DATA_TYPE_FP16;
  } else if (dtype == dl_bfloat16) {
    return Data_type::DATA_TYPE_BF16;
  } else if (dtype == dl_float8_e4m3fn) {
    return Data_type::DATA_TYPE_E4M3;
  } else if (dtype == dl_float8_e5m2) {
    return Data_type::DATA_TYPE_E5M2;
  } else if (dtype == dl_uint8) {
    // fp4 tensor is not supported in torch and use uint8_t as container.
    return Data_type::DATA_TYPE_E2M1;
  }
  return Data_type::DATA_TYPE_UNKNOWN;
}

inline bool is_4bit(Data_type data_type) { return data_type == Data_type::DATA_TYPE_E2M1; }

void trtllm_paged_attention_decode(TensorView out, Optional<TensorView> out_scale_factor,
                                   TensorView query, TensorView key_cache, TensorView value_cache,
                                   TensorView workspace_buffer, TensorView block_tables,
                                   TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
                                   Variant<double, ffi::Tensor> bmm1_scale,
                                   Variant<double, ffi::Tensor> bmm2_scale, double o_sf_scale,
                                   int64_t o_sf_vec_size, int64_t o_sf_start_index,
                                   int64_t batch_size, int64_t window_left,
                                   int64_t sparse_mla_top_k, int64_t sm_count, bool enable_pdl,
                                   int64_t workspace_size, Optional<TensorView> attention_sinks,
                                   Optional<TensorView> cum_seq_lens_q) {
  auto q_data_type = dl_dtype_to_tllm_data_type(query.dtype());
  auto kv_data_type = dl_dtype_to_tllm_data_type(key_cache.dtype());
  TVM_FFI_ICHECK_EQ(key_cache.ndim(), value_cache.ndim());
  for (int i = 0; i < key_cache.ndim(); i++) {
    TVM_FFI_ICHECK_EQ(key_cache.size(i), value_cache.size(i));
  }
  auto o_data_type = dl_dtype_to_tllm_data_type(out.dtype());
  int sum_seq_q = query.size(0);
  int num_qo_heads = query.size(1);
  // the cum_seq_lens_q is optional, and can be nullptr when all sequences have the same query
  // length
  int* cum_seq_lens_q_ptr =
      cum_seq_lens_q.has_value() ? static_cast<int*>(cum_seq_lens_q.value().data_ptr()) : nullptr;
  // Multiply by two for FP4 tensor as it is stored as UINT8 dtype. Assume the dim is even.
  int head_dim_k = is_4bit(kv_data_type) ? key_cache.size(-1) * 2 : key_cache.size(-1);
  int head_dim_q = is_4bit(q_data_type) ? query.size(-1) * 2 : query.size(-1);
  int head_dim_v = is_4bit(kv_data_type) ? value_cache.size(-1) * 2 : value_cache.size(-1);
  int head_dim_o = is_4bit(o_data_type) ? out.size(-1) * 2 : out.size(-1);
  TVM_FFI_ICHECK_EQ(head_dim_k, head_dim_q)
      << "head_dim_k and head_dim_q must be the same, got " << std::to_string(head_dim_k) << " and "
      << std::to_string(head_dim_q);
  TVM_FFI_ICHECK((head_dim_v == 576 && head_dim_o == 512) || head_dim_v == head_dim_o)
      << "head_dim_v and head_dim_o must be the same for non-MLA attention, got "
      << std::to_string(head_dim_v) << " and " << std::to_string(head_dim_o);
  int max_num_blocks_per_seq = block_tables.size(-1);
  bool is_shared_kv = key_cache.data_ptr() == value_cache.data_ptr();
  int num_pages_in_mem_pool = is_shared_kv ? key_cache.size(0) : key_cache.size(0) * 2;

  // Assume NHD layout: [..., H, N, D]
  int page_size = key_cache.size(-2);
  int num_kv_heads = key_cache.size(-3);
  int kv_stride_keys_values = key_cache.stride(-2);  // key/values
  int kv_stride_heads = key_cache.stride(-3);        // head

  int kv_stride_batch = key_cache.stride(0);  // batch

  const auto stream = get_stream(query.device());
  void* output_sf_ptr =
      out_scale_factor.has_value() ? out_scale_factor.value().data_ptr() : nullptr;

  float* attention_sinks_ptr = nullptr;
  if (attention_sinks.has_value()) {
    TVM_FFI_ICHECK_EQ(attention_sinks.value().dtype(), dl_float32)
        << "attention_sinks must be a float tensor";
    attention_sinks_ptr = static_cast<float*>(attention_sinks.value().data_ptr());
  }
  auto maybe_bmm1_scale_value = bmm1_scale.as<double>();
  auto maybe_bmm2_scale_value = bmm2_scale.as<double>();
  auto maybe_bmm1_scale_log2_tensor = bmm1_scale.as<ffi::Tensor>();
  auto maybe_bmm2_scale_tensor = bmm2_scale.as<ffi::Tensor>();
  TVM_FFI_CHECK(maybe_bmm1_scale_value.has_value() || maybe_bmm1_scale_log2_tensor.has_value(),
                "bmm1_scale must be either a double or a tensor");
  TVM_FFI_CHECK(maybe_bmm2_scale_value.has_value() || maybe_bmm2_scale_tensor.has_value(),
                "bmm2_scale must be either a double or a tensor");
  double bmm1_scale_value =
      maybe_bmm1_scale_value.has_value() ? maybe_bmm1_scale_value.value() : 1.0;
  double bmm2_scale_value =
      maybe_bmm2_scale_value.has_value() ? maybe_bmm2_scale_value.value() : 1.0;
  float* bmm1_scale_log2_ptr =
      maybe_bmm1_scale_log2_tensor.has_value()
          ? static_cast<float*>(maybe_bmm1_scale_log2_tensor.value().data_ptr())
          : nullptr;
  float* bmm2_scale_ptr = maybe_bmm2_scale_tensor.has_value()
                              ? static_cast<float*>(maybe_bmm2_scale_tensor.value().data_ptr())
                              : nullptr;
  trtllm_paged_attention_launcher(
      out.data_ptr(), output_sf_ptr, query.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
      workspace_buffer.data_ptr(), static_cast<int*>(block_tables.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()), cum_seq_lens_q_ptr,
      /*cum_seq_lens_kv*/ nullptr, attention_sinks_ptr, q_data_type, kv_data_type, o_data_type,
      TllmPagedAttentionMode::ForGen, batch_size, max_q_len, max_kv_len, num_pages_in_mem_pool,
      num_qo_heads, num_kv_heads, head_dim_q, head_dim_o, page_size, kv_stride_keys_values,
      kv_stride_heads, kv_stride_batch, max_num_blocks_per_seq, bmm1_scale_value, bmm2_scale_value,
      bmm1_scale_log2_ptr, bmm2_scale_ptr, o_sf_scale, o_sf_vec_size, o_sf_start_index, window_left,
      sum_seq_q, sparse_mla_top_k, sm_count, enable_pdl, workspace_size, stream);
}

void trtllm_paged_attention_context(
    TensorView out, Optional<TensorView> out_scale_factor, TensorView query, TensorView key_cache,
    TensorView value_cache, TensorView workspace_buffer, TensorView block_tables,
    TensorView seq_lens, int64_t max_q_len, int64_t max_kv_len,
    Variant<double, ffi::Tensor> bmm1_scale, Variant<double, ffi::Tensor> bmm2_scale,
    double o_sf_scale, int64_t o_sf_vec_size, int64_t o_sf_start_index, int64_t batch_size,
    int64_t window_left, TensorView cum_seq_lens_q, TensorView cum_seq_lens_kv, int64_t sm_count,
    bool enable_pdl, int64_t workspace_size, Optional<TensorView> attention_sinks) {
  auto q_data_type = dl_dtype_to_tllm_data_type(query.dtype());
  auto kv_data_type = dl_dtype_to_tllm_data_type(key_cache.dtype());
  auto o_data_type = dl_dtype_to_tllm_data_type(out.dtype());
  int num_qo_heads = query.size(1);
  int sum_seq_q = query.size(0);
  // Multiply by two for FP4 tensor as it is stored as UINT8 dtype. Assume the dim is even.
  int head_dim_k = is_4bit(kv_data_type) ? key_cache.size(-1) * 2 : key_cache.size(-1);
  int head_dim_q = is_4bit(q_data_type) ? query.size(-1) * 2 : query.size(-1);
  int head_dim_v = is_4bit(kv_data_type) ? value_cache.size(-1) * 2 : value_cache.size(-1);
  int head_dim_o = is_4bit(o_data_type) ? out.size(-1) * 2 : out.size(-1);
  TVM_FFI_ICHECK_EQ(head_dim_k, head_dim_q)
      << "head_dim_k and head_dim_q must be the same, got " << std::to_string(head_dim_k) << " and "
      << std::to_string(head_dim_q);
  TVM_FFI_ICHECK_EQ(head_dim_v, head_dim_o)
      << "head_dim_v and head_dim_o must be the same, got " << std::to_string(head_dim_v) << " and "
      << std::to_string(head_dim_o);
  int max_num_blocks_per_seq = block_tables.size(-1);
  bool is_shared_kv = key_cache.data_ptr() == value_cache.data_ptr();
  int num_pages_in_mem_pool = is_shared_kv ? key_cache.size(0) : key_cache.size(0) * 2;

  // Assume NHD layout: [..., H, N, D]
  int page_size = key_cache.size(-2);
  int num_kv_heads = key_cache.size(-3);
  int kv_stride_keys_values = key_cache.stride(-2);  // key/values
  int kv_stride_heads = key_cache.stride(-3);        // head
  int kv_stride_batch = key_cache.stride(0);         // batch

  const auto stream = get_stream(query.device());
  void* output_sf_ptr =
      out_scale_factor.has_value() ? out_scale_factor.value().data_ptr() : nullptr;

  float* attention_sinks_ptr = nullptr;
  if (attention_sinks.has_value()) {
    TVM_FFI_ICHECK_EQ(attention_sinks.value().dtype(), dl_float32)
        << "attention_sinks must be a float tensor";
    attention_sinks_ptr = static_cast<float*>(attention_sinks.value().data_ptr());
  }

  auto maybe_bmm1_scale_value = bmm1_scale.as<double>();
  auto maybe_bmm2_scale_value = bmm2_scale.as<double>();
  auto maybe_bmm1_scale_log2_tensor = bmm1_scale.as<ffi::Tensor>();
  auto maybe_bmm2_scale_tensor = bmm2_scale.as<ffi::Tensor>();
  TVM_FFI_CHECK(maybe_bmm1_scale_value.has_value() || maybe_bmm1_scale_log2_tensor.has_value(),
                "bmm1_scale must be either a double or a tensor");
  TVM_FFI_CHECK(maybe_bmm2_scale_value.has_value() || maybe_bmm2_scale_tensor.has_value(),
                "bmm2_scale must be either a double or a tensor");
  double bmm1_scale_value =
      maybe_bmm1_scale_value.has_value() ? maybe_bmm1_scale_value.value() : 1.0;
  double bmm2_scale_value =
      maybe_bmm2_scale_value.has_value() ? maybe_bmm2_scale_value.value() : 1.0;
  float* bmm1_scale_log2_ptr =
      maybe_bmm1_scale_log2_tensor.has_value()
          ? static_cast<float*>(maybe_bmm1_scale_log2_tensor.value().data_ptr())
          : nullptr;
  float* bmm2_scale_ptr = maybe_bmm2_scale_tensor.has_value()
                              ? static_cast<float*>(maybe_bmm2_scale_tensor.value().data_ptr())
                              : nullptr;

  trtllm_paged_attention_launcher(
      out.data_ptr(), output_sf_ptr, query.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
      workspace_buffer.data_ptr(), static_cast<int*>(block_tables.data_ptr()),
      static_cast<int*>(seq_lens.data_ptr()),
      /*cum_seq_lens_q=*/static_cast<int*>(cum_seq_lens_q.data_ptr()),
      /*cum_seq_lens_kv=*/static_cast<int*>(cum_seq_lens_kv.data_ptr()), attention_sinks_ptr,
      q_data_type, kv_data_type, o_data_type, TllmPagedAttentionMode::Context, batch_size,
      max_q_len, max_kv_len, num_pages_in_mem_pool, num_qo_heads, num_kv_heads, head_dim_q,
      head_dim_o, page_size, kv_stride_keys_values, kv_stride_heads, kv_stride_batch,
      max_num_blocks_per_seq, bmm1_scale_value, bmm2_scale_value, bmm1_scale_log2_ptr,
      bmm2_scale_ptr, o_sf_scale, o_sf_vec_size, o_sf_start_index, window_left, sum_seq_q,
      /*sparse_mla_top_k=*/0, sm_count, enable_pdl, workspace_size, stream);
}

void trtllm_ragged_attention_launcher(
    void* out, void* query, void* key, void* value, void* workspace_buffer, int* seq_lens,
    int* cum_seq_lens_q, int* cum_seq_lens_kv, float* attention_sinks, float* lse,
    Data_type q_data_type, Data_type kv_data_type, Data_type o_data_type, int64_t max_q_len,
    int64_t max_kv_len, int64_t num_qo_heads, int64_t num_kv_heads, int64_t head_dim_qk,
    int64_t head_dim_v, int64_t sum_seq_q, int64_t sum_seq_kv, double bmm1_scale, double bmm2_scale,
    const float* bmm1_scale_log2_ptr, const float* bmm2_scale_ptr, double o_sf_scale,
    int64_t batch_size, int64_t window_left, int64_t sm_count, bool enable_pdl, bool is_causal,
    int64_t k_stride_keys_values, int64_t k_stride_heads, int64_t k_stride_batch,
    int64_t v_stride_keys_values, int64_t v_stride_heads, int64_t v_stride_batch,
    int64_t workspace_size, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: " << num_kv_heads
            << " and num_qo_heads: " << num_qo_heads;
    FLASHINFER_ERROR(err_msg.str());
  }
  auto fmha_runner = TllmGenFmhaRunnerCache::get(q_data_type, kv_data_type, o_data_type);
  TllmGenFmhaRunnerParams runner_params;

  runner_params.qPtr = query;
  runner_params.kPtr = key;
  runner_params.vPtr = value;
  runner_params.kvPageIdxPtr = nullptr;
  runner_params.seqLensKvPtr = seq_lens;
  runner_params.oPtr = out;
  runner_params.mHeadDimQk = head_dim_qk;
  runner_params.mHeadDimV = head_dim_v;
  runner_params.mNumHeadsQ = num_qo_heads;
  runner_params.mNumHeadsKv = num_kv_heads;
  runner_params.mNumHeadsQPerKv = num_qo_heads / num_kv_heads;
  runner_params.mBatchSize = batch_size;
  runner_params.mMaxSeqLenKv = max_kv_len;
  runner_params.mQkvLayout = QkvLayout::SeparateQkv;
  runner_params.mMultiProcessorCount = sm_count;
  runner_params.stream = stream;
  // the scaleSoftmaxLog2Ptr and outputScalePtr have higher priority than the scaleSoftmaxLog2 and
  // outputScale. if they are not nullptr, then scaleSoftmaxLog2 and outputScale will be ignored
  runner_params.outputScale = bmm2_scale;
  runner_params.outputScalePtr = bmm2_scale_ptr;
  runner_params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  runner_params.scaleSoftmaxLog2Ptr = bmm1_scale_log2_ptr;
  runner_params.mScaleSfO = o_sf_scale;
  runner_params.mChunkedAttentionSize = INT_MAX;  // disable chunked attention by INT_MAX
  runner_params.mAttentionWindowSize =
      window_left == -1 ? INT_MAX : window_left + 1;  // disable window attention by INT_MAX
  runner_params.mMaxSeqLenQ = max_q_len;
  runner_params.mSumOfSeqLensQ = sum_seq_q;
  runner_params.mSumOfSeqLensKv = sum_seq_kv;
  runner_params.cumSeqLensKvPtr = cum_seq_lens_kv;
  runner_params.cumSeqLensQPtr = cum_seq_lens_q;
  runner_params.ptrAttentionSinks = attention_sinks;
  runner_params.enable_pdl = enable_pdl;

  runner_params.kStrideKeysValues = k_stride_keys_values;
  runner_params.kStrideHeads = k_stride_heads;
  runner_params.kStrideBatch = k_stride_batch;
  runner_params.vStrideKeysValues = v_stride_keys_values;
  runner_params.vStrideHeads = v_stride_heads;
  runner_params.vStrideBatch = v_stride_batch;

  runner_params.mKernelType = FmhaKernelType::Context;
  runner_params.mTileScheduler = TileScheduler::Persistent;
  runner_params.mMaskType =
      is_causal ? TrtllmGenAttentionMaskType::Causal : TrtllmGenAttentionMaskType::Dense;
  runner_params.lsePtr = lse;

  AlignedAllocator float_allocator(workspace_buffer, workspace_size);
  size_t max_batch_size = 8192;
  size_t max_num_qo_heads = 256;
  size_t num_semaphores =
      round_up(max_batch_size * max_num_qo_heads, 8);  // max 8MB, should align to 16 bytes
  // semaphores be at the first 8MB of workspace buffer: counter | softmax | scratch
  runner_params.multiCtasKvCounterPtr = float_allocator.aligned_alloc<int32_t>(
      num_semaphores * sizeof(uint32_t), 16, "trtllm_gen_counter_workspace");
  runner_params.softmaxStatsPtr = float_allocator.aligned_alloc<float2>(
      sizeof(float2) * num_qo_heads * runner_params.mSumOfSeqLensQ, 16,
      "trtllm_gen_softmax_workspace");
  // scratch takes the rest of the workspace buffer
  runner_params.multiCtasKvScratchPtr =
      float_allocator.aligned_alloc<void>(0, 16, "trtllm_gen_scratch_workspace");

  auto [foundKernels, kinfo] = fmha_runner->isSupportedWithInfo(runner_params);
  if (!foundKernels) {
    std::ostringstream err_msg;
    err_msg << "Missing TRTLLM-GEN kernel ragged attention: " << kinfo;
    FLASHINFER_ERROR(err_msg.str());
  }

  fmha_runner->run(runner_params);
}

void trtllm_ragged_attention(TensorView out, TensorView query, TensorView key, TensorView value,
                             TensorView workspace_buffer, TensorView seq_lens, int64_t max_q_len,
                             int64_t max_kv_len, Variant<double, ffi::Tensor> bmm1_scale,
                             Variant<double, ffi::Tensor> bmm2_scale, double o_sf_scale,
                             int64_t batch_size, int64_t window_left, TensorView cum_seq_lens_q,
                             TensorView cum_seq_lens_kv, int64_t sm_count, bool enable_pdl,
                             bool is_causal, int64_t workspace_size,
                             Optional<TensorView> attention_sinks, Optional<TensorView> lse) {
  float* attention_sinks_ptr = nullptr;
  if (attention_sinks.has_value()) {
    TVM_FFI_ICHECK_EQ(attention_sinks.value().dtype(), dl_float32)
        << "attention_sinks must be a float tensor";
    attention_sinks_ptr = static_cast<float*>(attention_sinks.value().data_ptr());
  }
  float* lse_ptr = nullptr;
  if (lse.has_value()) {
    TVM_FFI_ICHECK_EQ(lse.value().dtype(), dl_float32) << "lse must be a float tensor";
    lse_ptr = static_cast<float*>(lse.value().data_ptr());
  }
  TVM_FFI_ICHECK_EQ(out.ndim(), 3) << "out must be a 3D tensor";
  TVM_FFI_ICHECK_EQ(query.ndim(), 3) << "query must be a 3D tensor";
  TVM_FFI_ICHECK_EQ(key.ndim(), 3) << "key must be a 3D tensor";
  TVM_FFI_ICHECK_EQ(value.ndim(), 3) << "value must be a 3D tensor";

  auto q_data_type = dl_dtype_to_tllm_data_type(query.dtype());
  auto kv_data_type = dl_dtype_to_tllm_data_type(key.dtype());
  auto o_data_type = dl_dtype_to_tllm_data_type(out.dtype());
  const auto stream = get_stream(query.device());
  int num_qo_heads = query.size(1);
  int num_kv_heads = key.size(1);
  int sum_seq_q = query.size(0);
  int sum_seq_kv = key.size(0);
  int head_dim_qk = query.size(2);
  int head_dim_v = value.size(2);
  int k_stride_keys_values = key.stride(0);
  int k_stride_heads = key.stride(1);
  int k_stride_batch = key.numel();
  int v_stride_keys_values = value.stride(0);
  int v_stride_heads = value.stride(1);
  int v_stride_batch = value.numel();

  auto maybe_bmm1_scale_value = bmm1_scale.as<double>();
  auto maybe_bmm2_scale_value = bmm2_scale.as<double>();
  auto maybe_bmm1_scale_log2_tensor = bmm1_scale.as<ffi::Tensor>();
  auto maybe_bmm2_scale_tensor = bmm2_scale.as<ffi::Tensor>();
  TVM_FFI_CHECK(maybe_bmm1_scale_value.has_value() || maybe_bmm1_scale_log2_tensor.has_value(),
                "bmm1_scale must be either a double or a tensor");
  TVM_FFI_CHECK(maybe_bmm2_scale_value.has_value() || maybe_bmm2_scale_tensor.has_value(),
                "bmm2_scale must be either a double or a tensor");
  double bmm1_scale_value =
      maybe_bmm1_scale_value.has_value() ? maybe_bmm1_scale_value.value() : 1.0;
  double bmm2_scale_value =
      maybe_bmm2_scale_value.has_value() ? maybe_bmm2_scale_value.value() : 1.0;
  float* bmm1_scale_log2_ptr =
      maybe_bmm1_scale_log2_tensor.has_value()
          ? static_cast<float*>(maybe_bmm1_scale_log2_tensor.value().data_ptr())
          : nullptr;
  float* bmm2_scale_ptr = maybe_bmm2_scale_tensor.has_value()
                              ? static_cast<float*>(maybe_bmm2_scale_tensor.value().data_ptr())
                              : nullptr;
  trtllm_ragged_attention_launcher(
      out.data_ptr(), query.data_ptr(), key.data_ptr(), value.data_ptr(),
      workspace_buffer.data_ptr(), static_cast<int*>(seq_lens.data_ptr()),
      static_cast<int*>(cum_seq_lens_q.data_ptr()), static_cast<int*>(cum_seq_lens_kv.data_ptr()),
      attention_sinks_ptr, lse_ptr, q_data_type, kv_data_type, o_data_type, max_q_len, max_kv_len,
      num_qo_heads, num_kv_heads, head_dim_qk, head_dim_v, sum_seq_q, sum_seq_kv, bmm1_scale_value,
      bmm2_scale_value, bmm1_scale_log2_ptr, bmm2_scale_ptr, o_sf_scale, batch_size, window_left,
      sm_count, enable_pdl, is_causal, k_stride_keys_values, k_stride_heads, k_stride_batch,
      v_stride_keys_values, v_stride_heads, v_stride_batch, workspace_size, stream);
}

namespace trtllm_cubin_loader {
#include <flashinfer/cubin_loader.h>
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_paged_attention_decode, trtllm_paged_attention_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_paged_attention_context, trtllm_paged_attention_context);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_ragged_attention, trtllm_ragged_attention);

}  // namespace flashinfer
