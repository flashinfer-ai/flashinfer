#include <thrust/device_vector.h>

#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/quantization/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/utils.cuh>

using namespace flashinfer;

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void run_fwd_flashinfer(thrust::device_vector<DTypeQ>& q_d, thrust::device_vector<DTypeKV>& k_d,
                        thrust::device_vector<DTypeKV>& v_d, thrust::device_vector<DTypeO>& o_d,
                        thrust::device_vector<float>& scale_q_d,
                        thrust::device_vector<float>& scale_k_d,
                        thrust::device_vector<float>& scale_v_d, int32_t qo_len, int32_t kv_len,
                        int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim,
                        float sm_scale, MaskMode mask_mode, QKVLayout kv_layout) {
  using IdType = int32_t;
  using DTypeScale = float;

  constexpr bool USE_SLIDING_WINDOW = false;
  using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO, IdType>;
  using AttentionVariant = DefaultFP8Attention;

  Params params;
  params.q_ptr = static_cast<DTypeQ*>(thrust::raw_pointer_cast(q_d.data()));
  params.k_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(k_d.data()));
  params.v_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(v_d.data()));
  params.o_ptr = static_cast<DTypeO*>(thrust::raw_pointer_cast(o_d.data()));
  params.lse_ptr = nullptr;
  // q NHD
  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.o_stride_n = num_qo_heads * head_dim;
  params.o_stride_h = head_dim;
  if (kv_layout == QKVLayout::kNHD) {
    params.k_stride_n = num_kv_heads * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = num_kv_heads * head_dim;
    params.v_stride_h = head_dim;
  } else {
    // k HND
    params.k_stride_h = kv_len * head_dim;
    params.k_stride_n = head_dim;
    params.v_stride_h = kv_len * head_dim;
    params.v_stride_n = head_dim;
  }
  params.qo_len = qo_len;
  params.kv_len = kv_len;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.causal = (mask_mode == MaskMode::kCausal);
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = 0;

  params.additional_params.scale_q = thrust::raw_pointer_cast(scale_q_d.data());
  params.additional_params.scale_k = thrust::raw_pointer_cast(scale_k_d.data());
  params.additional_params.scale_v = thrust::raw_pointer_cast(scale_v_d.data());
  params.additional_params.sm_scale = sm_scale;

  cudaError_t status;
  cudaStream_t stream = 0;
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
      status = SingleFP8PrefillWithKVCacheDispatched<HEAD_DIM, MASK_MODE, USE_SLIDING_WINDOW,
                                                     AttentionVariant, Params>(params, stream);
    });
  });
  if (status != cudaSuccess) {
    throw std::runtime_error("Failed to run SingleFP8PrefillWithKVCacheDispatched");
  }
}

void run_fwd(thrust::device_vector<cutlass::float_e4m3_t>& q_d,
             thrust::device_vector<cutlass::float_e4m3_t>& k_d,
             thrust::device_vector<cutlass::float_e4m3_t>& v_d,
             thrust::device_vector<cutlass::half_t>& o_d, thrust::device_vector<float>& scale_q_d,
             thrust::device_vector<float>& scale_k_d, thrust::device_vector<float>& scale_v_d,
             int32_t qo_len, int32_t kv_len, int32_t num_qo_heads, int32_t num_kv_heads,
             int32_t head_dim, float sm_scale, MaskMode mask_mode, QKVLayout kv_layout) {
  run_fwd_flashinfer<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t>(
      q_d, k_d, v_d, o_d, scale_q_d, scale_k_d, scale_v_d, qo_len, kv_len, num_qo_heads,
      num_kv_heads, head_dim, sm_scale, mask_mode, kv_layout);
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void run_fwd_flashinfer(thrust::device_vector<DTypeQ>& q_d, thrust::device_vector<DTypeKV>& k_d,
                        thrust::device_vector<DTypeKV>& v_d, thrust::device_vector<DTypeO>& o_d,
                        int32_t qo_len, int32_t kv_len, int32_t num_qo_heads, int32_t num_kv_heads,
                        int32_t head_dim, float sm_scale, MaskMode mask_mode, QKVLayout kv_layout) {
  using IdType = int32_t;

  constexpr bool USE_SLIDING_WINDOW = false;
  using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO, IdType>;
  using AttentionVariant = StandardAttention;

  Params params;
  params.q_ptr = static_cast<DTypeQ*>(thrust::raw_pointer_cast(q_d.data()));
  params.k_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(k_d.data()));
  params.v_ptr = static_cast<DTypeKV*>(thrust::raw_pointer_cast(v_d.data()));
  params.o_ptr = static_cast<DTypeO*>(thrust::raw_pointer_cast(o_d.data()));
  params.lse_ptr = nullptr;
  // q NHD
  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.o_stride_n = num_qo_heads * head_dim;
  params.o_stride_h = head_dim;
  if (kv_layout == QKVLayout::kNHD) {
    params.k_stride_n = num_kv_heads * head_dim;
    params.k_stride_h = head_dim;
    params.v_stride_n = num_kv_heads * head_dim;
    params.v_stride_h = head_dim;
  } else {
    // k HND
    params.k_stride_h = kv_len * head_dim;
    params.k_stride_n = head_dim;
    params.v_stride_h = kv_len * head_dim;
    params.v_stride_n = head_dim;
  }
  params.qo_len = qo_len;
  params.kv_len = kv_len;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.causal = (mask_mode == MaskMode::kCausal);
  params.group_size = params.num_qo_heads / params.num_kv_heads;
  params.window_left = 0;
  params.additional_params.sm_scale = sm_scale;

  cudaError_t status;
  cudaStream_t stream = 0;
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_MASK_MODE(mask_mode, MASK_MODE, {
      status = SinglePrefillWithKVCacheDispatched<HEAD_DIM, HEAD_DIM, MASK_MODE, USE_SLIDING_WINDOW,
                                                  AttentionVariant, Params>(params, stream);
    });
  });
  if (status != cudaSuccess) {
    throw std::runtime_error("Failed to run SingleFP16PrefillWithKVCacheDispatched");
  }
}

void run_fwd(thrust::device_vector<cutlass::half_t>& q_d,
             thrust::device_vector<cutlass::half_t>& k_d,
             thrust::device_vector<cutlass::half_t>& v_d,
             thrust::device_vector<cutlass::half_t>& o_d, int32_t qo_len, int32_t kv_len,
             int32_t num_qo_heads, int32_t num_kv_heads, int32_t head_dim, float sm_scale,
             MaskMode mask_mode, QKVLayout kv_layout) {
  run_fwd_flashinfer<cutlass::half_t, cutlass::half_t, cutlass::half_t>(
      q_d, k_d, v_d, o_d, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, sm_scale, mask_mode,
      kv_layout);
}
