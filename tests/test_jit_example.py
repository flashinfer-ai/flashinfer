import flashinfer
import flashinfer.jit
import jinja2

single_decode_templ = r"""
#include <torch/extension.h>
#include <optional>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/decode_params.cuh>
#include "pytorch_extension_utils.h"

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
struct ParamsT : public DecodeParamsBase<DTypeQ, DTypeKV, DTypeO> {
  using IdType = int32_t;
  DTypeKV* k;
  DTypeKV* v;
  float* alibi_slopes;
  uint8_t* custom_mask;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  uint32_t head_dim;
  int32_t window_left;
  float logits_soft_cap;
  float rope_rcp_scale;
  float rope_rcp_theta;
  uint32_t kv_chunk_size;

  __device__ __host__ ParamsT(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o,
                                         uint8_t* custom_mask,
                                         float* alibi_slopes, uint32_t seq_len,
                                         uint32_t num_qo_heads, uint32_t num_kv_heads,
                                         QKVLayout kv_layout, uint32_t head_dim,
                                         int32_t window_left, float logits_soft_cap, float sm_scale,
                                         float rope_scale, float rope_theta)
      : DecodeParamsBase<DTypeQ, DTypeKV, DTypeO>{q, o, /*lse=*/nullptr, sm_scale},
        k(k),
        v(v),
        custom_mask(custom_mask),
        alibi_slopes(alibi_slopes),
        kv_len(seq_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(num_qo_heads * head_dim),
        q_stride_h(head_dim),
        kv_stride_n((kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim),
        kv_stride_h((kv_layout == QKVLayout::kNHD) ? head_dim : seq_len * head_dim),
        head_dim(head_dim),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        kv_chunk_size(0) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }

  __host__ __device__ __forceinline__ uint8_t* get_batch_local_mask_ptr(uint32_t batch_idx) const {
    return this->custom_mask;
  }
};

{% set use_alibi = "true" if pos_encoding_mode == "PosEncodingMode::kALiBi" else "false" %}
torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, torch::Tensor custom_mask,
                                          std::optional<torch::Tensor> alibi_slopes,
                                          unsigned int layout, int window_left,
                                          float logits_soft_cap, float sm_scale, float rope_scale,
                                          float rope_theta) {
  auto device = q.device();
  unsigned int num_qo_heads = q.size(0);
  unsigned int head_dim = q.size(1);
  unsigned int kv_len, num_kv_heads;
  QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  if (kv_layout == QKVLayout::kNHD) {
    kv_len = k.size(0);
    num_kv_heads = k.size(1);
  } else {
    num_kv_heads = k.size(0);
    kv_len = k.size(1);
  }
  cudaStream_t torch_current_stream = c10::cuda::getCurrentCUDAStream(device.index());
  auto o = torch::empty_like(q);

  ParamsT params(
      static_cast<{{ dtype_q }}*>(q.data_ptr()), static_cast<{{ dtype_kv }}*>(k.data_ptr()),
      static_cast<{{ dtype_kv }}*>(v.data_ptr()), static_cast<{{ dtype_o }}*>(o.data_ptr()),
      static_cast<uint8_t*>(custom_mask.data_ptr()),
      {% if use_alibi == "true" %}static_cast<float*>(alibi_slopes->data_ptr()){% else %}nullptr{% endif %},
      kv_len, num_qo_heads, num_kv_heads, kv_layout, head_dim, window_left,
      logits_soft_cap, sm_scale, rope_scale, rope_theta);
  using AttentionVariant = typename flashinfer::CustomMaskAttention<ParamsT>;
  
  cudaError_t status = SingleDecodeWithKVCacheDispatched<{{ head_dim }}, {{ pos_encoding_mode }}, AttentionVariant>(
      params, static_cast<{{ dtype_o }}*>(tmp.data_ptr()), torch_current_stream);
  TORCH_CHECK(status == cudaSuccess,
              "SingleDecodeWithKVCache kernel launch failed, error: " +
              std::string(cudaGetErrorString(status)));

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &single_decode_with_kv_cache,
        "Single-request decode with KV-Cache operator");
}
"""

def test_single_decode_mask():
    template = jinja2.Template(single_decode_templ)
    gen_directory = flashinfer.jit.FLASHINFER_GEN_SRC_DIR
    cuda_ops_str = template.render(
        dtype_q="half",
        dtype_kv="half",
        dtype_o="half",
        head_dim=128,
        pos_encoding_mode="PosEncodingMode::kNone",
        use_sliding_window="false",
        use_logits_soft_cap="false"
    )
    flashinfer.jit.utils.write_if_different(
        gen_directory / "single_decode_with_custom_mask.cu",
        cuda_ops_str,
    )
    
    ops = flashinfer.jit.load_cuda_ops(
        "single_decode_with_custom_mask",
        [gen_directory / "single_decode_with_custom_mask.cu"],
    )
    
    print(ops.__doc__)

if __name__ == "__main__":
    test_single_decode_mask()
