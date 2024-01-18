#include <thrust/device_vector.h>

#include <flashinfer/prefill.cuh>

using flashinfer::QKVLayout;
using flashinfer::RotaryMode;

template <typename dtype_in, typename dtype_out, size_t rotary_mode, size_t layout>
void bench_flashinfer_single_prefill() {
  size_t qo_len = 32768;
  size_t kv_len = 32768;
  size_t num_heads = 32;
  size_t head_dim = 128;
  bool causal = false;
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(qo_len * num_heads * head_dim);
  thrust::device_vector<dtype_in> K(kv_len * num_heads * head_dim);
  thrust::device_vector<dtype_in> V(kv_len * num_heads * head_dim);
  thrust::device_vector<dtype_out> O(qo_len * num_heads * head_dim);
  // thrust::device_vector<float> tmp(512 * num_heads * head_dim);

  cudaError_t status = flashinfer::SinglePrefillWithKVCache<dtype_in, dtype_out>(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()), nullptr, num_heads,
      qo_len, kv_len, head_dim, causal, QKVLayout(layout), RotaryMode(rotary_mode), 1.f, 1e4,
      nullptr);
}

int main() { bench_flashinfer_single_prefill<half, half, 0U, 0U>(); }
