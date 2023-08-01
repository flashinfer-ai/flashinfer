#include <nvbench/nvbench.cuh>
#include <flashinfer.cuh>
#include <thrust/device_vector.h>

template <typename dtype_in, typename dtype_out, int seq_len, int num_heads, int head_dim>
void bench_kv_parallel_flashattn_decode(nvbench::state &state)
{
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(num_heads * head_dim);
  thrust::device_vector<dtype_in> K(seq_len * num_heads * head_dim);
  thrust::device_vector<dtype_in> V(seq_len * num_heads * head_dim);
  thrust::device_vector<dtype_out> O(num_heads * head_dim);
  thrust::device_vector<float> m_global(num_heads * seq_len);
  thrust::device_vector<float> d_global(num_heads * seq_len);
  thrust::device_vector<int> mutex(num_heads * seq_len);

  // Provide throughput information:
  state.add_global_memory_reads<dtype_in>(num_heads * head_dim + 2 * seq_len * num_heads * head_dim, "DataSize");
  state.add_global_memory_writes<dtype_out>(num_heads * head_dim);

  state.exec([&](nvbench::launch &launch) {
    flashinfer::decoding_dispatch(
      thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
      thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
      thrust::raw_pointer_cast(m_global.data()), thrust::raw_pointer_cast(d_global.data()),
      thrust::raw_pointer_cast(mutex.data()), num_heads, seq_len, head_dim);
  });
}

auto bench_kv_parallel_flashattn_decode_f16f16_16384_32_128 = bench_kv_parallel_flashattn_decode<half, half, 16384, 32, 128>;

NVBENCH_BENCH(bench_kv_parallel_flashattn_decode_f16f16_16384_32_128);
