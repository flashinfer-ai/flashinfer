#include <thrust/device_vector.h>

#include <flashinfer.cuh>
#include <nvbench/nvbench.cuh>

template <typename dtype_in, typename dtype_out, size_t seq_len, size_t num_heads, size_t head_dim>
void bench_flashinfer_decode(nvbench::state &state) {
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(num_heads * head_dim);
  thrust::device_vector<dtype_in> K(seq_len * num_heads * head_dim);
  thrust::device_vector<dtype_in> V(seq_len * num_heads * head_dim);
  thrust::device_vector<dtype_out> O(num_heads * head_dim);
  thrust::device_vector<float> tmp(512 * num_heads * head_dim);

  // Provide throughput information:
  state.add_global_memory_reads<dtype_in>(num_heads * head_dim + 2 * seq_len * num_heads * head_dim,
                                          "DataSize");
  state.add_global_memory_writes<dtype_out>(num_heads * head_dim);

  state.exec([&](nvbench::launch &launch) {
    flashinfer::SingleDecodeWithKVCache(
        thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
        thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()),
        thrust::raw_pointer_cast(tmp.data()), num_heads, seq_len, head_dim);
  });
}

#define CREATE_BENCH_F16F16(SEQLEN, NUMHEADS, HEADDIM)                    \
  auto bench_flashinfer_decode_f16f16_##SEQLEN##_##NUMHEADS##_##HEADDIM = \
      bench_flashinfer_decode<half, half, SEQLEN, NUMHEADS, HEADDIM>;

CREATE_BENCH_F16F16(32, 32, 128);
CREATE_BENCH_F16F16(64, 32, 128);
CREATE_BENCH_F16F16(128, 32, 128);
CREATE_BENCH_F16F16(256, 32, 128);
CREATE_BENCH_F16F16(512, 32, 128);
CREATE_BENCH_F16F16(1024, 32, 128);
CREATE_BENCH_F16F16(2048, 32, 128);
CREATE_BENCH_F16F16(4096, 32, 128);
CREATE_BENCH_F16F16(8192, 32, 128);
CREATE_BENCH_F16F16(16384, 32, 128);
CREATE_BENCH_F16F16(32768, 32, 128);

NVBENCH_BENCH(bench_flashinfer_decode_f16f16_32_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_64_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_128_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_256_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_512_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_1024_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_2048_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_4096_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_8192_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_16384_32_128);
NVBENCH_BENCH(bench_flashinfer_decode_f16f16_32768_32_128);
