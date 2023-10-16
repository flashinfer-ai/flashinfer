#include <thrust/device_vector.h>

#include <flashinfer/prefill.cuh>
#include <nvbench/nvbench.cuh>

using flashinfer::QKVLayout;
using flashinfer::RotaryMode;

template <typename dtype_in, typename dtype_out>
void bench_flashinfer_single_prefill(nvbench::state &state) {
  size_t qo_len = state.get_int64("seq_len");
  size_t kv_len = state.get_int64("seq_len");
  size_t num_qo_heads = state.get_int64("num_qo_heads");
  size_t num_kv_heads = state.get_int64("num_kv_heads");
  size_t head_dim = state.get_int64("head_dim");
  size_t rotary_mode = state.get_int64("rotary_mode");
  size_t layout = state.get_int64("layout");
  bool causal = state.get_int64("causal");
  // Allocate input data:
  thrust::device_vector<dtype_in> Q(qo_len * num_qo_heads * head_dim);
  thrust::device_vector<dtype_in> K(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_in> V(kv_len * num_kv_heads * head_dim);
  thrust::device_vector<dtype_out> O(qo_len * num_qo_heads * head_dim);
  // thrust::device_vector<float> tmp(512 * num_heads * head_dim);

  // Provide throughput information:
  state.add_global_memory_reads<dtype_in>(
      (2 * qo_len * num_qo_heads + 2 * kv_len * num_kv_heads) * head_dim, "Read");
  state.add_global_memory_writes<dtype_out>(qo_len * num_qo_heads * head_dim, "Write");

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
    timer.start();
    cudaError_t status = flashinfer::SinglePrefillWithKVCache<dtype_in, dtype_out>(
        thrust::raw_pointer_cast(Q.data()), thrust::raw_pointer_cast(K.data()),
        thrust::raw_pointer_cast(V.data()), thrust::raw_pointer_cast(O.data()), nullptr,
        num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal, QKVLayout(layout),
        RotaryMode(rotary_mode), 1.f, 1e4, launch.get_stream());
    if (status != cudaSuccess) {
      state.skip("CUDA error: " + std::string(cudaGetErrorString(status)));
    }
    timer.stop();
  });
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define BENCH_FLASHINFER_PREFILL(dtype_in, dtype_out)                                           \
  auto bench_flashinfer_single_prefill_##dtype_in##_##dtype_out##_ =                            \
      bench_flashinfer_single_prefill<dtype_in, dtype_out>;                                     \
  NVBENCH_BENCH(bench_flashinfer_single_prefill_##dtype_in##_##dtype_out##_)                    \
      .set_name(("bench_flashinfer_single_prefill_" STR(dtype_in) "_" STR(dtype_out) "_"))      \
      .add_int64_axis("seq_len", {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}) \
      .add_int64_axis("num_heads", {32})                                                        \
      .add_int64_axis("head_dim", {128})                                                        \
      .add_int64_axis("causal", {0, 1})                                                         \
      .add_int64_axis("layout", {0, 1})                                                         \
      .add_int64_axis("rotary_mode", {0, 1})

BENCH_FLASHINFER_PREFILL(half, half);
