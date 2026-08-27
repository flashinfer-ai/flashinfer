// VibeCUDA binding TU for the generated FlashKDA M64 prefill kernel
// (vibecuda_flashkda_bf16_fused_m64.cu). The M64 variant is specialized for
// fixed-layout N=1, H=64 with a two-CTA grid.
#include "vibecuda_flashkda_tma.cuh"

#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#include "vibecuda_flashkda_bf16_fused_m64.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_flash {

static_assert(THREADS == 1024);
static_assert(SMEM_TOTAL >= 219136);

void RunM64(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
            const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
            const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
            const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
            void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
            int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
            int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
            int64_t beta_tma_dim1, int64_t cuda_stream) {
  if (!(num_heads > 0 && num_heads % 8 == 0)) {
    throw std::runtime_error("kda_flash: the M64 FlashKDA variant requires H % 8 == 0");
  }
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  kda_flash::CheckCuda(
      cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m64,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
      "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m64)");

  const dim3 grid(static_cast<uint32_t>(2 * num_seqs * num_heads), 1, 1);
  const dim3 block(THREADS, 1, 1);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<64>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out_ptr=*/nullptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);

  kernel_flashkda_bf16_fused_m64<<<grid, block, kSmemBytes, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(q_ptr)), tma.q,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(k_ptr)), tma.k,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v_ptr)), tma.v,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(g_ptr)), tma.g,
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(beta_ptr)), tma.beta,
      reinterpret_cast<float*>(const_cast<void*>(A_log_ptr)),
      reinterpret_cast<float*>(const_cast<void*>(dt_bias_ptr)),
      reinterpret_cast<long long*>(const_cast<void*>(cu_seqlens_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(seq_order_ptr)),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(initial_state_ptr)),
      reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out,
      reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
      static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_fused_m64 launch");
}

}  // namespace kda_flash
