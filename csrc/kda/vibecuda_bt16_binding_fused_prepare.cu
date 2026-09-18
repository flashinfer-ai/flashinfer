// Round-149 (supervisor directive): session-local binding TU for the vendored
// Cake generated BT=16 prepare kernel. The kernel source is a verbatim copy
// of FlashInfer's frozen BT16 prepare schedule; only the __global__ symbol is
// macro-renamed so this TU links side by side with the evolved m128/m64 TUs.
#include "vibecuda_bt16_tma.cuh"

// The generated standalone source declares its own fixed-width typedefs;
// isolate them so they do not collide with this TU's CUDA headers.
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
#define KDA_BT16_PLAN_INLINE 1
#define kernel_flashkda_bf16_bt16_prepare kernel_flashkda_bf16_bt16_prepare_fn16_local
#define CakeTensorMap flashkda_generated_CakeTensorMap
#define CakeTensorMapPack flashkda_generated_CakeTensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "vibecuda_cakebt16_prepare_gen.cu"
#undef kernel_flashkda_bf16_bt16_prepare
#undef CUtensorMap
#undef CakeTensorMapPack
#undef CakeTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_bt16_prepare_fused {

void Run(const void* q_ptr, const void* k_ptr, const void* g_ptr, const void* beta_ptr,
         const void* a_log_ptr, const void* dt_bias_ptr, const void* cu_seqlens_ptr,
         const void* cu_chunks_ptr, const void* chunk_to_seq_ptr, void* ws_qd_ptr,
         void* ws_kd_ptr, void* ws_w_ptr, void* ws_qk_ptr, void* ws_diag_ptr,
         void* descriptor_storage, int64_t prepare_descriptors, int64_t tokens,
         int64_t total_chunks, int64_t num_heads, double gate_lower_bound,
         int64_t prepare_total_ctas, int64_t cuda_stream, int64_t num_seqs) {
  using namespace kda_bt16;
  const cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_bt16_prepare_fn16_local,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL),
            "cudaFuncSetAttribute(BT16 prepare)");

  std::array<CUtensorMap, kTensorMapCount> maps{};
  if (prepare_descriptors != 0) {
    const CUtensorMap gate_map = EncodeGate16(g_ptr, tokens, num_heads);
    // The scalar-beta frozen kernel only acquires the beta descriptor slot;
    // it reads beta_logits through its raw pointer. Keep the seven-slot ABI
    // stable with a valid descriptor so H < 8 needs no pitch padding.
    maps = {EncodeQk16(q_ptr, tokens, num_heads, "q"),
            EncodeQk16(k_ptr, tokens, num_heads, "k"),
            gate_map,
            gate_map,
            EncodeFactor16(ws_qd_ptr, num_heads, total_chunks * kChunkTokens, 2, "ws_qd"),
            EncodeFactor16(ws_kd_ptr, num_heads, total_chunks * kChunkTokens, 2, "ws_kd"),
            EncodeFactor16(ws_w_ptr, num_heads, total_chunks * kChunkTokens, 1, "ws_w")};
  }
  PublishMaps(maps, descriptor_storage, stream, prepare_descriptors);
  const uint8_t* maps_bytes = static_cast<const uint8_t*>(descriptor_storage);
  constexpr size_t kStride = sizeof(CUtensorMap);

  const dim3 grid(static_cast<uint32_t>(prepare_total_ctas), 1, 1);
  const dim3 block(THREADS, 1, 1);
  kernel_flashkda_bf16_bt16_prepare_fn16_local<<<grid, block, SMEM_TOTAL, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(q_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 0 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(k_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 1 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(g_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 2 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(beta_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 3 * kStride),
      reinterpret_cast<float*>(const_cast<void*>(a_log_ptr)),
      reinterpret_cast<float*>(const_cast<void*>(dt_bias_ptr)),
      reinterpret_cast<long long*>(const_cast<void*>(cu_seqlens_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(cu_chunks_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(chunk_to_seq_ptr)),
      reinterpret_cast<__nv_bfloat16*>(ws_qd_ptr),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 4 * kStride),
      reinterpret_cast<__nv_bfloat16*>(ws_kd_ptr),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 5 * kStride),
      reinterpret_cast<__nv_bfloat16*>(ws_w_ptr),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 6 * kStride),
      reinterpret_cast<__nv_bfloat16*>(ws_qk_ptr),
      reinterpret_cast<float*>(ws_diag_ptr), static_cast<int>(total_chunks),
      static_cast<int>(num_heads), static_cast<float>(gate_lower_bound), static_cast<int>(num_seqs));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_bt16_prepare_fn16_local launch");
}

}  // namespace kda_bt16_prepare_fused
