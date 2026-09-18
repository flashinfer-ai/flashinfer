// Round-149 (supervisor directive): session-local binding TU for the vendored
// Cake generated BT=16 chain kernel, S9 pipeline schedule (9 stages, m64
// value-split). The kernel source is a verbatim copy of FlashInfer's frozen
// schedule; only the __global__ symbol is macro-renamed so this TU links side
// by side with the other BT16 chain schedule TUs.
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
#define kernel_flashkda_bf16_bt16_chain_m64 kernel_flashkda_bf16_bt16_chain_m64_fn16_s9_local
#define CakeTensorMap flashkda_generated_CakeTensorMap
#define CakeTensorMapPack flashkda_generated_CakeTensorMapPack
#define CUtensorMap flashkda_generated_CUtensorMap
#include "vibecuda_cakebt16_chain_s9_gen.cu"
#undef kernel_flashkda_bf16_bt16_chain_m64
#undef CUtensorMap
#undef CakeTensorMapPack
#undef CakeTensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_bt16_chain_fn16_s9 {

// Full launch entry with explicit grid_x = 2 * num_seqs * num_heads CTAs.
void RunFull(const void* ws_qd_ptr, const void* ws_kd_ptr, const void* ws_w_ptr,
             const void* ws_qk_ptr, const void* ws_diag_ptr, const void* v_ptr,
             const void* cu_seqlens_ptr, const void* cu_chunks_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage, int64_t prepare_descriptors, int64_t tokens,
             int64_t total_chunks, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, int64_t grid_x, int64_t cuda_stream,
             int64_t num_seqs) {
  using namespace kda_bt16;
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_bt16_chain_m64_fn16_s9_local,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL),
            "cudaFuncSetAttribute(BT16 chain s9)");

  std::array<CUtensorMap, kTensorMapCount> maps{};
  if (prepare_descriptors != 0) {
    maps = {EncodeFactor16(ws_qd_ptr, num_heads, total_chunks * kChunkTokens, 2, "ws_qd"),
            EncodeFactor16(ws_kd_ptr, num_heads, total_chunks * kChunkTokens, 2, "ws_kd"),
            EncodeFactor16(ws_w_ptr, num_heads, total_chunks * kChunkTokens, 2, "ws_w"),
            EncodeQkWs(ws_qk_ptr, num_heads, total_chunks),
            EncodeDiagWs(ws_diag_ptr, num_heads, total_chunks),
            EncodeValue64x16(v_ptr, tokens, num_heads),
            EncodeOut64x16(out_ptr, tokens, num_heads)};
  }
  PublishMaps(maps, descriptor_storage, stream, prepare_descriptors);
  const uint8_t* maps_bytes = static_cast<const uint8_t*>(descriptor_storage);
  constexpr size_t kStride = sizeof(CUtensorMap);

  const dim3 grid(static_cast<uint32_t>(grid_x), 1, 1);
  const dim3 block(THREADS, 1, 1);
  kernel_flashkda_bf16_bt16_chain_m64_fn16_s9_local<<<grid, block, SMEM_TOTAL, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(ws_qd_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 0 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(ws_kd_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 1 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(ws_w_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 2 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(ws_qk_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 3 * kStride),
      reinterpret_cast<float*>(const_cast<void*>(ws_diag_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 4 * kStride),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(v_ptr)),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 5 * kStride),
      reinterpret_cast<long long*>(const_cast<void*>(cu_seqlens_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(cu_chunks_ptr)),
      reinterpret_cast<int*>(const_cast<void*>(seq_order_ptr)),
      reinterpret_cast<__nv_bfloat16*>(const_cast<void*>(initial_state_ptr)),
      reinterpret_cast<__nv_bfloat16*>(out_ptr),
      reinterpret_cast<const flashkda_generated_CakeTensorMap*>(maps_bytes + 6 * kStride),
      reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
      static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
      static_cast<float>(scale), static_cast<int>(num_seqs));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_bt16_chain_m64_fn16_s9_local launch");
}

}  // namespace kda_bt16_chain_fn16_s9
