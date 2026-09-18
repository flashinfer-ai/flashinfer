// Round-182 (supervisor directive): fam v2 head-family path, H=12
// instantiation. Unlike the round-181 fam v1 (verbatim CAKE schedule), this
// binding compiles the EVOLVED slab H12 image (flashkda_bf16_fam2h12_gen.cu,
// a clone of flashkda_bf16_fused_m128slabh12_gen.cu with the KDA_FAM pins),
// so every measured device-side bake from R10-R88 (initial-state prefetch,
// first-chunk TMA issue placement, prep ILP, barrier-count reductions,
// lane-parallel mbarrier init, H12 compile-time head specialization) is
// retained. On top of that the kernel dead-code-eliminates all
// split/map/correction/finish/band-scan machinery via the textual
// fixup_mode/split_num_parts pins (single-kernel per-(sequence, head)
// recurrence; grid = num_seqs*heads, grid_y == 1).
#include <cstdlib>

#include "vibecuda_flashkda_tma.cuh"

// The generated standalone source declares its own fixed-width typedefs;
// isolate them so they do not collide with this TU's CUDA headers.
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
// Rename every __global__ symbol in the fam2 copy so all TUs link together.
#define kernel_flashkda_bf16_fused_m128 kernel_flashkda_fam2_h12
#define kernel_flashkda_split_scan_m128 kernel_flashkda_split_scan_m128_fam2h12
#define kernel_flashkda_split_scan_bf16_m128 kernel_flashkda_split_scan_bf16_m128_fam2h12
#define kernel_flashkda_split_lookback_m128 kernel_flashkda_split_lookback_m128_fam2h12
#define kernel_flashkda_split_out_add_m128 kernel_flashkda_split_out_add_m128_fam2h12
#define KDA_FAM
#include "vibecuda_flashkda_fam2h12_gen.cu"
// The kernel-name renames stay active for the whole TU so every reference
// below resolves to this TU's fam2 kernels. They are #undef'd at the end.
#undef KDA_FAM
#undef kernel_flashkda_bf16_fused_m128
#undef kernel_flashkda_split_scan_m128
#undef kernel_flashkda_split_scan_bf16_m128
#undef kernel_flashkda_split_lookback_m128
#undef kernel_flashkda_split_out_add_m128
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_fam2 {
using namespace kda_flash;

static_assert(THREADS == 1024);
#if defined(KDA_TF32_U2)
#if defined(KDA_PREP_PIPE2)
static_assert(SMEM_TOTAL == 218624 + 3200);
#else
static_assert(SMEM_TOTAL == 218624);
#endif
#else
#if defined(KDA_PREP_PIPE2)
static_assert(SMEM_TOTAL == 227328 + 3200 + 512);
#else
static_assert(SMEM_TOTAL == 227328 + 512);
#endif
#endif

void RunFam2H12(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
                const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
                void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
                int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
                int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
                int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream) {
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  static const bool kSmemAttrOnce = [] {
    CheckCuda(cudaFuncSetAttribute(kernel_flashkda_fam2_h12,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
              "cudaFuncSetAttribute(kernel_flashkda_fam2_h12)");
    return true;
  }();
  (void)kSmemAttrOnce;

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<128>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out_ptr=*/nullptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);
  const dim3 grid(static_cast<uint32_t>(num_seqs * num_heads), 1, 1);
  const dim3 block(THREADS, 1, 1);
#if defined(KDA_PDL)
  static const bool kPdlOn = [] {
    const char* e = std::getenv("KDA_PDL");
    return e == nullptr || std::atoi(e) != 0;
  }();
  if (kPdlOn) {
    cudaLaunchConfig_t pdl_cfg = {};
    cudaLaunchAttribute pdl_attr[1];
    pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;
    pdl_cfg.gridDim = grid;
    pdl_cfg.blockDim = block;
    pdl_cfg.dynamicSmemBytes = SMEM_TOTAL;
    pdl_cfg.stream = stream;
    pdl_cfg.attrs = pdl_attr;
    pdl_cfg.numAttrs = 1;
    CheckCuda(cudaLaunchKernelEx(
                  &pdl_cfg, kernel_flashkda_fam2_h12,
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
                  reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out, tma.out,
                  reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
                  static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
                  static_cast<float>(scale), static_cast<float>(lower_bound),
                  /*split_num_parts=*/1, /*split_state=*/nullptr, /*split_gamma=*/nullptr,
                  /*fixup_mode=*/0, /*progress_flags=*/nullptr,
                  /*map_state_bf16=*/nullptr, static_cast<int>(ft_slab)),
              "kernel_flashkda_fam2_h12 PDL launch");
    return;
  }
#endif
  kernel_flashkda_fam2_h12<<<grid, block, SMEM_TOTAL, stream>>>(
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
      reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out, tma.out,
      reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
      static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound),
      /*split_num_parts=*/1, /*split_state=*/nullptr, /*split_gamma=*/nullptr,
      /*fixup_mode=*/0, /*progress_flags=*/nullptr,
      /*map_state_bf16=*/nullptr, static_cast<int>(ft_slab));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_fam2_h12 launch");
}

}  // namespace kda_fam2
