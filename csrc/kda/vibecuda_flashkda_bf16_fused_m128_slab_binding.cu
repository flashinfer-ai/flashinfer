// VibeCUDA binding TU for the compile-time SLAB-SPECIALIZED FlashKDA M128
// prefill kernel (vibecuda_flashkda_bf16_fused_m128_slab.cu: the runtime-
// regime union with the combined N=160 UMMA-4 branch removed). All
// __global__ symbols are macro-renamed at include time so this TU links
// alongside the union TU (vibecuda_flashkda_bf16_fused_m128_binding.cu) in
// one module. The host dispatches here whenever the slab regime rule
// selects ft_slab=1.
//
// Split-seq (hierarchical affine prefix) orchestration: RunM128SplitFull
// issues the multi-launch pipeline for small-BH ultra-long workloads:
//   (1) main pass, grid.y = P parts with zero initial state exporting each
//       part's additive transform (fp32 zero-init end state S_p);
//   (2) map pass, grid.y = P parts, V := 0 and identity initial state,
//       exporting each part's exact linear operator M_p (skipped at P == 2);
//   (3) dense scan composing per-part prefix (carry) states
//       c_{p+1} = S_p + M_p x c_p (an exact S_0 -> carry copy at P == 2);
//   (4) correction pass, grid.y = P-1, relaunching the same kernel with V := 0
//       and the scanned carry as initial state, accumulating the exact carry
//       contribution onto the zero-init main-pass output. Exactness follows
//       from the state recurrence being affine in the state given the tokens.
#include <algorithm>

#include "vibecuda_flashkda_tma.cuh"

// The generated standalone source declares its own fixed-width typedefs;
// isolate them so they do not collide with this TU's CUDA headers.
#define uint8_t flashkda_generated_uint8_t
#define uint16_t flashkda_generated_uint16_t
#define uint32_t flashkda_generated_uint32_t
#define uint64_t flashkda_generated_uint64_t
#define int32_t flashkda_generated_int32_t
#define int16_t flashkda_generated_int16_t
// Rename every __global__ symbol in the slab copy so both TUs link together.
#define kernel_flashkda_bf16_fused_m128 kernel_flashkda_bf16_fused_m128_slab
#define kernel_flashkda_split_scan_m128 kernel_flashkda_split_scan_m128_slab
#define kernel_flashkda_split_scan_bf16_m128 kernel_flashkda_split_scan_bf16_m128_slab
#define kernel_flashkda_split_lookback_m128 kernel_flashkda_split_lookback_m128_slab
#define kernel_flashkda_split_out_add_m128 kernel_flashkda_split_out_add_m128_slab
#include "vibecuda_flashkda_bf16_fused_m128_slab.cu"
// NOTE: the kernel-name renames stay active for the whole TU so every
// reference below (launch, cudaFuncSetAttribute, template instantiation)
// resolves to this TU's slab kernels, not the union TU's symbols. They are
// #undef'd at the end of the file.
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace kda_flash_slab {
using namespace kda_flash;

// P=2 specialization of the split pipeline: at num_parts == 2 the (map +
// scan) middle of the split pipeline is dead
// work: the map pass's exported M panels are only consumed by the scan, whose
// only EMITTED carry is c_1 = S_0 (the bf16 register-carry scan initializes
// its register carry from split_state slot 0 and emits it BEFORE the
// discarded M_1 matvec; c_2 would be the final state, which split callers
// never store). Replace both launches with the exact fp32 slab copy of S_0
// into carry slot 0, preserving the scan's seq_order task mapping
// (split_state export slot is [bid*P + 0]; carry slot 0 is
// [seq_idx*num_heads + head_idx]; the correction pass reads part_y = 0 at
// offset task*16384 with the same mapping).
// (This kernel ALSO lives in the union m128 binding TU; keep both in sync.)
__global__ __launch_bounds__(256) void kernel_flashkda_split_carry_copy_m128_slab(
    const float* __restrict__ split_state, float* __restrict__ carry,
    const int* __restrict__ seq_order, int num_heads) {
  const long long task = blockIdx.x;
  const long long seq_idx = static_cast<long long>(seq_order[task / num_heads]);
  const long long head_idx = task % num_heads;
  const float4* src = reinterpret_cast<const float4*>(split_state + task * 2 * 16384);
  float4* dst = reinterpret_cast<float4*>(
      carry + (seq_idx * static_cast<long long>(num_heads) + head_idx) * 16384);
#pragma unroll
  for (int i = 0; i < 16; i++) dst[threadIdx.x + i * 256] = src[threadIdx.x + i * 256];
}

static_assert(THREADS == 1024);

static_assert(SMEM_TOTAL == 227328 + 512);

static void LaunchM128(const TmaPointers& tma, const void* q_ptr, const void* k_ptr,
                       const void* v_ptr, const void* g_ptr, const void* beta_ptr,
                       const void* A_log_ptr, const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                       const void* seq_order_ptr, const void* initial_state_ptr, void* out_ptr,
                       void* final_state_ptr, int64_t num_seqs, int64_t num_heads,
                       int64_t use_initial_state, int64_t store_final_state, double scale,
                       double lower_bound, int64_t split_num_parts, void* split_state_ptr,
                       void* split_gamma_ptr, int64_t fixup_mode, int64_t grid_y,
                       cudaStream_t stream, int* progress_flags, void* map_state_bf16_ptr,
                       int64_t ft_slab) {
  const dim3 grid(static_cast<uint32_t>(num_seqs * num_heads), static_cast<uint32_t>(grid_y), 1);
  const dim3 block(THREADS, 1, 1);
  // Full-chunk correction stores target the scratch-out descriptor (slot 6);
  // every other launch targets out (same address for non-split callers).
  const void* out2_tma = ((fixup_mode & 1) == 1) ? tma.scratch_out : tma.out;
  kernel_flashkda_bf16_fused_m128<<<grid, block, SMEM_TOTAL, stream>>>(
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
      reinterpret_cast<__nv_bfloat16*>(out_ptr), tma.out, out2_tma,
      reinterpret_cast<__nv_bfloat16*>(final_state_ptr), static_cast<int>(num_heads),
      static_cast<int>(use_initial_state), static_cast<int>(store_final_state),
      static_cast<float>(scale), static_cast<float>(lower_bound), static_cast<int>(split_num_parts),
      reinterpret_cast<float*>(split_state_ptr), reinterpret_cast<float*>(split_gamma_ptr),
      static_cast<int>(fixup_mode), progress_flags,
      reinterpret_cast<__nv_bfloat16*>(map_state_bf16_ptr), static_cast<int>(ft_slab));
  CheckCuda(cudaGetLastError(), "kernel_flashkda_bf16_fused_m128 launch");
}

void RunM128(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
             const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
             const void* dt_bias_ptr, const void* cu_seqlens_ptr, const void* seq_order_ptr,
             const void* initial_state_ptr, void* out_ptr, void* final_state_ptr,
             void* descriptor_storage_ptr, int64_t token_count, int64_t num_seqs,
             int64_t prepare_descriptors, int64_t num_heads, int64_t use_initial_state,
             int64_t store_final_state, double scale, double lower_bound, int64_t beta_tma_rows,
             int64_t beta_tma_dim1, int64_t ft_slab, int64_t cuda_stream) {
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  kda_flash::CheckCuda(
      cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
      "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<128>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out_ptr=*/nullptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);
  LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
             seq_order_ptr, initial_state_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
             use_initial_state, store_final_state, scale, lower_bound,
             /*split_num_parts=*/1, /*split_state_ptr=*/nullptr, /*split_gamma_ptr=*/nullptr,
             /*fixup_mode=*/0, /*grid_y=*/1, stream, /*progress_flags=*/nullptr,
             /*map_state_bf16_ptr=*/nullptr, ft_slab);
}

void RunM128SplitFull(const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* g_ptr,
                      const void* beta_ptr, const void* beta_tma_ptr, const void* A_log_ptr,
                      const void* dt_bias_ptr, const void* cu_seqlens_ptr,
                      const void* seq_order_ptr, const void* initial_state_ptr, void* out_ptr,
                      void* final_state_ptr, void* descriptor_storage_ptr, int64_t token_count,
                      int64_t num_seqs, int64_t prepare_descriptors, int64_t num_heads,
                      int64_t use_initial_state, double scale, double lower_bound,
                      int64_t beta_tma_rows, int64_t beta_tma_dim1, int64_t num_parts,
                      void* split_state_ptr, void* map_state_ptr, void* carry_ptr,
                      void* split_out_ptr, void* lookback_flags_ptr, void* map_state_bf16_ptr,
                      int64_t ft_slab, int64_t cuda_stream) {
  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  kda_flash::CheckCuda(
      cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
      "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128)");
  kda_flash::CheckCuda(
      cudaFuncSetAttribute(kernel_flashkda_split_scan_bf16_m128,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           65552),  // BF16 M: 2*32KB panels + 16B mbarriers (register carry)
      "cudaFuncSetAttribute(kernel_flashkda_split_scan_bf16_m128)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointersAll<128>(
      q_ptr, k_ptr, v_ptr, g_ptr, beta_tma_ptr, out_ptr, /*scratch_out=*/split_out_ptr, token_count,
      num_heads, beta_tma_rows * beta_tma_dim1, beta_tma_dim1, descriptor_storage_ptr,
      prepare_descriptors, stream);
  const int64_t num_tasks = num_seqs * num_heads;
  // (1) Main pass: parts [0, P) with zero initial state and the real inputs;
  // the exported end states are the parts' additive transforms S_p.
  LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
             seq_order_ptr, initial_state_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
             use_initial_state, /*store_final_state=*/0, scale, lower_bound, num_parts,
             split_state_ptr, /*split_gamma_ptr=*/nullptr, /*fixup_mode=*/0,
             /*grid_y=*/num_parts, stream, /*progress_flags=*/nullptr,
             /*map_state_bf16_ptr=*/nullptr, ft_slab);
  // (2) Map pass: parts [0, P), V := 0, identity initial state; the exported
  // end states are the parts' exact linear operators M_p (their row v is the
  // evolved row basis vector e_v).
  //
  // P=2 specialization: at P == 2 the map pass is dead work. The scan's
  // carry recurrence starts from c_1 = S_0 (the main pass folded the real
  // initial state into part 0; see the init guard `fixup_mode != 0 ||
  // split_part == 0` in the kernel and the scan's "c_1 = end state of part
  // 0" comment), so carry slot 0 = c_1 needs no M_0. The scan's only
  // remaining M read produces c_2 = the FINAL state, which split callers
  // never store (store_final_state=0), so M_1's product is discarded too.
  // Skipping the map launch removes one full duplicated q/k/g-prep walk of
  // every token, and the scan is replaced by the exact S_0->carry copy
  // kernel. (This gate lives in both m128 binding TUs; keep them in sync.)
  const bool kP2Spec = (num_parts == 2);
  if (!kP2Spec) {
    LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
               seq_order_ptr, initial_state_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
               /*use_initial_state=*/0, /*store_final_state=*/0, scale, lower_bound, num_parts,
               map_state_ptr, /*split_gamma_ptr=*/nullptr, /*fixup_mode=*/2,
               /*grid_y=*/num_parts, stream, /*progress_flags=*/nullptr, map_state_bf16_ptr,
               ft_slab);
  }
  // (3) Dense scan: c_0 = initial_state, c_{p+1} = S_p + M_p x c_p; emits the
  // fp32 per-part prefix states consumed by the correction pass.
  // (3) Dense scan: c_0 = initial_state, c_{p+1} = S_p + M_p x c_p; emits the
  // fp32 per-part prefix states consumed by the correction pass.
  // Round-73 P=2 specialization: at P == 2 the only emitted carry is c_1 =
  // S_0, so the whole scan launch collapses to the exact fp32 S_0->carry copy
  // (bit-identical to the bf16 register-carry scan's c_1 emission, which
  // copies split_state slot 0 into the register carry and stores it before
  // the discarded M_1 matvec). The copy also skips the map-pass M-panel
  // export above, eliminating both middle passes.
  if (kP2Spec) {
    kernel_flashkda_split_carry_copy_m128_slab<<<dim3(static_cast<uint32_t>(num_tasks)), 256, 0,
                                                 stream>>>(
        reinterpret_cast<const float*>(split_state_ptr), reinterpret_cast<float*>(carry_ptr),
        reinterpret_cast<const int*>(seq_order_ptr), static_cast<int>(num_heads));
    CheckCuda(cudaGetLastError(), "kernel_flashkda_split_carry_copy_m128_slab launch");
  } else {
    // bf16 register-carry dense scan: the map pass exported M_p rounded to
    // bf16 and the scan stages 32KB bf16 panels with the carry kept in
    // REGISTERS (no SMEM c panels, one shfl broadcast per step); 128-thread
    // CTAs, 65552B SMEM -> 3 CTAs/SM. Requires map_state_bf16 from the
    // caller (always provided by the wrapper).
    KDA_FFI_CHECK(map_state_bf16_ptr != nullptr,
                  "run_m128_split requires a bf16 map-state scratch buffer");
    kernel_flashkda_split_scan_bf16_m128<<<dim3(static_cast<uint32_t>(num_tasks), 32), 128, 65552,
                                           stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(initial_state_ptr),
        reinterpret_cast<const float*>(split_state_ptr),
        reinterpret_cast<const __nv_bfloat16*>(map_state_bf16_ptr),
        reinterpret_cast<float*>(carry_ptr), reinterpret_cast<const int*>(seq_order_ptr),
        static_cast<int>(num_heads), static_cast<int>(num_parts),
        static_cast<int>(use_initial_state));
    CheckCuda(cudaGetLastError(), "kernel_flashkda_split_scan_bf16_m128 launch");
  }
  // (4) Correction pass: parts [1, P), V := 0, scanned carry as initial state.
  // Full chunks TMA-store the exact carry contribution into the split-out
  // scratch buffer (out2_tma); partial tail chunks accumulate into out
  // directly (scalar path), and the add-out kernel below folds the scratch
  // contribution into out over the full-chunk rows of parts [1, P).
  {
    LaunchM128(tma, q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, A_log_ptr, dt_bias_ptr, cu_seqlens_ptr,
               seq_order_ptr, carry_ptr, out_ptr, final_state_ptr, num_seqs, num_heads,
               // Round-79 contract correction: the last correction part emits the
               // exact serial final state (M_{P-1} * c_{P-1} + S_{P-1}) in place
               // into the caller's initial_state buffer (gated in the kernel;
               // only safe when the caller supplied a real initial_state).
               /*use_initial_state=*/1,
               /*store_final_state=*/(use_initial_state != 0) ? 1 : 0, scale, lower_bound,
               num_parts, split_state_ptr, /*split_gamma_ptr=*/nullptr, /*fixup_mode=*/1,
               /*grid_y=*/num_parts - 1, stream, /*progress_flags=*/nullptr,
               /*map_state_bf16_ptr=*/nullptr, ft_slab);
  }
  {
    // Fixed layout only (split gate): uniform tokens per sequence.
    const int64_t tokens_per_seq = token_count / num_seqs;
    const int64_t total_vec = token_count * num_heads * 16;
    const int threads = 256;
    const int gridsize =
        static_cast<int>(std::min<int64_t>((total_vec + threads - 1) / threads, 65535LL * 8));
    if (gridsize > 0) {
      kernel_flashkda_split_out_add_m128<<<gridsize, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(split_out_ptr),
          reinterpret_cast<__nv_bfloat16*>(out_ptr), tokens_per_seq, static_cast<int>(num_seqs),
          static_cast<int>(num_parts), static_cast<int>(num_heads));
      CheckCuda(cudaGetLastError(), "kernel_flashkda_split_out_add_m128 launch");
    }
  }
}

}  // namespace kda_flash_slab

// End-of-TU cleanup: release the kernel-name renames (see include block).
#undef kernel_flashkda_bf16_fused_m128
#undef kernel_flashkda_split_scan_m128
#undef kernel_flashkda_split_scan_bf16_m128
#undef kernel_flashkda_split_lookback_m128
#undef kernel_flashkda_split_out_add_m128
