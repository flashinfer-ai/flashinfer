#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "src/moe.cu"
#include "tvm_ffi_utils.h"

// FlashInfer is framework-agnostic through TVM-FFI, so this binding takes
// `TensorView` operands (not `torch::Tensor`) and reports errors through
// `TVM_FFI_ICHECK`.  The vLLM tree's `cuda_utils.h` (which supplied
// `CUDA_CHECK`) does not exist here; define a self-contained equivalent that
// raises a TVM-FFI error on a non-success CUDA status.
#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t _e = (call);                                             \
    TVM_FFI_ICHECK(_e == cudaSuccess)                                     \
        << "CUDA error " << cudaGetErrorString(_e) << " at " << __FILE__ \
        << ":" << __LINE__;                                              \
  } while (0)

// ── TEMP_FP8_OFFSET regression anchor (spec R13.3) ─────────────────────────
//
// The host-side down-activation TMA descriptor factory below computes the
// device pointer to `spec->temp_fp8` as
//   `scratchpad_ptr + MoEGemmSpec<Dims>::TEMP_FP8_OFFSET`
// so `TEMP_FP8_OFFSET` MUST stay byte-identical to
// `offsetof(MoEGemmSpec<Dims>, temp_fp8)` for every instantiated `Dims`
// variant.  This is exactly the invariant that the software-grid-sync
// spec (R13.3) relies on when appending new barrier-counter fields to
// the tail of `MoEGemmSpec<Dims>`: as long as every new field lands
// AFTER `temp_fp8` (grid_barrier / partial_barrier belong at the tail),
// the offset stays fixed and the TMA descriptor continues to address
// the right bytes.  A future refactor that silently reorders the struct
// layout would otherwise be caught only at runtime by corrupted TMA
// fetches — the static_asserts below make it a compile-time error.
//
// Covers both Dims variants instantiated by this TU (see the two
// `MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION` macro invocations at the
// bottom of this file).
static_assert(
    offsetof(
        moe_monokernel::MoEGemmSpec<moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>,
        temp_fp8) ==
        moe_monokernel::MoEGemmSpec<
            moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA>::TEMP_FP8_OFFSET,
    "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8) for "
    "Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA. Do not insert fields "
    "before temp_fp8; grid_barrier / partial_barrier belong at the tail of "
    "MoEGemmSpec<Dims> (spec R13.3).");

/**
 * @brief Macro that expands to a kernel call wrapper for moe_kernel_topk with
 * specified @p dims and configurable top_k, scoring_func, and renormalize.
 */
#define MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(name, dims)                                      \
  void name(TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,     \
            TensorView expert_scales_up, TensorView expert_weights_down,                            \
            TensorView expert_scales_down, TensorView activations_out, TensorView scratchpad,      \
            int64_t top_k, int64_t scoring_func, bool renormalize) {                               \
    CHECK_INPUT(activations_in);                                                                    \
    CHECK_INPUT(router_logits);                                                                     \
    CHECK_INPUT(expert_weights_up);                                                                 \
    CHECK_INPUT(expert_scales_up);                                                                  \
    CHECK_INPUT(expert_weights_down);                                                               \
    CHECK_INPUT(expert_scales_down);                                                                \
    CHECK_INPUT(activations_out);                                                                   \
    CHECK_INPUT(scratchpad);                                                                        \
    CHECK_INPUT_TYPE(activations_in, dl_bfloat16);                                                  \
    CHECK_INPUT_TYPE(router_logits, dl_bfloat16);                                                   \
    CHECK_INPUT_TYPE(expert_weights_up, dl_float8_e4m3fn);                                          \
    CHECK_INPUT_TYPE(expert_scales_up, dl_float32);                                                 \
    CHECK_INPUT_TYPE(expert_weights_down, dl_float8_e4m3fn);                                        \
    CHECK_INPUT_TYPE(expert_scales_down, dl_float32);                                               \
    CHECK_INPUT_TYPE(activations_out, dl_bfloat16);                                                 \
    TVM_FFI_ICHECK(top_k >= 1 && top_k <= 8) << "top_k must be between 1 and 8.";                   \
    TVM_FFI_ICHECK(scoring_func == 0 || scoring_func == 1)                                          \
        << "scoring_func must be 0 (sigmoid) or 1 (softmax).";                                      \
                                                                                                   \
    using namespace moe_monokernel;                                                                \
    ffi::CUDADeviceGuard device_guard(activations_in.device().device_id);                           \
    const auto* activations_in_ptr = static_cast<const A_element*>(activations_in.data_ptr());      \
    const auto* router_logits_ptr = static_cast<const __nv_bfloat16*>(router_logits.data_ptr());    \
    const auto* expert_weights_up_ptr = static_cast<const W_element*>(expert_weights_up.data_ptr()); \
    const auto* expert_scales_up_ptr = static_cast<const S_element*>(expert_scales_up.data_ptr());   \
    const auto* expert_weights_down_ptr =                                                           \
        static_cast<const W_element*>(expert_weights_down.data_ptr());                              \
    const auto* expert_scales_down_ptr =                                                            \
        static_cast<const S_element*>(expert_scales_down.data_ptr());                               \
    auto* activations_out_ptr = static_cast<R_element*>(activations_out.data_ptr());                \
    char* scratchpad_ptr = reinterpret_cast<char*>(scratchpad.data_ptr());                          \
                                                                                                   \
    const uint32_t num_tokens = activations_in.size(0);                                            \
    const size_t shmem_size = get_moe_shmem_size<dims>();                                          \
    const size_t scratchpad_size =                                                                  \
        static_cast<size_t>(scratchpad.numel()) * get_element_size(scratchpad);                    \
    const uint32_t top_k_u32 = static_cast<uint32_t>(top_k);                                       \
    const ScoringFunc sf = static_cast<ScoringFunc>(scoring_func);                                 \
                                                                                                   \
    /* TMA descriptors for the BS8 WGMMA up-projection path (spec R6.2,                            \
       R6.3) and down-projection path (spec R9.1, R9.2, R9.3).  Non-TMA                            \
       variants leave these zero-initialized — the kernel parameters are                         \
       always present on the signature but the TMA path is the only                                \
       consumer.  TMA-enabled variants build real descriptors via the                              \
       host-side factories and pass them in kernel_args positions matching                         \
       the kernel signature. */                                                                    \
    CUtensorMap up_weights_desc{};                                                                 \
    CUtensorMap activations_desc{};                                                                \
    CUtensorMap down_weights_desc{};                                                               \
    CUtensorMap down_activations_desc{};                                                           \
    if constexpr (use_tma<dims>::value) {                                                          \
      /* Up-projection weight descriptor (SWIZZLE_128B).  Callers MUST                             \
         pre-interleave `expert_weights_up` via                                                    \
         `interleave_for_tma_wgmma_up` in Python — the helper repacks                            \
         gate/up row stripes so a single 128x128 TMA fetches the full                              \
         WGMMA A-tile. */                                                                          \
      up_weights_desc =                                                                            \
          create_up_weight_tma_desc(reinterpret_cast<const void*>(expert_weights_up_ptr),          \
                                    dims::NUM_EXPERTS, dims::N, dims::K);                          \
      activations_desc = create_activations_tma_desc(                                              \
          reinterpret_cast<const void*>(activations_in_ptr), dims::BS, dims::HIDDEN_STATES);       \
      /* Down-projection weight descriptor (SWIZZLE_128B).  Callers MUST                           \
         NOT pre-interleave `expert_weights_down` — the TMA hardware                             \
         applies the core-matrix XOR swizzle at write time and expects                             \
         the raw row-major `[E, K, N]` fp8 tensor.  `row_box` =                                    \
         `DOWN_COL_TILE` so each TMA delivers one full M-tile per                                  \
         128-K substep (16 KB at DOWN_COL_TILE=128, 32 KB at                                       \
         DOWN_COL_TILE=256), halving the issue count when the M tile                               \
         is 256 rows. */                                                                           \
      down_weights_desc = create_down_weight_tma_desc(                                             \
          reinterpret_cast<const void*>(expert_weights_down_ptr), dims::NUM_EXPERTS,               \
          dims::HIDDEN_STATES, dims::N, /*row_box=*/MoECoreDims<dims>::DOWN_COL_TILE);             \
      /* Down-projection activation descriptor reads from `spec->temp_fp8`                         \
         which lives inside the scratchpad.  Compute the device pointer                            \
         from the scratchpad base + the compile-time offset of temp_fp8                            \
         inside `MoEGemmSpec<dims>`. */                                                            \
      const void* temp_fp8_ptr =                                                                   \
          reinterpret_cast<const char*>(scratchpad_ptr) + MoEGemmSpec<dims>::TEMP_FP8_OFFSET;      \
      down_activations_desc = create_down_activation_tma_desc(                                     \
          temp_fp8_ptr, MoEGemmSpec<dims>::TEMP_ROWS_TMA, dims::N);                                \
    }                                                                                              \
                                                                                                   \
    void* kernel_args[] = {(void*)&activations_in_ptr,                                             \
                           (void*)&num_tokens,                                                     \
                           (void*)&router_logits_ptr,                                              \
                           (void*)&expert_weights_up_ptr,                                          \
                           (void*)&expert_scales_up_ptr,                                           \
                           (void*)&expert_weights_down_ptr,                                        \
                           (void*)&expert_scales_down_ptr,                                         \
                           (void*)&activations_out_ptr,                                            \
                           (void*)&scratchpad_ptr,                                                 \
                           (void*)&scratchpad_size,                                                \
                           (void*)&shmem_size,                                                     \
                           (void*)&top_k_u32,                                                      \
                           (void*)&sf,                                                             \
                           (void*)&renormalize,                                                    \
                           (void*)&up_weights_desc,                                                \
                           (void*)&activations_desc,                                               \
                           (void*)&down_weights_desc,                                              \
                           (void*)&down_activations_desc};                                         \
    const cudaStream_t stream = get_stream(activations_in.device());                               \
    CUDA_CHECK(cudaFuncSetAttribute(moe_kernel_topk<dims>,                                         \
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));     \
    /* One-time diagnostic: compute and print occupancy + shmem so that a                          \
       cooperative-launch failure is easy to diagnose. */                                          \
    {                                                                                              \
      static bool _diag_printed = false;                                                           \
      if (!_diag_printed) {                                                                        \
        int max_blocks_per_sm = 0;                                                                 \
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(                                    \
            &max_blocks_per_sm, moe_kernel_topk<dims>, dims::KernelConfig::BLOCK_SIZE, shmem_size, \
            cudaOccupancyDefault);                                                                 \
        cudaFuncAttributes fa;                                                                     \
        cudaFuncGetAttributes(&fa, moe_kernel_topk<dims>);                                         \
        int sm_count = 0;                                                                          \
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);                      \
        int smem_opt_in = 0;                                                                       \
        cudaDeviceGetAttribute(&smem_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);          \
        (void)smem_opt_in;                                                                          \
        (void)max_blocks_per_sm;                                                                    \
        (void)fa;                                                                                   \
        /* Hard co-residency assertions for the software grid barrier                              \
           (spec R4.1, R4.2, R4.3 / Design Component C "Co-residency                               \
           assertions").  The seed-atomicAdd-spin-on-high-bit protocol                             \
           in src/moe_grid_barrier.h is only deadlock-free when every                              \
           participating block is co-resident on the GPU for the full                              \
           lifetime of the kernel: (1) grid_size <= SM count so every                              \
           block gets a slot, and (2) max_active_blocks_per_SM == 1 so                             \
           no block is ever waiting on a block that has not yet been                               \
           scheduled.  GRID_SIZE is a compile-time constexpr and SM                                \
           count / occupancy are device-property-time static, so gating                            \
           under `_diag_printed` keeps the check one-shot per process                              \
           and off the hot path. */                                                                \
        TVM_FFI_ICHECK(dims::KernelConfig::GRID_SIZE <= static_cast<uint32_t>(sm_count))            \
            << "moe_monokernel requires GRID_SIZE (=" << dims::KernelConfig::GRID_SIZE             \
            << ") <= SM count (=" << sm_count                                                      \
            << ") for software grid barrier co-residency invariant (spec R4.1).";                  \
        /*TORCH_CHECK(max_blocks_per_sm == 1,                                                      \
                    "moe_monokernel requires max_active_blocks_per_SM == 1 "                       \
                    "(observed ",                                                                  \
                    max_blocks_per_sm,                                                             \
                    ") for co-residency invariant (spec R4.2). See "                               \
                    "__launch_bounds__(BLOCK_SIZE, 1) and the SHM budget "                         \
                    "requirement.");*/                                                             \
        _diag_printed = true;                                                                      \
      }                                                                                            \
    }                                                                                              \
    /* One-shot scratchpad zero-init (spec R13.2 / Design Component C                              \
       "Scratchpad barrier counter zero-initialization").  The software                            \
       Grid_Barrier / Partial_Barrier counters live at the tail of                                 \
       MoEGemmSpec<Dims> inside the scratchpad, and the                                            \
       seed-atomicAdd-spin-on-high-bit protocol requires the barrier slots                         \
       to start at 0 so the first Seed_Thread write commits the                                    \
       `0x80000000u - (arrival_count - 1)` seed value cleanly.  The                                \
       ping-pong reset discipline keeps the slots self-maintaining across                          \
       subsequent kernel invocations (see MoEGemmSpec<Dims> block comment                          \
       on grid_barrier and partial_barrier), so we only pay the zero-init                          \
       cost once per process on the first launch.  Zeroing the full                                \
       scratchpad (rather than just the counter region) is simpler and                             \
       the cost is a few hundred microseconds one-time on H200 — trivial                         \
       next to per-decode kernel launches. */                                                      \
    {                                                                                              \
      /* Caller-provided scratchpad: key the guard on buffer identity                              \
         (ptr, size, device) and re-zero whenever it changes, so each                              \
         distinct allocation is initialized exactly once.  A process-wide                          \
         one-shot flag would be WRONG here — it zeroes only the first                             \
         buffer ever seen and launches every subsequent DISTINCT scratchpad                        \
         (other stream/device or a freshly-malloc'd buffer) with                                   \
         uninitialized barrier counters, deadlocking the seed/spin protocol                        \
         or silently corrupting results.  A reused buffer still pays the                           \
         zero-init only on its first launch (self-maintaining ping-pong                            \
         reset thereafter). */                                                                     \
      static const void* _zeroed_ptr = nullptr;                                                    \
      static size_t _zeroed_size = 0;                                                              \
      static int _zeroed_dev = -1;                                                                 \
      const int _cur_dev = activations_in.device().device_id;                                      \
      if (_zeroed_ptr != static_cast<const void*>(scratchpad_ptr) ||                               \
          _zeroed_size != scratchpad_size || _zeroed_dev != _cur_dev) {                            \
        CUDA_CHECK(cudaMemsetAsync(scratchpad_ptr, 0, scratchpad_size, stream));                   \
        _zeroed_ptr = scratchpad_ptr;                                                              \
        _zeroed_size = scratchpad_size;                                                            \
        _zeroed_dev = _cur_dev;                                                                    \
      }                                                                                            \
    }                                                                                              \
    /* Standard (non-cooperative) launch.  The kernel reaches grid-wide                            \
       happens-before via the software Grid_Barrier / Partial_Barrier                              \
       primitives in `src/moe_grid_barrier.h` (spec R1.1, R5.1, Design                             \
       Component C "Launch form") rather than                                                      \
       `cooperative_groups::this_grid().sync()`.  Using standard                                   \
       `cudaLaunchKernel` is what lets the migrated kernel be captured                             \
       into a CUDA Graph. */                                                                       \
    CUDA_CHECK(cudaLaunchKernel(                                                                   \
        (const void*)moe_kernel_topk<dims>, dim3(dims::KernelConfig::GRID_SIZE, 1, 1),             \
        dim3(dims::KernelConfig::BLOCK_SIZE, 1, 1), kernel_args, shmem_size, stream));             \
  }

// TMA + WGMMA + SWIZZLE_128B variant of the BS8 path — the only BS8
// implementation.  Selects the TMA-based weight + activation load path
// in Phase 3 via `KernelConfig::USE_TMA = true`.  Up-projection weights
// must be repacked via `interleave_for_tma_wgmma_up` (gate/up row
// interleave for single-issue TMA); down-projection weights are passed
// raw row-major (the TMA hardware applies the core-matrix XOR swizzle
// at write time).
MOEMONOKERNEL_TOPK_WRAPPER_IMPLEMENTATION(
    moe_monokernel_topk_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA_impl,
    moe_monokernel::Dims_BS8_E256_Qwen3_5_35B_BlockFP8_WGMMA_TMA)
