#ifndef MOE_TMA_H
#define MOE_TMA_H

#pragma once

// Host-side factory declarations for TMA (Tensor Memory Accelerator)
// `CUtensorMap` descriptors used by the BS8 WGMMA up-projection path.
//
// These descriptors are built once per kernel launch from the torch-binding
// wrapper and passed to the device as `__grid_constant__ CUtensorMap const`
// kernel parameters.  They describe a tiled 2D view of the full
// `[E, 2*N, K]` fp8 up-projection weight tensor and the full
// `[BS, K_hidden]` bf16 activation tensor respectively.
//
// The definitions live in `moe_tma.cu` / `moe_tma.cpp` (host-side translation
// unit) and call `cuTensorMapEncodeTiled` from the CUDA Driver API, which
// requires CUDA 12.0 or later.

#include <cstdint>

// CUDA Driver API: provides `CUtensorMap` and `cuTensorMapEncodeTiled`.
// `CUtensorMap` itself was introduced in CUDA 12.0.
#include <cuda.h>

// Device-side PTX wrappers (`tma_load_2d`, `mbarrier_*`, ...) used by the
// device-side TMA load helpers below.  Safe to include from any
// nvcc-compiled TU (pulls in `<cuda/pipeline>`, `<cuda_bf16.h>`, etc.);
// not safe from pure host `.cpp` TUs.  All current includers of this
// header (`moe_tma.cu`, `tma_descriptor_factory_test.cu`, and the device
// TUs that will consume the loaders) are CUDA-compiled.
#include "ptx_utils.h"

// `A_element` (= `__nv_bfloat16`) is referenced in the
// `moe_load_full_bf16_input` template signature below.  `moe_interface.h`
// is the canonical declaration site for the kernel's element-type
// aliases and is a lightweight header (no
// `INSIDE_MONOMOE_IMPLEMENTATION` guard), so including it here is safe
// for every existing consumer of `moe_tma.h` (the host-only `moe_tma.cu`, the
// device-side `.cu` files in the kernel's whole-program-inlined chain, and the
// standalone `tma_descriptor_factory_test.cu`).
#include "moe_interface.h"

// Build-time guard: TMA descriptor encoding requires CUDA toolkit 12.0+.
// The CUDA Driver API exposes `CUtensorMap` only on 12.0+, so fail fast
// with a clear message on older toolchains.
#if defined(CUDA_VERSION) && (CUDA_VERSION < 12000)
#error "moe_tma.h requires CUDA 12.0 or later for CUtensorMap / cuTensorMapEncodeTiled"
#endif

namespace monomoe {

/**
 * @brief Build a `CUtensorMap` describing the fp8 up-projection weight
 *        tensor `[E, 2*N, K]` for TMA tile loads in Phase 3.
 *
 * Uses `CU_TENSOR_MAP_SWIZZLE_128B`, paired with the gate/up Python
 * pre-interleave `interleave_for_tma_wgmma_up` that packs each 128-row
 * WGMMA tile into a contiguous 128-row block of the weight tensor.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_UINT8` with
 * `rank = 2`, `globalDim = [K, num_experts * 2 * N]`,
 * `boxDim = [128, 128]`, `elementStrides = [1, 1]`,
 * `CU_TENSOR_MAP_INTERLEAVE_NONE`, `CU_TENSOR_MAP_SWIZZLE_128B`,
 * `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.
 *
 * The TMA hardware applies the 8-row × 128-byte core-matrix XOR
 * swizzle at write time, so the caller MUST pass the weight tensor
 * through `interleave_for_tma_wgmma_up` — the helper repacks gate/up
 * row stripes into the layout expected by this descriptor.  The
 * matching WGMMA A-descriptor uses `swizzle=1`, `A_LBO=16`,
 * `A_SBO=1024` (CUTLASS Major::K B128 layout).
 *
 * @param weights_ptr   Device pointer to the base of the pre-interleaved
 *                      `expert_weights_up` (fp8 e4m3 values, row-major
 *                      `[E, 2*N, K]`).
 * @param num_experts   Number of experts (`E`).
 * @param N             Half of the fused gate+up intermediate size (so the
 *                      flattened row axis has length `num_experts * 2 * N`).
 * @param K             Hidden size / reduction dimension.
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "up-projection weights" as the failing tensor.
 */
CUtensorMap create_up_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t N,
                                      uint32_t K);

/**
 * @brief Build a `CUtensorMap` describing the bf16 activation tensor
 *        `[BS, K_hidden]` for TMA tile loads in Phase 3.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_BFLOAT16` with
 * `rank = 2`, `globalDim = [K_hidden, batch_size_cap]`,
 * `boxDim = [128, 8]`, `elementStrides = [1, 1]`,
 * `CU_TENSOR_MAP_INTERLEAVE_NONE`, `CU_TENSOR_MAP_SWIZZLE_NONE`,
 * `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.
 *
 * @param activations_ptr Device pointer to the base of `activations_in`
 *                        (bf16 values, row-major `[BS, K_hidden]`).
 * @param batch_size_cap  Maximum batch size the kernel will process
 *                        (used as the outer axis of `globalDim`).
 * @param K_hidden        Hidden size of the activation tensor.
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "activations" as the failing tensor.
 */
CUtensorMap create_activations_tma_desc(const void* activations_ptr, uint32_t batch_size_cap,
                                        uint32_t K_hidden);

/**
 * @brief Build a `CUtensorMap` describing the fp8 down-projection weight
 *        tensor `[E, K, N]` for TMA tile loads in Phase 4.
 *
 * Uses `CU_TENSOR_MAP_SWIZZLE_128B`.  The TMA hardware applies the
 * 8-row × 128-byte core-matrix XOR swizzle at write time, so the caller
 * MUST NOT pre-interleave the weight tensor — the raw row-major
 * `[E, K, N]` fp8 tensor is expected.  The matching WGMMA A-descriptor
 * uses `swizzle=1`, `A_LBO=16`, `A_SBO=1024` (CUTLASS Major::K B128
 * layout).
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_UINT8` with
 * `rank = 2`, `globalDim = [N, num_experts * K]`,
 * `boxDim = [128, row_box]`, `elementStrides = [1, 1]`,
 * `CU_TENSOR_MAP_INTERLEAVE_NONE`, `CU_TENSOR_MAP_SWIZZLE_128B`,
 * `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.
 *
 * `row_box` controls how many rows (=output cols) each
 * `cp.async.bulk.tensor.2d` issue delivers.  Valid values:
 *   * 128 — one 128×128 fp8 atom = 16 KB per issue (legacy).  The
 *     kernel's M-axis loop issues `DOWN_COL_TILE / 128` TMAs back-to-
 *     back to populate one 128-K K-substep.
 *   * 256 — two stacked 128×128 atoms = 32 KB per issue.  Halves the
 *     issue count when `DOWN_COL_TILE = 256` (one TMA covers the full
 *     M tile per K-substep).  The two 128-row sub-atoms still observe
 *     the SWZ128 byte layout — the 256-row box simply concatenates two
 *     sub-atoms along the M axis, so the consumer-side WGMMA A
 *     descriptor (which addresses one 128-row sub-atom per WGMMA) is
 *     unchanged.
 * The maximum legal `row_box` is 256 (TMA `boxDim` is capped at 256
 * per axis).
 *
 * NOTE on axis ordering: unlike the up-projection weight descriptor
 * (where `K` is the innermost reduction axis and the outer axis is
 * `num_experts * 2 * N` row-major rows), the down-projection descriptor
 * flips this — here `N` is the innermost reduction axis (the TMA tile
 * covers 128 N-elements per row) and the outer axis is the flattened
 * `expert_id * K + k_row` row index.  This matches the down-proj's GM
 * layout `[E, K, N]` where each expert's `K` output rows are each a
 * contiguous `N`-element fp8 vector, and the down-proj's WGMMA consumes
 * 128 N-elements at a time as the reduction dimension.
 *
 * SHM must be 1024-byte aligned (satisfied by `alignas(1024)` on
 * `w_down_wgmma` in `moe_internal.h`).
 *
 * @param weights_ptr   Device pointer to the base of `expert_weights_down`
 *                      (fp8 e4m3 values, row-major `[E, K, N]`).
 * @param num_experts   Number of experts (`E`).
 * @param K             Down-projection output dimension (= hidden size).
 * @param N             Down-projection reduction dimension (= up-proj
 *                      intermediate output size).
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "down-projection weights" as the failing tensor.
 */
CUtensorMap create_down_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t K,
                                        uint32_t N, uint32_t row_box = 128u);

/**
 * @brief Build a `CUtensorMap` describing the fp8 intermediate-activation
 *        tensor `spec->temp_fp8[TEMP_ROWS, N]` for TMA tile loads in
 *        Phase 4.
 *
 * The returned descriptor targets `CU_TENSOR_MAP_DATA_TYPE_UINT8` with
 * `rank = 2`, `globalDim = [N, temp_rows]`, `boxDim = [128, 8]`,
 * `elementStrides = [1, 1]`, `CU_TENSOR_MAP_INTERLEAVE_NONE`,
 * `CU_TENSOR_MAP_SWIZZLE_NONE`, `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`, and
 * `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`.
 *
 * Axis ordering: innermost = `N` (128 fp8 K-values per token row),
 * outer = `sorted_slot` row index into `spec->temp_fp8`.  `boxDim[1] = 8`
 * sets the maximum rows per bulk TMA issue; the actual per-issue row
 * count is controlled at runtime by the instruction's dynamic row-count
 * operand and equals `routed_token_count ∈ [1, 8]` for the current
 * expert.
 *
 * The `temp_fp8` tensor is written by the Phase-3 up-projection epilogue
 * into a reorganized `[expert, token]` layout, so each expert's
 * routed tokens occupy a contiguous slab of rows
 * `[expert_slot_start[id], expert_slot_start[id] + routed_token_count[id])`
 * that this descriptor can fetch in a single bulk TMA.
 *
 * @param activations_ptr Device pointer to the base of `spec->temp_fp8`
 *                        (fp8 e4m3 values, row-major `[TEMP_ROWS, N]`).
 * @param temp_rows       Outer-axis length `Dims::BS * MAX_TOPK` (e.g. 64
 *                        for `BS=8, MAX_TOPK=8`).
 * @param N               Down-projection reduction dimension.
 * @return A 128-byte POD `CUtensorMap`, safe to pass by value as a
 *         `__grid_constant__ CUtensorMap const` kernel parameter.
 *
 * On `cuTensorMapEncodeTiled` failure, raises `TORCH_CHECK` identifying
 * "down-projection activations" as the failing tensor.
 */
CUtensorMap create_down_activation_tma_desc(const void* activations_ptr, uint32_t temp_rows,
                                            uint32_t N);

// ─── Device-side TMA load helpers ────────────────────────────────────────
//
// Thin wrappers around the low-level PTX TMA primitive `tma_load_2d`
// (in `ptx_utils.h`) that bake in the MoE up-projection path's coordinate
// convention and per-stripe semantics.  All helpers below are `__device__
// __forceinline__` and are expected to inline fully into the kernel's TMA
// launcher thread.
//
// ── Caller contract (critical) ──
// Every helper in this group MUST be invoked by exactly one thread in the
// block — the "TMA launcher thread" (warp 8, lane 0 in the BS8 WGMMA path,
// selected via `is_tma_launcher_thread<Dims>()`).  These helpers do NOT
// internally gate on `threadIdx`; calling them from multiple threads will
// issue duplicate `cp.async.bulk.tensor.2d` instructions and corrupt the
// barrier's transaction-bytes accounting.  The caller is also responsible
// for pre-arming the target mbarrier exactly once with the total expected
// byte count before issuing the TMA(s) that target it.

/**
 * @brief Single-issue 128×128 up-projection weight TMA loader.
 *
 * A single `cp.async.bulk.tensor.2d` issue fetches the full 128-row ×
 * 128-K fp8 weight tile into one 16 KB SHM slot (byte-aligned to the
 * 1024-B swizzle atom).  Used on the `USE_TMA` path with the
 * `interleave_for_tma_wgmma_up` Python pre-interleave.
 *
 * Expected GM layout (written by the pre-interleave):
 *   For every expert and every 64-gate-row block k in [0, N/64):
 *     new_rows[128k +  0 .. 128k + 32) = gate[64k     .. 64k + 32)
 *     new_rows[128k + 32 .. 128k + 64) =   up[64k     .. 64k + 32)
 *     new_rows[128k + 64 .. 128k + 96) = gate[64k + 32 .. 64k + 64)
 *     new_rows[128k + 96 .. 128k +128) =   up[64k + 32 .. 64k + 64)
 *
 * With this packing, `base_row_up` (the first WG0-gate row inside the
 * expert, in ORIGINAL gate coordinates) of 64k maps to interleaved
 * GM row offset `2 * base_row_up = 128k` within the expert, i.e. the
 * start of the k-th 128-row WGMMA tile.  The descriptor's flattened
 * outer-axis coordinate for the expert is
 *   `expert_id * 2 * N + 2 * base_row_up`.
 *
 * Caller contract:
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     typically warp 8 lane 0).  Does NOT gate on `threadIdx`.
 *   - `bar_smem_ptr` MUST be pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 16384)` BEFORE calling.
 *   - `dest_slot_smem_ptr` MUST be 1024-B aligned (the swizzle atom
 *     alignment) and point at the base of `shm->w_wgmma[slot][0][0]`.
 *     Callers that use the `alignas(1024)` `TinyDataWGMMA_TMA` layout
 *     already satisfy this.
 *   - `desc` MUST be the descriptor produced by
 *     `create_up_weight_tma_desc`.
 *   - Valid inputs: `0 ≤ expert_id < num_experts`,
 *     `0 ≤ base_row_up + 64 ≤ N`, `0 ≤ k_start` with
 *     `k_start + 128 ≤ K`.
 */
__device__ __forceinline__ void tma_load_up_wgmma_tile(
    CUtensorMap const& desc, std::uint32_t expert_id, std::uint32_t N, std::uint32_t base_row_up,
    std::uint32_t k_start, void* dest_slot_smem_ptr, std::uint64_t* bar_smem_ptr) {
  // Flattened outer-axis coordinate into the interleaved
  // `[num_experts * 2 * N, K]` descriptor view.  `2 * base_row_up`
  // skips `base_row_up` original gate rows (plus their paired up rows)
  // inside the expert's interleaved slab.
  const std::uint32_t global_row = expert_id * 2u * N + 2u * base_row_up;
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_slot_smem_ptr, bar_smem_ptr);
}

/**
 * @brief Issue one TMA load for the bf16 activation tile used by Phase-3
 *        streaming quantize.
 *
 * Issues one `cp.async.bulk.tensor.2d` to fetch one 128-K × 8-token
 * rectangular box from the full `[batch_size_cap, K_hidden]` activation
 * tensor into a 16-B aligned SHM
 * region (2048 B total: 8 tokens × 128 K × 2 B per bf16) using exactly
 * one `cp.async.bulk.tensor.2d` instruction.
 *
 * Coordinate convention for the activation descriptor (built by
 * `create_activations_tma_desc`): the innermost axis is K and the outer
 * axis is the token index.  Every K-step `s` fetches all
 * 8 tokens starting at token 0, so the tile coordinates are
 * `(k_start, 0) = (s * K_STEP, 0)`.  Per `cuTensorMapEncodeTiled`, the
 * coordinate operand order matches that of `globalDim`, so this wrapper
 * passes `coord0 = k_start` and `coord1 = 0` at the PTX boundary.
 *
 * Activations are expert-invariant: the coordinates depend only
 * on `k_start` and are the same across the outer expert loop.
 *
 * Caller contract (critical):
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     typically warp 8 lane 0).  This function does NOT gate on
 *     `threadIdx`; calling from multiple threads issues duplicate TMAs
 *     and corrupts the barrier's transaction-bytes accounting.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 2048)` BEFORE calling
 *     this function.  This function itself does NOT call
 *     `mbarrier_arrive_expect_tx`; it only issues the TMA load.
 *   - `dest_smem_ptr` MUST be 16-B aligned and point at a 2048-B
 *     activation-tile slot in SHM.  Used by
 *     `moe_load_full_bf16_input` to populate
 *     `shm->bf16_in_full[kblk][0][0]` (one 8×128 BF16 box per call).
 *   - `desc` MUST be the descriptor produced by
 *     `create_activations_tma_desc`, typically passed to the kernel as
 *     a `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs: `0 ≤ k_start` with `k_start + 128 ≤ K_hidden`.
 *
 * @param desc          Host-built activation TMA descriptor
 *                      (`__grid_constant__`).
 * @param k_start       Innermost-axis starting K column (multiple of 128).
 * @param dest_smem_ptr 16-B aligned SHM destination pointer to a
 *                      2048-B activation-tile slot (e.g.
 *                      `shm->bf16_in_full[kblk][0][0]`).
 * @param bar_smem_ptr  16-B aligned SHM mbarrier pre-armed by the caller
 *                      with `expect_tx = 2048`.
 */
__device__ __forceinline__ void tma_load_bf16_input_tile(CUtensorMap const& desc,
                                                         std::uint32_t k_start, void* dest_smem_ptr,
                                                         std::uint64_t* bar_smem_ptr) {
  // Descriptor axis order (innermost first): coord0 = K, coord1 = token.
  // We always fetch all 8 tokens starting at token 0.
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/0u, dest_smem_ptr, bar_smem_ptr);
}

/**
 * @brief Issue a full per-block BF16 input tile TMA load via
 *        `K_BLOCKS_TOTAL` back-to-back 128-K-wide bulk TMA issues
 *        (Phase-1 routing-window prefetch).
 *
 * Covers the entire `[Dims::BS, Dims::HIDDEN_STATES]` BF16 activation
 * tile — `K_BLOCKS_TOTAL = Dims::HIDDEN_STATES / 128` issues of the
 * existing `tma_load_bf16_input_tile` helper, one per 128-K substep
 * `k_start ∈ {0, 128, 256, …, (K_BLOCKS_TOTAL - 1) * 128}`.  Reuses the
 * same `CUtensorMap` produced by `create_activations_tma_desc` (no new
 * descriptor) and the same `(coord0 = k_start, coord1 = 0)` coordinate
 * convention.
 *
 * Each per-substep issue writes a `BS × 128`-element BF16 box into a
 * disjoint slab of the row-major `[BS][HIDDEN_STATES]` SHM destination
 * at offset `k_start * sizeof(A_element)` along the inner axis (i.e.
 * `&dest[0][k_start]`); the natural row-major layout of the destination
 * places token `t`'s K-stripe `[k_start, k_start + 128)` at SHM offset
 * `(t * HIDDEN_STATES + k_start) * sizeof(A_element)`, matching what
 * the activation descriptor's `boxDim = (128, 8)` produces.
 *
 * Why Option B (16 × 2 KB issues) instead of one 32 KB issue:
 * `cuTensorMapEncodeTiled` caps per-axis `boxDim` at 256 elements
 * regardless of swizzle mode, so a single-issue load with innermost
 * axis = `HIDDEN_STATES = 2048` is rejected by the Driver API.
 *
 * This helper consumes only a `CUtensorMap` and a typed SHM reference;
 * it deliberately does NOT depend on `MoE_SHM` or `MoECoreDims` so that
 * it stays usable from `moe_tma.h` without a heavy include.  The 128-K
 * substep width is the SWZ128 atom width and is hardcoded to match the
 * activation descriptor's `boxDim[0] = 128` (`create_activations_tma_desc`
 * in `moe_tma.cu`); a `static_assert` guards the descriptor invariant
 * (`Dims::HIDDEN_STATES % 128 == 0`).
 *
 * Caller contract (CRITICAL):
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     warp 8 lane 0 in the BS8 TMA+WGMMA path, selected via
 *     `is_tma_launcher_thread<Dims>()`).  Does NOT gate on `threadIdx`;
 *     calling from multiple threads issues duplicate
 *     `cp.async.bulk.tensor.2d` instructions and corrupts the
 *     mbarrier's transaction-bytes accounting.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE by the
 *     caller, BEFORE calling this helper, via
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr,
 *                                K_BLOCKS_TOTAL * Dims::BS * 128 *
 *                                    sizeof(A_element))`
 *     (= 32 768 bytes for Qwen3.5: `BS=8`, `HIDDEN_STATES=2048`).  This
 *     helper itself does NOT call `mbarrier_arrive_expect_tx`; it only
 *     issues the `K_BLOCKS_TOTAL` TMA loads, and the cumulative
 *     transaction count is what the caller's single arm covers.
 *   - `desc` MUST be the descriptor produced by
 *     `create_activations_tma_desc`, typically passed to the kernel as
 *     a `__grid_constant__ CUtensorMap const` parameter.  No new
 *     descriptor is required.
 *   - `dest` MUST be the BS8-sized BF16 input tile in SHM (Phase-1/2
 *     buffer); the caller's typed SHM layout supplies the underlying
 *     `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` storage (the union view in
 *     `TinyDataWGMMA_TMA::bf16_in_full`).  Aligned to at least 1024 B
 *     by the union's `alignas(1024)`.
 *   - This helper is consumed only by the BS8 TMA+WGMMA path; do NOT
 *     instantiate it from any other variant.
 *
 * @tparam Dims          The MoE Dims tag (provides `BS` and
 *                       `HIDDEN_STATES`).
 * @param  activations_desc Host-built activation TMA descriptor
 *                          (`__grid_constant__`), reused unchanged from
 *                          `create_activations_tma_desc`.
 * @param  dest          Reference to the tile-major BF16 SHM
 *                       destination tile shaped
 *                       `[K_BLOCKS_TOTAL][BS][K_STEP_WGMMA]` (typically
 *                       `shmem->tiny_wgmma_tma.bf16_in_full`).
 * @param  bar_smem_ptr  Pointer to the routing-window mbarrier in SHM
 *                       (`shmem->tiny_wgmma_tma.bar_rwin`), pre-armed
 *                       by the caller with the cumulative `tx_bytes`.
 */
template <typename Dims>
__device__ __forceinline__ void moe_load_full_bf16_input(
    CUtensorMap const& activations_desc,
    A_element (&dest)[Dims::HIDDEN_STATES / 128u][Dims::BS][128u], std::uint64_t* bar_smem_ptr) {
  // 128-K SWZ128 atom width — matches the activation descriptor's
  // `boxDim[0] = 128` baked into `create_activations_tma_desc` and the
  // `K_STEP_WGMMA` constant in `MoECoreDims`.  Hardcoded here so this
  // helper does not depend on `MoECoreDims` / `MoE_SHM`.
  constexpr std::uint32_t K_STEP_WGMMA = 128u;
  static_assert(Dims::HIDDEN_STATES % K_STEP_WGMMA == 0,
                "moe_load_full_bf16_input requires HIDDEN_STATES to be a "
                "multiple of 128 (the SWZ128 atom K-width).");
  constexpr std::uint32_t K_BLOCKS_TOTAL = Dims::HIDDEN_STATES / K_STEP_WGMMA;

#pragma unroll
  for (std::uint32_t kk = 0u; kk < K_BLOCKS_TOTAL; ++kk) {
    const std::uint32_t k_start = kk * K_STEP_WGMMA;
    // Each issue writes a self-contained 2 KB box (BS × 128 BF16) into
    // tile-major slot `kk` (`&dest[kk][0][0]`), matching the
    // `boxDim=(128,8)` byte layout exactly (docs/design_docs/monomoe_kernel.md §5).
    tma_load_bf16_input_tile(activations_desc, /*k_start=*/k_start,
                             /*dest_smem_ptr=*/&dest[kk][0][0],
                             /*bar_smem_ptr=*/bar_smem_ptr);
  }
}

// ─── Down-projection (Phase 4) TMA load helpers ──────────────────────────
//
// Mirror of the up-projection helpers above, specialized for the fp8
// down-projection weight and intermediate-activation descriptors built by
// `create_down_weight_tma_desc` and `create_down_activation_tma_desc`.
//
// Key differences vs the up-proj helpers:
//   * Weight descriptor axis order is (innermost=N, outer=expert*K + row)
//     instead of (innermost=K, outer=expert*2*N + row).  That is, the
//     down-proj's reduction dimension N is the innermost (K-for-WGMMA)
//     axis, and its output dimension K is the outer row axis.
//   * No Python pre-interleave needed on this side: the TMA hardware
//     applies SWIZZLE_128B at write time and the raw row-major
//     `[E, K, N]` fp8 tensor is fed directly.
//   * The 128×128 weight tile is fetched in ONE TMA issue
//     (16 384 B = `boxDim = (128, 128)`).
//   * The activation descriptor uses `boxDim = (128, 8)`; each issue
//     fetches up to 8 contiguous rows starting at `expert_slot_start[id]`
//     in the Phase-3-reorganized `temp_fp8` layout.
//
// Caller contract (same as up-proj helpers): every helper MUST be invoked
// by exactly ONE thread per block — the TMA launcher thread (warp 8,
// lane 0 on the BS8 WGMMA_TMA path, selected via
// `is_tma_launcher_thread<Dims>()`).  They do NOT internally gate on
// `threadIdx` and the caller is responsible for pre-arming the target
// mbarrier with the total expected byte count before issuing the TMA(s).

/**
 * @brief Issue one TMA load for the full 128×128 fp8 down-projection
 *        weight tile.
 *
 * A single `cp.async.bulk.tensor.2d` issue fetches one `(128 N, 128 K)`
 * box (16 384 bytes) from the descriptor built by
 * `create_down_weight_tma_desc` into the caller-supplied 16 KB SHM slot.
 * The descriptor uses `CU_TENSOR_MAP_SWIZZLE_128B`, so the TMA hardware
 * applies the 8-row × 128-byte core-matrix XOR swizzle at write time —
 * callers pass the raw row-major `[E, K, N]` fp8 weight tensor, and the
 * matching WGMMA A-descriptor uses `swizzle=1`, `A_LBO=16`,
 * `A_SBO=1024` (CUTLASS Major::K B128 layout).
 *
 * Coordinate convention for the down-weight descriptor: innermost axis
 * is N and outer axis is the flattened row index
 * `expert_id * K + output_row`.  Per `cuTensorMapEncodeTiled` the
 * coordinate operand order matches `globalDim`, so this wrapper emits
 * `coord0 = k_start` (innermost, the starting N column of the tile) and
 * `coord1 = expert_id * K + base_col` (outer, the starting output row).
 *
 * NOTE on parameter names: the outer axis is here called `base_col`
 * because it represents a column in the down-projection's output
 * (= hidden-size K of the layer).  Likewise `k_start` is a column in
 * the reduction dimension N.  The kernel-level spec uses these names
 * consistently with the mathematical roles of the dimensions.
 *
 * Caller contract:
 *   - Must be called by exactly ONE thread per block (the TMA launcher,
 *     typically warp 8 lane 0).  This function does NOT gate on
 *     `threadIdx`.
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 16384)` BEFORE calling
 *     this function.  This function itself does NOT call
 *     `mbarrier_arrive_expect_tx`; it only issues the single TMA load.
 *   - `dest_smem_ptr` MUST be 1024-B aligned (the swizzle atom
 *     alignment) and point at the base of `shm->w_down_wgmma[slot][0][0]`
 *     (the 16 384-B weight-tile slot).
 *   - `desc` MUST be the descriptor produced by
 *     `create_down_weight_tma_desc`, typically passed to the kernel as
 *     a `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs: `0 ≤ expert_id < num_experts`,
 *     `0 ≤ base_col` with `base_col + 128 ≤ K`, and `0 ≤ k_start` with
 *     `k_start + 128 ≤ N`.
 *
 * @param desc          Host-built down-weight TMA descriptor
 *                      (`__grid_constant__`).
 * @param expert_id     Expert index (`0 ≤ expert_id < num_experts`).
 * @param K             Down-projection output dimension (= hidden size).
 *                      Used to flatten the outer-axis coordinate as
 *                      `expert_id * K + base_col`.
 * @param base_col      First output row of the 128-row tile (multiple
 *                      of 128).
 * @param k_start       First reduction-dim column of the tile (multiple
 *                      of 128).
 * @param dest_smem_ptr 1024-B aligned SHM destination pointer to
 *                      `shm->w_down_wgmma[slot][0][0]` (16 384 B).
 * @param bar_smem_ptr  16-B aligned SHM mbarrier pre-armed by the caller
 *                      with `expect_tx = 16384`.
 */
__device__ __forceinline__ void tma_load_down_wgmma_tile(CUtensorMap const& desc,
                                                         std::uint32_t expert_id, std::uint32_t K,
                                                         std::uint32_t base_col,
                                                         std::uint32_t k_start, void* dest_smem_ptr,
                                                         std::uint64_t* bar_smem_ptr) {
  // Descriptor axis order (innermost first): coord0 = N, coord1 = row.
  // Flattened outer-axis row for this expert: `expert_id * K + base_col`.
  const std::uint32_t global_row = expert_id * K + base_col;
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_smem_ptr, bar_smem_ptr);
}

/**
 * @brief Issue one bulk TMA load for up to 8 contiguous rows of the fp8
 *        intermediate-activation tile consumed by the Phase-4 WGMMA
 *        down-projection.
 *
 * A single `cp.async.bulk.tensor.2d` issue fetches one
 * `(128 N, boxDim[1] = 8)` box (1024 bytes) from the descriptor built
 * by `create_down_activation_tma_desc` into the caller-supplied 1 KB
 * SHM slot.  The descriptor uses `CU_TENSOR_MAP_SWIZZLE_128B`, so the
 * TMA hardware applies the 8-row × 128-byte core-matrix XOR swizzle
 * at write time — the matching WGMMA B descriptor uses `swizzle=1`,
 * `B_LBO=16`, `B_SBO=128`.
 *
 * The fetched rows start at `expert_slot_start` in the
 * Phase-3-reorganized `spec->temp_fp8` layout, where each
 * expert's routed tokens occupy a contiguous slab
 * `[expert_slot_start[id], expert_slot_start[id] + routed_token_count[id])`.
 *
 * Since `boxDim[1] = 8` is baked into the descriptor, this issue always
 * fetches the full 8-row × 128-N atom.  When the expert's
 * `routed_token_count < 8`, the "unused" rows
 * `[routed_token_count, 8)` in the SHM slot land with TMA-fetched bytes
 * beyond the expert's own slab.  The kernel's WGMMA path tolerates
 * that garbage: each calc thread's accumulators map to a fixed rank
 * via lane id, and the rank-filtered accumulate at end-of-expert
 * (`if (rank_for_tok[tok] != 0xFF)`) only reads `down_out` columns
 * for ranks `< routed_count`.  fp8 e4m3 has no NaN encoding, so the
 * WGMMA cannot trip an exception on garbage input.
 *
 * Coordinate convention for the down-activation descriptor: innermost
 * axis is N and outer axis is the `sorted_slot` row index into
 * `spec->temp_fp8`.  Per `cuTensorMapEncodeTiled` the coordinate operand
 * order matches `globalDim`, so this wrapper emits
 * `coord0 = k_start` (innermost, starting N column) and
 * `coord1 = expert_slot_start` (outer, first reorganized row for this
 * expert).
 *
 * Caller contract:
 *   - Must be called by exactly ONE thread per block (the TMA launcher).
 *     This function does NOT gate on `threadIdx`.
 *   - MUST only be invoked when `routed_token_count > 0`.  When no
 *     tokens route to this expert, the caller SHALL neither arm
 *     `bar_smem_ptr` nor call this helper; the SHM slot is left
 *     uninitialized and its contents are tolerated by the WGMMA
 *     path (see preamble above).
 *   - `bar_smem_ptr` MUST have been pre-armed EXACTLY ONCE with
 *     `mbarrier_arrive_expect_tx(bar_smem_ptr, 1024)` BEFORE calling
 *     this function.  The TMA atom is always a full 8-row × 128-B
 *     slab (1024 B) regardless of `routed_token_count`.
 *   - `dest_smem_ptr` MUST be 1024-B aligned (the swizzle atom
 *     alignment) and point at `&shm->a_down_wgmma[slot][0][0][0]`
 *     (the 1024-B activation-tile slot).
 *   - `desc` MUST be the descriptor produced by
 *     `create_down_activation_tma_desc`, typically passed to the kernel
 *     as a `__grid_constant__ CUtensorMap const` parameter.
 *   - Valid inputs: `0 ≤ k_start` with `k_start + 128 ≤ N`;
 *     `expert_slot_start + 8 ≤ TEMP_ROWS_TMA`.
 *
 * SHM byte layout produced: the TMA + SWZ128 write places each
 * logical (tok, kc, ki) byte at `tok*128 + kc*16 + ki` XOR the SWZ128
 * permutation — this is the CUTLASS Major::K B128 canonical layout
 * that the WGMMA B descriptor reads via `LBO=16`, `SBO=128`,
 * `swizzle=1`.
 *
 * @param desc               Host-built down-activation TMA descriptor
 *                           (`__grid_constant__`, boxDim=(128, 8)).
 * @param k_start            First reduction-dim column of the K-step
 *                           (multiple of 128).
 * @param expert_slot_start  First reorganized row of the current
 *                           expert's contiguous slab in `temp_fp8`.
 * @param dest_smem_ptr      1024-B aligned SHM destination pointer to
 *                           `&shm->a_down_wgmma[slot][0][0][0]` (1024 B).
 * @param bar_smem_ptr       16-B aligned SHM mbarrier pre-armed by the
 *                           caller with `expect_tx = 1024`.
 */
__device__ __forceinline__ void tma_load_down_wgmma_activation_bulk(CUtensorMap const& desc,
                                                                    std::uint32_t k_start,
                                                                    std::uint32_t expert_slot_start,
                                                                    void* dest_smem_ptr,
                                                                    std::uint64_t* bar_smem_ptr) {
  // Descriptor axis order (innermost first): coord0 = N, coord1 = row.
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/expert_slot_start, dest_smem_ptr, bar_smem_ptr);
}

}  // namespace monomoe

#endif  // MOE_TMA_H
