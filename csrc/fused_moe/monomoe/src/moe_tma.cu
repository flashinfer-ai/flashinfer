/**
 * Host-side TMA `CUtensorMap` descriptor factories for the BS8 WGMMA
 * up-projection path.
 *
 * This translation unit is pure host code: the only CUDA interaction is
 * through the Driver API (`cuTensorMapEncodeTiled`), which runs on the CPU
 * to populate a 128-byte POD descriptor.  The returned descriptors are later
 * passed to the device as `__grid_constant__ CUtensorMap const` kernel
 * parameters.
 *
 * Implements the factories declared in `moe_tma.h`.
 *
 * Unlike the other `.cu` files in this directory (which are `#include`d into
 * `moe.cu` for whole-program inlining), this file is a standalone host-side
 * translation unit: it is compiled directly by `monomoe_wrapper.cu` via the
 * build system.  Hence no `#pragma once` / include guards — this file is
 * never `#include`d by other TUs.
 */

// Host-side TMA descriptor factories.  The only error path is a failed
// `cuTensorMapEncodeTiled`, reported via TVM-FFI's `TVM_FFI_ICHECK` so the
// failure surfaces as a Python exception (FlashInfer is framework-agnostic
// through TVM-FFI; no Torch/ATen headers in this tree).
#include <cuda.h>

#include "moe_tma.h"
#include "tvm_ffi_utils.h"

namespace monomoe {

CUtensorMap create_up_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t N,
                                      uint32_t K) {
  // SWIZZLE_128B up-weight descriptor, paired with the gate/up Python
  // pre-interleave that packs each 128-row WGMMA tile into a
  // contiguous 128-row GM block.
  //
  // Expected GM layout (produced by `interleave_for_tma_wgmma_up` in
  // Python):
  //   For every expert and every 64-gate-row block k in [0, N/64):
  //     new_rows[128k +  0 .. 128k + 32) = gate[64k     .. 64k + 32)
  //     new_rows[128k + 32 .. 128k + 64) =   up[64k     .. 64k + 32)
  //     new_rows[128k + 64 .. 128k + 96) = gate[64k + 32 .. 64k + 64)
  //     new_rows[128k + 96 .. 128k +128) =   up[64k + 32 .. 64k + 64)
  //   (Total 2N rows per expert, same footprint as the original
  //   `[gate[0..N), up[0..N)]` layout.)
  //
  // With this layout, a single `boxDim = (128, 128)` TMA at
  //   coord0 = k_start
  //   coord1 = expert_id * 2 * N + 2 * base_row_up
  // fetches the full 128×128 fp8 A operand for one WGMMA step in one
  // issue.  The SHM bytes after the TMA's SWZ128 XOR match the
  // canonical Major::K B128 layout expected by the WGMMA A-descriptor
  // with `swizzle=1`, `A_LBO=16`, `A_SBO=1024`.
  //
  // Zero-initialize so any unfilled bytes in the 128-B POD have a
  // defined value before we return by value.
  CUtensorMap desc{};

  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(K),
      static_cast<uint64_t>(num_experts) * 2ULL * static_cast<uint64_t>(N),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(K),  // K bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*rows   */ 128u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      // The Driver API takes a non-const void*; the descriptor only
      // reads the pointer value, it never writes through it.
      const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // On failure, raise a TVM_FFI_ICHECK naming the failing tensor so the
  // Python stack trace points directly at "up-projection weights".
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for up-projection weights: CUresult="
      << static_cast<int>(res) << " (num_experts=" << num_experts << ", N=" << N << ", K=" << K
      << ")";

  return desc;
}

CUtensorMap create_activations_tma_desc(const void* activations_ptr, uint32_t batch_size_cap,
                                        uint32_t K_hidden) {
  // Zero-initialize the POD so any bytes not explicitly written by the
  // Driver API have a defined value before we return by value.
  CUtensorMap desc{};

  // --- rank-2 descriptor describing the bf16 activation tensor ----------
  //
  // GM layout of `activations_in` is `[BS, K_hidden]` row-major bf16, which
  // we view as a 2D matrix with innermost = K_hidden, outer = batch row.
  //
  // `cuTensorMapEncodeTiled` uses innermost-first ordering:
  //   globalDim[0]     = innermost (K_hidden, bf16 elements)
  //   globalDim[1]     = outer     (batch_size_cap rows)
  //   boxDim[0]        = innermost tile extent (128 bf16 elements)
  //   boxDim[1]        = outer tile extent (8 tokens)
  //   elementStrides[i] = 1 means "no sub-sampling" along axis i
  //
  // For `rank = 2`, `globalStrides` is a length-1 array giving the byte
  // stride between successive rows along the outer axis; for bf16 (2 B/elem)
  // the row stride is `K_hidden * 2` bytes.
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(K_hidden),
      static_cast<uint64_t>(batch_size_cap),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(K_hidden) * 2ULL,  // bf16 = 2 B/element
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*tokens */ 8u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kRank,
      // The Driver API takes a non-const void*; the descriptor only reads
      // the pointer value, it never writes through it.
      const_cast<void*>(activations_ptr), global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // On failure, raise a TVM_FFI_ICHECK naming the failing tensor so the
  // Python stack trace points directly at "activations".
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for activations: CUresult=" << static_cast<int>(res)
      << " (batch_size_cap=" << batch_size_cap << ", K_hidden=" << K_hidden << ")";

  return desc;
}

CUtensorMap create_down_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t K,
                                        uint32_t N, uint32_t row_box) {
  // SWIZZLE_128B down-weight descriptor.  The TMA hardware applies the
  // 8-row × 128-byte core-matrix XOR swizzle at write time, so the
  // caller MUST NOT pre-interleave the weight tensor — the raw
  // row-major `[E, K, N]` fp8 tensor is expected.
  //
  // `row_box` controls how many output rows (= output cols of the
  // down-proj) one TMA delivers:
  //   * 128 — one 128×128 fp8 atom = 16 KB per TMA (legacy).
  //   * 256 — two stacked 128×128 atoms = 32 KB per TMA (DOWN_COL_TILE
  //     = 256 only).  Halves the issue count.
  // The maximum legal value is 256 (TMA boxDim cap).  Both values
  // produce the same canonical Major::K B128 SHM layout from the
  // consumer's perspective; the WGMMA A descriptor still references a
  // single 128-row sub-atom per WGMMA call.
  TVM_FFI_ICHECK(row_box == 128u || row_box == 256u)
      << "create_down_weight_tma_desc: row_box must be 128 or 256, got " << row_box;
  TVM_FFI_ICHECK(K % row_box == 0)
      << "create_down_weight_tma_desc: K=" << K << " must be a multiple of row_box=" << row_box;
  //
  // Axis ordering: innermost = N (reduction), outer = flattened
  // `expert_id * K + output_row`.  This matches the down-proj GM
  // layout `[E, K, N]` where each expert's `K` output rows are each a
  // contiguous `N`-element fp8 vector, and the WGMMA consumes 128
  // N-elements at a time as the reduction dimension.
  CUtensorMap desc{};

  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(N),
      static_cast<uint64_t>(num_experts) * static_cast<uint64_t>(K),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(N),  // N bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*N-inner*/ 128u,
      /*rows   */ row_box,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank, const_cast<void*>(weights_ptr), global_dim,
      global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for down-projection weights: CUresult="
      << static_cast<int>(res) << " (num_experts=" << num_experts << ", K=" << K << ", N=" << N
      << ", row_box=" << row_box << ")";

  return desc;
}

CUtensorMap create_down_activation_tma_desc(const void* activations_ptr, uint32_t temp_rows,
                                            uint32_t N) {
  // SWIZZLE_128B down-activation descriptor over token-major
  // `spec->temp_fp8[TEMP_ROWS, N]` (row-major fp8 with `byte(row, col)
  // = row * N + col`).
  //
  // One TMA with `boxDim = (128, 8)` fetches exactly one 1024-B SWZ128
  // atom (8 rows × 128 K-bytes) per K-step.  The TMA hardware applies
  // the 8-row × 128-byte core-matrix XOR swizzle at write time into
  // `shm->a_down_wgmma[slot]`, producing the CUTLASS Major::K B128
  // canonical layout the WGMMA B descriptor reads with `LBO=16`,
  // `SBO=128`, `swizzle=1`:
  //
  //   logical byte(tok, kc, ki) = tok * 128 + kc * 16 + ki
  //
  // Descriptor axes (innermost first):
  //   globalDim[0]  = N             (innermost fp8 bytes per row)
  //   globalDim[1]  = temp_rows     (outer sorted_slot row index)
  //   globalStrides[0] = N bytes    (fp8 row stride)
  //   boxDim[0]  = 128              (one K-step worth of K bytes)
  //   boxDim[1]  = 8                (8 rows per atom)
  //
  // Coordinates from the kernel side for one K-step at expert id:
  //   coord0 = k_start
  //   coord1 = expert_slot_start[id]
  //
  // SWZ128 preconditions:
  //   - `boxDim[0] * sizeof(fp8) = 128 B` ✓
  //   - `globalStrides[0] = N` must be a multiple of 128 ✓ for the
  //     Qwen3.5-35B shape (N=512).
  //   - `activations_ptr` must be 16-B aligned; the scratchpad start
  //     is already 256-B aligned by PyTorch's allocator.
  //
  // Zero-initialize the POD for defined-value bytes before return.
  CUtensorMap desc{};

  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(N),
      static_cast<uint64_t>(temp_rows),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(N),  // N bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*rows   */ 8u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      // The Driver API takes a non-const void*; the descriptor only
      // reads the pointer value, it never writes through it.
      const_cast<void*>(activations_ptr), global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for down-projection activations: CUresult="
      << static_cast<int>(res) << " (temp_rows=" << temp_rows << ", N=" << N << ")";

  return desc;
}

}  // namespace monomoe
