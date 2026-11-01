// SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo (modifications)
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION.
// SPDX-License-Identifier: Apache-2.0
// Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad,
// csrc/common/fp4/nvfp4_utils.cuh (NVIDIA/vLLM-derived helpers).
#pragma once
#include "quant_utils.cuh"
namespace flashinfer::resinfer {
constexpr int CVT_FP4_ELTS_PER_UINT32 = 8;

__host__ __device__ __forceinline__ int round_up(int x, int n) { return ((x + n - 1) / n) * n; }

__host__ __device__ __forceinline__ int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}

template <int RESIDUE_PER_8_ELTS>
constexpr int get_residue_per_thread() {
  return RESIDUE_PER_8_ELTS * (CVT_FP4_ELTS_PER_THREAD / 8);
}

template <int THREADS_PER_GROUP>
constexpr uint32_t compute_group_mask(int group_id) {
  return ((1u << THREADS_PER_GROUP) - 1u) << (group_id * THREADS_PER_GROUP);
}

template <int GROUP_THREADS>
__device__ __forceinline__ float warp_group_max(float v, int laneId) {
  int groupId = laneId / GROUP_THREADS;
  uint32_t mask = compute_group_mask<GROUP_THREADS>(groupId);
#pragma unroll
  for (int off = 1; off < GROUP_THREADS; off <<= 1) {
    v = fmaxf(v, __shfl_xor_sync(mask, v, off));
  }
  return v;
}

template <class ResidueType, int RESIDUE_PER_8_ELTS>
__device__ ResidueType* cvt_residue_data_get_offset(int rowIdx, int colIdx, int numMainCols,
                                                    int numTotalCols, ResidueType* outputStart) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();
  constexpr int RESIDUES_PER_WARP = 32 * RESIDUE_PER_THREAD;
  constexpr int UINT32_PER_WARP = (RESIDUES_PER_WARP + 7) / 8;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  if (laneId < UINT32_PER_WARP) {
    int64_t uint32s_per_row = numTotalCols / CVT_FP4_ELTS_PER_UINT32;
    int64_t base_uint32s_per_row = numMainCols / CVT_FP4_ELTS_PER_UINT32;

    int64_t residueDataIdx =
        rowIdx * uint32s_per_row + base_uint32s_per_row + warpId * UINT32_PER_WARP + laneId;
    return outputStart + residueDataIdx;
  }
  return nullptr;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx, int numTotalCols,
                                                       SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 || CVT_FP4_NUM_THREADS_PER_SF == 2);

  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    int64_t SFOffset = compute_tiled_sf_offset<SFType>(mIdx, kIdx, numTotalCols);

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}

template <class ResidueSFType, int RESIDUE_PER_8_ELTS>
__device__ ResidueSFType* cvt_residue_sf_get_offset(int rowIdx, int colIdx, int numTotalCols,
                                                    int numMainCols, ResidueSFType* residueSFOut) {
  constexpr int RESIDUE_PER_THREAD = get_residue_per_thread<RESIDUE_PER_8_ELTS>();

  constexpr int THREADS_PER_SF_GROUP = 16 / RESIDUE_PER_THREAD;

  int warpId = colIdx / 32;
  int laneId = colIdx % 32;

  int sfGroupId = laneId / THREADS_PER_SF_GROUP;
  int sfGroupThreadId = laneId % THREADS_PER_SF_GROUP;

  if (sfGroupThreadId != 0) return nullptr;

  int warpBaseResidueIdx = warpId * 32 * RESIDUE_PER_THREAD;
  int sfGroupResidueIdx = sfGroupId * 16;
  int residueElementIdx = numMainCols + warpBaseResidueIdx + sfGroupResidueIdx;

  int32_t kIdx = residueElementIdx / CVT_FP4_SF_VEC_SIZE;
  int32_t mIdx = rowIdx;
  int64_t SFOffset = compute_tiled_sf_offset<ResidueSFType>(mIdx, kIdx, numTotalCols);

  return residueSFOut + SFOffset;
}

}  // namespace flashinfer::resinfer
