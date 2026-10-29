// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../model/dsv41_layout.cuh"
#include "../../model/kv_cache_traits.cuh"
#include "../../pipeline/stage_contracts.cuh"

namespace flashinfer::sparse_mla_sm120::kernels::dsv41_fp8 {

using pipeline::RoleSync;
using pipeline::SlotRelease;
using pipeline::StoreHandoff;

struct Dsv41MixedCacheSharedLayout {
  using Raw = Dsv41Fp4Layout;
  using KV = KVCacheTraits<ModelType::DSV4_1>;
  static constexpr int CONVERSION_GROUPS_PER_ROW = Raw::DIMS / KV::QUANT_TILE;
  static constexpr int STAGES = 2;
  static constexpr int BI = 32;
  static constexpr int LOGICAL_CANDIDATES = 64;
  static constexpr int SUBTILES = LOGICAL_CANDIDATES / BI;
  static constexpr int RAW_SLOT_BYTES = BI * Raw::BYTES_PER_TOKEN;
  static constexpr int RAW_SCALE_OFFSET = BI * Raw::DATA_BYTES;
  static constexpr int READY_OFFSET = STAGES * RAW_SLOT_BYTES;
  static constexpr int EXTRA_SMEM = READY_OFFSET + STAGES * sizeof(uint64_t);
  static_assert(READY_OFFSET % alignof(uint64_t) == 0);
  static_assert(LOGICAL_CANDIDATES % BI == 0);
  __device__ static uint64_t* ready(char* scratch) {
    return reinterpret_cast<uint64_t*>(scratch + READY_OFFSET);
  }
  __device__ static uint8_t* raw(char* scratch, int slot) {
    return reinterpret_cast<uint8_t*>(scratch) + slot * RAW_SLOT_BYTES;
  }
  template <size_t BaseBytes>
  struct Layout {
    static constexpr size_t RAW_OFFSET = BaseBytes;
    static constexpr size_t TOTAL_BYTES = RAW_OFFSET + EXTRA_SMEM;
    static_assert(RAW_OFFSET % 16 == 0);
  };
};

template <int MathGroups, int MathRegs>
struct Dsv41MixedCacheRoleResources : Dsv41MixedCacheSharedLayout {
  static constexpr int WARP_THREADS = 32;
  static constexpr int GROUP_THREADS = 4 * WARP_THREADS;
  static constexpr int MATH_THREADS = MathGroups * GROUP_THREADS;
  static constexpr int CONVERT_BASE = MATH_THREADS;
  static constexpr int GATHER_BASE = CONVERT_BASE + GROUP_THREADS;
  static constexpr int BLOCK_THREADS = GATHER_BASE + GROUP_THREADS;
  static constexpr int MATH_REGS = MathRegs;
  static constexpr int CONVERT_REGS = 112;
  static constexpr int GATHER_REGS = 32;
  static constexpr int MIN_BLOCKS = 1;
  static_assert(GROUP_THREADS * (MathGroups * MATH_REGS + CONVERT_REGS + GATHER_REGS) <= 65536);
  __device__ static int role_tid() { return threadIdx.x & (GROUP_THREADS - 1); }
  __device__ static bool is_gather() { return threadIdx.x >= GATHER_BASE; }
  __device__ static void gather_registers();
  __device__ static void convert_registers();
  __device__ static void math_registers();
};

struct Dsv41MixedCacheDecodeResources : Dsv41MixedCacheRoleResources<1, 240> {
  using RawFree = SlotRelease<8, 9, GROUP_THREADS, GROUP_THREADS>;
  using KvFree = SlotRelease<4, 5, GROUP_THREADS, MATH_THREADS>;
  using KvReady = StoreHandoff<6, 7, GROUP_THREADS, MATH_THREADS>;
  using GatherSync = RoleSync<10, GROUP_THREADS>;
};

struct Dsv41MixedCachePrefillResources : Dsv41MixedCacheRoleResources<2, 176> {
  static constexpr bool RAW_PIPELINE = true;
  static constexpr int QK_THREADS = GROUP_THREADS;
  static constexpr int XV_THREADS = GROUP_THREADS;
  using RawFree = SlotRelease<14, 15, GROUP_THREADS, GROUP_THREADS>;
  using KvFree = SlotRelease<1, 5, GROUP_THREADS, QK_THREADS + XV_THREADS>;
  using KvReady = StoreHandoff<12, 13, GROUP_THREADS, QK_THREADS>;
  using GatherSync = RoleSync<11, GROUP_THREADS>;
};

}  // namespace flashinfer::sparse_mla_sm120::kernels::dsv41_fp8
