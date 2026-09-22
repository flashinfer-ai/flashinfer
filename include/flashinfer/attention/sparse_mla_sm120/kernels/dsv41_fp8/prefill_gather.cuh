// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../pipeline/staged_pipeline.cuh"
#include "../fp8_prefill/prefill_common.cuh"

struct Dsv41PrefillGatherSchedule {
  static constexpr int GATHER_BARRIER = 11;
  static constexpr bool RAW_PIPELINE = false;
  static constexpr int EXTRA_THREADS = 0;
  static constexpr int EXTRA_SMEM = 0;
  static constexpr int MATH_REGS = 232;
  template <typename Cfg, typename Smem>
  __device__ static void wait(Smem sm, const PrefillSection&, int tile) {
    using namespace flashinfer::sparse_mla_sm120::pipeline;
    BulkReady::wait(sm.mbar_kv + TwoSlotCursor::slot(tile), TwoSlotCursor::phase(tile));
  }

  template <typename Cfg, typename Smem>
  __device__ static void issue(Smem sm, const PrefillSection& sec, int tile, int tid,
                               uint64_t policy) {
    using KV = KVCacheTraits<ModelType::DSV4_1>;
    const int buf = tile & 1;
    const int idx = tid < Cfg::BI ? sec.index(tid, Cfg::BI) : -1;
    const uint8_t* data = sparse_mla_zero_row;
    const uint8_t* scales = sparse_mla_zero_row;
    if (idx >= 0) {
      const uint8_t* page = sec.kv + (size_t)(idx / sec.page_block_size) * sec.block_stride;
      const int local = idx % sec.page_block_size;
      data = page + Dsv41Fp8Layout::data_offset(local);
      scales = page + Dsv41Fp8Layout::scale_offset(sec.page_block_size, local);
    }
    if (tid < Cfg::BI)
      *reinterpret_cast<uint4*>(sm.kv_scale_buf(buf) + tid * KV::SCALE_BYTES_PER_TOKEN) =
          __ldg(reinterpret_cast<const uint4*>(scales));
    __threadfence_block();
    flashinfer::sparse_mla_sm120::pipeline::RoleSync<GATHER_BARRIER, Cfg::IO_THREADS>::wait();
    if (tid == 0)
      flashinfer::sparse_mla_sm120::pipeline::BulkReady::expect(sm.mbar_kv + buf,
                                                                Cfg::BI * KV::D_NOPE);
    if (tid < Cfg::BI)
      cp_async_bulk_g2s_l2hint(sm.kv_buf(buf) + tid * KV::KV_SMEM_STRIDE, data, KV::D_NOPE,
                               sm.mbar_kv + buf, policy);
  }
};
