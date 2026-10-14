// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../arch/common.cuh"
#include "../../common/kv_cache_io.cuh"
#include "convert.cuh"
#include "resources.cuh"

struct Dsv41MixedCacheDecodeSchedule
    : flashinfer::sparse_mla_sm120::kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources {
  using Plan = flashinfer::sparse_mla_sm120::kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources;
  static constexpr bool RAW_PIPELINE = true;

  template <typename Smem, typename Section>
  __device__ __forceinline__ static void produce(Smem sm, char* scratch, Section section,
                                                 int count) {
    using namespace flashinfer::sparse_mla_sm120;
    using pipeline::BulkReady;
    using pipeline::TwoSlotCursor;
    auto* ready = Plan::ready(scratch);
    int tid = Plan::role_tid();
    if (Plan::is_gather()) {
      Plan::gather_registers();
      for (int t = 0; t < count; ++t) {
        int buf = TwoSlotCursor::slot(t);
        auto sec = section(t);
        if (TwoSlotCursor::reuses(t)) RawFree::acquire(buf);
        if (!sec.extra && TwoSlotCursor::reuses(t)) KvFree::acquire(buf);
        int idx = tid < BI ? sec.index(tid) : -1;
        const uint8_t* data = sparse_mla_zero_row;
        const uint8_t* scales = sparse_mla_zero_row;
        if (idx >= 0) {
          const bool power_of_two = (sec.pbs & (sec.pbs - 1)) == 0;
          const int page_idx = power_of_two ? (idx >> (__ffs(sec.pbs) - 1)) : (idx / sec.pbs);
          const uint8_t* page = sec.kv + (size_t)page_idx * sec.stride;
          int local = idx - page_idx * sec.pbs;
          data = page + Raw::selected_data_offset<Dsv41Fp8Layout>(sec.extra, local);
          scales = page + Raw::selected_scale_offset<Dsv41Fp8Layout>(sec.extra, sec.pbs, local);
        }
        if (!sec.extra && tid < BI)
          *reinterpret_cast<uint4*>(sm.kv_sc(buf) + tid * KV::SCALE_BYTES_PER_TOKEN) =
              __ldg(reinterpret_cast<const uint4*>(scales));
        if (!sec.extra) {
          __threadfence_block();
          GatherSync::wait();
        }
        auto* barrier = sec.extra ? ready + buf : sm.mbar_full(buf);
        if (tid == 0)
          BulkReady::expect(barrier, BI * (sec.extra ? Raw::BYTES_PER_TOKEN : KV::D_NOPE));
        if (tid < BI) {
          if (sec.extra) {
            auto* raw = Plan::raw(scratch, buf);
            cp_async_bulk_g2s(raw + tid * Raw::DATA_BYTES, data, Raw::DATA_BYTES, barrier);
            cp_async_bulk_g2s(raw + RAW_SCALE_OFFSET + tid * Raw::SCALE_BYTES, scales,
                              Raw::SCALE_BYTES, barrier);
          } else {
            cp_async_bulk_g2s(sm.kv_fp8(buf) + tid * KV::KV_SMEM_STRIDE, data, KV::D_NOPE, barrier);
          }
        }
      }
    } else {
      Plan::convert_registers();
      pipeline::SelectiveSlotPhases phases;
      for (int t = 0; t < count; ++t) {
        int buf = TwoSlotCursor::slot(t);
        auto sec = section(t);
        if (sec.extra) {
          BulkReady::wait(ready + buf, phases.current(buf));
          phases.advance(buf);
          if (TwoSlotCursor::reuses(t)) KvFree::acquire(buf);
          kernels::dsv41_fp8::convert_raw<Plan>(Plan::raw(scratch, buf), sm.kv_fp8(buf),
                                                sm.kv_sc(buf), tid, [](int) { return true; });
        } else {
          BulkReady::wait(sm.mbar_full(buf), TwoSlotCursor::phase(t));
        }
        RawFree::release(buf);
        KvReady::publish(buf);
      }
    }
  }
};
