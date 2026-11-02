// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../../arch/common.cuh"
#include "../../common/kv_cache_io.cuh"
#include "convert.cuh"
#include "resources.cuh"

// MT selects the scale placement of both caches: DSV4_1 reads footer-scale
// pages, DSV4_1_INLINE reads inline-scale records, where a single bulk copy
// brings a row's data and scales (528B FP8, 288B FP4).
template <ModelType MT>
struct Dsv41MixedCacheDecodeSchedule
    : flashinfer::sparse_mla_sm120::kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources {
  using Plan = flashinfer::sparse_mla_sm120::kernels::dsv41_fp8::Dsv41MixedCacheDecodeResources;
  using Main = KVCacheTraits<MT>;
  static constexpr bool INLINE = Main::SCALE_IN_KV_SMEM;
  using RawLayout = flashinfer::sparse_mla_sm120::Dsv41Fp4LayoutT<INLINE>;
  static_assert(is_dsv41(MT));
  static_assert(RawLayout::BYTES_PER_TOKEN == Raw::BYTES_PER_TOKEN);
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
          if constexpr (INLINE) {
            // Flat rows at the section's validated 16B-aligned row pitch.
            data = sec.kv + ((size_t)idx * (sec.stride >> 4) << 4);
          } else {
            const bool power_of_two = (sec.pbs & (sec.pbs - 1)) == 0;
            const int page_idx = power_of_two ? (idx >> (__ffs(sec.pbs) - 1)) : (idx / sec.pbs);
            const uint8_t* page = sec.kv + (size_t)page_idx * sec.stride;
            int local = idx - page_idx * sec.pbs;
            data = page + Raw::selected_data_offset<Dsv41Fp8Layout>(sec.extra, local);
            scales = page + Raw::selected_scale_offset<Dsv41Fp8Layout>(sec.extra, sec.pbs, local);
          }
        }
        if constexpr (!INLINE) {
          if (!sec.extra && tid < BI)
            *reinterpret_cast<uint4*>(sm.kv_sc(buf) + tid * KV::SCALE_BYTES_PER_TOKEN) =
                __ldg(reinterpret_cast<const uint4*>(scales));
          if (!sec.extra) {
            __threadfence_block();
            GatherSync::wait();
          }
        }
        auto* barrier = sec.extra ? ready + buf : sm.mbar_full(buf);
        if (tid == 0)
          BulkReady::expect(barrier,
                            BI * (sec.extra ? Raw::BYTES_PER_TOKEN : Main::KV_SMEM_COPY_BYTES));
        if (tid < BI) {
          if (sec.extra) {
            auto* raw = Plan::raw(scratch, buf);
            if constexpr (INLINE) {
              cp_async_bulk_g2s(raw + RawLayout::data_offset(tid), data, RawLayout::BYTES_PER_TOKEN,
                                barrier);
            } else {
              cp_async_bulk_g2s(raw + tid * Raw::DATA_BYTES, data, Raw::DATA_BYTES, barrier);
              cp_async_bulk_g2s(raw + RAW_SCALE_OFFSET + tid * Raw::SCALE_BYTES, scales,
                                Raw::SCALE_BYTES, barrier);
            }
          } else {
            cp_async_bulk_g2s(sm.kv_fp8(buf) + tid * KV::KV_SMEM_STRIDE, data,
                              Main::KV_SMEM_COPY_BYTES, barrier);
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
          kernels::dsv41_fp8::convert_raw<Plan, INLINE, Smem::SCALE_STRIDE>(
              Plan::raw(scratch, buf), sm.kv_fp8(buf), sm.kv_scales(buf), tid,
              [](int) { return true; });
        } else {
          BulkReady::wait(sm.mbar_full(buf), TwoSlotCursor::phase(t));
        }
        RawFree::release(buf);
        KvReady::publish(buf);
      }
    }
  }
};
