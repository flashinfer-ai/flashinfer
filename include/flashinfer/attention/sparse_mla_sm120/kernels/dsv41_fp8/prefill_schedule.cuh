// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "convert.cuh"
#include "prefill_gather.cuh"
#include "resources.cuh"

struct Dsv41MixedCachePrefillSchedule
    : flashinfer::sparse_mla_sm120::kernels::dsv41_fp8::Dsv41MixedCachePrefillResources {
  using Plan = flashinfer::sparse_mla_sm120::kernels::dsv41_fp8::Dsv41MixedCachePrefillResources;
  using Cursor = flashinfer::sparse_mla_sm120::pipeline::TwoSlotCursor;
  using Transaction = flashinfer::sparse_mla_sm120::pipeline::BulkReady;

  template <typename Cfg, typename Smem>
  __device__ static void wait(Smem sm, const PrefillSection& sec, int tile) {
    static_assert(Cfg::QK_THREADS == QK_THREADS);
    if (!sec.extra) Transaction::wait(sm.mbar_kv + Cursor::slot(tile), Cursor::phase(tile));
    KvReady::wait(Cursor::slot(tile));
  }

  template <typename Cfg, typename Smem, typename Section>
  __device__ static void produce(Smem sm, char* scratch, Section section, int tiles) {
    using namespace flashinfer::sparse_mla_sm120;
    static_assert(Cfg::BI == BI && Cfg::MATH_THREADS == MATH_THREADS &&
                  Cfg::IO_THREADS == GROUP_THREADS);
    auto* ready = Plan::ready(scratch);
    const int role_tid = Plan::role_tid();
    if (Plan::is_gather()) {
      Plan::gather_registers();
      const uint64_t policy = create_l2_evict_first_policy();
      for (int tile = 0; tile < tiles; ++tile) {
        const int buf = Cursor::slot(tile);
        const auto sec = section(tile);
        if (Cursor::reuses(tile)) RawFree::acquire(buf);
        if (!sec.extra && Cursor::reuses(tile)) KvFree::acquire(buf);
        const int idx = role_tid < BI ? sec.index(role_tid, BI) : -1;
        const uint8_t* data = sparse_mla_zero_row;
        const uint8_t* scales = sparse_mla_zero_row;
        if (idx >= 0) {
          const uint8_t* page = sec.kv + (size_t)(idx / sec.page_block_size) * sec.block_stride;
          const int local = idx % sec.page_block_size;
          data = page + Raw::selected_data_offset<Dsv41Fp8Layout>(sec.extra, local);
          scales = page + Raw::selected_scale_offset<Dsv41Fp8Layout>(sec.extra, sec.page_block_size,
                                                                     local);
        }
        if (!sec.extra && role_tid < BI)
          *reinterpret_cast<uint4*>(sm.kv_scale_buf(buf) + role_tid * KV::SCALE_BYTES_PER_TOKEN) =
              __ldg(reinterpret_cast<const uint4*>(scales));
        __threadfence_block();
        GatherSync::wait();
        if (role_tid == 0)
          Transaction::expect(sec.extra ? ready + buf : sm.mbar_kv + buf,
                              BI * (sec.extra ? Raw::BYTES_PER_TOKEN : KV::D_NOPE));
        if (role_tid < BI) {
          if (sec.extra) {
            uint8_t* raw = Plan::raw(scratch, buf);
            cp_async_bulk_g2s_l2hint(raw + role_tid * Raw::DATA_BYTES, data, Raw::DATA_BYTES,
                                     ready + buf, policy);
            cp_async_bulk_g2s_l2hint(raw + RAW_SCALE_OFFSET + role_tid * Raw::SCALE_BYTES, scales,
                                     Raw::SCALE_BYTES, ready + buf, policy);
          } else {
            cp_async_bulk_g2s_l2hint(sm.kv_buf(buf) + role_tid * KV::KV_SMEM_STRIDE, data,
                                     KV::D_NOPE, sm.mbar_kv + buf, policy);
          }
        }
      }
    } else {
      Plan::convert_registers();
      int extra_tile = 0;
      for (int tile = 0; tile < tiles; ++tile) {
        const int buf = Cursor::slot(tile);
        const auto sec = section(tile);
        if (!sec.extra) Transaction::wait(sm.mbar_kv + buf, Cursor::phase(tile));
        if (sec.extra) {
          Transaction::wait(ready + buf, Cursor::phase(extra_tile++));
          if (Cursor::reuses(tile)) KvFree::acquire(buf);
          kernels::dsv41_fp8::convert_raw<Plan>(Plan::raw(scratch, buf), sm.kv_buf(buf),
                                                sm.kv_scale_buf(buf), role_tid,
                                                [&](int row) { return sec.index(row, BI) >= 0; });
        }
        RawFree::release(buf);
        KvReady::publish(buf);
      }
    }
  }
};
