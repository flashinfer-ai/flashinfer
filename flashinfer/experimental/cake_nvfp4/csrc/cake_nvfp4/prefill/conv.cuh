/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
// Nemotron 3.5 Super NVFP4-KV prefill, sm_100a.
// Dequantizes the visible NVFP4 pages of a request to FP8 E4M3 pages in the workspace: the
// compact K and V cache the attention kernels read. One CTA per (cache page, KV head).
#include "common.cuh"

namespace flashinfer {
namespace cake_nvfp4 {
namespace prefill {
namespace conv {

constexpr int kThreads = 256;

extern "C" {

__global__ __launch_bounds__(kThreads, 4) void cake_nvfp4_prefill_conv(
    int* __restrict__ k, int* __restrict__ v, int* __restrict__ ksf, int* __restrict__ vsf,
    uint8_t* __restrict__ k8, uint8_t* __restrict__ v8, int kv_page_words, int kv_head_words,
    int sf_page_words, int sf_head_words, const int* page_table, const int* kv_lengths,
    int num_pool_pages, int cache_pages, int table_stride) {
  const int tid = threadIdx.x;
  const int warp = broadcast_lane0(tid / 32);
  const int lane = tid % 32;

  // ---- Role: kwork ----
  if (warp <= 3) {
    {  // kwork_main
      int ph = blockIdx.x;
      int local_page = (ph / 2) % cache_pages;
      int batch = (ph / 2) / cache_pages;
      int page = page_table[batch * table_stride + local_page];
      int hh = ph % 2;
      int t = tid;
      long long kpage =
          (long long)page * (long long)kv_page_words + (long long)(hh * kv_head_words);
      long long spage =
          (long long)page * (long long)sf_page_words + (long long)(hh * sf_head_words);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        int it = i * 128 + t;
        int row = it / 8;
        int ch = it % 8;
        bool valid =
            page >= 0 && page < num_pool_pages && local_page * 128 + row < kv_lengths[batch];
        int kw[2];
        kw[0] = valid ? k[kpage + (long long)(row * 16) + (long long)(ch * 2)] : 0;
        kw[1] = valid ? k[kpage + (long long)(row * 16) + (long long)(ch * 2) + 1] : 0;
        int ksw = valid ? ksf[spage + (long long)(row * 2) + (long long)(ch / 4)] : 0;
        unsigned int ksc = (unsigned int)ksw >> (unsigned int)(ch % 4 * 8);
        unsigned int ko[4];
        {
          uint32_t _qmul4_0_src = kw[0];
          uint32_t _qmul4_0_sf = ksc;
          uint16_t _qmul4_0_a01;
          uint16_t _qmul4_0_a23;
          uint16_t _qmul4_0_s01;
          uint16_t _qmul4_0_s23;
          _qmul4_0_a01 = (uint16_t)(_qmul4_0_src & 0x000000FFu);
          _qmul4_0_a23 = (uint16_t)((_qmul4_0_src >> 8) & 0x000000FFu);
          uint16_t _qmul4_0_sb = (uint16_t)((_qmul4_0_sf >> 0) & 0xFFu);
          _qmul4_0_s01 = (uint16_t)(_qmul4_0_sb | (_qmul4_0_sb << 8));
          _qmul4_0_s23 = _qmul4_0_s01;
          uint32_t _qmul4_0_out;
          asm volatile(
              "{\n"
              ".reg .b8 ab0, ab1;\n"
              ".reg .b32 ah0, ah1, sh0, sh1;\n"
              ".reg .b16 e0, e1;\n"
              "mov.b16 {ab0, _}, %1;\n"
              "mov.b16 {ab1, _}, %2;\n"
              "cvt.rn.f16x2.e2m1x2 ah0, ab0;\n"
              "cvt.rn.f16x2.e2m1x2 ah1, ab1;\n"
              "cvt.rn.f16x2.e4m3x2 sh0, %3;\n"
              "cvt.rn.f16x2.e4m3x2 sh1, %4;\n"
              "mul.rn.f16x2 ah0, ah0, sh0;\n"
              "mul.rn.f16x2 ah1, ah1, sh1;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
              "mov.b32 %0, {e0, e1};\n"
              "}\n"
              : "=r"(_qmul4_0_out)
              : "h"(_qmul4_0_a01), "h"(_qmul4_0_a23), "h"(_qmul4_0_s01), "h"(_qmul4_0_s23));
          ko[0] = _qmul4_0_out;
        }
        {
          uint32_t _qmul4_1_src = kw[0];
          uint32_t _qmul4_1_sf = ksc;
          uint16_t _qmul4_1_a01;
          uint16_t _qmul4_1_a23;
          uint16_t _qmul4_1_s01;
          uint16_t _qmul4_1_s23;
          _qmul4_1_a01 = (uint16_t)((_qmul4_1_src >> 16) & 0x000000FFu);
          _qmul4_1_a23 = (uint16_t)((_qmul4_1_src >> 24) & 0x000000FFu);
          uint16_t _qmul4_1_sb = (uint16_t)((_qmul4_1_sf >> 0) & 0xFFu);
          _qmul4_1_s01 = (uint16_t)(_qmul4_1_sb | (_qmul4_1_sb << 8));
          _qmul4_1_s23 = _qmul4_1_s01;
          uint32_t _qmul4_1_out;
          asm volatile(
              "{\n"
              ".reg .b8 ab0, ab1;\n"
              ".reg .b32 ah0, ah1, sh0, sh1;\n"
              ".reg .b16 e0, e1;\n"
              "mov.b16 {ab0, _}, %1;\n"
              "mov.b16 {ab1, _}, %2;\n"
              "cvt.rn.f16x2.e2m1x2 ah0, ab0;\n"
              "cvt.rn.f16x2.e2m1x2 ah1, ab1;\n"
              "cvt.rn.f16x2.e4m3x2 sh0, %3;\n"
              "cvt.rn.f16x2.e4m3x2 sh1, %4;\n"
              "mul.rn.f16x2 ah0, ah0, sh0;\n"
              "mul.rn.f16x2 ah1, ah1, sh1;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
              "mov.b32 %0, {e0, e1};\n"
              "}\n"
              : "=r"(_qmul4_1_out)
              : "h"(_qmul4_1_a01), "h"(_qmul4_1_a23), "h"(_qmul4_1_s01), "h"(_qmul4_1_s23));
          ko[1] = _qmul4_1_out;
        }
        {
          uint32_t _qmul4_2_src = kw[1];
          uint32_t _qmul4_2_sf = ksc;
          uint16_t _qmul4_2_a01;
          uint16_t _qmul4_2_a23;
          uint16_t _qmul4_2_s01;
          uint16_t _qmul4_2_s23;
          _qmul4_2_a01 = (uint16_t)(_qmul4_2_src & 0x000000FFu);
          _qmul4_2_a23 = (uint16_t)((_qmul4_2_src >> 8) & 0x000000FFu);
          uint16_t _qmul4_2_sb = (uint16_t)((_qmul4_2_sf >> 0) & 0xFFu);
          _qmul4_2_s01 = (uint16_t)(_qmul4_2_sb | (_qmul4_2_sb << 8));
          _qmul4_2_s23 = _qmul4_2_s01;
          uint32_t _qmul4_2_out;
          asm volatile(
              "{\n"
              ".reg .b8 ab0, ab1;\n"
              ".reg .b32 ah0, ah1, sh0, sh1;\n"
              ".reg .b16 e0, e1;\n"
              "mov.b16 {ab0, _}, %1;\n"
              "mov.b16 {ab1, _}, %2;\n"
              "cvt.rn.f16x2.e2m1x2 ah0, ab0;\n"
              "cvt.rn.f16x2.e2m1x2 ah1, ab1;\n"
              "cvt.rn.f16x2.e4m3x2 sh0, %3;\n"
              "cvt.rn.f16x2.e4m3x2 sh1, %4;\n"
              "mul.rn.f16x2 ah0, ah0, sh0;\n"
              "mul.rn.f16x2 ah1, ah1, sh1;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
              "mov.b32 %0, {e0, e1};\n"
              "}\n"
              : "=r"(_qmul4_2_out)
              : "h"(_qmul4_2_a01), "h"(_qmul4_2_a23), "h"(_qmul4_2_s01), "h"(_qmul4_2_s23));
          ko[2] = _qmul4_2_out;
        }
        {
          uint32_t _qmul4_3_src = kw[1];
          uint32_t _qmul4_3_sf = ksc;
          uint16_t _qmul4_3_a01;
          uint16_t _qmul4_3_a23;
          uint16_t _qmul4_3_s01;
          uint16_t _qmul4_3_s23;
          _qmul4_3_a01 = (uint16_t)((_qmul4_3_src >> 16) & 0x000000FFu);
          _qmul4_3_a23 = (uint16_t)((_qmul4_3_src >> 24) & 0x000000FFu);
          uint16_t _qmul4_3_sb = (uint16_t)((_qmul4_3_sf >> 0) & 0xFFu);
          _qmul4_3_s01 = (uint16_t)(_qmul4_3_sb | (_qmul4_3_sb << 8));
          _qmul4_3_s23 = _qmul4_3_s01;
          uint32_t _qmul4_3_out;
          asm volatile(
              "{\n"
              ".reg .b8 ab0, ab1;\n"
              ".reg .b32 ah0, ah1, sh0, sh1;\n"
              ".reg .b16 e0, e1;\n"
              "mov.b16 {ab0, _}, %1;\n"
              "mov.b16 {ab1, _}, %2;\n"
              "cvt.rn.f16x2.e2m1x2 ah0, ab0;\n"
              "cvt.rn.f16x2.e2m1x2 ah1, ab1;\n"
              "cvt.rn.f16x2.e4m3x2 sh0, %3;\n"
              "cvt.rn.f16x2.e4m3x2 sh1, %4;\n"
              "mul.rn.f16x2 ah0, ah0, sh0;\n"
              "mul.rn.f16x2 ah1, ah1, sh1;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
              "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
              "mov.b32 %0, {e0, e1};\n"
              "}\n"
              : "=r"(_qmul4_3_out)
              : "h"(_qmul4_3_a01), "h"(_qmul4_3_a23), "h"(_qmul4_3_s01), "h"(_qmul4_3_s23));
          ko[3] = _qmul4_3_out;
        }
        int kprow = ((row & 31) >> 1) * 8 + (row >> 5) * 2 + (row & 1);
        reinterpret_cast<int4*>(
            k8 + (((long long)ph * 128 + (long long)kprow) * 128 + (long long)(ch * 16)))[0] =
            reinterpret_cast<int4*>(ko + 0)[0];
      }
    }
    // ---- Role: vwork ----
  } else if (warp >= 4 && warp <= 7) {
    {  // vwork_main
      int phv = blockIdx.x;
      int local_page_v = (phv / 2) % cache_pages;
      int batch_v = (phv / 2) / cache_pages;
      int pagev = page_table[batch_v * table_stride + local_page_v];
      int hv = phv % 2;
      int tv = tid - 128;
      long long vpage =
          (long long)pagev * (long long)kv_page_words + (long long)(hv * kv_head_words);
      long long vspage =
          (long long)pagev * (long long)sf_page_words + (long long)(hv * sf_head_words);
#pragma unroll
      for (int i_1 = 0; i_1 < 8; i_1++) {
        int itv = i_1 * 128 + tv;
        int rv = itv / 8;
        int cv = itv % 8;
        unsigned int qv[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          bool valid_v =
              pagev >= 0 && pagev < num_pool_pages && local_page_v * 128 + rv < kv_lengths[batch_v];
          int vwd = valid_v
                        ? v[vpage + (long long)(rv * 16) + (long long)(4 * u) + (long long)(cv / 2)]
                        : 0;
          unsigned int piece = (unsigned int)vwd >> (unsigned int)(cv % 2 * 16) & 65535;
          int vsw =
              valid_v
                  ? vsf[vspage + (long long)(rv / 4 * 8) + (long long)(2 * u) + (long long)(cv / 4)]
                  : 0;
          unsigned int vsc = (unsigned int)vsw >> (unsigned int)((rv & 3) * 8);
          {
            uint32_t _qmul4_0_src = piece;
            uint32_t _qmul4_0_sf = vsc;
            uint16_t _qmul4_0_a01;
            uint16_t _qmul4_0_a23;
            uint16_t _qmul4_0_s01;
            uint16_t _qmul4_0_s23;
            _qmul4_0_a01 = (uint16_t)(_qmul4_0_src & 0x000000FFu);
            _qmul4_0_a23 = (uint16_t)((_qmul4_0_src >> 8) & 0x000000FFu);
            uint16_t _qmul4_0_sb = (uint16_t)((_qmul4_0_sf >> 0) & 0xFFu);
            _qmul4_0_s01 = (uint16_t)(_qmul4_0_sb | (_qmul4_0_sb << 8));
            _qmul4_0_s23 = _qmul4_0_s01;
            uint32_t _qmul4_0_out;
            asm volatile(
                "{\n"
                ".reg .b8 ab0, ab1;\n"
                ".reg .b32 ah0, ah1, sh0, sh1;\n"
                ".reg .b16 e0, e1;\n"
                "mov.b16 {ab0, _}, %1;\n"
                "mov.b16 {ab1, _}, %2;\n"
                "cvt.rn.f16x2.e2m1x2 ah0, ab0;\n"
                "cvt.rn.f16x2.e2m1x2 ah1, ab1;\n"
                "cvt.rn.f16x2.e4m3x2 sh0, %3;\n"
                "cvt.rn.f16x2.e4m3x2 sh1, %4;\n"
                "mul.rn.f16x2 ah0, ah0, sh0;\n"
                "mul.rn.f16x2 ah1, ah1, sh1;\n"
                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                "mov.b32 %0, {e0, e1};\n"
                "}\n"
                : "=r"(_qmul4_0_out)
                : "h"(_qmul4_0_a01), "h"(_qmul4_0_a23), "h"(_qmul4_0_s01), "h"(_qmul4_0_s23));
            qv[u] = _qmul4_0_out;
          }
        }
        unsigned int vp[4];
        vp[0] = __byte_perm(qv[0], qv[1], 0x5410);
        vp[1] = __byte_perm(qv[2], qv[3], 0x5410);
        vp[2] = __byte_perm(qv[0], qv[1], 0x7632);
        vp[3] = __byte_perm(qv[2], qv[3], 0x7632);
        reinterpret_cast<int4*>(
            v8 + (((long long)phv * 128 + (long long)rv) * 128 + (long long)(cv * 16)))[0] =
            reinterpret_cast<int4*>(vp + 0)[0];
      }
    }
  }
}

}  // extern "C"

}  // namespace conv
}  // namespace prefill
}  // namespace cake_nvfp4
}  // namespace flashinfer
