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
// FP8 attention on the compact FP8 cache with split scheduling: a tile is 256 query rows of
// one head, and the CTAs left over by the last wave of tiles help with its KV range. A split
// tile leaves two partial results in the workspace, which comb.cuh merges.
#include "common.cuh"

namespace flashinfer {
namespace cake_nvfp4 {
namespace prefill {
namespace attn {

constexpr int kThreads = 640;
// Dynamic shared memory: the barriers and the TMEM address in the first 1024 bytes, then the
// stages of Q, K, V and P, 16384 bytes each.
constexpr int kSmemQOffset = 1024;
constexpr int kSmemKOffset = 33792;
constexpr int kSmemVOffset = 66560;
constexpr int kSmemPOffset = 132096;
constexpr int kSharedBytes = 197632;

extern "C" {

__global__ __launch_bounds__(kThreads, 1) void cake_nvfp4_prefill_attn(
    uint8_t* __restrict__ out, float* __restrict__ part_o, float* __restrict__ part_ml,
    float* __restrict__ pmax, int pmax_chunks, int* __restrict__ block_tables,
    int* __restrict__ q_indptr, int* __restrict__ kv_lengths, int table_stride, int n_qtiles,
    int nctas, int qf, int tile_base, int nrem, float x_frac, float scale_log2, float out_scale,
    const float* scale_log2_ptr, const float* out_scale_ptr,
    __grid_constant__ TensorMaps<3> const tensor_maps) {
  if (scale_log2_ptr) scale_log2 = *scale_log2_ptr;
  if (out_scale_ptr) out_scale = *out_scale_ptr;
  uint64_t tensor_maps_addr;
  asm volatile("mov.b64 %0, %1;" : "=l"(tensor_maps_addr) : "l"((uint64_t)(&tensor_maps)));

  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  int smem;
  smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

  // Shared memory stages
  const int smem_q_addr = smem + kSmemQOffset;
  const int smem_k8_addr = smem + kSmemKOffset;
  const int smem_v8_addr = smem + kSmemVOffset;
  const int smem_p_addr = smem + kSmemPOffset;

  // Mbarrier init (12 groups, 28 barriers)
  // Mbarriers at smem_raw[0..224)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    // q_full: 1 barriers, init_count=1
    mbarrier_init_pred(smem + 0, 1, leader);
    // q_empty: 1 barriers, init_count=2
    mbarrier_init_pred(smem + 8, 2, leader);
    // k_full: 2 barriers, init_count=1
    mbarrier_init_pred(smem + 16, 1, leader);
    mbarrier_init_pred(smem + 24, 1, leader);
    // k_empty: 2 barriers, init_count=2
    mbarrier_init_pred(smem + 32, 2, leader);
    mbarrier_init_pred(smem + 40, 2, leader);
    // v_full: 4 barriers, init_count=1
    mbarrier_init_pred(smem + 48, 1, leader);
    mbarrier_init_pred(smem + 56, 1, leader);
    mbarrier_init_pred(smem + 64, 1, leader);
    mbarrier_init_pred(smem + 72, 1, leader);
    // v_empty: 4 barriers, init_count=2
    mbarrier_init_pred(smem + 80, 2, leader);
    mbarrier_init_pred(smem + 88, 2, leader);
    mbarrier_init_pred(smem + 96, 2, leader);
    mbarrier_init_pred(smem + 104, 2, leader);
    // s_full: 2 barriers, init_count=1
    mbarrier_init_pred(smem + 112, 1, leader);
    mbarrier_init_pred(smem + 120, 1, leader);
    // s_free: 2 barriers, init_count=256
    mbarrier_init_pred(smem + 128, 256, leader);
    mbarrier_init_pred(smem + 136, 256, leader);
    // p_full: 4 barriers, init_count=256
    mbarrier_init_pred(smem + 144, 256, leader);
    mbarrier_init_pred(smem + 152, 256, leader);
    mbarrier_init_pred(smem + 160, 256, leader);
    mbarrier_init_pred(smem + 168, 256, leader);
    // tail_done: 2 barriers, init_count=1
    mbarrier_init_pred(smem + 176, 1, leader);
    mbarrier_init_pred(smem + 184, 1, leader);
    // o_full: 2 barriers, init_count=1
    mbarrier_init_pred(smem + 192, 1, leader);
    mbarrier_init_pred(smem + 200, 1, leader);
    // o_free: 2 barriers, init_count=256
    mbarrier_init_pred(smem + 208, 256, leader);
    mbarrier_init_pred(smem + 216, 256, leader);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }

  __syncthreads();

  // TMEM alloc (512 columns, 512 used)
  volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
  if (warp == 0) {
    int _tmem_hold = smem + 224;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(_tmem_hold),
        "r"(512)
        : "memory");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");

  const int mbar_base = smem;
  const int q_full_addr = mbar_base + 0;
  const int q_empty_addr = mbar_base + 8;
  const int k_full_addr = mbar_base + 16;
  const int k_empty_addr = mbar_base + 32;
  const int v_full_addr = mbar_base + 48;
  const int v_empty_addr = mbar_base + 80;
  const int s_full_addr = mbar_base + 112;
  const int s_free_addr = mbar_base + 128;
  const int p_full_addr = mbar_base + 144;
  const int tail_done_addr = mbar_base + 176;
  const int o_full_addr = mbar_base + 192;
  const int o_free_addr = mbar_base + 208;
  const int taddr = tmem_addr_storage[0];

  // TMEM columns
  const int tmem_tmem_s0 = taddr;
  const int tmem_tmem_s1 = taddr + 128;
  const int tmem_tmem_o0 = taddr + 256;
  const int tmem_tmem_o1 = taddr + 384;

  // ---- Role: soft ----
  if (warp <= 15) {
    {  // soft_main
      int cta_s = blockIdx.x;
      int nit_s = qf;
      if (nrem > 0) {
        if (cta_s < nrem) {
          nit_s = qf + 1;
        } else if (cta_s - nrem < nrem) {
          nit_s = qf + (nrem - 1 - (cta_s - nrem)) / (nctas - nrem) + 1;
        }
      }
      int gs_s = 0;
      int gp_s = 0;
      int qi = warp / 8;
      int hf = warp / 4 & 1;
      int lg = warp % 4;
      int uu = lane & 3;
      int lbase = lg * 32 + hf * 16;
      int ra = lbase + (lane >> 2);
      int rb = ra + 8;
      int s_addr = taddr + (unsigned int)(qi * 128) + (unsigned int)(lbase << 16);
      int o_addr = taddr + 256 + (unsigned int)(qi * 128) + (unsigned int)(lbase << 16);
      // A quarter of the exponentials avoid the exp2 instruction: adding and subtracting
      // 1.5 * 2^23 rounds x to an integer n, 2^(x - n) is a quadratic in x - n, and n goes into
      // the exponent bits (the low mantissa bits of x + 1.5 * 2^23 hold it).
      float2 _f2_0 = make_float2(12582912.0f, 12582912.0f);
      float2 _f2_1 = make_float2(0.0f, 0.0f);
      float2 _f2_2 = make_float2(0.24229444f, 0.24229444f);
      float2 _f2_3 = make_float2(0.70150868f, 0.70150868f);
      float2 _f2_4 = make_float2(0.99944824f, 0.99944824f);
      float2 _f2_5 = make_float2(scale_log2, scale_log2);
#pragma unroll 1
      for (int it_s = 0; it_s < nit_s; it_s++) {
        int tile_s = tile_base + cta_s + it_s * nctas;
        int kind_s = 0;
        if (it_s >= qf) {
          if (cta_s < nrem) {
            tile_s = tile_base + qf * nctas + cta_s;
            kind_s = 1;
          } else {
            tile_s = tile_base + qf * nctas + (cta_s - nrem) + (it_s - qf) * (nctas - nrem);
            kind_s = 2;
          }
        }
        int bz_s = tile_s / (32 * n_qtiles);
        int rem_s = tile_s % (32 * n_qtiles);
        int head_s = rem_s % 32;
        int q0_s = ((tile_base > 0) ? n_qtiles - 1 - rem_s / 32 : rem_s / 32) * 256;
        int qs_s = q_indptr[bz_s];
        int ql_s = q_indptr[bz_s + 1] - qs_s;
        int kvl_s = kv_lengths[bz_s];
        int _min_6 = ((q0_s + 255) < (ql_s - 1) ? (q0_s + 255) : (ql_s - 1));
        int last_s = _min_6;
        int _max_6 = ((kvl_s - ql_s + last_s) > (0) ? (kvl_s - ql_s + last_s) : (0));
        int lim_s = _max_6;
        int ntot_s = lim_s / 128 + 1;
        int _max_7 =
            (((int)(x_frac * (float)ntot_s)) > (1) ? ((int)(x_frac * (float)ntot_s)) : (1));
        int _min_7 = ((_max_7) < (ntot_s - 1) ? (_max_7) : (ntot_s - 1));
        int xs_s = _min_7;
        int beg_s = 0;
        int nst_s = ntot_s;
        if (kind_s == 1) {
          beg_s = xs_s;
        }
        // A one-page tile has xs=0. Process its sole page in both partials:
        // combining identical distributions preserves O, and every MMA has a K arrival.
        if (kind_s == 2) {
          nst_s = xs_s > 0 ? xs_s : 1;
        }
        int qrow_a = q0_s + qi * 128 + ra;
        int qrow_b = qrow_a + 8;
        int lim_a = kvl_s - ql_s + qrow_a;
        int lim_b = lim_a + 8;
        float m_a = -3.4028234663852886e+38f;
        float m_b = -3.4028234663852886e+38f;
        if (kind_s == 1) {
          long long pmrow =
              (long long)(tile_s - tile_base - qf * nctas) * (long long)pmax_chunks * 256 +
              (long long)(qi * 128) + (long long)ra;
#pragma unroll 1
          for (int pc = 0; pc < pmax_chunks; pc++) {
            float _max_8 = max_noftz(m_a, pmax[pmrow + (long long)(pc * 256)]);
            m_a = _max_8;
            float _max_9 = max_noftz(m_b, pmax[pmrow + (long long)(pc * 256) + 8]);
            m_b = _max_9;
          }
        }
        float l_a = 0.0f;
        float l_b = 0.0f;
#pragma unroll 1
        for (int js = beg_s; js < nst_s; js++) {
          mbarrier_wait(s_full_addr + (qi) * 8, gs_s & 1);
          asm volatile("tcgen05.fence::after_thread_sync;");
          float _tmem_load_0[64];
          asm volatile(
              "tcgen05.ld.sync.aligned.16x256b.x16.b32"
              " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
              "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, "
              "%34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, "
              "%50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])),
                "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
              : "r"(s_addr)
              : "memory");
          asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
          asm volatile("tcgen05.fence::before_thread_sync;");
          mbarrier_arrive(s_free_addr + (qi) * 8);
          gs_s = gs_s + 1;
          int kvbase = js * 128 + uu * 32;
          if (lim_a < kvbase + 31) {
#pragma unroll
            for (int c = 0; c < 16; c++) {
              _tmem_load_0[4 * c] =
                  ((lim_a >= kvbase + 2 * c) ? _tmem_load_0[4 * c] : -3.4028234663852886e+38f);
              _tmem_load_0[4 * c + 1] = ((lim_a >= kvbase + 2 * c + 1) ? _tmem_load_0[4 * c + 1]
                                                                       : -3.4028234663852886e+38f);
              _tmem_load_0[4 * c + 2] =
                  ((lim_b >= kvbase + 2 * c) ? _tmem_load_0[4 * c + 2] : -3.4028234663852886e+38f);
              _tmem_load_0[4 * c + 3] = ((lim_b >= kvbase + 2 * c + 1) ? _tmem_load_0[4 * c + 3]
                                                                       : -3.4028234663852886e+38f);
            }
          }
          float xa[4];
          float xb[4];
#pragma unroll
          for (int g = 0; g < 4; g++) {
            float _max3_0;
            asm volatile("max.f32 %0, %1, %2, %3;"
                         : "=f"(_max3_0)
                         : "f"(_tmem_load_0[16 * g]), "f"(_tmem_load_0[16 * g + 1]),
                           "f"(_tmem_load_0[16 * g + 4]));
            xa[g] = _max3_0;
            float _max3_1;
            asm volatile("max.f32 %0, %1, %2, %3;"
                         : "=f"(_max3_1)
                         : "f"(xa[g]), "f"(_tmem_load_0[16 * g + 5]),
                           "f"(_tmem_load_0[16 * g + 8]));
            xa[g] = _max3_1;
            float _max3_2;
            asm volatile("max.f32 %0, %1, %2, %3;"
                         : "=f"(_max3_2)
                         : "f"(xa[g]), "f"(_tmem_load_0[16 * g + 9]),
                           "f"(_tmem_load_0[16 * g + 12]));
            xa[g] = _max3_2;
            float _max3_3;
            asm volatile("max.f32 %0, %1, %2, %3;"
                         : "=f"(_max3_3)
                         : "f"(_tmem_load_0[16 * g + 2]), "f"(_tmem_load_0[16 * g + 3]),
                           "f"(_tmem_load_0[16 * g + 6]));
            xb[g] = _max3_3;
            float _max3_4;
            asm volatile("max.f32 %0, %1, %2, %3;"
                         : "=f"(_max3_4)
                         : "f"(xb[g]), "f"(_tmem_load_0[16 * g + 7]),
                           "f"(_tmem_load_0[16 * g + 10]));
            xb[g] = _max3_4;
            float _max3_5;
            asm volatile("max.f32 %0, %1, %2, %3;"
                         : "=f"(_max3_5)
                         : "f"(xb[g]), "f"(_tmem_load_0[16 * g + 11]),
                           "f"(_tmem_load_0[16 * g + 14]));
            xb[g] = _max3_5;
          }
          float _max3_6;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_6)
                       : "f"(xa[0]), "f"(xa[1]), "f"(xa[2]));
          float ma = _max3_6;
          float _max3_7;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_7)
                       : "f"(ma), "f"(xa[3]), "f"(_tmem_load_0[13]));
          ma = _max3_7;
          float _max3_8;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_8)
                       : "f"(ma), "f"(_tmem_load_0[29]), "f"(_tmem_load_0[45]));
          ma = _max3_8;
          float _max_10 = max_noftz(ma, _tmem_load_0[61]);
          ma = _max_10;
          float _max3_9;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_9)
                       : "f"(xb[0]), "f"(xb[1]), "f"(xb[2]));
          float mb = _max3_9;
          float _max3_10;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_10)
                       : "f"(mb), "f"(xb[3]), "f"(_tmem_load_0[15]));
          mb = _max3_10;
          float _max3_11;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_11)
                       : "f"(mb), "f"(_tmem_load_0[31]), "f"(_tmem_load_0[47]));
          mb = _max3_11;
          float _max_11 = max_noftz(mb, _tmem_load_0[63]);
          mb = _max_11;
          float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ma, 1);
          float _max_12 = max_noftz(ma, _shfl_xor_0);
          ma = _max_12;
          float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, mb, 1);
          float _max_13 = max_noftz(mb, _shfl_xor_1);
          mb = _max_13;
          float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ma, 2);
          float _max_14 = max_noftz(ma, _shfl_xor_2);
          ma = _max_14;
          float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, mb, 2);
          float _max_15 = max_noftz(mb, _shfl_xor_3);
          mb = _max_15;
          float _max_16 = max_noftz(m_a, ma);
          float mna = _max_16;
          float _max_17 = max_noftz(m_b, mb);
          float mnb = _max_17;
          int _vote_0 = __any_sync(0xFFFFFFFF, mna > m_a || mnb > m_b);
          int need = _vote_0;
          float alpha_a = 1.0f;
          float alpha_b = 1.0f;
          if (beg_s < js) {
            if (need != 0) {
              float _exp2_0 = approx_exp2((m_a - mna) * scale_log2);
              alpha_a = _exp2_0;
              float _exp2_1 = approx_exp2((m_b - mnb) * scale_log2);
              alpha_b = _exp2_1;
              l_a = l_a * alpha_a;
              l_b = l_b * alpha_b;
            }
          }
          m_a = mna;
          m_b = mnb;
          // log2(448): P is scaled into the E4M3 range.
          float nega = 8.807354922057604f - mna * scale_log2;
          float negb = 8.807354922057604f - mnb * scale_log2;
          float2 _f2_6 = make_float2(nega, nega);
          float2 _f2_7 = make_float2(negb, negb);
#pragma unroll
          for (int c_1 = 0; c_1 < 16; c_1++) {
            float2 _f2_8 = make_float2(_tmem_load_0[4 * c_1], _tmem_load_0[4 * c_1 + 1]);
            float2 _f2_9 = make_float2(_tmem_load_0[4 * c_1 + 2], _tmem_load_0[4 * c_1 + 3]);
            float2 ya = fma_f32x2(_f2_8, _f2_5, _f2_6);
            float2 yb = fma_f32x2(_f2_9, _f2_5, _f2_7);
            _tmem_load_0[4 * c_1] = ya.x;
            _tmem_load_0[4 * c_1 + 1] = ya.y;
            _tmem_load_0[4 * c_1 + 2] = yb.x;
            _tmem_load_0[4 * c_1 + 3] = yb.y;
          }
#pragma unroll
          for (int c2 = 0; c2 < 32; c2++) {
            if (c2 % 8 < 2) {
              float _max_18 = max_noftz(_tmem_load_0[2 * c2], -125.0f);
              float xa0 = _max_18;
              float _max_19 = max_noftz(_tmem_load_0[2 * c2 + 1], -125.0f);
              float xb0 = _max_19;
              float2 _f2_10 = make_float2(xa0, xb0);
              float2 tt = add_f32x2(_f2_10, _f2_0);
              float2 rr = sub_f32x2(tt, _f2_0);
              float2 ff = sub_f32x2(_f2_10, rr);
              float2 q1 = fma_f32x2(ff, _f2_2, _f2_3);
              float2 q0 = fma_f32x2(q1, ff, _f2_4);
              float pa = q0.x;
              float pb_ = q0.y;
              float ta = tt.x;
              float tb = tt.y;
              unsigned int ba = as_u32(pa) + (as_u32(ta) << 23);
              unsigned int bb = as_u32(pb_) + (as_u32(tb) << 23);
              _tmem_load_0[2 * c2] = *reinterpret_cast<float*>(&ba);
              _tmem_load_0[2 * c2 + 1] = *reinterpret_cast<float*>(&bb);
            } else {
              float _exp2_2 = approx_exp2(_tmem_load_0[2 * c2]);
              _tmem_load_0[2 * c2] = _exp2_2;
              float _exp2_3 = approx_exp2(_tmem_load_0[2 * c2 + 1]);
              _tmem_load_0[2 * c2 + 1] = _exp2_3;
            }
          }
          float2 _f2_11 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
          float2 sa0 = _f2_11;
          float2 _f2_12 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
          float2 sb0 = _f2_12;
          float2 _f2_13 = make_float2(_tmem_load_0[4], _tmem_load_0[5]);
          float2 sa1 = _f2_13;
          float2 _f2_14 = make_float2(_tmem_load_0[6], _tmem_load_0[7]);
          float2 sb1 = _f2_14;
#pragma unroll
          for (int c_2 = 1; c_2 < 8; c_2++) {
            float2 _f2_15 = make_float2(_tmem_load_0[8 * c_2], _tmem_load_0[8 * c_2 + 1]);
            float2 _f2_16 = make_float2(_tmem_load_0[8 * c_2 + 2], _tmem_load_0[8 * c_2 + 3]);
            float2 _f2_17 = make_float2(_tmem_load_0[8 * c_2 + 4], _tmem_load_0[8 * c_2 + 5]);
            float2 _f2_18 = make_float2(_tmem_load_0[8 * c_2 + 6], _tmem_load_0[8 * c_2 + 7]);
            sa0 = add_f32x2(sa0, _f2_15);
            sb0 = add_f32x2(sb0, _f2_16);
            sa1 = add_f32x2(sa1, _f2_17);
            sb1 = add_f32x2(sb1, _f2_18);
          }
          float2 sat = add_f32x2(sa0, sa1);
          float2 sbt = add_f32x2(sb0, sb1);
          l_a = l_a + (sat.x + sat.y);
          l_b = l_b + (sbt.x + sbt.y);
          float pva[32];
          float pvb[32];
#pragma unroll
          for (int c_3 = 0; c_3 < 16; c_3++) {
            pva[2 * c_3] = _tmem_load_0[4 * c_3];
            pva[2 * c_3 + 1] = _tmem_load_0[4 * c_3 + 1];
            pvb[2 * c_3] = _tmem_load_0[4 * c_3 + 2];
            pvb[2 * c_3 + 1] = _tmem_load_0[4 * c_3 + 3];
          }
          unsigned int pka[8];
          unsigned int pkb[8];
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[1]), "f"(pva[0]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[3]), "f"(pva[2]));
            pka[0] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[5]), "f"(pva[4]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[7]), "f"(pva[6]));
            pka[1] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[9]), "f"(pva[8]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[11]), "f"(pva[10]));
            pka[2] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[13]), "f"(pva[12]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[15]), "f"(pva[14]));
            pka[3] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[17]), "f"(pva[16]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[19]), "f"(pva[18]));
            pka[4] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[21]), "f"(pva[20]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[23]), "f"(pva[22]));
            pka[5] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[25]), "f"(pva[24]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[27]), "f"(pva[26]));
            pka[6] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pva[29]), "f"(pva[28]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pva[31]), "f"(pva[30]));
            pka[7] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[1]), "f"(pvb[0]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[3]), "f"(pvb[2]));
            pkb[0] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[5]), "f"(pvb[4]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[7]), "f"(pvb[6]));
            pkb[1] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[9]), "f"(pvb[8]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[11]), "f"(pvb[10]));
            pkb[2] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[13]), "f"(pvb[12]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[15]), "f"(pvb[14]));
            pkb[3] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[17]), "f"(pvb[16]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[19]), "f"(pvb[18]));
            pkb[4] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[21]), "f"(pvb[20]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[23]), "f"(pvb[22]));
            pkb[5] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[25]), "f"(pvb[24]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[27]), "f"(pvb[26]));
            pkb[6] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(pvb[29]), "f"(pvb[28]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(pvb[31]), "f"(pvb[30]));
            pkb[7] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          int pbase = (qi * 2 + (gp_s & 1)) * 16384;
          int pra = pbase + ra * 128;
          int prb = pbase + rb * 128;
#pragma unroll
          for (int h = 0; h < 2; h++) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                             smem_p_addr + (unsigned int)(pra + (uu * 2 + h ^ ra & 7) * 16)),
                         "r"(pka[4 * h]), "r"(pka[4 * h + 1]), "r"(pka[4 * h + 2]),
                         "r"(pka[4 * h + 3])
                         : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(
                             smem_p_addr + (unsigned int)(prb + (uu * 2 + h ^ rb & 7) * 16)),
                         "r"(pkb[4 * h]), "r"(pkb[4 * h + 1]), "r"(pkb[4 * h + 2]),
                         "r"(pkb[4 * h + 3])
                         : "memory");
          }
          if (beg_s < js) {
            if (need != 0) {
              if (nst_s > js + 1) {
                mbarrier_wait(s_full_addr + (qi) * 8, gs_s & 1);
              } else {
                mbarrier_wait(tail_done_addr + (qi) * 8, it_s & 1);
              }
              asm volatile("tcgen05.fence::after_thread_sync;");
#pragma unroll
              for (int c4 = 0; c4 < 2; c4++) {
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, "
                    "%17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
                    "[%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])),
                      "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(o_addr + c4 * 64)
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
                for (int c_4 = 0; c_4 < 8; c_4++) {
                  _tmem_load_1[4 * c_4] = _tmem_load_1[4 * c_4] * alpha_a;
                  _tmem_load_1[4 * c_4 + 1] = _tmem_load_1[4 * c_4 + 1] * alpha_a;
                  _tmem_load_1[4 * c_4 + 2] = _tmem_load_1[4 * c_4 + 2] * alpha_b;
                  _tmem_load_1[4 * c_4 + 3] = _tmem_load_1[4 * c_4 + 3] * alpha_b;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                    "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, "
                    "%31, %32};" ::"r"(o_addr + c4 * 64),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])),
                    "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31]))
                    : "memory");
              }
            }
          }
          asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
          asm volatile("tcgen05.fence::before_thread_sync;");
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          mbarrier_arrive(p_full_addr + (qi * 2 + (gp_s & 1)) * 8);
          gp_s = gp_s + 1;
        }
        mbarrier_wait(tail_done_addr + (qi) * 8, it_s & 1);
        mbarrier_wait(o_full_addr + (qi) * 8, it_s & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        float _tmem_load_2[64];
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
            "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
            "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, "
            "%52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])),
              "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
            : "r"(o_addr)
            : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        mbarrier_arrive(o_free_addr + (qi) * 8);
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, l_a, 1);
        l_a = l_a + _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, l_b, 1);
        l_b = l_b + _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, l_a, 2);
        l_a = l_a + _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, l_b, 2);
        l_b = l_b + _shfl_xor_7;
        if (kind_s == 0) {
          float inv_a = ((l_a > 0.0f) ? out_scale / l_a : 0.0f);
          float inv_b = ((l_b > 0.0f) ? out_scale / l_b : 0.0f);
          float oa[32];
          float ob[32];
#pragma unroll
          for (int c_5 = 0; c_5 < 16; c_5++) {
            oa[2 * c_5] = _tmem_load_2[4 * c_5] * inv_a;
            oa[2 * c_5 + 1] = _tmem_load_2[4 * c_5 + 1] * inv_a;
            ob[2 * c_5] = _tmem_load_2[4 * c_5 + 2] * inv_b;
            ob[2 * c_5 + 1] = _tmem_load_2[4 * c_5 + 3] * inv_b;
          }
          unsigned int oka[8];
          unsigned int okb[8];
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[1]), "f"(oa[0]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[3]), "f"(oa[2]));
            oka[0] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[5]), "f"(oa[4]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[7]), "f"(oa[6]));
            oka[1] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[9]), "f"(oa[8]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[11]), "f"(oa[10]));
            oka[2] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[13]), "f"(oa[12]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[15]), "f"(oa[14]));
            oka[3] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[17]), "f"(oa[16]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[19]), "f"(oa[18]));
            oka[4] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[21]), "f"(oa[20]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[23]), "f"(oa[22]));
            oka[5] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[25]), "f"(oa[24]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[27]), "f"(oa[26]));
            oka[6] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(oa[29]), "f"(oa[28]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(oa[31]), "f"(oa[30]));
            oka[7] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[1]), "f"(ob[0]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[3]), "f"(ob[2]));
            okb[0] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[5]), "f"(ob[4]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[7]), "f"(ob[6]));
            okb[1] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[9]), "f"(ob[8]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[11]), "f"(ob[10]));
            okb[2] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[13]), "f"(ob[12]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[15]), "f"(ob[14]));
            okb[3] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[17]), "f"(ob[16]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[19]), "f"(ob[18]));
            okb[4] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[21]), "f"(ob[20]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[23]), "f"(ob[22]));
            okb[5] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[25]), "f"(ob[24]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[27]), "f"(ob[26]));
            okb[6] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          {
            unsigned short _lo, _hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(ob[29]), "f"(ob[28]));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(ob[31]), "f"(ob[30]));
            okb[7] = (unsigned)_lo | ((unsigned)_hi << 16);
          }
          if (qrow_a < ql_s) {
            long long obase_a =
                ((long long)(qs_s + qrow_a) * 32 + (long long)head_s) * 128 + (long long)(uu * 32);
            reinterpret_cast<int4*>(out + obase_a)[0] = reinterpret_cast<int4*>(oka + 0)[0];
            reinterpret_cast<int4*>(out + (obase_a + 16))[0] = reinterpret_cast<int4*>(oka + 4)[0];
          }
          if (qrow_b < ql_s) {
            long long obase_b =
                ((long long)(qs_s + qrow_b) * 32 + (long long)head_s) * 128 + (long long)(uu * 32);
            reinterpret_cast<int4*>(out + obase_b)[0] = reinterpret_cast<int4*>(okb + 0)[0];
            reinterpret_cast<int4*>(out + (obase_b + 16))[0] = reinterpret_cast<int4*>(okb + 4)[0];
          }
        } else {
          int slot_s = 2 * (tile_s - tile_base - qf * nctas) + ((kind_s == 1) ? 1 : 0);
          long long prow_a = (long long)slot_s * 256 + (long long)(qi * 128) + (long long)ra;
          long long prow_b = prow_a + 8;
          float pa32[32];
          float pb32[32];
#pragma unroll
          for (int c_6 = 0; c_6 < 16; c_6++) {
            pa32[2 * c_6] = _tmem_load_2[4 * c_6];
            pa32[2 * c_6 + 1] = _tmem_load_2[4 * c_6 + 1];
            pb32[2 * c_6] = _tmem_load_2[4 * c_6 + 2];
            pb32[2 * c_6 + 1] = _tmem_load_2[4 * c_6 + 3];
          }
#pragma unroll
          for (int c8 = 0; c8 < 8; c8++) {
            reinterpret_cast<int4*>(
                part_o + (prow_a * 128 + (long long)(uu * 32) + (long long)(c8 * 4)))[0] =
                reinterpret_cast<int4*>(pa32 + 4 * c8)[0];
            reinterpret_cast<int4*>(
                part_o + (prow_b * 128 + (long long)(uu * 32) + (long long)(c8 * 4)))[0] =
                reinterpret_cast<int4*>(pb32 + 4 * c8)[0];
          }
          if (uu == 0) {
            part_ml[prow_a * 2] = m_a;
            part_ml[prow_a * 2 + 1] = l_a;
            part_ml[prow_b * 2] = m_b;
            part_ml[prow_b * 2 + 1] = l_b;
          }
        }
      }
    }
    // ---- Role: mma0 ----
  } else if (warp == 16) {
    {  // mma0_main
      int cta_m0 = blockIdx.x;
      int nit_m0 = qf;
      if (nrem > 0) {
        if (cta_m0 < nrem) {
          nit_m0 = qf + 1;
        } else if (cta_m0 - nrem < nrem) {
          nit_m0 = qf + (nrem - 1 - (cta_m0 - nrem)) / (nctas - nrem) + 1;
        }
      }
      int gk_m0 = 0;
      int gs_m0 = 0;
      int gv_m0 = 0;
      int gp_m0 = 0;
#pragma unroll 1
      for (int it_m0 = 0; it_m0 < nit_m0; it_m0++) {
        int tile_m0 = tile_base + cta_m0 + it_m0 * nctas;
        int kind_m0 = 0;
        if (it_m0 >= qf) {
          if (cta_m0 < nrem) {
            tile_m0 = tile_base + qf * nctas + cta_m0;
            kind_m0 = 1;
          } else {
            tile_m0 = tile_base + qf * nctas + (cta_m0 - nrem) + (it_m0 - qf) * (nctas - nrem);
            kind_m0 = 2;
          }
        }
        int bz_m0 = tile_m0 / (32 * n_qtiles);
        int rem_m0 = tile_m0 % (32 * n_qtiles);
        int head_m0 = rem_m0 % 32;
        int q0_m0 = ((tile_base > 0) ? n_qtiles - 1 - rem_m0 / 32 : rem_m0 / 32) * 256;
        int qs_m0 = q_indptr[bz_m0];
        int ql_m0 = q_indptr[bz_m0 + 1] - qs_m0;
        int kvl_m0 = kv_lengths[bz_m0];
        int _min_2 = ((q0_m0 + 255) < (ql_m0 - 1) ? (q0_m0 + 255) : (ql_m0 - 1));
        int last_m0 = _min_2;
        int _max_2 = ((kvl_m0 - ql_m0 + last_m0) > (0) ? (kvl_m0 - ql_m0 + last_m0) : (0));
        int lim_m0 = _max_2;
        int ntot_m0 = lim_m0 / 128 + 1;
        int _max_3 =
            (((int)(x_frac * (float)ntot_m0)) > (1) ? ((int)(x_frac * (float)ntot_m0)) : (1));
        int _min_3 = ((_max_3) < (ntot_m0 - 1) ? (_max_3) : (ntot_m0 - 1));
        int xs_m0 = _min_3;
        int beg_m0 = 0;
        int nst_m0 = ntot_m0;
        if (kind_m0 == 1) {
          beg_m0 = xs_m0;
        }
        if (kind_m0 == 2) {
          nst_m0 = xs_m0 > 0 ? xs_m0 : 1;
        }
        mbarrier_wait(q_full_addr, it_m0 & 1);
        int sk0_m0 = gk_m0 % 2;
        mbarrier_wait(k_full_addr + (sk0_m0) * 8, gk_m0 / 2 & 1);
        if (gs_m0 > 0) {
          mbarrier_wait(s_free_addr, gs_m0 - 1 & 1);
        }
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (0) * 1024);
        int _mma_b_lo_0 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk0_m0) * 1024);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136314896;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
            "}\n" ::"r"(_mma_a_lo_0),
            "r"(_mma_b_lo_0), "r"(tmem_tmem_s0), "r"(0));
        elect_commit(s_full_addr);
        elect_commit(k_empty_addr + (sk0_m0) * 8);
        gk_m0 = gk_m0 + 1;
        gs_m0 = gs_m0 + 1;
#pragma unroll 1
        for (int jm0 = beg_m0; jm0 < nst_m0; jm0++) {
          if (nst_m0 > jm0 + 1) {
            int sk_m0 = gk_m0 % 2;
            mbarrier_wait(k_full_addr + (sk_m0) * 8, gk_m0 / 2 & 1);
            mbarrier_wait(s_free_addr, gs_m0 - 1 & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (0) * 1024);
            int _mma_b_lo_1 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk_m0) * 1024);
            asm volatile(
                "{\n\t"
                ".reg .pred leader, p0, p1;\n\t"
                ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                ".reg .b64 da, db;\n\t"
                "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                "setp.ne.b32 p0, %3, 0;\n\t"
                "setp.ne.b32 p1, 1, 0;\n\t"
                ""
                "mov.b32 adhi, 0x40004040;\n\t"
                "mov.b32 bdhi, 0x40004040;\n\t"
                "mov.b32 id, 136314896;\n\t"
                "mov.b32 alo, %0;\n\t"
                "mov.b32 blo, %1;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                "add.u32 alo, alo, 2;\n\t"
                "add.u32 blo, blo, 2;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                "add.u32 alo, alo, 2;\n\t"
                "add.u32 blo, blo, 2;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                "add.u32 alo, alo, 2;\n\t"
                "add.u32 blo, blo, 2;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                "}\n" ::"r"(_mma_a_lo_1),
                "r"(_mma_b_lo_1), "r"(tmem_tmem_s0), "r"(0));
            elect_commit(s_full_addr);
            elect_commit(k_empty_addr + (sk_m0) * 8);
            gk_m0 = gk_m0 + 1;
            gs_m0 = gs_m0 + 1;
          } else {
            elect_commit(tail_done_addr);
          }
          if (jm0 == beg_m0) {
            if (it_m0 > 0) {
              mbarrier_wait(o_free_addr, it_m0 - 1 & 1);
            }
          }
          int sv_m0 = gv_m0 % 4;
          int pb_m0 = gp_m0 & 1;
          int first_m0 = ((jm0 == beg_m0) ? 1 : 0);
          mbarrier_wait(v_full_addr + (sv_m0) * 8, gv_m0 / 4 & 1);
          mbarrier_wait(p_full_addr + (pb_m0) * 8, gp_m0 >> 1 & 1);
          asm volatile("tcgen05.fence::after_thread_sync;");
          int _mma_a_lo_2 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (pb_m0) * 1024);
          int _mma_b_lo_2 =
              make_warp_uniform(((((smem_v8_addr) >> 4) & 0x3FFF) | 0x4000000) + (sv_m0) * 1024);
          asm volatile(
              "{\n\t"
              ".reg .pred leader, p0, p1;\n\t"
              ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
              ".reg .b64 da, db;\n\t"
              "elect.sync _|leader, 0xFFFFFFFF;\n\t"
              "setp.ne.b32 p0, %3, 0;\n\t"
              "setp.ne.b32 p1, 1, 0;\n\t"
              ""
              "mov.b32 adhi, 0x40004040;\n\t"
              "mov.b32 bdhi, 0x40004040;\n\t"
              "mov.b32 id, 136380432;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
              "add.u32 alo, alo, 2;\n\t"
              "add.u32 blo, blo, 256;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
              "add.u32 alo, alo, 2;\n\t"
              "add.u32 blo, blo, 256;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
              "add.u32 alo, alo, 2;\n\t"
              "add.u32 blo, blo, 256;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
              "}\n" ::"r"(_mma_a_lo_2),
              "r"(_mma_b_lo_2), "r"(tmem_tmem_o0), "r"(((first_m0) ? 0 : 1)));
          elect_commit(v_empty_addr + (sv_m0) * 8);
          gv_m0 = gv_m0 + 1;
          gp_m0 = gp_m0 + 1;
        }
        elect_commit(o_full_addr);
        elect_commit(q_empty_addr);
      }
    }
    // ---- Role: load ----
  } else if (warp == 17) {
    {  // load_main
      int cta_l = blockIdx.x;
      int nit_l = qf;
      if (nrem > 0) {
        if (cta_l < nrem) {
          nit_l = qf + 1;
        } else if (cta_l - nrem < nrem) {
          nit_l = qf + (nrem - 1 - (cta_l - nrem)) / (nctas - nrem) + 1;
        }
      }
      int gk_l = 0;
      int gv_l = 0;
      if (elect_sync()) {
#pragma unroll 1
        for (int it_l = 0; it_l < nit_l; it_l++) {
          int tile_l = tile_base + cta_l + it_l * nctas;
          int kind_l = 0;
          if (it_l >= qf) {
            if (cta_l < nrem) {
              tile_l = tile_base + qf * nctas + cta_l;
              kind_l = 1;
            } else {
              tile_l = tile_base + qf * nctas + (cta_l - nrem) + (it_l - qf) * (nctas - nrem);
              kind_l = 2;
            }
          }
          int bz_l = tile_l / (32 * n_qtiles);
          int rem_l = tile_l % (32 * n_qtiles);
          int head_l = rem_l % 32;
          int q0_l = ((tile_base > 0) ? n_qtiles - 1 - rem_l / 32 : rem_l / 32) * 256;
          int qs_l = q_indptr[bz_l];
          int ql_l = q_indptr[bz_l + 1] - qs_l;
          int kvl_l = kv_lengths[bz_l];
          int _min_0 = ((q0_l + 255) < (ql_l - 1) ? (q0_l + 255) : (ql_l - 1));
          int last_l = _min_0;
          int _max_0 = ((kvl_l - ql_l + last_l) > (0) ? (kvl_l - ql_l + last_l) : (0));
          int lim_l = _max_0;
          int ntot_l = lim_l / 128 + 1;
          int _max_1 =
              (((int)(x_frac * (float)ntot_l)) > (1) ? ((int)(x_frac * (float)ntot_l)) : (1));
          int _min_1 = ((_max_1) < (ntot_l - 1) ? (_max_1) : (ntot_l - 1));
          int xs_l = _min_1;
          int beg_l = 0;
          int nst_l = ntot_l;
          if (kind_l == 1) {
            beg_l = xs_l;
          }
          if (kind_l == 2) {
            nst_l = xs_l > 0 ? xs_l : 1;
          }
          int kvh_l = head_l / 16;
          long long bt_l = (long long)bz_l * (long long)table_stride;
          if (it_l > 0) {
            mbarrier_wait(q_empty_addr, it_l - 1 & 1);
          }
          mbarrier_arrive_expect_tx(q_full_addr, 32768);
          tma_4d_gmem2smem(smem_q_addr, ((const void*)(tensor_maps_addr + 0)), 0, 0, head_l,
                           qs_l + q0_l, q_full_addr);
          tma_4d_gmem2smem(smem_q_addr + 16384, ((const void*)(tensor_maps_addr + 0)), 0, 0, head_l,
                           qs_l + q0_l + 128, q_full_addr);
#pragma unroll 1
          for (int jl = beg_l; jl < nst_l; jl++) {
            int page_l = int(bt_l + (long long)jl);
            int sk_l = gk_l % 2;
            mbarrier_wait(k_empty_addr + (sk_l) * 8, gk_l / 2 & 1 ^ 1);
            mbarrier_arrive_expect_tx(k_full_addr + (sk_l) * 8, 16384);
            tma_4d_gmem2smem(smem_k8_addr + (unsigned int)(sk_l * 16384),
                             ((const void*)(tensor_maps_addr + 128)), 0, 0, kvh_l, page_l,
                             k_full_addr + (sk_l) * 8);
            gk_l = gk_l + 1;
            int sv_l = gv_l % 4;
            mbarrier_wait(v_empty_addr + (sv_l) * 8, gv_l / 4 & 1 ^ 1);
            mbarrier_arrive_expect_tx(v_full_addr + (sv_l) * 8, 16384);
            tma_4d_gmem2smem(smem_v8_addr + (unsigned int)(sv_l * 16384),
                             ((const void*)(tensor_maps_addr + 256)), 0, 0, kvh_l, page_l,
                             v_full_addr + (sv_l) * 8);
            gv_l = gv_l + 1;
          }
        }
      }
    }
    // ---- Role: mma1 ----
  } else if (warp == 18) {
    {  // mma1_main
      int cta_m1 = blockIdx.x;
      int nit_m1 = qf;
      if (nrem > 0) {
        if (cta_m1 < nrem) {
          nit_m1 = qf + 1;
        } else if (cta_m1 - nrem < nrem) {
          nit_m1 = qf + (nrem - 1 - (cta_m1 - nrem)) / (nctas - nrem) + 1;
        }
      }
      int gk_m1 = 0;
      int gs_m1 = 0;
      int gv_m1 = 0;
      int gp_m1 = 0;
#pragma unroll 1
      for (int it_m1 = 0; it_m1 < nit_m1; it_m1++) {
        int tile_m1 = tile_base + cta_m1 + it_m1 * nctas;
        int kind_m1 = 0;
        if (it_m1 >= qf) {
          if (cta_m1 < nrem) {
            tile_m1 = tile_base + qf * nctas + cta_m1;
            kind_m1 = 1;
          } else {
            tile_m1 = tile_base + qf * nctas + (cta_m1 - nrem) + (it_m1 - qf) * (nctas - nrem);
            kind_m1 = 2;
          }
        }
        int bz_m1 = tile_m1 / (32 * n_qtiles);
        int rem_m1 = tile_m1 % (32 * n_qtiles);
        int head_m1 = rem_m1 % 32;
        int q0_m1 = ((tile_base > 0) ? n_qtiles - 1 - rem_m1 / 32 : rem_m1 / 32) * 256;
        int qs_m1 = q_indptr[bz_m1];
        int ql_m1 = q_indptr[bz_m1 + 1] - qs_m1;
        int kvl_m1 = kv_lengths[bz_m1];
        int _min_4 = ((q0_m1 + 255) < (ql_m1 - 1) ? (q0_m1 + 255) : (ql_m1 - 1));
        int last_m1 = _min_4;
        int _max_4 = ((kvl_m1 - ql_m1 + last_m1) > (0) ? (kvl_m1 - ql_m1 + last_m1) : (0));
        int lim_m1 = _max_4;
        int ntot_m1 = lim_m1 / 128 + 1;
        int _max_5 =
            (((int)(x_frac * (float)ntot_m1)) > (1) ? ((int)(x_frac * (float)ntot_m1)) : (1));
        int _min_5 = ((_max_5) < (ntot_m1 - 1) ? (_max_5) : (ntot_m1 - 1));
        int xs_m1 = _min_5;
        int beg_m1 = 0;
        int nst_m1 = ntot_m1;
        if (kind_m1 == 1) {
          beg_m1 = xs_m1;
        }
        if (kind_m1 == 2) {
          nst_m1 = xs_m1 > 0 ? xs_m1 : 1;
        }
        mbarrier_wait(q_full_addr, it_m1 & 1);
        int sk0_m1 = gk_m1 % 2;
        mbarrier_wait(k_full_addr + (sk0_m1) * 8, gk_m1 / 2 & 1);
        if (gs_m1 > 0) {
          mbarrier_wait(s_free_addr + 8, gs_m1 - 1 & 1);
        }
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (1) * 1024);
        int _mma_b_lo_3 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk0_m1) * 1024);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
            ".reg .b64 da, db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 adhi, 0x40004040;\n\t"
            "mov.b32 bdhi, 0x40004040;\n\t"
            "mov.b32 id, 136314896;\n\t"
            "mov.b32 alo, %0;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
            "add.u32 alo, alo, 2;\n\t"
            "add.u32 blo, blo, 2;\n\t"
            "mov.b64 da, {alo, adhi};\n\t"
            "mov.b64 db, {blo, bdhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
            "}\n" ::"r"(_mma_a_lo_3),
            "r"(_mma_b_lo_3), "r"(tmem_tmem_s1), "r"(0));
        elect_commit(s_full_addr + 8);
        elect_commit(k_empty_addr + (sk0_m1) * 8);
        gk_m1 = gk_m1 + 1;
        gs_m1 = gs_m1 + 1;
#pragma unroll 1
        for (int jm1 = beg_m1; jm1 < nst_m1; jm1++) {
          if (nst_m1 > jm1 + 1) {
            int sk_m1 = gk_m1 % 2;
            mbarrier_wait(k_full_addr + (sk_m1) * 8, gk_m1 / 2 & 1);
            mbarrier_wait(s_free_addr + 8, gs_m1 - 1 & 1);
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _mma_a_lo_4 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (1) * 1024);
            int _mma_b_lo_4 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk_m1) * 1024);
            asm volatile(
                "{\n\t"
                ".reg .pred leader, p0, p1;\n\t"
                ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                ".reg .b64 da, db;\n\t"
                "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                "setp.ne.b32 p0, %3, 0;\n\t"
                "setp.ne.b32 p1, 1, 0;\n\t"
                ""
                "mov.b32 adhi, 0x40004040;\n\t"
                "mov.b32 bdhi, 0x40004040;\n\t"
                "mov.b32 id, 136314896;\n\t"
                "mov.b32 alo, %0;\n\t"
                "mov.b32 blo, %1;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                "add.u32 alo, alo, 2;\n\t"
                "add.u32 blo, blo, 2;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                "add.u32 alo, alo, 2;\n\t"
                "add.u32 blo, blo, 2;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                "add.u32 alo, alo, 2;\n\t"
                "add.u32 blo, blo, 2;\n\t"
                "mov.b64 da, {alo, adhi};\n\t"
                "mov.b64 db, {blo, bdhi};\n\t"
                "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                "}\n" ::"r"(_mma_a_lo_4),
                "r"(_mma_b_lo_4), "r"(tmem_tmem_s1), "r"(0));
            elect_commit(s_full_addr + 8);
            elect_commit(k_empty_addr + (sk_m1) * 8);
            gk_m1 = gk_m1 + 1;
            gs_m1 = gs_m1 + 1;
          } else {
            elect_commit(tail_done_addr + 8);
          }
          if (jm1 == beg_m1) {
            if (it_m1 > 0) {
              mbarrier_wait(o_free_addr + 8, it_m1 - 1 & 1);
            }
          }
          int sv_m1 = gv_m1 % 4;
          int pb_m1 = gp_m1 & 1;
          int first_m1 = ((jm1 == beg_m1) ? 1 : 0);
          mbarrier_wait(v_full_addr + (sv_m1) * 8, gv_m1 / 4 & 1);
          mbarrier_wait(p_full_addr + (2 + pb_m1) * 8, gp_m1 >> 1 & 1);
          asm volatile("tcgen05.fence::after_thread_sync;");
          int _mma_a_lo_5 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) + (2 + pb_m1) * 1024);
          int _mma_b_lo_5 =
              make_warp_uniform(((((smem_v8_addr) >> 4) & 0x3FFF) | 0x4000000) + (sv_m1) * 1024);
          asm volatile(
              "{\n\t"
              ".reg .pred leader, p0, p1;\n\t"
              ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
              ".reg .b64 da, db;\n\t"
              "elect.sync _|leader, 0xFFFFFFFF;\n\t"
              "setp.ne.b32 p0, %3, 0;\n\t"
              "setp.ne.b32 p1, 1, 0;\n\t"
              ""
              "mov.b32 adhi, 0x40004040;\n\t"
              "mov.b32 bdhi, 0x40004040;\n\t"
              "mov.b32 id, 136380432;\n\t"
              "mov.b32 alo, %0;\n\t"
              "mov.b32 blo, %1;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
              "add.u32 alo, alo, 2;\n\t"
              "add.u32 blo, blo, 256;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
              "add.u32 alo, alo, 2;\n\t"
              "add.u32 blo, blo, 256;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
              "add.u32 alo, alo, 2;\n\t"
              "add.u32 blo, blo, 256;\n\t"
              "mov.b64 da, {alo, adhi};\n\t"
              "mov.b64 db, {blo, bdhi};\n\t"
              "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
              "}\n" ::"r"(_mma_a_lo_5),
              "r"(_mma_b_lo_5), "r"(tmem_tmem_o1), "r"(((first_m1) ? 0 : 1)));
          elect_commit(v_empty_addr + (sv_m1) * 8);
          gv_m1 = gv_m1 + 1;
          gp_m1 = gp_m1 + 1;
        }
        elect_commit(o_full_addr + 8);
        elect_commit(q_empty_addr);
      }
    }
    // ---- Role: spare ----
  } else if (warp == 19) {
    // idle — no tasks assigned
  }

  // Cleanup
  __syncthreads();  // barrier before TMEM dealloc

  if (warp == 0) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_addr_storage[0]),
        "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
}

}  // extern "C"

}  // namespace attn
}  // namespace prefill
}  // namespace cake_nvfp4
}  // namespace flashinfer
