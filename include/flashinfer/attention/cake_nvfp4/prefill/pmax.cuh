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
// Row maxima of the split tiles: the QK pass alone, over the KV range of each tile in
// `pchunks` pieces, for the split attention kernel (attn.cuh) to normalize against.
#include "common.cuh"

namespace flashinfer {
namespace cake_nvfp4 {
namespace prefill {
namespace pmax {

constexpr int kThreads = 576;
// Dynamic shared memory: the barriers and the TMEM address in the first 1024 bytes, then the
// stages of Q and K, 16384 bytes each.
constexpr int kSmemQOffset = 1024;
constexpr int kSmemKOffset = 33792;
constexpr int kSharedBytes = 197632;

extern "C" {

__global__ __launch_bounds__(kThreads, 1) void cake_nvfp4_prefill_pmax(
    float* __restrict__ pmax, int* __restrict__ block_tables, int* __restrict__ q_indptr,
    int* __restrict__ kv_lengths, int table_stride, int n_qtiles, int nctas, int qf, int tile_base,
    float x_frac, int pchunks, __grid_constant__ TensorMaps<2> const tensor_maps) {
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

  // Mbarrier init (5 groups, 29 barriers)
  // Mbarriers at smem_raw[0..232)

  if (warp == 0) {
    uint32_t leader = elect_sync();
    // q_full: 1 barriers, init_count=1
    mbarrier_init_pred(smem + 0, 1, leader);
    // k_full: 10 barriers, init_count=1
    mbarrier_init_pred(smem + 8, 1, leader);
    mbarrier_init_pred(smem + 16, 1, leader);
    mbarrier_init_pred(smem + 24, 1, leader);
    mbarrier_init_pred(smem + 32, 1, leader);
    mbarrier_init_pred(smem + 40, 1, leader);
    mbarrier_init_pred(smem + 48, 1, leader);
    mbarrier_init_pred(smem + 56, 1, leader);
    mbarrier_init_pred(smem + 64, 1, leader);
    mbarrier_init_pred(smem + 72, 1, leader);
    mbarrier_init_pred(smem + 80, 1, leader);
    // k_empty: 10 barriers, init_count=1
    mbarrier_init_pred(smem + 88, 1, leader);
    mbarrier_init_pred(smem + 96, 1, leader);
    mbarrier_init_pred(smem + 104, 1, leader);
    mbarrier_init_pred(smem + 112, 1, leader);
    mbarrier_init_pred(smem + 120, 1, leader);
    mbarrier_init_pred(smem + 128, 1, leader);
    mbarrier_init_pred(smem + 136, 1, leader);
    mbarrier_init_pred(smem + 144, 1, leader);
    mbarrier_init_pred(smem + 152, 1, leader);
    mbarrier_init_pred(smem + 160, 1, leader);
    // s_full: 4 barriers, init_count=1
    mbarrier_init_pred(smem + 168, 1, leader);
    mbarrier_init_pred(smem + 176, 1, leader);
    mbarrier_init_pred(smem + 184, 1, leader);
    mbarrier_init_pred(smem + 192, 1, leader);
    // s_free: 4 barriers, init_count=256
    mbarrier_init_pred(smem + 200, 256, leader);
    mbarrier_init_pred(smem + 208, 256, leader);
    mbarrier_init_pred(smem + 216, 256, leader);
    mbarrier_init_pred(smem + 224, 256, leader);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }

  __syncthreads();

  // TMEM alloc (512 columns, 512 used)
  volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
  if (warp == 0) {
    int _tmem_hold = smem + 232;
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(_tmem_hold),
        "r"(512)
        : "memory");
  }

  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");

  const int mbar_base = smem;
  const int q_full_addr = mbar_base + 0;
  const int k_full_addr = mbar_base + 8;
  const int k_empty_addr = mbar_base + 88;
  const int s_full_addr = mbar_base + 168;
  const int s_free_addr = mbar_base + 200;
  const int taddr = tmem_addr_storage[0];

  // TMEM columns
  const int tmem_tmem_s00 = taddr;
  const int tmem_tmem_s10 = taddr + 128;
  const int tmem_tmem_s01 = taddr + 256;
  const int tmem_tmem_s11 = taddr + 384;

  // ---- Role: soft ----
  if (warp <= 15) {
    {  // soft_main
      int w_s = blockIdx.x;
      int r_s = w_s / pchunks;
      int c_s = w_s % pchunks;
      int tile_s = tile_base + qf * nctas + r_s;
      int bz_s = tile_s / (32 * n_qtiles);
      int rem_s = tile_s % (32 * n_qtiles);
      int q0_s = ((tile_base > 0) ? n_qtiles - 1 - rem_s / 32 : rem_s / 32) * 256;
      int ql_s = q_indptr[bz_s + 1] - q_indptr[bz_s];
      int kvl_s = kv_lengths[bz_s];
      int _min_4 = ((q0_s + 255) < (ql_s - 1) ? (q0_s + 255) : (ql_s - 1));
      int last_s = _min_4;
      int _max_4 = ((kvl_s - ql_s + last_s) > (0) ? (kvl_s - ql_s + last_s) : (0));
      int lim_s = _max_4;
      int ntot_s = lim_s / 128 + 1;
      int _max_5 = (((int)(x_frac * (float)ntot_s)) > (1) ? ((int)(x_frac * (float)ntot_s)) : (1));
      int _min_5 = ((_max_5) < (ntot_s - 1) ? (_max_5) : (ntot_s - 1));
      int xs_s = _min_5;
      int c0_s = xs_s * c_s / pchunks;
      int cnt_s = xs_s * (c_s + 1) / pchunks - c0_s;
      int qi = warp / 8;
      int hf = warp / 4 & 1;
      int lg = warp % 4;
      int uu = lane & 3;
      int lbase = lg * 32 + hf * 16;
      int ra = lbase + (lane >> 2);
      int qrow_a = q0_s + qi * 128 + ra;
      int lim_a = kvl_s - ql_s + qrow_a;
      int lim_b = lim_a + 8;
      int s_addr = taddr + (unsigned int)(qi * 128) + (unsigned int)(lbase << 16);
      float m_a = -3.4028234663852886e+38f;
      float m_b = -3.4028234663852886e+38f;
#pragma unroll 1
      for (int js = 0; js < cnt_s; js++) {
        int bf_s = js & 1;
        mbarrier_wait(s_full_addr + (qi * 2 + bf_s) * 8, js >> 1 & 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        float _tmem_load_0[64];
        asm volatile(
            "tcgen05.ld.sync.aligned.16x256b.x16.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, "
            "%18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, "
            "%35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, "
            "%52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
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
            : "r"(s_addr + bf_s * 256)
            : "memory");
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        mbarrier_arrive(s_free_addr + (qi * 2 + bf_s) * 8);
        int kvbase = (c0_s + js) * 128 + uu * 32;
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
                       : "f"(xa[g]), "f"(_tmem_load_0[16 * g + 5]), "f"(_tmem_load_0[16 * g + 8]));
          xa[g] = _max3_1;
          float _max3_2;
          asm volatile("max.f32 %0, %1, %2, %3;"
                       : "=f"(_max3_2)
                       : "f"(xa[g]), "f"(_tmem_load_0[16 * g + 9]), "f"(_tmem_load_0[16 * g + 12]));
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
                       : "f"(xb[g]), "f"(_tmem_load_0[16 * g + 7]), "f"(_tmem_load_0[16 * g + 10]));
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
        float ta = _max3_6;
        float _max3_7;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_7)
                     : "f"(ta), "f"(xa[3]), "f"(_tmem_load_0[13]));
        ta = _max3_7;
        float _max3_8;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_8)
                     : "f"(ta), "f"(_tmem_load_0[29]), "f"(_tmem_load_0[45]));
        ta = _max3_8;
        float _max3_9;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_9)
                     : "f"(m_a), "f"(ta), "f"(_tmem_load_0[61]));
        m_a = _max3_9;
        float _max3_10;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_10)
                     : "f"(xb[0]), "f"(xb[1]), "f"(xb[2]));
        float tb = _max3_10;
        float _max3_11;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_11)
                     : "f"(tb), "f"(xb[3]), "f"(_tmem_load_0[15]));
        tb = _max3_11;
        float _max3_12;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_12)
                     : "f"(tb), "f"(_tmem_load_0[31]), "f"(_tmem_load_0[47]));
        tb = _max3_12;
        float _max3_13;
        asm volatile("max.f32 %0, %1, %2, %3;"
                     : "=f"(_max3_13)
                     : "f"(m_b), "f"(tb), "f"(_tmem_load_0[63]));
        m_b = _max3_13;
      }
      float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, m_a, 1);
      float _max_6 = max_noftz(m_a, _shfl_xor_0);
      m_a = _max_6;
      float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, m_b, 1);
      float _max_7 = max_noftz(m_b, _shfl_xor_1);
      m_b = _max_7;
      float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, m_a, 2);
      float _max_8 = max_noftz(m_a, _shfl_xor_2);
      m_a = _max_8;
      float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, m_b, 2);
      float _max_9 = max_noftz(m_b, _shfl_xor_3);
      m_b = _max_9;
      if (uu == 0) {
        long long prow = (long long)w_s * 256 + (long long)(qi * 128) + (long long)ra;
        pmax[prow] = m_a;
        pmax[prow + 8] = m_b;
      }
    }
    // ---- Role: mma ----
  } else if (warp == 16) {
    {  // mma_main
      int w_m = blockIdx.x;
      int r_m = w_m / pchunks;
      int c_m = w_m % pchunks;
      int tile_m = tile_base + qf * nctas + r_m;
      int bz_m = tile_m / (32 * n_qtiles);
      int rem_m = tile_m % (32 * n_qtiles);
      int q0_m = ((tile_base > 0) ? n_qtiles - 1 - rem_m / 32 : rem_m / 32) * 256;
      int ql_m = q_indptr[bz_m + 1] - q_indptr[bz_m];
      int kvl_m = kv_lengths[bz_m];
      int _min_2 = ((q0_m + 255) < (ql_m - 1) ? (q0_m + 255) : (ql_m - 1));
      int last_m = _min_2;
      int _max_2 = ((kvl_m - ql_m + last_m) > (0) ? (kvl_m - ql_m + last_m) : (0));
      int lim_m = _max_2;
      int ntot_m = lim_m / 128 + 1;
      int _max_3 = (((int)(x_frac * (float)ntot_m)) > (1) ? ((int)(x_frac * (float)ntot_m)) : (1));
      int _min_3 = ((_max_3) < (ntot_m - 1) ? (_max_3) : (ntot_m - 1));
      int xs_m = _min_3;
      int c0_m = xs_m * c_m / pchunks;
      int cnt_m = xs_m * (c_m + 1) / pchunks - c0_m;
      mbarrier_wait(q_full_addr, 0);
#pragma unroll 1
      for (int jm = 0; jm < cnt_m; jm++) {
        int sk_m = jm % 10;
        int bf_m = jm & 1;
        mbarrier_wait(k_full_addr + (sk_m) * 8, jm / 10 & 1);
        if (jm >= 2) {
          mbarrier_wait(s_free_addr + (bf_m) * 8, (jm >> 1) - 1 & 1);
          mbarrier_wait(s_free_addr + (2 + bf_m) * 8, (jm >> 1) - 1 & 1);
        }
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (bf_m == 0) {
          int _mma_a_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (0) * 1024);
          int _mma_b_lo_0 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk_m) * 1024);
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
              "r"(_mma_b_lo_0), "r"(tmem_tmem_s00), "r"(0));
          elect_commit(s_full_addr);
          int _mma_a_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (1) * 1024);
          int _mma_b_lo_1 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk_m) * 1024);
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
              "r"(_mma_b_lo_1), "r"(tmem_tmem_s10), "r"(0));
          elect_commit(s_full_addr + 16);
        } else {
          int _mma_a_lo_2 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (0) * 1024);
          int _mma_b_lo_2 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk_m) * 1024);
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
              "}\n" ::"r"(_mma_a_lo_2),
              "r"(_mma_b_lo_2), "r"(tmem_tmem_s01), "r"(0));
          elect_commit(s_full_addr + 8);
          int _mma_a_lo_3 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (1) * 1024);
          int _mma_b_lo_3 = make_warp_uniform((((smem_k8_addr) >> 4) & 0x3FFF) + (sk_m) * 1024);
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
              "r"(_mma_b_lo_3), "r"(tmem_tmem_s11), "r"(0));
          elect_commit(s_full_addr + 24);
        }
        elect_commit(k_empty_addr + (sk_m) * 8);
      }
    }
    // ---- Role: load ----
  } else if (warp == 17) {
    {  // load_main
      int w_l = blockIdx.x;
      int r_l = w_l / pchunks;
      int c_l = w_l % pchunks;
      int tile_l = tile_base + qf * nctas + r_l;
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
      int _max_1 = (((int)(x_frac * (float)ntot_l)) > (1) ? ((int)(x_frac * (float)ntot_l)) : (1));
      int _min_1 = ((_max_1) < (ntot_l - 1) ? (_max_1) : (ntot_l - 1));
      int xs_l = _min_1;
      int c0_l = xs_l * c_l / pchunks;
      int cnt_l = xs_l * (c_l + 1) / pchunks - c0_l;
      int kvh_l = head_l / 16;
      long long bt_l = (long long)bz_l * (long long)table_stride + (long long)c0_l;
      if (elect_sync()) {
        mbarrier_arrive_expect_tx(q_full_addr, 32768);
        tma_4d_gmem2smem(smem_q_addr, ((const void*)(tensor_maps_addr + 0)), 0, 0, head_l,
                         qs_l + q0_l, q_full_addr);
        tma_4d_gmem2smem(smem_q_addr + 16384, ((const void*)(tensor_maps_addr + 0)), 0, 0, head_l,
                         qs_l + q0_l + 128, q_full_addr);
#pragma unroll 1
        for (int jl = 0; jl < cnt_l; jl++) {
          int page_l = int(bt_l + (long long)jl);
          int sk_l = jl % 10;
          mbarrier_wait(k_empty_addr + (sk_l) * 8, jl / 10 & 1 ^ 1);
          mbarrier_arrive_expect_tx(k_full_addr + (sk_l) * 8, 16384);
          tma_4d_gmem2smem(smem_k8_addr + (unsigned int)(sk_l * 16384),
                           ((const void*)(tensor_maps_addr + 128)), 0, 0, kvh_l, page_l,
                           k_full_addr + (sk_l) * 8);
        }
      }
    }
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

}  // namespace pmax
}  // namespace prefill
}  // namespace cake_nvfp4
}  // namespace flashinfer
