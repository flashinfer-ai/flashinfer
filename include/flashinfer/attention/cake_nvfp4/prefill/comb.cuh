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
// Merges the two partial results of every split tile (256 query rows of one head) into the
// FP8 output: the partial O and (max, sum) that attn.cuh left in the workspace.
#include "common.cuh"

namespace flashinfer {
namespace cake_nvfp4 {
namespace prefill {
namespace comb {

constexpr int kThreads = 256;

extern "C" {

__global__ __launch_bounds__(kThreads, 2) void cake_nvfp4_prefill_comb(
    uint8_t* __restrict__ out, float* __restrict__ part_o, float* __restrict__ part_ml,
    int* __restrict__ q_indptr, int n_qtiles, int nctas, int qf, int tile_base, float scale_log2,
    float out_scale, const float* scale_log2_ptr, const float* out_scale_ptr) {
  if (scale_log2_ptr) scale_log2 = *scale_log2_ptr;
  if (out_scale_ptr) out_scale = *out_scale_ptr;
  const int tid = threadIdx.x;
  const int warp = broadcast_lane0(tid / 32);
  const int lane = tid % 32;

  int r = blockIdx.x;
  int row = tid;
  int tile = tile_base + qf * nctas + r;
  int bz = tile / (32 * n_qtiles);
  int rem = tile % (32 * n_qtiles);
  int head = rem % 32;
  int q0 = ((tile_base > 0) ? n_qtiles - 1 - rem / 32 : rem / 32) * 256;
  int qs = q_indptr[bz];
  int ql = q_indptr[bz + 1] - qs;
  long long ph = (long long)(2 * r) * 256 + (long long)row;
  long long pt = ph + 256;
  float mh = part_ml[ph * 2];
  float lh = part_ml[ph * 2 + 1];
  float mt = part_ml[pt * 2];
  float lt = part_ml[pt * 2 + 1];
  float _max_0 = max_noftz(mh, mt);
  float mm = _max_0;
  float _exp2_0 = approx_exp2((mh - mm) * scale_log2);
  float eh = _exp2_0;
  float _exp2_1 = approx_exp2((mt - mm) * scale_log2);
  float et = _exp2_1;
  float ll = lh * eh + lt * et;
  float inv = ((ll > 0.0f) ? out_scale / ll : 0.0f);
  float ch = eh * inv;
  float ct = et * inv;
  int qrow = q0 + row;
  if (qrow < ql) {
    long long obase = ((long long)(qs + qrow) * 32 + (long long)head) * 128;
#pragma unroll
    for (int c = 0; c < 8; c++) {
      float va[16];
      float vb[16];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        {
          float4 _v4 = *reinterpret_cast<const float4*>(part_o + ph * 128 + (long long)(c * 16) +
                                                        (long long)(i * 4));
          va[i * 4 + 0] = _v4.x;
          va[i * 4 + 1] = _v4.y;
          va[i * 4 + 2] = _v4.z;
          va[i * 4 + 3] = _v4.w;
        }
        {
          float4 _v4 = *reinterpret_cast<const float4*>(part_o + pt * 128 + (long long)(c * 16) +
                                                        (long long)(i * 4));
          vb[i * 4 + 0] = _v4.x;
          vb[i * 4 + 1] = _v4.y;
          vb[i * 4 + 2] = _v4.z;
          vb[i * 4 + 3] = _v4.w;
        }
      }
      float vo[16];
#pragma unroll
      for (int i_1 = 0; i_1 < 16; i_1++) {
        vo[i_1] = va[i_1] * ch + vb[i_1] * ct;
      }
      unsigned int pk[4];
      {
        unsigned short _lo, _hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(vo[1]), "f"(vo[0]));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(vo[3]), "f"(vo[2]));
        pk[0] = (unsigned)_lo | ((unsigned)_hi << 16);
      }
      {
        unsigned short _lo, _hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(vo[5]), "f"(vo[4]));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(vo[7]), "f"(vo[6]));
        pk[1] = (unsigned)_lo | ((unsigned)_hi << 16);
      }
      {
        unsigned short _lo, _hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(vo[9]), "f"(vo[8]));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(vo[11]), "f"(vo[10]));
        pk[2] = (unsigned)_lo | ((unsigned)_hi << 16);
      }
      {
        unsigned short _lo, _hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(vo[13]), "f"(vo[12]));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(vo[15]), "f"(vo[14]));
        pk[3] = (unsigned)_lo | ((unsigned)_hi << 16);
      }
      reinterpret_cast<int4*>(out + (obase + (long long)(c * 16)))[0] =
          reinterpret_cast<int4*>(pk + 0)[0];
    }
  }
}

}  // extern "C"

}  // namespace comb
}  // namespace prefill
}  // namespace cake_nvfp4
}  // namespace flashinfer
