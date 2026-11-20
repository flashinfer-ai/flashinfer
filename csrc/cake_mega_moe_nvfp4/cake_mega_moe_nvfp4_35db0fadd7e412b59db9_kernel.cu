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

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_mega_moe_nvfp4_35db0fadd7e412b59db9(unsigned int* __restrict__ x, unsigned int* __restrict__ data, uint8_t* __restrict__ sf, unsigned int* __restrict__ route_ids, unsigned int* __restrict__ route_scores, unsigned int* __restrict__ table_ids, unsigned int* __restrict__ table_scores, float norm, float mult)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int t = bid;
    if (tid < 6) {
        int slot = t * 6 + tid;
        unsigned int route[2];
        {
            route[0] = *reinterpret_cast<const unsigned int*>(route_ids + slot + 0);
        }
        {
            route[1] = *reinterpret_cast<const unsigned int*>(route_scores + slot + 0);
        }
        *(reinterpret_cast<unsigned int*>(table_ids + slot) + (0)) = route[0];
        *(reinterpret_cast<unsigned int*>(table_scores + slot) + (0)) = route[1];
    }
    int blk_0 = tid;
    unsigned int words_0[8];
    unsigned int half_0[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(x + (t * 3584 + blk_0 * 8) + 0);
        half_0[0 + 0] = _uv4_0.x;
        half_0[0 + 1] = _uv4_0.y;
        half_0[0 + 2] = _uv4_0.z;
        half_0[0 + 3] = _uv4_0.w;
    }
    words_0[0] = half_0[0];
    words_0[1] = half_0[1];
    words_0[2] = half_0[2];
    words_0[3] = half_0[3];
    unsigned int half_1[4];
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(x + (t * 3584 + blk_0 * 8 + 4) + 0);
        half_1[0 + 0] = _uv4_1.x;
        half_1[0 + 1] = _uv4_1.y;
        half_1[0 + 2] = _uv4_1.z;
        half_1[0 + 3] = _uv4_1.w;
    }
    words_0[4] = half_1[0];
    words_0[5] = half_1[1];
    words_0[6] = half_1[2];
    words_0[7] = half_1[3];
    float vals_0[16];
    #pragma unroll
    for (int _pair = 0; _pair < 8; _pair++) {
        (&vals_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) << 16);
        (&vals_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) & 0xffff0000u);
    }
    float amax_0 = vals_0[0];
    float _max_2;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(amax_0), "f"(vals_0[1]));
    amax_0 = _max_2;
    float _max_3;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(amax_0), "f"(vals_0[2]));
    amax_0 = _max_3;
    float _max_4;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(amax_0), "f"(vals_0[3]));
    amax_0 = _max_4;
    float _max_5;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(amax_0), "f"(vals_0[4]));
    amax_0 = _max_5;
    float _max_6;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(amax_0), "f"(vals_0[5]));
    amax_0 = _max_6;
    float _max_7;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(amax_0), "f"(vals_0[6]));
    amax_0 = _max_7;
    float _max_8;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(amax_0), "f"(vals_0[7]));
    amax_0 = _max_8;
    float _max_9;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(amax_0), "f"(vals_0[8]));
    amax_0 = _max_9;
    float _max_10;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(amax_0), "f"(vals_0[9]));
    amax_0 = _max_10;
    float _max_11;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(amax_0), "f"(vals_0[10]));
    amax_0 = _max_11;
    float _max_12;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(amax_0), "f"(vals_0[11]));
    amax_0 = _max_12;
    float _max_13;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(amax_0), "f"(vals_0[12]));
    amax_0 = _max_13;
    float _max_14;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(amax_0), "f"(vals_0[13]));
    amax_0 = _max_14;
    float _max_15;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_15) : "f"(amax_0), "f"(vals_0[14]));
    amax_0 = _max_15;
    float _max_16;
    asm("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(amax_0), "f"(vals_0[15]));
    amax_0 = _max_16;
    float amag_0 = __uint_as_float(__as_u32(amax_0) & 2147483647);
    float sfc_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(sfc_0) : "f"(amag_0), "f"(mult));
    uint16_t sf_pair_0;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sf_pair_0) : "f"(0.0f), "f"(sfc_0));
    unsigned int sf_byte_0 = (unsigned int)sf_pair_0 & 255;
    uint32_t sf_f16_0;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sf_f16_0) : "h"((uint16_t)((uint16_t)sf_byte_0)));
    float sf_rt_0;
    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rt_0) : "h"((uint16_t)(sf_f16_0)));
    float _rcp_17 = approx_rcp(sf_rt_0);
    float enc_0;
    asm("min.f32 %0, %1, %2;" : "=f"(enc_0) : "f"(mul_f32_rn_noftz(norm, _rcp_17)), "f"(3.4028235e+38f));
    float mask_0;
    asm("min.f32 %0, %1, %2;" : "=f"(mask_0) : "f"(mul_f32_rn_noftz(sf_rt_0, 1e+30f)), "f"(1.0f));
    float enc_m_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(enc_m_0) : "f"(enc_0), "f"(mask_0));
    vals_0[0] = mul_f32_rn_noftz(vals_0[0], enc_m_0);
    vals_0[1] = mul_f32_rn_noftz(vals_0[1], enc_m_0);
    vals_0[2] = mul_f32_rn_noftz(vals_0[2], enc_m_0);
    vals_0[3] = mul_f32_rn_noftz(vals_0[3], enc_m_0);
    vals_0[4] = mul_f32_rn_noftz(vals_0[4], enc_m_0);
    vals_0[5] = mul_f32_rn_noftz(vals_0[5], enc_m_0);
    vals_0[6] = mul_f32_rn_noftz(vals_0[6], enc_m_0);
    vals_0[7] = mul_f32_rn_noftz(vals_0[7], enc_m_0);
    vals_0[8] = mul_f32_rn_noftz(vals_0[8], enc_m_0);
    vals_0[9] = mul_f32_rn_noftz(vals_0[9], enc_m_0);
    vals_0[10] = mul_f32_rn_noftz(vals_0[10], enc_m_0);
    vals_0[11] = mul_f32_rn_noftz(vals_0[11], enc_m_0);
    vals_0[12] = mul_f32_rn_noftz(vals_0[12], enc_m_0);
    vals_0[13] = mul_f32_rn_noftz(vals_0[13], enc_m_0);
    vals_0[14] = mul_f32_rn_noftz(vals_0[14], enc_m_0);
    vals_0[15] = mul_f32_rn_noftz(vals_0[15], enc_m_0);
    unsigned int packed_0[2];
    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_0[0]) : "f"(vals_0[0]), "f"(vals_0[1]), "f"(vals_0[2]), "f"(vals_0[3]), "f"(vals_0[4]), "f"(vals_0[5]), "f"(vals_0[6]), "f"(vals_0[7]));
    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_0[1]) : "f"(vals_0[8]), "f"(vals_0[9]), "f"(vals_0[10]), "f"(vals_0[11]), "f"(vals_0[12]), "f"(vals_0[13]), "f"(vals_0[14]), "f"(vals_0[15]));
    reinterpret_cast<int2*>(data + (t * 896 + blk_0 * 2))[0] = reinterpret_cast<int2*>(packed_0)[0];
    *(reinterpret_cast<unsigned char*>(sf + (t * 448 + blk_0)) + (0)) = (unsigned char)(sf_byte_0);
}

} // extern "C"
