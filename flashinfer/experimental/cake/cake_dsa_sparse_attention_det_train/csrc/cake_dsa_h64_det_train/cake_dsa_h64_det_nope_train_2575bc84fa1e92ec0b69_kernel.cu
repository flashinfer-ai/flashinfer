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

#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>


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

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_det_nope_train_2575bc84fa1e92ec0b69(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ delta, __nv_bfloat16* __restrict__ q_latent, float* __restrict__ qstat, unsigned int* __restrict__ det_globals, int num_tokens, int dout_token_stride, int dout_head_stride, int dout_offset, int q_token_stride, int q_head_stride, int q_offset, int qstat_stride, float sm_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* part = reinterpret_cast<float*>(smem_raw + 0);
    const int part_addr = smem + 0;

    // === Task calls (declaration order) ===
    int token = blockIdx.x;
    int warp_1 = warp;
    int lane_1 = lane;
    if (token == 0 && tid < 16) {
        det_globals[tid] = 0;
    }
    if (token < num_tokens) {
        float a_lat = 0.0f;
        float b_lat = 0.0f;
        float a_rope = 0.0f;
        float b_rope = 0.0f;
        #pragma unroll 1
        for (int i = 0; i < 8; i++) {
            int head = i * 8 + warp_1;
            int row = token * 64 + head;
            long long base = (long long)row * 512ll + (long long)(lane_1 * 16);
            long long gbase = (long long)dout_offset + (long long)token * (long long)dout_token_stride + (long long)head * (long long)dout_head_stride + (long long)(lane_1 * 16);
            long long qbase = (long long)q_offset + (long long)token * (long long)q_token_stride + (long long)head * (long long)q_head_stride + (long long)(lane_1 * 16);
            float acc = 0.0f;
            float sq = 0.0f;
            float dmax = 0.0f;
            float qmax = 0.0f;
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(dout + gbase + 0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_6_0[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(out + base + 0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_6_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                        (&_vec_load_6_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_7_0[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(o_lo + base + 0);
                uint4 _vld_2[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_2[_blk] = _vptr_2[_blk];
                    uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_7_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                        (&_vec_load_7_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                    }
                }
            }
            float q8_0[8];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(q_latent + qbase + 0);
                uint4 _vld_3[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&q8_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                        (&q8_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                    }
                }
            }
            acc = __fmaf_rn(_vec_load_0[0], _vec_load_6_0[0] + _vec_load_7_0[0], acc);
            acc = __fmaf_rn(_vec_load_0[1], _vec_load_6_0[1] + _vec_load_7_0[1], acc);
            acc = __fmaf_rn(_vec_load_0[2], _vec_load_6_0[2] + _vec_load_7_0[2], acc);
            acc = __fmaf_rn(_vec_load_0[3], _vec_load_6_0[3] + _vec_load_7_0[3], acc);
            acc = __fmaf_rn(_vec_load_0[4], _vec_load_6_0[4] + _vec_load_7_0[4], acc);
            acc = __fmaf_rn(_vec_load_0[5], _vec_load_6_0[5] + _vec_load_7_0[5], acc);
            acc = __fmaf_rn(_vec_load_0[6], _vec_load_6_0[6] + _vec_load_7_0[6], acc);
            acc = __fmaf_rn(_vec_load_0[7], _vec_load_6_0[7] + _vec_load_7_0[7], acc);
            sq = __fmaf_rn(_vec_load_0[0], _vec_load_0[0], sq);
            float _abs_5;
            asm("abs.f32 %0, %1;" : "=f"(_abs_5) : "f"(_vec_load_0[0]));
            float _max_4;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(dmax), "f"(_abs_5));
            dmax = _max_4;
            float _abs_7;
            asm("abs.f32 %0, %1;" : "=f"(_abs_7) : "f"(q8_0[0]));
            float _max_6;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(qmax), "f"(_abs_7));
            qmax = _max_6;
            sq = __fmaf_rn(_vec_load_0[1], _vec_load_0[1], sq);
            float _abs_9;
            asm("abs.f32 %0, %1;" : "=f"(_abs_9) : "f"(_vec_load_0[1]));
            float _max_8;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(dmax), "f"(_abs_9));
            dmax = _max_8;
            float _abs_11;
            asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(q8_0[1]));
            float _max_10;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(qmax), "f"(_abs_11));
            qmax = _max_10;
            sq = __fmaf_rn(_vec_load_0[2], _vec_load_0[2], sq);
            float _abs_13;
            asm("abs.f32 %0, %1;" : "=f"(_abs_13) : "f"(_vec_load_0[2]));
            float _max_12;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(dmax), "f"(_abs_13));
            dmax = _max_12;
            float _abs_15;
            asm("abs.f32 %0, %1;" : "=f"(_abs_15) : "f"(q8_0[2]));
            float _max_14;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(qmax), "f"(_abs_15));
            qmax = _max_14;
            sq = __fmaf_rn(_vec_load_0[3], _vec_load_0[3], sq);
            float _abs_17;
            asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(_vec_load_0[3]));
            float _max_16;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(dmax), "f"(_abs_17));
            dmax = _max_16;
            float _abs_19;
            asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(q8_0[3]));
            float _max_18;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_18) : "f"(qmax), "f"(_abs_19));
            qmax = _max_18;
            sq = __fmaf_rn(_vec_load_0[4], _vec_load_0[4], sq);
            float _abs_21;
            asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(_vec_load_0[4]));
            float _max_20;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_20) : "f"(dmax), "f"(_abs_21));
            dmax = _max_20;
            float _abs_23;
            asm("abs.f32 %0, %1;" : "=f"(_abs_23) : "f"(q8_0[4]));
            float _max_22;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_22) : "f"(qmax), "f"(_abs_23));
            qmax = _max_22;
            sq = __fmaf_rn(_vec_load_0[5], _vec_load_0[5], sq);
            float _abs_25;
            asm("abs.f32 %0, %1;" : "=f"(_abs_25) : "f"(_vec_load_0[5]));
            float _max_24;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_24) : "f"(dmax), "f"(_abs_25));
            dmax = _max_24;
            float _abs_27;
            asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(q8_0[5]));
            float _max_26;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_26) : "f"(qmax), "f"(_abs_27));
            qmax = _max_26;
            sq = __fmaf_rn(_vec_load_0[6], _vec_load_0[6], sq);
            float _abs_29;
            asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(_vec_load_0[6]));
            float _max_28;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(dmax), "f"(_abs_29));
            dmax = _max_28;
            float _abs_31;
            asm("abs.f32 %0, %1;" : "=f"(_abs_31) : "f"(q8_0[6]));
            float _max_30;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(qmax), "f"(_abs_31));
            qmax = _max_30;
            sq = __fmaf_rn(_vec_load_0[7], _vec_load_0[7], sq);
            float _abs_33;
            asm("abs.f32 %0, %1;" : "=f"(_abs_33) : "f"(_vec_load_0[7]));
            float _max_32;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_32) : "f"(dmax), "f"(_abs_33));
            dmax = _max_32;
            float _abs_35;
            asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(q8_0[7]));
            float _max_34;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(qmax), "f"(_abs_35));
            qmax = _max_34;
            float _vec_load_1[8];
            {
                const uint4* _vptr_36 = reinterpret_cast<const uint4*>(dout + (gbase + 8ll) + 0);
                uint4 _vld_36[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_36[_blk] = _vptr_36[_blk];
                    uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                        (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_6_1[8];
            {
                const uint4* _vptr_37 = reinterpret_cast<const uint4*>(out + (base + 8ll) + 0);
                uint4 _vld_37[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_37[_blk] = _vptr_37[_blk];
                    uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_6_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) << 16);
                        (&_vec_load_6_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_37[_pair]) & 0xffff0000u);
                    }
                }
            }
            float _vec_load_7_1[8];
            {
                const uint4* _vptr_38 = reinterpret_cast<const uint4*>(o_lo + (base + 8ll) + 0);
                uint4 _vld_38[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_38[_blk] = _vptr_38[_blk];
                    uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_vec_load_7_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                        (&_vec_load_7_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                    }
                }
            }
            float q8_1[8];
            {
                const uint4* _vptr_39 = reinterpret_cast<const uint4*>(q_latent + (qbase + 8ll) + 0);
                uint4 _vld_39[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_39[_blk] = _vptr_39[_blk];
                    uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&q8_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) << 16);
                        (&q8_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_39[_pair]) & 0xffff0000u);
                    }
                }
            }
            acc = __fmaf_rn(_vec_load_1[0], _vec_load_6_1[0] + _vec_load_7_1[0], acc);
            acc = __fmaf_rn(_vec_load_1[1], _vec_load_6_1[1] + _vec_load_7_1[1], acc);
            acc = __fmaf_rn(_vec_load_1[2], _vec_load_6_1[2] + _vec_load_7_1[2], acc);
            acc = __fmaf_rn(_vec_load_1[3], _vec_load_6_1[3] + _vec_load_7_1[3], acc);
            acc = __fmaf_rn(_vec_load_1[4], _vec_load_6_1[4] + _vec_load_7_1[4], acc);
            acc = __fmaf_rn(_vec_load_1[5], _vec_load_6_1[5] + _vec_load_7_1[5], acc);
            acc = __fmaf_rn(_vec_load_1[6], _vec_load_6_1[6] + _vec_load_7_1[6], acc);
            acc = __fmaf_rn(_vec_load_1[7], _vec_load_6_1[7] + _vec_load_7_1[7], acc);
            sq = __fmaf_rn(_vec_load_1[0], _vec_load_1[0], sq);
            float _abs_41;
            asm("abs.f32 %0, %1;" : "=f"(_abs_41) : "f"(_vec_load_1[0]));
            float _max_40;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(dmax), "f"(_abs_41));
            dmax = _max_40;
            float _abs_43;
            asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(q8_1[0]));
            float _max_42;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_42) : "f"(qmax), "f"(_abs_43));
            qmax = _max_42;
            sq = __fmaf_rn(_vec_load_1[1], _vec_load_1[1], sq);
            float _abs_45;
            asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(_vec_load_1[1]));
            float _max_44;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(dmax), "f"(_abs_45));
            dmax = _max_44;
            float _abs_47;
            asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(q8_1[1]));
            float _max_46;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_46) : "f"(qmax), "f"(_abs_47));
            qmax = _max_46;
            sq = __fmaf_rn(_vec_load_1[2], _vec_load_1[2], sq);
            float _abs_49;
            asm("abs.f32 %0, %1;" : "=f"(_abs_49) : "f"(_vec_load_1[2]));
            float _max_48;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_48) : "f"(dmax), "f"(_abs_49));
            dmax = _max_48;
            float _abs_51;
            asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(q8_1[2]));
            float _max_50;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_50) : "f"(qmax), "f"(_abs_51));
            qmax = _max_50;
            sq = __fmaf_rn(_vec_load_1[3], _vec_load_1[3], sq);
            float _abs_53;
            asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(_vec_load_1[3]));
            float _max_52;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(dmax), "f"(_abs_53));
            dmax = _max_52;
            float _abs_55;
            asm("abs.f32 %0, %1;" : "=f"(_abs_55) : "f"(q8_1[3]));
            float _max_54;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_54) : "f"(qmax), "f"(_abs_55));
            qmax = _max_54;
            sq = __fmaf_rn(_vec_load_1[4], _vec_load_1[4], sq);
            float _abs_57;
            asm("abs.f32 %0, %1;" : "=f"(_abs_57) : "f"(_vec_load_1[4]));
            float _max_56;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_56) : "f"(dmax), "f"(_abs_57));
            dmax = _max_56;
            float _abs_59;
            asm("abs.f32 %0, %1;" : "=f"(_abs_59) : "f"(q8_1[4]));
            float _max_58;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(qmax), "f"(_abs_59));
            qmax = _max_58;
            sq = __fmaf_rn(_vec_load_1[5], _vec_load_1[5], sq);
            float _abs_61;
            asm("abs.f32 %0, %1;" : "=f"(_abs_61) : "f"(_vec_load_1[5]));
            float _max_60;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_60) : "f"(dmax), "f"(_abs_61));
            dmax = _max_60;
            float _abs_63;
            asm("abs.f32 %0, %1;" : "=f"(_abs_63) : "f"(q8_1[5]));
            float _max_62;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_62) : "f"(qmax), "f"(_abs_63));
            qmax = _max_62;
            sq = __fmaf_rn(_vec_load_1[6], _vec_load_1[6], sq);
            float _abs_65;
            asm("abs.f32 %0, %1;" : "=f"(_abs_65) : "f"(_vec_load_1[6]));
            float _max_64;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_64) : "f"(dmax), "f"(_abs_65));
            dmax = _max_64;
            float _abs_67;
            asm("abs.f32 %0, %1;" : "=f"(_abs_67) : "f"(q8_1[6]));
            float _max_66;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_66) : "f"(qmax), "f"(_abs_67));
            qmax = _max_66;
            sq = __fmaf_rn(_vec_load_1[7], _vec_load_1[7], sq);
            float _abs_69;
            asm("abs.f32 %0, %1;" : "=f"(_abs_69) : "f"(_vec_load_1[7]));
            float _max_68;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_68) : "f"(dmax), "f"(_abs_69));
            dmax = _max_68;
            float _abs_71;
            asm("abs.f32 %0, %1;" : "=f"(_abs_71) : "f"(q8_1[7]));
            float _max_70;
            asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_70) : "f"(qmax), "f"(_abs_71));
            qmax = _max_70;
            float total = acc;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                total += __shfl_xor_sync(0xFFFFFFFF, total, offset);
            if (lane_1 == 0) {
                delta[row] = total;
            }
            float sqt = sq;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sqt += __shfl_xor_sync(0xFFFFFFFF, sqt, offset);
            float dl2;
            asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(dl2) : "f"(sqt));
            unsigned int _warp_redux_72;
            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_72) : "r"(__as_u32(dmax)));
            float dinf = __uint_as_float(_warp_redux_72);
            unsigned int _warp_redux_73;
            asm volatile("redux.sync.max.u32 %0, %1, 0xFFFFFFFF;" : "=r"(_warp_redux_73) : "r"(__as_u32(qmax)));
            float qlinf = __uint_as_float(_warp_redux_73);
            float dabs;
            asm("abs.f32 %0, %1;" : "=f"(dabs) : "f"(total));
            float s_lat = sm_scale * qlinf;
            a_lat = a_lat + (dinf + s_lat * dabs);
            b_lat = b_lat + s_lat * dl2;
        }
        if (lane_1 == 0) {
            part[warp_1 * 4] = a_lat;
            part[warp_1 * 4 + 1] = b_lat;
            part[warp_1 * 4 + 2] = a_rope;
            part[warp_1 * 4 + 3] = b_rope;
        }
        __syncthreads();
        if (tid < 2) {
            float term = part[tid];
            term = term + part[4 + tid];
            term = term + part[8 + tid];
            term = term + part[12 + tid];
            term = term + part[16 + tid];
            term = term + part[20 + tid];
            term = term + part[24 + tid];
            term = term + part[28 + tid];
            qstat[(long long)token * (long long)qstat_stride + (long long)tid] = term;
        }
    }
}

} // extern "C"
