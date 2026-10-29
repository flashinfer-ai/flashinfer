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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_STATS_OFF 0
#define SMEM_STATS_STAGE_BYTES 64
#define SMEM_STATS_STRIDE 64
#define SMEM_OUT_STATS_OFF 64
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 128
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_kimi_k3_attn_res_fc22fa228a3b146277e8
#endif
extern "C" {

__global__ __launch_bounds__(256) void
CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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

    // Kernel setup ops
    float* stats = reinterpret_cast<float*>(smem_raw + 0);
    const int stats_addr = smem + 0;
    float* out_stats = reinterpret_cast<float*>(smem_raw + 64);
    const int out_stats_addr = smem + 64;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = warp;
    int token = bid;
    int group = warp_0 / 4;
    int thread = (warp_0 * 32 + lane) % 128;
    if (token < M) {
        unsigned long long token64 = (unsigned long long)token;
        unsigned long long row_base = token64 * 7168;
        unsigned long long block_base = token64 * blocks_m_stride;
        unsigned int words[14];
        unsigned int dwords[14];
        float q[28];
        float wout[28];
        int base_0 = group * 1024 + thread * 8;
        const int woff_0 = 0;
        unsigned int raw_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            raw_0[0 + 0] = _uv4_0.x;
            raw_0[0 + 1] = _uv4_0.y;
            raw_0[0 + 2] = _uv4_0.z;
            raw_0[0 + 3] = _uv4_0.w;
        }
        words[woff_0] = raw_0[0];
        words[woff_0 + 1] = raw_0[1];
        words[woff_0 + 2] = raw_0[2];
        words[woff_0 + 3] = raw_0[3];
        unsigned int raw_1[4];
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            raw_1[0 + 0] = _uv4_1.x;
            raw_1[0 + 1] = _uv4_1.y;
            raw_1[0 + 2] = _uv4_1.z;
            raw_1[0 + 3] = _uv4_1.w;
        }
        dwords[woff_0] = raw_1[0];
        dwords[woff_0 + 1] = raw_1[1];
        dwords[woff_0 + 2] = raw_1[2];
        dwords[woff_0 + 3] = raw_1[3];
        float onw_0[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&onw_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&onw_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        wout[0] = onw_0[0];
        wout[1] = onw_0[1];
        wout[2] = onw_0[2];
        wout[3] = onw_0[3];
        wout[4] = onw_0[4];
        wout[5] = onw_0[5];
        wout[6] = onw_0[6];
        wout[7] = onw_0[7];
        int base_1 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int raw_2[4];
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_1)) + 0);
            raw_2[0 + 0] = _uv4_3.x;
            raw_2[0 + 1] = _uv4_3.y;
            raw_2[0 + 2] = _uv4_3.z;
            raw_2[0 + 3] = _uv4_3.w;
        }
        words[woff_1] = raw_2[0];
        words[woff_1 + 1] = raw_2[1];
        words[woff_1 + 2] = raw_2[2];
        words[woff_1 + 3] = raw_2[3];
        unsigned int raw_3[4];
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_1)) + 0);
            raw_3[0 + 0] = _uv4_4.x;
            raw_3[0 + 1] = _uv4_4.y;
            raw_3[0 + 2] = _uv4_4.z;
            raw_3[0 + 3] = _uv4_4.w;
        }
        dwords[woff_1] = raw_3[0];
        dwords[woff_1 + 1] = raw_3[1];
        dwords[woff_1 + 2] = raw_3[2];
        dwords[woff_1 + 3] = raw_3[3];
        float onw_1[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(output_norm_weight + base_1);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&onw_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&onw_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        wout[8] = onw_1[0];
        wout[9] = onw_1[1];
        wout[10] = onw_1[2];
        wout[11] = onw_1[3];
        wout[12] = onw_1[4];
        wout[13] = onw_1[5];
        wout[14] = onw_1[6];
        wout[15] = onw_1[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_2 = 8;
        unsigned int raw_4[4];
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            raw_4[0 + 0] = _uv4_6.x;
            raw_4[0 + 1] = _uv4_6.y;
            raw_4[0 + 2] = _uv4_6.z;
            raw_4[0 + 3] = _uv4_6.w;
        }
        words[woff_2] = raw_4[0];
        words[woff_2 + 1] = raw_4[1];
        words[woff_2 + 2] = raw_4[2];
        words[woff_2 + 3] = raw_4[3];
        unsigned int raw_5[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            raw_5[0 + 0] = _uv4_7.x;
            raw_5[0 + 1] = _uv4_7.y;
            raw_5[0 + 2] = _uv4_7.z;
            raw_5[0 + 3] = _uv4_7.w;
        }
        dwords[woff_2] = raw_5[0];
        dwords[woff_2 + 1] = raw_5[1];
        dwords[woff_2 + 2] = raw_5[2];
        dwords[woff_2 + 3] = raw_5[3];
        float onw_2[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_8[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_8[_blk] = _vptr_8[_blk];
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&onw_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&onw_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        wout[16] = onw_2[0];
        wout[17] = onw_2[1];
        wout[18] = onw_2[2];
        wout[19] = onw_2[3];
        wout[20] = onw_2[4];
        wout[21] = onw_2[5];
        wout[22] = onw_2[6];
        wout[23] = onw_2[7];
        int base_3 = 6144 + group * 512 + thread * 4;
        const int woff_3 = 12;
        unsigned int raw1_0[1];
        {
            raw1_0[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_3)) + 0);
        }
        words[woff_3] = raw1_0[0];
        unsigned int raw1_1[1];
        {
            raw1_1[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_3)) + 1);
        }
        words[woff_3 + 1] = raw1_1[0];
        unsigned int raw1_2[1];
        {
            raw1_2[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_3)) + 0);
        }
        dwords[woff_3] = raw1_2[0];
        unsigned int raw1_3[1];
        {
            raw1_3[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_3)) + 1);
        }
        dwords[woff_3 + 1] = raw1_3[0];
        float onw_3[4];
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(output_norm_weight + base_3);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&onw_3[0 + _pair * 2])[0]), "=f"((&onw_3[0 + _pair * 2])[1])
                    : "r"(_vpairs_9[_pair]));
            }
        }
        wout[24] = onw_3[0];
        wout[25] = onw_3[1];
        wout[26] = onw_3[2];
        wout[27] = onw_3[3];
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        float acc[28];
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        acc[8] = 0.0f;
        acc[9] = 0.0f;
        acc[10] = 0.0f;
        acc[11] = 0.0f;
        acc[12] = 0.0f;
        acc[13] = 0.0f;
        acc[14] = 0.0f;
        acc[15] = 0.0f;
        acc[16] = 0.0f;
        acc[17] = 0.0f;
        acc[18] = 0.0f;
        acc[19] = 0.0f;
        acc[20] = 0.0f;
        acc[21] = 0.0f;
        acc[22] = 0.0f;
        acc[23] = 0.0f;
        acc[24] = 0.0f;
        acc[25] = 0.0f;
        acc[26] = 0.0f;
        acc[27] = 0.0f;
        float updated[28];
        float2 local_sum_pair = make_float2(0.0f, 0.0f);
        int base_4 = group * 1024 + thread * 8;
        const int woff_4 = 0;
        unsigned int pw_0[4];
        unsigned int dw_0[4];
        pw_0[0] = words[woff_4];
        dw_0[0] = dwords[woff_4];
        pw_0[1] = words[woff_4 + 1];
        dw_0[1] = dwords[woff_4 + 1];
        pw_0[2] = words[woff_4 + 2];
        dw_0[2] = dwords[woff_4 + 2];
        pw_0[3] = words[woff_4 + 3];
        dw_0[3] = dwords[woff_4 + 3];
        float pv_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&pv_0[_pair * 2])[0]), "=f"((&pv_0[_pair * 2])[1])
                : "r"(pw_0[_pair]));
        }
        float dv_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dv_0[_pair * 2])[0]), "=f"((&dv_0[_pair * 2])[1])
                : "r"(dw_0[_pair]));
        }
        const int value_idx_0 = 0;
        const int acc_idx_0 = value_idx_0;
        float2 updated_pair_0 = add_f32x2_noftz(make_float2(pv_0[value_idx_0], pv_0[value_idx_0 + 1]), make_float2(dv_0[value_idx_0], dv_0[value_idx_0 + 1]));
        updated[acc_idx_0] = (float)(__nv_bfloat16)updated_pair_0.x;
        updated[acc_idx_0 + 1] = (float)(__nv_bfloat16)updated_pair_0.y;
        const int value_idx_1 = 2;
        const int acc_idx_1 = value_idx_1;
        float2 updated_pair_1 = add_f32x2_noftz(make_float2(pv_0[value_idx_1], pv_0[value_idx_1 + 1]), make_float2(dv_0[value_idx_1], dv_0[value_idx_1 + 1]));
        updated[acc_idx_1] = (float)(__nv_bfloat16)updated_pair_1.x;
        updated[acc_idx_1 + 1] = (float)(__nv_bfloat16)updated_pair_1.y;
        const int value_idx_2 = 4;
        const int acc_idx_2 = value_idx_2;
        float2 updated_pair_2 = add_f32x2_noftz(make_float2(pv_0[value_idx_2], pv_0[value_idx_2 + 1]), make_float2(dv_0[value_idx_2], dv_0[value_idx_2 + 1]));
        updated[acc_idx_2] = (float)(__nv_bfloat16)updated_pair_2.x;
        updated[acc_idx_2 + 1] = (float)(__nv_bfloat16)updated_pair_2.y;
        const int value_idx_3 = 6;
        const int acc_idx_3 = value_idx_3;
        float2 updated_pair_3 = add_f32x2_noftz(make_float2(pv_0[value_idx_3], pv_0[value_idx_3 + 1]), make_float2(dv_0[value_idx_3], dv_0[value_idx_3 + 1]));
        updated[acc_idx_3] = (float)(__nv_bfloat16)updated_pair_3.x;
        updated[acc_idx_3 + 1] = (float)(__nv_bfloat16)updated_pair_3.y;
        unsigned int packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 0], updated[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 2], updated[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 4], updated[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 6], updated[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(prefix + (row_base + (unsigned long long)base_4))[0] = reinterpret_cast<int4*>(packed_0)[0];
        const int acc_idx_4 = 0;
        float2 value_pair_0 = make_float2(updated[acc_idx_4], updated[acc_idx_4 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_0, value_pair_0, local_sum_pair);
        const int acc_idx_5 = 2;
        float2 value_pair_1 = make_float2(updated[acc_idx_5], updated[acc_idx_5 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_1, value_pair_1, local_sum_pair);
        const int acc_idx_6 = 4;
        float2 value_pair_2 = make_float2(updated[acc_idx_6], updated[acc_idx_6 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_2, value_pair_2, local_sum_pair);
        const int acc_idx_7 = 6;
        float2 value_pair_3 = make_float2(updated[acc_idx_7], updated[acc_idx_7 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_3, value_pair_3, local_sum_pair);
        int base_5 = (2 + group) * 1024 + thread * 8;
        const int woff_5 = 4;
        unsigned int pw_1[4];
        unsigned int dw_1[4];
        pw_1[0] = words[woff_5];
        dw_1[0] = dwords[woff_5];
        pw_1[1] = words[woff_5 + 1];
        dw_1[1] = dwords[woff_5 + 1];
        pw_1[2] = words[woff_5 + 2];
        dw_1[2] = dwords[woff_5 + 2];
        pw_1[3] = words[woff_5 + 3];
        dw_1[3] = dwords[woff_5 + 3];
        float pv_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&pv_1[_pair * 2])[0]), "=f"((&pv_1[_pair * 2])[1])
                : "r"(pw_1[_pair]));
        }
        float dv_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dv_1[_pair * 2])[0]), "=f"((&dv_1[_pair * 2])[1])
                : "r"(dw_1[_pair]));
        }
        const int value_idx_4 = 0;
        const int acc_idx_8 = 8 + value_idx_4;
        float2 updated_pair_4 = add_f32x2_noftz(make_float2(pv_1[value_idx_4], pv_1[value_idx_4 + 1]), make_float2(dv_1[value_idx_4], dv_1[value_idx_4 + 1]));
        updated[acc_idx_8] = (float)(__nv_bfloat16)updated_pair_4.x;
        updated[acc_idx_8 + 1] = (float)(__nv_bfloat16)updated_pair_4.y;
        const int value_idx_5 = 2;
        const int acc_idx_9 = 8 + value_idx_5;
        float2 updated_pair_5 = add_f32x2_noftz(make_float2(pv_1[value_idx_5], pv_1[value_idx_5 + 1]), make_float2(dv_1[value_idx_5], dv_1[value_idx_5 + 1]));
        updated[acc_idx_9] = (float)(__nv_bfloat16)updated_pair_5.x;
        updated[acc_idx_9 + 1] = (float)(__nv_bfloat16)updated_pair_5.y;
        const int value_idx_6 = 4;
        const int acc_idx_10 = 8 + value_idx_6;
        float2 updated_pair_6 = add_f32x2_noftz(make_float2(pv_1[value_idx_6], pv_1[value_idx_6 + 1]), make_float2(dv_1[value_idx_6], dv_1[value_idx_6 + 1]));
        updated[acc_idx_10] = (float)(__nv_bfloat16)updated_pair_6.x;
        updated[acc_idx_10 + 1] = (float)(__nv_bfloat16)updated_pair_6.y;
        const int value_idx_7 = 6;
        const int acc_idx_11 = 8 + value_idx_7;
        float2 updated_pair_7 = add_f32x2_noftz(make_float2(pv_1[value_idx_7], pv_1[value_idx_7 + 1]), make_float2(dv_1[value_idx_7], dv_1[value_idx_7 + 1]));
        updated[acc_idx_11] = (float)(__nv_bfloat16)updated_pair_7.x;
        updated[acc_idx_11 + 1] = (float)(__nv_bfloat16)updated_pair_7.y;
        unsigned int packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 8], updated[_lp*2+1 + 8]));
            packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 10], updated[_lp*2+1 + 10]));
            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 12], updated[_lp*2+1 + 12]));
            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 14], updated[_lp*2+1 + 14]));
            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(prefix + (row_base + (unsigned long long)base_5))[0] = reinterpret_cast<int4*>(packed_1)[0];
        const int acc_idx_12 = 8;
        float2 value_pair_4 = make_float2(updated[acc_idx_12], updated[acc_idx_12 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_4, value_pair_4, local_sum_pair);
        const int acc_idx_13 = 10;
        float2 value_pair_5 = make_float2(updated[acc_idx_13], updated[acc_idx_13 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_5, value_pair_5, local_sum_pair);
        const int acc_idx_14 = 12;
        float2 value_pair_6 = make_float2(updated[acc_idx_14], updated[acc_idx_14 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_6, value_pair_6, local_sum_pair);
        const int acc_idx_15 = 14;
        float2 value_pair_7 = make_float2(updated[acc_idx_15], updated[acc_idx_15 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_7, value_pair_7, local_sum_pair);
        int base_6 = (4 + group) * 1024 + thread * 8;
        const int woff_6 = 8;
        unsigned int pw_2[4];
        unsigned int dw_2[4];
        pw_2[0] = words[woff_6];
        dw_2[0] = dwords[woff_6];
        pw_2[1] = words[woff_6 + 1];
        dw_2[1] = dwords[woff_6 + 1];
        pw_2[2] = words[woff_6 + 2];
        dw_2[2] = dwords[woff_6 + 2];
        pw_2[3] = words[woff_6 + 3];
        dw_2[3] = dwords[woff_6 + 3];
        float pv_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&pv_2[_pair * 2])[0]), "=f"((&pv_2[_pair * 2])[1])
                : "r"(pw_2[_pair]));
        }
        float dv_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dv_2[_pair * 2])[0]), "=f"((&dv_2[_pair * 2])[1])
                : "r"(dw_2[_pair]));
        }
        const int value_idx_8 = 0;
        const int acc_idx_16 = 16 + value_idx_8;
        float2 updated_pair_8 = add_f32x2_noftz(make_float2(pv_2[value_idx_8], pv_2[value_idx_8 + 1]), make_float2(dv_2[value_idx_8], dv_2[value_idx_8 + 1]));
        updated[acc_idx_16] = (float)(__nv_bfloat16)updated_pair_8.x;
        updated[acc_idx_16 + 1] = (float)(__nv_bfloat16)updated_pair_8.y;
        const int value_idx_9 = 2;
        const int acc_idx_17 = 16 + value_idx_9;
        float2 updated_pair_9 = add_f32x2_noftz(make_float2(pv_2[value_idx_9], pv_2[value_idx_9 + 1]), make_float2(dv_2[value_idx_9], dv_2[value_idx_9 + 1]));
        updated[acc_idx_17] = (float)(__nv_bfloat16)updated_pair_9.x;
        updated[acc_idx_17 + 1] = (float)(__nv_bfloat16)updated_pair_9.y;
        const int value_idx_10 = 4;
        const int acc_idx_18 = 16 + value_idx_10;
        float2 updated_pair_10 = add_f32x2_noftz(make_float2(pv_2[value_idx_10], pv_2[value_idx_10 + 1]), make_float2(dv_2[value_idx_10], dv_2[value_idx_10 + 1]));
        updated[acc_idx_18] = (float)(__nv_bfloat16)updated_pair_10.x;
        updated[acc_idx_18 + 1] = (float)(__nv_bfloat16)updated_pair_10.y;
        const int value_idx_11 = 6;
        const int acc_idx_19 = 16 + value_idx_11;
        float2 updated_pair_11 = add_f32x2_noftz(make_float2(pv_2[value_idx_11], pv_2[value_idx_11 + 1]), make_float2(dv_2[value_idx_11], dv_2[value_idx_11 + 1]));
        updated[acc_idx_19] = (float)(__nv_bfloat16)updated_pair_11.x;
        updated[acc_idx_19 + 1] = (float)(__nv_bfloat16)updated_pair_11.y;
        unsigned int packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 16], updated[_lp*2+1 + 16]));
            packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 18], updated[_lp*2+1 + 18]));
            packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 20], updated[_lp*2+1 + 20]));
            packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 22], updated[_lp*2+1 + 22]));
            packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(prefix + (row_base + (unsigned long long)base_6))[0] = reinterpret_cast<int4*>(packed_2)[0];
        const int acc_idx_20 = 16;
        float2 value_pair_8 = make_float2(updated[acc_idx_20], updated[acc_idx_20 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_8, value_pair_8, local_sum_pair);
        const int acc_idx_21 = 18;
        float2 value_pair_9 = make_float2(updated[acc_idx_21], updated[acc_idx_21 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_9, value_pair_9, local_sum_pair);
        const int acc_idx_22 = 20;
        float2 value_pair_10 = make_float2(updated[acc_idx_22], updated[acc_idx_22 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_10, value_pair_10, local_sum_pair);
        const int acc_idx_23 = 22;
        float2 value_pair_11 = make_float2(updated[acc_idx_23], updated[acc_idx_23 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_11, value_pair_11, local_sum_pair);
        int base_7 = 6144 + group * 512 + thread * 4;
        const int woff_7 = 12;
        unsigned int pw_3[4];
        unsigned int dw_3[4];
        pw_3[0] = words[woff_7];
        dw_3[0] = dwords[woff_7];
        pw_3[1] = words[woff_7 + 1];
        dw_3[1] = dwords[woff_7 + 1];
        float pv_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&pv_3[_pair * 2])[0]), "=f"((&pv_3[_pair * 2])[1])
                : "r"(pw_3[_pair]));
        }
        float dv_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&dv_3[_pair * 2])[0]), "=f"((&dv_3[_pair * 2])[1])
                : "r"(dw_3[_pair]));
        }
        const int value_idx_12 = 0;
        const int acc_idx_24 = 24 + value_idx_12;
        float2 updated_pair_12 = add_f32x2_noftz(make_float2(pv_3[value_idx_12], pv_3[value_idx_12 + 1]), make_float2(dv_3[value_idx_12], dv_3[value_idx_12 + 1]));
        updated[acc_idx_24] = (float)(__nv_bfloat16)updated_pair_12.x;
        updated[acc_idx_24 + 1] = (float)(__nv_bfloat16)updated_pair_12.y;
        const int value_idx_13 = 2;
        const int acc_idx_25 = 24 + value_idx_13;
        float2 updated_pair_13 = add_f32x2_noftz(make_float2(pv_3[value_idx_13], pv_3[value_idx_13 + 1]), make_float2(dv_3[value_idx_13], dv_3[value_idx_13 + 1]));
        updated[acc_idx_25] = (float)(__nv_bfloat16)updated_pair_13.x;
        updated[acc_idx_25 + 1] = (float)(__nv_bfloat16)updated_pair_13.y;
        unsigned int packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 24], updated[_lp*2+1 + 24]));
            packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 26], updated[_lp*2+1 + 26]));
            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(prefix + (row_base + (unsigned long long)base_7))[0] = reinterpret_cast<int2*>(packed_3)[0];
        const int acc_idx_26 = 24;
        float2 value_pair_12 = make_float2(updated[acc_idx_26], updated[acc_idx_26 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_12, value_pair_12, local_sum_pair);
        const int acc_idx_27 = 26;
        float2 value_pair_13 = make_float2(updated[acc_idx_27], updated[acc_idx_27 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_13, value_pair_13, local_sum_pair);
        acc[0] = updated[0];
        acc[1] = updated[1];
        acc[2] = updated[2];
        acc[3] = updated[3];
        acc[4] = updated[4];
        acc[5] = updated[5];
        acc[6] = updated[6];
        acc[7] = updated[7];
        acc[8] = updated[8];
        acc[9] = updated[9];
        acc[10] = updated[10];
        acc[11] = updated[11];
        acc[12] = updated[12];
        acc[13] = updated[13];
        acc[14] = updated[14];
        acc[15] = updated[15];
        acc[16] = updated[16];
        acc[17] = updated[17];
        acc[18] = updated[18];
        acc[19] = updated[19];
        acc[20] = updated[20];
        acc[21] = updated[21];
        acc[22] = updated[22];
        acc[23] = updated[23];
        acc[24] = updated[24];
        acc[25] = updated[25];
        acc[26] = updated[26];
        acc[27] = updated[27];
        float sum_running = 1.0f;
        float output_sq = local_sum_pair.x + local_sum_pair.y;
        float _warp_reduce_10 = output_sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_10 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_10, offset);
        output_sq = _warp_reduce_10;
        if (lane == 0) {
            out_stats[warp_0] = output_sq;
        }
        __syncthreads();
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_11 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_11;
        float _shfl_down_sync_12 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_12;
        float _shfl_down_sync_13 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_13;
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_14 = rsqrtf(output_total / 7168.0f + output_norm_eps);
            rsigma_lane = _rsqrt_14;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_8 = group * 1024 + thread * 8;
        float output_values_0[8];
        const int value_idx_14 = 0;
        const int acc_idx_28 = value_idx_14;
        float2 _f32x2_mul_a_15 = make_float2(acc[acc_idx_28], acc[acc_idx_28 + 1]);
        float2 scaled_pair_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_16 = make_float2(wout[acc_idx_28], wout[acc_idx_28 + 1]);
        float2 normalized_pair_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_0) : "l"(*(const unsigned long long*)&scaled_pair_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        output_values_0[value_idx_14] = normalized_pair_0.x;
        output_values_0[value_idx_14 + 1] = normalized_pair_0.y;
        const int value_idx_15 = 2;
        const int acc_idx_29 = value_idx_15;
        float2 _f32x2_mul_a_17 = make_float2(acc[acc_idx_29], acc[acc_idx_29 + 1]);
        float2 scaled_pair_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_18 = make_float2(wout[acc_idx_29], wout[acc_idx_29 + 1]);
        float2 normalized_pair_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1) : "l"(*(const unsigned long long*)&scaled_pair_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        output_values_0[value_idx_15] = normalized_pair_1.x;
        output_values_0[value_idx_15 + 1] = normalized_pair_1.y;
        const int value_idx_16 = 4;
        const int acc_idx_30 = value_idx_16;
        float2 _f32x2_mul_a_19 = make_float2(acc[acc_idx_30], acc[acc_idx_30 + 1]);
        float2 scaled_pair_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_20 = make_float2(wout[acc_idx_30], wout[acc_idx_30 + 1]);
        float2 normalized_pair_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_2) : "l"(*(const unsigned long long*)&scaled_pair_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        output_values_0[value_idx_16] = normalized_pair_2.x;
        output_values_0[value_idx_16 + 1] = normalized_pair_2.y;
        const int value_idx_17 = 6;
        const int acc_idx_31 = value_idx_17;
        float2 _f32x2_mul_a_21 = make_float2(acc[acc_idx_31], acc[acc_idx_31 + 1]);
        float2 scaled_pair_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_22 = make_float2(wout[acc_idx_31], wout[acc_idx_31 + 1]);
        float2 normalized_pair_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_3) : "l"(*(const unsigned long long*)&scaled_pair_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        output_values_0[value_idx_17] = normalized_pair_3.x;
        output_values_0[value_idx_17 + 1] = normalized_pair_3.y;
        unsigned int packed_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 0], output_values_0[_lp*2+1 + 0]));
            packed_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 2], output_values_0[_lp*2+1 + 2]));
            packed_4[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 4], output_values_0[_lp*2+1 + 4]));
            packed_4[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 6], output_values_0[_lp*2+1 + 6]));
            packed_4[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_8))[0] = reinterpret_cast<int4*>(packed_4)[0];
        int base_9 = (2 + group) * 1024 + thread * 8;
        float output_values_1[8];
        const int value_idx_18 = 0;
        const int acc_idx_32 = 8 + value_idx_18;
        float2 _f32x2_mul_a_23 = make_float2(acc[acc_idx_32], acc[acc_idx_32 + 1]);
        float2 scaled_pair_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_24 = make_float2(wout[acc_idx_32], wout[acc_idx_32 + 1]);
        float2 normalized_pair_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_4) : "l"(*(const unsigned long long*)&scaled_pair_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        output_values_1[value_idx_18] = normalized_pair_4.x;
        output_values_1[value_idx_18 + 1] = normalized_pair_4.y;
        const int value_idx_19 = 2;
        const int acc_idx_33 = 8 + value_idx_19;
        float2 _f32x2_mul_a_25 = make_float2(acc[acc_idx_33], acc[acc_idx_33 + 1]);
        float2 scaled_pair_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_26 = make_float2(wout[acc_idx_33], wout[acc_idx_33 + 1]);
        float2 normalized_pair_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_5) : "l"(*(const unsigned long long*)&scaled_pair_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        output_values_1[value_idx_19] = normalized_pair_5.x;
        output_values_1[value_idx_19 + 1] = normalized_pair_5.y;
        const int value_idx_20 = 4;
        const int acc_idx_34 = 8 + value_idx_20;
        float2 _f32x2_mul_a_27 = make_float2(acc[acc_idx_34], acc[acc_idx_34 + 1]);
        float2 scaled_pair_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_28 = make_float2(wout[acc_idx_34], wout[acc_idx_34 + 1]);
        float2 normalized_pair_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_6) : "l"(*(const unsigned long long*)&scaled_pair_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        output_values_1[value_idx_20] = normalized_pair_6.x;
        output_values_1[value_idx_20 + 1] = normalized_pair_6.y;
        const int value_idx_21 = 6;
        const int acc_idx_35 = 8 + value_idx_21;
        float2 _f32x2_mul_a_29 = make_float2(acc[acc_idx_35], acc[acc_idx_35 + 1]);
        float2 scaled_pair_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_30 = make_float2(wout[acc_idx_35], wout[acc_idx_35 + 1]);
        float2 normalized_pair_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_7) : "l"(*(const unsigned long long*)&scaled_pair_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        output_values_1[value_idx_21] = normalized_pair_7.x;
        output_values_1[value_idx_21 + 1] = normalized_pair_7.y;
        unsigned int packed_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 0], output_values_1[_lp*2+1 + 0]));
            packed_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 2], output_values_1[_lp*2+1 + 2]));
            packed_5[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 4], output_values_1[_lp*2+1 + 4]));
            packed_5[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 6], output_values_1[_lp*2+1 + 6]));
            packed_5[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_9))[0] = reinterpret_cast<int4*>(packed_5)[0];
        int base_10 = (4 + group) * 1024 + thread * 8;
        float output_values_2[8];
        const int value_idx_22 = 0;
        const int acc_idx_36 = 16 + value_idx_22;
        float2 _f32x2_mul_a_31 = make_float2(acc[acc_idx_36], acc[acc_idx_36 + 1]);
        float2 scaled_pair_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_32 = make_float2(wout[acc_idx_36], wout[acc_idx_36 + 1]);
        float2 normalized_pair_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_8) : "l"(*(const unsigned long long*)&scaled_pair_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        output_values_2[value_idx_22] = normalized_pair_8.x;
        output_values_2[value_idx_22 + 1] = normalized_pair_8.y;
        const int value_idx_23 = 2;
        const int acc_idx_37 = 16 + value_idx_23;
        float2 _f32x2_mul_a_33 = make_float2(acc[acc_idx_37], acc[acc_idx_37 + 1]);
        float2 scaled_pair_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_34 = make_float2(wout[acc_idx_37], wout[acc_idx_37 + 1]);
        float2 normalized_pair_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_9) : "l"(*(const unsigned long long*)&scaled_pair_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        output_values_2[value_idx_23] = normalized_pair_9.x;
        output_values_2[value_idx_23 + 1] = normalized_pair_9.y;
        const int value_idx_24 = 4;
        const int acc_idx_38 = 16 + value_idx_24;
        float2 _f32x2_mul_a_35 = make_float2(acc[acc_idx_38], acc[acc_idx_38 + 1]);
        float2 scaled_pair_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_36 = make_float2(wout[acc_idx_38], wout[acc_idx_38 + 1]);
        float2 normalized_pair_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_10) : "l"(*(const unsigned long long*)&scaled_pair_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        output_values_2[value_idx_24] = normalized_pair_10.x;
        output_values_2[value_idx_24 + 1] = normalized_pair_10.y;
        const int value_idx_25 = 6;
        const int acc_idx_39 = 16 + value_idx_25;
        float2 _f32x2_mul_a_37 = make_float2(acc[acc_idx_39], acc[acc_idx_39 + 1]);
        float2 scaled_pair_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_38 = make_float2(wout[acc_idx_39], wout[acc_idx_39 + 1]);
        float2 normalized_pair_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_11) : "l"(*(const unsigned long long*)&scaled_pair_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        output_values_2[value_idx_25] = normalized_pair_11.x;
        output_values_2[value_idx_25 + 1] = normalized_pair_11.y;
        unsigned int packed_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 0], output_values_2[_lp*2+1 + 0]));
            packed_6[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 2], output_values_2[_lp*2+1 + 2]));
            packed_6[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 4], output_values_2[_lp*2+1 + 4]));
            packed_6[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 6], output_values_2[_lp*2+1 + 6]));
            packed_6[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_10))[0] = reinterpret_cast<int4*>(packed_6)[0];
        int base_11 = 6144 + group * 512 + thread * 4;
        float output_values_3[8];
        const int value_idx_26 = 0;
        const int acc_idx_40 = 24 + value_idx_26;
        float2 _f32x2_mul_a_39 = make_float2(acc[acc_idx_40], acc[acc_idx_40 + 1]);
        float2 scaled_pair_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_40 = make_float2(wout[acc_idx_40], wout[acc_idx_40 + 1]);
        float2 normalized_pair_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_12) : "l"(*(const unsigned long long*)&scaled_pair_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        output_values_3[value_idx_26] = normalized_pair_12.x;
        output_values_3[value_idx_26 + 1] = normalized_pair_12.y;
        const int value_idx_27 = 2;
        const int acc_idx_41 = 24 + value_idx_27;
        float2 _f32x2_mul_a_41 = make_float2(acc[acc_idx_41], acc[acc_idx_41 + 1]);
        float2 scaled_pair_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_42 = make_float2(wout[acc_idx_41], wout[acc_idx_41 + 1]);
        float2 normalized_pair_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_13) : "l"(*(const unsigned long long*)&scaled_pair_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        output_values_3[value_idx_27] = normalized_pair_13.x;
        output_values_3[value_idx_27 + 1] = normalized_pair_13.y;
        unsigned int packed_7[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_3[_lp*2 + 0], output_values_3[_lp*2+1 + 0]));
            packed_7[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_3[_lp*2 + 2], output_values_3[_lp*2+1 + 2]));
            packed_7[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_11))[0] = reinterpret_cast<int2*>(packed_7)[0];
    }
}

} // extern "C"
