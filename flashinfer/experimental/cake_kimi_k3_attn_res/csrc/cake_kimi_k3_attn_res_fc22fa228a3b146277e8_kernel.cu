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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_attn_res_fc22fa228a3b146277e8(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
        int base = group * 1024 + thread * 8;
        const int woff = 0;
        unsigned int _vec_load_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
            _vec_load_0[0 + 0] = _uv4_0.x;
            _vec_load_0[0 + 1] = _uv4_0.y;
            _vec_load_0[0 + 2] = _uv4_0.z;
            _vec_load_0[0 + 3] = _uv4_0.w;
        }
        words[woff] = _vec_load_0[0];
        words[woff + 1] = _vec_load_0[1];
        words[woff + 2] = _vec_load_0[2];
        words[woff + 3] = _vec_load_0[3];
        unsigned int _vec_load_1[4];
        {
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_1[0 + 0] = _uv4_1.x;
            _vec_load_1[0 + 1] = _uv4_1.y;
            _vec_load_1[0 + 2] = _uv4_1.z;
            _vec_load_1[0 + 3] = _uv4_1.w;
        }
        dwords[woff] = _vec_load_1[0];
        dwords[woff + 1] = _vec_load_1[1];
        dwords[woff + 2] = _vec_load_1[2];
        dwords[woff + 3] = _vec_load_1[3];
        float _vec_load_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(output_norm_weight + base);
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
                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        wout[0] = _vec_load_2[0];
        wout[1] = _vec_load_2[1];
        wout[2] = _vec_load_2[2];
        wout[3] = _vec_load_2[3];
        wout[4] = _vec_load_2[4];
        wout[5] = _vec_load_2[5];
        wout[6] = _vec_load_2[6];
        wout[7] = _vec_load_2[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_3[4];
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_3[0 + 0] = _uv4_3.x;
            _vec_load_3[0 + 1] = _uv4_3.y;
            _vec_load_3[0 + 2] = _uv4_3.z;
            _vec_load_3[0 + 3] = _uv4_3.w;
        }
        words[woff_1] = _vec_load_3[0];
        words[woff_1 + 1] = _vec_load_3[1];
        words[woff_1 + 2] = _vec_load_3[2];
        words[woff_1 + 3] = _vec_load_3[3];
        unsigned int _vec_load_4[4];
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_4[0 + 0] = _uv4_4.x;
            _vec_load_4[0 + 1] = _uv4_4.y;
            _vec_load_4[0 + 2] = _uv4_4.z;
            _vec_load_4[0 + 3] = _uv4_4.w;
        }
        dwords[woff_1] = _vec_load_4[0];
        dwords[woff_1 + 1] = _vec_load_4[1];
        dwords[woff_1 + 2] = _vec_load_4[2];
        dwords[woff_1 + 3] = _vec_load_4[3];
        float _vec_load_5[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        wout[8] = _vec_load_5[0];
        wout[9] = _vec_load_5[1];
        wout[10] = _vec_load_5[2];
        wout[11] = _vec_load_5[3];
        wout[12] = _vec_load_5[4];
        wout[13] = _vec_load_5[5];
        wout[14] = _vec_load_5[6];
        wout[15] = _vec_load_5[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_6[4];
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_6[0 + 0] = _uv4_6.x;
            _vec_load_6[0 + 1] = _uv4_6.y;
            _vec_load_6[0 + 2] = _uv4_6.z;
            _vec_load_6[0 + 3] = _uv4_6.w;
        }
        words[woff_3] = _vec_load_6[0];
        words[woff_3 + 1] = _vec_load_6[1];
        words[woff_3 + 2] = _vec_load_6[2];
        words[woff_3 + 3] = _vec_load_6[3];
        unsigned int _vec_load_7[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_7[0 + 0] = _uv4_7.x;
            _vec_load_7[0 + 1] = _uv4_7.y;
            _vec_load_7[0 + 2] = _uv4_7.z;
            _vec_load_7[0 + 3] = _uv4_7.w;
        }
        dwords[woff_3] = _vec_load_7[0];
        dwords[woff_3 + 1] = _vec_load_7[1];
        dwords[woff_3 + 2] = _vec_load_7[2];
        dwords[woff_3 + 3] = _vec_load_7[3];
        float _vec_load_8[8];
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
                        : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        wout[16] = _vec_load_8[0];
        wout[17] = _vec_load_8[1];
        wout[18] = _vec_load_8[2];
        wout[19] = _vec_load_8[3];
        wout[20] = _vec_load_8[4];
        wout[21] = _vec_load_8[5];
        wout[22] = _vec_load_8[6];
        wout[23] = _vec_load_8[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_9[1];
        {
            _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_9[0];
        unsigned int _vec_load_10[1];
        {
            _vec_load_10[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_10[0];
        unsigned int _vec_load_11[1];
        {
            _vec_load_11[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_11[0];
        unsigned int _vec_load_12[1];
        {
            _vec_load_12[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_12[0];
        float _vec_load_13[4];
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_13[0 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _pair * 2])[1])
                    : "r"(_vpairs_9[_pair]));
            }
        }
        wout[24] = _vec_load_13[0];
        wout[25] = _vec_load_13[1];
        wout[26] = _vec_load_13[2];
        wout[27] = _vec_load_13[3];
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
        int base_6 = group * 1024 + thread * 8;
        const int woff_7 = 0;
        unsigned int pw[4];
        unsigned int dw[4];
        pw[0] = words[woff_7];
        dw[0] = dwords[woff_7];
        pw[1] = words[woff_7 + 1];
        dw[1] = dwords[woff_7 + 1];
        pw[2] = words[woff_7 + 2];
        dw[2] = dwords[woff_7 + 2];
        pw[3] = words[woff_7 + 3];
        dw[3] = dwords[woff_7 + 3];
        float _unpack_bf16_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_0[_pair * 2])[0]), "=f"((&_unpack_bf16_0[_pair * 2])[1])
                : "r"(pw[_pair]));
        }
        float _unpack_bf16_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
                : "r"(dw[_pair]));
        }
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 updated_pair = add_f32x2_noftz(make_float2(_unpack_bf16_0[value_idx], _unpack_bf16_0[value_idx + 1]), make_float2(_unpack_bf16_1[value_idx], _unpack_bf16_1[value_idx + 1]));
        updated[acc_idx] = (float)(__nv_bfloat16)updated_pair.x;
        updated[acc_idx + 1] = (float)(__nv_bfloat16)updated_pair.y;
        const int value_idx_8 = 2;
        const int acc_idx_9 = value_idx_8;
        float2 updated_pair_10 = add_f32x2_noftz(make_float2(_unpack_bf16_0[value_idx_8], _unpack_bf16_0[value_idx_8 + 1]), make_float2(_unpack_bf16_1[value_idx_8], _unpack_bf16_1[value_idx_8 + 1]));
        updated[acc_idx_9] = (float)(__nv_bfloat16)updated_pair_10.x;
        updated[acc_idx_9 + 1] = (float)(__nv_bfloat16)updated_pair_10.y;
        const int value_idx_11 = 4;
        const int acc_idx_12 = value_idx_11;
        float2 updated_pair_13 = add_f32x2_noftz(make_float2(_unpack_bf16_0[value_idx_11], _unpack_bf16_0[value_idx_11 + 1]), make_float2(_unpack_bf16_1[value_idx_11], _unpack_bf16_1[value_idx_11 + 1]));
        updated[acc_idx_12] = (float)(__nv_bfloat16)updated_pair_13.x;
        updated[acc_idx_12 + 1] = (float)(__nv_bfloat16)updated_pair_13.y;
        const int value_idx_14 = 6;
        const int acc_idx_15 = value_idx_14;
        float2 updated_pair_16 = add_f32x2_noftz(make_float2(_unpack_bf16_0[value_idx_14], _unpack_bf16_0[value_idx_14 + 1]), make_float2(_unpack_bf16_1[value_idx_14], _unpack_bf16_1[value_idx_14 + 1]));
        updated[acc_idx_15] = (float)(__nv_bfloat16)updated_pair_16.x;
        updated[acc_idx_15 + 1] = (float)(__nv_bfloat16)updated_pair_16.y;
        unsigned int _packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 0], updated[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 2], updated[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 4], updated[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 6], updated[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(prefix + (row_base + (unsigned long long)base_6))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        const int acc_idx_17 = 0;
        float2 value_pair = make_float2(updated[acc_idx_17], updated[acc_idx_17 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair, value_pair, local_sum_pair);
        const int acc_idx_18 = 2;
        float2 value_pair_19 = make_float2(updated[acc_idx_18], updated[acc_idx_18 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_19, value_pair_19, local_sum_pair);
        const int acc_idx_20 = 4;
        float2 value_pair_21 = make_float2(updated[acc_idx_20], updated[acc_idx_20 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_21, value_pair_21, local_sum_pair);
        const int acc_idx_22 = 6;
        float2 value_pair_23 = make_float2(updated[acc_idx_22], updated[acc_idx_22 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_23, value_pair_23, local_sum_pair);
        int base_24 = (2 + group) * 1024 + thread * 8;
        const int woff_25 = 4;
        unsigned int pw_26[4];
        unsigned int dw_27[4];
        pw_26[0] = words[woff_25];
        dw_27[0] = dwords[woff_25];
        pw_26[1] = words[woff_25 + 1];
        dw_27[1] = dwords[woff_25 + 1];
        pw_26[2] = words[woff_25 + 2];
        dw_27[2] = dwords[woff_25 + 2];
        pw_26[3] = words[woff_25 + 3];
        dw_27[3] = dwords[woff_25 + 3];
        float _unpack_bf16_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                : "r"(pw_26[_pair]));
        }
        float _unpack_bf16_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                : "r"(dw_27[_pair]));
        }
        const int value_idx_28 = 0;
        const int acc_idx_29 = 8 + value_idx_28;
        float2 updated_pair_30 = add_f32x2_noftz(make_float2(_unpack_bf16_2[value_idx_28], _unpack_bf16_2[value_idx_28 + 1]), make_float2(_unpack_bf16_3[value_idx_28], _unpack_bf16_3[value_idx_28 + 1]));
        updated[acc_idx_29] = (float)(__nv_bfloat16)updated_pair_30.x;
        updated[acc_idx_29 + 1] = (float)(__nv_bfloat16)updated_pair_30.y;
        const int value_idx_31 = 2;
        const int acc_idx_32 = 8 + value_idx_31;
        float2 updated_pair_33 = add_f32x2_noftz(make_float2(_unpack_bf16_2[value_idx_31], _unpack_bf16_2[value_idx_31 + 1]), make_float2(_unpack_bf16_3[value_idx_31], _unpack_bf16_3[value_idx_31 + 1]));
        updated[acc_idx_32] = (float)(__nv_bfloat16)updated_pair_33.x;
        updated[acc_idx_32 + 1] = (float)(__nv_bfloat16)updated_pair_33.y;
        const int value_idx_34 = 4;
        const int acc_idx_35 = 8 + value_idx_34;
        float2 updated_pair_36 = add_f32x2_noftz(make_float2(_unpack_bf16_2[value_idx_34], _unpack_bf16_2[value_idx_34 + 1]), make_float2(_unpack_bf16_3[value_idx_34], _unpack_bf16_3[value_idx_34 + 1]));
        updated[acc_idx_35] = (float)(__nv_bfloat16)updated_pair_36.x;
        updated[acc_idx_35 + 1] = (float)(__nv_bfloat16)updated_pair_36.y;
        const int value_idx_37 = 6;
        const int acc_idx_38 = 8 + value_idx_37;
        float2 updated_pair_39 = add_f32x2_noftz(make_float2(_unpack_bf16_2[value_idx_37], _unpack_bf16_2[value_idx_37 + 1]), make_float2(_unpack_bf16_3[value_idx_37], _unpack_bf16_3[value_idx_37 + 1]));
        updated[acc_idx_38] = (float)(__nv_bfloat16)updated_pair_39.x;
        updated[acc_idx_38 + 1] = (float)(__nv_bfloat16)updated_pair_39.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 8], updated[_lp*2+1 + 8]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 10], updated[_lp*2+1 + 10]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 12], updated[_lp*2+1 + 12]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 14], updated[_lp*2+1 + 14]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(prefix + (row_base + (unsigned long long)base_24))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        const int acc_idx_40 = 8;
        float2 value_pair_41 = make_float2(updated[acc_idx_40], updated[acc_idx_40 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_41, value_pair_41, local_sum_pair);
        const int acc_idx_42 = 10;
        float2 value_pair_43 = make_float2(updated[acc_idx_42], updated[acc_idx_42 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_43, value_pair_43, local_sum_pair);
        const int acc_idx_44 = 12;
        float2 value_pair_45 = make_float2(updated[acc_idx_44], updated[acc_idx_44 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_45, value_pair_45, local_sum_pair);
        const int acc_idx_46 = 14;
        float2 value_pair_47 = make_float2(updated[acc_idx_46], updated[acc_idx_46 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_47, value_pair_47, local_sum_pair);
        int base_48 = (4 + group) * 1024 + thread * 8;
        const int woff_49 = 8;
        unsigned int pw_50[4];
        unsigned int dw_51[4];
        pw_50[0] = words[woff_49];
        dw_51[0] = dwords[woff_49];
        pw_50[1] = words[woff_49 + 1];
        dw_51[1] = dwords[woff_49 + 1];
        pw_50[2] = words[woff_49 + 2];
        dw_51[2] = dwords[woff_49 + 2];
        pw_50[3] = words[woff_49 + 3];
        dw_51[3] = dwords[woff_49 + 3];
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                : "r"(pw_50[_pair]));
        }
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                : "r"(dw_51[_pair]));
        }
        const int value_idx_52 = 0;
        const int acc_idx_53 = 16 + value_idx_52;
        float2 updated_pair_54 = add_f32x2_noftz(make_float2(_unpack_bf16_4[value_idx_52], _unpack_bf16_4[value_idx_52 + 1]), make_float2(_unpack_bf16_5[value_idx_52], _unpack_bf16_5[value_idx_52 + 1]));
        updated[acc_idx_53] = (float)(__nv_bfloat16)updated_pair_54.x;
        updated[acc_idx_53 + 1] = (float)(__nv_bfloat16)updated_pair_54.y;
        const int value_idx_55 = 2;
        const int acc_idx_56 = 16 + value_idx_55;
        float2 updated_pair_57 = add_f32x2_noftz(make_float2(_unpack_bf16_4[value_idx_55], _unpack_bf16_4[value_idx_55 + 1]), make_float2(_unpack_bf16_5[value_idx_55], _unpack_bf16_5[value_idx_55 + 1]));
        updated[acc_idx_56] = (float)(__nv_bfloat16)updated_pair_57.x;
        updated[acc_idx_56 + 1] = (float)(__nv_bfloat16)updated_pair_57.y;
        const int value_idx_58 = 4;
        const int acc_idx_59 = 16 + value_idx_58;
        float2 updated_pair_60 = add_f32x2_noftz(make_float2(_unpack_bf16_4[value_idx_58], _unpack_bf16_4[value_idx_58 + 1]), make_float2(_unpack_bf16_5[value_idx_58], _unpack_bf16_5[value_idx_58 + 1]));
        updated[acc_idx_59] = (float)(__nv_bfloat16)updated_pair_60.x;
        updated[acc_idx_59 + 1] = (float)(__nv_bfloat16)updated_pair_60.y;
        const int value_idx_61 = 6;
        const int acc_idx_62 = 16 + value_idx_61;
        float2 updated_pair_63 = add_f32x2_noftz(make_float2(_unpack_bf16_4[value_idx_61], _unpack_bf16_4[value_idx_61 + 1]), make_float2(_unpack_bf16_5[value_idx_61], _unpack_bf16_5[value_idx_61 + 1]));
        updated[acc_idx_62] = (float)(__nv_bfloat16)updated_pair_63.x;
        updated[acc_idx_62 + 1] = (float)(__nv_bfloat16)updated_pair_63.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 16], updated[_lp*2+1 + 16]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 18], updated[_lp*2+1 + 18]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 20], updated[_lp*2+1 + 20]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 22], updated[_lp*2+1 + 22]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(prefix + (row_base + (unsigned long long)base_48))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        const int acc_idx_64 = 16;
        float2 value_pair_65 = make_float2(updated[acc_idx_64], updated[acc_idx_64 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_65, value_pair_65, local_sum_pair);
        const int acc_idx_66 = 18;
        float2 value_pair_67 = make_float2(updated[acc_idx_66], updated[acc_idx_66 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_67, value_pair_67, local_sum_pair);
        const int acc_idx_68 = 20;
        float2 value_pair_69 = make_float2(updated[acc_idx_68], updated[acc_idx_68 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_69, value_pair_69, local_sum_pair);
        const int acc_idx_70 = 22;
        float2 value_pair_71 = make_float2(updated[acc_idx_70], updated[acc_idx_70 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_71, value_pair_71, local_sum_pair);
        int base_72 = 6144 + group * 512 + thread * 4;
        const int woff_73 = 12;
        unsigned int pw_74[4];
        unsigned int dw_75[4];
        pw_74[0] = words[woff_73];
        dw_75[0] = dwords[woff_73];
        pw_74[1] = words[woff_73 + 1];
        dw_75[1] = dwords[woff_73 + 1];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(pw_74[_pair]));
        }
        float _unpack_bf16_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                : "r"(dw_75[_pair]));
        }
        const int value_idx_76 = 0;
        const int acc_idx_77 = 24 + value_idx_76;
        float2 updated_pair_78 = add_f32x2_noftz(make_float2(_unpack_bf16_6[value_idx_76], _unpack_bf16_6[value_idx_76 + 1]), make_float2(_unpack_bf16_7[value_idx_76], _unpack_bf16_7[value_idx_76 + 1]));
        updated[acc_idx_77] = (float)(__nv_bfloat16)updated_pair_78.x;
        updated[acc_idx_77 + 1] = (float)(__nv_bfloat16)updated_pair_78.y;
        const int value_idx_79 = 2;
        const int acc_idx_80 = 24 + value_idx_79;
        float2 updated_pair_81 = add_f32x2_noftz(make_float2(_unpack_bf16_6[value_idx_79], _unpack_bf16_6[value_idx_79 + 1]), make_float2(_unpack_bf16_7[value_idx_79], _unpack_bf16_7[value_idx_79 + 1]));
        updated[acc_idx_80] = (float)(__nv_bfloat16)updated_pair_81.x;
        updated[acc_idx_80 + 1] = (float)(__nv_bfloat16)updated_pair_81.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 24], updated[_lp*2+1 + 24]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(updated[_lp*2 + 26], updated[_lp*2+1 + 26]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(prefix + (row_base + (unsigned long long)base_72))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        const int acc_idx_82 = 24;
        float2 value_pair_83 = make_float2(updated[acc_idx_82], updated[acc_idx_82 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_83, value_pair_83, local_sum_pair);
        const int acc_idx_84 = 26;
        float2 value_pair_85 = make_float2(updated[acc_idx_84], updated[acc_idx_84 + 1]);
        local_sum_pair = fma_f32x2_rn_ftz(value_pair_85, value_pair_85, local_sum_pair);
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
        int base_86 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx_87 = 0;
        const int acc_idx_88 = value_idx_87;
        float2 _f32x2_mul_a_15 = make_float2(acc[acc_idx_88], acc[acc_idx_88 + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_16 = make_float2(wout[acc_idx_88], wout[acc_idx_88 + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        output_values[value_idx_87] = normalized_pair.x;
        output_values[value_idx_87 + 1] = normalized_pair.y;
        const int value_idx_89 = 2;
        const int acc_idx_90 = value_idx_89;
        float2 _f32x2_mul_a_17 = make_float2(acc[acc_idx_90], acc[acc_idx_90 + 1]);
        float2 scaled_pair_91;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_91) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_18 = make_float2(wout[acc_idx_90], wout[acc_idx_90 + 1]);
        float2 normalized_pair_92;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_92) : "l"(*(const unsigned long long*)&scaled_pair_91), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        output_values[value_idx_89] = normalized_pair_92.x;
        output_values[value_idx_89 + 1] = normalized_pair_92.y;
        const int value_idx_93 = 4;
        const int acc_idx_94 = value_idx_93;
        float2 _f32x2_mul_a_19 = make_float2(acc[acc_idx_94], acc[acc_idx_94 + 1]);
        float2 scaled_pair_95;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_95) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_20 = make_float2(wout[acc_idx_94], wout[acc_idx_94 + 1]);
        float2 normalized_pair_96;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_96) : "l"(*(const unsigned long long*)&scaled_pair_95), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        output_values[value_idx_93] = normalized_pair_96.x;
        output_values[value_idx_93 + 1] = normalized_pair_96.y;
        const int value_idx_97 = 6;
        const int acc_idx_98 = value_idx_97;
        float2 _f32x2_mul_a_21 = make_float2(acc[acc_idx_98], acc[acc_idx_98 + 1]);
        float2 scaled_pair_99;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_99) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_22 = make_float2(wout[acc_idx_98], wout[acc_idx_98 + 1]);
        float2 normalized_pair_100;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_100) : "l"(*(const unsigned long long*)&scaled_pair_99), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        output_values[value_idx_97] = normalized_pair_100.x;
        output_values[value_idx_97 + 1] = normalized_pair_100.y;
        unsigned int _packed_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 0], output_values[_lp*2+1 + 0]));
            _packed_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 2], output_values[_lp*2+1 + 2]));
            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 4], output_values[_lp*2+1 + 4]));
            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 6], output_values[_lp*2+1 + 6]));
            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_86))[0] = reinterpret_cast<int4*>(_packed_4)[0];
        int base_101 = (2 + group) * 1024 + thread * 8;
        float output_values_102[8];
        const int value_idx_103 = 0;
        const int acc_idx_104 = 8 + value_idx_103;
        float2 _f32x2_mul_a_23 = make_float2(acc[acc_idx_104], acc[acc_idx_104 + 1]);
        float2 scaled_pair_105;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_105) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_24 = make_float2(wout[acc_idx_104], wout[acc_idx_104 + 1]);
        float2 normalized_pair_106;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_106) : "l"(*(const unsigned long long*)&scaled_pair_105), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        output_values_102[value_idx_103] = normalized_pair_106.x;
        output_values_102[value_idx_103 + 1] = normalized_pair_106.y;
        const int value_idx_107 = 2;
        const int acc_idx_108 = 8 + value_idx_107;
        float2 _f32x2_mul_a_25 = make_float2(acc[acc_idx_108], acc[acc_idx_108 + 1]);
        float2 scaled_pair_109;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_109) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_26 = make_float2(wout[acc_idx_108], wout[acc_idx_108 + 1]);
        float2 normalized_pair_110;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_110) : "l"(*(const unsigned long long*)&scaled_pair_109), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        output_values_102[value_idx_107] = normalized_pair_110.x;
        output_values_102[value_idx_107 + 1] = normalized_pair_110.y;
        const int value_idx_111 = 4;
        const int acc_idx_112 = 8 + value_idx_111;
        float2 _f32x2_mul_a_27 = make_float2(acc[acc_idx_112], acc[acc_idx_112 + 1]);
        float2 scaled_pair_113;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_113) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_28 = make_float2(wout[acc_idx_112], wout[acc_idx_112 + 1]);
        float2 normalized_pair_114;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_114) : "l"(*(const unsigned long long*)&scaled_pair_113), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        output_values_102[value_idx_111] = normalized_pair_114.x;
        output_values_102[value_idx_111 + 1] = normalized_pair_114.y;
        const int value_idx_115 = 6;
        const int acc_idx_116 = 8 + value_idx_115;
        float2 _f32x2_mul_a_29 = make_float2(acc[acc_idx_116], acc[acc_idx_116 + 1]);
        float2 scaled_pair_117;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_117) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_30 = make_float2(wout[acc_idx_116], wout[acc_idx_116 + 1]);
        float2 normalized_pair_118;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_118) : "l"(*(const unsigned long long*)&scaled_pair_117), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        output_values_102[value_idx_115] = normalized_pair_118.x;
        output_values_102[value_idx_115 + 1] = normalized_pair_118.y;
        unsigned int _packed_5[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_102[_lp*2 + 0], output_values_102[_lp*2+1 + 0]));
            _packed_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_102[_lp*2 + 2], output_values_102[_lp*2+1 + 2]));
            _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_102[_lp*2 + 4], output_values_102[_lp*2+1 + 4]));
            _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_102[_lp*2 + 6], output_values_102[_lp*2+1 + 6]));
            _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_101))[0] = reinterpret_cast<int4*>(_packed_5)[0];
        int base_119 = (4 + group) * 1024 + thread * 8;
        float output_values_120[8];
        const int value_idx_121 = 0;
        const int acc_idx_122 = 16 + value_idx_121;
        float2 _f32x2_mul_a_31 = make_float2(acc[acc_idx_122], acc[acc_idx_122 + 1]);
        float2 scaled_pair_123;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_123) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_32 = make_float2(wout[acc_idx_122], wout[acc_idx_122 + 1]);
        float2 normalized_pair_124;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_124) : "l"(*(const unsigned long long*)&scaled_pair_123), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        output_values_120[value_idx_121] = normalized_pair_124.x;
        output_values_120[value_idx_121 + 1] = normalized_pair_124.y;
        const int value_idx_125 = 2;
        const int acc_idx_126 = 16 + value_idx_125;
        float2 _f32x2_mul_a_33 = make_float2(acc[acc_idx_126], acc[acc_idx_126 + 1]);
        float2 scaled_pair_127;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_127) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_34 = make_float2(wout[acc_idx_126], wout[acc_idx_126 + 1]);
        float2 normalized_pair_128;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_128) : "l"(*(const unsigned long long*)&scaled_pair_127), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        output_values_120[value_idx_125] = normalized_pair_128.x;
        output_values_120[value_idx_125 + 1] = normalized_pair_128.y;
        const int value_idx_129 = 4;
        const int acc_idx_130 = 16 + value_idx_129;
        float2 _f32x2_mul_a_35 = make_float2(acc[acc_idx_130], acc[acc_idx_130 + 1]);
        float2 scaled_pair_131;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_131) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_36 = make_float2(wout[acc_idx_130], wout[acc_idx_130 + 1]);
        float2 normalized_pair_132;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_132) : "l"(*(const unsigned long long*)&scaled_pair_131), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        output_values_120[value_idx_129] = normalized_pair_132.x;
        output_values_120[value_idx_129 + 1] = normalized_pair_132.y;
        const int value_idx_133 = 6;
        const int acc_idx_134 = 16 + value_idx_133;
        float2 _f32x2_mul_a_37 = make_float2(acc[acc_idx_134], acc[acc_idx_134 + 1]);
        float2 scaled_pair_135;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_135) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_38 = make_float2(wout[acc_idx_134], wout[acc_idx_134 + 1]);
        float2 normalized_pair_136;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_136) : "l"(*(const unsigned long long*)&scaled_pair_135), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        output_values_120[value_idx_133] = normalized_pair_136.x;
        output_values_120[value_idx_133 + 1] = normalized_pair_136.y;
        unsigned int _packed_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_120[_lp*2 + 0], output_values_120[_lp*2+1 + 0]));
            _packed_6[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_120[_lp*2 + 2], output_values_120[_lp*2+1 + 2]));
            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_120[_lp*2 + 4], output_values_120[_lp*2+1 + 4]));
            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_120[_lp*2 + 6], output_values_120[_lp*2+1 + 6]));
            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_119))[0] = reinterpret_cast<int4*>(_packed_6)[0];
        int base_137 = 6144 + group * 512 + thread * 4;
        float output_values_138[8];
        const int value_idx_139 = 0;
        const int acc_idx_140 = 24 + value_idx_139;
        float2 _f32x2_mul_a_39 = make_float2(acc[acc_idx_140], acc[acc_idx_140 + 1]);
        float2 scaled_pair_141;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_141) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_40 = make_float2(wout[acc_idx_140], wout[acc_idx_140 + 1]);
        float2 normalized_pair_142;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_142) : "l"(*(const unsigned long long*)&scaled_pair_141), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        output_values_138[value_idx_139] = normalized_pair_142.x;
        output_values_138[value_idx_139 + 1] = normalized_pair_142.y;
        const int value_idx_143 = 2;
        const int acc_idx_144 = 24 + value_idx_143;
        float2 _f32x2_mul_a_41 = make_float2(acc[acc_idx_144], acc[acc_idx_144 + 1]);
        float2 scaled_pair_145;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_145) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_42 = make_float2(wout[acc_idx_144], wout[acc_idx_144 + 1]);
        float2 normalized_pair_146;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_146) : "l"(*(const unsigned long long*)&scaled_pair_145), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        output_values_138[value_idx_143] = normalized_pair_146.x;
        output_values_138[value_idx_143 + 1] = normalized_pair_146.y;
        unsigned int _packed_7[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_138[_lp*2 + 0], output_values_138[_lp*2+1 + 0]));
            _packed_7[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_138[_lp*2 + 2], output_values_138[_lp*2+1 + 2]));
            _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_137))[0] = reinterpret_cast<int2*>(_packed_7)[0];
    }
}

} // extern "C"
