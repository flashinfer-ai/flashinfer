/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define SMEM_STATS_STAGE_BYTES 128
#define SMEM_STATS_STRIDE 128
#define SMEM_OUT_STATS_OFF 128
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 256
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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_attn_res_6c023a0cead93de42945(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 128);
    const int out_stats_addr = smem + 128;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = warp;
    int token = bid;
    int group = warp_0 / 4;
    int thread = (warp_0 * 32 + lane) % 128;
    if (token < M) {
        unsigned long long token64 = (unsigned long long)token;
        unsigned long long row_base = token64 * 7168;
        unsigned long long block_base = token64 * blocks_m_stride;
        unsigned int words[28];
        unsigned int dwords[14];
        float q[28];
        float wout[28];
        int base = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        const int woff = 0;
        {
            unsigned int _vec_load_0[4];
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base))) + 0);
                _vec_load_0[0 + 0] = _uv4_0.x;
                _vec_load_0[0 + 1] = _uv4_0.y;
                _vec_load_0[0 + 2] = _uv4_0.z;
                _vec_load_0[0 + 3] = _uv4_0.w;
            }
            words[woff] = _vec_load_0[0];
            words[woff + 1] = _vec_load_0[1];
            words[woff + 2] = _vec_load_0[2];
            words[woff + 3] = _vec_load_0[3];
        }
        {
            unsigned int _vec_load_3[4];
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_3[0 + 0] = _uv4_1.x;
                _vec_load_3[0 + 1] = _uv4_1.y;
                _vec_load_3[0 + 2] = _uv4_1.z;
                _vec_load_3[0 + 3] = _uv4_1.w;
            }
            words[14 + woff] = _vec_load_3[0];
            words[14 + woff + 1] = _vec_load_3[1];
            words[14 + woff + 2] = _vec_load_3[2];
            words[14 + woff + 3] = _vec_load_3[3];
        }
        {
            unsigned int _vec_load_6[4];
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_6[0 + 0] = _uv4_2.x;
                _vec_load_6[0 + 1] = _uv4_2.y;
                _vec_load_6[0 + 2] = _uv4_2.z;
                _vec_load_6[0 + 3] = _uv4_2.w;
            }
            dwords[woff] = _vec_load_6[0];
            dwords[woff + 1] = _vec_load_6[1];
            dwords[woff + 2] = _vec_load_6[2];
            dwords[woff + 3] = _vec_load_6[3];
        }
        float _vec_load_9[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(norm_weight + base);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
        float _vec_load_10[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(qk_weight + base);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        q[0] = _vec_load_9[0] * _vec_load_10[0];
        q[1] = _vec_load_9[1] * _vec_load_10[1];
        q[2] = _vec_load_9[2] * _vec_load_10[2];
        q[3] = _vec_load_9[3] * _vec_load_10[3];
        q[4] = _vec_load_9[4] * _vec_load_10[4];
        q[5] = _vec_load_9[5] * _vec_load_10[5];
        q[6] = _vec_load_9[6] * _vec_load_10[6];
        q[7] = _vec_load_9[7] * _vec_load_10[7];
        float _vec_load_11[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(output_norm_weight + base);
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
                        : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        wout[0] = _vec_load_11[0];
        wout[1] = _vec_load_11[1];
        wout[2] = _vec_load_11[2];
        wout[3] = _vec_load_11[3];
        wout[4] = _vec_load_11[4];
        wout[5] = _vec_load_11[5];
        wout[6] = _vec_load_11[6];
        wout[7] = _vec_load_11[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_12[4];
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_12[0 + 0] = _uv4_6.x;
                _vec_load_12[0 + 1] = _uv4_6.y;
                _vec_load_12[0 + 2] = _uv4_6.z;
                _vec_load_12[0 + 3] = _uv4_6.w;
            }
            words[woff_1] = _vec_load_12[0];
            words[woff_1 + 1] = _vec_load_12[1];
            words[woff_1 + 2] = _vec_load_12[2];
            words[woff_1 + 3] = _vec_load_12[3];
        }
        {
            unsigned int _vec_load_15[4];
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_15[0 + 0] = _uv4_7.x;
                _vec_load_15[0 + 1] = _uv4_7.y;
                _vec_load_15[0 + 2] = _uv4_7.z;
                _vec_load_15[0 + 3] = _uv4_7.w;
            }
            words[14 + woff_1] = _vec_load_15[0];
            words[14 + woff_1 + 1] = _vec_load_15[1];
            words[14 + woff_1 + 2] = _vec_load_15[2];
            words[14 + woff_1 + 3] = _vec_load_15[3];
        }
        {
            unsigned int _vec_load_18[4];
            {
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_18[0 + 0] = _uv4_8.x;
                _vec_load_18[0 + 1] = _uv4_8.y;
                _vec_load_18[0 + 2] = _uv4_8.z;
                _vec_load_18[0 + 3] = _uv4_8.w;
            }
            dwords[woff_1] = _vec_load_18[0];
            dwords[woff_1 + 1] = _vec_load_18[1];
            dwords[woff_1 + 2] = _vec_load_18[2];
            dwords[woff_1 + 3] = _vec_load_18[3];
        }
        float _vec_load_21[8];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(norm_weight + base_0);
            uint4 _vld_9[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_9[_blk] = _vptr_9[_blk];
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        float _vec_load_22[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(qk_weight + base_0);
            uint4 _vld_10[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_10[_blk] = _vptr_10[_blk];
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        q[8] = _vec_load_21[0] * _vec_load_22[0];
        q[9] = _vec_load_21[1] * _vec_load_22[1];
        q[10] = _vec_load_21[2] * _vec_load_22[2];
        q[11] = _vec_load_21[3] * _vec_load_22[3];
        q[12] = _vec_load_21[4] * _vec_load_22[4];
        q[13] = _vec_load_21[5] * _vec_load_22[5];
        q[14] = _vec_load_21[6] * _vec_load_22[6];
        q[15] = _vec_load_21[7] * _vec_load_22[7];
        float _vec_load_23[8];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
            uint4 _vld_11[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_11[_blk] = _vptr_11[_blk];
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        wout[8] = _vec_load_23[0];
        wout[9] = _vec_load_23[1];
        wout[10] = _vec_load_23[2];
        wout[11] = _vec_load_23[3];
        wout[12] = _vec_load_23[4];
        wout[13] = _vec_load_23[5];
        wout[14] = _vec_load_23[6];
        wout[15] = _vec_load_23[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_24[4];
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_24[0 + 0] = _uv4_12.x;
                _vec_load_24[0 + 1] = _uv4_12.y;
                _vec_load_24[0 + 2] = _uv4_12.z;
                _vec_load_24[0 + 3] = _uv4_12.w;
            }
            words[woff_3] = _vec_load_24[0];
            words[woff_3 + 1] = _vec_load_24[1];
            words[woff_3 + 2] = _vec_load_24[2];
            words[woff_3 + 3] = _vec_load_24[3];
        }
        {
            unsigned int _vec_load_27[4];
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_27[0 + 0] = _uv4_13.x;
                _vec_load_27[0 + 1] = _uv4_13.y;
                _vec_load_27[0 + 2] = _uv4_13.z;
                _vec_load_27[0 + 3] = _uv4_13.w;
            }
            words[14 + woff_3] = _vec_load_27[0];
            words[14 + woff_3 + 1] = _vec_load_27[1];
            words[14 + woff_3 + 2] = _vec_load_27[2];
            words[14 + woff_3 + 3] = _vec_load_27[3];
        }
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_30[0 + 0] = _uv4_14.x;
                _vec_load_30[0 + 1] = _uv4_14.y;
                _vec_load_30[0 + 2] = _uv4_14.z;
                _vec_load_30[0 + 3] = _uv4_14.w;
            }
            dwords[woff_3] = _vec_load_30[0];
            dwords[woff_3 + 1] = _vec_load_30[1];
            dwords[woff_3 + 2] = _vec_load_30[2];
            dwords[woff_3 + 3] = _vec_load_30[3];
        }
        float _vec_load_33[8];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_15[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_15[_blk] = _vptr_15[_blk];
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_33[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_33[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
        }
        float _vec_load_34[8];
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_16[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_16[_blk] = _vptr_16[_blk];
                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_34[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_34[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_16[_pair]));
                }
            }
        }
        q[16] = _vec_load_33[0] * _vec_load_34[0];
        q[17] = _vec_load_33[1] * _vec_load_34[1];
        q[18] = _vec_load_33[2] * _vec_load_34[2];
        q[19] = _vec_load_33[3] * _vec_load_34[3];
        q[20] = _vec_load_33[4] * _vec_load_34[4];
        q[21] = _vec_load_33[5] * _vec_load_34[5];
        q[22] = _vec_load_33[6] * _vec_load_34[6];
        q[23] = _vec_load_33[7] * _vec_load_34[7];
        float _vec_load_35[8];
        {
            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_17[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_17[_blk] = _vptr_17[_blk];
                uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_35[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_35[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_17[_pair]));
                }
            }
        }
        wout[16] = _vec_load_35[0];
        wout[17] = _vec_load_35[1];
        wout[18] = _vec_load_35[2];
        wout[19] = _vec_load_35[3];
        wout[20] = _vec_load_35[4];
        wout[21] = _vec_load_35[5];
        wout[22] = _vec_load_35[6];
        wout[23] = _vec_load_35[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_37[1];
            {
                _vec_load_37[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_37[0];
            unsigned int _vec_load_38[1];
            {
                _vec_load_38[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_38[0];
        }
        {
            unsigned int _vec_load_40[1];
            {
                _vec_load_40[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_40[0];
            unsigned int _vec_load_41[1];
            {
                _vec_load_41[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_41[0];
        }
        {
            unsigned int _vec_load_43[1];
            {
                _vec_load_43[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_43[0];
            unsigned int _vec_load_44[1];
            {
                _vec_load_44[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_44[0];
        }
        float _vec_load_45[4];
        {
            uint2 _vld_18;
            _vld_18 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_45[0 + _pair * 2])[0]), "=f"((&_vec_load_45[0 + _pair * 2])[1])
                    : "r"(_vpairs_18[_pair]));
            }
        }
        float _vec_load_46[4];
        {
            uint2 _vld_19;
            _vld_19 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_46[0 + _pair * 2])[0]), "=f"((&_vec_load_46[0 + _pair * 2])[1])
                    : "r"(_vpairs_19[_pair]));
            }
        }
        q[24] = _vec_load_45[0] * _vec_load_46[0];
        q[25] = _vec_load_45[1] * _vec_load_46[1];
        q[26] = _vec_load_45[2] * _vec_load_46[2];
        q[27] = _vec_load_45[3] * _vec_load_46[3];
        float _vec_load_47[4];
        {
            uint2 _vld_20;
            _vld_20 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_47[0 + _pair * 2])[0]), "=f"((&_vec_load_47[0 + _pair * 2])[1])
                    : "r"(_vpairs_20[_pair]));
            }
        }
        wout[24] = _vec_load_47[0];
        wout[25] = _vec_load_47[1];
        wout[26] = _vec_load_47[2];
        wout[27] = _vec_load_47[3];
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
        float2 sq[2];
        float2 dot[2];
        float fsrc[((1) ? 56 : 1)];
        float2 _f2_0 = make_float2(0.0f, 0.0f);
        sq[0] = _f2_0;
        float2 _f2_1 = make_float2(0.0f, 0.0f);
        dot[0] = _f2_1;
        float2 _f2_2 = make_float2(0.0f, 0.0f);
        sq[1] = _f2_2;
        float2 _f2_3 = make_float2(0.0f, 0.0f);
        dot[1] = _f2_3;
        int base_6 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        const int woff_7 = 0;
        unsigned int sw[4];
        sw[0] = words[woff_7];
        sw[1] = words[woff_7 + 1];
        sw[2] = words[woff_7 + 2];
        sw[3] = words[woff_7 + 3];
        float sw_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_f32[_pair * 2])[0]), "=f"((&sw_f32[_pair * 2])[1])
                : "r"(sw[_pair]));
        }
        {
            fsrc[0] = sw_f32[0];
            fsrc[1] = sw_f32[1];
            fsrc[2] = sw_f32[2];
            fsrc[3] = sw_f32[3];
            fsrc[4] = sw_f32[4];
            fsrc[5] = sw_f32[5];
            fsrc[6] = sw_f32[6];
            fsrc[7] = sw_f32[7];
        }
        {
            float2 _f2_10 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_10;
            float2 _f2_11 = make_float2(q[0], q[1]);
            float2 qp = _f2_11;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_12 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_12;
            float2 _f2_13 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_13;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_14 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_14;
            float2 _f2_15 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_15;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_16 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_16;
            float2 _f2_17 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_17;
            sq[0] = fma_f32x2_rn_noftz(v_4, v_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4, qp_5, dot[0]);
        }
        unsigned int sw_8[4];
        sw_8[0] = words[14 + woff_7];
        sw_8[1] = words[14 + woff_7 + 1];
        sw_8[2] = words[14 + woff_7 + 2];
        sw_8[3] = words[14 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_8[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_8[0] = __as_u32(mixed);
            words[14 + woff_7] = sw_8[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_8[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_8[1] = __as_u32(mixed_2);
            words[14 + woff_7 + 1] = sw_8[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_8[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_8[2] = __as_u32(mixed_5);
            words[14 + woff_7 + 2] = sw_8[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_8[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_8[3] = __as_u32(mixed_8);
            words[14 + woff_7 + 3] = sw_8[3];
            {
                int4 _iv4 = make_int4(sw_8[0 + 0], sw_8[0 + 1], sw_8[0 + 2], sw_8[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
        float sw_8_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_8_f32[_pair * 2])[0]), "=f"((&sw_8_f32[_pair * 2])[1])
                : "r"(sw_8[_pair]));
        }
        {
            fsrc[28] = sw_8_f32[0];
            fsrc[29] = sw_8_f32[1];
            fsrc[30] = sw_8_f32[2];
            fsrc[31] = sw_8_f32[3];
            fsrc[32] = sw_8_f32[4];
            fsrc[33] = sw_8_f32[5];
            fsrc[34] = sw_8_f32[6];
            fsrc[35] = sw_8_f32[7];
        }
        {
            float2 _f2_24 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_24;
            float2 _f2_25 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_25;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_26 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_26;
            float2 _f2_27 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_27;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_28 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_28;
            float2 _f2_29 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_29;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_30 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_30;
            float2 _f2_31 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_31;
            sq[1] = fma_f32x2_rn_noftz(v_4_1, v_4_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_1, qp_5_1, dot[1]);
        }
        int base_9 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_10 = 4;
        unsigned int sw_11[4];
        sw_11[0] = words[woff_10];
        sw_11[1] = words[woff_10 + 1];
        sw_11[2] = words[woff_10 + 2];
        sw_11[3] = words[woff_10 + 3];
        float sw_11_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_11_f32[_pair * 2])[0]), "=f"((&sw_11_f32[_pair * 2])[1])
                : "r"(sw_11[_pair]));
        }
        {
            fsrc[8] = sw_11_f32[0];
            fsrc[9] = sw_11_f32[1];
            fsrc[10] = sw_11_f32[2];
            fsrc[11] = sw_11_f32[3];
            fsrc[12] = sw_11_f32[4];
            fsrc[13] = sw_11_f32[5];
            fsrc[14] = sw_11_f32[6];
            fsrc[15] = sw_11_f32[7];
        }
        {
            float2 _f2_38 = make_float2(sw_11_f32[0], sw_11_f32[1]);
            float2 v_3 = _f2_38;
            float2 _f2_39 = make_float2(q[8], q[9]);
            float2 qp_4 = _f2_39;
            sq[0] = fma_f32x2_rn_noftz(v_3, v_3, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_3, qp_4, dot[0]);
            float2 _f2_40 = make_float2(sw_11_f32[2], sw_11_f32[3]);
            float2 v_0_2 = _f2_40;
            float2 _f2_41 = make_float2(q[10], q[11]);
            float2 qp_1_2 = _f2_41;
            sq[0] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[0]);
            float2 _f2_42 = make_float2(sw_11_f32[4], sw_11_f32[5]);
            float2 v_2_2 = _f2_42;
            float2 _f2_43 = make_float2(q[12], q[13]);
            float2 qp_3_2 = _f2_43;
            sq[0] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[0]);
            float2 _f2_44 = make_float2(sw_11_f32[6], sw_11_f32[7]);
            float2 v_4_2 = _f2_44;
            float2 _f2_45 = make_float2(q[14], q[15]);
            float2 qp_5_2 = _f2_45;
            sq[0] = fma_f32x2_rn_noftz(v_4_2, v_4_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_2, qp_5_2, dot[0]);
        }
        unsigned int sw_12[4];
        sw_12[0] = words[14 + woff_10];
        sw_12[1] = words[14 + woff_10 + 1];
        sw_12[2] = words[14 + woff_10 + 2];
        sw_12[3] = words[14 + woff_10 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_12[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_10]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_12[0] = __as_u32(mixed_1);
            words[14 + woff_10] = sw_12[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_12[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_10 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_12[1] = __as_u32(mixed_2_1);
            words[14 + woff_10 + 1] = sw_12[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_12[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_10 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_12[2] = __as_u32(mixed_5_1);
            words[14 + woff_10 + 2] = sw_12[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_12[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_10 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_12[3] = __as_u32(mixed_8_1);
            words[14 + woff_10 + 3] = sw_12[3];
            {
                int4 _iv4 = make_int4(sw_12[0 + 0], sw_12[0 + 1], sw_12[0 + 2], sw_12[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_9)) + 0) = _iv4;
            }
        }
        float sw_12_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_12_f32[_pair * 2])[0]), "=f"((&sw_12_f32[_pair * 2])[1])
                : "r"(sw_12[_pair]));
        }
        {
            fsrc[36] = sw_12_f32[0];
            fsrc[37] = sw_12_f32[1];
            fsrc[38] = sw_12_f32[2];
            fsrc[39] = sw_12_f32[3];
            fsrc[40] = sw_12_f32[4];
            fsrc[41] = sw_12_f32[5];
            fsrc[42] = sw_12_f32[6];
            fsrc[43] = sw_12_f32[7];
        }
        {
            float2 _f2_52 = make_float2(sw_12_f32[0], sw_12_f32[1]);
            float2 v_5 = _f2_52;
            float2 _f2_53 = make_float2(q[8], q[9]);
            float2 qp_6 = _f2_53;
            sq[1] = fma_f32x2_rn_noftz(v_5, v_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_5, qp_6, dot[1]);
            float2 _f2_54 = make_float2(sw_12_f32[2], sw_12_f32[3]);
            float2 v_0_3 = _f2_54;
            float2 _f2_55 = make_float2(q[10], q[11]);
            float2 qp_1_3 = _f2_55;
            sq[1] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[1]);
            float2 _f2_56 = make_float2(sw_12_f32[4], sw_12_f32[5]);
            float2 v_2_3 = _f2_56;
            float2 _f2_57 = make_float2(q[12], q[13]);
            float2 qp_3_3 = _f2_57;
            sq[1] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[1]);
            float2 _f2_58 = make_float2(sw_12_f32[6], sw_12_f32[7]);
            float2 v_4_3 = _f2_58;
            float2 _f2_59 = make_float2(q[14], q[15]);
            float2 qp_5_3 = _f2_59;
            sq[1] = fma_f32x2_rn_noftz(v_4_3, v_4_3, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_3, qp_5_3, dot[1]);
        }
        int base_13 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_14 = 8;
        unsigned int sw_15[4];
        sw_15[0] = words[woff_14];
        sw_15[1] = words[woff_14 + 1];
        sw_15[2] = words[woff_14 + 2];
        sw_15[3] = words[woff_14 + 3];
        float sw_15_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_15_f32[_pair * 2])[0]), "=f"((&sw_15_f32[_pair * 2])[1])
                : "r"(sw_15[_pair]));
        }
        {
            fsrc[16] = sw_15_f32[0];
            fsrc[17] = sw_15_f32[1];
            fsrc[18] = sw_15_f32[2];
            fsrc[19] = sw_15_f32[3];
            fsrc[20] = sw_15_f32[4];
            fsrc[21] = sw_15_f32[5];
            fsrc[22] = sw_15_f32[6];
            fsrc[23] = sw_15_f32[7];
        }
        {
            float2 _f2_66 = make_float2(sw_15_f32[0], sw_15_f32[1]);
            float2 v_6 = _f2_66;
            float2 _f2_67 = make_float2(q[16], q[17]);
            float2 qp_7 = _f2_67;
            sq[0] = fma_f32x2_rn_noftz(v_6, v_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_6, qp_7, dot[0]);
            float2 _f2_68 = make_float2(sw_15_f32[2], sw_15_f32[3]);
            float2 v_0_4 = _f2_68;
            float2 _f2_69 = make_float2(q[18], q[19]);
            float2 qp_1_4 = _f2_69;
            sq[0] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[0]);
            float2 _f2_70 = make_float2(sw_15_f32[4], sw_15_f32[5]);
            float2 v_2_4 = _f2_70;
            float2 _f2_71 = make_float2(q[20], q[21]);
            float2 qp_3_4 = _f2_71;
            sq[0] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[0]);
            float2 _f2_72 = make_float2(sw_15_f32[6], sw_15_f32[7]);
            float2 v_4_4 = _f2_72;
            float2 _f2_73 = make_float2(q[22], q[23]);
            float2 qp_5_4 = _f2_73;
            sq[0] = fma_f32x2_rn_noftz(v_4_4, v_4_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_4, qp_5_4, dot[0]);
        }
        unsigned int sw_16[4];
        sw_16[0] = words[14 + woff_14];
        sw_16[1] = words[14 + woff_14 + 1];
        sw_16[2] = words[14 + woff_14 + 2];
        sw_16[3] = words[14 + woff_14 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_16[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_14]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_16[0] = __as_u32(mixed_3);
            words[14 + woff_14] = sw_16[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_16[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_14 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_16[1] = __as_u32(mixed_2_2);
            words[14 + woff_14 + 1] = sw_16[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_16[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_14 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_16[2] = __as_u32(mixed_5_2);
            words[14 + woff_14 + 2] = sw_16[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_16[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_14 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_16[3] = __as_u32(mixed_8_2);
            words[14 + woff_14 + 3] = sw_16[3];
            {
                int4 _iv4 = make_int4(sw_16[0 + 0], sw_16[0 + 1], sw_16[0 + 2], sw_16[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_13)) + 0) = _iv4;
            }
        }
        float sw_16_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_16_f32[_pair * 2])[0]), "=f"((&sw_16_f32[_pair * 2])[1])
                : "r"(sw_16[_pair]));
        }
        {
            fsrc[44] = sw_16_f32[0];
            fsrc[45] = sw_16_f32[1];
            fsrc[46] = sw_16_f32[2];
            fsrc[47] = sw_16_f32[3];
            fsrc[48] = sw_16_f32[4];
            fsrc[49] = sw_16_f32[5];
            fsrc[50] = sw_16_f32[6];
            fsrc[51] = sw_16_f32[7];
        }
        {
            float2 _f2_80 = make_float2(sw_16_f32[0], sw_16_f32[1]);
            float2 v_7 = _f2_80;
            float2 _f2_81 = make_float2(q[16], q[17]);
            float2 qp_8 = _f2_81;
            sq[1] = fma_f32x2_rn_noftz(v_7, v_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_7, qp_8, dot[1]);
            float2 _f2_82 = make_float2(sw_16_f32[2], sw_16_f32[3]);
            float2 v_0_5 = _f2_82;
            float2 _f2_83 = make_float2(q[18], q[19]);
            float2 qp_1_5 = _f2_83;
            sq[1] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[1]);
            float2 _f2_84 = make_float2(sw_16_f32[4], sw_16_f32[5]);
            float2 v_2_5 = _f2_84;
            float2 _f2_85 = make_float2(q[20], q[21]);
            float2 qp_3_5 = _f2_85;
            sq[1] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[1]);
            float2 _f2_86 = make_float2(sw_16_f32[6], sw_16_f32[7]);
            float2 v_4_5 = _f2_86;
            float2 _f2_87 = make_float2(q[22], q[23]);
            float2 qp_5_5 = _f2_87;
            sq[1] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[1]);
        }
        int base_17 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_18 = 12;
        unsigned int sw_19[4];
        sw_19[0] = words[woff_18];
        sw_19[1] = words[woff_18 + 1];
        float sw_19_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_19_f32[_pair * 2])[0]), "=f"((&sw_19_f32[_pair * 2])[1])
                : "r"(sw_19[_pair]));
        }
        {
            fsrc[24] = sw_19_f32[0];
            fsrc[25] = sw_19_f32[1];
            fsrc[26] = sw_19_f32[2];
            fsrc[27] = sw_19_f32[3];
        }
        {
            float2 _f2_88 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_8 = _f2_88;
            sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
            float2 _f2_89 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_6 = _f2_89;
            sq[0] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[0]);
            float2 _f2_90 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_1_1 = _f2_90;
            float2 _f2_91 = make_float2(q[24], q[25]);
            float2 qp_9 = _f2_91;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_9, dot[0]);
            float2 _f2_92 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_2_6 = _f2_92;
            float2 _f2_93 = make_float2(q[26], q[27]);
            float2 qp_3_6 = _f2_93;
            dot[0] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[0]);
        }
        unsigned int sw_20[4];
        sw_20[0] = words[14 + woff_18];
        sw_20[1] = words[14 + woff_18 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_20[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_18]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_20[0] = __as_u32(mixed_4);
            words[14 + woff_18] = sw_20[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_20[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_18 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_20[1] = __as_u32(mixed_2_3);
            words[14 + woff_18 + 1] = sw_20[1];
            {
                int2 _iv2 = make_int2(sw_20[0 + 0], sw_20[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_17)) + 0) = _iv2;
            }
        }
        float sw_20_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_20_f32[_pair * 2])[0]), "=f"((&sw_20_f32[_pair * 2])[1])
                : "r"(sw_20[_pair]));
        }
        {
            fsrc[52] = sw_20_f32[0];
            fsrc[53] = sw_20_f32[1];
            fsrc[54] = sw_20_f32[2];
            fsrc[55] = sw_20_f32[3];
        }
        {
            float2 _f2_102 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_9 = _f2_102;
            sq[1] = fma_f32x2_rn_noftz(v_9, v_9, sq[1]);
            float2 _f2_103 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_0_7 = _f2_103;
            sq[1] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[1]);
            float2 _f2_104 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_1_2 = _f2_104;
            float2 _f2_105 = make_float2(q[24], q[25]);
            float2 qp_10 = _f2_105;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_10, dot[1]);
            float2 _f2_106 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_2_7 = _f2_106;
            float2 _f2_107 = make_float2(q[26], q[27]);
            float2 qp_3_7 = _f2_107;
            dot[1] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[1]);
        }
        float2 pairs[2];
        float2 _f2_116 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_116;
        float2 _f2_117 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_117;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_118 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_118;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_21 = 0;
        bits_21 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_21, 16);
        unsigned long long peerbits_22 = _shfl_xor_1;
        float2 _f2_119 = make_float2(0.0f, 0.0f);
        float2 peer_23 = _f2_119;
        peer_23 = reinterpret_cast<float2*>(&peerbits_22)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_23);
        unsigned long long bits_24 = 0;
        bits_24 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_24, 8);
        unsigned long long peerbits_25 = _shfl_xor_2;
        float2 _f2_120 = make_float2(0.0f, 0.0f);
        float2 peer_26 = _f2_120;
        peer_26 = reinterpret_cast<float2*>(&peerbits_25)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_26);
        unsigned long long bits_27 = 0;
        bits_27 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_27, 8);
        unsigned long long peerbits_28 = _shfl_xor_3;
        float2 _f2_121 = make_float2(0.0f, 0.0f);
        float2 peer_29 = _f2_121;
        peer_29 = reinterpret_cast<float2*>(&peerbits_28)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_29);
        unsigned long long bits_30 = 0;
        bits_30 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_30, 4);
        unsigned long long peerbits_31 = _shfl_xor_4;
        float2 _f2_122 = make_float2(0.0f, 0.0f);
        float2 peer_32 = _f2_122;
        peer_32 = reinterpret_cast<float2*>(&peerbits_31)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_32);
        unsigned long long bits_33 = 0;
        bits_33 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_33, 4);
        unsigned long long peerbits_34 = _shfl_xor_5;
        float2 _f2_123 = make_float2(0.0f, 0.0f);
        float2 peer_35 = _f2_123;
        peer_35 = reinterpret_cast<float2*>(&peerbits_34)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_35);
        unsigned long long bits_36 = 0;
        bits_36 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_36, 2);
        unsigned long long peerbits_37 = _shfl_xor_6;
        float2 _f2_124 = make_float2(0.0f, 0.0f);
        float2 peer_38 = _f2_124;
        peer_38 = reinterpret_cast<float2*>(&peerbits_37)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_38);
        unsigned long long bits_39 = 0;
        bits_39 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_39, 2);
        unsigned long long peerbits_40 = _shfl_xor_7;
        float2 _f2_125 = make_float2(0.0f, 0.0f);
        float2 peer_41 = _f2_125;
        peer_41 = reinterpret_cast<float2*>(&peerbits_40)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_41);
        unsigned long long bits_42 = 0;
        bits_42 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_42, 1);
        unsigned long long peerbits_43 = _shfl_xor_8;
        float2 _f2_126 = make_float2(0.0f, 0.0f);
        float2 peer_44 = _f2_126;
        peer_44 = reinterpret_cast<float2*>(&peerbits_43)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_44);
        unsigned long long bits_45 = 0;
        bits_45 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_45, 1);
        unsigned long long peerbits_46 = _shfl_xor_9;
        float2 _f2_127 = make_float2(0.0f, 0.0f);
        float2 peer_47 = _f2_127;
        peer_47 = reinterpret_cast<float2*>(&peerbits_46)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_47);
        if (lane == 0) {
            stats[warp_0 * 2 * 2] = pairs[0].x;
            stats[warp_0 * 2 * 2 + 1] = pairs[0].y;
            stats[(warp_0 * 2 + 1) * 2] = pairs[1].x;
            stats[(warp_0 * 2 + 1) * 2 + 1] = pairs[1].y;
        }
        __syncthreads();
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 2) {
            total_sq = stats[(stat_w * 2 + stat_n) * 2];
            total_dot = stats[(stat_w * 2 + stat_n) * 2 + 1];
        }
        float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_0;
        float _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_1;
        float _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_2;
        float _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_3;
        float _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_4;
        float _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_5;
        float logit = 0.0f;
        if (stat_n < 2 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[2];
        float _shfl_0;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(logit), "r"(0));
        logits[0] = _shfl_0;
        float _shfl_1;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(logit), "r"(8));
        logits[1] = _shfl_1;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        float max_running = -3.4028234663852886e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028234663852886e+38f;
        float _fmax_0 = fmaxf(max_chunk, logits[0]);
        max_chunk = _fmax_0;
        float _fmax_1 = fmaxf(max_chunk, logits[1]);
        max_chunk = _fmax_1;
        float _fmax_2 = fmaxf(max_running, max_chunk);
        float max_new = _fmax_2;
        float _exp2_0 = approx_exp2((max_running - max_new) * 1.4426950408889634f);
        float correction = _exp2_0;
        float weights[2];
        float sum_weights = 0.0f;
        float _exp2_1 = approx_exp2((logits[0] - max_new) * 1.4426950408889634f);
        weights[0] = _exp2_1;
        sum_weights += weights[0];
        float _exp2_2 = approx_exp2((logits[1] - max_new) * 1.4426950408889634f);
        weights[1] = _exp2_2;
        sum_weights += weights[1];
        float2 _f2_128 = make_float2(correction, correction);
        float2 corr = _f2_128;
        const int woff_48 = 0;
        int base_49 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_129 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_129;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_130 = make_float2(acc[2], acc[3]);
        float2 previous_50 = _f2_130;
        a_5[1] = mul_f32x2_noftz(previous_50, corr);
        float2 _f2_131 = make_float2(acc[4], acc[5]);
        float2 previous_51 = _f2_131;
        a_5[2] = mul_f32x2_noftz(previous_51, corr);
        float2 _f2_132 = make_float2(acc[6], acc[7]);
        float2 previous_52 = _f2_132;
        a_5[3] = mul_f32x2_noftz(previous_52, corr);
        float2 _f2_133 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_133;
        {
            float2 _f2_134 = make_float2(fsrc[0], fsrc[1]);
            float2 v_10 = _f2_134;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_10, a_5[0]);
            float2 _f2_135 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_8 = _f2_135;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_8, a_5[1]);
            float2 _f2_136 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_3 = _f2_136;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_3, a_5[2]);
            float2 _f2_137 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_8 = _f2_137;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_8, a_5[3]);
        }
        float2 _f2_142 = make_float2(weights[1], weights[1]);
        float2 weight_53 = _f2_142;
        {
            float2 _f2_143 = make_float2(fsrc[28], fsrc[29]);
            float2 v_11 = _f2_143;
            a_5[0] = fma_f32x2_rn_noftz(weight_53, v_11, a_5[0]);
            float2 _f2_144 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_9 = _f2_144;
            a_5[1] = fma_f32x2_rn_noftz(weight_53, v_0_9, a_5[1]);
            float2 _f2_145 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_4 = _f2_145;
            a_5[2] = fma_f32x2_rn_noftz(weight_53, v_1_4, a_5[2]);
            float2 _f2_146 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_9 = _f2_146;
            a_5[3] = fma_f32x2_rn_noftz(weight_53, v_2_9, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_54 = 4;
        int base_55 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_56[4];
        float2 _f2_151 = make_float2(acc[8], acc[9]);
        float2 previous_57 = _f2_151;
        a_56[0] = mul_f32x2_noftz(previous_57, corr);
        float2 _f2_152 = make_float2(acc[10], acc[11]);
        float2 previous_58 = _f2_152;
        a_56[1] = mul_f32x2_noftz(previous_58, corr);
        float2 _f2_153 = make_float2(acc[12], acc[13]);
        float2 previous_59 = _f2_153;
        a_56[2] = mul_f32x2_noftz(previous_59, corr);
        float2 _f2_154 = make_float2(acc[14], acc[15]);
        float2 previous_60 = _f2_154;
        a_56[3] = mul_f32x2_noftz(previous_60, corr);
        float2 _f2_155 = make_float2(weights[0], weights[0]);
        float2 weight_61 = _f2_155;
        {
            float2 _f2_156 = make_float2(fsrc[8], fsrc[9]);
            float2 v_12 = _f2_156;
            a_56[0] = fma_f32x2_rn_noftz(weight_61, v_12, a_56[0]);
            float2 _f2_157 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_10 = _f2_157;
            a_56[1] = fma_f32x2_rn_noftz(weight_61, v_0_10, a_56[1]);
            float2 _f2_158 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_5 = _f2_158;
            a_56[2] = fma_f32x2_rn_noftz(weight_61, v_1_5, a_56[2]);
            float2 _f2_159 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_10 = _f2_159;
            a_56[3] = fma_f32x2_rn_noftz(weight_61, v_2_10, a_56[3]);
        }
        float2 _f2_164 = make_float2(weights[1], weights[1]);
        float2 weight_62 = _f2_164;
        {
            float2 _f2_165 = make_float2(fsrc[36], fsrc[37]);
            float2 v_13 = _f2_165;
            a_56[0] = fma_f32x2_rn_noftz(weight_62, v_13, a_56[0]);
            float2 _f2_166 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_11 = _f2_166;
            a_56[1] = fma_f32x2_rn_noftz(weight_62, v_0_11, a_56[1]);
            float2 _f2_167 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_6 = _f2_167;
            a_56[2] = fma_f32x2_rn_noftz(weight_62, v_1_6, a_56[2]);
            float2 _f2_168 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_11 = _f2_168;
            a_56[3] = fma_f32x2_rn_noftz(weight_62, v_2_11, a_56[3]);
        }
        acc[8] = a_56[0].x;
        acc[9] = a_56[0].y;
        acc[10] = a_56[1].x;
        acc[11] = a_56[1].y;
        acc[12] = a_56[2].x;
        acc[13] = a_56[2].y;
        acc[14] = a_56[3].x;
        acc[15] = a_56[3].y;
        const int woff_63 = 8;
        int base_64 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_65[4];
        float2 _f2_173 = make_float2(acc[16], acc[17]);
        float2 previous_66 = _f2_173;
        a_65[0] = mul_f32x2_noftz(previous_66, corr);
        float2 _f2_174 = make_float2(acc[18], acc[19]);
        float2 previous_67 = _f2_174;
        a_65[1] = mul_f32x2_noftz(previous_67, corr);
        float2 _f2_175 = make_float2(acc[20], acc[21]);
        float2 previous_68 = _f2_175;
        a_65[2] = mul_f32x2_noftz(previous_68, corr);
        float2 _f2_176 = make_float2(acc[22], acc[23]);
        float2 previous_69 = _f2_176;
        a_65[3] = mul_f32x2_noftz(previous_69, corr);
        float2 _f2_177 = make_float2(weights[0], weights[0]);
        float2 weight_70 = _f2_177;
        {
            float2 _f2_178 = make_float2(fsrc[16], fsrc[17]);
            float2 v_14 = _f2_178;
            a_65[0] = fma_f32x2_rn_noftz(weight_70, v_14, a_65[0]);
            float2 _f2_179 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_12 = _f2_179;
            a_65[1] = fma_f32x2_rn_noftz(weight_70, v_0_12, a_65[1]);
            float2 _f2_180 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_7 = _f2_180;
            a_65[2] = fma_f32x2_rn_noftz(weight_70, v_1_7, a_65[2]);
            float2 _f2_181 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_12 = _f2_181;
            a_65[3] = fma_f32x2_rn_noftz(weight_70, v_2_12, a_65[3]);
        }
        float2 _f2_186 = make_float2(weights[1], weights[1]);
        float2 weight_71 = _f2_186;
        {
            float2 _f2_187 = make_float2(fsrc[44], fsrc[45]);
            float2 v_15 = _f2_187;
            a_65[0] = fma_f32x2_rn_noftz(weight_71, v_15, a_65[0]);
            float2 _f2_188 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_13 = _f2_188;
            a_65[1] = fma_f32x2_rn_noftz(weight_71, v_0_13, a_65[1]);
            float2 _f2_189 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_8 = _f2_189;
            a_65[2] = fma_f32x2_rn_noftz(weight_71, v_1_8, a_65[2]);
            float2 _f2_190 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_13 = _f2_190;
            a_65[3] = fma_f32x2_rn_noftz(weight_71, v_2_13, a_65[3]);
        }
        acc[16] = a_65[0].x;
        acc[17] = a_65[0].y;
        acc[18] = a_65[1].x;
        acc[19] = a_65[1].y;
        acc[20] = a_65[2].x;
        acc[21] = a_65[2].y;
        acc[22] = a_65[3].x;
        acc[23] = a_65[3].y;
        const int woff_72 = 12;
        int base_73 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_74[4];
        float2 _f2_195 = make_float2(acc[24], acc[25]);
        float2 previous_75 = _f2_195;
        a_74[0] = mul_f32x2_noftz(previous_75, corr);
        float2 _f2_196 = make_float2(acc[26], acc[27]);
        float2 previous_76 = _f2_196;
        a_74[1] = mul_f32x2_noftz(previous_76, corr);
        float2 _f2_197 = make_float2(weights[0], weights[0]);
        float2 weight_77 = _f2_197;
        {
            float2 _f2_198 = make_float2(fsrc[24], fsrc[25]);
            float2 v_16 = _f2_198;
            a_74[0] = fma_f32x2_rn_noftz(weight_77, v_16, a_74[0]);
            float2 _f2_199 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_14 = _f2_199;
            a_74[1] = fma_f32x2_rn_noftz(weight_77, v_0_14, a_74[1]);
        }
        float2 _f2_202 = make_float2(weights[1], weights[1]);
        float2 weight_78 = _f2_202;
        {
            float2 _f2_203 = make_float2(fsrc[52], fsrc[53]);
            float2 v_17 = _f2_203;
            a_74[0] = fma_f32x2_rn_noftz(weight_78, v_17, a_74[0]);
            float2 _f2_204 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_15 = _f2_204;
            a_74[1] = fma_f32x2_rn_noftz(weight_78, v_0_15, a_74[1]);
        }
        acc[24] = a_74[0].x;
        acc[25] = a_74[0].y;
        acc[26] = a_74[1].x;
        acc[27] = a_74[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float2 _f2_207 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_207;
        float2 _f2_208 = make_float2(acc[0], acc[1]);
        float2 v_18 = _f2_208;
        output_sq_pair = fma_f32x2_rn_noftz(v_18, v_18, output_sq_pair);
        float2 _f2_209 = make_float2(acc[2], acc[3]);
        float2 v_79 = _f2_209;
        output_sq_pair = fma_f32x2_rn_noftz(v_79, v_79, output_sq_pair);
        float2 _f2_210 = make_float2(acc[4], acc[5]);
        float2 v_80 = _f2_210;
        output_sq_pair = fma_f32x2_rn_noftz(v_80, v_80, output_sq_pair);
        float2 _f2_211 = make_float2(acc[6], acc[7]);
        float2 v_81 = _f2_211;
        output_sq_pair = fma_f32x2_rn_noftz(v_81, v_81, output_sq_pair);
        float2 _f2_212 = make_float2(acc[8], acc[9]);
        float2 v_82 = _f2_212;
        output_sq_pair = fma_f32x2_rn_noftz(v_82, v_82, output_sq_pair);
        float2 _f2_213 = make_float2(acc[10], acc[11]);
        float2 v_83 = _f2_213;
        output_sq_pair = fma_f32x2_rn_noftz(v_83, v_83, output_sq_pair);
        float2 _f2_214 = make_float2(acc[12], acc[13]);
        float2 v_84 = _f2_214;
        output_sq_pair = fma_f32x2_rn_noftz(v_84, v_84, output_sq_pair);
        float2 _f2_215 = make_float2(acc[14], acc[15]);
        float2 v_85 = _f2_215;
        output_sq_pair = fma_f32x2_rn_noftz(v_85, v_85, output_sq_pair);
        float2 _f2_216 = make_float2(acc[16], acc[17]);
        float2 v_86 = _f2_216;
        output_sq_pair = fma_f32x2_rn_noftz(v_86, v_86, output_sq_pair);
        float2 _f2_217 = make_float2(acc[18], acc[19]);
        float2 v_87 = _f2_217;
        output_sq_pair = fma_f32x2_rn_noftz(v_87, v_87, output_sq_pair);
        float2 _f2_218 = make_float2(acc[20], acc[21]);
        float2 v_88 = _f2_218;
        output_sq_pair = fma_f32x2_rn_noftz(v_88, v_88, output_sq_pair);
        float2 _f2_219 = make_float2(acc[22], acc[23]);
        float2 v_89 = _f2_219;
        output_sq_pair = fma_f32x2_rn_noftz(v_89, v_89, output_sq_pair);
        float2 _f2_220 = make_float2(acc[24], acc[25]);
        float2 v_90 = _f2_220;
        output_sq_pair = fma_f32x2_rn_noftz(v_90, v_90, output_sq_pair);
        float2 _f2_221 = make_float2(acc[26], acc[27]);
        float2 v_91 = _f2_221;
        output_sq_pair = fma_f32x2_rn_noftz(v_91, v_91, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_10;
        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_11;
        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_12;
        float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_13;
        float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_14;
        if (lane == 0) {
            out_stats[warp_0] = output_sq;
        }
        __syncthreads();
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_6;
        float _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_7;
        float _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_8;
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_1 = rsqrtf(output_total / 7168.0f + output_norm_eps * sum_running * sum_running);
            rsigma_lane = _rsqrt_1;
        }
        float _shfl_2;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(rsigma_lane), "r"(0));
        float rsigma = _shfl_2;
        float2 _f2_222 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_222;
        int base_92 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_223 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_223, rsigma_pair);
        float2 _f2_224 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_224);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_93 = 2;
        const int acc_idx_94 = value_idx_93;
        float2 _f2_225 = make_float2(acc[acc_idx_94], acc[acc_idx_94 + 1]);
        float2 scaled_pair_95 = mul_f32x2_noftz(_f2_225, rsigma_pair);
        float2 _f2_226 = make_float2(wout[acc_idx_94], wout[acc_idx_94 + 1]);
        float2 normalized_pair_96 = mul_f32x2_noftz(scaled_pair_95, _f2_226);
        output_values[value_idx_93] = normalized_pair_96.x;
        output_values[value_idx_93 + 1] = normalized_pair_96.y;
        const int value_idx_97 = 4;
        const int acc_idx_98 = value_idx_97;
        float2 _f2_227 = make_float2(acc[acc_idx_98], acc[acc_idx_98 + 1]);
        float2 scaled_pair_99 = mul_f32x2_noftz(_f2_227, rsigma_pair);
        float2 _f2_228 = make_float2(wout[acc_idx_98], wout[acc_idx_98 + 1]);
        float2 normalized_pair_100 = mul_f32x2_noftz(scaled_pair_99, _f2_228);
        output_values[value_idx_97] = normalized_pair_100.x;
        output_values[value_idx_97 + 1] = normalized_pair_100.y;
        const int value_idx_101 = 6;
        const int acc_idx_102 = value_idx_101;
        float2 _f2_229 = make_float2(acc[acc_idx_102], acc[acc_idx_102 + 1]);
        float2 scaled_pair_103 = mul_f32x2_noftz(_f2_229, rsigma_pair);
        float2 _f2_230 = make_float2(wout[acc_idx_102], wout[acc_idx_102 + 1]);
        float2 normalized_pair_104 = mul_f32x2_noftz(scaled_pair_103, _f2_230);
        output_values[value_idx_101] = normalized_pair_104.x;
        output_values[value_idx_101 + 1] = normalized_pair_104.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_92 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_105 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_106[8];
        const int value_idx_107 = 0;
        const int acc_idx_108 = 8 + value_idx_107;
        float2 _f2_231 = make_float2(acc[acc_idx_108], acc[acc_idx_108 + 1]);
        float2 scaled_pair_109 = mul_f32x2_noftz(_f2_231, rsigma_pair);
        float2 _f2_232 = make_float2(wout[acc_idx_108], wout[acc_idx_108 + 1]);
        float2 normalized_pair_110 = mul_f32x2_noftz(scaled_pair_109, _f2_232);
        output_values_106[value_idx_107] = normalized_pair_110.x;
        output_values_106[value_idx_107 + 1] = normalized_pair_110.y;
        const int value_idx_111 = 2;
        const int acc_idx_112 = 8 + value_idx_111;
        float2 _f2_233 = make_float2(acc[acc_idx_112], acc[acc_idx_112 + 1]);
        float2 scaled_pair_113 = mul_f32x2_noftz(_f2_233, rsigma_pair);
        float2 _f2_234 = make_float2(wout[acc_idx_112], wout[acc_idx_112 + 1]);
        float2 normalized_pair_114 = mul_f32x2_noftz(scaled_pair_113, _f2_234);
        output_values_106[value_idx_111] = normalized_pair_114.x;
        output_values_106[value_idx_111 + 1] = normalized_pair_114.y;
        const int value_idx_115 = 4;
        const int acc_idx_116 = 8 + value_idx_115;
        float2 _f2_235 = make_float2(acc[acc_idx_116], acc[acc_idx_116 + 1]);
        float2 scaled_pair_117 = mul_f32x2_noftz(_f2_235, rsigma_pair);
        float2 _f2_236 = make_float2(wout[acc_idx_116], wout[acc_idx_116 + 1]);
        float2 normalized_pair_118 = mul_f32x2_noftz(scaled_pair_117, _f2_236);
        output_values_106[value_idx_115] = normalized_pair_118.x;
        output_values_106[value_idx_115 + 1] = normalized_pair_118.y;
        const int value_idx_119 = 6;
        const int acc_idx_120 = 8 + value_idx_119;
        float2 _f2_237 = make_float2(acc[acc_idx_120], acc[acc_idx_120 + 1]);
        float2 scaled_pair_121 = mul_f32x2_noftz(_f2_237, rsigma_pair);
        float2 _f2_238 = make_float2(wout[acc_idx_120], wout[acc_idx_120 + 1]);
        float2 normalized_pair_122 = mul_f32x2_noftz(scaled_pair_121, _f2_238);
        output_values_106[value_idx_119] = normalized_pair_122.x;
        output_values_106[value_idx_119 + 1] = normalized_pair_122.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_106[0 + 0], output_values_106[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_106[0 + 2], output_values_106[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_106[0 + 4], output_values_106[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_106[0 + 6], output_values_106[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_105 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_123 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_124[8];
        const int value_idx_125 = 0;
        const int acc_idx_126 = 16 + value_idx_125;
        float2 _f2_239 = make_float2(acc[acc_idx_126], acc[acc_idx_126 + 1]);
        float2 scaled_pair_127 = mul_f32x2_noftz(_f2_239, rsigma_pair);
        float2 _f2_240 = make_float2(wout[acc_idx_126], wout[acc_idx_126 + 1]);
        float2 normalized_pair_128 = mul_f32x2_noftz(scaled_pair_127, _f2_240);
        output_values_124[value_idx_125] = normalized_pair_128.x;
        output_values_124[value_idx_125 + 1] = normalized_pair_128.y;
        const int value_idx_129 = 2;
        const int acc_idx_130 = 16 + value_idx_129;
        float2 _f2_241 = make_float2(acc[acc_idx_130], acc[acc_idx_130 + 1]);
        float2 scaled_pair_131 = mul_f32x2_noftz(_f2_241, rsigma_pair);
        float2 _f2_242 = make_float2(wout[acc_idx_130], wout[acc_idx_130 + 1]);
        float2 normalized_pair_132 = mul_f32x2_noftz(scaled_pair_131, _f2_242);
        output_values_124[value_idx_129] = normalized_pair_132.x;
        output_values_124[value_idx_129 + 1] = normalized_pair_132.y;
        const int value_idx_133 = 4;
        const int acc_idx_134 = 16 + value_idx_133;
        float2 _f2_243 = make_float2(acc[acc_idx_134], acc[acc_idx_134 + 1]);
        float2 scaled_pair_135 = mul_f32x2_noftz(_f2_243, rsigma_pair);
        float2 _f2_244 = make_float2(wout[acc_idx_134], wout[acc_idx_134 + 1]);
        float2 normalized_pair_136 = mul_f32x2_noftz(scaled_pair_135, _f2_244);
        output_values_124[value_idx_133] = normalized_pair_136.x;
        output_values_124[value_idx_133 + 1] = normalized_pair_136.y;
        const int value_idx_137 = 6;
        const int acc_idx_138 = 16 + value_idx_137;
        float2 _f2_245 = make_float2(acc[acc_idx_138], acc[acc_idx_138 + 1]);
        float2 scaled_pair_139 = mul_f32x2_noftz(_f2_245, rsigma_pair);
        float2 _f2_246 = make_float2(wout[acc_idx_138], wout[acc_idx_138 + 1]);
        float2 normalized_pair_140 = mul_f32x2_noftz(scaled_pair_139, _f2_246);
        output_values_124[value_idx_137] = normalized_pair_140.x;
        output_values_124[value_idx_137 + 1] = normalized_pair_140.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_124[0 + 0], output_values_124[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_124[0 + 2], output_values_124[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_124[0 + 4], output_values_124[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_124[0 + 6], output_values_124[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_123 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_141 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_142[8];
        const int value_idx_143 = 0;
        const int acc_idx_144 = 24 + value_idx_143;
        float2 _f2_247 = make_float2(acc[acc_idx_144], acc[acc_idx_144 + 1]);
        float2 scaled_pair_145 = mul_f32x2_noftz(_f2_247, rsigma_pair);
        float2 _f2_248 = make_float2(wout[acc_idx_144], wout[acc_idx_144 + 1]);
        float2 normalized_pair_146 = mul_f32x2_noftz(scaled_pair_145, _f2_248);
        output_values_142[value_idx_143] = normalized_pair_146.x;
        output_values_142[value_idx_143 + 1] = normalized_pair_146.y;
        const int value_idx_147 = 2;
        const int acc_idx_148 = 24 + value_idx_147;
        float2 _f2_249 = make_float2(acc[acc_idx_148], acc[acc_idx_148 + 1]);
        float2 scaled_pair_149 = mul_f32x2_noftz(_f2_249, rsigma_pair);
        float2 _f2_250 = make_float2(wout[acc_idx_148], wout[acc_idx_148 + 1]);
        float2 normalized_pair_150 = mul_f32x2_noftz(scaled_pair_149, _f2_250);
        output_values_142[value_idx_147] = normalized_pair_150.x;
        output_values_142[value_idx_147 + 1] = normalized_pair_150.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_142[0 + 0], output_values_142[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_142[0 + 2], output_values_142[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_141]) = _pk2;
        }
    }
}

} // extern "C"
