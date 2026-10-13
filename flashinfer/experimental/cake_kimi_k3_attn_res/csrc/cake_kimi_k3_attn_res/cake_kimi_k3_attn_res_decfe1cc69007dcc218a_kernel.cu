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
#define SMEM_STATS_STAGE_BYTES 192
#define SMEM_STATS_STRIDE 192
#define SMEM_OUT_STATS_OFF 192
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
kernel_cake_kimi_k3_attn_res_decfe1cc69007dcc218a(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 192);
    const int out_stats_addr = smem + 192;

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
        unsigned int words[42];
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
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_6[0 + 0] = _uv4_2.x;
                _vec_load_6[0 + 1] = _uv4_2.y;
                _vec_load_6[0 + 2] = _uv4_2.z;
                _vec_load_6[0 + 3] = _uv4_2.w;
            }
            words[28 + woff] = _vec_load_6[0];
            words[28 + woff + 1] = _vec_load_6[1];
            words[28 + woff + 2] = _vec_load_6[2];
            words[28 + woff + 3] = _vec_load_6[3];
        }
        {
            unsigned int _vec_load_9[4];
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_9[0 + 0] = _uv4_3.x;
                _vec_load_9[0 + 1] = _uv4_3.y;
                _vec_load_9[0 + 2] = _uv4_3.z;
                _vec_load_9[0 + 3] = _uv4_3.w;
            }
            dwords[woff] = _vec_load_9[0];
            dwords[woff + 1] = _vec_load_9[1];
            dwords[woff + 2] = _vec_load_9[2];
            dwords[woff + 3] = _vec_load_9[3];
        }
        float _vec_load_12[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        float _vec_load_13[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        q[0] = _vec_load_12[0] * _vec_load_13[0];
        q[1] = _vec_load_12[1] * _vec_load_13[1];
        q[2] = _vec_load_12[2] * _vec_load_13[2];
        q[3] = _vec_load_12[3] * _vec_load_13[3];
        q[4] = _vec_load_12[4] * _vec_load_13[4];
        q[5] = _vec_load_12[5] * _vec_load_13[5];
        q[6] = _vec_load_12[6] * _vec_load_13[6];
        q[7] = _vec_load_12[7] * _vec_load_13[7];
        float _vec_load_14[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(output_norm_weight + base);
            uint4 _vld_6[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_6[_blk] = _vptr_6[_blk];
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_14[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_14[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        wout[0] = _vec_load_14[0];
        wout[1] = _vec_load_14[1];
        wout[2] = _vec_load_14[2];
        wout[3] = _vec_load_14[3];
        wout[4] = _vec_load_14[4];
        wout[5] = _vec_load_14[5];
        wout[6] = _vec_load_14[6];
        wout[7] = _vec_load_14[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_15[4];
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_15[0 + 0] = _uv4_7.x;
                _vec_load_15[0 + 1] = _uv4_7.y;
                _vec_load_15[0 + 2] = _uv4_7.z;
                _vec_load_15[0 + 3] = _uv4_7.w;
            }
            words[woff_1] = _vec_load_15[0];
            words[woff_1 + 1] = _vec_load_15[1];
            words[woff_1 + 2] = _vec_load_15[2];
            words[woff_1 + 3] = _vec_load_15[3];
        }
        {
            unsigned int _vec_load_18[4];
            {
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_18[0 + 0] = _uv4_8.x;
                _vec_load_18[0 + 1] = _uv4_8.y;
                _vec_load_18[0 + 2] = _uv4_8.z;
                _vec_load_18[0 + 3] = _uv4_8.w;
            }
            words[14 + woff_1] = _vec_load_18[0];
            words[14 + woff_1 + 1] = _vec_load_18[1];
            words[14 + woff_1 + 2] = _vec_load_18[2];
            words[14 + woff_1 + 3] = _vec_load_18[3];
        }
        {
            unsigned int _vec_load_21[4];
            {
                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_21[0 + 0] = _uv4_9.x;
                _vec_load_21[0 + 1] = _uv4_9.y;
                _vec_load_21[0 + 2] = _uv4_9.z;
                _vec_load_21[0 + 3] = _uv4_9.w;
            }
            words[28 + woff_1] = _vec_load_21[0];
            words[28 + woff_1 + 1] = _vec_load_21[1];
            words[28 + woff_1 + 2] = _vec_load_21[2];
            words[28 + woff_1 + 3] = _vec_load_21[3];
        }
        {
            unsigned int _vec_load_24[4];
            {
                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_24[0 + 0] = _uv4_10.x;
                _vec_load_24[0 + 1] = _uv4_10.y;
                _vec_load_24[0 + 2] = _uv4_10.z;
                _vec_load_24[0 + 3] = _uv4_10.w;
            }
            dwords[woff_1] = _vec_load_24[0];
            dwords[woff_1 + 1] = _vec_load_24[1];
            dwords[woff_1 + 2] = _vec_load_24[2];
            dwords[woff_1 + 3] = _vec_load_24[3];
        }
        float _vec_load_27[8];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_27[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_27[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        float _vec_load_28[8];
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(qk_weight + base_0);
            uint4 _vld_12[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_12[_blk] = _vptr_12[_blk];
                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_28[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_28[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_12[_pair]));
                }
            }
        }
        q[8] = _vec_load_27[0] * _vec_load_28[0];
        q[9] = _vec_load_27[1] * _vec_load_28[1];
        q[10] = _vec_load_27[2] * _vec_load_28[2];
        q[11] = _vec_load_27[3] * _vec_load_28[3];
        q[12] = _vec_load_27[4] * _vec_load_28[4];
        q[13] = _vec_load_27[5] * _vec_load_28[5];
        q[14] = _vec_load_27[6] * _vec_load_28[6];
        q[15] = _vec_load_27[7] * _vec_load_28[7];
        float _vec_load_29[8];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
            uint4 _vld_13[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_13[_blk] = _vptr_13[_blk];
                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_29[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_29[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_13[_pair]));
                }
            }
        }
        wout[8] = _vec_load_29[0];
        wout[9] = _vec_load_29[1];
        wout[10] = _vec_load_29[2];
        wout[11] = _vec_load_29[3];
        wout[12] = _vec_load_29[4];
        wout[13] = _vec_load_29[5];
        wout[14] = _vec_load_29[6];
        wout[15] = _vec_load_29[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_30[0 + 0] = _uv4_14.x;
                _vec_load_30[0 + 1] = _uv4_14.y;
                _vec_load_30[0 + 2] = _uv4_14.z;
                _vec_load_30[0 + 3] = _uv4_14.w;
            }
            words[woff_3] = _vec_load_30[0];
            words[woff_3 + 1] = _vec_load_30[1];
            words[woff_3 + 2] = _vec_load_30[2];
            words[woff_3 + 3] = _vec_load_30[3];
        }
        {
            unsigned int _vec_load_33[4];
            {
                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_33[0 + 0] = _uv4_15.x;
                _vec_load_33[0 + 1] = _uv4_15.y;
                _vec_load_33[0 + 2] = _uv4_15.z;
                _vec_load_33[0 + 3] = _uv4_15.w;
            }
            words[14 + woff_3] = _vec_load_33[0];
            words[14 + woff_3 + 1] = _vec_load_33[1];
            words[14 + woff_3 + 2] = _vec_load_33[2];
            words[14 + woff_3 + 3] = _vec_load_33[3];
        }
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_36[0 + 0] = _uv4_16.x;
                _vec_load_36[0 + 1] = _uv4_16.y;
                _vec_load_36[0 + 2] = _uv4_16.z;
                _vec_load_36[0 + 3] = _uv4_16.w;
            }
            words[28 + woff_3] = _vec_load_36[0];
            words[28 + woff_3 + 1] = _vec_load_36[1];
            words[28 + woff_3 + 2] = _vec_load_36[2];
            words[28 + woff_3 + 3] = _vec_load_36[3];
        }
        {
            unsigned int _vec_load_39[4];
            {
                uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_39[0 + 0] = _uv4_17.x;
                _vec_load_39[0 + 1] = _uv4_17.y;
                _vec_load_39[0 + 2] = _uv4_17.z;
                _vec_load_39[0 + 3] = _uv4_17.w;
            }
            dwords[woff_3] = _vec_load_39[0];
            dwords[woff_3 + 1] = _vec_load_39[1];
            dwords[woff_3 + 2] = _vec_load_39[2];
            dwords[woff_3 + 3] = _vec_load_39[3];
        }
        float _vec_load_42[8];
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_18[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_18[_blk] = _vptr_18[_blk];
                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_42[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_42[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_18[_pair]));
                }
            }
        }
        float _vec_load_43[8];
        {
            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_19[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_19[_blk] = _vptr_19[_blk];
                uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_43[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_43[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        q[16] = _vec_load_42[0] * _vec_load_43[0];
        q[17] = _vec_load_42[1] * _vec_load_43[1];
        q[18] = _vec_load_42[2] * _vec_load_43[2];
        q[19] = _vec_load_42[3] * _vec_load_43[3];
        q[20] = _vec_load_42[4] * _vec_load_43[4];
        q[21] = _vec_load_42[5] * _vec_load_43[5];
        q[22] = _vec_load_42[6] * _vec_load_43[6];
        q[23] = _vec_load_42[7] * _vec_load_43[7];
        float _vec_load_44[8];
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_20[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_20[_blk] = _vptr_20[_blk];
                uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_44[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_44[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_20[_pair]));
                }
            }
        }
        wout[16] = _vec_load_44[0];
        wout[17] = _vec_load_44[1];
        wout[18] = _vec_load_44[2];
        wout[19] = _vec_load_44[3];
        wout[20] = _vec_load_44[4];
        wout[21] = _vec_load_44[5];
        wout[22] = _vec_load_44[6];
        wout[23] = _vec_load_44[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_46[1];
            {
                _vec_load_46[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_46[0];
            unsigned int _vec_load_47[1];
            {
                _vec_load_47[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_47[0];
        }
        {
            unsigned int _vec_load_49[1];
            {
                _vec_load_49[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_49[0];
            unsigned int _vec_load_50[1];
            {
                _vec_load_50[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_50[0];
        }
        {
            unsigned int _vec_load_52[1];
            {
                _vec_load_52[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_52[0];
            unsigned int _vec_load_53[1];
            {
                _vec_load_53[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_53[0];
        }
        {
            unsigned int _vec_load_55[1];
            {
                _vec_load_55[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_55[0];
            unsigned int _vec_load_56[1];
            {
                _vec_load_56[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_56[0];
        }
        float _vec_load_57[4];
        {
            uint2 _vld_21;
            _vld_21 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_57[0 + _pair * 2])[0]), "=f"((&_vec_load_57[0 + _pair * 2])[1])
                    : "r"(_vpairs_21[_pair]));
            }
        }
        float _vec_load_58[4];
        {
            uint2 _vld_22;
            _vld_22 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_58[0 + _pair * 2])[0]), "=f"((&_vec_load_58[0 + _pair * 2])[1])
                    : "r"(_vpairs_22[_pair]));
            }
        }
        q[24] = _vec_load_57[0] * _vec_load_58[0];
        q[25] = _vec_load_57[1] * _vec_load_58[1];
        q[26] = _vec_load_57[2] * _vec_load_58[2];
        q[27] = _vec_load_57[3] * _vec_load_58[3];
        float _vec_load_59[4];
        {
            uint2 _vld_23;
            _vld_23 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_59[0 + _pair * 2])[0]), "=f"((&_vec_load_59[0 + _pair * 2])[1])
                    : "r"(_vpairs_23[_pair]));
            }
        }
        wout[24] = _vec_load_59[0];
        wout[25] = _vec_load_59[1];
        wout[26] = _vec_load_59[2];
        wout[27] = _vec_load_59[3];
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
        float2 sq[3];
        float2 dot[3];
        float fsrc[((1) ? 84 : 1)];
        float2 _f2_0 = make_float2(0.0f, 0.0f);
        sq[0] = _f2_0;
        float2 _f2_1 = make_float2(0.0f, 0.0f);
        dot[0] = _f2_1;
        float2 _f2_2 = make_float2(0.0f, 0.0f);
        sq[1] = _f2_2;
        float2 _f2_3 = make_float2(0.0f, 0.0f);
        dot[1] = _f2_3;
        float2 _f2_4 = make_float2(0.0f, 0.0f);
        sq[2] = _f2_4;
        float2 _f2_5 = make_float2(0.0f, 0.0f);
        dot[2] = _f2_5;
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
            float2 _f2_12 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_12;
            float2 _f2_13 = make_float2(q[0], q[1]);
            float2 qp = _f2_13;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_14 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_14;
            float2 _f2_15 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_15;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_16 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_16;
            float2 _f2_17 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_17;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_18 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_18;
            float2 _f2_19 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_19;
            sq[0] = fma_f32x2_rn_noftz(v_4, v_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4, qp_5, dot[0]);
        }
        unsigned int sw_8[4];
        sw_8[0] = words[14 + woff_7];
        sw_8[1] = words[14 + woff_7 + 1];
        sw_8[2] = words[14 + woff_7 + 2];
        sw_8[3] = words[14 + woff_7 + 3];
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
            float2 _f2_26 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_26;
            float2 _f2_27 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_27;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_28 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_28;
            float2 _f2_29 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_29;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_30 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_30;
            float2 _f2_31 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_31;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_32 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_32;
            float2 _f2_33 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_33;
            sq[1] = fma_f32x2_rn_noftz(v_4_1, v_4_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_1, qp_5_1, dot[1]);
        }
        unsigned int sw_9[4];
        sw_9[0] = words[28 + woff_7];
        sw_9[1] = words[28 + woff_7 + 1];
        sw_9[2] = words[28 + woff_7 + 2];
        sw_9[3] = words[28 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_9[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_9[0] = __as_u32(mixed);
            words[28 + woff_7] = sw_9[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_9[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_9[1] = __as_u32(mixed_2);
            words[28 + woff_7 + 1] = sw_9[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_9[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_9[2] = __as_u32(mixed_5);
            words[28 + woff_7 + 2] = sw_9[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_9[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_9[3] = __as_u32(mixed_8);
            words[28 + woff_7 + 3] = sw_9[3];
            {
                int4 _iv4 = make_int4(sw_9[0 + 0], sw_9[0 + 1], sw_9[0 + 2], sw_9[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
        float sw_9_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_9_f32[_pair * 2])[0]), "=f"((&sw_9_f32[_pair * 2])[1])
                : "r"(sw_9[_pair]));
        }
        {
            fsrc[56] = sw_9_f32[0];
            fsrc[57] = sw_9_f32[1];
            fsrc[58] = sw_9_f32[2];
            fsrc[59] = sw_9_f32[3];
            fsrc[60] = sw_9_f32[4];
            fsrc[61] = sw_9_f32[5];
            fsrc[62] = sw_9_f32[6];
            fsrc[63] = sw_9_f32[7];
        }
        {
            float2 _f2_40 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_40;
            float2 _f2_41 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_41;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_42 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_42;
            float2 _f2_43 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_43;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_44 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_44;
            float2 _f2_45 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_45;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_46 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_46;
            float2 _f2_47 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_47;
            sq[2] = fma_f32x2_rn_noftz(v_4_2, v_4_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_2, qp_5_2, dot[2]);
        }
        int base_10 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_11 = 4;
        unsigned int sw_12[4];
        sw_12[0] = words[woff_11];
        sw_12[1] = words[woff_11 + 1];
        sw_12[2] = words[woff_11 + 2];
        sw_12[3] = words[woff_11 + 3];
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
            fsrc[8] = sw_12_f32[0];
            fsrc[9] = sw_12_f32[1];
            fsrc[10] = sw_12_f32[2];
            fsrc[11] = sw_12_f32[3];
            fsrc[12] = sw_12_f32[4];
            fsrc[13] = sw_12_f32[5];
            fsrc[14] = sw_12_f32[6];
            fsrc[15] = sw_12_f32[7];
        }
        {
            float2 _f2_54 = make_float2(sw_12_f32[0], sw_12_f32[1]);
            float2 v_5 = _f2_54;
            float2 _f2_55 = make_float2(q[8], q[9]);
            float2 qp_6 = _f2_55;
            sq[0] = fma_f32x2_rn_noftz(v_5, v_5, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_5, qp_6, dot[0]);
            float2 _f2_56 = make_float2(sw_12_f32[2], sw_12_f32[3]);
            float2 v_0_3 = _f2_56;
            float2 _f2_57 = make_float2(q[10], q[11]);
            float2 qp_1_3 = _f2_57;
            sq[0] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[0]);
            float2 _f2_58 = make_float2(sw_12_f32[4], sw_12_f32[5]);
            float2 v_2_3 = _f2_58;
            float2 _f2_59 = make_float2(q[12], q[13]);
            float2 qp_3_3 = _f2_59;
            sq[0] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[0]);
            float2 _f2_60 = make_float2(sw_12_f32[6], sw_12_f32[7]);
            float2 v_4_3 = _f2_60;
            float2 _f2_61 = make_float2(q[14], q[15]);
            float2 qp_5_3 = _f2_61;
            sq[0] = fma_f32x2_rn_noftz(v_4_3, v_4_3, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_3, qp_5_3, dot[0]);
        }
        unsigned int sw_13[4];
        sw_13[0] = words[14 + woff_11];
        sw_13[1] = words[14 + woff_11 + 1];
        sw_13[2] = words[14 + woff_11 + 2];
        sw_13[3] = words[14 + woff_11 + 3];
        float sw_13_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_13_f32[_pair * 2])[0]), "=f"((&sw_13_f32[_pair * 2])[1])
                : "r"(sw_13[_pair]));
        }
        {
            fsrc[36] = sw_13_f32[0];
            fsrc[37] = sw_13_f32[1];
            fsrc[38] = sw_13_f32[2];
            fsrc[39] = sw_13_f32[3];
            fsrc[40] = sw_13_f32[4];
            fsrc[41] = sw_13_f32[5];
            fsrc[42] = sw_13_f32[6];
            fsrc[43] = sw_13_f32[7];
        }
        {
            float2 _f2_68 = make_float2(sw_13_f32[0], sw_13_f32[1]);
            float2 v_6 = _f2_68;
            float2 _f2_69 = make_float2(q[8], q[9]);
            float2 qp_7 = _f2_69;
            sq[1] = fma_f32x2_rn_noftz(v_6, v_6, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_6, qp_7, dot[1]);
            float2 _f2_70 = make_float2(sw_13_f32[2], sw_13_f32[3]);
            float2 v_0_4 = _f2_70;
            float2 _f2_71 = make_float2(q[10], q[11]);
            float2 qp_1_4 = _f2_71;
            sq[1] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[1]);
            float2 _f2_72 = make_float2(sw_13_f32[4], sw_13_f32[5]);
            float2 v_2_4 = _f2_72;
            float2 _f2_73 = make_float2(q[12], q[13]);
            float2 qp_3_4 = _f2_73;
            sq[1] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[1]);
            float2 _f2_74 = make_float2(sw_13_f32[6], sw_13_f32[7]);
            float2 v_4_4 = _f2_74;
            float2 _f2_75 = make_float2(q[14], q[15]);
            float2 qp_5_4 = _f2_75;
            sq[1] = fma_f32x2_rn_noftz(v_4_4, v_4_4, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_4, qp_5_4, dot[1]);
        }
        unsigned int sw_14[4];
        sw_14[0] = words[28 + woff_11];
        sw_14[1] = words[28 + woff_11 + 1];
        sw_14[2] = words[28 + woff_11 + 2];
        sw_14[3] = words[28 + woff_11 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_14[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_11]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_14[0] = __as_u32(mixed_1);
            words[28 + woff_11] = sw_14[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_14[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_11 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_14[1] = __as_u32(mixed_2_1);
            words[28 + woff_11 + 1] = sw_14[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_14[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_11 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_14[2] = __as_u32(mixed_5_1);
            words[28 + woff_11 + 2] = sw_14[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_14[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_11 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_14[3] = __as_u32(mixed_8_1);
            words[28 + woff_11 + 3] = sw_14[3];
            {
                int4 _iv4 = make_int4(sw_14[0 + 0], sw_14[0 + 1], sw_14[0 + 2], sw_14[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_10)) + 0) = _iv4;
            }
        }
        float sw_14_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_14_f32[_pair * 2])[0]), "=f"((&sw_14_f32[_pair * 2])[1])
                : "r"(sw_14[_pair]));
        }
        {
            fsrc[64] = sw_14_f32[0];
            fsrc[65] = sw_14_f32[1];
            fsrc[66] = sw_14_f32[2];
            fsrc[67] = sw_14_f32[3];
            fsrc[68] = sw_14_f32[4];
            fsrc[69] = sw_14_f32[5];
            fsrc[70] = sw_14_f32[6];
            fsrc[71] = sw_14_f32[7];
        }
        {
            float2 _f2_82 = make_float2(sw_14_f32[0], sw_14_f32[1]);
            float2 v_7 = _f2_82;
            float2 _f2_83 = make_float2(q[8], q[9]);
            float2 qp_8 = _f2_83;
            sq[2] = fma_f32x2_rn_noftz(v_7, v_7, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_7, qp_8, dot[2]);
            float2 _f2_84 = make_float2(sw_14_f32[2], sw_14_f32[3]);
            float2 v_0_5 = _f2_84;
            float2 _f2_85 = make_float2(q[10], q[11]);
            float2 qp_1_5 = _f2_85;
            sq[2] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[2]);
            float2 _f2_86 = make_float2(sw_14_f32[4], sw_14_f32[5]);
            float2 v_2_5 = _f2_86;
            float2 _f2_87 = make_float2(q[12], q[13]);
            float2 qp_3_5 = _f2_87;
            sq[2] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[2]);
            float2 _f2_88 = make_float2(sw_14_f32[6], sw_14_f32[7]);
            float2 v_4_5 = _f2_88;
            float2 _f2_89 = make_float2(q[14], q[15]);
            float2 qp_5_5 = _f2_89;
            sq[2] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[2]);
        }
        int base_15 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_16 = 8;
        unsigned int sw_17[4];
        sw_17[0] = words[woff_16];
        sw_17[1] = words[woff_16 + 1];
        sw_17[2] = words[woff_16 + 2];
        sw_17[3] = words[woff_16 + 3];
        float sw_17_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_17_f32[_pair * 2])[0]), "=f"((&sw_17_f32[_pair * 2])[1])
                : "r"(sw_17[_pair]));
        }
        {
            fsrc[16] = sw_17_f32[0];
            fsrc[17] = sw_17_f32[1];
            fsrc[18] = sw_17_f32[2];
            fsrc[19] = sw_17_f32[3];
            fsrc[20] = sw_17_f32[4];
            fsrc[21] = sw_17_f32[5];
            fsrc[22] = sw_17_f32[6];
            fsrc[23] = sw_17_f32[7];
        }
        {
            float2 _f2_96 = make_float2(sw_17_f32[0], sw_17_f32[1]);
            float2 v_8 = _f2_96;
            float2 _f2_97 = make_float2(q[16], q[17]);
            float2 qp_9 = _f2_97;
            sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_8, qp_9, dot[0]);
            float2 _f2_98 = make_float2(sw_17_f32[2], sw_17_f32[3]);
            float2 v_0_6 = _f2_98;
            float2 _f2_99 = make_float2(q[18], q[19]);
            float2 qp_1_6 = _f2_99;
            sq[0] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[0]);
            float2 _f2_100 = make_float2(sw_17_f32[4], sw_17_f32[5]);
            float2 v_2_6 = _f2_100;
            float2 _f2_101 = make_float2(q[20], q[21]);
            float2 qp_3_6 = _f2_101;
            sq[0] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[0]);
            float2 _f2_102 = make_float2(sw_17_f32[6], sw_17_f32[7]);
            float2 v_4_6 = _f2_102;
            float2 _f2_103 = make_float2(q[22], q[23]);
            float2 qp_5_6 = _f2_103;
            sq[0] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[0]);
        }
        unsigned int sw_18[4];
        sw_18[0] = words[14 + woff_16];
        sw_18[1] = words[14 + woff_16 + 1];
        sw_18[2] = words[14 + woff_16 + 2];
        sw_18[3] = words[14 + woff_16 + 3];
        float sw_18_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_18_f32[_pair * 2])[0]), "=f"((&sw_18_f32[_pair * 2])[1])
                : "r"(sw_18[_pair]));
        }
        {
            fsrc[44] = sw_18_f32[0];
            fsrc[45] = sw_18_f32[1];
            fsrc[46] = sw_18_f32[2];
            fsrc[47] = sw_18_f32[3];
            fsrc[48] = sw_18_f32[4];
            fsrc[49] = sw_18_f32[5];
            fsrc[50] = sw_18_f32[6];
            fsrc[51] = sw_18_f32[7];
        }
        {
            float2 _f2_110 = make_float2(sw_18_f32[0], sw_18_f32[1]);
            float2 v_9 = _f2_110;
            float2 _f2_111 = make_float2(q[16], q[17]);
            float2 qp_10 = _f2_111;
            sq[1] = fma_f32x2_rn_noftz(v_9, v_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_9, qp_10, dot[1]);
            float2 _f2_112 = make_float2(sw_18_f32[2], sw_18_f32[3]);
            float2 v_0_7 = _f2_112;
            float2 _f2_113 = make_float2(q[18], q[19]);
            float2 qp_1_7 = _f2_113;
            sq[1] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[1]);
            float2 _f2_114 = make_float2(sw_18_f32[4], sw_18_f32[5]);
            float2 v_2_7 = _f2_114;
            float2 _f2_115 = make_float2(q[20], q[21]);
            float2 qp_3_7 = _f2_115;
            sq[1] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[1]);
            float2 _f2_116 = make_float2(sw_18_f32[6], sw_18_f32[7]);
            float2 v_4_7 = _f2_116;
            float2 _f2_117 = make_float2(q[22], q[23]);
            float2 qp_5_7 = _f2_117;
            sq[1] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[1]);
        }
        unsigned int sw_19[4];
        sw_19[0] = words[28 + woff_16];
        sw_19[1] = words[28 + woff_16 + 1];
        sw_19[2] = words[28 + woff_16 + 2];
        sw_19[3] = words[28 + woff_16 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_19[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_16]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_19[0] = __as_u32(mixed_3);
            words[28 + woff_16] = sw_19[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_19[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_16 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_19[1] = __as_u32(mixed_2_2);
            words[28 + woff_16 + 1] = sw_19[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_19[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_16 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_19[2] = __as_u32(mixed_5_2);
            words[28 + woff_16 + 2] = sw_19[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_19[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_16 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_19[3] = __as_u32(mixed_8_2);
            words[28 + woff_16 + 3] = sw_19[3];
            {
                int4 _iv4 = make_int4(sw_19[0 + 0], sw_19[0 + 1], sw_19[0 + 2], sw_19[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_15)) + 0) = _iv4;
            }
        }
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
            fsrc[72] = sw_19_f32[0];
            fsrc[73] = sw_19_f32[1];
            fsrc[74] = sw_19_f32[2];
            fsrc[75] = sw_19_f32[3];
            fsrc[76] = sw_19_f32[4];
            fsrc[77] = sw_19_f32[5];
            fsrc[78] = sw_19_f32[6];
            fsrc[79] = sw_19_f32[7];
        }
        {
            float2 _f2_124 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_10 = _f2_124;
            float2 _f2_125 = make_float2(q[16], q[17]);
            float2 qp_11 = _f2_125;
            sq[2] = fma_f32x2_rn_noftz(v_10, v_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_10, qp_11, dot[2]);
            float2 _f2_126 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_8 = _f2_126;
            float2 _f2_127 = make_float2(q[18], q[19]);
            float2 qp_1_8 = _f2_127;
            sq[2] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[2]);
            float2 _f2_128 = make_float2(sw_19_f32[4], sw_19_f32[5]);
            float2 v_2_8 = _f2_128;
            float2 _f2_129 = make_float2(q[20], q[21]);
            float2 qp_3_8 = _f2_129;
            sq[2] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[2]);
            float2 _f2_130 = make_float2(sw_19_f32[6], sw_19_f32[7]);
            float2 v_4_8 = _f2_130;
            float2 _f2_131 = make_float2(q[22], q[23]);
            float2 qp_5_8 = _f2_131;
            sq[2] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[2]);
        }
        int base_20 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_21 = 12;
        unsigned int sw_22[4];
        sw_22[0] = words[woff_21];
        sw_22[1] = words[woff_21 + 1];
        float sw_22_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_22_f32[_pair * 2])[0]), "=f"((&sw_22_f32[_pair * 2])[1])
                : "r"(sw_22[_pair]));
        }
        {
            fsrc[24] = sw_22_f32[0];
            fsrc[25] = sw_22_f32[1];
            fsrc[26] = sw_22_f32[2];
            fsrc[27] = sw_22_f32[3];
        }
        {
            float2 _f2_132 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_11 = _f2_132;
            sq[0] = fma_f32x2_rn_noftz(v_11, v_11, sq[0]);
            float2 _f2_133 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_0_9 = _f2_133;
            sq[0] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[0]);
            float2 _f2_134 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_1_1 = _f2_134;
            float2 _f2_135 = make_float2(q[24], q[25]);
            float2 qp_12 = _f2_135;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_12, dot[0]);
            float2 _f2_136 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_2_9 = _f2_136;
            float2 _f2_137 = make_float2(q[26], q[27]);
            float2 qp_3_9 = _f2_137;
            dot[0] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[0]);
        }
        unsigned int sw_23[4];
        sw_23[0] = words[14 + woff_21];
        sw_23[1] = words[14 + woff_21 + 1];
        float sw_23_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_23_f32[_pair * 2])[0]), "=f"((&sw_23_f32[_pair * 2])[1])
                : "r"(sw_23[_pair]));
        }
        {
            fsrc[52] = sw_23_f32[0];
            fsrc[53] = sw_23_f32[1];
            fsrc[54] = sw_23_f32[2];
            fsrc[55] = sw_23_f32[3];
        }
        {
            float2 _f2_146 = make_float2(sw_23_f32[0], sw_23_f32[1]);
            float2 v_12 = _f2_146;
            sq[1] = fma_f32x2_rn_noftz(v_12, v_12, sq[1]);
            float2 _f2_147 = make_float2(sw_23_f32[2], sw_23_f32[3]);
            float2 v_0_10 = _f2_147;
            sq[1] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[1]);
            float2 _f2_148 = make_float2(sw_23_f32[0], sw_23_f32[1]);
            float2 v_1_2 = _f2_148;
            float2 _f2_149 = make_float2(q[24], q[25]);
            float2 qp_13 = _f2_149;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_13, dot[1]);
            float2 _f2_150 = make_float2(sw_23_f32[2], sw_23_f32[3]);
            float2 v_2_10 = _f2_150;
            float2 _f2_151 = make_float2(q[26], q[27]);
            float2 qp_3_10 = _f2_151;
            dot[1] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[1]);
        }
        unsigned int sw_24[4];
        sw_24[0] = words[28 + woff_21];
        sw_24[1] = words[28 + woff_21 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_24[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_21]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_24[0] = __as_u32(mixed_4);
            words[28 + woff_21] = sw_24[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_24[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_21 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_24[1] = __as_u32(mixed_2_3);
            words[28 + woff_21 + 1] = sw_24[1];
            {
                int2 _iv2 = make_int2(sw_24[0 + 0], sw_24[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_20)) + 0) = _iv2;
            }
        }
        float sw_24_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_24_f32[_pair * 2])[0]), "=f"((&sw_24_f32[_pair * 2])[1])
                : "r"(sw_24[_pair]));
        }
        {
            fsrc[80] = sw_24_f32[0];
            fsrc[81] = sw_24_f32[1];
            fsrc[82] = sw_24_f32[2];
            fsrc[83] = sw_24_f32[3];
        }
        {
            float2 _f2_160 = make_float2(sw_24_f32[0], sw_24_f32[1]);
            float2 v_13 = _f2_160;
            sq[2] = fma_f32x2_rn_noftz(v_13, v_13, sq[2]);
            float2 _f2_161 = make_float2(sw_24_f32[2], sw_24_f32[3]);
            float2 v_0_11 = _f2_161;
            sq[2] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[2]);
            float2 _f2_162 = make_float2(sw_24_f32[0], sw_24_f32[1]);
            float2 v_1_3 = _f2_162;
            float2 _f2_163 = make_float2(q[24], q[25]);
            float2 qp_14 = _f2_163;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_14, dot[2]);
            float2 _f2_164 = make_float2(sw_24_f32[2], sw_24_f32[3]);
            float2 v_2_11 = _f2_164;
            float2 _f2_165 = make_float2(q[26], q[27]);
            float2 qp_3_11 = _f2_165;
            dot[2] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[2]);
        }
        float2 pairs[3];
        float2 _f2_174 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_174;
        float2 _f2_175 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_175;
        float2 _f2_176 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_176;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_177 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_177;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_25 = 0;
        bits_25 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_25, 16);
        unsigned long long peerbits_26 = _shfl_xor_1;
        float2 _f2_178 = make_float2(0.0f, 0.0f);
        float2 peer_27 = _f2_178;
        peer_27 = reinterpret_cast<float2*>(&peerbits_26)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_27);
        unsigned long long bits_28 = 0;
        bits_28 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_28, 16);
        unsigned long long peerbits_29 = _shfl_xor_2;
        float2 _f2_179 = make_float2(0.0f, 0.0f);
        float2 peer_30 = _f2_179;
        peer_30 = reinterpret_cast<float2*>(&peerbits_29)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_30);
        unsigned long long bits_31 = 0;
        bits_31 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_31, 8);
        unsigned long long peerbits_32 = _shfl_xor_3;
        float2 _f2_180 = make_float2(0.0f, 0.0f);
        float2 peer_33 = _f2_180;
        peer_33 = reinterpret_cast<float2*>(&peerbits_32)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_33);
        unsigned long long bits_34 = 0;
        bits_34 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_34, 8);
        unsigned long long peerbits_35 = _shfl_xor_4;
        float2 _f2_181 = make_float2(0.0f, 0.0f);
        float2 peer_36 = _f2_181;
        peer_36 = reinterpret_cast<float2*>(&peerbits_35)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_36);
        unsigned long long bits_37 = 0;
        bits_37 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_37, 8);
        unsigned long long peerbits_38 = _shfl_xor_5;
        float2 _f2_182 = make_float2(0.0f, 0.0f);
        float2 peer_39 = _f2_182;
        peer_39 = reinterpret_cast<float2*>(&peerbits_38)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_39);
        unsigned long long bits_40 = 0;
        bits_40 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_40, 4);
        unsigned long long peerbits_41 = _shfl_xor_6;
        float2 _f2_183 = make_float2(0.0f, 0.0f);
        float2 peer_42 = _f2_183;
        peer_42 = reinterpret_cast<float2*>(&peerbits_41)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_42);
        unsigned long long bits_43 = 0;
        bits_43 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_43, 4);
        unsigned long long peerbits_44 = _shfl_xor_7;
        float2 _f2_184 = make_float2(0.0f, 0.0f);
        float2 peer_45 = _f2_184;
        peer_45 = reinterpret_cast<float2*>(&peerbits_44)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_45);
        unsigned long long bits_46 = 0;
        bits_46 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_46, 4);
        unsigned long long peerbits_47 = _shfl_xor_8;
        float2 _f2_185 = make_float2(0.0f, 0.0f);
        float2 peer_48 = _f2_185;
        peer_48 = reinterpret_cast<float2*>(&peerbits_47)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_48);
        unsigned long long bits_49 = 0;
        bits_49 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_49, 2);
        unsigned long long peerbits_50 = _shfl_xor_9;
        float2 _f2_186 = make_float2(0.0f, 0.0f);
        float2 peer_51 = _f2_186;
        peer_51 = reinterpret_cast<float2*>(&peerbits_50)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_51);
        unsigned long long bits_52 = 0;
        bits_52 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_52, 2);
        unsigned long long peerbits_53 = _shfl_xor_10;
        float2 _f2_187 = make_float2(0.0f, 0.0f);
        float2 peer_54 = _f2_187;
        peer_54 = reinterpret_cast<float2*>(&peerbits_53)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_54);
        unsigned long long bits_55 = 0;
        bits_55 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_55, 2);
        unsigned long long peerbits_56 = _shfl_xor_11;
        float2 _f2_188 = make_float2(0.0f, 0.0f);
        float2 peer_57 = _f2_188;
        peer_57 = reinterpret_cast<float2*>(&peerbits_56)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_57);
        unsigned long long bits_58 = 0;
        bits_58 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_58, 1);
        unsigned long long peerbits_59 = _shfl_xor_12;
        float2 _f2_189 = make_float2(0.0f, 0.0f);
        float2 peer_60 = _f2_189;
        peer_60 = reinterpret_cast<float2*>(&peerbits_59)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_60);
        unsigned long long bits_61 = 0;
        bits_61 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_61, 1);
        unsigned long long peerbits_62 = _shfl_xor_13;
        float2 _f2_190 = make_float2(0.0f, 0.0f);
        float2 peer_63 = _f2_190;
        peer_63 = reinterpret_cast<float2*>(&peerbits_62)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_63);
        unsigned long long bits_64 = 0;
        bits_64 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_64, 1);
        unsigned long long peerbits_65 = _shfl_xor_14;
        float2 _f2_191 = make_float2(0.0f, 0.0f);
        float2 peer_66 = _f2_191;
        peer_66 = reinterpret_cast<float2*>(&peerbits_65)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_66);
        if (lane == 0) {
            stats[warp_0 * 3 * 2] = pairs[0].x;
            stats[warp_0 * 3 * 2 + 1] = pairs[0].y;
            stats[(warp_0 * 3 + 1) * 2] = pairs[1].x;
            stats[(warp_0 * 3 + 1) * 2 + 1] = pairs[1].y;
            stats[(warp_0 * 3 + 2) * 2] = pairs[2].x;
            stats[(warp_0 * 3 + 2) * 2 + 1] = pairs[2].y;
        }
        __syncthreads();
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 3) {
            total_sq = stats[(stat_w * 3 + stat_n) * 2];
            total_dot = stats[(stat_w * 3 + stat_n) * 2 + 1];
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
        if (stat_n < 3 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[3];
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_0;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_1;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_2;
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
        float _fmax_2 = fmaxf(max_chunk, logits[2]);
        max_chunk = _fmax_2;
        float _fmax_3 = fmaxf(max_running, max_chunk);
        float max_new = _fmax_3;
        float _exp2_0 = approx_exp2((max_running - max_new) * 1.4426950408889634f);
        float correction = _exp2_0;
        float weights[3];
        float sum_weights = 0.0f;
        float _exp2_1 = approx_exp2((logits[0] - max_new) * 1.4426950408889634f);
        weights[0] = _exp2_1;
        sum_weights += weights[0];
        float _exp2_2 = approx_exp2((logits[1] - max_new) * 1.4426950408889634f);
        weights[1] = _exp2_2;
        sum_weights += weights[1];
        float _exp2_3 = approx_exp2((logits[2] - max_new) * 1.4426950408889634f);
        weights[2] = _exp2_3;
        sum_weights += weights[2];
        float2 _f2_192 = make_float2(correction, correction);
        float2 corr = _f2_192;
        const int woff_67 = 0;
        int base_68 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_193 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_193;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_194 = make_float2(acc[2], acc[3]);
        float2 previous_69 = _f2_194;
        a_5[1] = mul_f32x2_noftz(previous_69, corr);
        float2 _f2_195 = make_float2(acc[4], acc[5]);
        float2 previous_70 = _f2_195;
        a_5[2] = mul_f32x2_noftz(previous_70, corr);
        float2 _f2_196 = make_float2(acc[6], acc[7]);
        float2 previous_71 = _f2_196;
        a_5[3] = mul_f32x2_noftz(previous_71, corr);
        float2 _f2_197 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_197;
        {
            float2 _f2_198 = make_float2(fsrc[0], fsrc[1]);
            float2 v_14 = _f2_198;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_14, a_5[0]);
            float2 _f2_199 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_12 = _f2_199;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_12, a_5[1]);
            float2 _f2_200 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_4 = _f2_200;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_4, a_5[2]);
            float2 _f2_201 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_12 = _f2_201;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_12, a_5[3]);
        }
        float2 _f2_206 = make_float2(weights[1], weights[1]);
        float2 weight_72 = _f2_206;
        {
            float2 _f2_207 = make_float2(fsrc[28], fsrc[29]);
            float2 v_15 = _f2_207;
            a_5[0] = fma_f32x2_rn_noftz(weight_72, v_15, a_5[0]);
            float2 _f2_208 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_13 = _f2_208;
            a_5[1] = fma_f32x2_rn_noftz(weight_72, v_0_13, a_5[1]);
            float2 _f2_209 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_5 = _f2_209;
            a_5[2] = fma_f32x2_rn_noftz(weight_72, v_1_5, a_5[2]);
            float2 _f2_210 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_13 = _f2_210;
            a_5[3] = fma_f32x2_rn_noftz(weight_72, v_2_13, a_5[3]);
        }
        float2 _f2_215 = make_float2(weights[2], weights[2]);
        float2 weight_73 = _f2_215;
        {
            float2 _f2_216 = make_float2(fsrc[56], fsrc[57]);
            float2 v_16 = _f2_216;
            a_5[0] = fma_f32x2_rn_noftz(weight_73, v_16, a_5[0]);
            float2 _f2_217 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_14 = _f2_217;
            a_5[1] = fma_f32x2_rn_noftz(weight_73, v_0_14, a_5[1]);
            float2 _f2_218 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_6 = _f2_218;
            a_5[2] = fma_f32x2_rn_noftz(weight_73, v_1_6, a_5[2]);
            float2 _f2_219 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_14 = _f2_219;
            a_5[3] = fma_f32x2_rn_noftz(weight_73, v_2_14, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_74 = 4;
        int base_75 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_76[4];
        float2 _f2_224 = make_float2(acc[8], acc[9]);
        float2 previous_77 = _f2_224;
        a_76[0] = mul_f32x2_noftz(previous_77, corr);
        float2 _f2_225 = make_float2(acc[10], acc[11]);
        float2 previous_78 = _f2_225;
        a_76[1] = mul_f32x2_noftz(previous_78, corr);
        float2 _f2_226 = make_float2(acc[12], acc[13]);
        float2 previous_79 = _f2_226;
        a_76[2] = mul_f32x2_noftz(previous_79, corr);
        float2 _f2_227 = make_float2(acc[14], acc[15]);
        float2 previous_80 = _f2_227;
        a_76[3] = mul_f32x2_noftz(previous_80, corr);
        float2 _f2_228 = make_float2(weights[0], weights[0]);
        float2 weight_81 = _f2_228;
        {
            float2 _f2_229 = make_float2(fsrc[8], fsrc[9]);
            float2 v_17 = _f2_229;
            a_76[0] = fma_f32x2_rn_noftz(weight_81, v_17, a_76[0]);
            float2 _f2_230 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_15 = _f2_230;
            a_76[1] = fma_f32x2_rn_noftz(weight_81, v_0_15, a_76[1]);
            float2 _f2_231 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_7 = _f2_231;
            a_76[2] = fma_f32x2_rn_noftz(weight_81, v_1_7, a_76[2]);
            float2 _f2_232 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_15 = _f2_232;
            a_76[3] = fma_f32x2_rn_noftz(weight_81, v_2_15, a_76[3]);
        }
        float2 _f2_237 = make_float2(weights[1], weights[1]);
        float2 weight_82 = _f2_237;
        {
            float2 _f2_238 = make_float2(fsrc[36], fsrc[37]);
            float2 v_18 = _f2_238;
            a_76[0] = fma_f32x2_rn_noftz(weight_82, v_18, a_76[0]);
            float2 _f2_239 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_16 = _f2_239;
            a_76[1] = fma_f32x2_rn_noftz(weight_82, v_0_16, a_76[1]);
            float2 _f2_240 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_8 = _f2_240;
            a_76[2] = fma_f32x2_rn_noftz(weight_82, v_1_8, a_76[2]);
            float2 _f2_241 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_16 = _f2_241;
            a_76[3] = fma_f32x2_rn_noftz(weight_82, v_2_16, a_76[3]);
        }
        float2 _f2_246 = make_float2(weights[2], weights[2]);
        float2 weight_83 = _f2_246;
        {
            float2 _f2_247 = make_float2(fsrc[64], fsrc[65]);
            float2 v_19 = _f2_247;
            a_76[0] = fma_f32x2_rn_noftz(weight_83, v_19, a_76[0]);
            float2 _f2_248 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_17 = _f2_248;
            a_76[1] = fma_f32x2_rn_noftz(weight_83, v_0_17, a_76[1]);
            float2 _f2_249 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_9 = _f2_249;
            a_76[2] = fma_f32x2_rn_noftz(weight_83, v_1_9, a_76[2]);
            float2 _f2_250 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_17 = _f2_250;
            a_76[3] = fma_f32x2_rn_noftz(weight_83, v_2_17, a_76[3]);
        }
        acc[8] = a_76[0].x;
        acc[9] = a_76[0].y;
        acc[10] = a_76[1].x;
        acc[11] = a_76[1].y;
        acc[12] = a_76[2].x;
        acc[13] = a_76[2].y;
        acc[14] = a_76[3].x;
        acc[15] = a_76[3].y;
        const int woff_84 = 8;
        int base_85 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_86[4];
        float2 _f2_255 = make_float2(acc[16], acc[17]);
        float2 previous_87 = _f2_255;
        a_86[0] = mul_f32x2_noftz(previous_87, corr);
        float2 _f2_256 = make_float2(acc[18], acc[19]);
        float2 previous_88 = _f2_256;
        a_86[1] = mul_f32x2_noftz(previous_88, corr);
        float2 _f2_257 = make_float2(acc[20], acc[21]);
        float2 previous_89 = _f2_257;
        a_86[2] = mul_f32x2_noftz(previous_89, corr);
        float2 _f2_258 = make_float2(acc[22], acc[23]);
        float2 previous_90 = _f2_258;
        a_86[3] = mul_f32x2_noftz(previous_90, corr);
        float2 _f2_259 = make_float2(weights[0], weights[0]);
        float2 weight_91 = _f2_259;
        {
            float2 _f2_260 = make_float2(fsrc[16], fsrc[17]);
            float2 v_20 = _f2_260;
            a_86[0] = fma_f32x2_rn_noftz(weight_91, v_20, a_86[0]);
            float2 _f2_261 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_18 = _f2_261;
            a_86[1] = fma_f32x2_rn_noftz(weight_91, v_0_18, a_86[1]);
            float2 _f2_262 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_10 = _f2_262;
            a_86[2] = fma_f32x2_rn_noftz(weight_91, v_1_10, a_86[2]);
            float2 _f2_263 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_18 = _f2_263;
            a_86[3] = fma_f32x2_rn_noftz(weight_91, v_2_18, a_86[3]);
        }
        float2 _f2_268 = make_float2(weights[1], weights[1]);
        float2 weight_92 = _f2_268;
        {
            float2 _f2_269 = make_float2(fsrc[44], fsrc[45]);
            float2 v_21 = _f2_269;
            a_86[0] = fma_f32x2_rn_noftz(weight_92, v_21, a_86[0]);
            float2 _f2_270 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_19 = _f2_270;
            a_86[1] = fma_f32x2_rn_noftz(weight_92, v_0_19, a_86[1]);
            float2 _f2_271 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_11 = _f2_271;
            a_86[2] = fma_f32x2_rn_noftz(weight_92, v_1_11, a_86[2]);
            float2 _f2_272 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_19 = _f2_272;
            a_86[3] = fma_f32x2_rn_noftz(weight_92, v_2_19, a_86[3]);
        }
        float2 _f2_277 = make_float2(weights[2], weights[2]);
        float2 weight_93 = _f2_277;
        {
            float2 _f2_278 = make_float2(fsrc[72], fsrc[73]);
            float2 v_22 = _f2_278;
            a_86[0] = fma_f32x2_rn_noftz(weight_93, v_22, a_86[0]);
            float2 _f2_279 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_20 = _f2_279;
            a_86[1] = fma_f32x2_rn_noftz(weight_93, v_0_20, a_86[1]);
            float2 _f2_280 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_12 = _f2_280;
            a_86[2] = fma_f32x2_rn_noftz(weight_93, v_1_12, a_86[2]);
            float2 _f2_281 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_20 = _f2_281;
            a_86[3] = fma_f32x2_rn_noftz(weight_93, v_2_20, a_86[3]);
        }
        acc[16] = a_86[0].x;
        acc[17] = a_86[0].y;
        acc[18] = a_86[1].x;
        acc[19] = a_86[1].y;
        acc[20] = a_86[2].x;
        acc[21] = a_86[2].y;
        acc[22] = a_86[3].x;
        acc[23] = a_86[3].y;
        const int woff_94 = 12;
        int base_95 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_96[4];
        float2 _f2_286 = make_float2(acc[24], acc[25]);
        float2 previous_97 = _f2_286;
        a_96[0] = mul_f32x2_noftz(previous_97, corr);
        float2 _f2_287 = make_float2(acc[26], acc[27]);
        float2 previous_98 = _f2_287;
        a_96[1] = mul_f32x2_noftz(previous_98, corr);
        float2 _f2_288 = make_float2(weights[0], weights[0]);
        float2 weight_99 = _f2_288;
        {
            float2 _f2_289 = make_float2(fsrc[24], fsrc[25]);
            float2 v_23 = _f2_289;
            a_96[0] = fma_f32x2_rn_noftz(weight_99, v_23, a_96[0]);
            float2 _f2_290 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_21 = _f2_290;
            a_96[1] = fma_f32x2_rn_noftz(weight_99, v_0_21, a_96[1]);
        }
        float2 _f2_293 = make_float2(weights[1], weights[1]);
        float2 weight_100 = _f2_293;
        {
            float2 _f2_294 = make_float2(fsrc[52], fsrc[53]);
            float2 v_24 = _f2_294;
            a_96[0] = fma_f32x2_rn_noftz(weight_100, v_24, a_96[0]);
            float2 _f2_295 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_22 = _f2_295;
            a_96[1] = fma_f32x2_rn_noftz(weight_100, v_0_22, a_96[1]);
        }
        float2 _f2_298 = make_float2(weights[2], weights[2]);
        float2 weight_101 = _f2_298;
        {
            float2 _f2_299 = make_float2(fsrc[80], fsrc[81]);
            float2 v_25 = _f2_299;
            a_96[0] = fma_f32x2_rn_noftz(weight_101, v_25, a_96[0]);
            float2 _f2_300 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_23 = _f2_300;
            a_96[1] = fma_f32x2_rn_noftz(weight_101, v_0_23, a_96[1]);
        }
        acc[24] = a_96[0].x;
        acc[25] = a_96[0].y;
        acc[26] = a_96[1].x;
        acc[27] = a_96[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float2 _f2_303 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_303;
        float2 _f2_304 = make_float2(acc[0], acc[1]);
        float2 v_26 = _f2_304;
        output_sq_pair = fma_f32x2_rn_noftz(v_26, v_26, output_sq_pair);
        float2 _f2_305 = make_float2(acc[2], acc[3]);
        float2 v_102 = _f2_305;
        output_sq_pair = fma_f32x2_rn_noftz(v_102, v_102, output_sq_pair);
        float2 _f2_306 = make_float2(acc[4], acc[5]);
        float2 v_103 = _f2_306;
        output_sq_pair = fma_f32x2_rn_noftz(v_103, v_103, output_sq_pair);
        float2 _f2_307 = make_float2(acc[6], acc[7]);
        float2 v_104 = _f2_307;
        output_sq_pair = fma_f32x2_rn_noftz(v_104, v_104, output_sq_pair);
        float2 _f2_308 = make_float2(acc[8], acc[9]);
        float2 v_105 = _f2_308;
        output_sq_pair = fma_f32x2_rn_noftz(v_105, v_105, output_sq_pair);
        float2 _f2_309 = make_float2(acc[10], acc[11]);
        float2 v_106 = _f2_309;
        output_sq_pair = fma_f32x2_rn_noftz(v_106, v_106, output_sq_pair);
        float2 _f2_310 = make_float2(acc[12], acc[13]);
        float2 v_107 = _f2_310;
        output_sq_pair = fma_f32x2_rn_noftz(v_107, v_107, output_sq_pair);
        float2 _f2_311 = make_float2(acc[14], acc[15]);
        float2 v_108 = _f2_311;
        output_sq_pair = fma_f32x2_rn_noftz(v_108, v_108, output_sq_pair);
        float2 _f2_312 = make_float2(acc[16], acc[17]);
        float2 v_109 = _f2_312;
        output_sq_pair = fma_f32x2_rn_noftz(v_109, v_109, output_sq_pair);
        float2 _f2_313 = make_float2(acc[18], acc[19]);
        float2 v_110 = _f2_313;
        output_sq_pair = fma_f32x2_rn_noftz(v_110, v_110, output_sq_pair);
        float2 _f2_314 = make_float2(acc[20], acc[21]);
        float2 v_111 = _f2_314;
        output_sq_pair = fma_f32x2_rn_noftz(v_111, v_111, output_sq_pair);
        float2 _f2_315 = make_float2(acc[22], acc[23]);
        float2 v_112 = _f2_315;
        output_sq_pair = fma_f32x2_rn_noftz(v_112, v_112, output_sq_pair);
        float2 _f2_316 = make_float2(acc[24], acc[25]);
        float2 v_113 = _f2_316;
        output_sq_pair = fma_f32x2_rn_noftz(v_113, v_113, output_sq_pair);
        float2 _f2_317 = make_float2(acc[26], acc[27]);
        float2 v_114 = _f2_317;
        output_sq_pair = fma_f32x2_rn_noftz(v_114, v_114, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_15;
        float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_16;
        float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_17;
        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_18;
        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_19;
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
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_3;
        float2 _f2_318 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_318;
        int base_115 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_319 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_319, rsigma_pair);
        float2 _f2_320 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_320);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_116 = 2;
        const int acc_idx_117 = value_idx_116;
        float2 _f2_321 = make_float2(acc[acc_idx_117], acc[acc_idx_117 + 1]);
        float2 scaled_pair_118 = mul_f32x2_noftz(_f2_321, rsigma_pair);
        float2 _f2_322 = make_float2(wout[acc_idx_117], wout[acc_idx_117 + 1]);
        float2 normalized_pair_119 = mul_f32x2_noftz(scaled_pair_118, _f2_322);
        output_values[value_idx_116] = normalized_pair_119.x;
        output_values[value_idx_116 + 1] = normalized_pair_119.y;
        const int value_idx_120 = 4;
        const int acc_idx_121 = value_idx_120;
        float2 _f2_323 = make_float2(acc[acc_idx_121], acc[acc_idx_121 + 1]);
        float2 scaled_pair_122 = mul_f32x2_noftz(_f2_323, rsigma_pair);
        float2 _f2_324 = make_float2(wout[acc_idx_121], wout[acc_idx_121 + 1]);
        float2 normalized_pair_123 = mul_f32x2_noftz(scaled_pair_122, _f2_324);
        output_values[value_idx_120] = normalized_pair_123.x;
        output_values[value_idx_120 + 1] = normalized_pair_123.y;
        const int value_idx_124 = 6;
        const int acc_idx_125 = value_idx_124;
        float2 _f2_325 = make_float2(acc[acc_idx_125], acc[acc_idx_125 + 1]);
        float2 scaled_pair_126 = mul_f32x2_noftz(_f2_325, rsigma_pair);
        float2 _f2_326 = make_float2(wout[acc_idx_125], wout[acc_idx_125 + 1]);
        float2 normalized_pair_127 = mul_f32x2_noftz(scaled_pair_126, _f2_326);
        output_values[value_idx_124] = normalized_pair_127.x;
        output_values[value_idx_124 + 1] = normalized_pair_127.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_115 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_128 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_129[8];
        const int value_idx_130 = 0;
        const int acc_idx_131 = 8 + value_idx_130;
        float2 _f2_327 = make_float2(acc[acc_idx_131], acc[acc_idx_131 + 1]);
        float2 scaled_pair_132 = mul_f32x2_noftz(_f2_327, rsigma_pair);
        float2 _f2_328 = make_float2(wout[acc_idx_131], wout[acc_idx_131 + 1]);
        float2 normalized_pair_133 = mul_f32x2_noftz(scaled_pair_132, _f2_328);
        output_values_129[value_idx_130] = normalized_pair_133.x;
        output_values_129[value_idx_130 + 1] = normalized_pair_133.y;
        const int value_idx_134 = 2;
        const int acc_idx_135 = 8 + value_idx_134;
        float2 _f2_329 = make_float2(acc[acc_idx_135], acc[acc_idx_135 + 1]);
        float2 scaled_pair_136 = mul_f32x2_noftz(_f2_329, rsigma_pair);
        float2 _f2_330 = make_float2(wout[acc_idx_135], wout[acc_idx_135 + 1]);
        float2 normalized_pair_137 = mul_f32x2_noftz(scaled_pair_136, _f2_330);
        output_values_129[value_idx_134] = normalized_pair_137.x;
        output_values_129[value_idx_134 + 1] = normalized_pair_137.y;
        const int value_idx_138 = 4;
        const int acc_idx_139 = 8 + value_idx_138;
        float2 _f2_331 = make_float2(acc[acc_idx_139], acc[acc_idx_139 + 1]);
        float2 scaled_pair_140 = mul_f32x2_noftz(_f2_331, rsigma_pair);
        float2 _f2_332 = make_float2(wout[acc_idx_139], wout[acc_idx_139 + 1]);
        float2 normalized_pair_141 = mul_f32x2_noftz(scaled_pair_140, _f2_332);
        output_values_129[value_idx_138] = normalized_pair_141.x;
        output_values_129[value_idx_138 + 1] = normalized_pair_141.y;
        const int value_idx_142 = 6;
        const int acc_idx_143 = 8 + value_idx_142;
        float2 _f2_333 = make_float2(acc[acc_idx_143], acc[acc_idx_143 + 1]);
        float2 scaled_pair_144 = mul_f32x2_noftz(_f2_333, rsigma_pair);
        float2 _f2_334 = make_float2(wout[acc_idx_143], wout[acc_idx_143 + 1]);
        float2 normalized_pair_145 = mul_f32x2_noftz(scaled_pair_144, _f2_334);
        output_values_129[value_idx_142] = normalized_pair_145.x;
        output_values_129[value_idx_142 + 1] = normalized_pair_145.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_129[0 + 0], output_values_129[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_129[0 + 2], output_values_129[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_129[0 + 4], output_values_129[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_129[0 + 6], output_values_129[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_128 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_146 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_147[8];
        const int value_idx_148 = 0;
        const int acc_idx_149 = 16 + value_idx_148;
        float2 _f2_335 = make_float2(acc[acc_idx_149], acc[acc_idx_149 + 1]);
        float2 scaled_pair_150 = mul_f32x2_noftz(_f2_335, rsigma_pair);
        float2 _f2_336 = make_float2(wout[acc_idx_149], wout[acc_idx_149 + 1]);
        float2 normalized_pair_151 = mul_f32x2_noftz(scaled_pair_150, _f2_336);
        output_values_147[value_idx_148] = normalized_pair_151.x;
        output_values_147[value_idx_148 + 1] = normalized_pair_151.y;
        const int value_idx_152 = 2;
        const int acc_idx_153 = 16 + value_idx_152;
        float2 _f2_337 = make_float2(acc[acc_idx_153], acc[acc_idx_153 + 1]);
        float2 scaled_pair_154 = mul_f32x2_noftz(_f2_337, rsigma_pair);
        float2 _f2_338 = make_float2(wout[acc_idx_153], wout[acc_idx_153 + 1]);
        float2 normalized_pair_155 = mul_f32x2_noftz(scaled_pair_154, _f2_338);
        output_values_147[value_idx_152] = normalized_pair_155.x;
        output_values_147[value_idx_152 + 1] = normalized_pair_155.y;
        const int value_idx_156 = 4;
        const int acc_idx_157 = 16 + value_idx_156;
        float2 _f2_339 = make_float2(acc[acc_idx_157], acc[acc_idx_157 + 1]);
        float2 scaled_pair_158 = mul_f32x2_noftz(_f2_339, rsigma_pair);
        float2 _f2_340 = make_float2(wout[acc_idx_157], wout[acc_idx_157 + 1]);
        float2 normalized_pair_159 = mul_f32x2_noftz(scaled_pair_158, _f2_340);
        output_values_147[value_idx_156] = normalized_pair_159.x;
        output_values_147[value_idx_156 + 1] = normalized_pair_159.y;
        const int value_idx_160 = 6;
        const int acc_idx_161 = 16 + value_idx_160;
        float2 _f2_341 = make_float2(acc[acc_idx_161], acc[acc_idx_161 + 1]);
        float2 scaled_pair_162 = mul_f32x2_noftz(_f2_341, rsigma_pair);
        float2 _f2_342 = make_float2(wout[acc_idx_161], wout[acc_idx_161 + 1]);
        float2 normalized_pair_163 = mul_f32x2_noftz(scaled_pair_162, _f2_342);
        output_values_147[value_idx_160] = normalized_pair_163.x;
        output_values_147[value_idx_160 + 1] = normalized_pair_163.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_147[0 + 0], output_values_147[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_147[0 + 2], output_values_147[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_147[0 + 4], output_values_147[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_147[0 + 6], output_values_147[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_146 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_164 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_165[8];
        const int value_idx_166 = 0;
        const int acc_idx_167 = 24 + value_idx_166;
        float2 _f2_343 = make_float2(acc[acc_idx_167], acc[acc_idx_167 + 1]);
        float2 scaled_pair_168 = mul_f32x2_noftz(_f2_343, rsigma_pair);
        float2 _f2_344 = make_float2(wout[acc_idx_167], wout[acc_idx_167 + 1]);
        float2 normalized_pair_169 = mul_f32x2_noftz(scaled_pair_168, _f2_344);
        output_values_165[value_idx_166] = normalized_pair_169.x;
        output_values_165[value_idx_166 + 1] = normalized_pair_169.y;
        const int value_idx_170 = 2;
        const int acc_idx_171 = 24 + value_idx_170;
        float2 _f2_345 = make_float2(acc[acc_idx_171], acc[acc_idx_171 + 1]);
        float2 scaled_pair_172 = mul_f32x2_noftz(_f2_345, rsigma_pair);
        float2 _f2_346 = make_float2(wout[acc_idx_171], wout[acc_idx_171 + 1]);
        float2 normalized_pair_173 = mul_f32x2_noftz(scaled_pair_172, _f2_346);
        output_values_165[value_idx_170] = normalized_pair_173.x;
        output_values_165[value_idx_170 + 1] = normalized_pair_173.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_165[0 + 0], output_values_165[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_165[0 + 2], output_values_165[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_164]) = _pk2;
        }
    }
}

} // extern "C"
