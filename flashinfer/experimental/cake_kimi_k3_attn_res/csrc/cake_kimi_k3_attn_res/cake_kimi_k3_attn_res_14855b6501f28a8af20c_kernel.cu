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
#define SMEM_STATS_STAGE_BYTES 256
#define SMEM_STATS_STRIDE 256
#define SMEM_OUT_STATS_OFF 256
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 384
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
kernel_cake_kimi_k3_attn_res_14855b6501f28a8af20c(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 256);
    const int out_stats_addr = smem + 256;

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
        unsigned int words[56];
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
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_9[0 + 0] = _uv4_3.x;
                _vec_load_9[0 + 1] = _uv4_3.y;
                _vec_load_9[0 + 2] = _uv4_3.z;
                _vec_load_9[0 + 3] = _uv4_3.w;
            }
            words[42 + woff] = _vec_load_9[0];
            words[42 + woff + 1] = _vec_load_9[1];
            words[42 + woff + 2] = _vec_load_9[2];
            words[42 + woff + 3] = _vec_load_9[3];
        }
        {
            unsigned int _vec_load_12[4];
            {
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_12[0 + 0] = _uv4_4.x;
                _vec_load_12[0 + 1] = _uv4_4.y;
                _vec_load_12[0 + 2] = _uv4_4.z;
                _vec_load_12[0 + 3] = _uv4_4.w;
            }
            dwords[woff] = _vec_load_12[0];
            dwords[woff + 1] = _vec_load_12[1];
            dwords[woff + 2] = _vec_load_12[2];
            dwords[woff + 3] = _vec_load_12[3];
        }
        float _vec_load_15[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        float _vec_load_16[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        q[0] = _vec_load_15[0] * _vec_load_16[0];
        q[1] = _vec_load_15[1] * _vec_load_16[1];
        q[2] = _vec_load_15[2] * _vec_load_16[2];
        q[3] = _vec_load_15[3] * _vec_load_16[3];
        q[4] = _vec_load_15[4] * _vec_load_16[4];
        q[5] = _vec_load_15[5] * _vec_load_16[5];
        q[6] = _vec_load_15[6] * _vec_load_16[6];
        q[7] = _vec_load_15[7] * _vec_load_16[7];
        float _vec_load_17[8];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(output_norm_weight + base);
            uint4 _vld_7[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_7[_blk] = _vptr_7[_blk];
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
        }
        wout[0] = _vec_load_17[0];
        wout[1] = _vec_load_17[1];
        wout[2] = _vec_load_17[2];
        wout[3] = _vec_load_17[3];
        wout[4] = _vec_load_17[4];
        wout[5] = _vec_load_17[5];
        wout[6] = _vec_load_17[6];
        wout[7] = _vec_load_17[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_18[4];
            {
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_18[0 + 0] = _uv4_8.x;
                _vec_load_18[0 + 1] = _uv4_8.y;
                _vec_load_18[0 + 2] = _uv4_8.z;
                _vec_load_18[0 + 3] = _uv4_8.w;
            }
            words[woff_1] = _vec_load_18[0];
            words[woff_1 + 1] = _vec_load_18[1];
            words[woff_1 + 2] = _vec_load_18[2];
            words[woff_1 + 3] = _vec_load_18[3];
        }
        {
            unsigned int _vec_load_21[4];
            {
                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_21[0 + 0] = _uv4_9.x;
                _vec_load_21[0 + 1] = _uv4_9.y;
                _vec_load_21[0 + 2] = _uv4_9.z;
                _vec_load_21[0 + 3] = _uv4_9.w;
            }
            words[14 + woff_1] = _vec_load_21[0];
            words[14 + woff_1 + 1] = _vec_load_21[1];
            words[14 + woff_1 + 2] = _vec_load_21[2];
            words[14 + woff_1 + 3] = _vec_load_21[3];
        }
        {
            unsigned int _vec_load_24[4];
            {
                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_24[0 + 0] = _uv4_10.x;
                _vec_load_24[0 + 1] = _uv4_10.y;
                _vec_load_24[0 + 2] = _uv4_10.z;
                _vec_load_24[0 + 3] = _uv4_10.w;
            }
            words[28 + woff_1] = _vec_load_24[0];
            words[28 + woff_1 + 1] = _vec_load_24[1];
            words[28 + woff_1 + 2] = _vec_load_24[2];
            words[28 + woff_1 + 3] = _vec_load_24[3];
        }
        {
            unsigned int _vec_load_27[4];
            {
                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_27[0 + 0] = _uv4_11.x;
                _vec_load_27[0 + 1] = _uv4_11.y;
                _vec_load_27[0 + 2] = _uv4_11.z;
                _vec_load_27[0 + 3] = _uv4_11.w;
            }
            words[42 + woff_1] = _vec_load_27[0];
            words[42 + woff_1 + 1] = _vec_load_27[1];
            words[42 + woff_1 + 2] = _vec_load_27[2];
            words[42 + woff_1 + 3] = _vec_load_27[3];
        }
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_30[0 + 0] = _uv4_12.x;
                _vec_load_30[0 + 1] = _uv4_12.y;
                _vec_load_30[0 + 2] = _uv4_12.z;
                _vec_load_30[0 + 3] = _uv4_12.w;
            }
            dwords[woff_1] = _vec_load_30[0];
            dwords[woff_1 + 1] = _vec_load_30[1];
            dwords[woff_1 + 2] = _vec_load_30[2];
            dwords[woff_1 + 3] = _vec_load_30[3];
        }
        float _vec_load_33[8];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_33[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_33[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_13[_pair]));
                }
            }
        }
        float _vec_load_34[8];
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(qk_weight + base_0);
            uint4 _vld_14[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_14[_blk] = _vptr_14[_blk];
                uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_34[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_34[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_14[_pair]));
                }
            }
        }
        q[8] = _vec_load_33[0] * _vec_load_34[0];
        q[9] = _vec_load_33[1] * _vec_load_34[1];
        q[10] = _vec_load_33[2] * _vec_load_34[2];
        q[11] = _vec_load_33[3] * _vec_load_34[3];
        q[12] = _vec_load_33[4] * _vec_load_34[4];
        q[13] = _vec_load_33[5] * _vec_load_34[5];
        q[14] = _vec_load_33[6] * _vec_load_34[6];
        q[15] = _vec_load_33[7] * _vec_load_34[7];
        float _vec_load_35[8];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_35[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_35[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
        }
        wout[8] = _vec_load_35[0];
        wout[9] = _vec_load_35[1];
        wout[10] = _vec_load_35[2];
        wout[11] = _vec_load_35[3];
        wout[12] = _vec_load_35[4];
        wout[13] = _vec_load_35[5];
        wout[14] = _vec_load_35[6];
        wout[15] = _vec_load_35[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_36[0 + 0] = _uv4_16.x;
                _vec_load_36[0 + 1] = _uv4_16.y;
                _vec_load_36[0 + 2] = _uv4_16.z;
                _vec_load_36[0 + 3] = _uv4_16.w;
            }
            words[woff_3] = _vec_load_36[0];
            words[woff_3 + 1] = _vec_load_36[1];
            words[woff_3 + 2] = _vec_load_36[2];
            words[woff_3 + 3] = _vec_load_36[3];
        }
        {
            unsigned int _vec_load_39[4];
            {
                uint4 _uv4_17 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_39[0 + 0] = _uv4_17.x;
                _vec_load_39[0 + 1] = _uv4_17.y;
                _vec_load_39[0 + 2] = _uv4_17.z;
                _vec_load_39[0 + 3] = _uv4_17.w;
            }
            words[14 + woff_3] = _vec_load_39[0];
            words[14 + woff_3 + 1] = _vec_load_39[1];
            words[14 + woff_3 + 2] = _vec_load_39[2];
            words[14 + woff_3 + 3] = _vec_load_39[3];
        }
        {
            unsigned int _vec_load_42[4];
            {
                uint4 _uv4_18 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_42[0 + 0] = _uv4_18.x;
                _vec_load_42[0 + 1] = _uv4_18.y;
                _vec_load_42[0 + 2] = _uv4_18.z;
                _vec_load_42[0 + 3] = _uv4_18.w;
            }
            words[28 + woff_3] = _vec_load_42[0];
            words[28 + woff_3 + 1] = _vec_load_42[1];
            words[28 + woff_3 + 2] = _vec_load_42[2];
            words[28 + woff_3 + 3] = _vec_load_42[3];
        }
        {
            unsigned int _vec_load_45[4];
            {
                uint4 _uv4_19 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_45[0 + 0] = _uv4_19.x;
                _vec_load_45[0 + 1] = _uv4_19.y;
                _vec_load_45[0 + 2] = _uv4_19.z;
                _vec_load_45[0 + 3] = _uv4_19.w;
            }
            words[42 + woff_3] = _vec_load_45[0];
            words[42 + woff_3 + 1] = _vec_load_45[1];
            words[42 + woff_3 + 2] = _vec_load_45[2];
            words[42 + woff_3 + 3] = _vec_load_45[3];
        }
        {
            unsigned int _vec_load_48[4];
            {
                uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_48[0 + 0] = _uv4_20.x;
                _vec_load_48[0 + 1] = _uv4_20.y;
                _vec_load_48[0 + 2] = _uv4_20.z;
                _vec_load_48[0 + 3] = _uv4_20.w;
            }
            dwords[woff_3] = _vec_load_48[0];
            dwords[woff_3 + 1] = _vec_load_48[1];
            dwords[woff_3 + 2] = _vec_load_48[2];
            dwords[woff_3 + 3] = _vec_load_48[3];
        }
        float _vec_load_51[8];
        {
            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_21[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_21[_blk] = _vptr_21[_blk];
                uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_51[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_51[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_21[_pair]));
                }
            }
        }
        float _vec_load_52[8];
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_22[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_22[_blk] = _vptr_22[_blk];
                uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_52[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_52[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_22[_pair]));
                }
            }
        }
        q[16] = _vec_load_51[0] * _vec_load_52[0];
        q[17] = _vec_load_51[1] * _vec_load_52[1];
        q[18] = _vec_load_51[2] * _vec_load_52[2];
        q[19] = _vec_load_51[3] * _vec_load_52[3];
        q[20] = _vec_load_51[4] * _vec_load_52[4];
        q[21] = _vec_load_51[5] * _vec_load_52[5];
        q[22] = _vec_load_51[6] * _vec_load_52[6];
        q[23] = _vec_load_51[7] * _vec_load_52[7];
        float _vec_load_53[8];
        {
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_23[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_23[_blk] = _vptr_23[_blk];
                uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_53[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_53[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_23[_pair]));
                }
            }
        }
        wout[16] = _vec_load_53[0];
        wout[17] = _vec_load_53[1];
        wout[18] = _vec_load_53[2];
        wout[19] = _vec_load_53[3];
        wout[20] = _vec_load_53[4];
        wout[21] = _vec_load_53[5];
        wout[22] = _vec_load_53[6];
        wout[23] = _vec_load_53[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_55[1];
            {
                _vec_load_55[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_55[0];
            unsigned int _vec_load_56[1];
            {
                _vec_load_56[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_56[0];
        }
        {
            unsigned int _vec_load_58[1];
            {
                _vec_load_58[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_58[0];
            unsigned int _vec_load_59[1];
            {
                _vec_load_59[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_59[0];
        }
        {
            unsigned int _vec_load_61[1];
            {
                _vec_load_61[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_61[0];
            unsigned int _vec_load_62[1];
            {
                _vec_load_62[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_62[0];
        }
        {
            unsigned int _vec_load_64[1];
            {
                _vec_load_64[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[42 + woff_5] = _vec_load_64[0];
            unsigned int _vec_load_65[1];
            {
                _vec_load_65[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[42 + woff_5 + 1] = _vec_load_65[0];
        }
        {
            unsigned int _vec_load_67[1];
            {
                _vec_load_67[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_67[0];
            unsigned int _vec_load_68[1];
            {
                _vec_load_68[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_68[0];
        }
        float _vec_load_69[4];
        {
            uint2 _vld_24;
            _vld_24 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_69[0 + _pair * 2])[0]), "=f"((&_vec_load_69[0 + _pair * 2])[1])
                    : "r"(_vpairs_24[_pair]));
            }
        }
        float _vec_load_70[4];
        {
            uint2 _vld_25;
            _vld_25 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_70[0 + _pair * 2])[0]), "=f"((&_vec_load_70[0 + _pair * 2])[1])
                    : "r"(_vpairs_25[_pair]));
            }
        }
        q[24] = _vec_load_69[0] * _vec_load_70[0];
        q[25] = _vec_load_69[1] * _vec_load_70[1];
        q[26] = _vec_load_69[2] * _vec_load_70[2];
        q[27] = _vec_load_69[3] * _vec_load_70[3];
        float _vec_load_71[4];
        {
            uint2 _vld_26;
            _vld_26 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_71[0 + _pair * 2])[0]), "=f"((&_vec_load_71[0 + _pair * 2])[1])
                    : "r"(_vpairs_26[_pair]));
            }
        }
        wout[24] = _vec_load_71[0];
        wout[25] = _vec_load_71[1];
        wout[26] = _vec_load_71[2];
        wout[27] = _vec_load_71[3];
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
        float2 sq[4];
        float2 dot[4];
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
        float2 _f2_6 = make_float2(0.0f, 0.0f);
        sq[3] = _f2_6;
        float2 _f2_7 = make_float2(0.0f, 0.0f);
        dot[3] = _f2_7;
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
            float2 _f2_14 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_14;
            float2 _f2_15 = make_float2(q[0], q[1]);
            float2 qp = _f2_15;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_16 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_16;
            float2 _f2_17 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_17;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_18 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_18;
            float2 _f2_19 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_19;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_20 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_20;
            float2 _f2_21 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_21;
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
            float2 _f2_28 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_28;
            float2 _f2_29 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_29;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_30 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_30;
            float2 _f2_31 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_31;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_32 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_32;
            float2 _f2_33 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_33;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_34 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_34;
            float2 _f2_35 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_35;
            sq[1] = fma_f32x2_rn_noftz(v_4_1, v_4_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_1, qp_5_1, dot[1]);
        }
        unsigned int sw_9[4];
        sw_9[0] = words[28 + woff_7];
        sw_9[1] = words[28 + woff_7 + 1];
        sw_9[2] = words[28 + woff_7 + 2];
        sw_9[3] = words[28 + woff_7 + 3];
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
            float2 _f2_42 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_42;
            float2 _f2_43 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_43;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_44 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_44;
            float2 _f2_45 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_45;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_46 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_46;
            float2 _f2_47 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_47;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_48 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_48;
            float2 _f2_49 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_49;
            sq[2] = fma_f32x2_rn_noftz(v_4_2, v_4_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_2, qp_5_2, dot[2]);
        }
        unsigned int sw_10[4];
        sw_10[0] = words[42 + woff_7];
        sw_10[1] = words[42 + woff_7 + 1];
        sw_10[2] = words[42 + woff_7 + 2];
        sw_10[3] = words[42 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_10[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_10[0] = __as_u32(mixed);
            words[42 + woff_7] = sw_10[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_10[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_10[1] = __as_u32(mixed_2);
            words[42 + woff_7 + 1] = sw_10[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_10[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_10[2] = __as_u32(mixed_5);
            words[42 + woff_7 + 2] = sw_10[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_10[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_10[3] = __as_u32(mixed_8);
            words[42 + woff_7 + 3] = sw_10[3];
            {
                int4 _iv4 = make_int4(sw_10[0 + 0], sw_10[0 + 1], sw_10[0 + 2], sw_10[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
        float sw_10_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_10_f32[_pair * 2])[0]), "=f"((&sw_10_f32[_pair * 2])[1])
                : "r"(sw_10[_pair]));
        }
        {
            float2 _f2_56 = make_float2(sw_10_f32[0], sw_10_f32[1]);
            float2 v_5 = _f2_56;
            float2 _f2_57 = make_float2(q[0], q[1]);
            float2 qp_6 = _f2_57;
            sq[3] = fma_f32x2_rn_noftz(v_5, v_5, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_5, qp_6, dot[3]);
            float2 _f2_58 = make_float2(sw_10_f32[2], sw_10_f32[3]);
            float2 v_0_3 = _f2_58;
            float2 _f2_59 = make_float2(q[2], q[3]);
            float2 qp_1_3 = _f2_59;
            sq[3] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[3]);
            float2 _f2_60 = make_float2(sw_10_f32[4], sw_10_f32[5]);
            float2 v_2_3 = _f2_60;
            float2 _f2_61 = make_float2(q[4], q[5]);
            float2 qp_3_3 = _f2_61;
            sq[3] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[3]);
            float2 _f2_62 = make_float2(sw_10_f32[6], sw_10_f32[7]);
            float2 v_4_3 = _f2_62;
            float2 _f2_63 = make_float2(q[6], q[7]);
            float2 qp_5_3 = _f2_63;
            sq[3] = fma_f32x2_rn_noftz(v_4_3, v_4_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_3, qp_5_3, dot[3]);
        }
        int base_11 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_12 = 4;
        unsigned int sw_13[4];
        sw_13[0] = words[woff_12];
        sw_13[1] = words[woff_12 + 1];
        sw_13[2] = words[woff_12 + 2];
        sw_13[3] = words[woff_12 + 3];
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
            fsrc[8] = sw_13_f32[0];
            fsrc[9] = sw_13_f32[1];
            fsrc[10] = sw_13_f32[2];
            fsrc[11] = sw_13_f32[3];
            fsrc[12] = sw_13_f32[4];
            fsrc[13] = sw_13_f32[5];
            fsrc[14] = sw_13_f32[6];
            fsrc[15] = sw_13_f32[7];
        }
        {
            float2 _f2_70 = make_float2(sw_13_f32[0], sw_13_f32[1]);
            float2 v_6 = _f2_70;
            float2 _f2_71 = make_float2(q[8], q[9]);
            float2 qp_7 = _f2_71;
            sq[0] = fma_f32x2_rn_noftz(v_6, v_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_6, qp_7, dot[0]);
            float2 _f2_72 = make_float2(sw_13_f32[2], sw_13_f32[3]);
            float2 v_0_4 = _f2_72;
            float2 _f2_73 = make_float2(q[10], q[11]);
            float2 qp_1_4 = _f2_73;
            sq[0] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[0]);
            float2 _f2_74 = make_float2(sw_13_f32[4], sw_13_f32[5]);
            float2 v_2_4 = _f2_74;
            float2 _f2_75 = make_float2(q[12], q[13]);
            float2 qp_3_4 = _f2_75;
            sq[0] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[0]);
            float2 _f2_76 = make_float2(sw_13_f32[6], sw_13_f32[7]);
            float2 v_4_4 = _f2_76;
            float2 _f2_77 = make_float2(q[14], q[15]);
            float2 qp_5_4 = _f2_77;
            sq[0] = fma_f32x2_rn_noftz(v_4_4, v_4_4, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_4, qp_5_4, dot[0]);
        }
        unsigned int sw_14[4];
        sw_14[0] = words[14 + woff_12];
        sw_14[1] = words[14 + woff_12 + 1];
        sw_14[2] = words[14 + woff_12 + 2];
        sw_14[3] = words[14 + woff_12 + 3];
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
            fsrc[36] = sw_14_f32[0];
            fsrc[37] = sw_14_f32[1];
            fsrc[38] = sw_14_f32[2];
            fsrc[39] = sw_14_f32[3];
            fsrc[40] = sw_14_f32[4];
            fsrc[41] = sw_14_f32[5];
            fsrc[42] = sw_14_f32[6];
            fsrc[43] = sw_14_f32[7];
        }
        {
            float2 _f2_84 = make_float2(sw_14_f32[0], sw_14_f32[1]);
            float2 v_7 = _f2_84;
            float2 _f2_85 = make_float2(q[8], q[9]);
            float2 qp_8 = _f2_85;
            sq[1] = fma_f32x2_rn_noftz(v_7, v_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_7, qp_8, dot[1]);
            float2 _f2_86 = make_float2(sw_14_f32[2], sw_14_f32[3]);
            float2 v_0_5 = _f2_86;
            float2 _f2_87 = make_float2(q[10], q[11]);
            float2 qp_1_5 = _f2_87;
            sq[1] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[1]);
            float2 _f2_88 = make_float2(sw_14_f32[4], sw_14_f32[5]);
            float2 v_2_5 = _f2_88;
            float2 _f2_89 = make_float2(q[12], q[13]);
            float2 qp_3_5 = _f2_89;
            sq[1] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[1]);
            float2 _f2_90 = make_float2(sw_14_f32[6], sw_14_f32[7]);
            float2 v_4_5 = _f2_90;
            float2 _f2_91 = make_float2(q[14], q[15]);
            float2 qp_5_5 = _f2_91;
            sq[1] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[1]);
        }
        unsigned int sw_15[4];
        sw_15[0] = words[28 + woff_12];
        sw_15[1] = words[28 + woff_12 + 1];
        sw_15[2] = words[28 + woff_12 + 2];
        sw_15[3] = words[28 + woff_12 + 3];
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
            fsrc[64] = sw_15_f32[0];
            fsrc[65] = sw_15_f32[1];
            fsrc[66] = sw_15_f32[2];
            fsrc[67] = sw_15_f32[3];
            fsrc[68] = sw_15_f32[4];
            fsrc[69] = sw_15_f32[5];
            fsrc[70] = sw_15_f32[6];
            fsrc[71] = sw_15_f32[7];
        }
        {
            float2 _f2_98 = make_float2(sw_15_f32[0], sw_15_f32[1]);
            float2 v_8 = _f2_98;
            float2 _f2_99 = make_float2(q[8], q[9]);
            float2 qp_9 = _f2_99;
            sq[2] = fma_f32x2_rn_noftz(v_8, v_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_8, qp_9, dot[2]);
            float2 _f2_100 = make_float2(sw_15_f32[2], sw_15_f32[3]);
            float2 v_0_6 = _f2_100;
            float2 _f2_101 = make_float2(q[10], q[11]);
            float2 qp_1_6 = _f2_101;
            sq[2] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[2]);
            float2 _f2_102 = make_float2(sw_15_f32[4], sw_15_f32[5]);
            float2 v_2_6 = _f2_102;
            float2 _f2_103 = make_float2(q[12], q[13]);
            float2 qp_3_6 = _f2_103;
            sq[2] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[2]);
            float2 _f2_104 = make_float2(sw_15_f32[6], sw_15_f32[7]);
            float2 v_4_6 = _f2_104;
            float2 _f2_105 = make_float2(q[14], q[15]);
            float2 qp_5_6 = _f2_105;
            sq[2] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[2]);
        }
        unsigned int sw_16[4];
        sw_16[0] = words[42 + woff_12];
        sw_16[1] = words[42 + woff_12 + 1];
        sw_16[2] = words[42 + woff_12 + 2];
        sw_16[3] = words[42 + woff_12 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_16[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_12]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_16[0] = __as_u32(mixed_1);
            words[42 + woff_12] = sw_16[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_16[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_12 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_16[1] = __as_u32(mixed_2_1);
            words[42 + woff_12 + 1] = sw_16[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_16[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_12 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_16[2] = __as_u32(mixed_5_1);
            words[42 + woff_12 + 2] = sw_16[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_16[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_12 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_16[3] = __as_u32(mixed_8_1);
            words[42 + woff_12 + 3] = sw_16[3];
            {
                int4 _iv4 = make_int4(sw_16[0 + 0], sw_16[0 + 1], sw_16[0 + 2], sw_16[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_11)) + 0) = _iv4;
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
            float2 _f2_112 = make_float2(sw_16_f32[0], sw_16_f32[1]);
            float2 v_9 = _f2_112;
            float2 _f2_113 = make_float2(q[8], q[9]);
            float2 qp_10 = _f2_113;
            sq[3] = fma_f32x2_rn_noftz(v_9, v_9, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_9, qp_10, dot[3]);
            float2 _f2_114 = make_float2(sw_16_f32[2], sw_16_f32[3]);
            float2 v_0_7 = _f2_114;
            float2 _f2_115 = make_float2(q[10], q[11]);
            float2 qp_1_7 = _f2_115;
            sq[3] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[3]);
            float2 _f2_116 = make_float2(sw_16_f32[4], sw_16_f32[5]);
            float2 v_2_7 = _f2_116;
            float2 _f2_117 = make_float2(q[12], q[13]);
            float2 qp_3_7 = _f2_117;
            sq[3] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[3]);
            float2 _f2_118 = make_float2(sw_16_f32[6], sw_16_f32[7]);
            float2 v_4_7 = _f2_118;
            float2 _f2_119 = make_float2(q[14], q[15]);
            float2 qp_5_7 = _f2_119;
            sq[3] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[3]);
        }
        int base_17 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_18 = 8;
        unsigned int sw_19[4];
        sw_19[0] = words[woff_18];
        sw_19[1] = words[woff_18 + 1];
        sw_19[2] = words[woff_18 + 2];
        sw_19[3] = words[woff_18 + 3];
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
            fsrc[16] = sw_19_f32[0];
            fsrc[17] = sw_19_f32[1];
            fsrc[18] = sw_19_f32[2];
            fsrc[19] = sw_19_f32[3];
            fsrc[20] = sw_19_f32[4];
            fsrc[21] = sw_19_f32[5];
            fsrc[22] = sw_19_f32[6];
            fsrc[23] = sw_19_f32[7];
        }
        {
            float2 _f2_126 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_10 = _f2_126;
            float2 _f2_127 = make_float2(q[16], q[17]);
            float2 qp_11 = _f2_127;
            sq[0] = fma_f32x2_rn_noftz(v_10, v_10, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_10, qp_11, dot[0]);
            float2 _f2_128 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_8 = _f2_128;
            float2 _f2_129 = make_float2(q[18], q[19]);
            float2 qp_1_8 = _f2_129;
            sq[0] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[0]);
            float2 _f2_130 = make_float2(sw_19_f32[4], sw_19_f32[5]);
            float2 v_2_8 = _f2_130;
            float2 _f2_131 = make_float2(q[20], q[21]);
            float2 qp_3_8 = _f2_131;
            sq[0] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[0]);
            float2 _f2_132 = make_float2(sw_19_f32[6], sw_19_f32[7]);
            float2 v_4_8 = _f2_132;
            float2 _f2_133 = make_float2(q[22], q[23]);
            float2 qp_5_8 = _f2_133;
            sq[0] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[0]);
        }
        unsigned int sw_20[4];
        sw_20[0] = words[14 + woff_18];
        sw_20[1] = words[14 + woff_18 + 1];
        sw_20[2] = words[14 + woff_18 + 2];
        sw_20[3] = words[14 + woff_18 + 3];
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
            fsrc[44] = sw_20_f32[0];
            fsrc[45] = sw_20_f32[1];
            fsrc[46] = sw_20_f32[2];
            fsrc[47] = sw_20_f32[3];
            fsrc[48] = sw_20_f32[4];
            fsrc[49] = sw_20_f32[5];
            fsrc[50] = sw_20_f32[6];
            fsrc[51] = sw_20_f32[7];
        }
        {
            float2 _f2_140 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_11 = _f2_140;
            float2 _f2_141 = make_float2(q[16], q[17]);
            float2 qp_12 = _f2_141;
            sq[1] = fma_f32x2_rn_noftz(v_11, v_11, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_11, qp_12, dot[1]);
            float2 _f2_142 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_0_9 = _f2_142;
            float2 _f2_143 = make_float2(q[18], q[19]);
            float2 qp_1_9 = _f2_143;
            sq[1] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_9, qp_1_9, dot[1]);
            float2 _f2_144 = make_float2(sw_20_f32[4], sw_20_f32[5]);
            float2 v_2_9 = _f2_144;
            float2 _f2_145 = make_float2(q[20], q[21]);
            float2 qp_3_9 = _f2_145;
            sq[1] = fma_f32x2_rn_noftz(v_2_9, v_2_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[1]);
            float2 _f2_146 = make_float2(sw_20_f32[6], sw_20_f32[7]);
            float2 v_4_9 = _f2_146;
            float2 _f2_147 = make_float2(q[22], q[23]);
            float2 qp_5_9 = _f2_147;
            sq[1] = fma_f32x2_rn_noftz(v_4_9, v_4_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_9, qp_5_9, dot[1]);
        }
        unsigned int sw_21[4];
        sw_21[0] = words[28 + woff_18];
        sw_21[1] = words[28 + woff_18 + 1];
        sw_21[2] = words[28 + woff_18 + 2];
        sw_21[3] = words[28 + woff_18 + 3];
        float sw_21_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_21_f32[_pair * 2])[0]), "=f"((&sw_21_f32[_pair * 2])[1])
                : "r"(sw_21[_pair]));
        }
        {
            fsrc[72] = sw_21_f32[0];
            fsrc[73] = sw_21_f32[1];
            fsrc[74] = sw_21_f32[2];
            fsrc[75] = sw_21_f32[3];
            fsrc[76] = sw_21_f32[4];
            fsrc[77] = sw_21_f32[5];
            fsrc[78] = sw_21_f32[6];
            fsrc[79] = sw_21_f32[7];
        }
        {
            float2 _f2_154 = make_float2(sw_21_f32[0], sw_21_f32[1]);
            float2 v_12 = _f2_154;
            float2 _f2_155 = make_float2(q[16], q[17]);
            float2 qp_13 = _f2_155;
            sq[2] = fma_f32x2_rn_noftz(v_12, v_12, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_12, qp_13, dot[2]);
            float2 _f2_156 = make_float2(sw_21_f32[2], sw_21_f32[3]);
            float2 v_0_10 = _f2_156;
            float2 _f2_157 = make_float2(q[18], q[19]);
            float2 qp_1_10 = _f2_157;
            sq[2] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_10, qp_1_10, dot[2]);
            float2 _f2_158 = make_float2(sw_21_f32[4], sw_21_f32[5]);
            float2 v_2_10 = _f2_158;
            float2 _f2_159 = make_float2(q[20], q[21]);
            float2 qp_3_10 = _f2_159;
            sq[2] = fma_f32x2_rn_noftz(v_2_10, v_2_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[2]);
            float2 _f2_160 = make_float2(sw_21_f32[6], sw_21_f32[7]);
            float2 v_4_10 = _f2_160;
            float2 _f2_161 = make_float2(q[22], q[23]);
            float2 qp_5_10 = _f2_161;
            sq[2] = fma_f32x2_rn_noftz(v_4_10, v_4_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_10, qp_5_10, dot[2]);
        }
        unsigned int sw_22[4];
        sw_22[0] = words[42 + woff_18];
        sw_22[1] = words[42 + woff_18 + 1];
        sw_22[2] = words[42 + woff_18 + 2];
        sw_22[3] = words[42 + woff_18 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_22[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_18]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_22[0] = __as_u32(mixed_3);
            words[42 + woff_18] = sw_22[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_22[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_18 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_22[1] = __as_u32(mixed_2_2);
            words[42 + woff_18 + 1] = sw_22[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_22[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_18 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_22[2] = __as_u32(mixed_5_2);
            words[42 + woff_18 + 2] = sw_22[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_22[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_18 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_22[3] = __as_u32(mixed_8_2);
            words[42 + woff_18 + 3] = sw_22[3];
            {
                int4 _iv4 = make_int4(sw_22[0 + 0], sw_22[0 + 1], sw_22[0 + 2], sw_22[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_17)) + 0) = _iv4;
            }
        }
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
            float2 _f2_168 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_13 = _f2_168;
            float2 _f2_169 = make_float2(q[16], q[17]);
            float2 qp_14 = _f2_169;
            sq[3] = fma_f32x2_rn_noftz(v_13, v_13, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_13, qp_14, dot[3]);
            float2 _f2_170 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_0_11 = _f2_170;
            float2 _f2_171 = make_float2(q[18], q[19]);
            float2 qp_1_11 = _f2_171;
            sq[3] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_11, qp_1_11, dot[3]);
            float2 _f2_172 = make_float2(sw_22_f32[4], sw_22_f32[5]);
            float2 v_2_11 = _f2_172;
            float2 _f2_173 = make_float2(q[20], q[21]);
            float2 qp_3_11 = _f2_173;
            sq[3] = fma_f32x2_rn_noftz(v_2_11, v_2_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[3]);
            float2 _f2_174 = make_float2(sw_22_f32[6], sw_22_f32[7]);
            float2 v_4_11 = _f2_174;
            float2 _f2_175 = make_float2(q[22], q[23]);
            float2 qp_5_11 = _f2_175;
            sq[3] = fma_f32x2_rn_noftz(v_4_11, v_4_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_11, qp_5_11, dot[3]);
        }
        int base_23 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_24 = 12;
        unsigned int sw_25[4];
        sw_25[0] = words[woff_24];
        sw_25[1] = words[woff_24 + 1];
        float sw_25_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_25_f32[_pair * 2])[0]), "=f"((&sw_25_f32[_pair * 2])[1])
                : "r"(sw_25[_pair]));
        }
        {
            fsrc[24] = sw_25_f32[0];
            fsrc[25] = sw_25_f32[1];
            fsrc[26] = sw_25_f32[2];
            fsrc[27] = sw_25_f32[3];
        }
        {
            float2 _f2_176 = make_float2(sw_25_f32[0], sw_25_f32[1]);
            float2 v_14 = _f2_176;
            sq[0] = fma_f32x2_rn_noftz(v_14, v_14, sq[0]);
            float2 _f2_177 = make_float2(sw_25_f32[2], sw_25_f32[3]);
            float2 v_0_12 = _f2_177;
            sq[0] = fma_f32x2_rn_noftz(v_0_12, v_0_12, sq[0]);
            float2 _f2_178 = make_float2(sw_25_f32[0], sw_25_f32[1]);
            float2 v_1_1 = _f2_178;
            float2 _f2_179 = make_float2(q[24], q[25]);
            float2 qp_15 = _f2_179;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_15, dot[0]);
            float2 _f2_180 = make_float2(sw_25_f32[2], sw_25_f32[3]);
            float2 v_2_12 = _f2_180;
            float2 _f2_181 = make_float2(q[26], q[27]);
            float2 qp_3_12 = _f2_181;
            dot[0] = fma_f32x2_rn_noftz(v_2_12, qp_3_12, dot[0]);
        }
        unsigned int sw_26[4];
        sw_26[0] = words[14 + woff_24];
        sw_26[1] = words[14 + woff_24 + 1];
        float sw_26_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_26_f32[_pair * 2])[0]), "=f"((&sw_26_f32[_pair * 2])[1])
                : "r"(sw_26[_pair]));
        }
        {
            fsrc[52] = sw_26_f32[0];
            fsrc[53] = sw_26_f32[1];
            fsrc[54] = sw_26_f32[2];
            fsrc[55] = sw_26_f32[3];
        }
        {
            float2 _f2_190 = make_float2(sw_26_f32[0], sw_26_f32[1]);
            float2 v_15 = _f2_190;
            sq[1] = fma_f32x2_rn_noftz(v_15, v_15, sq[1]);
            float2 _f2_191 = make_float2(sw_26_f32[2], sw_26_f32[3]);
            float2 v_0_13 = _f2_191;
            sq[1] = fma_f32x2_rn_noftz(v_0_13, v_0_13, sq[1]);
            float2 _f2_192 = make_float2(sw_26_f32[0], sw_26_f32[1]);
            float2 v_1_2 = _f2_192;
            float2 _f2_193 = make_float2(q[24], q[25]);
            float2 qp_16 = _f2_193;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_16, dot[1]);
            float2 _f2_194 = make_float2(sw_26_f32[2], sw_26_f32[3]);
            float2 v_2_13 = _f2_194;
            float2 _f2_195 = make_float2(q[26], q[27]);
            float2 qp_3_13 = _f2_195;
            dot[1] = fma_f32x2_rn_noftz(v_2_13, qp_3_13, dot[1]);
        }
        unsigned int sw_27[4];
        sw_27[0] = words[28 + woff_24];
        sw_27[1] = words[28 + woff_24 + 1];
        float sw_27_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_27_f32[_pair * 2])[0]), "=f"((&sw_27_f32[_pair * 2])[1])
                : "r"(sw_27[_pair]));
        }
        {
            fsrc[80] = sw_27_f32[0];
            fsrc[81] = sw_27_f32[1];
            fsrc[82] = sw_27_f32[2];
            fsrc[83] = sw_27_f32[3];
        }
        {
            float2 _f2_204 = make_float2(sw_27_f32[0], sw_27_f32[1]);
            float2 v_16 = _f2_204;
            sq[2] = fma_f32x2_rn_noftz(v_16, v_16, sq[2]);
            float2 _f2_205 = make_float2(sw_27_f32[2], sw_27_f32[3]);
            float2 v_0_14 = _f2_205;
            sq[2] = fma_f32x2_rn_noftz(v_0_14, v_0_14, sq[2]);
            float2 _f2_206 = make_float2(sw_27_f32[0], sw_27_f32[1]);
            float2 v_1_3 = _f2_206;
            float2 _f2_207 = make_float2(q[24], q[25]);
            float2 qp_17 = _f2_207;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_17, dot[2]);
            float2 _f2_208 = make_float2(sw_27_f32[2], sw_27_f32[3]);
            float2 v_2_14 = _f2_208;
            float2 _f2_209 = make_float2(q[26], q[27]);
            float2 qp_3_14 = _f2_209;
            dot[2] = fma_f32x2_rn_noftz(v_2_14, qp_3_14, dot[2]);
        }
        unsigned int sw_28[4];
        sw_28[0] = words[42 + woff_24];
        sw_28[1] = words[42 + woff_24 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_28[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_24]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_28[0] = __as_u32(mixed_4);
            words[42 + woff_24] = sw_28[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_28[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_24 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_28[1] = __as_u32(mixed_2_3);
            words[42 + woff_24 + 1] = sw_28[1];
            {
                int2 _iv2 = make_int2(sw_28[0 + 0], sw_28[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_23)) + 0) = _iv2;
            }
        }
        float sw_28_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_28_f32[_pair * 2])[0]), "=f"((&sw_28_f32[_pair * 2])[1])
                : "r"(sw_28[_pair]));
        }
        {
            float2 _f2_218 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_17 = _f2_218;
            sq[3] = fma_f32x2_rn_noftz(v_17, v_17, sq[3]);
            float2 _f2_219 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_0_15 = _f2_219;
            sq[3] = fma_f32x2_rn_noftz(v_0_15, v_0_15, sq[3]);
            float2 _f2_220 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_1_4 = _f2_220;
            float2 _f2_221 = make_float2(q[24], q[25]);
            float2 qp_18 = _f2_221;
            dot[3] = fma_f32x2_rn_noftz(v_1_4, qp_18, dot[3]);
            float2 _f2_222 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_2_15 = _f2_222;
            float2 _f2_223 = make_float2(q[26], q[27]);
            float2 qp_3_15 = _f2_223;
            dot[3] = fma_f32x2_rn_noftz(v_2_15, qp_3_15, dot[3]);
        }
        float2 pairs[4];
        float2 _f2_232 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_232;
        float2 _f2_233 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_233;
        float2 _f2_234 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_234;
        float2 _f2_235 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_235;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_236 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_236;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_29 = 0;
        bits_29 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_29, 16);
        unsigned long long peerbits_30 = _shfl_xor_1;
        float2 _f2_237 = make_float2(0.0f, 0.0f);
        float2 peer_31 = _f2_237;
        peer_31 = reinterpret_cast<float2*>(&peerbits_30)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_31);
        unsigned long long bits_32 = 0;
        bits_32 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_32, 16);
        unsigned long long peerbits_33 = _shfl_xor_2;
        float2 _f2_238 = make_float2(0.0f, 0.0f);
        float2 peer_34 = _f2_238;
        peer_34 = reinterpret_cast<float2*>(&peerbits_33)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_34);
        unsigned long long bits_35 = 0;
        bits_35 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_35, 16);
        unsigned long long peerbits_36 = _shfl_xor_3;
        float2 _f2_239 = make_float2(0.0f, 0.0f);
        float2 peer_37 = _f2_239;
        peer_37 = reinterpret_cast<float2*>(&peerbits_36)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_37);
        unsigned long long bits_38 = 0;
        bits_38 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_38, 8);
        unsigned long long peerbits_39 = _shfl_xor_4;
        float2 _f2_240 = make_float2(0.0f, 0.0f);
        float2 peer_40 = _f2_240;
        peer_40 = reinterpret_cast<float2*>(&peerbits_39)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_40);
        unsigned long long bits_41 = 0;
        bits_41 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_41, 8);
        unsigned long long peerbits_42 = _shfl_xor_5;
        float2 _f2_241 = make_float2(0.0f, 0.0f);
        float2 peer_43 = _f2_241;
        peer_43 = reinterpret_cast<float2*>(&peerbits_42)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_43);
        unsigned long long bits_44 = 0;
        bits_44 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_44, 8);
        unsigned long long peerbits_45 = _shfl_xor_6;
        float2 _f2_242 = make_float2(0.0f, 0.0f);
        float2 peer_46 = _f2_242;
        peer_46 = reinterpret_cast<float2*>(&peerbits_45)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_46);
        unsigned long long bits_47 = 0;
        bits_47 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_47, 8);
        unsigned long long peerbits_48 = _shfl_xor_7;
        float2 _f2_243 = make_float2(0.0f, 0.0f);
        float2 peer_49 = _f2_243;
        peer_49 = reinterpret_cast<float2*>(&peerbits_48)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_49);
        unsigned long long bits_50 = 0;
        bits_50 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_50, 4);
        unsigned long long peerbits_51 = _shfl_xor_8;
        float2 _f2_244 = make_float2(0.0f, 0.0f);
        float2 peer_52 = _f2_244;
        peer_52 = reinterpret_cast<float2*>(&peerbits_51)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_52);
        unsigned long long bits_53 = 0;
        bits_53 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_53, 4);
        unsigned long long peerbits_54 = _shfl_xor_9;
        float2 _f2_245 = make_float2(0.0f, 0.0f);
        float2 peer_55 = _f2_245;
        peer_55 = reinterpret_cast<float2*>(&peerbits_54)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_55);
        unsigned long long bits_56 = 0;
        bits_56 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_56, 4);
        unsigned long long peerbits_57 = _shfl_xor_10;
        float2 _f2_246 = make_float2(0.0f, 0.0f);
        float2 peer_58 = _f2_246;
        peer_58 = reinterpret_cast<float2*>(&peerbits_57)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_58);
        unsigned long long bits_59 = 0;
        bits_59 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_59, 4);
        unsigned long long peerbits_60 = _shfl_xor_11;
        float2 _f2_247 = make_float2(0.0f, 0.0f);
        float2 peer_61 = _f2_247;
        peer_61 = reinterpret_cast<float2*>(&peerbits_60)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_61);
        unsigned long long bits_62 = 0;
        bits_62 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_62, 2);
        unsigned long long peerbits_63 = _shfl_xor_12;
        float2 _f2_248 = make_float2(0.0f, 0.0f);
        float2 peer_64 = _f2_248;
        peer_64 = reinterpret_cast<float2*>(&peerbits_63)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_64);
        unsigned long long bits_65 = 0;
        bits_65 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_65, 2);
        unsigned long long peerbits_66 = _shfl_xor_13;
        float2 _f2_249 = make_float2(0.0f, 0.0f);
        float2 peer_67 = _f2_249;
        peer_67 = reinterpret_cast<float2*>(&peerbits_66)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_67);
        unsigned long long bits_68 = 0;
        bits_68 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_68, 2);
        unsigned long long peerbits_69 = _shfl_xor_14;
        float2 _f2_250 = make_float2(0.0f, 0.0f);
        float2 peer_70 = _f2_250;
        peer_70 = reinterpret_cast<float2*>(&peerbits_69)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_70);
        unsigned long long bits_71 = 0;
        bits_71 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_71, 2);
        unsigned long long peerbits_72 = _shfl_xor_15;
        float2 _f2_251 = make_float2(0.0f, 0.0f);
        float2 peer_73 = _f2_251;
        peer_73 = reinterpret_cast<float2*>(&peerbits_72)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_73);
        unsigned long long bits_74 = 0;
        bits_74 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_74, 1);
        unsigned long long peerbits_75 = _shfl_xor_16;
        float2 _f2_252 = make_float2(0.0f, 0.0f);
        float2 peer_76 = _f2_252;
        peer_76 = reinterpret_cast<float2*>(&peerbits_75)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_76);
        unsigned long long bits_77 = 0;
        bits_77 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_77, 1);
        unsigned long long peerbits_78 = _shfl_xor_17;
        float2 _f2_253 = make_float2(0.0f, 0.0f);
        float2 peer_79 = _f2_253;
        peer_79 = reinterpret_cast<float2*>(&peerbits_78)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_79);
        unsigned long long bits_80 = 0;
        bits_80 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_80, 1);
        unsigned long long peerbits_81 = _shfl_xor_18;
        float2 _f2_254 = make_float2(0.0f, 0.0f);
        float2 peer_82 = _f2_254;
        peer_82 = reinterpret_cast<float2*>(&peerbits_81)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_82);
        unsigned long long bits_83 = 0;
        bits_83 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_83, 1);
        unsigned long long peerbits_84 = _shfl_xor_19;
        float2 _f2_255 = make_float2(0.0f, 0.0f);
        float2 peer_85 = _f2_255;
        peer_85 = reinterpret_cast<float2*>(&peerbits_84)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_85);
        if (lane == 0) {
            stats[warp_0 * 4 * 2] = pairs[0].x;
            stats[warp_0 * 4 * 2 + 1] = pairs[0].y;
            stats[(warp_0 * 4 + 1) * 2] = pairs[1].x;
            stats[(warp_0 * 4 + 1) * 2 + 1] = pairs[1].y;
            stats[(warp_0 * 4 + 2) * 2] = pairs[2].x;
            stats[(warp_0 * 4 + 2) * 2 + 1] = pairs[2].y;
            stats[(warp_0 * 4 + 3) * 2] = pairs[3].x;
            stats[(warp_0 * 4 + 3) * 2 + 1] = pairs[3].y;
        }
        __syncthreads();
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 4) {
            total_sq = stats[(stat_w * 4 + stat_n) * 2];
            total_dot = stats[(stat_w * 4 + stat_n) * 2 + 1];
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
        if (stat_n < 4 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[4];
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_0;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_1;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_2;
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_3;
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
        float _fmax_3 = fmaxf(max_chunk, logits[3]);
        max_chunk = _fmax_3;
        float _fmax_4 = fmaxf(max_running, max_chunk);
        float max_new = _fmax_4;
        float _exp2_0 = approx_exp2((max_running - max_new) * 1.4426950408889634f);
        float correction = _exp2_0;
        float weights[4];
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
        float _exp2_4 = approx_exp2((logits[3] - max_new) * 1.4426950408889634f);
        weights[3] = _exp2_4;
        sum_weights += weights[3];
        float2 _f2_256 = make_float2(correction, correction);
        float2 corr = _f2_256;
        const int woff_86 = 0;
        int base_87 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_257 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_257;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_258 = make_float2(acc[2], acc[3]);
        float2 previous_88 = _f2_258;
        a_5[1] = mul_f32x2_noftz(previous_88, corr);
        float2 _f2_259 = make_float2(acc[4], acc[5]);
        float2 previous_89 = _f2_259;
        a_5[2] = mul_f32x2_noftz(previous_89, corr);
        float2 _f2_260 = make_float2(acc[6], acc[7]);
        float2 previous_90 = _f2_260;
        a_5[3] = mul_f32x2_noftz(previous_90, corr);
        float2 _f2_261 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_261;
        {
            float2 _f2_262 = make_float2(fsrc[0], fsrc[1]);
            float2 v_18 = _f2_262;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_18, a_5[0]);
            float2 _f2_263 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_16 = _f2_263;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_16, a_5[1]);
            float2 _f2_264 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_5 = _f2_264;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_5, a_5[2]);
            float2 _f2_265 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_16 = _f2_265;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_16, a_5[3]);
        }
        float2 _f2_270 = make_float2(weights[1], weights[1]);
        float2 weight_91 = _f2_270;
        {
            float2 _f2_271 = make_float2(fsrc[28], fsrc[29]);
            float2 v_19 = _f2_271;
            a_5[0] = fma_f32x2_rn_noftz(weight_91, v_19, a_5[0]);
            float2 _f2_272 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_17 = _f2_272;
            a_5[1] = fma_f32x2_rn_noftz(weight_91, v_0_17, a_5[1]);
            float2 _f2_273 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_6 = _f2_273;
            a_5[2] = fma_f32x2_rn_noftz(weight_91, v_1_6, a_5[2]);
            float2 _f2_274 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_17 = _f2_274;
            a_5[3] = fma_f32x2_rn_noftz(weight_91, v_2_17, a_5[3]);
        }
        float2 _f2_279 = make_float2(weights[2], weights[2]);
        float2 weight_92 = _f2_279;
        {
            float2 _f2_280 = make_float2(fsrc[56], fsrc[57]);
            float2 v_20 = _f2_280;
            a_5[0] = fma_f32x2_rn_noftz(weight_92, v_20, a_5[0]);
            float2 _f2_281 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_18 = _f2_281;
            a_5[1] = fma_f32x2_rn_noftz(weight_92, v_0_18, a_5[1]);
            float2 _f2_282 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_7 = _f2_282;
            a_5[2] = fma_f32x2_rn_noftz(weight_92, v_1_7, a_5[2]);
            float2 _f2_283 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_18 = _f2_283;
            a_5[3] = fma_f32x2_rn_noftz(weight_92, v_2_18, a_5[3]);
        }
        float2 _f2_288 = make_float2(weights[3], weights[3]);
        float2 weight_93 = _f2_288;
        {
            unsigned int sw2[4];
            sw2[0] = words[42 + woff_86];
            sw2[1] = words[42 + woff_86 + 1];
            sw2[2] = words[42 + woff_86 + 2];
            sw2[3] = words[42 + woff_86 + 3];
            float sw2_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32[_pair * 2])[0]), "=f"((&sw2_f32[_pair * 2])[1])
                    : "r"(sw2[_pair]));
            }
            float2 _f2_293 = make_float2(sw2_f32[0], sw2_f32[1]);
            float2 v_21 = _f2_293;
            a_5[0] = fma_f32x2_rn_noftz(weight_93, v_21, a_5[0]);
            float2 _f2_294 = make_float2(sw2_f32[2], sw2_f32[3]);
            float2 v_0_19 = _f2_294;
            a_5[1] = fma_f32x2_rn_noftz(weight_93, v_0_19, a_5[1]);
            float2 _f2_295 = make_float2(sw2_f32[4], sw2_f32[5]);
            float2 v_1_8 = _f2_295;
            a_5[2] = fma_f32x2_rn_noftz(weight_93, v_1_8, a_5[2]);
            float2 _f2_296 = make_float2(sw2_f32[6], sw2_f32[7]);
            float2 v_2_19 = _f2_296;
            a_5[3] = fma_f32x2_rn_noftz(weight_93, v_2_19, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_94 = 4;
        int base_95 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_96[4];
        float2 _f2_297 = make_float2(acc[8], acc[9]);
        float2 previous_97 = _f2_297;
        a_96[0] = mul_f32x2_noftz(previous_97, corr);
        float2 _f2_298 = make_float2(acc[10], acc[11]);
        float2 previous_98 = _f2_298;
        a_96[1] = mul_f32x2_noftz(previous_98, corr);
        float2 _f2_299 = make_float2(acc[12], acc[13]);
        float2 previous_99 = _f2_299;
        a_96[2] = mul_f32x2_noftz(previous_99, corr);
        float2 _f2_300 = make_float2(acc[14], acc[15]);
        float2 previous_100 = _f2_300;
        a_96[3] = mul_f32x2_noftz(previous_100, corr);
        float2 _f2_301 = make_float2(weights[0], weights[0]);
        float2 weight_101 = _f2_301;
        {
            float2 _f2_302 = make_float2(fsrc[8], fsrc[9]);
            float2 v_22 = _f2_302;
            a_96[0] = fma_f32x2_rn_noftz(weight_101, v_22, a_96[0]);
            float2 _f2_303 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_20 = _f2_303;
            a_96[1] = fma_f32x2_rn_noftz(weight_101, v_0_20, a_96[1]);
            float2 _f2_304 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_9 = _f2_304;
            a_96[2] = fma_f32x2_rn_noftz(weight_101, v_1_9, a_96[2]);
            float2 _f2_305 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_20 = _f2_305;
            a_96[3] = fma_f32x2_rn_noftz(weight_101, v_2_20, a_96[3]);
        }
        float2 _f2_310 = make_float2(weights[1], weights[1]);
        float2 weight_102 = _f2_310;
        {
            float2 _f2_311 = make_float2(fsrc[36], fsrc[37]);
            float2 v_23 = _f2_311;
            a_96[0] = fma_f32x2_rn_noftz(weight_102, v_23, a_96[0]);
            float2 _f2_312 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_21 = _f2_312;
            a_96[1] = fma_f32x2_rn_noftz(weight_102, v_0_21, a_96[1]);
            float2 _f2_313 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_10 = _f2_313;
            a_96[2] = fma_f32x2_rn_noftz(weight_102, v_1_10, a_96[2]);
            float2 _f2_314 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_21 = _f2_314;
            a_96[3] = fma_f32x2_rn_noftz(weight_102, v_2_21, a_96[3]);
        }
        float2 _f2_319 = make_float2(weights[2], weights[2]);
        float2 weight_103 = _f2_319;
        {
            float2 _f2_320 = make_float2(fsrc[64], fsrc[65]);
            float2 v_24 = _f2_320;
            a_96[0] = fma_f32x2_rn_noftz(weight_103, v_24, a_96[0]);
            float2 _f2_321 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_22 = _f2_321;
            a_96[1] = fma_f32x2_rn_noftz(weight_103, v_0_22, a_96[1]);
            float2 _f2_322 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_11 = _f2_322;
            a_96[2] = fma_f32x2_rn_noftz(weight_103, v_1_11, a_96[2]);
            float2 _f2_323 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_22 = _f2_323;
            a_96[3] = fma_f32x2_rn_noftz(weight_103, v_2_22, a_96[3]);
        }
        float2 _f2_328 = make_float2(weights[3], weights[3]);
        float2 weight_104 = _f2_328;
        {
            unsigned int sw2_1[4];
            sw2_1[0] = words[42 + woff_94];
            sw2_1[1] = words[42 + woff_94 + 1];
            sw2_1[2] = words[42 + woff_94 + 2];
            sw2_1[3] = words[42 + woff_94 + 3];
            float sw2_f32_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_1[_pair * 2])[0]), "=f"((&sw2_f32_1[_pair * 2])[1])
                    : "r"(sw2_1[_pair]));
            }
            float2 _f2_333 = make_float2(sw2_f32_1[0], sw2_f32_1[1]);
            float2 v_25 = _f2_333;
            a_96[0] = fma_f32x2_rn_noftz(weight_104, v_25, a_96[0]);
            float2 _f2_334 = make_float2(sw2_f32_1[2], sw2_f32_1[3]);
            float2 v_0_23 = _f2_334;
            a_96[1] = fma_f32x2_rn_noftz(weight_104, v_0_23, a_96[1]);
            float2 _f2_335 = make_float2(sw2_f32_1[4], sw2_f32_1[5]);
            float2 v_1_12 = _f2_335;
            a_96[2] = fma_f32x2_rn_noftz(weight_104, v_1_12, a_96[2]);
            float2 _f2_336 = make_float2(sw2_f32_1[6], sw2_f32_1[7]);
            float2 v_2_23 = _f2_336;
            a_96[3] = fma_f32x2_rn_noftz(weight_104, v_2_23, a_96[3]);
        }
        acc[8] = a_96[0].x;
        acc[9] = a_96[0].y;
        acc[10] = a_96[1].x;
        acc[11] = a_96[1].y;
        acc[12] = a_96[2].x;
        acc[13] = a_96[2].y;
        acc[14] = a_96[3].x;
        acc[15] = a_96[3].y;
        const int woff_105 = 8;
        int base_106 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_107[4];
        float2 _f2_337 = make_float2(acc[16], acc[17]);
        float2 previous_108 = _f2_337;
        a_107[0] = mul_f32x2_noftz(previous_108, corr);
        float2 _f2_338 = make_float2(acc[18], acc[19]);
        float2 previous_109 = _f2_338;
        a_107[1] = mul_f32x2_noftz(previous_109, corr);
        float2 _f2_339 = make_float2(acc[20], acc[21]);
        float2 previous_110 = _f2_339;
        a_107[2] = mul_f32x2_noftz(previous_110, corr);
        float2 _f2_340 = make_float2(acc[22], acc[23]);
        float2 previous_111 = _f2_340;
        a_107[3] = mul_f32x2_noftz(previous_111, corr);
        float2 _f2_341 = make_float2(weights[0], weights[0]);
        float2 weight_112 = _f2_341;
        {
            float2 _f2_342 = make_float2(fsrc[16], fsrc[17]);
            float2 v_26 = _f2_342;
            a_107[0] = fma_f32x2_rn_noftz(weight_112, v_26, a_107[0]);
            float2 _f2_343 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_24 = _f2_343;
            a_107[1] = fma_f32x2_rn_noftz(weight_112, v_0_24, a_107[1]);
            float2 _f2_344 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_13 = _f2_344;
            a_107[2] = fma_f32x2_rn_noftz(weight_112, v_1_13, a_107[2]);
            float2 _f2_345 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_24 = _f2_345;
            a_107[3] = fma_f32x2_rn_noftz(weight_112, v_2_24, a_107[3]);
        }
        float2 _f2_350 = make_float2(weights[1], weights[1]);
        float2 weight_113 = _f2_350;
        {
            float2 _f2_351 = make_float2(fsrc[44], fsrc[45]);
            float2 v_27 = _f2_351;
            a_107[0] = fma_f32x2_rn_noftz(weight_113, v_27, a_107[0]);
            float2 _f2_352 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_25 = _f2_352;
            a_107[1] = fma_f32x2_rn_noftz(weight_113, v_0_25, a_107[1]);
            float2 _f2_353 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_14 = _f2_353;
            a_107[2] = fma_f32x2_rn_noftz(weight_113, v_1_14, a_107[2]);
            float2 _f2_354 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_25 = _f2_354;
            a_107[3] = fma_f32x2_rn_noftz(weight_113, v_2_25, a_107[3]);
        }
        float2 _f2_359 = make_float2(weights[2], weights[2]);
        float2 weight_114 = _f2_359;
        {
            float2 _f2_360 = make_float2(fsrc[72], fsrc[73]);
            float2 v_28 = _f2_360;
            a_107[0] = fma_f32x2_rn_noftz(weight_114, v_28, a_107[0]);
            float2 _f2_361 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_26 = _f2_361;
            a_107[1] = fma_f32x2_rn_noftz(weight_114, v_0_26, a_107[1]);
            float2 _f2_362 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_15 = _f2_362;
            a_107[2] = fma_f32x2_rn_noftz(weight_114, v_1_15, a_107[2]);
            float2 _f2_363 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_26 = _f2_363;
            a_107[3] = fma_f32x2_rn_noftz(weight_114, v_2_26, a_107[3]);
        }
        float2 _f2_368 = make_float2(weights[3], weights[3]);
        float2 weight_115 = _f2_368;
        {
            unsigned int sw2_2[4];
            sw2_2[0] = words[42 + woff_105];
            sw2_2[1] = words[42 + woff_105 + 1];
            sw2_2[2] = words[42 + woff_105 + 2];
            sw2_2[3] = words[42 + woff_105 + 3];
            float sw2_f32_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_2[_pair * 2])[0]), "=f"((&sw2_f32_2[_pair * 2])[1])
                    : "r"(sw2_2[_pair]));
            }
            float2 _f2_373 = make_float2(sw2_f32_2[0], sw2_f32_2[1]);
            float2 v_29 = _f2_373;
            a_107[0] = fma_f32x2_rn_noftz(weight_115, v_29, a_107[0]);
            float2 _f2_374 = make_float2(sw2_f32_2[2], sw2_f32_2[3]);
            float2 v_0_27 = _f2_374;
            a_107[1] = fma_f32x2_rn_noftz(weight_115, v_0_27, a_107[1]);
            float2 _f2_375 = make_float2(sw2_f32_2[4], sw2_f32_2[5]);
            float2 v_1_16 = _f2_375;
            a_107[2] = fma_f32x2_rn_noftz(weight_115, v_1_16, a_107[2]);
            float2 _f2_376 = make_float2(sw2_f32_2[6], sw2_f32_2[7]);
            float2 v_2_27 = _f2_376;
            a_107[3] = fma_f32x2_rn_noftz(weight_115, v_2_27, a_107[3]);
        }
        acc[16] = a_107[0].x;
        acc[17] = a_107[0].y;
        acc[18] = a_107[1].x;
        acc[19] = a_107[1].y;
        acc[20] = a_107[2].x;
        acc[21] = a_107[2].y;
        acc[22] = a_107[3].x;
        acc[23] = a_107[3].y;
        const int woff_116 = 12;
        int base_117 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_118[4];
        float2 _f2_377 = make_float2(acc[24], acc[25]);
        float2 previous_119 = _f2_377;
        a_118[0] = mul_f32x2_noftz(previous_119, corr);
        float2 _f2_378 = make_float2(acc[26], acc[27]);
        float2 previous_120 = _f2_378;
        a_118[1] = mul_f32x2_noftz(previous_120, corr);
        float2 _f2_379 = make_float2(weights[0], weights[0]);
        float2 weight_121 = _f2_379;
        {
            float2 _f2_380 = make_float2(fsrc[24], fsrc[25]);
            float2 v_30 = _f2_380;
            a_118[0] = fma_f32x2_rn_noftz(weight_121, v_30, a_118[0]);
            float2 _f2_381 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_28 = _f2_381;
            a_118[1] = fma_f32x2_rn_noftz(weight_121, v_0_28, a_118[1]);
        }
        float2 _f2_384 = make_float2(weights[1], weights[1]);
        float2 weight_122 = _f2_384;
        {
            float2 _f2_385 = make_float2(fsrc[52], fsrc[53]);
            float2 v_31 = _f2_385;
            a_118[0] = fma_f32x2_rn_noftz(weight_122, v_31, a_118[0]);
            float2 _f2_386 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_29 = _f2_386;
            a_118[1] = fma_f32x2_rn_noftz(weight_122, v_0_29, a_118[1]);
        }
        float2 _f2_389 = make_float2(weights[2], weights[2]);
        float2 weight_123 = _f2_389;
        {
            float2 _f2_390 = make_float2(fsrc[80], fsrc[81]);
            float2 v_32 = _f2_390;
            a_118[0] = fma_f32x2_rn_noftz(weight_123, v_32, a_118[0]);
            float2 _f2_391 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_30 = _f2_391;
            a_118[1] = fma_f32x2_rn_noftz(weight_123, v_0_30, a_118[1]);
        }
        float2 _f2_394 = make_float2(weights[3], weights[3]);
        float2 weight_124 = _f2_394;
        {
            unsigned int sw2_3[4];
            sw2_3[0] = words[42 + woff_116];
            sw2_3[1] = words[42 + woff_116 + 1];
            float sw2_f32_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_3[_pair * 2])[0]), "=f"((&sw2_f32_3[_pair * 2])[1])
                    : "r"(sw2_3[_pair]));
            }
            float2 _f2_397 = make_float2(sw2_f32_3[0], sw2_f32_3[1]);
            float2 v_33 = _f2_397;
            a_118[0] = fma_f32x2_rn_noftz(weight_124, v_33, a_118[0]);
            float2 _f2_398 = make_float2(sw2_f32_3[2], sw2_f32_3[3]);
            float2 v_0_31 = _f2_398;
            a_118[1] = fma_f32x2_rn_noftz(weight_124, v_0_31, a_118[1]);
        }
        acc[24] = a_118[0].x;
        acc[25] = a_118[0].y;
        acc[26] = a_118[1].x;
        acc[27] = a_118[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float2 _f2_399 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_399;
        float2 _f2_400 = make_float2(acc[0], acc[1]);
        float2 v_34 = _f2_400;
        output_sq_pair = fma_f32x2_rn_noftz(v_34, v_34, output_sq_pair);
        float2 _f2_401 = make_float2(acc[2], acc[3]);
        float2 v_125 = _f2_401;
        output_sq_pair = fma_f32x2_rn_noftz(v_125, v_125, output_sq_pair);
        float2 _f2_402 = make_float2(acc[4], acc[5]);
        float2 v_126 = _f2_402;
        output_sq_pair = fma_f32x2_rn_noftz(v_126, v_126, output_sq_pair);
        float2 _f2_403 = make_float2(acc[6], acc[7]);
        float2 v_127 = _f2_403;
        output_sq_pair = fma_f32x2_rn_noftz(v_127, v_127, output_sq_pair);
        float2 _f2_404 = make_float2(acc[8], acc[9]);
        float2 v_128 = _f2_404;
        output_sq_pair = fma_f32x2_rn_noftz(v_128, v_128, output_sq_pair);
        float2 _f2_405 = make_float2(acc[10], acc[11]);
        float2 v_129 = _f2_405;
        output_sq_pair = fma_f32x2_rn_noftz(v_129, v_129, output_sq_pair);
        float2 _f2_406 = make_float2(acc[12], acc[13]);
        float2 v_130 = _f2_406;
        output_sq_pair = fma_f32x2_rn_noftz(v_130, v_130, output_sq_pair);
        float2 _f2_407 = make_float2(acc[14], acc[15]);
        float2 v_131 = _f2_407;
        output_sq_pair = fma_f32x2_rn_noftz(v_131, v_131, output_sq_pair);
        float2 _f2_408 = make_float2(acc[16], acc[17]);
        float2 v_132 = _f2_408;
        output_sq_pair = fma_f32x2_rn_noftz(v_132, v_132, output_sq_pair);
        float2 _f2_409 = make_float2(acc[18], acc[19]);
        float2 v_133 = _f2_409;
        output_sq_pair = fma_f32x2_rn_noftz(v_133, v_133, output_sq_pair);
        float2 _f2_410 = make_float2(acc[20], acc[21]);
        float2 v_134 = _f2_410;
        output_sq_pair = fma_f32x2_rn_noftz(v_134, v_134, output_sq_pair);
        float2 _f2_411 = make_float2(acc[22], acc[23]);
        float2 v_135 = _f2_411;
        output_sq_pair = fma_f32x2_rn_noftz(v_135, v_135, output_sq_pair);
        float2 _f2_412 = make_float2(acc[24], acc[25]);
        float2 v_136 = _f2_412;
        output_sq_pair = fma_f32x2_rn_noftz(v_136, v_136, output_sq_pair);
        float2 _f2_413 = make_float2(acc[26], acc[27]);
        float2 v_137 = _f2_413;
        output_sq_pair = fma_f32x2_rn_noftz(v_137, v_137, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_20;
        float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_21;
        float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_22;
        float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_23;
        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_24;
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
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_4;
        float2 _f2_414 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_414;
        int base_138 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_415 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_415, rsigma_pair);
        float2 _f2_416 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_416);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_139 = 2;
        const int acc_idx_140 = value_idx_139;
        float2 _f2_417 = make_float2(acc[acc_idx_140], acc[acc_idx_140 + 1]);
        float2 scaled_pair_141 = mul_f32x2_noftz(_f2_417, rsigma_pair);
        float2 _f2_418 = make_float2(wout[acc_idx_140], wout[acc_idx_140 + 1]);
        float2 normalized_pair_142 = mul_f32x2_noftz(scaled_pair_141, _f2_418);
        output_values[value_idx_139] = normalized_pair_142.x;
        output_values[value_idx_139 + 1] = normalized_pair_142.y;
        const int value_idx_143 = 4;
        const int acc_idx_144 = value_idx_143;
        float2 _f2_419 = make_float2(acc[acc_idx_144], acc[acc_idx_144 + 1]);
        float2 scaled_pair_145 = mul_f32x2_noftz(_f2_419, rsigma_pair);
        float2 _f2_420 = make_float2(wout[acc_idx_144], wout[acc_idx_144 + 1]);
        float2 normalized_pair_146 = mul_f32x2_noftz(scaled_pair_145, _f2_420);
        output_values[value_idx_143] = normalized_pair_146.x;
        output_values[value_idx_143 + 1] = normalized_pair_146.y;
        const int value_idx_147 = 6;
        const int acc_idx_148 = value_idx_147;
        float2 _f2_421 = make_float2(acc[acc_idx_148], acc[acc_idx_148 + 1]);
        float2 scaled_pair_149 = mul_f32x2_noftz(_f2_421, rsigma_pair);
        float2 _f2_422 = make_float2(wout[acc_idx_148], wout[acc_idx_148 + 1]);
        float2 normalized_pair_150 = mul_f32x2_noftz(scaled_pair_149, _f2_422);
        output_values[value_idx_147] = normalized_pair_150.x;
        output_values[value_idx_147 + 1] = normalized_pair_150.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_138 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_151 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_152[8];
        const int value_idx_153 = 0;
        const int acc_idx_154 = 8 + value_idx_153;
        float2 _f2_423 = make_float2(acc[acc_idx_154], acc[acc_idx_154 + 1]);
        float2 scaled_pair_155 = mul_f32x2_noftz(_f2_423, rsigma_pair);
        float2 _f2_424 = make_float2(wout[acc_idx_154], wout[acc_idx_154 + 1]);
        float2 normalized_pair_156 = mul_f32x2_noftz(scaled_pair_155, _f2_424);
        output_values_152[value_idx_153] = normalized_pair_156.x;
        output_values_152[value_idx_153 + 1] = normalized_pair_156.y;
        const int value_idx_157 = 2;
        const int acc_idx_158 = 8 + value_idx_157;
        float2 _f2_425 = make_float2(acc[acc_idx_158], acc[acc_idx_158 + 1]);
        float2 scaled_pair_159 = mul_f32x2_noftz(_f2_425, rsigma_pair);
        float2 _f2_426 = make_float2(wout[acc_idx_158], wout[acc_idx_158 + 1]);
        float2 normalized_pair_160 = mul_f32x2_noftz(scaled_pair_159, _f2_426);
        output_values_152[value_idx_157] = normalized_pair_160.x;
        output_values_152[value_idx_157 + 1] = normalized_pair_160.y;
        const int value_idx_161 = 4;
        const int acc_idx_162 = 8 + value_idx_161;
        float2 _f2_427 = make_float2(acc[acc_idx_162], acc[acc_idx_162 + 1]);
        float2 scaled_pair_163 = mul_f32x2_noftz(_f2_427, rsigma_pair);
        float2 _f2_428 = make_float2(wout[acc_idx_162], wout[acc_idx_162 + 1]);
        float2 normalized_pair_164 = mul_f32x2_noftz(scaled_pair_163, _f2_428);
        output_values_152[value_idx_161] = normalized_pair_164.x;
        output_values_152[value_idx_161 + 1] = normalized_pair_164.y;
        const int value_idx_165 = 6;
        const int acc_idx_166 = 8 + value_idx_165;
        float2 _f2_429 = make_float2(acc[acc_idx_166], acc[acc_idx_166 + 1]);
        float2 scaled_pair_167 = mul_f32x2_noftz(_f2_429, rsigma_pair);
        float2 _f2_430 = make_float2(wout[acc_idx_166], wout[acc_idx_166 + 1]);
        float2 normalized_pair_168 = mul_f32x2_noftz(scaled_pair_167, _f2_430);
        output_values_152[value_idx_165] = normalized_pair_168.x;
        output_values_152[value_idx_165 + 1] = normalized_pair_168.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_152[0 + 0], output_values_152[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_152[0 + 2], output_values_152[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_152[0 + 4], output_values_152[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_152[0 + 6], output_values_152[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_151 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_169 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_170[8];
        const int value_idx_171 = 0;
        const int acc_idx_172 = 16 + value_idx_171;
        float2 _f2_431 = make_float2(acc[acc_idx_172], acc[acc_idx_172 + 1]);
        float2 scaled_pair_173 = mul_f32x2_noftz(_f2_431, rsigma_pair);
        float2 _f2_432 = make_float2(wout[acc_idx_172], wout[acc_idx_172 + 1]);
        float2 normalized_pair_174 = mul_f32x2_noftz(scaled_pair_173, _f2_432);
        output_values_170[value_idx_171] = normalized_pair_174.x;
        output_values_170[value_idx_171 + 1] = normalized_pair_174.y;
        const int value_idx_175 = 2;
        const int acc_idx_176 = 16 + value_idx_175;
        float2 _f2_433 = make_float2(acc[acc_idx_176], acc[acc_idx_176 + 1]);
        float2 scaled_pair_177 = mul_f32x2_noftz(_f2_433, rsigma_pair);
        float2 _f2_434 = make_float2(wout[acc_idx_176], wout[acc_idx_176 + 1]);
        float2 normalized_pair_178 = mul_f32x2_noftz(scaled_pair_177, _f2_434);
        output_values_170[value_idx_175] = normalized_pair_178.x;
        output_values_170[value_idx_175 + 1] = normalized_pair_178.y;
        const int value_idx_179 = 4;
        const int acc_idx_180 = 16 + value_idx_179;
        float2 _f2_435 = make_float2(acc[acc_idx_180], acc[acc_idx_180 + 1]);
        float2 scaled_pair_181 = mul_f32x2_noftz(_f2_435, rsigma_pair);
        float2 _f2_436 = make_float2(wout[acc_idx_180], wout[acc_idx_180 + 1]);
        float2 normalized_pair_182 = mul_f32x2_noftz(scaled_pair_181, _f2_436);
        output_values_170[value_idx_179] = normalized_pair_182.x;
        output_values_170[value_idx_179 + 1] = normalized_pair_182.y;
        const int value_idx_183 = 6;
        const int acc_idx_184 = 16 + value_idx_183;
        float2 _f2_437 = make_float2(acc[acc_idx_184], acc[acc_idx_184 + 1]);
        float2 scaled_pair_185 = mul_f32x2_noftz(_f2_437, rsigma_pair);
        float2 _f2_438 = make_float2(wout[acc_idx_184], wout[acc_idx_184 + 1]);
        float2 normalized_pair_186 = mul_f32x2_noftz(scaled_pair_185, _f2_438);
        output_values_170[value_idx_183] = normalized_pair_186.x;
        output_values_170[value_idx_183 + 1] = normalized_pair_186.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_170[0 + 0], output_values_170[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_170[0 + 2], output_values_170[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_170[0 + 4], output_values_170[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_170[0 + 6], output_values_170[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_169 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_187 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_188[8];
        const int value_idx_189 = 0;
        const int acc_idx_190 = 24 + value_idx_189;
        float2 _f2_439 = make_float2(acc[acc_idx_190], acc[acc_idx_190 + 1]);
        float2 scaled_pair_191 = mul_f32x2_noftz(_f2_439, rsigma_pair);
        float2 _f2_440 = make_float2(wout[acc_idx_190], wout[acc_idx_190 + 1]);
        float2 normalized_pair_192 = mul_f32x2_noftz(scaled_pair_191, _f2_440);
        output_values_188[value_idx_189] = normalized_pair_192.x;
        output_values_188[value_idx_189 + 1] = normalized_pair_192.y;
        const int value_idx_193 = 2;
        const int acc_idx_194 = 24 + value_idx_193;
        float2 _f2_441 = make_float2(acc[acc_idx_194], acc[acc_idx_194 + 1]);
        float2 scaled_pair_195 = mul_f32x2_noftz(_f2_441, rsigma_pair);
        float2 _f2_442 = make_float2(wout[acc_idx_194], wout[acc_idx_194 + 1]);
        float2 normalized_pair_196 = mul_f32x2_noftz(scaled_pair_195, _f2_442);
        output_values_188[value_idx_193] = normalized_pair_196.x;
        output_values_188[value_idx_193 + 1] = normalized_pair_196.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_188[0 + 0], output_values_188[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_188[0 + 2], output_values_188[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_187]) = _pk2;
        }
    }
}

} // extern "C"
