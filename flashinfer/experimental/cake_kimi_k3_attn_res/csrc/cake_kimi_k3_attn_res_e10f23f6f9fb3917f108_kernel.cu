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
kernel_cake_kimi_k3_attn_res_e10f23f6f9fb3917f108(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
        unsigned int words[42];
        unsigned int dwords[14];
        float q[28];
        float wout[28];
        int base_0 = group * 1024 + thread * 8;
        const int woff_0 = 0;
        unsigned int raw_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
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
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            raw_1[0 + 0] = _uv4_1.x;
            raw_1[0 + 1] = _uv4_1.y;
            raw_1[0 + 2] = _uv4_1.z;
            raw_1[0 + 3] = _uv4_1.w;
        }
        words[14 + woff_0] = raw_1[0];
        words[14 + woff_0 + 1] = raw_1[1];
        words[14 + woff_0 + 2] = raw_1[2];
        words[14 + woff_0 + 3] = raw_1[3];
        unsigned int raw_2[4];
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            raw_2[0 + 0] = _uv4_2.x;
            raw_2[0 + 1] = _uv4_2.y;
            raw_2[0 + 2] = _uv4_2.z;
            raw_2[0 + 3] = _uv4_2.w;
        }
        words[28 + woff_0] = raw_2[0];
        words[28 + woff_0 + 1] = raw_2[1];
        words[28 + woff_0 + 2] = raw_2[2];
        words[28 + woff_0 + 3] = raw_2[3];
        unsigned int raw_3[4];
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            raw_3[0 + 0] = _uv4_3.x;
            raw_3[0 + 1] = _uv4_3.y;
            raw_3[0 + 2] = _uv4_3.z;
            raw_3[0 + 3] = _uv4_3.w;
        }
        dwords[woff_0] = raw_3[0];
        dwords[woff_0 + 1] = raw_3[1];
        dwords[woff_0 + 2] = raw_3[2];
        dwords[woff_0 + 3] = raw_3[3];
        float rms_0[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&rms_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&rms_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        float res_0[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(qk_weight + base_0);
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
                        : "=f"((&res_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&res_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        q[0] = rms_0[0] * res_0[0];
        q[1] = rms_0[1] * res_0[1];
        q[2] = rms_0[2] * res_0[2];
        q[3] = rms_0[3] * res_0[3];
        q[4] = rms_0[4] * res_0[4];
        q[5] = rms_0[5] * res_0[5];
        q[6] = rms_0[6] * res_0[6];
        q[7] = rms_0[7] * res_0[7];
        float onw_0[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&onw_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&onw_0[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
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
        unsigned int raw_4[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_1)) + 0);
            raw_4[0 + 0] = _uv4_7.x;
            raw_4[0 + 1] = _uv4_7.y;
            raw_4[0 + 2] = _uv4_7.z;
            raw_4[0 + 3] = _uv4_7.w;
        }
        words[woff_1] = raw_4[0];
        words[woff_1 + 1] = raw_4[1];
        words[woff_1 + 2] = raw_4[2];
        words[woff_1 + 3] = raw_4[3];
        unsigned int raw_5[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_1)) + 0);
            raw_5[0 + 0] = _uv4_8.x;
            raw_5[0 + 1] = _uv4_8.y;
            raw_5[0 + 2] = _uv4_8.z;
            raw_5[0 + 3] = _uv4_8.w;
        }
        words[14 + woff_1] = raw_5[0];
        words[14 + woff_1 + 1] = raw_5[1];
        words[14 + woff_1 + 2] = raw_5[2];
        words[14 + woff_1 + 3] = raw_5[3];
        unsigned int raw_6[4];
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_1)) + 0);
            raw_6[0 + 0] = _uv4_9.x;
            raw_6[0 + 1] = _uv4_9.y;
            raw_6[0 + 2] = _uv4_9.z;
            raw_6[0 + 3] = _uv4_9.w;
        }
        words[28 + woff_1] = raw_6[0];
        words[28 + woff_1 + 1] = raw_6[1];
        words[28 + woff_1 + 2] = raw_6[2];
        words[28 + woff_1 + 3] = raw_6[3];
        unsigned int raw_7[4];
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_1)) + 0);
            raw_7[0 + 0] = _uv4_10.x;
            raw_7[0 + 1] = _uv4_10.y;
            raw_7[0 + 2] = _uv4_10.z;
            raw_7[0 + 3] = _uv4_10.w;
        }
        dwords[woff_1] = raw_7[0];
        dwords[woff_1 + 1] = raw_7[1];
        dwords[woff_1 + 2] = raw_7[2];
        dwords[woff_1 + 3] = raw_7[3];
        float rms_1[8];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(norm_weight + base_1);
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
                        : "=f"((&rms_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&rms_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        float res_1[8];
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(qk_weight + base_1);
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
                        : "=f"((&res_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&res_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_12[_pair]));
                }
            }
        }
        q[8] = rms_1[0] * res_1[0];
        q[9] = rms_1[1] * res_1[1];
        q[10] = rms_1[2] * res_1[2];
        q[11] = rms_1[3] * res_1[3];
        q[12] = rms_1[4] * res_1[4];
        q[13] = rms_1[5] * res_1[5];
        q[14] = rms_1[6] * res_1[6];
        q[15] = rms_1[7] * res_1[7];
        float onw_1[8];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(output_norm_weight + base_1);
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
                        : "=f"((&onw_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&onw_1[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_13[_pair]));
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
        unsigned int raw_8[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            raw_8[0 + 0] = _uv4_14.x;
            raw_8[0 + 1] = _uv4_14.y;
            raw_8[0 + 2] = _uv4_14.z;
            raw_8[0 + 3] = _uv4_14.w;
        }
        words[woff_2] = raw_8[0];
        words[woff_2 + 1] = raw_8[1];
        words[woff_2 + 2] = raw_8[2];
        words[woff_2 + 3] = raw_8[3];
        unsigned int raw_9[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            raw_9[0 + 0] = _uv4_15.x;
            raw_9[0 + 1] = _uv4_15.y;
            raw_9[0 + 2] = _uv4_15.z;
            raw_9[0 + 3] = _uv4_15.w;
        }
        words[14 + woff_2] = raw_9[0];
        words[14 + woff_2 + 1] = raw_9[1];
        words[14 + woff_2 + 2] = raw_9[2];
        words[14 + woff_2 + 3] = raw_9[3];
        unsigned int raw_10[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            raw_10[0 + 0] = _uv4_16.x;
            raw_10[0 + 1] = _uv4_16.y;
            raw_10[0 + 2] = _uv4_16.z;
            raw_10[0 + 3] = _uv4_16.w;
        }
        words[28 + woff_2] = raw_10[0];
        words[28 + woff_2 + 1] = raw_10[1];
        words[28 + woff_2 + 2] = raw_10[2];
        words[28 + woff_2 + 3] = raw_10[3];
        unsigned int raw_11[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            raw_11[0 + 0] = _uv4_17.x;
            raw_11[0 + 1] = _uv4_17.y;
            raw_11[0 + 2] = _uv4_17.z;
            raw_11[0 + 3] = _uv4_17.w;
        }
        dwords[woff_2] = raw_11[0];
        dwords[woff_2 + 1] = raw_11[1];
        dwords[woff_2 + 2] = raw_11[2];
        dwords[woff_2 + 3] = raw_11[3];
        float rms_2[8];
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
                        : "=f"((&rms_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&rms_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_18[_pair]));
                }
            }
        }
        float res_2[8];
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
                        : "=f"((&res_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&res_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        q[16] = rms_2[0] * res_2[0];
        q[17] = rms_2[1] * res_2[1];
        q[18] = rms_2[2] * res_2[2];
        q[19] = rms_2[3] * res_2[3];
        q[20] = rms_2[4] * res_2[4];
        q[21] = rms_2[5] * res_2[5];
        q[22] = rms_2[6] * res_2[6];
        q[23] = rms_2[7] * res_2[7];
        float onw_2[8];
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
                        : "=f"((&onw_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&onw_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_20[_pair]));
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
            raw1_0[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_3)) + 0);
        }
        words[woff_3] = raw1_0[0];
        unsigned int raw1_1[1];
        {
            raw1_1[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_3)) + 1);
        }
        words[woff_3 + 1] = raw1_1[0];
        unsigned int raw1_2[1];
        {
            raw1_2[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_3)) + 0);
        }
        words[14 + woff_3] = raw1_2[0];
        unsigned int raw1_3[1];
        {
            raw1_3[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_3)) + 1);
        }
        words[14 + woff_3 + 1] = raw1_3[0];
        unsigned int raw1_4[1];
        {
            raw1_4[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_3)) + 0);
        }
        words[28 + woff_3] = raw1_4[0];
        unsigned int raw1_5[1];
        {
            raw1_5[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_3)) + 1);
        }
        words[28 + woff_3 + 1] = raw1_5[0];
        unsigned int raw1_6[1];
        {
            raw1_6[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_3)) + 0);
        }
        dwords[woff_3] = raw1_6[0];
        unsigned int raw1_7[1];
        {
            raw1_7[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_3)) + 1);
        }
        dwords[woff_3 + 1] = raw1_7[0];
        float rms_3[4];
        {
            uint2 _vld_21;
            _vld_21 = *reinterpret_cast<const uint2*>(norm_weight + base_3);
            uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rms_3[0 + _pair * 2])[0]), "=f"((&rms_3[0 + _pair * 2])[1])
                    : "r"(_vpairs_21[_pair]));
            }
        }
        float res_3[4];
        {
            uint2 _vld_22;
            _vld_22 = *reinterpret_cast<const uint2*>(qk_weight + base_3);
            uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&res_3[0 + _pair * 2])[0]), "=f"((&res_3[0 + _pair * 2])[1])
                    : "r"(_vpairs_22[_pair]));
            }
        }
        q[24] = rms_3[0] * res_3[0];
        q[25] = rms_3[1] * res_3[1];
        q[26] = rms_3[2] * res_3[2];
        q[27] = rms_3[3] * res_3[3];
        float onw_3[4];
        {
            uint2 _vld_23;
            _vld_23 = *reinterpret_cast<const uint2*>(output_norm_weight + base_3);
            uint32_t* _vpairs_23 = reinterpret_cast<uint32_t*>(&_vld_23);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&onw_3[0 + _pair * 2])[0]), "=f"((&onw_3[0 + _pair * 2])[1])
                    : "r"(_vpairs_23[_pair]));
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
        float2 sq[3];
        float2 dot[3];
        float fsrc[84];
        sq[0] = make_float2(0.0f, 0.0f);
        dot[0] = make_float2(0.0f, 0.0f);
        sq[1] = make_float2(0.0f, 0.0f);
        dot[1] = make_float2(0.0f, 0.0f);
        sq[2] = make_float2(0.0f, 0.0f);
        dot[2] = make_float2(0.0f, 0.0f);
        int base_4 = group * 1024 + thread * 8;
        const int woff_4 = 0;
        unsigned int sw_0[4];
        sw_0[0] = words[woff_4];
        sw_0[1] = words[woff_4 + 1];
        sw_0[2] = words[woff_4 + 2];
        sw_0[3] = words[woff_4 + 3];
        float vals_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_0[_pair * 2])[0]), "=f"((&vals_0[_pair * 2])[1])
                : "r"(sw_0[_pair]));
        }
        fsrc[0] = vals_0[0];
        fsrc[1] = vals_0[1];
        fsrc[2] = vals_0[2];
        fsrc[3] = vals_0[3];
        fsrc[4] = vals_0[4];
        fsrc[5] = vals_0[5];
        fsrc[6] = vals_0[6];
        fsrc[7] = vals_0[7];
        float2 v_0 = make_float2(vals_0[0], vals_0[1]);
        float2 qp_0 = make_float2(q[0], q[1]);
        sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_0, qp_0, dot[0]);
        float2 v_1 = make_float2(vals_0[2], vals_0[3]);
        float2 qp_1 = make_float2(q[2], q[3]);
        sq[0] = fma_f32x2_rn_noftz(v_1, v_1, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_1, qp_1, dot[0]);
        float2 v_2 = make_float2(vals_0[4], vals_0[5]);
        float2 qp_2 = make_float2(q[4], q[5]);
        sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_2, qp_2, dot[0]);
        float2 v_3 = make_float2(vals_0[6], vals_0[7]);
        float2 qp_3 = make_float2(q[6], q[7]);
        sq[0] = fma_f32x2_rn_noftz(v_3, v_3, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_3, qp_3, dot[0]);
        unsigned int sw_1[4];
        sw_1[0] = words[14 + woff_4];
        sw_1[1] = words[14 + woff_4 + 1];
        sw_1[2] = words[14 + woff_4 + 2];
        sw_1[3] = words[14 + woff_4 + 3];
        float vals_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_1[_pair * 2])[0]), "=f"((&vals_1[_pair * 2])[1])
                : "r"(sw_1[_pair]));
        }
        fsrc[28] = vals_1[0];
        fsrc[29] = vals_1[1];
        fsrc[30] = vals_1[2];
        fsrc[31] = vals_1[3];
        fsrc[32] = vals_1[4];
        fsrc[33] = vals_1[5];
        fsrc[34] = vals_1[6];
        fsrc[35] = vals_1[7];
        float2 v_4 = make_float2(vals_1[0], vals_1[1]);
        float2 qp_4 = make_float2(q[0], q[1]);
        sq[1] = fma_f32x2_rn_noftz(v_4, v_4, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_4, qp_4, dot[1]);
        float2 v_5 = make_float2(vals_1[2], vals_1[3]);
        float2 qp_5 = make_float2(q[2], q[3]);
        sq[1] = fma_f32x2_rn_noftz(v_5, v_5, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_5, qp_5, dot[1]);
        float2 v_6 = make_float2(vals_1[4], vals_1[5]);
        float2 qp_6 = make_float2(q[4], q[5]);
        sq[1] = fma_f32x2_rn_noftz(v_6, v_6, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_6, qp_6, dot[1]);
        float2 v_7 = make_float2(vals_1[6], vals_1[7]);
        float2 qp_7 = make_float2(q[6], q[7]);
        sq[1] = fma_f32x2_rn_noftz(v_7, v_7, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_7, qp_7, dot[1]);
        unsigned int sw_2[4];
        sw_2[0] = words[28 + woff_4];
        sw_2[1] = words[28 + woff_4 + 1];
        sw_2[2] = words[28 + woff_4 + 2];
        sw_2[3] = words[28 + woff_4 + 3];
        __nv_bfloat162 a_0 = __as_bf16x2(sw_2[0]);
        __nv_bfloat162 d_0 = __as_bf16x2(dwords[woff_4]);
        __nv_bfloat162 mixed_0 = a_0 + d_0;
        sw_2[0] = __as_u32(mixed_0);
        words[28 + woff_4] = sw_2[0];
        __nv_bfloat162 a_1 = __as_bf16x2(sw_2[1]);
        __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_4 + 1]);
        __nv_bfloat162 mixed_1 = a_1 + d_1;
        sw_2[1] = __as_u32(mixed_1);
        words[28 + woff_4 + 1] = sw_2[1];
        __nv_bfloat162 a_2 = __as_bf16x2(sw_2[2]);
        __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_4 + 2]);
        __nv_bfloat162 mixed_2 = a_2 + d_2;
        sw_2[2] = __as_u32(mixed_2);
        words[28 + woff_4 + 2] = sw_2[2];
        __nv_bfloat162 a_3 = __as_bf16x2(sw_2[3]);
        __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_4 + 3]);
        __nv_bfloat162 mixed_3 = a_3 + d_3;
        sw_2[3] = __as_u32(mixed_3);
        words[28 + woff_4 + 3] = sw_2[3];
        {
            int4 _iv4 = make_int4(sw_2[0 + 0], sw_2[0 + 1], sw_2[0 + 2], sw_2[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_4)) + 0) = _iv4;
        }
        float vals_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_2[_pair * 2])[0]), "=f"((&vals_2[_pair * 2])[1])
                : "r"(sw_2[_pair]));
        }
        fsrc[56] = vals_2[0];
        fsrc[57] = vals_2[1];
        fsrc[58] = vals_2[2];
        fsrc[59] = vals_2[3];
        fsrc[60] = vals_2[4];
        fsrc[61] = vals_2[5];
        fsrc[62] = vals_2[6];
        fsrc[63] = vals_2[7];
        float2 v_8 = make_float2(vals_2[0], vals_2[1]);
        float2 qp_8 = make_float2(q[0], q[1]);
        sq[2] = fma_f32x2_rn_noftz(v_8, v_8, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_8, qp_8, dot[2]);
        float2 v_9 = make_float2(vals_2[2], vals_2[3]);
        float2 qp_9 = make_float2(q[2], q[3]);
        sq[2] = fma_f32x2_rn_noftz(v_9, v_9, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_9, qp_9, dot[2]);
        float2 v_10 = make_float2(vals_2[4], vals_2[5]);
        float2 qp_10 = make_float2(q[4], q[5]);
        sq[2] = fma_f32x2_rn_noftz(v_10, v_10, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_10, qp_10, dot[2]);
        float2 v_11 = make_float2(vals_2[6], vals_2[7]);
        float2 qp_11 = make_float2(q[6], q[7]);
        sq[2] = fma_f32x2_rn_noftz(v_11, v_11, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_11, qp_11, dot[2]);
        int base_5 = (2 + group) * 1024 + thread * 8;
        const int woff_5 = 4;
        unsigned int sw_3[4];
        sw_3[0] = words[woff_5];
        sw_3[1] = words[woff_5 + 1];
        sw_3[2] = words[woff_5 + 2];
        sw_3[3] = words[woff_5 + 3];
        float vals_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_3[_pair * 2])[0]), "=f"((&vals_3[_pair * 2])[1])
                : "r"(sw_3[_pair]));
        }
        fsrc[8] = vals_3[0];
        fsrc[9] = vals_3[1];
        fsrc[10] = vals_3[2];
        fsrc[11] = vals_3[3];
        fsrc[12] = vals_3[4];
        fsrc[13] = vals_3[5];
        fsrc[14] = vals_3[6];
        fsrc[15] = vals_3[7];
        float2 v_12 = make_float2(vals_3[0], vals_3[1]);
        float2 qp_12 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_12, v_12, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_12, qp_12, dot[0]);
        float2 v_13 = make_float2(vals_3[2], vals_3[3]);
        float2 qp_13 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_13, v_13, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_13, qp_13, dot[0]);
        float2 v_14 = make_float2(vals_3[4], vals_3[5]);
        float2 qp_14 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_14, v_14, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_14, qp_14, dot[0]);
        float2 v_15 = make_float2(vals_3[6], vals_3[7]);
        float2 qp_15 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_15, v_15, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_15, qp_15, dot[0]);
        unsigned int sw_4[4];
        sw_4[0] = words[14 + woff_5];
        sw_4[1] = words[14 + woff_5 + 1];
        sw_4[2] = words[14 + woff_5 + 2];
        sw_4[3] = words[14 + woff_5 + 3];
        float vals_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_4[_pair * 2])[0]), "=f"((&vals_4[_pair * 2])[1])
                : "r"(sw_4[_pair]));
        }
        fsrc[36] = vals_4[0];
        fsrc[37] = vals_4[1];
        fsrc[38] = vals_4[2];
        fsrc[39] = vals_4[3];
        fsrc[40] = vals_4[4];
        fsrc[41] = vals_4[5];
        fsrc[42] = vals_4[6];
        fsrc[43] = vals_4[7];
        float2 v_16 = make_float2(vals_4[0], vals_4[1]);
        float2 qp_16 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_16, v_16, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_16, qp_16, dot[1]);
        float2 v_17 = make_float2(vals_4[2], vals_4[3]);
        float2 qp_17 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_17, v_17, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_17, qp_17, dot[1]);
        float2 v_18 = make_float2(vals_4[4], vals_4[5]);
        float2 qp_18 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_18, v_18, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_18, qp_18, dot[1]);
        float2 v_19 = make_float2(vals_4[6], vals_4[7]);
        float2 qp_19 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_19, v_19, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_19, qp_19, dot[1]);
        unsigned int sw_5[4];
        sw_5[0] = words[28 + woff_5];
        sw_5[1] = words[28 + woff_5 + 1];
        sw_5[2] = words[28 + woff_5 + 2];
        sw_5[3] = words[28 + woff_5 + 3];
        __nv_bfloat162 a_4 = __as_bf16x2(sw_5[0]);
        __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_5]);
        __nv_bfloat162 mixed_4 = a_4 + d_4;
        sw_5[0] = __as_u32(mixed_4);
        words[28 + woff_5] = sw_5[0];
        __nv_bfloat162 a_5 = __as_bf16x2(sw_5[1]);
        __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_5 + 1]);
        __nv_bfloat162 mixed_5 = a_5 + d_5;
        sw_5[1] = __as_u32(mixed_5);
        words[28 + woff_5 + 1] = sw_5[1];
        __nv_bfloat162 a_6 = __as_bf16x2(sw_5[2]);
        __nv_bfloat162 d_6 = __as_bf16x2(dwords[woff_5 + 2]);
        __nv_bfloat162 mixed_6 = a_6 + d_6;
        sw_5[2] = __as_u32(mixed_6);
        words[28 + woff_5 + 2] = sw_5[2];
        __nv_bfloat162 a_7 = __as_bf16x2(sw_5[3]);
        __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_5 + 3]);
        __nv_bfloat162 mixed_7 = a_7 + d_7;
        sw_5[3] = __as_u32(mixed_7);
        words[28 + woff_5 + 3] = sw_5[3];
        {
            int4 _iv4 = make_int4(sw_5[0 + 0], sw_5[0 + 1], sw_5[0 + 2], sw_5[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_5)) + 0) = _iv4;
        }
        float vals_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_5[_pair * 2])[0]), "=f"((&vals_5[_pair * 2])[1])
                : "r"(sw_5[_pair]));
        }
        fsrc[64] = vals_5[0];
        fsrc[65] = vals_5[1];
        fsrc[66] = vals_5[2];
        fsrc[67] = vals_5[3];
        fsrc[68] = vals_5[4];
        fsrc[69] = vals_5[5];
        fsrc[70] = vals_5[6];
        fsrc[71] = vals_5[7];
        float2 v_20 = make_float2(vals_5[0], vals_5[1]);
        float2 qp_20 = make_float2(q[8], q[9]);
        sq[2] = fma_f32x2_rn_noftz(v_20, v_20, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_20, qp_20, dot[2]);
        float2 v_21 = make_float2(vals_5[2], vals_5[3]);
        float2 qp_21 = make_float2(q[10], q[11]);
        sq[2] = fma_f32x2_rn_noftz(v_21, v_21, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_21, qp_21, dot[2]);
        float2 v_22 = make_float2(vals_5[4], vals_5[5]);
        float2 qp_22 = make_float2(q[12], q[13]);
        sq[2] = fma_f32x2_rn_noftz(v_22, v_22, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_22, qp_22, dot[2]);
        float2 v_23 = make_float2(vals_5[6], vals_5[7]);
        float2 qp_23 = make_float2(q[14], q[15]);
        sq[2] = fma_f32x2_rn_noftz(v_23, v_23, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_23, qp_23, dot[2]);
        int base_6 = (4 + group) * 1024 + thread * 8;
        const int woff_6 = 8;
        unsigned int sw_6[4];
        sw_6[0] = words[woff_6];
        sw_6[1] = words[woff_6 + 1];
        sw_6[2] = words[woff_6 + 2];
        sw_6[3] = words[woff_6 + 3];
        float vals_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_6[_pair * 2])[0]), "=f"((&vals_6[_pair * 2])[1])
                : "r"(sw_6[_pair]));
        }
        fsrc[16] = vals_6[0];
        fsrc[17] = vals_6[1];
        fsrc[18] = vals_6[2];
        fsrc[19] = vals_6[3];
        fsrc[20] = vals_6[4];
        fsrc[21] = vals_6[5];
        fsrc[22] = vals_6[6];
        fsrc[23] = vals_6[7];
        float2 v_24 = make_float2(vals_6[0], vals_6[1]);
        float2 qp_24 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_24, v_24, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_24, qp_24, dot[0]);
        float2 v_25 = make_float2(vals_6[2], vals_6[3]);
        float2 qp_25 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_25, v_25, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_25, qp_25, dot[0]);
        float2 v_26 = make_float2(vals_6[4], vals_6[5]);
        float2 qp_26 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_26, v_26, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_26, qp_26, dot[0]);
        float2 v_27 = make_float2(vals_6[6], vals_6[7]);
        float2 qp_27 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_27, v_27, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_27, qp_27, dot[0]);
        unsigned int sw_7[4];
        sw_7[0] = words[14 + woff_6];
        sw_7[1] = words[14 + woff_6 + 1];
        sw_7[2] = words[14 + woff_6 + 2];
        sw_7[3] = words[14 + woff_6 + 3];
        float vals_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_7[_pair * 2])[0]), "=f"((&vals_7[_pair * 2])[1])
                : "r"(sw_7[_pair]));
        }
        fsrc[44] = vals_7[0];
        fsrc[45] = vals_7[1];
        fsrc[46] = vals_7[2];
        fsrc[47] = vals_7[3];
        fsrc[48] = vals_7[4];
        fsrc[49] = vals_7[5];
        fsrc[50] = vals_7[6];
        fsrc[51] = vals_7[7];
        float2 v_28 = make_float2(vals_7[0], vals_7[1]);
        float2 qp_28 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_28, v_28, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_28, qp_28, dot[1]);
        float2 v_29 = make_float2(vals_7[2], vals_7[3]);
        float2 qp_29 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_29, v_29, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_29, qp_29, dot[1]);
        float2 v_30 = make_float2(vals_7[4], vals_7[5]);
        float2 qp_30 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_30, v_30, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_30, qp_30, dot[1]);
        float2 v_31 = make_float2(vals_7[6], vals_7[7]);
        float2 qp_31 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_31, v_31, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_31, qp_31, dot[1]);
        unsigned int sw_8[4];
        sw_8[0] = words[28 + woff_6];
        sw_8[1] = words[28 + woff_6 + 1];
        sw_8[2] = words[28 + woff_6 + 2];
        sw_8[3] = words[28 + woff_6 + 3];
        __nv_bfloat162 a_8 = __as_bf16x2(sw_8[0]);
        __nv_bfloat162 d_8 = __as_bf16x2(dwords[woff_6]);
        __nv_bfloat162 mixed_8 = a_8 + d_8;
        sw_8[0] = __as_u32(mixed_8);
        words[28 + woff_6] = sw_8[0];
        __nv_bfloat162 a_9 = __as_bf16x2(sw_8[1]);
        __nv_bfloat162 d_9 = __as_bf16x2(dwords[woff_6 + 1]);
        __nv_bfloat162 mixed_9 = a_9 + d_9;
        sw_8[1] = __as_u32(mixed_9);
        words[28 + woff_6 + 1] = sw_8[1];
        __nv_bfloat162 a_10 = __as_bf16x2(sw_8[2]);
        __nv_bfloat162 d_10 = __as_bf16x2(dwords[woff_6 + 2]);
        __nv_bfloat162 mixed_10 = a_10 + d_10;
        sw_8[2] = __as_u32(mixed_10);
        words[28 + woff_6 + 2] = sw_8[2];
        __nv_bfloat162 a_11 = __as_bf16x2(sw_8[3]);
        __nv_bfloat162 d_11 = __as_bf16x2(dwords[woff_6 + 3]);
        __nv_bfloat162 mixed_11 = a_11 + d_11;
        sw_8[3] = __as_u32(mixed_11);
        words[28 + woff_6 + 3] = sw_8[3];
        {
            int4 _iv4 = make_int4(sw_8[0 + 0], sw_8[0 + 1], sw_8[0 + 2], sw_8[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
        }
        float vals_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_8[_pair * 2])[0]), "=f"((&vals_8[_pair * 2])[1])
                : "r"(sw_8[_pair]));
        }
        fsrc[72] = vals_8[0];
        fsrc[73] = vals_8[1];
        fsrc[74] = vals_8[2];
        fsrc[75] = vals_8[3];
        fsrc[76] = vals_8[4];
        fsrc[77] = vals_8[5];
        fsrc[78] = vals_8[6];
        fsrc[79] = vals_8[7];
        float2 v_32 = make_float2(vals_8[0], vals_8[1]);
        float2 qp_32 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_32, v_32, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_32, qp_32, dot[2]);
        float2 v_33 = make_float2(vals_8[2], vals_8[3]);
        float2 qp_33 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_33, v_33, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_33, qp_33, dot[2]);
        float2 v_34 = make_float2(vals_8[4], vals_8[5]);
        float2 qp_34 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_34, v_34, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_34, qp_34, dot[2]);
        float2 v_35 = make_float2(vals_8[6], vals_8[7]);
        float2 qp_35 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_35, v_35, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_35, qp_35, dot[2]);
        int base_7 = 6144 + group * 512 + thread * 4;
        const int woff_7 = 12;
        unsigned int sw_9[4];
        sw_9[0] = words[woff_7];
        sw_9[1] = words[woff_7 + 1];
        float vals_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_9[_pair * 2])[0]), "=f"((&vals_9[_pair * 2])[1])
                : "r"(sw_9[_pair]));
        }
        fsrc[24] = vals_9[0];
        fsrc[25] = vals_9[1];
        fsrc[26] = vals_9[2];
        fsrc[27] = vals_9[3];
        float2 v_36 = make_float2(vals_9[0], vals_9[1]);
        sq[0] = fma_f32x2_rn_noftz(v_36, v_36, sq[0]);
        float2 v_37 = make_float2(vals_9[2], vals_9[3]);
        sq[0] = fma_f32x2_rn_noftz(v_37, v_37, sq[0]);
        float2 v_38 = make_float2(vals_9[0], vals_9[1]);
        float2 qp_36 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_38, qp_36, dot[0]);
        float2 v_39 = make_float2(vals_9[2], vals_9[3]);
        float2 qp_37 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_39, qp_37, dot[0]);
        unsigned int sw_10[4];
        sw_10[0] = words[14 + woff_7];
        sw_10[1] = words[14 + woff_7 + 1];
        float vals_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_10[_pair * 2])[0]), "=f"((&vals_10[_pair * 2])[1])
                : "r"(sw_10[_pair]));
        }
        fsrc[52] = vals_10[0];
        fsrc[53] = vals_10[1];
        fsrc[54] = vals_10[2];
        fsrc[55] = vals_10[3];
        float2 v_40 = make_float2(vals_10[0], vals_10[1]);
        sq[1] = fma_f32x2_rn_noftz(v_40, v_40, sq[1]);
        float2 v_41 = make_float2(vals_10[2], vals_10[3]);
        sq[1] = fma_f32x2_rn_noftz(v_41, v_41, sq[1]);
        float2 v_42 = make_float2(vals_10[0], vals_10[1]);
        float2 qp_38 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_42, qp_38, dot[1]);
        float2 v_43 = make_float2(vals_10[2], vals_10[3]);
        float2 qp_39 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_43, qp_39, dot[1]);
        unsigned int sw_11[4];
        sw_11[0] = words[28 + woff_7];
        sw_11[1] = words[28 + woff_7 + 1];
        __nv_bfloat162 a_12 = __as_bf16x2(sw_11[0]);
        __nv_bfloat162 d_12 = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed_12 = a_12 + d_12;
        sw_11[0] = __as_u32(mixed_12);
        words[28 + woff_7] = sw_11[0];
        __nv_bfloat162 a_13 = __as_bf16x2(sw_11[1]);
        __nv_bfloat162 d_13 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_13 = a_13 + d_13;
        sw_11[1] = __as_u32(mixed_13);
        words[28 + woff_7 + 1] = sw_11[1];
        {
            int2 _iv2 = make_int2(sw_11[0 + 0], sw_11[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_7)) + 0) = _iv2;
        }
        float vals_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&vals_11[_pair * 2])[0]), "=f"((&vals_11[_pair * 2])[1])
                : "r"(sw_11[_pair]));
        }
        fsrc[80] = vals_11[0];
        fsrc[81] = vals_11[1];
        fsrc[82] = vals_11[2];
        fsrc[83] = vals_11[3];
        float2 v_44 = make_float2(vals_11[0], vals_11[1]);
        sq[2] = fma_f32x2_rn_noftz(v_44, v_44, sq[2]);
        float2 v_45 = make_float2(vals_11[2], vals_11[3]);
        sq[2] = fma_f32x2_rn_noftz(v_45, v_45, sq[2]);
        float2 v_46 = make_float2(vals_11[0], vals_11[1]);
        float2 qp_40 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_46, qp_40, dot[2]);
        float2 v_47 = make_float2(vals_11[2], vals_11[3]);
        float2 qp_41 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_47, qp_41, dot[2]);
        float2 pairs[3];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[2] = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        unsigned long long bits_0 = 0;
        float2 _bit_cast_src_24 = pairs[0];
        bits_0 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_24)[0];
        unsigned long long peerbits_0 = __shfl_xor_sync(0xFFFFFFFF, bits_0, 16);
        float2 peer_0 = make_float2(0.0f, 0.0f);
        peer_0 = reinterpret_cast<float2*>(&peerbits_0)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_0);
        unsigned long long bits_1 = 0;
        float2 _bit_cast_src_25 = pairs[1];
        bits_1 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_25)[0];
        unsigned long long peerbits_1 = __shfl_xor_sync(0xFFFFFFFF, bits_1, 16);
        float2 peer_1 = make_float2(0.0f, 0.0f);
        peer_1 = reinterpret_cast<float2*>(&peerbits_1)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_1);
        unsigned long long bits_2 = 0;
        float2 _bit_cast_src_26 = pairs[2];
        bits_2 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_26)[0];
        unsigned long long peerbits_2 = __shfl_xor_sync(0xFFFFFFFF, bits_2, 16);
        float2 peer_2 = make_float2(0.0f, 0.0f);
        peer_2 = reinterpret_cast<float2*>(&peerbits_2)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_2);
        unsigned long long bits_3 = 0;
        float2 _bit_cast_src_27 = pairs[0];
        bits_3 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_27)[0];
        unsigned long long peerbits_3 = __shfl_xor_sync(0xFFFFFFFF, bits_3, 8);
        float2 peer_3 = make_float2(0.0f, 0.0f);
        peer_3 = reinterpret_cast<float2*>(&peerbits_3)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_3);
        unsigned long long bits_4 = 0;
        float2 _bit_cast_src_28 = pairs[1];
        bits_4 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_28)[0];
        unsigned long long peerbits_4 = __shfl_xor_sync(0xFFFFFFFF, bits_4, 8);
        float2 peer_4 = make_float2(0.0f, 0.0f);
        peer_4 = reinterpret_cast<float2*>(&peerbits_4)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_4);
        unsigned long long bits_5 = 0;
        float2 _bit_cast_src_29 = pairs[2];
        bits_5 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_29)[0];
        unsigned long long peerbits_5 = __shfl_xor_sync(0xFFFFFFFF, bits_5, 8);
        float2 peer_5 = make_float2(0.0f, 0.0f);
        peer_5 = reinterpret_cast<float2*>(&peerbits_5)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_5);
        unsigned long long bits_6 = 0;
        float2 _bit_cast_src_30 = pairs[0];
        bits_6 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_30)[0];
        unsigned long long peerbits_6 = __shfl_xor_sync(0xFFFFFFFF, bits_6, 4);
        float2 peer_6 = make_float2(0.0f, 0.0f);
        peer_6 = reinterpret_cast<float2*>(&peerbits_6)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_6);
        unsigned long long bits_7 = 0;
        float2 _bit_cast_src_31 = pairs[1];
        bits_7 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_31)[0];
        unsigned long long peerbits_7 = __shfl_xor_sync(0xFFFFFFFF, bits_7, 4);
        float2 peer_7 = make_float2(0.0f, 0.0f);
        peer_7 = reinterpret_cast<float2*>(&peerbits_7)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_7);
        unsigned long long bits_8 = 0;
        float2 _bit_cast_src_32 = pairs[2];
        bits_8 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_32)[0];
        unsigned long long peerbits_8 = __shfl_xor_sync(0xFFFFFFFF, bits_8, 4);
        float2 peer_8 = make_float2(0.0f, 0.0f);
        peer_8 = reinterpret_cast<float2*>(&peerbits_8)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_8);
        unsigned long long bits_9 = 0;
        float2 _bit_cast_src_33 = pairs[0];
        bits_9 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_33)[0];
        unsigned long long peerbits_9 = __shfl_xor_sync(0xFFFFFFFF, bits_9, 2);
        float2 peer_9 = make_float2(0.0f, 0.0f);
        peer_9 = reinterpret_cast<float2*>(&peerbits_9)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_9);
        unsigned long long bits_10 = 0;
        float2 _bit_cast_src_34 = pairs[1];
        bits_10 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_34)[0];
        unsigned long long peerbits_10 = __shfl_xor_sync(0xFFFFFFFF, bits_10, 2);
        float2 peer_10 = make_float2(0.0f, 0.0f);
        peer_10 = reinterpret_cast<float2*>(&peerbits_10)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_10);
        unsigned long long bits_11 = 0;
        float2 _bit_cast_src_35 = pairs[2];
        bits_11 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_35)[0];
        unsigned long long peerbits_11 = __shfl_xor_sync(0xFFFFFFFF, bits_11, 2);
        float2 peer_11 = make_float2(0.0f, 0.0f);
        peer_11 = reinterpret_cast<float2*>(&peerbits_11)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_11);
        unsigned long long bits_12 = 0;
        float2 _bit_cast_src_36 = pairs[0];
        bits_12 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_36)[0];
        unsigned long long peerbits_12 = __shfl_xor_sync(0xFFFFFFFF, bits_12, 1);
        float2 peer_12 = make_float2(0.0f, 0.0f);
        peer_12 = reinterpret_cast<float2*>(&peerbits_12)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_12);
        unsigned long long bits_13 = 0;
        float2 _bit_cast_src_37 = pairs[1];
        bits_13 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_37)[0];
        unsigned long long peerbits_13 = __shfl_xor_sync(0xFFFFFFFF, bits_13, 1);
        float2 peer_13 = make_float2(0.0f, 0.0f);
        peer_13 = reinterpret_cast<float2*>(&peerbits_13)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_13);
        unsigned long long bits_14 = 0;
        float2 _bit_cast_src_38 = pairs[2];
        bits_14 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_38)[0];
        unsigned long long peerbits_14 = __shfl_xor_sync(0xFFFFFFFF, bits_14, 1);
        float2 peer_14 = make_float2(0.0f, 0.0f);
        peer_14 = reinterpret_cast<float2*>(&peerbits_14)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_14);
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
        float _shfl_down_sync_39 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_39;
        float _shfl_down_sync_40 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_40;
        float _shfl_down_sync_41 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_41;
        float _shfl_down_sync_42 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_42;
        float _shfl_down_sync_43 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_43;
        float _shfl_down_sync_44 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_44;
        float logit = 0.0f;
        if (stat_n < 3 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[3];
        float _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_45;
        float _shfl_sync_46 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_46;
        float _shfl_sync_47 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_47;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk_0 = -3.4028235e+38f;
        max_chunk_0 = fmaxf(max_chunk_0, logits[0]);
        max_chunk_0 = fmaxf(max_chunk_0, logits[1]);
        max_chunk_0 = fmaxf(max_chunk_0, logits[2]);
        float max_new_0 = fmaxf(max_running, max_chunk_0);
        float correction_0 = approx_exp2((max_running - max_new_0) * 1.442695f);
        float weights_0[3];
        float sum_weights_0 = 0.0f;
        float _exp2_48 = approx_exp2((logits[0] - max_new_0) * 1.442695f);
        weights_0[0] = _exp2_48;
        sum_weights_0 += weights_0[0];
        float _exp2_49 = approx_exp2((logits[1] - max_new_0) * 1.442695f);
        weights_0[1] = _exp2_49;
        sum_weights_0 += weights_0[1];
        float _exp2_50 = approx_exp2((logits[2] - max_new_0) * 1.442695f);
        weights_0[2] = _exp2_50;
        sum_weights_0 += weights_0[2];
        float2 corr_0 = make_float2(correction_0, correction_0);
        const int woff_8 = 0;
        int base_8 = group * 1024 + thread * 8;
        float2 a_14[4];
        float2 previous_0 = make_float2(acc[0], acc[1]);
        a_14[0] = mul_f32x2_noftz(previous_0, corr_0);
        float2 previous_1 = make_float2(acc[2], acc[3]);
        a_14[1] = mul_f32x2_noftz(previous_1, corr_0);
        float2 previous_2 = make_float2(acc[4], acc[5]);
        a_14[2] = mul_f32x2_noftz(previous_2, corr_0);
        float2 previous_3 = make_float2(acc[6], acc[7]);
        a_14[3] = mul_f32x2_noftz(previous_3, corr_0);
        float2 weight_0 = make_float2(weights_0[0], weights_0[0]);
        float2 v_48 = make_float2(fsrc[0], fsrc[1]);
        a_14[0] = fma_f32x2_rn_noftz(weight_0, v_48, a_14[0]);
        float2 v_49 = make_float2(fsrc[2], fsrc[3]);
        a_14[1] = fma_f32x2_rn_noftz(weight_0, v_49, a_14[1]);
        float2 v_50 = make_float2(fsrc[4], fsrc[5]);
        a_14[2] = fma_f32x2_rn_noftz(weight_0, v_50, a_14[2]);
        float2 v_51 = make_float2(fsrc[6], fsrc[7]);
        a_14[3] = fma_f32x2_rn_noftz(weight_0, v_51, a_14[3]);
        float2 weight_1 = make_float2(weights_0[1], weights_0[1]);
        float2 v_52 = make_float2(fsrc[28], fsrc[29]);
        a_14[0] = fma_f32x2_rn_noftz(weight_1, v_52, a_14[0]);
        float2 v_53 = make_float2(fsrc[30], fsrc[31]);
        a_14[1] = fma_f32x2_rn_noftz(weight_1, v_53, a_14[1]);
        float2 v_54 = make_float2(fsrc[32], fsrc[33]);
        a_14[2] = fma_f32x2_rn_noftz(weight_1, v_54, a_14[2]);
        float2 v_55 = make_float2(fsrc[34], fsrc[35]);
        a_14[3] = fma_f32x2_rn_noftz(weight_1, v_55, a_14[3]);
        float2 weight_2 = make_float2(weights_0[2], weights_0[2]);
        float2 v_56 = make_float2(fsrc[56], fsrc[57]);
        a_14[0] = fma_f32x2_rn_noftz(weight_2, v_56, a_14[0]);
        float2 v_57 = make_float2(fsrc[58], fsrc[59]);
        a_14[1] = fma_f32x2_rn_noftz(weight_2, v_57, a_14[1]);
        float2 v_58 = make_float2(fsrc[60], fsrc[61]);
        a_14[2] = fma_f32x2_rn_noftz(weight_2, v_58, a_14[2]);
        float2 v_59 = make_float2(fsrc[62], fsrc[63]);
        a_14[3] = fma_f32x2_rn_noftz(weight_2, v_59, a_14[3]);
        acc[0] = a_14[0].x;
        acc[1] = a_14[0].y;
        acc[2] = a_14[1].x;
        acc[3] = a_14[1].y;
        acc[4] = a_14[2].x;
        acc[5] = a_14[2].y;
        acc[6] = a_14[3].x;
        acc[7] = a_14[3].y;
        const int woff_9 = 4;
        int base_9 = (2 + group) * 1024 + thread * 8;
        float2 a_15[4];
        float2 previous_4 = make_float2(acc[8], acc[9]);
        a_15[0] = mul_f32x2_noftz(previous_4, corr_0);
        float2 previous_5 = make_float2(acc[10], acc[11]);
        a_15[1] = mul_f32x2_noftz(previous_5, corr_0);
        float2 previous_6 = make_float2(acc[12], acc[13]);
        a_15[2] = mul_f32x2_noftz(previous_6, corr_0);
        float2 previous_7 = make_float2(acc[14], acc[15]);
        a_15[3] = mul_f32x2_noftz(previous_7, corr_0);
        float2 weight_3 = make_float2(weights_0[0], weights_0[0]);
        float2 v_60 = make_float2(fsrc[8], fsrc[9]);
        a_15[0] = fma_f32x2_rn_noftz(weight_3, v_60, a_15[0]);
        float2 v_61 = make_float2(fsrc[10], fsrc[11]);
        a_15[1] = fma_f32x2_rn_noftz(weight_3, v_61, a_15[1]);
        float2 v_62 = make_float2(fsrc[12], fsrc[13]);
        a_15[2] = fma_f32x2_rn_noftz(weight_3, v_62, a_15[2]);
        float2 v_63 = make_float2(fsrc[14], fsrc[15]);
        a_15[3] = fma_f32x2_rn_noftz(weight_3, v_63, a_15[3]);
        float2 weight_4 = make_float2(weights_0[1], weights_0[1]);
        float2 v_64 = make_float2(fsrc[36], fsrc[37]);
        a_15[0] = fma_f32x2_rn_noftz(weight_4, v_64, a_15[0]);
        float2 v_65 = make_float2(fsrc[38], fsrc[39]);
        a_15[1] = fma_f32x2_rn_noftz(weight_4, v_65, a_15[1]);
        float2 v_66 = make_float2(fsrc[40], fsrc[41]);
        a_15[2] = fma_f32x2_rn_noftz(weight_4, v_66, a_15[2]);
        float2 v_67 = make_float2(fsrc[42], fsrc[43]);
        a_15[3] = fma_f32x2_rn_noftz(weight_4, v_67, a_15[3]);
        float2 weight_5 = make_float2(weights_0[2], weights_0[2]);
        float2 v_68 = make_float2(fsrc[64], fsrc[65]);
        a_15[0] = fma_f32x2_rn_noftz(weight_5, v_68, a_15[0]);
        float2 v_69 = make_float2(fsrc[66], fsrc[67]);
        a_15[1] = fma_f32x2_rn_noftz(weight_5, v_69, a_15[1]);
        float2 v_70 = make_float2(fsrc[68], fsrc[69]);
        a_15[2] = fma_f32x2_rn_noftz(weight_5, v_70, a_15[2]);
        float2 v_71 = make_float2(fsrc[70], fsrc[71]);
        a_15[3] = fma_f32x2_rn_noftz(weight_5, v_71, a_15[3]);
        acc[8] = a_15[0].x;
        acc[9] = a_15[0].y;
        acc[10] = a_15[1].x;
        acc[11] = a_15[1].y;
        acc[12] = a_15[2].x;
        acc[13] = a_15[2].y;
        acc[14] = a_15[3].x;
        acc[15] = a_15[3].y;
        const int woff_10 = 8;
        int base_10 = (4 + group) * 1024 + thread * 8;
        float2 a_16[4];
        float2 previous_8 = make_float2(acc[16], acc[17]);
        a_16[0] = mul_f32x2_noftz(previous_8, corr_0);
        float2 previous_9 = make_float2(acc[18], acc[19]);
        a_16[1] = mul_f32x2_noftz(previous_9, corr_0);
        float2 previous_10 = make_float2(acc[20], acc[21]);
        a_16[2] = mul_f32x2_noftz(previous_10, corr_0);
        float2 previous_11 = make_float2(acc[22], acc[23]);
        a_16[3] = mul_f32x2_noftz(previous_11, corr_0);
        float2 weight_6 = make_float2(weights_0[0], weights_0[0]);
        float2 v_72 = make_float2(fsrc[16], fsrc[17]);
        a_16[0] = fma_f32x2_rn_noftz(weight_6, v_72, a_16[0]);
        float2 v_73 = make_float2(fsrc[18], fsrc[19]);
        a_16[1] = fma_f32x2_rn_noftz(weight_6, v_73, a_16[1]);
        float2 v_74 = make_float2(fsrc[20], fsrc[21]);
        a_16[2] = fma_f32x2_rn_noftz(weight_6, v_74, a_16[2]);
        float2 v_75 = make_float2(fsrc[22], fsrc[23]);
        a_16[3] = fma_f32x2_rn_noftz(weight_6, v_75, a_16[3]);
        float2 weight_7 = make_float2(weights_0[1], weights_0[1]);
        float2 v_76 = make_float2(fsrc[44], fsrc[45]);
        a_16[0] = fma_f32x2_rn_noftz(weight_7, v_76, a_16[0]);
        float2 v_77 = make_float2(fsrc[46], fsrc[47]);
        a_16[1] = fma_f32x2_rn_noftz(weight_7, v_77, a_16[1]);
        float2 v_78 = make_float2(fsrc[48], fsrc[49]);
        a_16[2] = fma_f32x2_rn_noftz(weight_7, v_78, a_16[2]);
        float2 v_79 = make_float2(fsrc[50], fsrc[51]);
        a_16[3] = fma_f32x2_rn_noftz(weight_7, v_79, a_16[3]);
        float2 weight_8 = make_float2(weights_0[2], weights_0[2]);
        float2 v_80 = make_float2(fsrc[72], fsrc[73]);
        a_16[0] = fma_f32x2_rn_noftz(weight_8, v_80, a_16[0]);
        float2 v_81 = make_float2(fsrc[74], fsrc[75]);
        a_16[1] = fma_f32x2_rn_noftz(weight_8, v_81, a_16[1]);
        float2 v_82 = make_float2(fsrc[76], fsrc[77]);
        a_16[2] = fma_f32x2_rn_noftz(weight_8, v_82, a_16[2]);
        float2 v_83 = make_float2(fsrc[78], fsrc[79]);
        a_16[3] = fma_f32x2_rn_noftz(weight_8, v_83, a_16[3]);
        acc[16] = a_16[0].x;
        acc[17] = a_16[0].y;
        acc[18] = a_16[1].x;
        acc[19] = a_16[1].y;
        acc[20] = a_16[2].x;
        acc[21] = a_16[2].y;
        acc[22] = a_16[3].x;
        acc[23] = a_16[3].y;
        const int woff_11 = 12;
        int base_11 = 6144 + group * 512 + thread * 4;
        float2 a_17[4];
        float2 previous_12 = make_float2(acc[24], acc[25]);
        a_17[0] = mul_f32x2_noftz(previous_12, corr_0);
        float2 previous_13 = make_float2(acc[26], acc[27]);
        a_17[1] = mul_f32x2_noftz(previous_13, corr_0);
        float2 weight_9 = make_float2(weights_0[0], weights_0[0]);
        float2 v_84 = make_float2(fsrc[24], fsrc[25]);
        a_17[0] = fma_f32x2_rn_noftz(weight_9, v_84, a_17[0]);
        float2 v_85 = make_float2(fsrc[26], fsrc[27]);
        a_17[1] = fma_f32x2_rn_noftz(weight_9, v_85, a_17[1]);
        float2 weight_10 = make_float2(weights_0[1], weights_0[1]);
        float2 v_86 = make_float2(fsrc[52], fsrc[53]);
        a_17[0] = fma_f32x2_rn_noftz(weight_10, v_86, a_17[0]);
        float2 v_87 = make_float2(fsrc[54], fsrc[55]);
        a_17[1] = fma_f32x2_rn_noftz(weight_10, v_87, a_17[1]);
        float2 weight_11 = make_float2(weights_0[2], weights_0[2]);
        float2 v_88 = make_float2(fsrc[80], fsrc[81]);
        a_17[0] = fma_f32x2_rn_noftz(weight_11, v_88, a_17[0]);
        float2 v_89 = make_float2(fsrc[82], fsrc[83]);
        a_17[1] = fma_f32x2_rn_noftz(weight_11, v_89, a_17[1]);
        acc[24] = a_17[0].x;
        acc[25] = a_17[0].y;
        acc[26] = a_17[1].x;
        acc[27] = a_17[1].y;
        sum_running = __fmaf_rn(sum_running, correction_0, sum_weights_0);
        max_running = max_new_0;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_90 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_90, v_90, output_sq_pair);
        float2 v_91 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_91, v_91, output_sq_pair);
        float2 v_92 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_92, v_92, output_sq_pair);
        float2 v_93 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_93, v_93, output_sq_pair);
        float2 v_94 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_94, v_94, output_sq_pair);
        float2 v_95 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_95, v_95, output_sq_pair);
        float2 v_96 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_96, v_96, output_sq_pair);
        float2 v_97 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_97, v_97, output_sq_pair);
        float2 v_98 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_98, v_98, output_sq_pair);
        float2 v_99 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_99, v_99, output_sq_pair);
        float2 v_100 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_100, v_100, output_sq_pair);
        float2 v_101 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_101, v_101, output_sq_pair);
        float2 v_102 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_102, v_102, output_sq_pair);
        float2 v_103 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_103, v_103, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_51;
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_53;
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_54;
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_55;
        if (lane == 0) {
            out_stats[warp_0] = output_sq;
        }
        __syncthreads();
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_56 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_56;
        float _shfl_down_sync_57 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_57;
        float _shfl_down_sync_58 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_58;
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_59 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_59;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_12 = group * 1024 + thread * 8;
        float output_values_0[8];
        const int value_idx_0 = 0;
        const int acc_idx_0 = value_idx_0;
        float2 _f32x2_mul_a_60 = make_float2(acc[acc_idx_0], acc[acc_idx_0 + 1]);
        float2 scaled_pair_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_61 = make_float2(wout[acc_idx_0], wout[acc_idx_0 + 1]);
        float2 normalized_pair_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_0) : "l"(*(const unsigned long long*)&scaled_pair_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
        output_values_0[value_idx_0] = normalized_pair_0.x;
        output_values_0[value_idx_0 + 1] = normalized_pair_0.y;
        const int value_idx_1 = 2;
        const int acc_idx_1 = value_idx_1;
        float2 _f32x2_mul_a_62 = make_float2(acc[acc_idx_1], acc[acc_idx_1 + 1]);
        float2 scaled_pair_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_63 = make_float2(wout[acc_idx_1], wout[acc_idx_1 + 1]);
        float2 normalized_pair_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_1) : "l"(*(const unsigned long long*)&scaled_pair_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
        output_values_0[value_idx_1] = normalized_pair_1.x;
        output_values_0[value_idx_1 + 1] = normalized_pair_1.y;
        const int value_idx_2 = 4;
        const int acc_idx_2 = value_idx_2;
        float2 _f32x2_mul_a_64 = make_float2(acc[acc_idx_2], acc[acc_idx_2 + 1]);
        float2 scaled_pair_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_65 = make_float2(wout[acc_idx_2], wout[acc_idx_2 + 1]);
        float2 normalized_pair_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_2) : "l"(*(const unsigned long long*)&scaled_pair_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
        output_values_0[value_idx_2] = normalized_pair_2.x;
        output_values_0[value_idx_2 + 1] = normalized_pair_2.y;
        const int value_idx_3 = 6;
        const int acc_idx_3 = value_idx_3;
        float2 _f32x2_mul_a_66 = make_float2(acc[acc_idx_3], acc[acc_idx_3 + 1]);
        float2 scaled_pair_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_67 = make_float2(wout[acc_idx_3], wout[acc_idx_3 + 1]);
        float2 normalized_pair_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_3) : "l"(*(const unsigned long long*)&scaled_pair_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
        output_values_0[value_idx_3] = normalized_pair_3.x;
        output_values_0[value_idx_3 + 1] = normalized_pair_3.y;
        unsigned int packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 0], output_values_0[_lp*2+1 + 0]));
            packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 2], output_values_0[_lp*2+1 + 2]));
            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 4], output_values_0[_lp*2+1 + 4]));
            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_0[_lp*2 + 6], output_values_0[_lp*2+1 + 6]));
            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_12))[0] = reinterpret_cast<int4*>(packed_0)[0];
        int base_13 = (2 + group) * 1024 + thread * 8;
        float output_values_1[8];
        const int value_idx_4 = 0;
        const int acc_idx_4 = 8 + value_idx_4;
        float2 _f32x2_mul_a_68 = make_float2(acc[acc_idx_4], acc[acc_idx_4 + 1]);
        float2 scaled_pair_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_69 = make_float2(wout[acc_idx_4], wout[acc_idx_4 + 1]);
        float2 normalized_pair_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_4) : "l"(*(const unsigned long long*)&scaled_pair_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
        output_values_1[value_idx_4] = normalized_pair_4.x;
        output_values_1[value_idx_4 + 1] = normalized_pair_4.y;
        const int value_idx_5 = 2;
        const int acc_idx_5 = 8 + value_idx_5;
        float2 _f32x2_mul_a_70 = make_float2(acc[acc_idx_5], acc[acc_idx_5 + 1]);
        float2 scaled_pair_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_71 = make_float2(wout[acc_idx_5], wout[acc_idx_5 + 1]);
        float2 normalized_pair_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_5) : "l"(*(const unsigned long long*)&scaled_pair_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
        output_values_1[value_idx_5] = normalized_pair_5.x;
        output_values_1[value_idx_5 + 1] = normalized_pair_5.y;
        const int value_idx_6 = 4;
        const int acc_idx_6 = 8 + value_idx_6;
        float2 _f32x2_mul_a_72 = make_float2(acc[acc_idx_6], acc[acc_idx_6 + 1]);
        float2 scaled_pair_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_73 = make_float2(wout[acc_idx_6], wout[acc_idx_6 + 1]);
        float2 normalized_pair_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_6) : "l"(*(const unsigned long long*)&scaled_pair_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
        output_values_1[value_idx_6] = normalized_pair_6.x;
        output_values_1[value_idx_6 + 1] = normalized_pair_6.y;
        const int value_idx_7 = 6;
        const int acc_idx_7 = 8 + value_idx_7;
        float2 _f32x2_mul_a_74 = make_float2(acc[acc_idx_7], acc[acc_idx_7 + 1]);
        float2 scaled_pair_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_75 = make_float2(wout[acc_idx_7], wout[acc_idx_7 + 1]);
        float2 normalized_pair_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_7) : "l"(*(const unsigned long long*)&scaled_pair_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
        output_values_1[value_idx_7] = normalized_pair_7.x;
        output_values_1[value_idx_7 + 1] = normalized_pair_7.y;
        unsigned int packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 0], output_values_1[_lp*2+1 + 0]));
            packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 2], output_values_1[_lp*2+1 + 2]));
            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 4], output_values_1[_lp*2+1 + 4]));
            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 6], output_values_1[_lp*2+1 + 6]));
            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_13))[0] = reinterpret_cast<int4*>(packed_1)[0];
        int base_14 = (4 + group) * 1024 + thread * 8;
        float output_values_2[8];
        const int value_idx_8 = 0;
        const int acc_idx_8 = 16 + value_idx_8;
        float2 _f32x2_mul_a_76 = make_float2(acc[acc_idx_8], acc[acc_idx_8 + 1]);
        float2 scaled_pair_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_77 = make_float2(wout[acc_idx_8], wout[acc_idx_8 + 1]);
        float2 normalized_pair_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_8) : "l"(*(const unsigned long long*)&scaled_pair_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
        output_values_2[value_idx_8] = normalized_pair_8.x;
        output_values_2[value_idx_8 + 1] = normalized_pair_8.y;
        const int value_idx_9 = 2;
        const int acc_idx_9 = 16 + value_idx_9;
        float2 _f32x2_mul_a_78 = make_float2(acc[acc_idx_9], acc[acc_idx_9 + 1]);
        float2 scaled_pair_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_78), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_79 = make_float2(wout[acc_idx_9], wout[acc_idx_9 + 1]);
        float2 normalized_pair_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_9) : "l"(*(const unsigned long long*)&scaled_pair_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_79));
        output_values_2[value_idx_9] = normalized_pair_9.x;
        output_values_2[value_idx_9 + 1] = normalized_pair_9.y;
        const int value_idx_10 = 4;
        const int acc_idx_10 = 16 + value_idx_10;
        float2 _f32x2_mul_a_80 = make_float2(acc[acc_idx_10], acc[acc_idx_10 + 1]);
        float2 scaled_pair_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_80), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_81 = make_float2(wout[acc_idx_10], wout[acc_idx_10 + 1]);
        float2 normalized_pair_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_10) : "l"(*(const unsigned long long*)&scaled_pair_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_81));
        output_values_2[value_idx_10] = normalized_pair_10.x;
        output_values_2[value_idx_10 + 1] = normalized_pair_10.y;
        const int value_idx_11 = 6;
        const int acc_idx_11 = 16 + value_idx_11;
        float2 _f32x2_mul_a_82 = make_float2(acc[acc_idx_11], acc[acc_idx_11 + 1]);
        float2 scaled_pair_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_82), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_83 = make_float2(wout[acc_idx_11], wout[acc_idx_11 + 1]);
        float2 normalized_pair_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_11) : "l"(*(const unsigned long long*)&scaled_pair_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_83));
        output_values_2[value_idx_11] = normalized_pair_11.x;
        output_values_2[value_idx_11 + 1] = normalized_pair_11.y;
        unsigned int packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 0], output_values_2[_lp*2+1 + 0]));
            packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 2], output_values_2[_lp*2+1 + 2]));
            packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 4], output_values_2[_lp*2+1 + 4]));
            packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_2[_lp*2 + 6], output_values_2[_lp*2+1 + 6]));
            packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_14))[0] = reinterpret_cast<int4*>(packed_2)[0];
        int base_15 = 6144 + group * 512 + thread * 4;
        float output_values_3[8];
        const int value_idx_12 = 0;
        const int acc_idx_12 = 24 + value_idx_12;
        float2 _f32x2_mul_a_84 = make_float2(acc[acc_idx_12], acc[acc_idx_12 + 1]);
        float2 scaled_pair_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_85 = make_float2(wout[acc_idx_12], wout[acc_idx_12 + 1]);
        float2 normalized_pair_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_12) : "l"(*(const unsigned long long*)&scaled_pair_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_85));
        output_values_3[value_idx_12] = normalized_pair_12.x;
        output_values_3[value_idx_12 + 1] = normalized_pair_12.y;
        const int value_idx_13 = 2;
        const int acc_idx_13 = 24 + value_idx_13;
        float2 _f32x2_mul_a_86 = make_float2(acc[acc_idx_13], acc[acc_idx_13 + 1]);
        float2 scaled_pair_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_87 = make_float2(wout[acc_idx_13], wout[acc_idx_13 + 1]);
        float2 normalized_pair_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_13) : "l"(*(const unsigned long long*)&scaled_pair_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
        output_values_3[value_idx_13] = normalized_pair_13.x;
        output_values_3[value_idx_13 + 1] = normalized_pair_13.y;
        unsigned int packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_3[_lp*2 + 0], output_values_3[_lp*2+1 + 0]));
            packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_3[_lp*2 + 2], output_values_3[_lp*2+1 + 2]));
            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_15))[0] = reinterpret_cast<int2*>(packed_3)[0];
    }
}

} // extern "C"
