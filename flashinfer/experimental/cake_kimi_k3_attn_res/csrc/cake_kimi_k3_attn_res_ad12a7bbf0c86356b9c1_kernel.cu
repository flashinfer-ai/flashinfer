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
#define SMEM_STATS_STAGE_BYTES 512
#define SMEM_STATS_STRIDE 512
#define SMEM_OUT_STATS_OFF 512
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 640
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
kernel_cake_kimi_k3_attn_res_ad12a7bbf0c86356b9c1(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 512);
    const int out_stats_addr = smem + 512;

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
        unsigned int words[112];
        unsigned int dwords[14];
        float q[28];
        float wout[28];
        int base = group * 1024 + thread * 8;
        const int woff = 0;
        unsigned int _vec_load_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_1[0 + 0] = _uv4_1.x;
            _vec_load_1[0 + 1] = _uv4_1.y;
            _vec_load_1[0 + 2] = _uv4_1.z;
            _vec_load_1[0 + 3] = _uv4_1.w;
        }
        words[14 + woff] = _vec_load_1[0];
        words[14 + woff + 1] = _vec_load_1[1];
        words[14 + woff + 2] = _vec_load_1[2];
        words[14 + woff + 3] = _vec_load_1[3];
        unsigned int _vec_load_2[4];
        {
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_2[0 + 0] = _uv4_2.x;
            _vec_load_2[0 + 1] = _uv4_2.y;
            _vec_load_2[0 + 2] = _uv4_2.z;
            _vec_load_2[0 + 3] = _uv4_2.w;
        }
        words[28 + woff] = _vec_load_2[0];
        words[28 + woff + 1] = _vec_load_2[1];
        words[28 + woff + 2] = _vec_load_2[2];
        words[28 + woff + 3] = _vec_load_2[3];
        unsigned int _vec_load_3[4];
        {
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_3[0 + 0] = _uv4_3.x;
            _vec_load_3[0 + 1] = _uv4_3.y;
            _vec_load_3[0 + 2] = _uv4_3.z;
            _vec_load_3[0 + 3] = _uv4_3.w;
        }
        words[42 + woff] = _vec_load_3[0];
        words[42 + woff + 1] = _vec_load_3[1];
        words[42 + woff + 2] = _vec_load_3[2];
        words[42 + woff + 3] = _vec_load_3[3];
        unsigned int _vec_load_4[4];
        {
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_4[0 + 0] = _uv4_4.x;
            _vec_load_4[0 + 1] = _uv4_4.y;
            _vec_load_4[0 + 2] = _uv4_4.z;
            _vec_load_4[0 + 3] = _uv4_4.w;
        }
        words[56 + woff] = _vec_load_4[0];
        words[56 + woff + 1] = _vec_load_4[1];
        words[56 + woff + 2] = _vec_load_4[2];
        words[56 + woff + 3] = _vec_load_4[3];
        unsigned int _vec_load_5[4];
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_5[0 + 0] = _uv4_5.x;
            _vec_load_5[0 + 1] = _uv4_5.y;
            _vec_load_5[0 + 2] = _uv4_5.z;
            _vec_load_5[0 + 3] = _uv4_5.w;
        }
        words[70 + woff] = _vec_load_5[0];
        words[70 + woff + 1] = _vec_load_5[1];
        words[70 + woff + 2] = _vec_load_5[2];
        words[70 + woff + 3] = _vec_load_5[3];
        unsigned int _vec_load_6[4];
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_6[0 + 0] = _uv4_6.x;
            _vec_load_6[0 + 1] = _uv4_6.y;
            _vec_load_6[0 + 2] = _uv4_6.z;
            _vec_load_6[0 + 3] = _uv4_6.w;
        }
        words[84 + woff] = _vec_load_6[0];
        words[84 + woff + 1] = _vec_load_6[1];
        words[84 + woff + 2] = _vec_load_6[2];
        words[84 + woff + 3] = _vec_load_6[3];
        unsigned int _vec_load_7[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
            _vec_load_7[0 + 0] = _uv4_7.x;
            _vec_load_7[0 + 1] = _uv4_7.y;
            _vec_load_7[0 + 2] = _uv4_7.z;
            _vec_load_7[0 + 3] = _uv4_7.w;
        }
        words[98 + woff] = _vec_load_7[0];
        words[98 + woff + 1] = _vec_load_7[1];
        words[98 + woff + 2] = _vec_load_7[2];
        words[98 + woff + 3] = _vec_load_7[3];
        unsigned int _vec_load_8[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_8[0 + 0] = _uv4_8.x;
            _vec_load_8[0 + 1] = _uv4_8.y;
            _vec_load_8[0 + 2] = _uv4_8.z;
            _vec_load_8[0 + 3] = _uv4_8.w;
        }
        dwords[woff] = _vec_load_8[0];
        dwords[woff + 1] = _vec_load_8[1];
        dwords[woff + 2] = _vec_load_8[2];
        dwords[woff + 3] = _vec_load_8[3];
        float _vec_load_9[8];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        float _vec_load_10[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
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
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(output_norm_weight + base);
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
                        : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
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
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_12[0 + 0] = _uv4_12.x;
            _vec_load_12[0 + 1] = _uv4_12.y;
            _vec_load_12[0 + 2] = _uv4_12.z;
            _vec_load_12[0 + 3] = _uv4_12.w;
        }
        words[woff_1] = _vec_load_12[0];
        words[woff_1 + 1] = _vec_load_12[1];
        words[woff_1 + 2] = _vec_load_12[2];
        words[woff_1 + 3] = _vec_load_12[3];
        unsigned int _vec_load_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_13[0 + 0] = _uv4_13.x;
            _vec_load_13[0 + 1] = _uv4_13.y;
            _vec_load_13[0 + 2] = _uv4_13.z;
            _vec_load_13[0 + 3] = _uv4_13.w;
        }
        words[14 + woff_1] = _vec_load_13[0];
        words[14 + woff_1 + 1] = _vec_load_13[1];
        words[14 + woff_1 + 2] = _vec_load_13[2];
        words[14 + woff_1 + 3] = _vec_load_13[3];
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        words[28 + woff_1] = _vec_load_14[0];
        words[28 + woff_1 + 1] = _vec_load_14[1];
        words[28 + woff_1 + 2] = _vec_load_14[2];
        words[28 + woff_1 + 3] = _vec_load_14[3];
        unsigned int _vec_load_15[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_15[0 + 0] = _uv4_15.x;
            _vec_load_15[0 + 1] = _uv4_15.y;
            _vec_load_15[0 + 2] = _uv4_15.z;
            _vec_load_15[0 + 3] = _uv4_15.w;
        }
        words[42 + woff_1] = _vec_load_15[0];
        words[42 + woff_1 + 1] = _vec_load_15[1];
        words[42 + woff_1 + 2] = _vec_load_15[2];
        words[42 + woff_1 + 3] = _vec_load_15[3];
        unsigned int _vec_load_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_16[0 + 0] = _uv4_16.x;
            _vec_load_16[0 + 1] = _uv4_16.y;
            _vec_load_16[0 + 2] = _uv4_16.z;
            _vec_load_16[0 + 3] = _uv4_16.w;
        }
        words[56 + woff_1] = _vec_load_16[0];
        words[56 + woff_1 + 1] = _vec_load_16[1];
        words[56 + woff_1 + 2] = _vec_load_16[2];
        words[56 + woff_1 + 3] = _vec_load_16[3];
        unsigned int _vec_load_17[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_17[0 + 0] = _uv4_17.x;
            _vec_load_17[0 + 1] = _uv4_17.y;
            _vec_load_17[0 + 2] = _uv4_17.z;
            _vec_load_17[0 + 3] = _uv4_17.w;
        }
        words[70 + woff_1] = _vec_load_17[0];
        words[70 + woff_1 + 1] = _vec_load_17[1];
        words[70 + woff_1 + 2] = _vec_load_17[2];
        words[70 + woff_1 + 3] = _vec_load_17[3];
        unsigned int _vec_load_18[4];
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_18[0 + 0] = _uv4_18.x;
            _vec_load_18[0 + 1] = _uv4_18.y;
            _vec_load_18[0 + 2] = _uv4_18.z;
            _vec_load_18[0 + 3] = _uv4_18.w;
        }
        words[84 + woff_1] = _vec_load_18[0];
        words[84 + woff_1 + 1] = _vec_load_18[1];
        words[84 + woff_1 + 2] = _vec_load_18[2];
        words[84 + woff_1 + 3] = _vec_load_18[3];
        unsigned int _vec_load_19[4];
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_19[0 + 0] = _uv4_19.x;
            _vec_load_19[0 + 1] = _uv4_19.y;
            _vec_load_19[0 + 2] = _uv4_19.z;
            _vec_load_19[0 + 3] = _uv4_19.w;
        }
        words[98 + woff_1] = _vec_load_19[0];
        words[98 + woff_1 + 1] = _vec_load_19[1];
        words[98 + woff_1 + 2] = _vec_load_19[2];
        words[98 + woff_1 + 3] = _vec_load_19[3];
        unsigned int _vec_load_20[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_20[0 + 0] = _uv4_20.x;
            _vec_load_20[0 + 1] = _uv4_20.y;
            _vec_load_20[0 + 2] = _uv4_20.z;
            _vec_load_20[0 + 3] = _uv4_20.w;
        }
        dwords[woff_1] = _vec_load_20[0];
        dwords[woff_1 + 1] = _vec_load_20[1];
        dwords[woff_1 + 2] = _vec_load_20[2];
        dwords[woff_1 + 3] = _vec_load_20[3];
        float _vec_load_21[8];
        {
            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_21[_pair]));
                }
            }
        }
        float _vec_load_22[8];
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(qk_weight + base_0);
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
                        : "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_22[_pair]));
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
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_23[_pair]));
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
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_24[4];
        {
            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_24[0 + 0] = _uv4_24.x;
            _vec_load_24[0 + 1] = _uv4_24.y;
            _vec_load_24[0 + 2] = _uv4_24.z;
            _vec_load_24[0 + 3] = _uv4_24.w;
        }
        words[woff_3] = _vec_load_24[0];
        words[woff_3 + 1] = _vec_load_24[1];
        words[woff_3 + 2] = _vec_load_24[2];
        words[woff_3 + 3] = _vec_load_24[3];
        unsigned int _vec_load_25[4];
        {
            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_25[0 + 0] = _uv4_25.x;
            _vec_load_25[0 + 1] = _uv4_25.y;
            _vec_load_25[0 + 2] = _uv4_25.z;
            _vec_load_25[0 + 3] = _uv4_25.w;
        }
        words[14 + woff_3] = _vec_load_25[0];
        words[14 + woff_3 + 1] = _vec_load_25[1];
        words[14 + woff_3 + 2] = _vec_load_25[2];
        words[14 + woff_3 + 3] = _vec_load_25[3];
        unsigned int _vec_load_26[4];
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_26[0 + 0] = _uv4_26.x;
            _vec_load_26[0 + 1] = _uv4_26.y;
            _vec_load_26[0 + 2] = _uv4_26.z;
            _vec_load_26[0 + 3] = _uv4_26.w;
        }
        words[28 + woff_3] = _vec_load_26[0];
        words[28 + woff_3 + 1] = _vec_load_26[1];
        words[28 + woff_3 + 2] = _vec_load_26[2];
        words[28 + woff_3 + 3] = _vec_load_26[3];
        unsigned int _vec_load_27[4];
        {
            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_27[0 + 0] = _uv4_27.x;
            _vec_load_27[0 + 1] = _uv4_27.y;
            _vec_load_27[0 + 2] = _uv4_27.z;
            _vec_load_27[0 + 3] = _uv4_27.w;
        }
        words[42 + woff_3] = _vec_load_27[0];
        words[42 + woff_3 + 1] = _vec_load_27[1];
        words[42 + woff_3 + 2] = _vec_load_27[2];
        words[42 + woff_3 + 3] = _vec_load_27[3];
        unsigned int _vec_load_28[4];
        {
            uint4 _uv4_28 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_28[0 + 0] = _uv4_28.x;
            _vec_load_28[0 + 1] = _uv4_28.y;
            _vec_load_28[0 + 2] = _uv4_28.z;
            _vec_load_28[0 + 3] = _uv4_28.w;
        }
        words[56 + woff_3] = _vec_load_28[0];
        words[56 + woff_3 + 1] = _vec_load_28[1];
        words[56 + woff_3 + 2] = _vec_load_28[2];
        words[56 + woff_3 + 3] = _vec_load_28[3];
        unsigned int _vec_load_29[4];
        {
            uint4 _uv4_29 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_29[0 + 0] = _uv4_29.x;
            _vec_load_29[0 + 1] = _uv4_29.y;
            _vec_load_29[0 + 2] = _uv4_29.z;
            _vec_load_29[0 + 3] = _uv4_29.w;
        }
        words[70 + woff_3] = _vec_load_29[0];
        words[70 + woff_3 + 1] = _vec_load_29[1];
        words[70 + woff_3 + 2] = _vec_load_29[2];
        words[70 + woff_3 + 3] = _vec_load_29[3];
        unsigned int _vec_load_30[4];
        {
            uint4 _uv4_30 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_30[0 + 0] = _uv4_30.x;
            _vec_load_30[0 + 1] = _uv4_30.y;
            _vec_load_30[0 + 2] = _uv4_30.z;
            _vec_load_30[0 + 3] = _uv4_30.w;
        }
        words[84 + woff_3] = _vec_load_30[0];
        words[84 + woff_3 + 1] = _vec_load_30[1];
        words[84 + woff_3 + 2] = _vec_load_30[2];
        words[84 + woff_3 + 3] = _vec_load_30[3];
        unsigned int _vec_load_31[4];
        {
            uint4 _uv4_31 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_31[0 + 0] = _uv4_31.x;
            _vec_load_31[0 + 1] = _uv4_31.y;
            _vec_load_31[0 + 2] = _uv4_31.z;
            _vec_load_31[0 + 3] = _uv4_31.w;
        }
        words[98 + woff_3] = _vec_load_31[0];
        words[98 + woff_3 + 1] = _vec_load_31[1];
        words[98 + woff_3 + 2] = _vec_load_31[2];
        words[98 + woff_3 + 3] = _vec_load_31[3];
        unsigned int _vec_load_32[4];
        {
            uint4 _uv4_32 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_32[0 + 0] = _uv4_32.x;
            _vec_load_32[0 + 1] = _uv4_32.y;
            _vec_load_32[0 + 2] = _uv4_32.z;
            _vec_load_32[0 + 3] = _uv4_32.w;
        }
        dwords[woff_3] = _vec_load_32[0];
        dwords[woff_3 + 1] = _vec_load_32[1];
        dwords[woff_3 + 2] = _vec_load_32[2];
        dwords[woff_3 + 3] = _vec_load_32[3];
        float _vec_load_33[8];
        {
            const uint4* _vptr_33 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_33[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_33[_blk] = _vptr_33[_blk];
                uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_33[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_33[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_33[_pair]));
                }
            }
        }
        float _vec_load_34[8];
        {
            const uint4* _vptr_34 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_34[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_34[_blk] = _vptr_34[_blk];
                uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_34[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_34[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_34[_pair]));
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
            const uint4* _vptr_35 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_35[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_35[_blk] = _vptr_35[_blk];
                uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_35[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_35[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_35[_pair]));
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
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_36[1];
        {
            _vec_load_36[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_36[0];
        unsigned int _vec_load_37[1];
        {
            _vec_load_37[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_37[0];
        unsigned int _vec_load_38[1];
        {
            _vec_load_38[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_38[0];
        unsigned int _vec_load_39[1];
        {
            _vec_load_39[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_39[0];
        unsigned int _vec_load_40[1];
        {
            _vec_load_40[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_40[0];
        unsigned int _vec_load_41[1];
        {
            _vec_load_41[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_41[0];
        unsigned int _vec_load_42[1];
        {
            _vec_load_42[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[42 + woff_5] = _vec_load_42[0];
        unsigned int _vec_load_43[1];
        {
            _vec_load_43[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[42 + woff_5 + 1] = _vec_load_43[0];
        unsigned int _vec_load_44[1];
        {
            _vec_load_44[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[56 + woff_5] = _vec_load_44[0];
        unsigned int _vec_load_45[1];
        {
            _vec_load_45[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[56 + woff_5 + 1] = _vec_load_45[0];
        unsigned int _vec_load_46[1];
        {
            _vec_load_46[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[70 + woff_5] = _vec_load_46[0];
        unsigned int _vec_load_47[1];
        {
            _vec_load_47[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[70 + woff_5 + 1] = _vec_load_47[0];
        unsigned int _vec_load_48[1];
        {
            _vec_load_48[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[84 + woff_5] = _vec_load_48[0];
        unsigned int _vec_load_49[1];
        {
            _vec_load_49[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[84 + woff_5 + 1] = _vec_load_49[0];
        unsigned int _vec_load_50[1];
        {
            _vec_load_50[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[98 + woff_5] = _vec_load_50[0];
        unsigned int _vec_load_51[1];
        {
            _vec_load_51[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[98 + woff_5 + 1] = _vec_load_51[0];
        unsigned int _vec_load_52[1];
        {
            _vec_load_52[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_52[0];
        unsigned int _vec_load_53[1];
        {
            _vec_load_53[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_53[0];
        float _vec_load_54[4];
        {
            uint2 _vld_36;
            _vld_36 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_54[0 + _pair * 2])[0]), "=f"((&_vec_load_54[0 + _pair * 2])[1])
                    : "r"(_vpairs_36[_pair]));
            }
        }
        float _vec_load_55[4];
        {
            uint2 _vld_37;
            _vld_37 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_55[0 + _pair * 2])[0]), "=f"((&_vec_load_55[0 + _pair * 2])[1])
                    : "r"(_vpairs_37[_pair]));
            }
        }
        q[24] = _vec_load_54[0] * _vec_load_55[0];
        q[25] = _vec_load_54[1] * _vec_load_55[1];
        q[26] = _vec_load_54[2] * _vec_load_55[2];
        q[27] = _vec_load_54[3] * _vec_load_55[3];
        float _vec_load_56[4];
        {
            uint2 _vld_38;
            _vld_38 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_56[0 + _pair * 2])[0]), "=f"((&_vec_load_56[0 + _pair * 2])[1])
                    : "r"(_vpairs_38[_pair]));
            }
        }
        wout[24] = _vec_load_56[0];
        wout[25] = _vec_load_56[1];
        wout[26] = _vec_load_56[2];
        wout[27] = _vec_load_56[3];
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
        float2 sq[8];
        float2 dot[8];
        float fsrc[84];
        sq[0] = make_float2(0.0f, 0.0f);
        dot[0] = make_float2(0.0f, 0.0f);
        sq[1] = make_float2(0.0f, 0.0f);
        dot[1] = make_float2(0.0f, 0.0f);
        sq[2] = make_float2(0.0f, 0.0f);
        dot[2] = make_float2(0.0f, 0.0f);
        sq[3] = make_float2(0.0f, 0.0f);
        dot[3] = make_float2(0.0f, 0.0f);
        sq[4] = make_float2(0.0f, 0.0f);
        dot[4] = make_float2(0.0f, 0.0f);
        sq[5] = make_float2(0.0f, 0.0f);
        dot[5] = make_float2(0.0f, 0.0f);
        sq[6] = make_float2(0.0f, 0.0f);
        dot[6] = make_float2(0.0f, 0.0f);
        sq[7] = make_float2(0.0f, 0.0f);
        dot[7] = make_float2(0.0f, 0.0f);
        int base_6 = group * 1024 + thread * 8;
        const int woff_7 = 0;
        unsigned int sw[4];
        sw[0] = words[woff_7];
        sw[1] = words[woff_7 + 1];
        sw[2] = words[woff_7 + 2];
        sw[3] = words[woff_7 + 3];
        float _unpack_bf16_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_0[_pair * 2])[0]), "=f"((&_unpack_bf16_0[_pair * 2])[1])
                : "r"(sw[_pair]));
        }
        fsrc[0] = _unpack_bf16_0[0];
        fsrc[1] = _unpack_bf16_0[1];
        fsrc[2] = _unpack_bf16_0[2];
        fsrc[3] = _unpack_bf16_0[3];
        fsrc[4] = _unpack_bf16_0[4];
        fsrc[5] = _unpack_bf16_0[5];
        fsrc[6] = _unpack_bf16_0[6];
        fsrc[7] = _unpack_bf16_0[7];
        float2 v = make_float2(_unpack_bf16_0[0], _unpack_bf16_0[1]);
        float2 qp = make_float2(q[0], q[1]);
        sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
        float2 v_8 = make_float2(_unpack_bf16_0[2], _unpack_bf16_0[3]);
        float2 qp_9 = make_float2(q[2], q[3]);
        sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_8, qp_9, dot[0]);
        float2 v_10 = make_float2(_unpack_bf16_0[4], _unpack_bf16_0[5]);
        float2 qp_11 = make_float2(q[4], q[5]);
        sq[0] = fma_f32x2_rn_noftz(v_10, v_10, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_10, qp_11, dot[0]);
        float2 v_12 = make_float2(_unpack_bf16_0[6], _unpack_bf16_0[7]);
        float2 qp_13 = make_float2(q[6], q[7]);
        sq[0] = fma_f32x2_rn_noftz(v_12, v_12, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_12, qp_13, dot[0]);
        unsigned int sw_14[4];
        sw_14[0] = words[14 + woff_7];
        sw_14[1] = words[14 + woff_7 + 1];
        sw_14[2] = words[14 + woff_7 + 2];
        sw_14[3] = words[14 + woff_7 + 3];
        float _unpack_bf16_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
                : "r"(sw_14[_pair]));
        }
        fsrc[28] = _unpack_bf16_1[0];
        fsrc[29] = _unpack_bf16_1[1];
        fsrc[30] = _unpack_bf16_1[2];
        fsrc[31] = _unpack_bf16_1[3];
        fsrc[32] = _unpack_bf16_1[4];
        fsrc[33] = _unpack_bf16_1[5];
        fsrc[34] = _unpack_bf16_1[6];
        fsrc[35] = _unpack_bf16_1[7];
        float2 v_15 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
        float2 qp_16 = make_float2(q[0], q[1]);
        sq[1] = fma_f32x2_rn_noftz(v_15, v_15, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_15, qp_16, dot[1]);
        float2 v_17 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
        float2 qp_18 = make_float2(q[2], q[3]);
        sq[1] = fma_f32x2_rn_noftz(v_17, v_17, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_17, qp_18, dot[1]);
        float2 v_19 = make_float2(_unpack_bf16_1[4], _unpack_bf16_1[5]);
        float2 qp_20 = make_float2(q[4], q[5]);
        sq[1] = fma_f32x2_rn_noftz(v_19, v_19, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_19, qp_20, dot[1]);
        float2 v_21 = make_float2(_unpack_bf16_1[6], _unpack_bf16_1[7]);
        float2 qp_22 = make_float2(q[6], q[7]);
        sq[1] = fma_f32x2_rn_noftz(v_21, v_21, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_21, qp_22, dot[1]);
        unsigned int sw_23[4];
        sw_23[0] = words[28 + woff_7];
        sw_23[1] = words[28 + woff_7 + 1];
        sw_23[2] = words[28 + woff_7 + 2];
        sw_23[3] = words[28 + woff_7 + 3];
        float _unpack_bf16_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                : "r"(sw_23[_pair]));
        }
        fsrc[56] = _unpack_bf16_2[0];
        fsrc[57] = _unpack_bf16_2[1];
        fsrc[58] = _unpack_bf16_2[2];
        fsrc[59] = _unpack_bf16_2[3];
        fsrc[60] = _unpack_bf16_2[4];
        fsrc[61] = _unpack_bf16_2[5];
        fsrc[62] = _unpack_bf16_2[6];
        fsrc[63] = _unpack_bf16_2[7];
        float2 v_24 = make_float2(_unpack_bf16_2[0], _unpack_bf16_2[1]);
        float2 qp_25 = make_float2(q[0], q[1]);
        sq[2] = fma_f32x2_rn_noftz(v_24, v_24, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_24, qp_25, dot[2]);
        float2 v_26 = make_float2(_unpack_bf16_2[2], _unpack_bf16_2[3]);
        float2 qp_27 = make_float2(q[2], q[3]);
        sq[2] = fma_f32x2_rn_noftz(v_26, v_26, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_26, qp_27, dot[2]);
        float2 v_28 = make_float2(_unpack_bf16_2[4], _unpack_bf16_2[5]);
        float2 qp_29 = make_float2(q[4], q[5]);
        sq[2] = fma_f32x2_rn_noftz(v_28, v_28, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_28, qp_29, dot[2]);
        float2 v_30 = make_float2(_unpack_bf16_2[6], _unpack_bf16_2[7]);
        float2 qp_31 = make_float2(q[6], q[7]);
        sq[2] = fma_f32x2_rn_noftz(v_30, v_30, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_30, qp_31, dot[2]);
        unsigned int sw_32[4];
        sw_32[0] = words[42 + woff_7];
        sw_32[1] = words[42 + woff_7 + 1];
        sw_32[2] = words[42 + woff_7 + 2];
        sw_32[3] = words[42 + woff_7 + 3];
        float _unpack_bf16_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                : "r"(sw_32[_pair]));
        }
        float2 v_33 = make_float2(_unpack_bf16_3[0], _unpack_bf16_3[1]);
        float2 qp_34 = make_float2(q[0], q[1]);
        sq[3] = fma_f32x2_rn_noftz(v_33, v_33, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_33, qp_34, dot[3]);
        float2 v_35 = make_float2(_unpack_bf16_3[2], _unpack_bf16_3[3]);
        float2 qp_36 = make_float2(q[2], q[3]);
        sq[3] = fma_f32x2_rn_noftz(v_35, v_35, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_35, qp_36, dot[3]);
        float2 v_37 = make_float2(_unpack_bf16_3[4], _unpack_bf16_3[5]);
        float2 qp_38 = make_float2(q[4], q[5]);
        sq[3] = fma_f32x2_rn_noftz(v_37, v_37, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_37, qp_38, dot[3]);
        float2 v_39 = make_float2(_unpack_bf16_3[6], _unpack_bf16_3[7]);
        float2 qp_40 = make_float2(q[6], q[7]);
        sq[3] = fma_f32x2_rn_noftz(v_39, v_39, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_39, qp_40, dot[3]);
        unsigned int sw_41[4];
        sw_41[0] = words[56 + woff_7];
        sw_41[1] = words[56 + woff_7 + 1];
        sw_41[2] = words[56 + woff_7 + 2];
        sw_41[3] = words[56 + woff_7 + 3];
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                : "r"(sw_41[_pair]));
        }
        float2 v_42 = make_float2(_unpack_bf16_4[0], _unpack_bf16_4[1]);
        float2 qp_43 = make_float2(q[0], q[1]);
        sq[4] = fma_f32x2_rn_noftz(v_42, v_42, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_42, qp_43, dot[4]);
        float2 v_44 = make_float2(_unpack_bf16_4[2], _unpack_bf16_4[3]);
        float2 qp_45 = make_float2(q[2], q[3]);
        sq[4] = fma_f32x2_rn_noftz(v_44, v_44, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_44, qp_45, dot[4]);
        float2 v_46 = make_float2(_unpack_bf16_4[4], _unpack_bf16_4[5]);
        float2 qp_47 = make_float2(q[4], q[5]);
        sq[4] = fma_f32x2_rn_noftz(v_46, v_46, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_46, qp_47, dot[4]);
        float2 v_48 = make_float2(_unpack_bf16_4[6], _unpack_bf16_4[7]);
        float2 qp_49 = make_float2(q[6], q[7]);
        sq[4] = fma_f32x2_rn_noftz(v_48, v_48, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_48, qp_49, dot[4]);
        unsigned int sw_50[4];
        sw_50[0] = words[70 + woff_7];
        sw_50[1] = words[70 + woff_7 + 1];
        sw_50[2] = words[70 + woff_7 + 2];
        sw_50[3] = words[70 + woff_7 + 3];
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                : "r"(sw_50[_pair]));
        }
        float2 v_51 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 qp_52 = make_float2(q[0], q[1]);
        sq[5] = fma_f32x2_rn_noftz(v_51, v_51, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_51, qp_52, dot[5]);
        float2 v_53 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 qp_54 = make_float2(q[2], q[3]);
        sq[5] = fma_f32x2_rn_noftz(v_53, v_53, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_53, qp_54, dot[5]);
        float2 v_55 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 qp_56 = make_float2(q[4], q[5]);
        sq[5] = fma_f32x2_rn_noftz(v_55, v_55, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_55, qp_56, dot[5]);
        float2 v_57 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 qp_58 = make_float2(q[6], q[7]);
        sq[5] = fma_f32x2_rn_noftz(v_57, v_57, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_57, qp_58, dot[5]);
        unsigned int sw_59[4];
        sw_59[0] = words[84 + woff_7];
        sw_59[1] = words[84 + woff_7 + 1];
        sw_59[2] = words[84 + woff_7 + 2];
        sw_59[3] = words[84 + woff_7 + 3];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(sw_59[_pair]));
        }
        float2 v_60 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 qp_61 = make_float2(q[0], q[1]);
        sq[6] = fma_f32x2_rn_noftz(v_60, v_60, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_60, qp_61, dot[6]);
        float2 v_62 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 qp_63 = make_float2(q[2], q[3]);
        sq[6] = fma_f32x2_rn_noftz(v_62, v_62, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_62, qp_63, dot[6]);
        float2 v_64 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 qp_65 = make_float2(q[4], q[5]);
        sq[6] = fma_f32x2_rn_noftz(v_64, v_64, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_64, qp_65, dot[6]);
        float2 v_66 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 qp_67 = make_float2(q[6], q[7]);
        sq[6] = fma_f32x2_rn_noftz(v_66, v_66, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_66, qp_67, dot[6]);
        unsigned int sw_68[4];
        sw_68[0] = words[98 + woff_7];
        sw_68[1] = words[98 + woff_7 + 1];
        sw_68[2] = words[98 + woff_7 + 2];
        sw_68[3] = words[98 + woff_7 + 3];
        __nv_bfloat162 a = __as_bf16x2(sw_68[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_68[0] = __as_u32(mixed);
        words[98 + woff_7] = sw_68[0];
        __nv_bfloat162 a_69 = __as_bf16x2(sw_68[1]);
        __nv_bfloat162 d_70 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_71 = a_69 + d_70;
        sw_68[1] = __as_u32(mixed_71);
        words[98 + woff_7 + 1] = sw_68[1];
        __nv_bfloat162 a_72 = __as_bf16x2(sw_68[2]);
        __nv_bfloat162 d_73 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_74 = a_72 + d_73;
        sw_68[2] = __as_u32(mixed_74);
        words[98 + woff_7 + 2] = sw_68[2];
        __nv_bfloat162 a_75 = __as_bf16x2(sw_68[3]);
        __nv_bfloat162 d_76 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_77 = a_75 + d_76;
        sw_68[3] = __as_u32(mixed_77);
        words[98 + woff_7 + 3] = sw_68[3];
        {
            int4 _iv4 = make_int4(sw_68[0 + 0], sw_68[0 + 1], sw_68[0 + 2], sw_68[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
        }
        unsigned long long write_base = block_base + 7 * blocks_k_stride + (unsigned long long)base_6;
        {
            int4 _iv4 = make_int4(sw_68[0 + 0], sw_68[0 + 1], sw_68[0 + 2], sw_68[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base) + 0) = _iv4;
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
                : "r"(sw_68[_pair]));
        }
        float2 v_78 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 qp_79 = make_float2(q[0], q[1]);
        sq[7] = fma_f32x2_rn_noftz(v_78, v_78, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_78, qp_79, dot[7]);
        float2 v_80 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 qp_81 = make_float2(q[2], q[3]);
        sq[7] = fma_f32x2_rn_noftz(v_80, v_80, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_80, qp_81, dot[7]);
        float2 v_82 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 qp_83 = make_float2(q[4], q[5]);
        sq[7] = fma_f32x2_rn_noftz(v_82, v_82, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_82, qp_83, dot[7]);
        float2 v_84 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 qp_85 = make_float2(q[6], q[7]);
        sq[7] = fma_f32x2_rn_noftz(v_84, v_84, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_84, qp_85, dot[7]);
        int base_86 = (2 + group) * 1024 + thread * 8;
        const int woff_87 = 4;
        unsigned int sw_88[4];
        sw_88[0] = words[woff_87];
        sw_88[1] = words[woff_87 + 1];
        sw_88[2] = words[woff_87 + 2];
        sw_88[3] = words[woff_87 + 3];
        float _unpack_bf16_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                : "r"(sw_88[_pair]));
        }
        fsrc[8] = _unpack_bf16_8[0];
        fsrc[9] = _unpack_bf16_8[1];
        fsrc[10] = _unpack_bf16_8[2];
        fsrc[11] = _unpack_bf16_8[3];
        fsrc[12] = _unpack_bf16_8[4];
        fsrc[13] = _unpack_bf16_8[5];
        fsrc[14] = _unpack_bf16_8[6];
        fsrc[15] = _unpack_bf16_8[7];
        float2 v_89 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 qp_90 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_89, v_89, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_89, qp_90, dot[0]);
        float2 v_91 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 qp_92 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_91, v_91, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_91, qp_92, dot[0]);
        float2 v_93 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 qp_94 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_93, v_93, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_93, qp_94, dot[0]);
        float2 v_95 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 qp_96 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_95, v_95, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_95, qp_96, dot[0]);
        unsigned int sw_97[4];
        sw_97[0] = words[14 + woff_87];
        sw_97[1] = words[14 + woff_87 + 1];
        sw_97[2] = words[14 + woff_87 + 2];
        sw_97[3] = words[14 + woff_87 + 3];
        float _unpack_bf16_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                : "r"(sw_97[_pair]));
        }
        fsrc[36] = _unpack_bf16_9[0];
        fsrc[37] = _unpack_bf16_9[1];
        fsrc[38] = _unpack_bf16_9[2];
        fsrc[39] = _unpack_bf16_9[3];
        fsrc[40] = _unpack_bf16_9[4];
        fsrc[41] = _unpack_bf16_9[5];
        fsrc[42] = _unpack_bf16_9[6];
        fsrc[43] = _unpack_bf16_9[7];
        float2 v_98 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 qp_99 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_98, v_98, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_98, qp_99, dot[1]);
        float2 v_100 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 qp_101 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_100, v_100, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_100, qp_101, dot[1]);
        float2 v_102 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 qp_103 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_102, v_102, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_102, qp_103, dot[1]);
        float2 v_104 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 qp_105 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_104, v_104, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_104, qp_105, dot[1]);
        unsigned int sw_106[4];
        sw_106[0] = words[28 + woff_87];
        sw_106[1] = words[28 + woff_87 + 1];
        sw_106[2] = words[28 + woff_87 + 2];
        sw_106[3] = words[28 + woff_87 + 3];
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                : "r"(sw_106[_pair]));
        }
        fsrc[64] = _unpack_bf16_10[0];
        fsrc[65] = _unpack_bf16_10[1];
        fsrc[66] = _unpack_bf16_10[2];
        fsrc[67] = _unpack_bf16_10[3];
        fsrc[68] = _unpack_bf16_10[4];
        fsrc[69] = _unpack_bf16_10[5];
        fsrc[70] = _unpack_bf16_10[6];
        fsrc[71] = _unpack_bf16_10[7];
        float2 v_107 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 qp_108 = make_float2(q[8], q[9]);
        sq[2] = fma_f32x2_rn_noftz(v_107, v_107, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_107, qp_108, dot[2]);
        float2 v_109 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 qp_110 = make_float2(q[10], q[11]);
        sq[2] = fma_f32x2_rn_noftz(v_109, v_109, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_109, qp_110, dot[2]);
        float2 v_111 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 qp_112 = make_float2(q[12], q[13]);
        sq[2] = fma_f32x2_rn_noftz(v_111, v_111, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_111, qp_112, dot[2]);
        float2 v_113 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 qp_114 = make_float2(q[14], q[15]);
        sq[2] = fma_f32x2_rn_noftz(v_113, v_113, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_113, qp_114, dot[2]);
        unsigned int sw_115[4];
        sw_115[0] = words[42 + woff_87];
        sw_115[1] = words[42 + woff_87 + 1];
        sw_115[2] = words[42 + woff_87 + 2];
        sw_115[3] = words[42 + woff_87 + 3];
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                : "r"(sw_115[_pair]));
        }
        float2 v_116 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 qp_117 = make_float2(q[8], q[9]);
        sq[3] = fma_f32x2_rn_noftz(v_116, v_116, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_116, qp_117, dot[3]);
        float2 v_118 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 qp_119 = make_float2(q[10], q[11]);
        sq[3] = fma_f32x2_rn_noftz(v_118, v_118, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_118, qp_119, dot[3]);
        float2 v_120 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 qp_121 = make_float2(q[12], q[13]);
        sq[3] = fma_f32x2_rn_noftz(v_120, v_120, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_120, qp_121, dot[3]);
        float2 v_122 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 qp_123 = make_float2(q[14], q[15]);
        sq[3] = fma_f32x2_rn_noftz(v_122, v_122, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_122, qp_123, dot[3]);
        unsigned int sw_124[4];
        sw_124[0] = words[56 + woff_87];
        sw_124[1] = words[56 + woff_87 + 1];
        sw_124[2] = words[56 + woff_87 + 2];
        sw_124[3] = words[56 + woff_87 + 3];
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                : "r"(sw_124[_pair]));
        }
        float2 v_125 = make_float2(_unpack_bf16_12[0], _unpack_bf16_12[1]);
        float2 qp_126 = make_float2(q[8], q[9]);
        sq[4] = fma_f32x2_rn_noftz(v_125, v_125, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_125, qp_126, dot[4]);
        float2 v_127 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 qp_128 = make_float2(q[10], q[11]);
        sq[4] = fma_f32x2_rn_noftz(v_127, v_127, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_127, qp_128, dot[4]);
        float2 v_129 = make_float2(_unpack_bf16_12[4], _unpack_bf16_12[5]);
        float2 qp_130 = make_float2(q[12], q[13]);
        sq[4] = fma_f32x2_rn_noftz(v_129, v_129, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_129, qp_130, dot[4]);
        float2 v_131 = make_float2(_unpack_bf16_12[6], _unpack_bf16_12[7]);
        float2 qp_132 = make_float2(q[14], q[15]);
        sq[4] = fma_f32x2_rn_noftz(v_131, v_131, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_131, qp_132, dot[4]);
        unsigned int sw_133[4];
        sw_133[0] = words[70 + woff_87];
        sw_133[1] = words[70 + woff_87 + 1];
        sw_133[2] = words[70 + woff_87 + 2];
        sw_133[3] = words[70 + woff_87 + 3];
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                : "r"(sw_133[_pair]));
        }
        float2 v_134 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 qp_135 = make_float2(q[8], q[9]);
        sq[5] = fma_f32x2_rn_noftz(v_134, v_134, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_134, qp_135, dot[5]);
        float2 v_136 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 qp_137 = make_float2(q[10], q[11]);
        sq[5] = fma_f32x2_rn_noftz(v_136, v_136, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_136, qp_137, dot[5]);
        float2 v_138 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
        float2 qp_139 = make_float2(q[12], q[13]);
        sq[5] = fma_f32x2_rn_noftz(v_138, v_138, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_138, qp_139, dot[5]);
        float2 v_140 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
        float2 qp_141 = make_float2(q[14], q[15]);
        sq[5] = fma_f32x2_rn_noftz(v_140, v_140, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_140, qp_141, dot[5]);
        unsigned int sw_142[4];
        sw_142[0] = words[84 + woff_87];
        sw_142[1] = words[84 + woff_87 + 1];
        sw_142[2] = words[84 + woff_87 + 2];
        sw_142[3] = words[84 + woff_87 + 3];
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                : "r"(sw_142[_pair]));
        }
        float2 v_143 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 qp_144 = make_float2(q[8], q[9]);
        sq[6] = fma_f32x2_rn_noftz(v_143, v_143, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_143, qp_144, dot[6]);
        float2 v_145 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 qp_146 = make_float2(q[10], q[11]);
        sq[6] = fma_f32x2_rn_noftz(v_145, v_145, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_145, qp_146, dot[6]);
        float2 v_147 = make_float2(_unpack_bf16_14[4], _unpack_bf16_14[5]);
        float2 qp_148 = make_float2(q[12], q[13]);
        sq[6] = fma_f32x2_rn_noftz(v_147, v_147, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_147, qp_148, dot[6]);
        float2 v_149 = make_float2(_unpack_bf16_14[6], _unpack_bf16_14[7]);
        float2 qp_150 = make_float2(q[14], q[15]);
        sq[6] = fma_f32x2_rn_noftz(v_149, v_149, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_149, qp_150, dot[6]);
        unsigned int sw_151[4];
        sw_151[0] = words[98 + woff_87];
        sw_151[1] = words[98 + woff_87 + 1];
        sw_151[2] = words[98 + woff_87 + 2];
        sw_151[3] = words[98 + woff_87 + 3];
        __nv_bfloat162 a_152 = __as_bf16x2(sw_151[0]);
        __nv_bfloat162 d_153 = __as_bf16x2(dwords[woff_87]);
        __nv_bfloat162 mixed_154 = a_152 + d_153;
        sw_151[0] = __as_u32(mixed_154);
        words[98 + woff_87] = sw_151[0];
        __nv_bfloat162 a_155 = __as_bf16x2(sw_151[1]);
        __nv_bfloat162 d_156 = __as_bf16x2(dwords[woff_87 + 1]);
        __nv_bfloat162 mixed_157 = a_155 + d_156;
        sw_151[1] = __as_u32(mixed_157);
        words[98 + woff_87 + 1] = sw_151[1];
        __nv_bfloat162 a_158 = __as_bf16x2(sw_151[2]);
        __nv_bfloat162 d_159 = __as_bf16x2(dwords[woff_87 + 2]);
        __nv_bfloat162 mixed_160 = a_158 + d_159;
        sw_151[2] = __as_u32(mixed_160);
        words[98 + woff_87 + 2] = sw_151[2];
        __nv_bfloat162 a_161 = __as_bf16x2(sw_151[3]);
        __nv_bfloat162 d_162 = __as_bf16x2(dwords[woff_87 + 3]);
        __nv_bfloat162 mixed_163 = a_161 + d_162;
        sw_151[3] = __as_u32(mixed_163);
        words[98 + woff_87 + 3] = sw_151[3];
        {
            int4 _iv4 = make_int4(sw_151[0 + 0], sw_151[0 + 1], sw_151[0 + 2], sw_151[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_86)) + 0) = _iv4;
        }
        unsigned long long write_base_164 = block_base + 7 * blocks_k_stride + (unsigned long long)base_86;
        {
            int4 _iv4 = make_int4(sw_151[0 + 0], sw_151[0 + 1], sw_151[0 + 2], sw_151[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_164) + 0) = _iv4;
        }
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                : "r"(sw_151[_pair]));
        }
        float2 v_165 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_166 = make_float2(q[8], q[9]);
        sq[7] = fma_f32x2_rn_noftz(v_165, v_165, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_165, qp_166, dot[7]);
        float2 v_167 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_168 = make_float2(q[10], q[11]);
        sq[7] = fma_f32x2_rn_noftz(v_167, v_167, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_167, qp_168, dot[7]);
        float2 v_169 = make_float2(_unpack_bf16_15[4], _unpack_bf16_15[5]);
        float2 qp_170 = make_float2(q[12], q[13]);
        sq[7] = fma_f32x2_rn_noftz(v_169, v_169, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_169, qp_170, dot[7]);
        float2 v_171 = make_float2(_unpack_bf16_15[6], _unpack_bf16_15[7]);
        float2 qp_172 = make_float2(q[14], q[15]);
        sq[7] = fma_f32x2_rn_noftz(v_171, v_171, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_171, qp_172, dot[7]);
        int base_173 = (4 + group) * 1024 + thread * 8;
        const int woff_174 = 8;
        unsigned int sw_175[4];
        sw_175[0] = words[woff_174];
        sw_175[1] = words[woff_174 + 1];
        sw_175[2] = words[woff_174 + 2];
        sw_175[3] = words[woff_174 + 3];
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(sw_175[_pair]));
        }
        fsrc[16] = _unpack_bf16_16[0];
        fsrc[17] = _unpack_bf16_16[1];
        fsrc[18] = _unpack_bf16_16[2];
        fsrc[19] = _unpack_bf16_16[3];
        fsrc[20] = _unpack_bf16_16[4];
        fsrc[21] = _unpack_bf16_16[5];
        fsrc[22] = _unpack_bf16_16[6];
        fsrc[23] = _unpack_bf16_16[7];
        float2 v_176 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_177 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_176, v_176, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_176, qp_177, dot[0]);
        float2 v_178 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_179 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_178, v_178, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_178, qp_179, dot[0]);
        float2 v_180 = make_float2(_unpack_bf16_16[4], _unpack_bf16_16[5]);
        float2 qp_181 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_180, v_180, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_180, qp_181, dot[0]);
        float2 v_182 = make_float2(_unpack_bf16_16[6], _unpack_bf16_16[7]);
        float2 qp_183 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_182, v_182, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_182, qp_183, dot[0]);
        unsigned int sw_184[4];
        sw_184[0] = words[14 + woff_174];
        sw_184[1] = words[14 + woff_174 + 1];
        sw_184[2] = words[14 + woff_174 + 2];
        sw_184[3] = words[14 + woff_174 + 3];
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(sw_184[_pair]));
        }
        fsrc[44] = _unpack_bf16_17[0];
        fsrc[45] = _unpack_bf16_17[1];
        fsrc[46] = _unpack_bf16_17[2];
        fsrc[47] = _unpack_bf16_17[3];
        fsrc[48] = _unpack_bf16_17[4];
        fsrc[49] = _unpack_bf16_17[5];
        fsrc[50] = _unpack_bf16_17[6];
        fsrc[51] = _unpack_bf16_17[7];
        float2 v_185 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_186 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_185, v_185, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_185, qp_186, dot[1]);
        float2 v_187 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_188 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_187, v_187, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_187, qp_188, dot[1]);
        float2 v_189 = make_float2(_unpack_bf16_17[4], _unpack_bf16_17[5]);
        float2 qp_190 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_189, v_189, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_189, qp_190, dot[1]);
        float2 v_191 = make_float2(_unpack_bf16_17[6], _unpack_bf16_17[7]);
        float2 qp_192 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_191, v_191, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_191, qp_192, dot[1]);
        unsigned int sw_193[4];
        sw_193[0] = words[28 + woff_174];
        sw_193[1] = words[28 + woff_174 + 1];
        sw_193[2] = words[28 + woff_174 + 2];
        sw_193[3] = words[28 + woff_174 + 3];
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                : "r"(sw_193[_pair]));
        }
        fsrc[72] = _unpack_bf16_18[0];
        fsrc[73] = _unpack_bf16_18[1];
        fsrc[74] = _unpack_bf16_18[2];
        fsrc[75] = _unpack_bf16_18[3];
        fsrc[76] = _unpack_bf16_18[4];
        fsrc[77] = _unpack_bf16_18[5];
        fsrc[78] = _unpack_bf16_18[6];
        fsrc[79] = _unpack_bf16_18[7];
        float2 v_194 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_195 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_194, v_194, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_194, qp_195, dot[2]);
        float2 v_196 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_197 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_196, v_196, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_196, qp_197, dot[2]);
        float2 v_198 = make_float2(_unpack_bf16_18[4], _unpack_bf16_18[5]);
        float2 qp_199 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_198, v_198, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_198, qp_199, dot[2]);
        float2 v_200 = make_float2(_unpack_bf16_18[6], _unpack_bf16_18[7]);
        float2 qp_201 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_200, v_200, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_200, qp_201, dot[2]);
        unsigned int sw_202[4];
        sw_202[0] = words[42 + woff_174];
        sw_202[1] = words[42 + woff_174 + 1];
        sw_202[2] = words[42 + woff_174 + 2];
        sw_202[3] = words[42 + woff_174 + 3];
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                : "r"(sw_202[_pair]));
        }
        float2 v_203 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 qp_204 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_203, v_203, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_203, qp_204, dot[3]);
        float2 v_205 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_206 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_205, v_205, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_205, qp_206, dot[3]);
        float2 v_207 = make_float2(_unpack_bf16_19[4], _unpack_bf16_19[5]);
        float2 qp_208 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_207, v_207, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_207, qp_208, dot[3]);
        float2 v_209 = make_float2(_unpack_bf16_19[6], _unpack_bf16_19[7]);
        float2 qp_210 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_209, v_209, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_209, qp_210, dot[3]);
        unsigned int sw_211[4];
        sw_211[0] = words[56 + woff_174];
        sw_211[1] = words[56 + woff_174 + 1];
        sw_211[2] = words[56 + woff_174 + 2];
        sw_211[3] = words[56 + woff_174 + 3];
        float _unpack_bf16_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                : "r"(sw_211[_pair]));
        }
        float2 v_212 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        float2 qp_213 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_212, v_212, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_212, qp_213, dot[4]);
        float2 v_214 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        float2 qp_215 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_214, v_214, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_214, qp_215, dot[4]);
        float2 v_216 = make_float2(_unpack_bf16_20[4], _unpack_bf16_20[5]);
        float2 qp_217 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_216, v_216, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_216, qp_217, dot[4]);
        float2 v_218 = make_float2(_unpack_bf16_20[6], _unpack_bf16_20[7]);
        float2 qp_219 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_218, v_218, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_218, qp_219, dot[4]);
        unsigned int sw_220[4];
        sw_220[0] = words[70 + woff_174];
        sw_220[1] = words[70 + woff_174 + 1];
        sw_220[2] = words[70 + woff_174 + 2];
        sw_220[3] = words[70 + woff_174 + 3];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw_220[_pair]));
        }
        float2 v_221 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        float2 qp_222 = make_float2(q[16], q[17]);
        sq[5] = fma_f32x2_rn_noftz(v_221, v_221, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_221, qp_222, dot[5]);
        float2 v_223 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        float2 qp_224 = make_float2(q[18], q[19]);
        sq[5] = fma_f32x2_rn_noftz(v_223, v_223, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_223, qp_224, dot[5]);
        float2 v_225 = make_float2(_unpack_bf16_21[4], _unpack_bf16_21[5]);
        float2 qp_226 = make_float2(q[20], q[21]);
        sq[5] = fma_f32x2_rn_noftz(v_225, v_225, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_225, qp_226, dot[5]);
        float2 v_227 = make_float2(_unpack_bf16_21[6], _unpack_bf16_21[7]);
        float2 qp_228 = make_float2(q[22], q[23]);
        sq[5] = fma_f32x2_rn_noftz(v_227, v_227, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_227, qp_228, dot[5]);
        unsigned int sw_229[4];
        sw_229[0] = words[84 + woff_174];
        sw_229[1] = words[84 + woff_174 + 1];
        sw_229[2] = words[84 + woff_174 + 2];
        sw_229[3] = words[84 + woff_174 + 3];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw_229[_pair]));
        }
        float2 v_230 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        float2 qp_231 = make_float2(q[16], q[17]);
        sq[6] = fma_f32x2_rn_noftz(v_230, v_230, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_230, qp_231, dot[6]);
        float2 v_232 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        float2 qp_233 = make_float2(q[18], q[19]);
        sq[6] = fma_f32x2_rn_noftz(v_232, v_232, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_232, qp_233, dot[6]);
        float2 v_234 = make_float2(_unpack_bf16_22[4], _unpack_bf16_22[5]);
        float2 qp_235 = make_float2(q[20], q[21]);
        sq[6] = fma_f32x2_rn_noftz(v_234, v_234, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_234, qp_235, dot[6]);
        float2 v_236 = make_float2(_unpack_bf16_22[6], _unpack_bf16_22[7]);
        float2 qp_237 = make_float2(q[22], q[23]);
        sq[6] = fma_f32x2_rn_noftz(v_236, v_236, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_236, qp_237, dot[6]);
        unsigned int sw_238[4];
        sw_238[0] = words[98 + woff_174];
        sw_238[1] = words[98 + woff_174 + 1];
        sw_238[2] = words[98 + woff_174 + 2];
        sw_238[3] = words[98 + woff_174 + 3];
        __nv_bfloat162 a_239 = __as_bf16x2(sw_238[0]);
        __nv_bfloat162 d_240 = __as_bf16x2(dwords[woff_174]);
        __nv_bfloat162 mixed_241 = a_239 + d_240;
        sw_238[0] = __as_u32(mixed_241);
        words[98 + woff_174] = sw_238[0];
        __nv_bfloat162 a_242 = __as_bf16x2(sw_238[1]);
        __nv_bfloat162 d_243 = __as_bf16x2(dwords[woff_174 + 1]);
        __nv_bfloat162 mixed_244 = a_242 + d_243;
        sw_238[1] = __as_u32(mixed_244);
        words[98 + woff_174 + 1] = sw_238[1];
        __nv_bfloat162 a_245 = __as_bf16x2(sw_238[2]);
        __nv_bfloat162 d_246 = __as_bf16x2(dwords[woff_174 + 2]);
        __nv_bfloat162 mixed_247 = a_245 + d_246;
        sw_238[2] = __as_u32(mixed_247);
        words[98 + woff_174 + 2] = sw_238[2];
        __nv_bfloat162 a_248 = __as_bf16x2(sw_238[3]);
        __nv_bfloat162 d_249 = __as_bf16x2(dwords[woff_174 + 3]);
        __nv_bfloat162 mixed_250 = a_248 + d_249;
        sw_238[3] = __as_u32(mixed_250);
        words[98 + woff_174 + 3] = sw_238[3];
        {
            int4 _iv4 = make_int4(sw_238[0 + 0], sw_238[0 + 1], sw_238[0 + 2], sw_238[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_173)) + 0) = _iv4;
        }
        unsigned long long write_base_251 = block_base + 7 * blocks_k_stride + (unsigned long long)base_173;
        {
            int4 _iv4 = make_int4(sw_238[0 + 0], sw_238[0 + 1], sw_238[0 + 2], sw_238[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_251) + 0) = _iv4;
        }
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                : "r"(sw_238[_pair]));
        }
        float2 v_252 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        float2 qp_253 = make_float2(q[16], q[17]);
        sq[7] = fma_f32x2_rn_noftz(v_252, v_252, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_252, qp_253, dot[7]);
        float2 v_254 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        float2 qp_255 = make_float2(q[18], q[19]);
        sq[7] = fma_f32x2_rn_noftz(v_254, v_254, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_254, qp_255, dot[7]);
        float2 v_256 = make_float2(_unpack_bf16_23[4], _unpack_bf16_23[5]);
        float2 qp_257 = make_float2(q[20], q[21]);
        sq[7] = fma_f32x2_rn_noftz(v_256, v_256, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_256, qp_257, dot[7]);
        float2 v_258 = make_float2(_unpack_bf16_23[6], _unpack_bf16_23[7]);
        float2 qp_259 = make_float2(q[22], q[23]);
        sq[7] = fma_f32x2_rn_noftz(v_258, v_258, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_258, qp_259, dot[7]);
        int base_260 = 6144 + group * 512 + thread * 4;
        const int woff_261 = 12;
        unsigned int sw_262[4];
        sw_262[0] = words[woff_261];
        sw_262[1] = words[woff_261 + 1];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw_262[_pair]));
        }
        fsrc[24] = _unpack_bf16_24[0];
        fsrc[25] = _unpack_bf16_24[1];
        fsrc[26] = _unpack_bf16_24[2];
        fsrc[27] = _unpack_bf16_24[3];
        float2 v_263 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        sq[0] = fma_f32x2_rn_noftz(v_263, v_263, sq[0]);
        float2 v_264 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        sq[0] = fma_f32x2_rn_noftz(v_264, v_264, sq[0]);
        float2 v_265 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        float2 qp_266 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_265, qp_266, dot[0]);
        float2 v_267 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        float2 qp_268 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_267, qp_268, dot[0]);
        unsigned int sw_269[4];
        sw_269[0] = words[14 + woff_261];
        sw_269[1] = words[14 + woff_261 + 1];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw_269[_pair]));
        }
        fsrc[52] = _unpack_bf16_25[0];
        fsrc[53] = _unpack_bf16_25[1];
        fsrc[54] = _unpack_bf16_25[2];
        fsrc[55] = _unpack_bf16_25[3];
        float2 v_270 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        sq[1] = fma_f32x2_rn_noftz(v_270, v_270, sq[1]);
        float2 v_271 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        sq[1] = fma_f32x2_rn_noftz(v_271, v_271, sq[1]);
        float2 v_272 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        float2 qp_273 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_272, qp_273, dot[1]);
        float2 v_274 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        float2 qp_275 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_274, qp_275, dot[1]);
        unsigned int sw_276[4];
        sw_276[0] = words[28 + woff_261];
        sw_276[1] = words[28 + woff_261 + 1];
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw_276[_pair]));
        }
        fsrc[80] = _unpack_bf16_26[0];
        fsrc[81] = _unpack_bf16_26[1];
        fsrc[82] = _unpack_bf16_26[2];
        fsrc[83] = _unpack_bf16_26[3];
        float2 v_277 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        sq[2] = fma_f32x2_rn_noftz(v_277, v_277, sq[2]);
        float2 v_278 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        sq[2] = fma_f32x2_rn_noftz(v_278, v_278, sq[2]);
        float2 v_279 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        float2 qp_280 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_279, qp_280, dot[2]);
        float2 v_281 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        float2 qp_282 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_281, qp_282, dot[2]);
        unsigned int sw_283[4];
        sw_283[0] = words[42 + woff_261];
        sw_283[1] = words[42 + woff_261 + 1];
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw_283[_pair]));
        }
        float2 v_284 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        sq[3] = fma_f32x2_rn_noftz(v_284, v_284, sq[3]);
        float2 v_285 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        sq[3] = fma_f32x2_rn_noftz(v_285, v_285, sq[3]);
        float2 v_286 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        float2 qp_287 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_286, qp_287, dot[3]);
        float2 v_288 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        float2 qp_289 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_288, qp_289, dot[3]);
        unsigned int sw_290[4];
        sw_290[0] = words[56 + woff_261];
        sw_290[1] = words[56 + woff_261 + 1];
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                : "r"(sw_290[_pair]));
        }
        float2 v_291 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        sq[4] = fma_f32x2_rn_noftz(v_291, v_291, sq[4]);
        float2 v_292 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        sq[4] = fma_f32x2_rn_noftz(v_292, v_292, sq[4]);
        float2 v_293 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        float2 qp_294 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_293, qp_294, dot[4]);
        float2 v_295 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        float2 qp_296 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_295, qp_296, dot[4]);
        unsigned int sw_297[4];
        sw_297[0] = words[70 + woff_261];
        sw_297[1] = words[70 + woff_261 + 1];
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                : "r"(sw_297[_pair]));
        }
        float2 v_298 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        sq[5] = fma_f32x2_rn_noftz(v_298, v_298, sq[5]);
        float2 v_299 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        sq[5] = fma_f32x2_rn_noftz(v_299, v_299, sq[5]);
        float2 v_300 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        float2 qp_301 = make_float2(q[24], q[25]);
        dot[5] = fma_f32x2_rn_noftz(v_300, qp_301, dot[5]);
        float2 v_302 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        float2 qp_303 = make_float2(q[26], q[27]);
        dot[5] = fma_f32x2_rn_noftz(v_302, qp_303, dot[5]);
        unsigned int sw_304[4];
        sw_304[0] = words[84 + woff_261];
        sw_304[1] = words[84 + woff_261 + 1];
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                : "r"(sw_304[_pair]));
        }
        float2 v_305 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        sq[6] = fma_f32x2_rn_noftz(v_305, v_305, sq[6]);
        float2 v_306 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        sq[6] = fma_f32x2_rn_noftz(v_306, v_306, sq[6]);
        float2 v_307 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        float2 qp_308 = make_float2(q[24], q[25]);
        dot[6] = fma_f32x2_rn_noftz(v_307, qp_308, dot[6]);
        float2 v_309 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        float2 qp_310 = make_float2(q[26], q[27]);
        dot[6] = fma_f32x2_rn_noftz(v_309, qp_310, dot[6]);
        unsigned int sw_311[4];
        sw_311[0] = words[98 + woff_261];
        sw_311[1] = words[98 + woff_261 + 1];
        __nv_bfloat162 a_312 = __as_bf16x2(sw_311[0]);
        __nv_bfloat162 d_313 = __as_bf16x2(dwords[woff_261]);
        __nv_bfloat162 mixed_314 = a_312 + d_313;
        sw_311[0] = __as_u32(mixed_314);
        words[98 + woff_261] = sw_311[0];
        __nv_bfloat162 a_315 = __as_bf16x2(sw_311[1]);
        __nv_bfloat162 d_316 = __as_bf16x2(dwords[woff_261 + 1]);
        __nv_bfloat162 mixed_317 = a_315 + d_316;
        sw_311[1] = __as_u32(mixed_317);
        words[98 + woff_261 + 1] = sw_311[1];
        {
            int2 _iv2 = make_int2(sw_311[0 + 0], sw_311[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_260)) + 0) = _iv2;
        }
        unsigned long long write_base_318 = block_base + 7 * blocks_k_stride + (unsigned long long)base_260;
        {
            int2 _iv2 = make_int2(sw_311[0 + 0], sw_311[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(blocks + write_base_318) + 0) = _iv2;
        }
        float _unpack_bf16_31[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                : "r"(sw_311[_pair]));
        }
        float2 v_319 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        sq[7] = fma_f32x2_rn_noftz(v_319, v_319, sq[7]);
        float2 v_320 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        sq[7] = fma_f32x2_rn_noftz(v_320, v_320, sq[7]);
        float2 v_321 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        float2 qp_322 = make_float2(q[24], q[25]);
        dot[7] = fma_f32x2_rn_noftz(v_321, qp_322, dot[7]);
        float2 v_323 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        float2 qp_324 = make_float2(q[26], q[27]);
        dot[7] = fma_f32x2_rn_noftz(v_323, qp_324, dot[7]);
        float2 pairs[8];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[2] = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[3] = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[4] = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[5] = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[6] = make_float2(sq[6].x + sq[6].y, dot[6].x + dot[6].y);
        pairs[7] = make_float2(sq[7].x + sq[7].y, dot[7].x + dot[7].y);
        unsigned long long bits = 0;
        float2 _bit_cast_src_39 = pairs[0];
        bits = reinterpret_cast<unsigned long long*>(&_bit_cast_src_39)[0];
        unsigned long long peerbits = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        float2 peer = make_float2(0.0f, 0.0f);
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_325 = 0;
        float2 _bit_cast_src_40 = pairs[1];
        bits_325 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_40)[0];
        unsigned long long peerbits_326 = __shfl_xor_sync(0xFFFFFFFF, bits_325, 16);
        float2 peer_327 = make_float2(0.0f, 0.0f);
        peer_327 = reinterpret_cast<float2*>(&peerbits_326)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_327);
        unsigned long long bits_328 = 0;
        float2 _bit_cast_src_41 = pairs[2];
        bits_328 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_41)[0];
        unsigned long long peerbits_329 = __shfl_xor_sync(0xFFFFFFFF, bits_328, 16);
        float2 peer_330 = make_float2(0.0f, 0.0f);
        peer_330 = reinterpret_cast<float2*>(&peerbits_329)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_330);
        unsigned long long bits_331 = 0;
        float2 _bit_cast_src_42 = pairs[3];
        bits_331 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits_332 = __shfl_xor_sync(0xFFFFFFFF, bits_331, 16);
        float2 peer_333 = make_float2(0.0f, 0.0f);
        peer_333 = reinterpret_cast<float2*>(&peerbits_332)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_333);
        unsigned long long bits_334 = 0;
        float2 _bit_cast_src_43 = pairs[4];
        bits_334 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_335 = __shfl_xor_sync(0xFFFFFFFF, bits_334, 16);
        float2 peer_336 = make_float2(0.0f, 0.0f);
        peer_336 = reinterpret_cast<float2*>(&peerbits_335)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_336);
        unsigned long long bits_337 = 0;
        float2 _bit_cast_src_44 = pairs[5];
        bits_337 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_338 = __shfl_xor_sync(0xFFFFFFFF, bits_337, 16);
        float2 peer_339 = make_float2(0.0f, 0.0f);
        peer_339 = reinterpret_cast<float2*>(&peerbits_338)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_339);
        unsigned long long bits_340 = 0;
        float2 _bit_cast_src_45 = pairs[6];
        bits_340 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_341 = __shfl_xor_sync(0xFFFFFFFF, bits_340, 16);
        float2 peer_342 = make_float2(0.0f, 0.0f);
        peer_342 = reinterpret_cast<float2*>(&peerbits_341)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_342);
        unsigned long long bits_343 = 0;
        float2 _bit_cast_src_46 = pairs[7];
        bits_343 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_344 = __shfl_xor_sync(0xFFFFFFFF, bits_343, 16);
        float2 peer_345 = make_float2(0.0f, 0.0f);
        peer_345 = reinterpret_cast<float2*>(&peerbits_344)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_345);
        unsigned long long bits_346 = 0;
        float2 _bit_cast_src_47 = pairs[0];
        bits_346 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_347 = __shfl_xor_sync(0xFFFFFFFF, bits_346, 8);
        float2 peer_348 = make_float2(0.0f, 0.0f);
        peer_348 = reinterpret_cast<float2*>(&peerbits_347)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_348);
        unsigned long long bits_349 = 0;
        float2 _bit_cast_src_48 = pairs[1];
        bits_349 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_350 = __shfl_xor_sync(0xFFFFFFFF, bits_349, 8);
        float2 peer_351 = make_float2(0.0f, 0.0f);
        peer_351 = reinterpret_cast<float2*>(&peerbits_350)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_351);
        unsigned long long bits_352 = 0;
        float2 _bit_cast_src_49 = pairs[2];
        bits_352 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_353 = __shfl_xor_sync(0xFFFFFFFF, bits_352, 8);
        float2 peer_354 = make_float2(0.0f, 0.0f);
        peer_354 = reinterpret_cast<float2*>(&peerbits_353)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_354);
        unsigned long long bits_355 = 0;
        float2 _bit_cast_src_50 = pairs[3];
        bits_355 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_356 = __shfl_xor_sync(0xFFFFFFFF, bits_355, 8);
        float2 peer_357 = make_float2(0.0f, 0.0f);
        peer_357 = reinterpret_cast<float2*>(&peerbits_356)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_357);
        unsigned long long bits_358 = 0;
        float2 _bit_cast_src_51 = pairs[4];
        bits_358 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_359 = __shfl_xor_sync(0xFFFFFFFF, bits_358, 8);
        float2 peer_360 = make_float2(0.0f, 0.0f);
        peer_360 = reinterpret_cast<float2*>(&peerbits_359)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_360);
        unsigned long long bits_361 = 0;
        float2 _bit_cast_src_52 = pairs[5];
        bits_361 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_362 = __shfl_xor_sync(0xFFFFFFFF, bits_361, 8);
        float2 peer_363 = make_float2(0.0f, 0.0f);
        peer_363 = reinterpret_cast<float2*>(&peerbits_362)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_363);
        unsigned long long bits_364 = 0;
        float2 _bit_cast_src_53 = pairs[6];
        bits_364 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_365 = __shfl_xor_sync(0xFFFFFFFF, bits_364, 8);
        float2 peer_366 = make_float2(0.0f, 0.0f);
        peer_366 = reinterpret_cast<float2*>(&peerbits_365)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_366);
        unsigned long long bits_367 = 0;
        float2 _bit_cast_src_54 = pairs[7];
        bits_367 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_368 = __shfl_xor_sync(0xFFFFFFFF, bits_367, 8);
        float2 peer_369 = make_float2(0.0f, 0.0f);
        peer_369 = reinterpret_cast<float2*>(&peerbits_368)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_369);
        unsigned long long bits_370 = 0;
        float2 _bit_cast_src_55 = pairs[0];
        bits_370 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_55)[0];
        unsigned long long peerbits_371 = __shfl_xor_sync(0xFFFFFFFF, bits_370, 4);
        float2 peer_372 = make_float2(0.0f, 0.0f);
        peer_372 = reinterpret_cast<float2*>(&peerbits_371)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_372);
        unsigned long long bits_373 = 0;
        float2 _bit_cast_src_56 = pairs[1];
        bits_373 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_56)[0];
        unsigned long long peerbits_374 = __shfl_xor_sync(0xFFFFFFFF, bits_373, 4);
        float2 peer_375 = make_float2(0.0f, 0.0f);
        peer_375 = reinterpret_cast<float2*>(&peerbits_374)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_375);
        unsigned long long bits_376 = 0;
        float2 _bit_cast_src_57 = pairs[2];
        bits_376 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_57)[0];
        unsigned long long peerbits_377 = __shfl_xor_sync(0xFFFFFFFF, bits_376, 4);
        float2 peer_378 = make_float2(0.0f, 0.0f);
        peer_378 = reinterpret_cast<float2*>(&peerbits_377)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_378);
        unsigned long long bits_379 = 0;
        float2 _bit_cast_src_58 = pairs[3];
        bits_379 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_58)[0];
        unsigned long long peerbits_380 = __shfl_xor_sync(0xFFFFFFFF, bits_379, 4);
        float2 peer_381 = make_float2(0.0f, 0.0f);
        peer_381 = reinterpret_cast<float2*>(&peerbits_380)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_381);
        unsigned long long bits_382 = 0;
        float2 _bit_cast_src_59 = pairs[4];
        bits_382 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_59)[0];
        unsigned long long peerbits_383 = __shfl_xor_sync(0xFFFFFFFF, bits_382, 4);
        float2 peer_384 = make_float2(0.0f, 0.0f);
        peer_384 = reinterpret_cast<float2*>(&peerbits_383)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_384);
        unsigned long long bits_385 = 0;
        float2 _bit_cast_src_60 = pairs[5];
        bits_385 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_60)[0];
        unsigned long long peerbits_386 = __shfl_xor_sync(0xFFFFFFFF, bits_385, 4);
        float2 peer_387 = make_float2(0.0f, 0.0f);
        peer_387 = reinterpret_cast<float2*>(&peerbits_386)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_387);
        unsigned long long bits_388 = 0;
        float2 _bit_cast_src_61 = pairs[6];
        bits_388 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_61)[0];
        unsigned long long peerbits_389 = __shfl_xor_sync(0xFFFFFFFF, bits_388, 4);
        float2 peer_390 = make_float2(0.0f, 0.0f);
        peer_390 = reinterpret_cast<float2*>(&peerbits_389)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_390);
        unsigned long long bits_391 = 0;
        float2 _bit_cast_src_62 = pairs[7];
        bits_391 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_62)[0];
        unsigned long long peerbits_392 = __shfl_xor_sync(0xFFFFFFFF, bits_391, 4);
        float2 peer_393 = make_float2(0.0f, 0.0f);
        peer_393 = reinterpret_cast<float2*>(&peerbits_392)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_393);
        unsigned long long bits_394 = 0;
        float2 _bit_cast_src_63 = pairs[0];
        bits_394 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_63)[0];
        unsigned long long peerbits_395 = __shfl_xor_sync(0xFFFFFFFF, bits_394, 2);
        float2 peer_396 = make_float2(0.0f, 0.0f);
        peer_396 = reinterpret_cast<float2*>(&peerbits_395)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_396);
        unsigned long long bits_397 = 0;
        float2 _bit_cast_src_64 = pairs[1];
        bits_397 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_64)[0];
        unsigned long long peerbits_398 = __shfl_xor_sync(0xFFFFFFFF, bits_397, 2);
        float2 peer_399 = make_float2(0.0f, 0.0f);
        peer_399 = reinterpret_cast<float2*>(&peerbits_398)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_399);
        unsigned long long bits_400 = 0;
        float2 _bit_cast_src_65 = pairs[2];
        bits_400 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_65)[0];
        unsigned long long peerbits_401 = __shfl_xor_sync(0xFFFFFFFF, bits_400, 2);
        float2 peer_402 = make_float2(0.0f, 0.0f);
        peer_402 = reinterpret_cast<float2*>(&peerbits_401)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_402);
        unsigned long long bits_403 = 0;
        float2 _bit_cast_src_66 = pairs[3];
        bits_403 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_66)[0];
        unsigned long long peerbits_404 = __shfl_xor_sync(0xFFFFFFFF, bits_403, 2);
        float2 peer_405 = make_float2(0.0f, 0.0f);
        peer_405 = reinterpret_cast<float2*>(&peerbits_404)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_405);
        unsigned long long bits_406 = 0;
        float2 _bit_cast_src_67 = pairs[4];
        bits_406 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_67)[0];
        unsigned long long peerbits_407 = __shfl_xor_sync(0xFFFFFFFF, bits_406, 2);
        float2 peer_408 = make_float2(0.0f, 0.0f);
        peer_408 = reinterpret_cast<float2*>(&peerbits_407)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_408);
        unsigned long long bits_409 = 0;
        float2 _bit_cast_src_68 = pairs[5];
        bits_409 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_68)[0];
        unsigned long long peerbits_410 = __shfl_xor_sync(0xFFFFFFFF, bits_409, 2);
        float2 peer_411 = make_float2(0.0f, 0.0f);
        peer_411 = reinterpret_cast<float2*>(&peerbits_410)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_411);
        unsigned long long bits_412 = 0;
        float2 _bit_cast_src_69 = pairs[6];
        bits_412 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_69)[0];
        unsigned long long peerbits_413 = __shfl_xor_sync(0xFFFFFFFF, bits_412, 2);
        float2 peer_414 = make_float2(0.0f, 0.0f);
        peer_414 = reinterpret_cast<float2*>(&peerbits_413)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_414);
        unsigned long long bits_415 = 0;
        float2 _bit_cast_src_70 = pairs[7];
        bits_415 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_70)[0];
        unsigned long long peerbits_416 = __shfl_xor_sync(0xFFFFFFFF, bits_415, 2);
        float2 peer_417 = make_float2(0.0f, 0.0f);
        peer_417 = reinterpret_cast<float2*>(&peerbits_416)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_417);
        unsigned long long bits_418 = 0;
        float2 _bit_cast_src_71 = pairs[0];
        bits_418 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_71)[0];
        unsigned long long peerbits_419 = __shfl_xor_sync(0xFFFFFFFF, bits_418, 1);
        float2 peer_420 = make_float2(0.0f, 0.0f);
        peer_420 = reinterpret_cast<float2*>(&peerbits_419)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_420);
        unsigned long long bits_421 = 0;
        float2 _bit_cast_src_72 = pairs[1];
        bits_421 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_72)[0];
        unsigned long long peerbits_422 = __shfl_xor_sync(0xFFFFFFFF, bits_421, 1);
        float2 peer_423 = make_float2(0.0f, 0.0f);
        peer_423 = reinterpret_cast<float2*>(&peerbits_422)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_423);
        unsigned long long bits_424 = 0;
        float2 _bit_cast_src_73 = pairs[2];
        bits_424 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_73)[0];
        unsigned long long peerbits_425 = __shfl_xor_sync(0xFFFFFFFF, bits_424, 1);
        float2 peer_426 = make_float2(0.0f, 0.0f);
        peer_426 = reinterpret_cast<float2*>(&peerbits_425)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_426);
        unsigned long long bits_427 = 0;
        float2 _bit_cast_src_74 = pairs[3];
        bits_427 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_74)[0];
        unsigned long long peerbits_428 = __shfl_xor_sync(0xFFFFFFFF, bits_427, 1);
        float2 peer_429 = make_float2(0.0f, 0.0f);
        peer_429 = reinterpret_cast<float2*>(&peerbits_428)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_429);
        unsigned long long bits_430 = 0;
        float2 _bit_cast_src_75 = pairs[4];
        bits_430 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_75)[0];
        unsigned long long peerbits_431 = __shfl_xor_sync(0xFFFFFFFF, bits_430, 1);
        float2 peer_432 = make_float2(0.0f, 0.0f);
        peer_432 = reinterpret_cast<float2*>(&peerbits_431)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_432);
        unsigned long long bits_433 = 0;
        float2 _bit_cast_src_76 = pairs[5];
        bits_433 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_76)[0];
        unsigned long long peerbits_434 = __shfl_xor_sync(0xFFFFFFFF, bits_433, 1);
        float2 peer_435 = make_float2(0.0f, 0.0f);
        peer_435 = reinterpret_cast<float2*>(&peerbits_434)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_435);
        unsigned long long bits_436 = 0;
        float2 _bit_cast_src_77 = pairs[6];
        bits_436 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_77)[0];
        unsigned long long peerbits_437 = __shfl_xor_sync(0xFFFFFFFF, bits_436, 1);
        float2 peer_438 = make_float2(0.0f, 0.0f);
        peer_438 = reinterpret_cast<float2*>(&peerbits_437)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_438);
        unsigned long long bits_439 = 0;
        float2 _bit_cast_src_78 = pairs[7];
        bits_439 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_78)[0];
        unsigned long long peerbits_440 = __shfl_xor_sync(0xFFFFFFFF, bits_439, 1);
        float2 peer_441 = make_float2(0.0f, 0.0f);
        peer_441 = reinterpret_cast<float2*>(&peerbits_440)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_441);
        if (lane == 0) {
            stats[warp_0 * 8 * 2] = pairs[0].x;
            stats[warp_0 * 8 * 2 + 1] = pairs[0].y;
            stats[(warp_0 * 8 + 1) * 2] = pairs[1].x;
            stats[(warp_0 * 8 + 1) * 2 + 1] = pairs[1].y;
            stats[(warp_0 * 8 + 2) * 2] = pairs[2].x;
            stats[(warp_0 * 8 + 2) * 2 + 1] = pairs[2].y;
            stats[(warp_0 * 8 + 3) * 2] = pairs[3].x;
            stats[(warp_0 * 8 + 3) * 2 + 1] = pairs[3].y;
            stats[(warp_0 * 8 + 4) * 2] = pairs[4].x;
            stats[(warp_0 * 8 + 4) * 2 + 1] = pairs[4].y;
            stats[(warp_0 * 8 + 5) * 2] = pairs[5].x;
            stats[(warp_0 * 8 + 5) * 2 + 1] = pairs[5].y;
            stats[(warp_0 * 8 + 6) * 2] = pairs[6].x;
            stats[(warp_0 * 8 + 6) * 2 + 1] = pairs[6].y;
            stats[(warp_0 * 8 + 7) * 2] = pairs[7].x;
            stats[(warp_0 * 8 + 7) * 2 + 1] = pairs[7].y;
        }
        __syncthreads();
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 8) {
            total_sq = stats[(stat_w * 8 + stat_n) * 2];
            total_dot = stats[(stat_w * 8 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_79 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_79;
        float _shfl_down_sync_80 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_80;
        float _shfl_down_sync_81 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_81;
        float _shfl_down_sync_82 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_82;
        float _shfl_down_sync_83 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_83;
        float _shfl_down_sync_84 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_84;
        float logit = 0.0f;
        if (stat_n < 8 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[8];
        float _shfl_sync_85 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_85;
        float _shfl_sync_86 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_86;
        float _shfl_sync_87 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_87;
        float _shfl_sync_88 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_sync_88;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 4) {
            total_sq2 = stats[(stat_w * 8 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 8 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_89 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_89;
        float _shfl_down_sync_90 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_90;
        float _shfl_down_sync_91 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_91;
        float _shfl_down_sync_92 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_92;
        float _shfl_down_sync_93 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_93;
        float _shfl_down_sync_94 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_94;
        logit2 = 0.0f;
        if (stat_n < 4 && stat_w == 0) {
            float sigma2 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_sync_95 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_sync_95;
        float _shfl_sync_96 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_sync_96;
        float _shfl_sync_97 = __shfl_sync(0xFFFFFFFF, logit2, 16);
        logits[6] = _shfl_sync_97;
        float _shfl_sync_98 = __shfl_sync(0xFFFFFFFF, logit2, 24);
        logits[7] = _shfl_sync_98;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028235e+38f;
        max_chunk = fmaxf(max_chunk, logits[0]);
        max_chunk = fmaxf(max_chunk, logits[1]);
        max_chunk = fmaxf(max_chunk, logits[2]);
        max_chunk = fmaxf(max_chunk, logits[3]);
        float max_new = fmaxf(max_running, max_chunk);
        float correction = approx_exp2((max_running - max_new) * 1.442695f);
        float weights[8];
        float sum_weights = 0.0f;
        float _exp2_99 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_99;
        sum_weights += weights[0];
        float _exp2_100 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_100;
        sum_weights += weights[1];
        float _exp2_101 = approx_exp2((logits[2] - max_new) * 1.442695f);
        weights[2] = _exp2_101;
        sum_weights += weights[2];
        float _exp2_102 = approx_exp2((logits[3] - max_new) * 1.442695f);
        weights[3] = _exp2_102;
        sum_weights += weights[3];
        float2 corr = make_float2(correction, correction);
        const int woff_442 = 0;
        int base_443 = group * 1024 + thread * 8;
        float2 a_444[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_444[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_445 = make_float2(acc[2], acc[3]);
        a_444[1] = mul_f32x2_noftz(previous_445, corr);
        float2 previous_446 = make_float2(acc[4], acc[5]);
        a_444[2] = mul_f32x2_noftz(previous_446, corr);
        float2 previous_447 = make_float2(acc[6], acc[7]);
        a_444[3] = mul_f32x2_noftz(previous_447, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_448 = make_float2(fsrc[0], fsrc[1]);
        a_444[0] = fma_f32x2_rn_noftz(weight, v_448, a_444[0]);
        float2 v_449 = make_float2(fsrc[2], fsrc[3]);
        a_444[1] = fma_f32x2_rn_noftz(weight, v_449, a_444[1]);
        float2 v_450 = make_float2(fsrc[4], fsrc[5]);
        a_444[2] = fma_f32x2_rn_noftz(weight, v_450, a_444[2]);
        float2 v_451 = make_float2(fsrc[6], fsrc[7]);
        a_444[3] = fma_f32x2_rn_noftz(weight, v_451, a_444[3]);
        float2 weight_452 = make_float2(weights[1], weights[1]);
        float2 v_453 = make_float2(fsrc[28], fsrc[29]);
        a_444[0] = fma_f32x2_rn_noftz(weight_452, v_453, a_444[0]);
        float2 v_454 = make_float2(fsrc[30], fsrc[31]);
        a_444[1] = fma_f32x2_rn_noftz(weight_452, v_454, a_444[1]);
        float2 v_455 = make_float2(fsrc[32], fsrc[33]);
        a_444[2] = fma_f32x2_rn_noftz(weight_452, v_455, a_444[2]);
        float2 v_456 = make_float2(fsrc[34], fsrc[35]);
        a_444[3] = fma_f32x2_rn_noftz(weight_452, v_456, a_444[3]);
        float2 weight_457 = make_float2(weights[2], weights[2]);
        float2 v_458 = make_float2(fsrc[56], fsrc[57]);
        a_444[0] = fma_f32x2_rn_noftz(weight_457, v_458, a_444[0]);
        float2 v_459 = make_float2(fsrc[58], fsrc[59]);
        a_444[1] = fma_f32x2_rn_noftz(weight_457, v_459, a_444[1]);
        float2 v_460 = make_float2(fsrc[60], fsrc[61]);
        a_444[2] = fma_f32x2_rn_noftz(weight_457, v_460, a_444[2]);
        float2 v_461 = make_float2(fsrc[62], fsrc[63]);
        a_444[3] = fma_f32x2_rn_noftz(weight_457, v_461, a_444[3]);
        float2 weight_462 = make_float2(weights[3], weights[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_442];
        sw2[1] = words[42 + woff_442 + 1];
        sw2[2] = words[42 + woff_442 + 2];
        sw2[3] = words[42 + woff_442 + 3];
        float _unpack_bf16_32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                : "r"(sw2[_pair]));
        }
        float2 v_463 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
        a_444[0] = fma_f32x2_rn_noftz(weight_462, v_463, a_444[0]);
        float2 v_464 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
        a_444[1] = fma_f32x2_rn_noftz(weight_462, v_464, a_444[1]);
        float2 v_465 = make_float2(_unpack_bf16_32[4], _unpack_bf16_32[5]);
        a_444[2] = fma_f32x2_rn_noftz(weight_462, v_465, a_444[2]);
        float2 v_466 = make_float2(_unpack_bf16_32[6], _unpack_bf16_32[7]);
        a_444[3] = fma_f32x2_rn_noftz(weight_462, v_466, a_444[3]);
        acc[0] = a_444[0].x;
        acc[1] = a_444[0].y;
        acc[2] = a_444[1].x;
        acc[3] = a_444[1].y;
        acc[4] = a_444[2].x;
        acc[5] = a_444[2].y;
        acc[6] = a_444[3].x;
        acc[7] = a_444[3].y;
        const int woff_467 = 4;
        int base_468 = (2 + group) * 1024 + thread * 8;
        float2 a_469[4];
        float2 previous_470 = make_float2(acc[8], acc[9]);
        a_469[0] = mul_f32x2_noftz(previous_470, corr);
        float2 previous_471 = make_float2(acc[10], acc[11]);
        a_469[1] = mul_f32x2_noftz(previous_471, corr);
        float2 previous_472 = make_float2(acc[12], acc[13]);
        a_469[2] = mul_f32x2_noftz(previous_472, corr);
        float2 previous_473 = make_float2(acc[14], acc[15]);
        a_469[3] = mul_f32x2_noftz(previous_473, corr);
        float2 weight_474 = make_float2(weights[0], weights[0]);
        float2 v_475 = make_float2(fsrc[8], fsrc[9]);
        a_469[0] = fma_f32x2_rn_noftz(weight_474, v_475, a_469[0]);
        float2 v_476 = make_float2(fsrc[10], fsrc[11]);
        a_469[1] = fma_f32x2_rn_noftz(weight_474, v_476, a_469[1]);
        float2 v_477 = make_float2(fsrc[12], fsrc[13]);
        a_469[2] = fma_f32x2_rn_noftz(weight_474, v_477, a_469[2]);
        float2 v_478 = make_float2(fsrc[14], fsrc[15]);
        a_469[3] = fma_f32x2_rn_noftz(weight_474, v_478, a_469[3]);
        float2 weight_479 = make_float2(weights[1], weights[1]);
        float2 v_480 = make_float2(fsrc[36], fsrc[37]);
        a_469[0] = fma_f32x2_rn_noftz(weight_479, v_480, a_469[0]);
        float2 v_481 = make_float2(fsrc[38], fsrc[39]);
        a_469[1] = fma_f32x2_rn_noftz(weight_479, v_481, a_469[1]);
        float2 v_482 = make_float2(fsrc[40], fsrc[41]);
        a_469[2] = fma_f32x2_rn_noftz(weight_479, v_482, a_469[2]);
        float2 v_483 = make_float2(fsrc[42], fsrc[43]);
        a_469[3] = fma_f32x2_rn_noftz(weight_479, v_483, a_469[3]);
        float2 weight_484 = make_float2(weights[2], weights[2]);
        float2 v_485 = make_float2(fsrc[64], fsrc[65]);
        a_469[0] = fma_f32x2_rn_noftz(weight_484, v_485, a_469[0]);
        float2 v_486 = make_float2(fsrc[66], fsrc[67]);
        a_469[1] = fma_f32x2_rn_noftz(weight_484, v_486, a_469[1]);
        float2 v_487 = make_float2(fsrc[68], fsrc[69]);
        a_469[2] = fma_f32x2_rn_noftz(weight_484, v_487, a_469[2]);
        float2 v_488 = make_float2(fsrc[70], fsrc[71]);
        a_469[3] = fma_f32x2_rn_noftz(weight_484, v_488, a_469[3]);
        float2 weight_489 = make_float2(weights[3], weights[3]);
        unsigned int sw2_490[4];
        sw2_490[0] = words[42 + woff_467];
        sw2_490[1] = words[42 + woff_467 + 1];
        sw2_490[2] = words[42 + woff_467 + 2];
        sw2_490[3] = words[42 + woff_467 + 3];
        float _unpack_bf16_33[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                : "r"(sw2_490[_pair]));
        }
        float2 v_491 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
        a_469[0] = fma_f32x2_rn_noftz(weight_489, v_491, a_469[0]);
        float2 v_492 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
        a_469[1] = fma_f32x2_rn_noftz(weight_489, v_492, a_469[1]);
        float2 v_493 = make_float2(_unpack_bf16_33[4], _unpack_bf16_33[5]);
        a_469[2] = fma_f32x2_rn_noftz(weight_489, v_493, a_469[2]);
        float2 v_494 = make_float2(_unpack_bf16_33[6], _unpack_bf16_33[7]);
        a_469[3] = fma_f32x2_rn_noftz(weight_489, v_494, a_469[3]);
        acc[8] = a_469[0].x;
        acc[9] = a_469[0].y;
        acc[10] = a_469[1].x;
        acc[11] = a_469[1].y;
        acc[12] = a_469[2].x;
        acc[13] = a_469[2].y;
        acc[14] = a_469[3].x;
        acc[15] = a_469[3].y;
        const int woff_495 = 8;
        int base_496 = (4 + group) * 1024 + thread * 8;
        float2 a_497[4];
        float2 previous_498 = make_float2(acc[16], acc[17]);
        a_497[0] = mul_f32x2_noftz(previous_498, corr);
        float2 previous_499 = make_float2(acc[18], acc[19]);
        a_497[1] = mul_f32x2_noftz(previous_499, corr);
        float2 previous_500 = make_float2(acc[20], acc[21]);
        a_497[2] = mul_f32x2_noftz(previous_500, corr);
        float2 previous_501 = make_float2(acc[22], acc[23]);
        a_497[3] = mul_f32x2_noftz(previous_501, corr);
        float2 weight_502 = make_float2(weights[0], weights[0]);
        float2 v_503 = make_float2(fsrc[16], fsrc[17]);
        a_497[0] = fma_f32x2_rn_noftz(weight_502, v_503, a_497[0]);
        float2 v_504 = make_float2(fsrc[18], fsrc[19]);
        a_497[1] = fma_f32x2_rn_noftz(weight_502, v_504, a_497[1]);
        float2 v_505 = make_float2(fsrc[20], fsrc[21]);
        a_497[2] = fma_f32x2_rn_noftz(weight_502, v_505, a_497[2]);
        float2 v_506 = make_float2(fsrc[22], fsrc[23]);
        a_497[3] = fma_f32x2_rn_noftz(weight_502, v_506, a_497[3]);
        float2 weight_507 = make_float2(weights[1], weights[1]);
        float2 v_508 = make_float2(fsrc[44], fsrc[45]);
        a_497[0] = fma_f32x2_rn_noftz(weight_507, v_508, a_497[0]);
        float2 v_509 = make_float2(fsrc[46], fsrc[47]);
        a_497[1] = fma_f32x2_rn_noftz(weight_507, v_509, a_497[1]);
        float2 v_510 = make_float2(fsrc[48], fsrc[49]);
        a_497[2] = fma_f32x2_rn_noftz(weight_507, v_510, a_497[2]);
        float2 v_511 = make_float2(fsrc[50], fsrc[51]);
        a_497[3] = fma_f32x2_rn_noftz(weight_507, v_511, a_497[3]);
        float2 weight_512 = make_float2(weights[2], weights[2]);
        float2 v_513 = make_float2(fsrc[72], fsrc[73]);
        a_497[0] = fma_f32x2_rn_noftz(weight_512, v_513, a_497[0]);
        float2 v_514 = make_float2(fsrc[74], fsrc[75]);
        a_497[1] = fma_f32x2_rn_noftz(weight_512, v_514, a_497[1]);
        float2 v_515 = make_float2(fsrc[76], fsrc[77]);
        a_497[2] = fma_f32x2_rn_noftz(weight_512, v_515, a_497[2]);
        float2 v_516 = make_float2(fsrc[78], fsrc[79]);
        a_497[3] = fma_f32x2_rn_noftz(weight_512, v_516, a_497[3]);
        float2 weight_517 = make_float2(weights[3], weights[3]);
        unsigned int sw2_518[4];
        sw2_518[0] = words[42 + woff_495];
        sw2_518[1] = words[42 + woff_495 + 1];
        sw2_518[2] = words[42 + woff_495 + 2];
        sw2_518[3] = words[42 + woff_495 + 3];
        float _unpack_bf16_34[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                : "r"(sw2_518[_pair]));
        }
        float2 v_519 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
        a_497[0] = fma_f32x2_rn_noftz(weight_517, v_519, a_497[0]);
        float2 v_520 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
        a_497[1] = fma_f32x2_rn_noftz(weight_517, v_520, a_497[1]);
        float2 v_521 = make_float2(_unpack_bf16_34[4], _unpack_bf16_34[5]);
        a_497[2] = fma_f32x2_rn_noftz(weight_517, v_521, a_497[2]);
        float2 v_522 = make_float2(_unpack_bf16_34[6], _unpack_bf16_34[7]);
        a_497[3] = fma_f32x2_rn_noftz(weight_517, v_522, a_497[3]);
        acc[16] = a_497[0].x;
        acc[17] = a_497[0].y;
        acc[18] = a_497[1].x;
        acc[19] = a_497[1].y;
        acc[20] = a_497[2].x;
        acc[21] = a_497[2].y;
        acc[22] = a_497[3].x;
        acc[23] = a_497[3].y;
        const int woff_523 = 12;
        int base_524 = 6144 + group * 512 + thread * 4;
        float2 a_525[4];
        float2 previous_526 = make_float2(acc[24], acc[25]);
        a_525[0] = mul_f32x2_noftz(previous_526, corr);
        float2 previous_527 = make_float2(acc[26], acc[27]);
        a_525[1] = mul_f32x2_noftz(previous_527, corr);
        float2 weight_528 = make_float2(weights[0], weights[0]);
        float2 v_529 = make_float2(fsrc[24], fsrc[25]);
        a_525[0] = fma_f32x2_rn_noftz(weight_528, v_529, a_525[0]);
        float2 v_530 = make_float2(fsrc[26], fsrc[27]);
        a_525[1] = fma_f32x2_rn_noftz(weight_528, v_530, a_525[1]);
        float2 weight_531 = make_float2(weights[1], weights[1]);
        float2 v_532 = make_float2(fsrc[52], fsrc[53]);
        a_525[0] = fma_f32x2_rn_noftz(weight_531, v_532, a_525[0]);
        float2 v_533 = make_float2(fsrc[54], fsrc[55]);
        a_525[1] = fma_f32x2_rn_noftz(weight_531, v_533, a_525[1]);
        float2 weight_534 = make_float2(weights[2], weights[2]);
        float2 v_535 = make_float2(fsrc[80], fsrc[81]);
        a_525[0] = fma_f32x2_rn_noftz(weight_534, v_535, a_525[0]);
        float2 v_536 = make_float2(fsrc[82], fsrc[83]);
        a_525[1] = fma_f32x2_rn_noftz(weight_534, v_536, a_525[1]);
        float2 weight_537 = make_float2(weights[3], weights[3]);
        unsigned int sw2_538[4];
        sw2_538[0] = words[42 + woff_523];
        sw2_538[1] = words[42 + woff_523 + 1];
        float _unpack_bf16_35[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                : "r"(sw2_538[_pair]));
        }
        float2 v_539 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
        a_525[0] = fma_f32x2_rn_noftz(weight_537, v_539, a_525[0]);
        float2 v_540 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
        a_525[1] = fma_f32x2_rn_noftz(weight_537, v_540, a_525[1]);
        acc[24] = a_525[0].x;
        acc[25] = a_525[0].y;
        acc[26] = a_525[1].x;
        acc[27] = a_525[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_541 = -3.4028235e+38f;
        max_chunk_541 = fmaxf(max_chunk_541, logits[4]);
        max_chunk_541 = fmaxf(max_chunk_541, logits[5]);
        max_chunk_541 = fmaxf(max_chunk_541, logits[6]);
        max_chunk_541 = fmaxf(max_chunk_541, logits[7]);
        float max_new_542 = fmaxf(max_running, max_chunk_541);
        float correction_543 = approx_exp2((max_running - max_new_542) * 1.442695f);
        float weights_544[8];
        float sum_weights_545 = 0.0f;
        float _exp2_103 = approx_exp2((logits[4] - max_new_542) * 1.442695f);
        weights_544[4] = _exp2_103;
        sum_weights_545 += weights_544[4];
        float _exp2_104 = approx_exp2((logits[5] - max_new_542) * 1.442695f);
        weights_544[5] = _exp2_104;
        sum_weights_545 += weights_544[5];
        float _exp2_105 = approx_exp2((logits[6] - max_new_542) * 1.442695f);
        weights_544[6] = _exp2_105;
        sum_weights_545 += weights_544[6];
        float _exp2_106 = approx_exp2((logits[7] - max_new_542) * 1.442695f);
        weights_544[7] = _exp2_106;
        sum_weights_545 += weights_544[7];
        float2 corr_546 = make_float2(correction_543, correction_543);
        const int woff_547 = 0;
        int base_548 = group * 1024 + thread * 8;
        float2 a_549[4];
        float2 previous_550 = make_float2(acc[0], acc[1]);
        a_549[0] = mul_f32x2_noftz(previous_550, corr_546);
        float2 previous_551 = make_float2(acc[2], acc[3]);
        a_549[1] = mul_f32x2_noftz(previous_551, corr_546);
        float2 previous_552 = make_float2(acc[4], acc[5]);
        a_549[2] = mul_f32x2_noftz(previous_552, corr_546);
        float2 previous_553 = make_float2(acc[6], acc[7]);
        a_549[3] = mul_f32x2_noftz(previous_553, corr_546);
        float2 weight_554 = make_float2(weights_544[4], weights_544[4]);
        unsigned int sw2_555[4];
        sw2_555[0] = words[56 + woff_547];
        sw2_555[1] = words[56 + woff_547 + 1];
        sw2_555[2] = words[56 + woff_547 + 2];
        sw2_555[3] = words[56 + woff_547 + 3];
        float _unpack_bf16_36[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                : "r"(sw2_555[_pair]));
        }
        float2 v_556 = make_float2(_unpack_bf16_36[0], _unpack_bf16_36[1]);
        a_549[0] = fma_f32x2_rn_noftz(weight_554, v_556, a_549[0]);
        float2 v_557 = make_float2(_unpack_bf16_36[2], _unpack_bf16_36[3]);
        a_549[1] = fma_f32x2_rn_noftz(weight_554, v_557, a_549[1]);
        float2 v_558 = make_float2(_unpack_bf16_36[4], _unpack_bf16_36[5]);
        a_549[2] = fma_f32x2_rn_noftz(weight_554, v_558, a_549[2]);
        float2 v_559 = make_float2(_unpack_bf16_36[6], _unpack_bf16_36[7]);
        a_549[3] = fma_f32x2_rn_noftz(weight_554, v_559, a_549[3]);
        float2 weight_560 = make_float2(weights_544[5], weights_544[5]);
        unsigned int sw2_561[4];
        sw2_561[0] = words[70 + woff_547];
        sw2_561[1] = words[70 + woff_547 + 1];
        sw2_561[2] = words[70 + woff_547 + 2];
        sw2_561[3] = words[70 + woff_547 + 3];
        float _unpack_bf16_37[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                : "r"(sw2_561[_pair]));
        }
        float2 v_562 = make_float2(_unpack_bf16_37[0], _unpack_bf16_37[1]);
        a_549[0] = fma_f32x2_rn_noftz(weight_560, v_562, a_549[0]);
        float2 v_563 = make_float2(_unpack_bf16_37[2], _unpack_bf16_37[3]);
        a_549[1] = fma_f32x2_rn_noftz(weight_560, v_563, a_549[1]);
        float2 v_564 = make_float2(_unpack_bf16_37[4], _unpack_bf16_37[5]);
        a_549[2] = fma_f32x2_rn_noftz(weight_560, v_564, a_549[2]);
        float2 v_565 = make_float2(_unpack_bf16_37[6], _unpack_bf16_37[7]);
        a_549[3] = fma_f32x2_rn_noftz(weight_560, v_565, a_549[3]);
        float2 weight_566 = make_float2(weights_544[6], weights_544[6]);
        unsigned int sw2_567[4];
        sw2_567[0] = words[84 + woff_547];
        sw2_567[1] = words[84 + woff_547 + 1];
        sw2_567[2] = words[84 + woff_547 + 2];
        sw2_567[3] = words[84 + woff_547 + 3];
        float _unpack_bf16_38[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                : "r"(sw2_567[_pair]));
        }
        float2 v_568 = make_float2(_unpack_bf16_38[0], _unpack_bf16_38[1]);
        a_549[0] = fma_f32x2_rn_noftz(weight_566, v_568, a_549[0]);
        float2 v_569 = make_float2(_unpack_bf16_38[2], _unpack_bf16_38[3]);
        a_549[1] = fma_f32x2_rn_noftz(weight_566, v_569, a_549[1]);
        float2 v_570 = make_float2(_unpack_bf16_38[4], _unpack_bf16_38[5]);
        a_549[2] = fma_f32x2_rn_noftz(weight_566, v_570, a_549[2]);
        float2 v_571 = make_float2(_unpack_bf16_38[6], _unpack_bf16_38[7]);
        a_549[3] = fma_f32x2_rn_noftz(weight_566, v_571, a_549[3]);
        float2 weight_572 = make_float2(weights_544[7], weights_544[7]);
        unsigned int sw2_573[4];
        sw2_573[0] = words[98 + woff_547];
        sw2_573[1] = words[98 + woff_547 + 1];
        sw2_573[2] = words[98 + woff_547 + 2];
        sw2_573[3] = words[98 + woff_547 + 3];
        float _unpack_bf16_39[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                : "r"(sw2_573[_pair]));
        }
        float2 v_574 = make_float2(_unpack_bf16_39[0], _unpack_bf16_39[1]);
        a_549[0] = fma_f32x2_rn_noftz(weight_572, v_574, a_549[0]);
        float2 v_575 = make_float2(_unpack_bf16_39[2], _unpack_bf16_39[3]);
        a_549[1] = fma_f32x2_rn_noftz(weight_572, v_575, a_549[1]);
        float2 v_576 = make_float2(_unpack_bf16_39[4], _unpack_bf16_39[5]);
        a_549[2] = fma_f32x2_rn_noftz(weight_572, v_576, a_549[2]);
        float2 v_577 = make_float2(_unpack_bf16_39[6], _unpack_bf16_39[7]);
        a_549[3] = fma_f32x2_rn_noftz(weight_572, v_577, a_549[3]);
        acc[0] = a_549[0].x;
        acc[1] = a_549[0].y;
        acc[2] = a_549[1].x;
        acc[3] = a_549[1].y;
        acc[4] = a_549[2].x;
        acc[5] = a_549[2].y;
        acc[6] = a_549[3].x;
        acc[7] = a_549[3].y;
        const int woff_578 = 4;
        int base_579 = (2 + group) * 1024 + thread * 8;
        float2 a_580[4];
        float2 previous_581 = make_float2(acc[8], acc[9]);
        a_580[0] = mul_f32x2_noftz(previous_581, corr_546);
        float2 previous_582 = make_float2(acc[10], acc[11]);
        a_580[1] = mul_f32x2_noftz(previous_582, corr_546);
        float2 previous_583 = make_float2(acc[12], acc[13]);
        a_580[2] = mul_f32x2_noftz(previous_583, corr_546);
        float2 previous_584 = make_float2(acc[14], acc[15]);
        a_580[3] = mul_f32x2_noftz(previous_584, corr_546);
        float2 weight_585 = make_float2(weights_544[4], weights_544[4]);
        unsigned int sw2_586[4];
        sw2_586[0] = words[56 + woff_578];
        sw2_586[1] = words[56 + woff_578 + 1];
        sw2_586[2] = words[56 + woff_578 + 2];
        sw2_586[3] = words[56 + woff_578 + 3];
        float _unpack_bf16_40[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                : "r"(sw2_586[_pair]));
        }
        float2 v_587 = make_float2(_unpack_bf16_40[0], _unpack_bf16_40[1]);
        a_580[0] = fma_f32x2_rn_noftz(weight_585, v_587, a_580[0]);
        float2 v_588 = make_float2(_unpack_bf16_40[2], _unpack_bf16_40[3]);
        a_580[1] = fma_f32x2_rn_noftz(weight_585, v_588, a_580[1]);
        float2 v_589 = make_float2(_unpack_bf16_40[4], _unpack_bf16_40[5]);
        a_580[2] = fma_f32x2_rn_noftz(weight_585, v_589, a_580[2]);
        float2 v_590 = make_float2(_unpack_bf16_40[6], _unpack_bf16_40[7]);
        a_580[3] = fma_f32x2_rn_noftz(weight_585, v_590, a_580[3]);
        float2 weight_591 = make_float2(weights_544[5], weights_544[5]);
        unsigned int sw2_592[4];
        sw2_592[0] = words[70 + woff_578];
        sw2_592[1] = words[70 + woff_578 + 1];
        sw2_592[2] = words[70 + woff_578 + 2];
        sw2_592[3] = words[70 + woff_578 + 3];
        float _unpack_bf16_41[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                : "r"(sw2_592[_pair]));
        }
        float2 v_593 = make_float2(_unpack_bf16_41[0], _unpack_bf16_41[1]);
        a_580[0] = fma_f32x2_rn_noftz(weight_591, v_593, a_580[0]);
        float2 v_594 = make_float2(_unpack_bf16_41[2], _unpack_bf16_41[3]);
        a_580[1] = fma_f32x2_rn_noftz(weight_591, v_594, a_580[1]);
        float2 v_595 = make_float2(_unpack_bf16_41[4], _unpack_bf16_41[5]);
        a_580[2] = fma_f32x2_rn_noftz(weight_591, v_595, a_580[2]);
        float2 v_596 = make_float2(_unpack_bf16_41[6], _unpack_bf16_41[7]);
        a_580[3] = fma_f32x2_rn_noftz(weight_591, v_596, a_580[3]);
        float2 weight_597 = make_float2(weights_544[6], weights_544[6]);
        unsigned int sw2_598[4];
        sw2_598[0] = words[84 + woff_578];
        sw2_598[1] = words[84 + woff_578 + 1];
        sw2_598[2] = words[84 + woff_578 + 2];
        sw2_598[3] = words[84 + woff_578 + 3];
        float _unpack_bf16_42[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                : "r"(sw2_598[_pair]));
        }
        float2 v_599 = make_float2(_unpack_bf16_42[0], _unpack_bf16_42[1]);
        a_580[0] = fma_f32x2_rn_noftz(weight_597, v_599, a_580[0]);
        float2 v_600 = make_float2(_unpack_bf16_42[2], _unpack_bf16_42[3]);
        a_580[1] = fma_f32x2_rn_noftz(weight_597, v_600, a_580[1]);
        float2 v_601 = make_float2(_unpack_bf16_42[4], _unpack_bf16_42[5]);
        a_580[2] = fma_f32x2_rn_noftz(weight_597, v_601, a_580[2]);
        float2 v_602 = make_float2(_unpack_bf16_42[6], _unpack_bf16_42[7]);
        a_580[3] = fma_f32x2_rn_noftz(weight_597, v_602, a_580[3]);
        float2 weight_603 = make_float2(weights_544[7], weights_544[7]);
        unsigned int sw2_604[4];
        sw2_604[0] = words[98 + woff_578];
        sw2_604[1] = words[98 + woff_578 + 1];
        sw2_604[2] = words[98 + woff_578 + 2];
        sw2_604[3] = words[98 + woff_578 + 3];
        float _unpack_bf16_43[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                : "r"(sw2_604[_pair]));
        }
        float2 v_605 = make_float2(_unpack_bf16_43[0], _unpack_bf16_43[1]);
        a_580[0] = fma_f32x2_rn_noftz(weight_603, v_605, a_580[0]);
        float2 v_606 = make_float2(_unpack_bf16_43[2], _unpack_bf16_43[3]);
        a_580[1] = fma_f32x2_rn_noftz(weight_603, v_606, a_580[1]);
        float2 v_607 = make_float2(_unpack_bf16_43[4], _unpack_bf16_43[5]);
        a_580[2] = fma_f32x2_rn_noftz(weight_603, v_607, a_580[2]);
        float2 v_608 = make_float2(_unpack_bf16_43[6], _unpack_bf16_43[7]);
        a_580[3] = fma_f32x2_rn_noftz(weight_603, v_608, a_580[3]);
        acc[8] = a_580[0].x;
        acc[9] = a_580[0].y;
        acc[10] = a_580[1].x;
        acc[11] = a_580[1].y;
        acc[12] = a_580[2].x;
        acc[13] = a_580[2].y;
        acc[14] = a_580[3].x;
        acc[15] = a_580[3].y;
        const int woff_609 = 8;
        int base_610 = (4 + group) * 1024 + thread * 8;
        float2 a_611[4];
        float2 previous_612 = make_float2(acc[16], acc[17]);
        a_611[0] = mul_f32x2_noftz(previous_612, corr_546);
        float2 previous_613 = make_float2(acc[18], acc[19]);
        a_611[1] = mul_f32x2_noftz(previous_613, corr_546);
        float2 previous_614 = make_float2(acc[20], acc[21]);
        a_611[2] = mul_f32x2_noftz(previous_614, corr_546);
        float2 previous_615 = make_float2(acc[22], acc[23]);
        a_611[3] = mul_f32x2_noftz(previous_615, corr_546);
        float2 weight_616 = make_float2(weights_544[4], weights_544[4]);
        unsigned int sw2_617[4];
        sw2_617[0] = words[56 + woff_609];
        sw2_617[1] = words[56 + woff_609 + 1];
        sw2_617[2] = words[56 + woff_609 + 2];
        sw2_617[3] = words[56 + woff_609 + 3];
        float _unpack_bf16_44[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_44[_pair * 2])[0]), "=f"((&_unpack_bf16_44[_pair * 2])[1])
                : "r"(sw2_617[_pair]));
        }
        float2 v_618 = make_float2(_unpack_bf16_44[0], _unpack_bf16_44[1]);
        a_611[0] = fma_f32x2_rn_noftz(weight_616, v_618, a_611[0]);
        float2 v_619 = make_float2(_unpack_bf16_44[2], _unpack_bf16_44[3]);
        a_611[1] = fma_f32x2_rn_noftz(weight_616, v_619, a_611[1]);
        float2 v_620 = make_float2(_unpack_bf16_44[4], _unpack_bf16_44[5]);
        a_611[2] = fma_f32x2_rn_noftz(weight_616, v_620, a_611[2]);
        float2 v_621 = make_float2(_unpack_bf16_44[6], _unpack_bf16_44[7]);
        a_611[3] = fma_f32x2_rn_noftz(weight_616, v_621, a_611[3]);
        float2 weight_622 = make_float2(weights_544[5], weights_544[5]);
        unsigned int sw2_623[4];
        sw2_623[0] = words[70 + woff_609];
        sw2_623[1] = words[70 + woff_609 + 1];
        sw2_623[2] = words[70 + woff_609 + 2];
        sw2_623[3] = words[70 + woff_609 + 3];
        float _unpack_bf16_45[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_45[_pair * 2])[0]), "=f"((&_unpack_bf16_45[_pair * 2])[1])
                : "r"(sw2_623[_pair]));
        }
        float2 v_624 = make_float2(_unpack_bf16_45[0], _unpack_bf16_45[1]);
        a_611[0] = fma_f32x2_rn_noftz(weight_622, v_624, a_611[0]);
        float2 v_625 = make_float2(_unpack_bf16_45[2], _unpack_bf16_45[3]);
        a_611[1] = fma_f32x2_rn_noftz(weight_622, v_625, a_611[1]);
        float2 v_626 = make_float2(_unpack_bf16_45[4], _unpack_bf16_45[5]);
        a_611[2] = fma_f32x2_rn_noftz(weight_622, v_626, a_611[2]);
        float2 v_627 = make_float2(_unpack_bf16_45[6], _unpack_bf16_45[7]);
        a_611[3] = fma_f32x2_rn_noftz(weight_622, v_627, a_611[3]);
        float2 weight_628 = make_float2(weights_544[6], weights_544[6]);
        unsigned int sw2_629[4];
        sw2_629[0] = words[84 + woff_609];
        sw2_629[1] = words[84 + woff_609 + 1];
        sw2_629[2] = words[84 + woff_609 + 2];
        sw2_629[3] = words[84 + woff_609 + 3];
        float _unpack_bf16_46[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_46[_pair * 2])[0]), "=f"((&_unpack_bf16_46[_pair * 2])[1])
                : "r"(sw2_629[_pair]));
        }
        float2 v_630 = make_float2(_unpack_bf16_46[0], _unpack_bf16_46[1]);
        a_611[0] = fma_f32x2_rn_noftz(weight_628, v_630, a_611[0]);
        float2 v_631 = make_float2(_unpack_bf16_46[2], _unpack_bf16_46[3]);
        a_611[1] = fma_f32x2_rn_noftz(weight_628, v_631, a_611[1]);
        float2 v_632 = make_float2(_unpack_bf16_46[4], _unpack_bf16_46[5]);
        a_611[2] = fma_f32x2_rn_noftz(weight_628, v_632, a_611[2]);
        float2 v_633 = make_float2(_unpack_bf16_46[6], _unpack_bf16_46[7]);
        a_611[3] = fma_f32x2_rn_noftz(weight_628, v_633, a_611[3]);
        float2 weight_634 = make_float2(weights_544[7], weights_544[7]);
        unsigned int sw2_635[4];
        sw2_635[0] = words[98 + woff_609];
        sw2_635[1] = words[98 + woff_609 + 1];
        sw2_635[2] = words[98 + woff_609 + 2];
        sw2_635[3] = words[98 + woff_609 + 3];
        float _unpack_bf16_47[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_47[_pair * 2])[0]), "=f"((&_unpack_bf16_47[_pair * 2])[1])
                : "r"(sw2_635[_pair]));
        }
        float2 v_636 = make_float2(_unpack_bf16_47[0], _unpack_bf16_47[1]);
        a_611[0] = fma_f32x2_rn_noftz(weight_634, v_636, a_611[0]);
        float2 v_637 = make_float2(_unpack_bf16_47[2], _unpack_bf16_47[3]);
        a_611[1] = fma_f32x2_rn_noftz(weight_634, v_637, a_611[1]);
        float2 v_638 = make_float2(_unpack_bf16_47[4], _unpack_bf16_47[5]);
        a_611[2] = fma_f32x2_rn_noftz(weight_634, v_638, a_611[2]);
        float2 v_639 = make_float2(_unpack_bf16_47[6], _unpack_bf16_47[7]);
        a_611[3] = fma_f32x2_rn_noftz(weight_634, v_639, a_611[3]);
        acc[16] = a_611[0].x;
        acc[17] = a_611[0].y;
        acc[18] = a_611[1].x;
        acc[19] = a_611[1].y;
        acc[20] = a_611[2].x;
        acc[21] = a_611[2].y;
        acc[22] = a_611[3].x;
        acc[23] = a_611[3].y;
        const int woff_640 = 12;
        int base_641 = 6144 + group * 512 + thread * 4;
        float2 a_642[4];
        float2 previous_643 = make_float2(acc[24], acc[25]);
        a_642[0] = mul_f32x2_noftz(previous_643, corr_546);
        float2 previous_644 = make_float2(acc[26], acc[27]);
        a_642[1] = mul_f32x2_noftz(previous_644, corr_546);
        float2 weight_645 = make_float2(weights_544[4], weights_544[4]);
        unsigned int sw2_646[4];
        sw2_646[0] = words[56 + woff_640];
        sw2_646[1] = words[56 + woff_640 + 1];
        float _unpack_bf16_48[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_48[_pair * 2])[0]), "=f"((&_unpack_bf16_48[_pair * 2])[1])
                : "r"(sw2_646[_pair]));
        }
        float2 v_647 = make_float2(_unpack_bf16_48[0], _unpack_bf16_48[1]);
        a_642[0] = fma_f32x2_rn_noftz(weight_645, v_647, a_642[0]);
        float2 v_648 = make_float2(_unpack_bf16_48[2], _unpack_bf16_48[3]);
        a_642[1] = fma_f32x2_rn_noftz(weight_645, v_648, a_642[1]);
        float2 weight_649 = make_float2(weights_544[5], weights_544[5]);
        unsigned int sw2_650[4];
        sw2_650[0] = words[70 + woff_640];
        sw2_650[1] = words[70 + woff_640 + 1];
        float _unpack_bf16_49[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_49[_pair * 2])[0]), "=f"((&_unpack_bf16_49[_pair * 2])[1])
                : "r"(sw2_650[_pair]));
        }
        float2 v_651 = make_float2(_unpack_bf16_49[0], _unpack_bf16_49[1]);
        a_642[0] = fma_f32x2_rn_noftz(weight_649, v_651, a_642[0]);
        float2 v_652 = make_float2(_unpack_bf16_49[2], _unpack_bf16_49[3]);
        a_642[1] = fma_f32x2_rn_noftz(weight_649, v_652, a_642[1]);
        float2 weight_653 = make_float2(weights_544[6], weights_544[6]);
        unsigned int sw2_654[4];
        sw2_654[0] = words[84 + woff_640];
        sw2_654[1] = words[84 + woff_640 + 1];
        float _unpack_bf16_50[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_50[_pair * 2])[0]), "=f"((&_unpack_bf16_50[_pair * 2])[1])
                : "r"(sw2_654[_pair]));
        }
        float2 v_655 = make_float2(_unpack_bf16_50[0], _unpack_bf16_50[1]);
        a_642[0] = fma_f32x2_rn_noftz(weight_653, v_655, a_642[0]);
        float2 v_656 = make_float2(_unpack_bf16_50[2], _unpack_bf16_50[3]);
        a_642[1] = fma_f32x2_rn_noftz(weight_653, v_656, a_642[1]);
        float2 weight_657 = make_float2(weights_544[7], weights_544[7]);
        unsigned int sw2_658[4];
        sw2_658[0] = words[98 + woff_640];
        sw2_658[1] = words[98 + woff_640 + 1];
        float _unpack_bf16_51[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_51[_pair * 2])[0]), "=f"((&_unpack_bf16_51[_pair * 2])[1])
                : "r"(sw2_658[_pair]));
        }
        float2 v_659 = make_float2(_unpack_bf16_51[0], _unpack_bf16_51[1]);
        a_642[0] = fma_f32x2_rn_noftz(weight_657, v_659, a_642[0]);
        float2 v_660 = make_float2(_unpack_bf16_51[2], _unpack_bf16_51[3]);
        a_642[1] = fma_f32x2_rn_noftz(weight_657, v_660, a_642[1]);
        acc[24] = a_642[0].x;
        acc[25] = a_642[0].y;
        acc[26] = a_642[1].x;
        acc[27] = a_642[1].y;
        sum_running = __fmaf_rn(sum_running, correction_543, sum_weights_545);
        max_running = max_new_542;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_661 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_661, v_661, output_sq_pair);
        float2 v_662 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_662, v_662, output_sq_pair);
        float2 v_663 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_663, v_663, output_sq_pair);
        float2 v_664 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_664, v_664, output_sq_pair);
        float2 v_665 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_665, v_665, output_sq_pair);
        float2 v_666 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_666, v_666, output_sq_pair);
        float2 v_667 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_667, v_667, output_sq_pair);
        float2 v_668 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_668, v_668, output_sq_pair);
        float2 v_669 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_669, v_669, output_sq_pair);
        float2 v_670 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_670, v_670, output_sq_pair);
        float2 v_671 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_671, v_671, output_sq_pair);
        float2 v_672 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_672, v_672, output_sq_pair);
        float2 v_673 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_673, v_673, output_sq_pair);
        float2 v_674 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_674, v_674, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_107;
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_108;
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_109;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_110;
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_111;
        if (lane == 0) {
            out_stats[warp_0] = output_sq;
        }
        __syncthreads();
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_112 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_112;
        float _shfl_down_sync_113 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_113;
        float _shfl_down_sync_114 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_114;
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_115 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_115;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_675 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_116 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_117 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_117));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_676 = 2;
        const int acc_idx_677 = value_idx_676;
        float2 _f32x2_mul_a_118 = make_float2(acc[acc_idx_677], acc[acc_idx_677 + 1]);
        float2 scaled_pair_678;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_678) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_119 = make_float2(wout[acc_idx_677], wout[acc_idx_677 + 1]);
        float2 normalized_pair_679;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_679) : "l"(*(const unsigned long long*)&scaled_pair_678), "l"(*(const unsigned long long*)&_f32x2_mul_b_119));
        output_values[value_idx_676] = normalized_pair_679.x;
        output_values[value_idx_676 + 1] = normalized_pair_679.y;
        const int value_idx_680 = 4;
        const int acc_idx_681 = value_idx_680;
        float2 _f32x2_mul_a_120 = make_float2(acc[acc_idx_681], acc[acc_idx_681 + 1]);
        float2 scaled_pair_682;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_682) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_121 = make_float2(wout[acc_idx_681], wout[acc_idx_681 + 1]);
        float2 normalized_pair_683;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_683) : "l"(*(const unsigned long long*)&scaled_pair_682), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
        output_values[value_idx_680] = normalized_pair_683.x;
        output_values[value_idx_680 + 1] = normalized_pair_683.y;
        const int value_idx_684 = 6;
        const int acc_idx_685 = value_idx_684;
        float2 _f32x2_mul_a_122 = make_float2(acc[acc_idx_685], acc[acc_idx_685 + 1]);
        float2 scaled_pair_686;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_686) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_123 = make_float2(wout[acc_idx_685], wout[acc_idx_685 + 1]);
        float2 normalized_pair_687;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_687) : "l"(*(const unsigned long long*)&scaled_pair_686), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
        output_values[value_idx_684] = normalized_pair_687.x;
        output_values[value_idx_684 + 1] = normalized_pair_687.y;
        unsigned int _packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 0], output_values[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 2], output_values[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 4], output_values[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 6], output_values[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_675))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_688 = (2 + group) * 1024 + thread * 8;
        float output_values_689[8];
        const int value_idx_690 = 0;
        const int acc_idx_691 = 8 + value_idx_690;
        float2 _f32x2_mul_a_124 = make_float2(acc[acc_idx_691], acc[acc_idx_691 + 1]);
        float2 scaled_pair_692;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_692) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_125 = make_float2(wout[acc_idx_691], wout[acc_idx_691 + 1]);
        float2 normalized_pair_693;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_693) : "l"(*(const unsigned long long*)&scaled_pair_692), "l"(*(const unsigned long long*)&_f32x2_mul_b_125));
        output_values_689[value_idx_690] = normalized_pair_693.x;
        output_values_689[value_idx_690 + 1] = normalized_pair_693.y;
        const int value_idx_694 = 2;
        const int acc_idx_695 = 8 + value_idx_694;
        float2 _f32x2_mul_a_126 = make_float2(acc[acc_idx_695], acc[acc_idx_695 + 1]);
        float2 scaled_pair_696;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_696) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_127 = make_float2(wout[acc_idx_695], wout[acc_idx_695 + 1]);
        float2 normalized_pair_697;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_697) : "l"(*(const unsigned long long*)&scaled_pair_696), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
        output_values_689[value_idx_694] = normalized_pair_697.x;
        output_values_689[value_idx_694 + 1] = normalized_pair_697.y;
        const int value_idx_698 = 4;
        const int acc_idx_699 = 8 + value_idx_698;
        float2 _f32x2_mul_a_128 = make_float2(acc[acc_idx_699], acc[acc_idx_699 + 1]);
        float2 scaled_pair_700;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_700) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_129 = make_float2(wout[acc_idx_699], wout[acc_idx_699 + 1]);
        float2 normalized_pair_701;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_701) : "l"(*(const unsigned long long*)&scaled_pair_700), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
        output_values_689[value_idx_698] = normalized_pair_701.x;
        output_values_689[value_idx_698 + 1] = normalized_pair_701.y;
        const int value_idx_702 = 6;
        const int acc_idx_703 = 8 + value_idx_702;
        float2 _f32x2_mul_a_130 = make_float2(acc[acc_idx_703], acc[acc_idx_703 + 1]);
        float2 scaled_pair_704;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_704) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_131 = make_float2(wout[acc_idx_703], wout[acc_idx_703 + 1]);
        float2 normalized_pair_705;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_705) : "l"(*(const unsigned long long*)&scaled_pair_704), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
        output_values_689[value_idx_702] = normalized_pair_705.x;
        output_values_689[value_idx_702 + 1] = normalized_pair_705.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_689[_lp*2 + 0], output_values_689[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_689[_lp*2 + 2], output_values_689[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_689[_lp*2 + 4], output_values_689[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_689[_lp*2 + 6], output_values_689[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_688))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_706 = (4 + group) * 1024 + thread * 8;
        float output_values_707[8];
        const int value_idx_708 = 0;
        const int acc_idx_709 = 16 + value_idx_708;
        float2 _f32x2_mul_a_132 = make_float2(acc[acc_idx_709], acc[acc_idx_709 + 1]);
        float2 scaled_pair_710;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_710) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_133 = make_float2(wout[acc_idx_709], wout[acc_idx_709 + 1]);
        float2 normalized_pair_711;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_711) : "l"(*(const unsigned long long*)&scaled_pair_710), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
        output_values_707[value_idx_708] = normalized_pair_711.x;
        output_values_707[value_idx_708 + 1] = normalized_pair_711.y;
        const int value_idx_712 = 2;
        const int acc_idx_713 = 16 + value_idx_712;
        float2 _f32x2_mul_a_134 = make_float2(acc[acc_idx_713], acc[acc_idx_713 + 1]);
        float2 scaled_pair_714;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_714) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_135 = make_float2(wout[acc_idx_713], wout[acc_idx_713 + 1]);
        float2 normalized_pair_715;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_715) : "l"(*(const unsigned long long*)&scaled_pair_714), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
        output_values_707[value_idx_712] = normalized_pair_715.x;
        output_values_707[value_idx_712 + 1] = normalized_pair_715.y;
        const int value_idx_716 = 4;
        const int acc_idx_717 = 16 + value_idx_716;
        float2 _f32x2_mul_a_136 = make_float2(acc[acc_idx_717], acc[acc_idx_717 + 1]);
        float2 scaled_pair_718;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_718) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_137 = make_float2(wout[acc_idx_717], wout[acc_idx_717 + 1]);
        float2 normalized_pair_719;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_719) : "l"(*(const unsigned long long*)&scaled_pair_718), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
        output_values_707[value_idx_716] = normalized_pair_719.x;
        output_values_707[value_idx_716 + 1] = normalized_pair_719.y;
        const int value_idx_720 = 6;
        const int acc_idx_721 = 16 + value_idx_720;
        float2 _f32x2_mul_a_138 = make_float2(acc[acc_idx_721], acc[acc_idx_721 + 1]);
        float2 scaled_pair_722;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_722) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_139 = make_float2(wout[acc_idx_721], wout[acc_idx_721 + 1]);
        float2 normalized_pair_723;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_723) : "l"(*(const unsigned long long*)&scaled_pair_722), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
        output_values_707[value_idx_720] = normalized_pair_723.x;
        output_values_707[value_idx_720 + 1] = normalized_pair_723.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_707[_lp*2 + 0], output_values_707[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_707[_lp*2 + 2], output_values_707[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_707[_lp*2 + 4], output_values_707[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_707[_lp*2 + 6], output_values_707[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_706))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_724 = 6144 + group * 512 + thread * 4;
        float output_values_725[8];
        const int value_idx_726 = 0;
        const int acc_idx_727 = 24 + value_idx_726;
        float2 _f32x2_mul_a_140 = make_float2(acc[acc_idx_727], acc[acc_idx_727 + 1]);
        float2 scaled_pair_728;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_728) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_141 = make_float2(wout[acc_idx_727], wout[acc_idx_727 + 1]);
        float2 normalized_pair_729;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_729) : "l"(*(const unsigned long long*)&scaled_pair_728), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
        output_values_725[value_idx_726] = normalized_pair_729.x;
        output_values_725[value_idx_726 + 1] = normalized_pair_729.y;
        const int value_idx_730 = 2;
        const int acc_idx_731 = 24 + value_idx_730;
        float2 _f32x2_mul_a_142 = make_float2(acc[acc_idx_731], acc[acc_idx_731 + 1]);
        float2 scaled_pair_732;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_732) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_143 = make_float2(wout[acc_idx_731], wout[acc_idx_731 + 1]);
        float2 normalized_pair_733;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_733) : "l"(*(const unsigned long long*)&scaled_pair_732), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
        output_values_725[value_idx_730] = normalized_pair_733.x;
        output_values_725[value_idx_730 + 1] = normalized_pair_733.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_725[_lp*2 + 0], output_values_725[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_725[_lp*2 + 2], output_values_725[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_724))[0] = reinterpret_cast<int2*>(_packed_3)[0];
    }
}

} // extern "C"
