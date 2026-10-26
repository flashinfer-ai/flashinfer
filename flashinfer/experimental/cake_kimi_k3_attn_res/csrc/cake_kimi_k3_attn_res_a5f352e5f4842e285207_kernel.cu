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
kernel_cake_kimi_k3_attn_res_a5f352e5f4842e285207(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
        float2 v_164 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_165 = make_float2(q[8], q[9]);
        sq[7] = fma_f32x2_rn_noftz(v_164, v_164, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_164, qp_165, dot[7]);
        float2 v_166 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_167 = make_float2(q[10], q[11]);
        sq[7] = fma_f32x2_rn_noftz(v_166, v_166, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_166, qp_167, dot[7]);
        float2 v_168 = make_float2(_unpack_bf16_15[4], _unpack_bf16_15[5]);
        float2 qp_169 = make_float2(q[12], q[13]);
        sq[7] = fma_f32x2_rn_noftz(v_168, v_168, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_168, qp_169, dot[7]);
        float2 v_170 = make_float2(_unpack_bf16_15[6], _unpack_bf16_15[7]);
        float2 qp_171 = make_float2(q[14], q[15]);
        sq[7] = fma_f32x2_rn_noftz(v_170, v_170, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_170, qp_171, dot[7]);
        int base_172 = (4 + group) * 1024 + thread * 8;
        const int woff_173 = 8;
        unsigned int sw_174[4];
        sw_174[0] = words[woff_173];
        sw_174[1] = words[woff_173 + 1];
        sw_174[2] = words[woff_173 + 2];
        sw_174[3] = words[woff_173 + 3];
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(sw_174[_pair]));
        }
        fsrc[16] = _unpack_bf16_16[0];
        fsrc[17] = _unpack_bf16_16[1];
        fsrc[18] = _unpack_bf16_16[2];
        fsrc[19] = _unpack_bf16_16[3];
        fsrc[20] = _unpack_bf16_16[4];
        fsrc[21] = _unpack_bf16_16[5];
        fsrc[22] = _unpack_bf16_16[6];
        fsrc[23] = _unpack_bf16_16[7];
        float2 v_175 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_176 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_175, v_175, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_175, qp_176, dot[0]);
        float2 v_177 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_178 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_177, v_177, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_177, qp_178, dot[0]);
        float2 v_179 = make_float2(_unpack_bf16_16[4], _unpack_bf16_16[5]);
        float2 qp_180 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_179, v_179, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_179, qp_180, dot[0]);
        float2 v_181 = make_float2(_unpack_bf16_16[6], _unpack_bf16_16[7]);
        float2 qp_182 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_181, v_181, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_181, qp_182, dot[0]);
        unsigned int sw_183[4];
        sw_183[0] = words[14 + woff_173];
        sw_183[1] = words[14 + woff_173 + 1];
        sw_183[2] = words[14 + woff_173 + 2];
        sw_183[3] = words[14 + woff_173 + 3];
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(sw_183[_pair]));
        }
        fsrc[44] = _unpack_bf16_17[0];
        fsrc[45] = _unpack_bf16_17[1];
        fsrc[46] = _unpack_bf16_17[2];
        fsrc[47] = _unpack_bf16_17[3];
        fsrc[48] = _unpack_bf16_17[4];
        fsrc[49] = _unpack_bf16_17[5];
        fsrc[50] = _unpack_bf16_17[6];
        fsrc[51] = _unpack_bf16_17[7];
        float2 v_184 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_185 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_184, v_184, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_184, qp_185, dot[1]);
        float2 v_186 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_187 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_186, v_186, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_186, qp_187, dot[1]);
        float2 v_188 = make_float2(_unpack_bf16_17[4], _unpack_bf16_17[5]);
        float2 qp_189 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_188, v_188, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_188, qp_189, dot[1]);
        float2 v_190 = make_float2(_unpack_bf16_17[6], _unpack_bf16_17[7]);
        float2 qp_191 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_190, v_190, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_190, qp_191, dot[1]);
        unsigned int sw_192[4];
        sw_192[0] = words[28 + woff_173];
        sw_192[1] = words[28 + woff_173 + 1];
        sw_192[2] = words[28 + woff_173 + 2];
        sw_192[3] = words[28 + woff_173 + 3];
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                : "r"(sw_192[_pair]));
        }
        fsrc[72] = _unpack_bf16_18[0];
        fsrc[73] = _unpack_bf16_18[1];
        fsrc[74] = _unpack_bf16_18[2];
        fsrc[75] = _unpack_bf16_18[3];
        fsrc[76] = _unpack_bf16_18[4];
        fsrc[77] = _unpack_bf16_18[5];
        fsrc[78] = _unpack_bf16_18[6];
        fsrc[79] = _unpack_bf16_18[7];
        float2 v_193 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_194 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_193, v_193, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_193, qp_194, dot[2]);
        float2 v_195 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_196 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_195, v_195, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_195, qp_196, dot[2]);
        float2 v_197 = make_float2(_unpack_bf16_18[4], _unpack_bf16_18[5]);
        float2 qp_198 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_197, v_197, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_197, qp_198, dot[2]);
        float2 v_199 = make_float2(_unpack_bf16_18[6], _unpack_bf16_18[7]);
        float2 qp_200 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_199, v_199, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_199, qp_200, dot[2]);
        unsigned int sw_201[4];
        sw_201[0] = words[42 + woff_173];
        sw_201[1] = words[42 + woff_173 + 1];
        sw_201[2] = words[42 + woff_173 + 2];
        sw_201[3] = words[42 + woff_173 + 3];
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                : "r"(sw_201[_pair]));
        }
        float2 v_202 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 qp_203 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_202, v_202, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_202, qp_203, dot[3]);
        float2 v_204 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_205 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_204, v_204, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_204, qp_205, dot[3]);
        float2 v_206 = make_float2(_unpack_bf16_19[4], _unpack_bf16_19[5]);
        float2 qp_207 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_206, v_206, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_206, qp_207, dot[3]);
        float2 v_208 = make_float2(_unpack_bf16_19[6], _unpack_bf16_19[7]);
        float2 qp_209 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_208, v_208, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_208, qp_209, dot[3]);
        unsigned int sw_210[4];
        sw_210[0] = words[56 + woff_173];
        sw_210[1] = words[56 + woff_173 + 1];
        sw_210[2] = words[56 + woff_173 + 2];
        sw_210[3] = words[56 + woff_173 + 3];
        float _unpack_bf16_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                : "r"(sw_210[_pair]));
        }
        float2 v_211 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        float2 qp_212 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_211, v_211, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_211, qp_212, dot[4]);
        float2 v_213 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        float2 qp_214 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_213, v_213, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_213, qp_214, dot[4]);
        float2 v_215 = make_float2(_unpack_bf16_20[4], _unpack_bf16_20[5]);
        float2 qp_216 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_215, v_215, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_215, qp_216, dot[4]);
        float2 v_217 = make_float2(_unpack_bf16_20[6], _unpack_bf16_20[7]);
        float2 qp_218 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_217, v_217, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_217, qp_218, dot[4]);
        unsigned int sw_219[4];
        sw_219[0] = words[70 + woff_173];
        sw_219[1] = words[70 + woff_173 + 1];
        sw_219[2] = words[70 + woff_173 + 2];
        sw_219[3] = words[70 + woff_173 + 3];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw_219[_pair]));
        }
        float2 v_220 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        float2 qp_221 = make_float2(q[16], q[17]);
        sq[5] = fma_f32x2_rn_noftz(v_220, v_220, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_220, qp_221, dot[5]);
        float2 v_222 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        float2 qp_223 = make_float2(q[18], q[19]);
        sq[5] = fma_f32x2_rn_noftz(v_222, v_222, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_222, qp_223, dot[5]);
        float2 v_224 = make_float2(_unpack_bf16_21[4], _unpack_bf16_21[5]);
        float2 qp_225 = make_float2(q[20], q[21]);
        sq[5] = fma_f32x2_rn_noftz(v_224, v_224, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_224, qp_225, dot[5]);
        float2 v_226 = make_float2(_unpack_bf16_21[6], _unpack_bf16_21[7]);
        float2 qp_227 = make_float2(q[22], q[23]);
        sq[5] = fma_f32x2_rn_noftz(v_226, v_226, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_226, qp_227, dot[5]);
        unsigned int sw_228[4];
        sw_228[0] = words[84 + woff_173];
        sw_228[1] = words[84 + woff_173 + 1];
        sw_228[2] = words[84 + woff_173 + 2];
        sw_228[3] = words[84 + woff_173 + 3];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw_228[_pair]));
        }
        float2 v_229 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        float2 qp_230 = make_float2(q[16], q[17]);
        sq[6] = fma_f32x2_rn_noftz(v_229, v_229, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_229, qp_230, dot[6]);
        float2 v_231 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        float2 qp_232 = make_float2(q[18], q[19]);
        sq[6] = fma_f32x2_rn_noftz(v_231, v_231, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_231, qp_232, dot[6]);
        float2 v_233 = make_float2(_unpack_bf16_22[4], _unpack_bf16_22[5]);
        float2 qp_234 = make_float2(q[20], q[21]);
        sq[6] = fma_f32x2_rn_noftz(v_233, v_233, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_233, qp_234, dot[6]);
        float2 v_235 = make_float2(_unpack_bf16_22[6], _unpack_bf16_22[7]);
        float2 qp_236 = make_float2(q[22], q[23]);
        sq[6] = fma_f32x2_rn_noftz(v_235, v_235, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_235, qp_236, dot[6]);
        unsigned int sw_237[4];
        sw_237[0] = words[98 + woff_173];
        sw_237[1] = words[98 + woff_173 + 1];
        sw_237[2] = words[98 + woff_173 + 2];
        sw_237[3] = words[98 + woff_173 + 3];
        __nv_bfloat162 a_238 = __as_bf16x2(sw_237[0]);
        __nv_bfloat162 d_239 = __as_bf16x2(dwords[woff_173]);
        __nv_bfloat162 mixed_240 = a_238 + d_239;
        sw_237[0] = __as_u32(mixed_240);
        words[98 + woff_173] = sw_237[0];
        __nv_bfloat162 a_241 = __as_bf16x2(sw_237[1]);
        __nv_bfloat162 d_242 = __as_bf16x2(dwords[woff_173 + 1]);
        __nv_bfloat162 mixed_243 = a_241 + d_242;
        sw_237[1] = __as_u32(mixed_243);
        words[98 + woff_173 + 1] = sw_237[1];
        __nv_bfloat162 a_244 = __as_bf16x2(sw_237[2]);
        __nv_bfloat162 d_245 = __as_bf16x2(dwords[woff_173 + 2]);
        __nv_bfloat162 mixed_246 = a_244 + d_245;
        sw_237[2] = __as_u32(mixed_246);
        words[98 + woff_173 + 2] = sw_237[2];
        __nv_bfloat162 a_247 = __as_bf16x2(sw_237[3]);
        __nv_bfloat162 d_248 = __as_bf16x2(dwords[woff_173 + 3]);
        __nv_bfloat162 mixed_249 = a_247 + d_248;
        sw_237[3] = __as_u32(mixed_249);
        words[98 + woff_173 + 3] = sw_237[3];
        {
            int4 _iv4 = make_int4(sw_237[0 + 0], sw_237[0 + 1], sw_237[0 + 2], sw_237[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_172)) + 0) = _iv4;
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
                : "r"(sw_237[_pair]));
        }
        float2 v_250 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        float2 qp_251 = make_float2(q[16], q[17]);
        sq[7] = fma_f32x2_rn_noftz(v_250, v_250, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_250, qp_251, dot[7]);
        float2 v_252 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        float2 qp_253 = make_float2(q[18], q[19]);
        sq[7] = fma_f32x2_rn_noftz(v_252, v_252, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_252, qp_253, dot[7]);
        float2 v_254 = make_float2(_unpack_bf16_23[4], _unpack_bf16_23[5]);
        float2 qp_255 = make_float2(q[20], q[21]);
        sq[7] = fma_f32x2_rn_noftz(v_254, v_254, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_254, qp_255, dot[7]);
        float2 v_256 = make_float2(_unpack_bf16_23[6], _unpack_bf16_23[7]);
        float2 qp_257 = make_float2(q[22], q[23]);
        sq[7] = fma_f32x2_rn_noftz(v_256, v_256, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_256, qp_257, dot[7]);
        int base_258 = 6144 + group * 512 + thread * 4;
        const int woff_259 = 12;
        unsigned int sw_260[4];
        sw_260[0] = words[woff_259];
        sw_260[1] = words[woff_259 + 1];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw_260[_pair]));
        }
        fsrc[24] = _unpack_bf16_24[0];
        fsrc[25] = _unpack_bf16_24[1];
        fsrc[26] = _unpack_bf16_24[2];
        fsrc[27] = _unpack_bf16_24[3];
        float2 v_261 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        sq[0] = fma_f32x2_rn_noftz(v_261, v_261, sq[0]);
        float2 v_262 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        sq[0] = fma_f32x2_rn_noftz(v_262, v_262, sq[0]);
        float2 v_263 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        float2 qp_264 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_263, qp_264, dot[0]);
        float2 v_265 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        float2 qp_266 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_265, qp_266, dot[0]);
        unsigned int sw_267[4];
        sw_267[0] = words[14 + woff_259];
        sw_267[1] = words[14 + woff_259 + 1];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw_267[_pair]));
        }
        fsrc[52] = _unpack_bf16_25[0];
        fsrc[53] = _unpack_bf16_25[1];
        fsrc[54] = _unpack_bf16_25[2];
        fsrc[55] = _unpack_bf16_25[3];
        float2 v_268 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        sq[1] = fma_f32x2_rn_noftz(v_268, v_268, sq[1]);
        float2 v_269 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        sq[1] = fma_f32x2_rn_noftz(v_269, v_269, sq[1]);
        float2 v_270 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        float2 qp_271 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_270, qp_271, dot[1]);
        float2 v_272 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        float2 qp_273 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_272, qp_273, dot[1]);
        unsigned int sw_274[4];
        sw_274[0] = words[28 + woff_259];
        sw_274[1] = words[28 + woff_259 + 1];
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw_274[_pair]));
        }
        fsrc[80] = _unpack_bf16_26[0];
        fsrc[81] = _unpack_bf16_26[1];
        fsrc[82] = _unpack_bf16_26[2];
        fsrc[83] = _unpack_bf16_26[3];
        float2 v_275 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        sq[2] = fma_f32x2_rn_noftz(v_275, v_275, sq[2]);
        float2 v_276 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        sq[2] = fma_f32x2_rn_noftz(v_276, v_276, sq[2]);
        float2 v_277 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        float2 qp_278 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_277, qp_278, dot[2]);
        float2 v_279 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        float2 qp_280 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_279, qp_280, dot[2]);
        unsigned int sw_281[4];
        sw_281[0] = words[42 + woff_259];
        sw_281[1] = words[42 + woff_259 + 1];
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw_281[_pair]));
        }
        float2 v_282 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        sq[3] = fma_f32x2_rn_noftz(v_282, v_282, sq[3]);
        float2 v_283 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        sq[3] = fma_f32x2_rn_noftz(v_283, v_283, sq[3]);
        float2 v_284 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        float2 qp_285 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_284, qp_285, dot[3]);
        float2 v_286 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        float2 qp_287 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_286, qp_287, dot[3]);
        unsigned int sw_288[4];
        sw_288[0] = words[56 + woff_259];
        sw_288[1] = words[56 + woff_259 + 1];
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                : "r"(sw_288[_pair]));
        }
        float2 v_289 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        sq[4] = fma_f32x2_rn_noftz(v_289, v_289, sq[4]);
        float2 v_290 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        sq[4] = fma_f32x2_rn_noftz(v_290, v_290, sq[4]);
        float2 v_291 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        float2 qp_292 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_291, qp_292, dot[4]);
        float2 v_293 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        float2 qp_294 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_293, qp_294, dot[4]);
        unsigned int sw_295[4];
        sw_295[0] = words[70 + woff_259];
        sw_295[1] = words[70 + woff_259 + 1];
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                : "r"(sw_295[_pair]));
        }
        float2 v_296 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        sq[5] = fma_f32x2_rn_noftz(v_296, v_296, sq[5]);
        float2 v_297 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        sq[5] = fma_f32x2_rn_noftz(v_297, v_297, sq[5]);
        float2 v_298 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        float2 qp_299 = make_float2(q[24], q[25]);
        dot[5] = fma_f32x2_rn_noftz(v_298, qp_299, dot[5]);
        float2 v_300 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        float2 qp_301 = make_float2(q[26], q[27]);
        dot[5] = fma_f32x2_rn_noftz(v_300, qp_301, dot[5]);
        unsigned int sw_302[4];
        sw_302[0] = words[84 + woff_259];
        sw_302[1] = words[84 + woff_259 + 1];
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                : "r"(sw_302[_pair]));
        }
        float2 v_303 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        sq[6] = fma_f32x2_rn_noftz(v_303, v_303, sq[6]);
        float2 v_304 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        sq[6] = fma_f32x2_rn_noftz(v_304, v_304, sq[6]);
        float2 v_305 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        float2 qp_306 = make_float2(q[24], q[25]);
        dot[6] = fma_f32x2_rn_noftz(v_305, qp_306, dot[6]);
        float2 v_307 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        float2 qp_308 = make_float2(q[26], q[27]);
        dot[6] = fma_f32x2_rn_noftz(v_307, qp_308, dot[6]);
        unsigned int sw_309[4];
        sw_309[0] = words[98 + woff_259];
        sw_309[1] = words[98 + woff_259 + 1];
        __nv_bfloat162 a_310 = __as_bf16x2(sw_309[0]);
        __nv_bfloat162 d_311 = __as_bf16x2(dwords[woff_259]);
        __nv_bfloat162 mixed_312 = a_310 + d_311;
        sw_309[0] = __as_u32(mixed_312);
        words[98 + woff_259] = sw_309[0];
        __nv_bfloat162 a_313 = __as_bf16x2(sw_309[1]);
        __nv_bfloat162 d_314 = __as_bf16x2(dwords[woff_259 + 1]);
        __nv_bfloat162 mixed_315 = a_313 + d_314;
        sw_309[1] = __as_u32(mixed_315);
        words[98 + woff_259 + 1] = sw_309[1];
        {
            int2 _iv2 = make_int2(sw_309[0 + 0], sw_309[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_258)) + 0) = _iv2;
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
                : "r"(sw_309[_pair]));
        }
        float2 v_316 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        sq[7] = fma_f32x2_rn_noftz(v_316, v_316, sq[7]);
        float2 v_317 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        sq[7] = fma_f32x2_rn_noftz(v_317, v_317, sq[7]);
        float2 v_318 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        float2 qp_319 = make_float2(q[24], q[25]);
        dot[7] = fma_f32x2_rn_noftz(v_318, qp_319, dot[7]);
        float2 v_320 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        float2 qp_321 = make_float2(q[26], q[27]);
        dot[7] = fma_f32x2_rn_noftz(v_320, qp_321, dot[7]);
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
        unsigned long long bits_322 = 0;
        float2 _bit_cast_src_40 = pairs[1];
        bits_322 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_40)[0];
        unsigned long long peerbits_323 = __shfl_xor_sync(0xFFFFFFFF, bits_322, 16);
        float2 peer_324 = make_float2(0.0f, 0.0f);
        peer_324 = reinterpret_cast<float2*>(&peerbits_323)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_324);
        unsigned long long bits_325 = 0;
        float2 _bit_cast_src_41 = pairs[2];
        bits_325 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_41)[0];
        unsigned long long peerbits_326 = __shfl_xor_sync(0xFFFFFFFF, bits_325, 16);
        float2 peer_327 = make_float2(0.0f, 0.0f);
        peer_327 = reinterpret_cast<float2*>(&peerbits_326)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_327);
        unsigned long long bits_328 = 0;
        float2 _bit_cast_src_42 = pairs[3];
        bits_328 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits_329 = __shfl_xor_sync(0xFFFFFFFF, bits_328, 16);
        float2 peer_330 = make_float2(0.0f, 0.0f);
        peer_330 = reinterpret_cast<float2*>(&peerbits_329)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_330);
        unsigned long long bits_331 = 0;
        float2 _bit_cast_src_43 = pairs[4];
        bits_331 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_332 = __shfl_xor_sync(0xFFFFFFFF, bits_331, 16);
        float2 peer_333 = make_float2(0.0f, 0.0f);
        peer_333 = reinterpret_cast<float2*>(&peerbits_332)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_333);
        unsigned long long bits_334 = 0;
        float2 _bit_cast_src_44 = pairs[5];
        bits_334 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_335 = __shfl_xor_sync(0xFFFFFFFF, bits_334, 16);
        float2 peer_336 = make_float2(0.0f, 0.0f);
        peer_336 = reinterpret_cast<float2*>(&peerbits_335)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_336);
        unsigned long long bits_337 = 0;
        float2 _bit_cast_src_45 = pairs[6];
        bits_337 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_338 = __shfl_xor_sync(0xFFFFFFFF, bits_337, 16);
        float2 peer_339 = make_float2(0.0f, 0.0f);
        peer_339 = reinterpret_cast<float2*>(&peerbits_338)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_339);
        unsigned long long bits_340 = 0;
        float2 _bit_cast_src_46 = pairs[7];
        bits_340 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_341 = __shfl_xor_sync(0xFFFFFFFF, bits_340, 16);
        float2 peer_342 = make_float2(0.0f, 0.0f);
        peer_342 = reinterpret_cast<float2*>(&peerbits_341)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_342);
        unsigned long long bits_343 = 0;
        float2 _bit_cast_src_47 = pairs[0];
        bits_343 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_344 = __shfl_xor_sync(0xFFFFFFFF, bits_343, 8);
        float2 peer_345 = make_float2(0.0f, 0.0f);
        peer_345 = reinterpret_cast<float2*>(&peerbits_344)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_345);
        unsigned long long bits_346 = 0;
        float2 _bit_cast_src_48 = pairs[1];
        bits_346 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_347 = __shfl_xor_sync(0xFFFFFFFF, bits_346, 8);
        float2 peer_348 = make_float2(0.0f, 0.0f);
        peer_348 = reinterpret_cast<float2*>(&peerbits_347)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_348);
        unsigned long long bits_349 = 0;
        float2 _bit_cast_src_49 = pairs[2];
        bits_349 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_350 = __shfl_xor_sync(0xFFFFFFFF, bits_349, 8);
        float2 peer_351 = make_float2(0.0f, 0.0f);
        peer_351 = reinterpret_cast<float2*>(&peerbits_350)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_351);
        unsigned long long bits_352 = 0;
        float2 _bit_cast_src_50 = pairs[3];
        bits_352 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_353 = __shfl_xor_sync(0xFFFFFFFF, bits_352, 8);
        float2 peer_354 = make_float2(0.0f, 0.0f);
        peer_354 = reinterpret_cast<float2*>(&peerbits_353)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_354);
        unsigned long long bits_355 = 0;
        float2 _bit_cast_src_51 = pairs[4];
        bits_355 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_356 = __shfl_xor_sync(0xFFFFFFFF, bits_355, 8);
        float2 peer_357 = make_float2(0.0f, 0.0f);
        peer_357 = reinterpret_cast<float2*>(&peerbits_356)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_357);
        unsigned long long bits_358 = 0;
        float2 _bit_cast_src_52 = pairs[5];
        bits_358 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_359 = __shfl_xor_sync(0xFFFFFFFF, bits_358, 8);
        float2 peer_360 = make_float2(0.0f, 0.0f);
        peer_360 = reinterpret_cast<float2*>(&peerbits_359)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_360);
        unsigned long long bits_361 = 0;
        float2 _bit_cast_src_53 = pairs[6];
        bits_361 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_362 = __shfl_xor_sync(0xFFFFFFFF, bits_361, 8);
        float2 peer_363 = make_float2(0.0f, 0.0f);
        peer_363 = reinterpret_cast<float2*>(&peerbits_362)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_363);
        unsigned long long bits_364 = 0;
        float2 _bit_cast_src_54 = pairs[7];
        bits_364 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_365 = __shfl_xor_sync(0xFFFFFFFF, bits_364, 8);
        float2 peer_366 = make_float2(0.0f, 0.0f);
        peer_366 = reinterpret_cast<float2*>(&peerbits_365)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_366);
        unsigned long long bits_367 = 0;
        float2 _bit_cast_src_55 = pairs[0];
        bits_367 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_55)[0];
        unsigned long long peerbits_368 = __shfl_xor_sync(0xFFFFFFFF, bits_367, 4);
        float2 peer_369 = make_float2(0.0f, 0.0f);
        peer_369 = reinterpret_cast<float2*>(&peerbits_368)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_369);
        unsigned long long bits_370 = 0;
        float2 _bit_cast_src_56 = pairs[1];
        bits_370 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_56)[0];
        unsigned long long peerbits_371 = __shfl_xor_sync(0xFFFFFFFF, bits_370, 4);
        float2 peer_372 = make_float2(0.0f, 0.0f);
        peer_372 = reinterpret_cast<float2*>(&peerbits_371)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_372);
        unsigned long long bits_373 = 0;
        float2 _bit_cast_src_57 = pairs[2];
        bits_373 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_57)[0];
        unsigned long long peerbits_374 = __shfl_xor_sync(0xFFFFFFFF, bits_373, 4);
        float2 peer_375 = make_float2(0.0f, 0.0f);
        peer_375 = reinterpret_cast<float2*>(&peerbits_374)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_375);
        unsigned long long bits_376 = 0;
        float2 _bit_cast_src_58 = pairs[3];
        bits_376 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_58)[0];
        unsigned long long peerbits_377 = __shfl_xor_sync(0xFFFFFFFF, bits_376, 4);
        float2 peer_378 = make_float2(0.0f, 0.0f);
        peer_378 = reinterpret_cast<float2*>(&peerbits_377)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_378);
        unsigned long long bits_379 = 0;
        float2 _bit_cast_src_59 = pairs[4];
        bits_379 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_59)[0];
        unsigned long long peerbits_380 = __shfl_xor_sync(0xFFFFFFFF, bits_379, 4);
        float2 peer_381 = make_float2(0.0f, 0.0f);
        peer_381 = reinterpret_cast<float2*>(&peerbits_380)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_381);
        unsigned long long bits_382 = 0;
        float2 _bit_cast_src_60 = pairs[5];
        bits_382 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_60)[0];
        unsigned long long peerbits_383 = __shfl_xor_sync(0xFFFFFFFF, bits_382, 4);
        float2 peer_384 = make_float2(0.0f, 0.0f);
        peer_384 = reinterpret_cast<float2*>(&peerbits_383)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_384);
        unsigned long long bits_385 = 0;
        float2 _bit_cast_src_61 = pairs[6];
        bits_385 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_61)[0];
        unsigned long long peerbits_386 = __shfl_xor_sync(0xFFFFFFFF, bits_385, 4);
        float2 peer_387 = make_float2(0.0f, 0.0f);
        peer_387 = reinterpret_cast<float2*>(&peerbits_386)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_387);
        unsigned long long bits_388 = 0;
        float2 _bit_cast_src_62 = pairs[7];
        bits_388 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_62)[0];
        unsigned long long peerbits_389 = __shfl_xor_sync(0xFFFFFFFF, bits_388, 4);
        float2 peer_390 = make_float2(0.0f, 0.0f);
        peer_390 = reinterpret_cast<float2*>(&peerbits_389)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_390);
        unsigned long long bits_391 = 0;
        float2 _bit_cast_src_63 = pairs[0];
        bits_391 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_63)[0];
        unsigned long long peerbits_392 = __shfl_xor_sync(0xFFFFFFFF, bits_391, 2);
        float2 peer_393 = make_float2(0.0f, 0.0f);
        peer_393 = reinterpret_cast<float2*>(&peerbits_392)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_393);
        unsigned long long bits_394 = 0;
        float2 _bit_cast_src_64 = pairs[1];
        bits_394 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_64)[0];
        unsigned long long peerbits_395 = __shfl_xor_sync(0xFFFFFFFF, bits_394, 2);
        float2 peer_396 = make_float2(0.0f, 0.0f);
        peer_396 = reinterpret_cast<float2*>(&peerbits_395)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_396);
        unsigned long long bits_397 = 0;
        float2 _bit_cast_src_65 = pairs[2];
        bits_397 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_65)[0];
        unsigned long long peerbits_398 = __shfl_xor_sync(0xFFFFFFFF, bits_397, 2);
        float2 peer_399 = make_float2(0.0f, 0.0f);
        peer_399 = reinterpret_cast<float2*>(&peerbits_398)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_399);
        unsigned long long bits_400 = 0;
        float2 _bit_cast_src_66 = pairs[3];
        bits_400 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_66)[0];
        unsigned long long peerbits_401 = __shfl_xor_sync(0xFFFFFFFF, bits_400, 2);
        float2 peer_402 = make_float2(0.0f, 0.0f);
        peer_402 = reinterpret_cast<float2*>(&peerbits_401)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_402);
        unsigned long long bits_403 = 0;
        float2 _bit_cast_src_67 = pairs[4];
        bits_403 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_67)[0];
        unsigned long long peerbits_404 = __shfl_xor_sync(0xFFFFFFFF, bits_403, 2);
        float2 peer_405 = make_float2(0.0f, 0.0f);
        peer_405 = reinterpret_cast<float2*>(&peerbits_404)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_405);
        unsigned long long bits_406 = 0;
        float2 _bit_cast_src_68 = pairs[5];
        bits_406 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_68)[0];
        unsigned long long peerbits_407 = __shfl_xor_sync(0xFFFFFFFF, bits_406, 2);
        float2 peer_408 = make_float2(0.0f, 0.0f);
        peer_408 = reinterpret_cast<float2*>(&peerbits_407)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_408);
        unsigned long long bits_409 = 0;
        float2 _bit_cast_src_69 = pairs[6];
        bits_409 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_69)[0];
        unsigned long long peerbits_410 = __shfl_xor_sync(0xFFFFFFFF, bits_409, 2);
        float2 peer_411 = make_float2(0.0f, 0.0f);
        peer_411 = reinterpret_cast<float2*>(&peerbits_410)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_411);
        unsigned long long bits_412 = 0;
        float2 _bit_cast_src_70 = pairs[7];
        bits_412 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_70)[0];
        unsigned long long peerbits_413 = __shfl_xor_sync(0xFFFFFFFF, bits_412, 2);
        float2 peer_414 = make_float2(0.0f, 0.0f);
        peer_414 = reinterpret_cast<float2*>(&peerbits_413)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_414);
        unsigned long long bits_415 = 0;
        float2 _bit_cast_src_71 = pairs[0];
        bits_415 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_71)[0];
        unsigned long long peerbits_416 = __shfl_xor_sync(0xFFFFFFFF, bits_415, 1);
        float2 peer_417 = make_float2(0.0f, 0.0f);
        peer_417 = reinterpret_cast<float2*>(&peerbits_416)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_417);
        unsigned long long bits_418 = 0;
        float2 _bit_cast_src_72 = pairs[1];
        bits_418 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_72)[0];
        unsigned long long peerbits_419 = __shfl_xor_sync(0xFFFFFFFF, bits_418, 1);
        float2 peer_420 = make_float2(0.0f, 0.0f);
        peer_420 = reinterpret_cast<float2*>(&peerbits_419)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_420);
        unsigned long long bits_421 = 0;
        float2 _bit_cast_src_73 = pairs[2];
        bits_421 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_73)[0];
        unsigned long long peerbits_422 = __shfl_xor_sync(0xFFFFFFFF, bits_421, 1);
        float2 peer_423 = make_float2(0.0f, 0.0f);
        peer_423 = reinterpret_cast<float2*>(&peerbits_422)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_423);
        unsigned long long bits_424 = 0;
        float2 _bit_cast_src_74 = pairs[3];
        bits_424 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_74)[0];
        unsigned long long peerbits_425 = __shfl_xor_sync(0xFFFFFFFF, bits_424, 1);
        float2 peer_426 = make_float2(0.0f, 0.0f);
        peer_426 = reinterpret_cast<float2*>(&peerbits_425)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_426);
        unsigned long long bits_427 = 0;
        float2 _bit_cast_src_75 = pairs[4];
        bits_427 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_75)[0];
        unsigned long long peerbits_428 = __shfl_xor_sync(0xFFFFFFFF, bits_427, 1);
        float2 peer_429 = make_float2(0.0f, 0.0f);
        peer_429 = reinterpret_cast<float2*>(&peerbits_428)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_429);
        unsigned long long bits_430 = 0;
        float2 _bit_cast_src_76 = pairs[5];
        bits_430 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_76)[0];
        unsigned long long peerbits_431 = __shfl_xor_sync(0xFFFFFFFF, bits_430, 1);
        float2 peer_432 = make_float2(0.0f, 0.0f);
        peer_432 = reinterpret_cast<float2*>(&peerbits_431)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_432);
        unsigned long long bits_433 = 0;
        float2 _bit_cast_src_77 = pairs[6];
        bits_433 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_77)[0];
        unsigned long long peerbits_434 = __shfl_xor_sync(0xFFFFFFFF, bits_433, 1);
        float2 peer_435 = make_float2(0.0f, 0.0f);
        peer_435 = reinterpret_cast<float2*>(&peerbits_434)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_435);
        unsigned long long bits_436 = 0;
        float2 _bit_cast_src_78 = pairs[7];
        bits_436 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_78)[0];
        unsigned long long peerbits_437 = __shfl_xor_sync(0xFFFFFFFF, bits_436, 1);
        float2 peer_438 = make_float2(0.0f, 0.0f);
        peer_438 = reinterpret_cast<float2*>(&peerbits_437)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_438);
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
        const int woff_439 = 0;
        int base_440 = group * 1024 + thread * 8;
        float2 a_441[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_441[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_442 = make_float2(acc[2], acc[3]);
        a_441[1] = mul_f32x2_noftz(previous_442, corr);
        float2 previous_443 = make_float2(acc[4], acc[5]);
        a_441[2] = mul_f32x2_noftz(previous_443, corr);
        float2 previous_444 = make_float2(acc[6], acc[7]);
        a_441[3] = mul_f32x2_noftz(previous_444, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_445 = make_float2(fsrc[0], fsrc[1]);
        a_441[0] = fma_f32x2_rn_noftz(weight, v_445, a_441[0]);
        float2 v_446 = make_float2(fsrc[2], fsrc[3]);
        a_441[1] = fma_f32x2_rn_noftz(weight, v_446, a_441[1]);
        float2 v_447 = make_float2(fsrc[4], fsrc[5]);
        a_441[2] = fma_f32x2_rn_noftz(weight, v_447, a_441[2]);
        float2 v_448 = make_float2(fsrc[6], fsrc[7]);
        a_441[3] = fma_f32x2_rn_noftz(weight, v_448, a_441[3]);
        float2 weight_449 = make_float2(weights[1], weights[1]);
        float2 v_450 = make_float2(fsrc[28], fsrc[29]);
        a_441[0] = fma_f32x2_rn_noftz(weight_449, v_450, a_441[0]);
        float2 v_451 = make_float2(fsrc[30], fsrc[31]);
        a_441[1] = fma_f32x2_rn_noftz(weight_449, v_451, a_441[1]);
        float2 v_452 = make_float2(fsrc[32], fsrc[33]);
        a_441[2] = fma_f32x2_rn_noftz(weight_449, v_452, a_441[2]);
        float2 v_453 = make_float2(fsrc[34], fsrc[35]);
        a_441[3] = fma_f32x2_rn_noftz(weight_449, v_453, a_441[3]);
        float2 weight_454 = make_float2(weights[2], weights[2]);
        float2 v_455 = make_float2(fsrc[56], fsrc[57]);
        a_441[0] = fma_f32x2_rn_noftz(weight_454, v_455, a_441[0]);
        float2 v_456 = make_float2(fsrc[58], fsrc[59]);
        a_441[1] = fma_f32x2_rn_noftz(weight_454, v_456, a_441[1]);
        float2 v_457 = make_float2(fsrc[60], fsrc[61]);
        a_441[2] = fma_f32x2_rn_noftz(weight_454, v_457, a_441[2]);
        float2 v_458 = make_float2(fsrc[62], fsrc[63]);
        a_441[3] = fma_f32x2_rn_noftz(weight_454, v_458, a_441[3]);
        float2 weight_459 = make_float2(weights[3], weights[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_439];
        sw2[1] = words[42 + woff_439 + 1];
        sw2[2] = words[42 + woff_439 + 2];
        sw2[3] = words[42 + woff_439 + 3];
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
        float2 v_460 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
        a_441[0] = fma_f32x2_rn_noftz(weight_459, v_460, a_441[0]);
        float2 v_461 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
        a_441[1] = fma_f32x2_rn_noftz(weight_459, v_461, a_441[1]);
        float2 v_462 = make_float2(_unpack_bf16_32[4], _unpack_bf16_32[5]);
        a_441[2] = fma_f32x2_rn_noftz(weight_459, v_462, a_441[2]);
        float2 v_463 = make_float2(_unpack_bf16_32[6], _unpack_bf16_32[7]);
        a_441[3] = fma_f32x2_rn_noftz(weight_459, v_463, a_441[3]);
        acc[0] = a_441[0].x;
        acc[1] = a_441[0].y;
        acc[2] = a_441[1].x;
        acc[3] = a_441[1].y;
        acc[4] = a_441[2].x;
        acc[5] = a_441[2].y;
        acc[6] = a_441[3].x;
        acc[7] = a_441[3].y;
        const int woff_464 = 4;
        int base_465 = (2 + group) * 1024 + thread * 8;
        float2 a_466[4];
        float2 previous_467 = make_float2(acc[8], acc[9]);
        a_466[0] = mul_f32x2_noftz(previous_467, corr);
        float2 previous_468 = make_float2(acc[10], acc[11]);
        a_466[1] = mul_f32x2_noftz(previous_468, corr);
        float2 previous_469 = make_float2(acc[12], acc[13]);
        a_466[2] = mul_f32x2_noftz(previous_469, corr);
        float2 previous_470 = make_float2(acc[14], acc[15]);
        a_466[3] = mul_f32x2_noftz(previous_470, corr);
        float2 weight_471 = make_float2(weights[0], weights[0]);
        float2 v_472 = make_float2(fsrc[8], fsrc[9]);
        a_466[0] = fma_f32x2_rn_noftz(weight_471, v_472, a_466[0]);
        float2 v_473 = make_float2(fsrc[10], fsrc[11]);
        a_466[1] = fma_f32x2_rn_noftz(weight_471, v_473, a_466[1]);
        float2 v_474 = make_float2(fsrc[12], fsrc[13]);
        a_466[2] = fma_f32x2_rn_noftz(weight_471, v_474, a_466[2]);
        float2 v_475 = make_float2(fsrc[14], fsrc[15]);
        a_466[3] = fma_f32x2_rn_noftz(weight_471, v_475, a_466[3]);
        float2 weight_476 = make_float2(weights[1], weights[1]);
        float2 v_477 = make_float2(fsrc[36], fsrc[37]);
        a_466[0] = fma_f32x2_rn_noftz(weight_476, v_477, a_466[0]);
        float2 v_478 = make_float2(fsrc[38], fsrc[39]);
        a_466[1] = fma_f32x2_rn_noftz(weight_476, v_478, a_466[1]);
        float2 v_479 = make_float2(fsrc[40], fsrc[41]);
        a_466[2] = fma_f32x2_rn_noftz(weight_476, v_479, a_466[2]);
        float2 v_480 = make_float2(fsrc[42], fsrc[43]);
        a_466[3] = fma_f32x2_rn_noftz(weight_476, v_480, a_466[3]);
        float2 weight_481 = make_float2(weights[2], weights[2]);
        float2 v_482 = make_float2(fsrc[64], fsrc[65]);
        a_466[0] = fma_f32x2_rn_noftz(weight_481, v_482, a_466[0]);
        float2 v_483 = make_float2(fsrc[66], fsrc[67]);
        a_466[1] = fma_f32x2_rn_noftz(weight_481, v_483, a_466[1]);
        float2 v_484 = make_float2(fsrc[68], fsrc[69]);
        a_466[2] = fma_f32x2_rn_noftz(weight_481, v_484, a_466[2]);
        float2 v_485 = make_float2(fsrc[70], fsrc[71]);
        a_466[3] = fma_f32x2_rn_noftz(weight_481, v_485, a_466[3]);
        float2 weight_486 = make_float2(weights[3], weights[3]);
        unsigned int sw2_487[4];
        sw2_487[0] = words[42 + woff_464];
        sw2_487[1] = words[42 + woff_464 + 1];
        sw2_487[2] = words[42 + woff_464 + 2];
        sw2_487[3] = words[42 + woff_464 + 3];
        float _unpack_bf16_33[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                : "r"(sw2_487[_pair]));
        }
        float2 v_488 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
        a_466[0] = fma_f32x2_rn_noftz(weight_486, v_488, a_466[0]);
        float2 v_489 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
        a_466[1] = fma_f32x2_rn_noftz(weight_486, v_489, a_466[1]);
        float2 v_490 = make_float2(_unpack_bf16_33[4], _unpack_bf16_33[5]);
        a_466[2] = fma_f32x2_rn_noftz(weight_486, v_490, a_466[2]);
        float2 v_491 = make_float2(_unpack_bf16_33[6], _unpack_bf16_33[7]);
        a_466[3] = fma_f32x2_rn_noftz(weight_486, v_491, a_466[3]);
        acc[8] = a_466[0].x;
        acc[9] = a_466[0].y;
        acc[10] = a_466[1].x;
        acc[11] = a_466[1].y;
        acc[12] = a_466[2].x;
        acc[13] = a_466[2].y;
        acc[14] = a_466[3].x;
        acc[15] = a_466[3].y;
        const int woff_492 = 8;
        int base_493 = (4 + group) * 1024 + thread * 8;
        float2 a_494[4];
        float2 previous_495 = make_float2(acc[16], acc[17]);
        a_494[0] = mul_f32x2_noftz(previous_495, corr);
        float2 previous_496 = make_float2(acc[18], acc[19]);
        a_494[1] = mul_f32x2_noftz(previous_496, corr);
        float2 previous_497 = make_float2(acc[20], acc[21]);
        a_494[2] = mul_f32x2_noftz(previous_497, corr);
        float2 previous_498 = make_float2(acc[22], acc[23]);
        a_494[3] = mul_f32x2_noftz(previous_498, corr);
        float2 weight_499 = make_float2(weights[0], weights[0]);
        float2 v_500 = make_float2(fsrc[16], fsrc[17]);
        a_494[0] = fma_f32x2_rn_noftz(weight_499, v_500, a_494[0]);
        float2 v_501 = make_float2(fsrc[18], fsrc[19]);
        a_494[1] = fma_f32x2_rn_noftz(weight_499, v_501, a_494[1]);
        float2 v_502 = make_float2(fsrc[20], fsrc[21]);
        a_494[2] = fma_f32x2_rn_noftz(weight_499, v_502, a_494[2]);
        float2 v_503 = make_float2(fsrc[22], fsrc[23]);
        a_494[3] = fma_f32x2_rn_noftz(weight_499, v_503, a_494[3]);
        float2 weight_504 = make_float2(weights[1], weights[1]);
        float2 v_505 = make_float2(fsrc[44], fsrc[45]);
        a_494[0] = fma_f32x2_rn_noftz(weight_504, v_505, a_494[0]);
        float2 v_506 = make_float2(fsrc[46], fsrc[47]);
        a_494[1] = fma_f32x2_rn_noftz(weight_504, v_506, a_494[1]);
        float2 v_507 = make_float2(fsrc[48], fsrc[49]);
        a_494[2] = fma_f32x2_rn_noftz(weight_504, v_507, a_494[2]);
        float2 v_508 = make_float2(fsrc[50], fsrc[51]);
        a_494[3] = fma_f32x2_rn_noftz(weight_504, v_508, a_494[3]);
        float2 weight_509 = make_float2(weights[2], weights[2]);
        float2 v_510 = make_float2(fsrc[72], fsrc[73]);
        a_494[0] = fma_f32x2_rn_noftz(weight_509, v_510, a_494[0]);
        float2 v_511 = make_float2(fsrc[74], fsrc[75]);
        a_494[1] = fma_f32x2_rn_noftz(weight_509, v_511, a_494[1]);
        float2 v_512 = make_float2(fsrc[76], fsrc[77]);
        a_494[2] = fma_f32x2_rn_noftz(weight_509, v_512, a_494[2]);
        float2 v_513 = make_float2(fsrc[78], fsrc[79]);
        a_494[3] = fma_f32x2_rn_noftz(weight_509, v_513, a_494[3]);
        float2 weight_514 = make_float2(weights[3], weights[3]);
        unsigned int sw2_515[4];
        sw2_515[0] = words[42 + woff_492];
        sw2_515[1] = words[42 + woff_492 + 1];
        sw2_515[2] = words[42 + woff_492 + 2];
        sw2_515[3] = words[42 + woff_492 + 3];
        float _unpack_bf16_34[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                : "r"(sw2_515[_pair]));
        }
        float2 v_516 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
        a_494[0] = fma_f32x2_rn_noftz(weight_514, v_516, a_494[0]);
        float2 v_517 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
        a_494[1] = fma_f32x2_rn_noftz(weight_514, v_517, a_494[1]);
        float2 v_518 = make_float2(_unpack_bf16_34[4], _unpack_bf16_34[5]);
        a_494[2] = fma_f32x2_rn_noftz(weight_514, v_518, a_494[2]);
        float2 v_519 = make_float2(_unpack_bf16_34[6], _unpack_bf16_34[7]);
        a_494[3] = fma_f32x2_rn_noftz(weight_514, v_519, a_494[3]);
        acc[16] = a_494[0].x;
        acc[17] = a_494[0].y;
        acc[18] = a_494[1].x;
        acc[19] = a_494[1].y;
        acc[20] = a_494[2].x;
        acc[21] = a_494[2].y;
        acc[22] = a_494[3].x;
        acc[23] = a_494[3].y;
        const int woff_520 = 12;
        int base_521 = 6144 + group * 512 + thread * 4;
        float2 a_522[4];
        float2 previous_523 = make_float2(acc[24], acc[25]);
        a_522[0] = mul_f32x2_noftz(previous_523, corr);
        float2 previous_524 = make_float2(acc[26], acc[27]);
        a_522[1] = mul_f32x2_noftz(previous_524, corr);
        float2 weight_525 = make_float2(weights[0], weights[0]);
        float2 v_526 = make_float2(fsrc[24], fsrc[25]);
        a_522[0] = fma_f32x2_rn_noftz(weight_525, v_526, a_522[0]);
        float2 v_527 = make_float2(fsrc[26], fsrc[27]);
        a_522[1] = fma_f32x2_rn_noftz(weight_525, v_527, a_522[1]);
        float2 weight_528 = make_float2(weights[1], weights[1]);
        float2 v_529 = make_float2(fsrc[52], fsrc[53]);
        a_522[0] = fma_f32x2_rn_noftz(weight_528, v_529, a_522[0]);
        float2 v_530 = make_float2(fsrc[54], fsrc[55]);
        a_522[1] = fma_f32x2_rn_noftz(weight_528, v_530, a_522[1]);
        float2 weight_531 = make_float2(weights[2], weights[2]);
        float2 v_532 = make_float2(fsrc[80], fsrc[81]);
        a_522[0] = fma_f32x2_rn_noftz(weight_531, v_532, a_522[0]);
        float2 v_533 = make_float2(fsrc[82], fsrc[83]);
        a_522[1] = fma_f32x2_rn_noftz(weight_531, v_533, a_522[1]);
        float2 weight_534 = make_float2(weights[3], weights[3]);
        unsigned int sw2_535[4];
        sw2_535[0] = words[42 + woff_520];
        sw2_535[1] = words[42 + woff_520 + 1];
        float _unpack_bf16_35[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                : "r"(sw2_535[_pair]));
        }
        float2 v_536 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
        a_522[0] = fma_f32x2_rn_noftz(weight_534, v_536, a_522[0]);
        float2 v_537 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
        a_522[1] = fma_f32x2_rn_noftz(weight_534, v_537, a_522[1]);
        acc[24] = a_522[0].x;
        acc[25] = a_522[0].y;
        acc[26] = a_522[1].x;
        acc[27] = a_522[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_538 = -3.4028235e+38f;
        max_chunk_538 = fmaxf(max_chunk_538, logits[4]);
        max_chunk_538 = fmaxf(max_chunk_538, logits[5]);
        max_chunk_538 = fmaxf(max_chunk_538, logits[6]);
        max_chunk_538 = fmaxf(max_chunk_538, logits[7]);
        float max_new_539 = fmaxf(max_running, max_chunk_538);
        float correction_540 = approx_exp2((max_running - max_new_539) * 1.442695f);
        float weights_541[8];
        float sum_weights_542 = 0.0f;
        float _exp2_103 = approx_exp2((logits[4] - max_new_539) * 1.442695f);
        weights_541[4] = _exp2_103;
        sum_weights_542 += weights_541[4];
        float _exp2_104 = approx_exp2((logits[5] - max_new_539) * 1.442695f);
        weights_541[5] = _exp2_104;
        sum_weights_542 += weights_541[5];
        float _exp2_105 = approx_exp2((logits[6] - max_new_539) * 1.442695f);
        weights_541[6] = _exp2_105;
        sum_weights_542 += weights_541[6];
        float _exp2_106 = approx_exp2((logits[7] - max_new_539) * 1.442695f);
        weights_541[7] = _exp2_106;
        sum_weights_542 += weights_541[7];
        float2 corr_543 = make_float2(correction_540, correction_540);
        const int woff_544 = 0;
        int base_545 = group * 1024 + thread * 8;
        float2 a_546[4];
        float2 previous_547 = make_float2(acc[0], acc[1]);
        a_546[0] = mul_f32x2_noftz(previous_547, corr_543);
        float2 previous_548 = make_float2(acc[2], acc[3]);
        a_546[1] = mul_f32x2_noftz(previous_548, corr_543);
        float2 previous_549 = make_float2(acc[4], acc[5]);
        a_546[2] = mul_f32x2_noftz(previous_549, corr_543);
        float2 previous_550 = make_float2(acc[6], acc[7]);
        a_546[3] = mul_f32x2_noftz(previous_550, corr_543);
        float2 weight_551 = make_float2(weights_541[4], weights_541[4]);
        unsigned int sw2_552[4];
        sw2_552[0] = words[56 + woff_544];
        sw2_552[1] = words[56 + woff_544 + 1];
        sw2_552[2] = words[56 + woff_544 + 2];
        sw2_552[3] = words[56 + woff_544 + 3];
        float _unpack_bf16_36[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                : "r"(sw2_552[_pair]));
        }
        float2 v_553 = make_float2(_unpack_bf16_36[0], _unpack_bf16_36[1]);
        a_546[0] = fma_f32x2_rn_noftz(weight_551, v_553, a_546[0]);
        float2 v_554 = make_float2(_unpack_bf16_36[2], _unpack_bf16_36[3]);
        a_546[1] = fma_f32x2_rn_noftz(weight_551, v_554, a_546[1]);
        float2 v_555 = make_float2(_unpack_bf16_36[4], _unpack_bf16_36[5]);
        a_546[2] = fma_f32x2_rn_noftz(weight_551, v_555, a_546[2]);
        float2 v_556 = make_float2(_unpack_bf16_36[6], _unpack_bf16_36[7]);
        a_546[3] = fma_f32x2_rn_noftz(weight_551, v_556, a_546[3]);
        float2 weight_557 = make_float2(weights_541[5], weights_541[5]);
        unsigned int sw2_558[4];
        sw2_558[0] = words[70 + woff_544];
        sw2_558[1] = words[70 + woff_544 + 1];
        sw2_558[2] = words[70 + woff_544 + 2];
        sw2_558[3] = words[70 + woff_544 + 3];
        float _unpack_bf16_37[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                : "r"(sw2_558[_pair]));
        }
        float2 v_559 = make_float2(_unpack_bf16_37[0], _unpack_bf16_37[1]);
        a_546[0] = fma_f32x2_rn_noftz(weight_557, v_559, a_546[0]);
        float2 v_560 = make_float2(_unpack_bf16_37[2], _unpack_bf16_37[3]);
        a_546[1] = fma_f32x2_rn_noftz(weight_557, v_560, a_546[1]);
        float2 v_561 = make_float2(_unpack_bf16_37[4], _unpack_bf16_37[5]);
        a_546[2] = fma_f32x2_rn_noftz(weight_557, v_561, a_546[2]);
        float2 v_562 = make_float2(_unpack_bf16_37[6], _unpack_bf16_37[7]);
        a_546[3] = fma_f32x2_rn_noftz(weight_557, v_562, a_546[3]);
        float2 weight_563 = make_float2(weights_541[6], weights_541[6]);
        unsigned int sw2_564[4];
        sw2_564[0] = words[84 + woff_544];
        sw2_564[1] = words[84 + woff_544 + 1];
        sw2_564[2] = words[84 + woff_544 + 2];
        sw2_564[3] = words[84 + woff_544 + 3];
        float _unpack_bf16_38[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                : "r"(sw2_564[_pair]));
        }
        float2 v_565 = make_float2(_unpack_bf16_38[0], _unpack_bf16_38[1]);
        a_546[0] = fma_f32x2_rn_noftz(weight_563, v_565, a_546[0]);
        float2 v_566 = make_float2(_unpack_bf16_38[2], _unpack_bf16_38[3]);
        a_546[1] = fma_f32x2_rn_noftz(weight_563, v_566, a_546[1]);
        float2 v_567 = make_float2(_unpack_bf16_38[4], _unpack_bf16_38[5]);
        a_546[2] = fma_f32x2_rn_noftz(weight_563, v_567, a_546[2]);
        float2 v_568 = make_float2(_unpack_bf16_38[6], _unpack_bf16_38[7]);
        a_546[3] = fma_f32x2_rn_noftz(weight_563, v_568, a_546[3]);
        float2 weight_569 = make_float2(weights_541[7], weights_541[7]);
        unsigned int sw2_570[4];
        sw2_570[0] = words[98 + woff_544];
        sw2_570[1] = words[98 + woff_544 + 1];
        sw2_570[2] = words[98 + woff_544 + 2];
        sw2_570[3] = words[98 + woff_544 + 3];
        float _unpack_bf16_39[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                : "r"(sw2_570[_pair]));
        }
        float2 v_571 = make_float2(_unpack_bf16_39[0], _unpack_bf16_39[1]);
        a_546[0] = fma_f32x2_rn_noftz(weight_569, v_571, a_546[0]);
        float2 v_572 = make_float2(_unpack_bf16_39[2], _unpack_bf16_39[3]);
        a_546[1] = fma_f32x2_rn_noftz(weight_569, v_572, a_546[1]);
        float2 v_573 = make_float2(_unpack_bf16_39[4], _unpack_bf16_39[5]);
        a_546[2] = fma_f32x2_rn_noftz(weight_569, v_573, a_546[2]);
        float2 v_574 = make_float2(_unpack_bf16_39[6], _unpack_bf16_39[7]);
        a_546[3] = fma_f32x2_rn_noftz(weight_569, v_574, a_546[3]);
        acc[0] = a_546[0].x;
        acc[1] = a_546[0].y;
        acc[2] = a_546[1].x;
        acc[3] = a_546[1].y;
        acc[4] = a_546[2].x;
        acc[5] = a_546[2].y;
        acc[6] = a_546[3].x;
        acc[7] = a_546[3].y;
        const int woff_575 = 4;
        int base_576 = (2 + group) * 1024 + thread * 8;
        float2 a_577[4];
        float2 previous_578 = make_float2(acc[8], acc[9]);
        a_577[0] = mul_f32x2_noftz(previous_578, corr_543);
        float2 previous_579 = make_float2(acc[10], acc[11]);
        a_577[1] = mul_f32x2_noftz(previous_579, corr_543);
        float2 previous_580 = make_float2(acc[12], acc[13]);
        a_577[2] = mul_f32x2_noftz(previous_580, corr_543);
        float2 previous_581 = make_float2(acc[14], acc[15]);
        a_577[3] = mul_f32x2_noftz(previous_581, corr_543);
        float2 weight_582 = make_float2(weights_541[4], weights_541[4]);
        unsigned int sw2_583[4];
        sw2_583[0] = words[56 + woff_575];
        sw2_583[1] = words[56 + woff_575 + 1];
        sw2_583[2] = words[56 + woff_575 + 2];
        sw2_583[3] = words[56 + woff_575 + 3];
        float _unpack_bf16_40[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                : "r"(sw2_583[_pair]));
        }
        float2 v_584 = make_float2(_unpack_bf16_40[0], _unpack_bf16_40[1]);
        a_577[0] = fma_f32x2_rn_noftz(weight_582, v_584, a_577[0]);
        float2 v_585 = make_float2(_unpack_bf16_40[2], _unpack_bf16_40[3]);
        a_577[1] = fma_f32x2_rn_noftz(weight_582, v_585, a_577[1]);
        float2 v_586 = make_float2(_unpack_bf16_40[4], _unpack_bf16_40[5]);
        a_577[2] = fma_f32x2_rn_noftz(weight_582, v_586, a_577[2]);
        float2 v_587 = make_float2(_unpack_bf16_40[6], _unpack_bf16_40[7]);
        a_577[3] = fma_f32x2_rn_noftz(weight_582, v_587, a_577[3]);
        float2 weight_588 = make_float2(weights_541[5], weights_541[5]);
        unsigned int sw2_589[4];
        sw2_589[0] = words[70 + woff_575];
        sw2_589[1] = words[70 + woff_575 + 1];
        sw2_589[2] = words[70 + woff_575 + 2];
        sw2_589[3] = words[70 + woff_575 + 3];
        float _unpack_bf16_41[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                : "r"(sw2_589[_pair]));
        }
        float2 v_590 = make_float2(_unpack_bf16_41[0], _unpack_bf16_41[1]);
        a_577[0] = fma_f32x2_rn_noftz(weight_588, v_590, a_577[0]);
        float2 v_591 = make_float2(_unpack_bf16_41[2], _unpack_bf16_41[3]);
        a_577[1] = fma_f32x2_rn_noftz(weight_588, v_591, a_577[1]);
        float2 v_592 = make_float2(_unpack_bf16_41[4], _unpack_bf16_41[5]);
        a_577[2] = fma_f32x2_rn_noftz(weight_588, v_592, a_577[2]);
        float2 v_593 = make_float2(_unpack_bf16_41[6], _unpack_bf16_41[7]);
        a_577[3] = fma_f32x2_rn_noftz(weight_588, v_593, a_577[3]);
        float2 weight_594 = make_float2(weights_541[6], weights_541[6]);
        unsigned int sw2_595[4];
        sw2_595[0] = words[84 + woff_575];
        sw2_595[1] = words[84 + woff_575 + 1];
        sw2_595[2] = words[84 + woff_575 + 2];
        sw2_595[3] = words[84 + woff_575 + 3];
        float _unpack_bf16_42[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                : "r"(sw2_595[_pair]));
        }
        float2 v_596 = make_float2(_unpack_bf16_42[0], _unpack_bf16_42[1]);
        a_577[0] = fma_f32x2_rn_noftz(weight_594, v_596, a_577[0]);
        float2 v_597 = make_float2(_unpack_bf16_42[2], _unpack_bf16_42[3]);
        a_577[1] = fma_f32x2_rn_noftz(weight_594, v_597, a_577[1]);
        float2 v_598 = make_float2(_unpack_bf16_42[4], _unpack_bf16_42[5]);
        a_577[2] = fma_f32x2_rn_noftz(weight_594, v_598, a_577[2]);
        float2 v_599 = make_float2(_unpack_bf16_42[6], _unpack_bf16_42[7]);
        a_577[3] = fma_f32x2_rn_noftz(weight_594, v_599, a_577[3]);
        float2 weight_600 = make_float2(weights_541[7], weights_541[7]);
        unsigned int sw2_601[4];
        sw2_601[0] = words[98 + woff_575];
        sw2_601[1] = words[98 + woff_575 + 1];
        sw2_601[2] = words[98 + woff_575 + 2];
        sw2_601[3] = words[98 + woff_575 + 3];
        float _unpack_bf16_43[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                : "r"(sw2_601[_pair]));
        }
        float2 v_602 = make_float2(_unpack_bf16_43[0], _unpack_bf16_43[1]);
        a_577[0] = fma_f32x2_rn_noftz(weight_600, v_602, a_577[0]);
        float2 v_603 = make_float2(_unpack_bf16_43[2], _unpack_bf16_43[3]);
        a_577[1] = fma_f32x2_rn_noftz(weight_600, v_603, a_577[1]);
        float2 v_604 = make_float2(_unpack_bf16_43[4], _unpack_bf16_43[5]);
        a_577[2] = fma_f32x2_rn_noftz(weight_600, v_604, a_577[2]);
        float2 v_605 = make_float2(_unpack_bf16_43[6], _unpack_bf16_43[7]);
        a_577[3] = fma_f32x2_rn_noftz(weight_600, v_605, a_577[3]);
        acc[8] = a_577[0].x;
        acc[9] = a_577[0].y;
        acc[10] = a_577[1].x;
        acc[11] = a_577[1].y;
        acc[12] = a_577[2].x;
        acc[13] = a_577[2].y;
        acc[14] = a_577[3].x;
        acc[15] = a_577[3].y;
        const int woff_606 = 8;
        int base_607 = (4 + group) * 1024 + thread * 8;
        float2 a_608[4];
        float2 previous_609 = make_float2(acc[16], acc[17]);
        a_608[0] = mul_f32x2_noftz(previous_609, corr_543);
        float2 previous_610 = make_float2(acc[18], acc[19]);
        a_608[1] = mul_f32x2_noftz(previous_610, corr_543);
        float2 previous_611 = make_float2(acc[20], acc[21]);
        a_608[2] = mul_f32x2_noftz(previous_611, corr_543);
        float2 previous_612 = make_float2(acc[22], acc[23]);
        a_608[3] = mul_f32x2_noftz(previous_612, corr_543);
        float2 weight_613 = make_float2(weights_541[4], weights_541[4]);
        unsigned int sw2_614[4];
        sw2_614[0] = words[56 + woff_606];
        sw2_614[1] = words[56 + woff_606 + 1];
        sw2_614[2] = words[56 + woff_606 + 2];
        sw2_614[3] = words[56 + woff_606 + 3];
        float _unpack_bf16_44[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_44[_pair * 2])[0]), "=f"((&_unpack_bf16_44[_pair * 2])[1])
                : "r"(sw2_614[_pair]));
        }
        float2 v_615 = make_float2(_unpack_bf16_44[0], _unpack_bf16_44[1]);
        a_608[0] = fma_f32x2_rn_noftz(weight_613, v_615, a_608[0]);
        float2 v_616 = make_float2(_unpack_bf16_44[2], _unpack_bf16_44[3]);
        a_608[1] = fma_f32x2_rn_noftz(weight_613, v_616, a_608[1]);
        float2 v_617 = make_float2(_unpack_bf16_44[4], _unpack_bf16_44[5]);
        a_608[2] = fma_f32x2_rn_noftz(weight_613, v_617, a_608[2]);
        float2 v_618 = make_float2(_unpack_bf16_44[6], _unpack_bf16_44[7]);
        a_608[3] = fma_f32x2_rn_noftz(weight_613, v_618, a_608[3]);
        float2 weight_619 = make_float2(weights_541[5], weights_541[5]);
        unsigned int sw2_620[4];
        sw2_620[0] = words[70 + woff_606];
        sw2_620[1] = words[70 + woff_606 + 1];
        sw2_620[2] = words[70 + woff_606 + 2];
        sw2_620[3] = words[70 + woff_606 + 3];
        float _unpack_bf16_45[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_45[_pair * 2])[0]), "=f"((&_unpack_bf16_45[_pair * 2])[1])
                : "r"(sw2_620[_pair]));
        }
        float2 v_621 = make_float2(_unpack_bf16_45[0], _unpack_bf16_45[1]);
        a_608[0] = fma_f32x2_rn_noftz(weight_619, v_621, a_608[0]);
        float2 v_622 = make_float2(_unpack_bf16_45[2], _unpack_bf16_45[3]);
        a_608[1] = fma_f32x2_rn_noftz(weight_619, v_622, a_608[1]);
        float2 v_623 = make_float2(_unpack_bf16_45[4], _unpack_bf16_45[5]);
        a_608[2] = fma_f32x2_rn_noftz(weight_619, v_623, a_608[2]);
        float2 v_624 = make_float2(_unpack_bf16_45[6], _unpack_bf16_45[7]);
        a_608[3] = fma_f32x2_rn_noftz(weight_619, v_624, a_608[3]);
        float2 weight_625 = make_float2(weights_541[6], weights_541[6]);
        unsigned int sw2_626[4];
        sw2_626[0] = words[84 + woff_606];
        sw2_626[1] = words[84 + woff_606 + 1];
        sw2_626[2] = words[84 + woff_606 + 2];
        sw2_626[3] = words[84 + woff_606 + 3];
        float _unpack_bf16_46[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_46[_pair * 2])[0]), "=f"((&_unpack_bf16_46[_pair * 2])[1])
                : "r"(sw2_626[_pair]));
        }
        float2 v_627 = make_float2(_unpack_bf16_46[0], _unpack_bf16_46[1]);
        a_608[0] = fma_f32x2_rn_noftz(weight_625, v_627, a_608[0]);
        float2 v_628 = make_float2(_unpack_bf16_46[2], _unpack_bf16_46[3]);
        a_608[1] = fma_f32x2_rn_noftz(weight_625, v_628, a_608[1]);
        float2 v_629 = make_float2(_unpack_bf16_46[4], _unpack_bf16_46[5]);
        a_608[2] = fma_f32x2_rn_noftz(weight_625, v_629, a_608[2]);
        float2 v_630 = make_float2(_unpack_bf16_46[6], _unpack_bf16_46[7]);
        a_608[3] = fma_f32x2_rn_noftz(weight_625, v_630, a_608[3]);
        float2 weight_631 = make_float2(weights_541[7], weights_541[7]);
        unsigned int sw2_632[4];
        sw2_632[0] = words[98 + woff_606];
        sw2_632[1] = words[98 + woff_606 + 1];
        sw2_632[2] = words[98 + woff_606 + 2];
        sw2_632[3] = words[98 + woff_606 + 3];
        float _unpack_bf16_47[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_47[_pair * 2])[0]), "=f"((&_unpack_bf16_47[_pair * 2])[1])
                : "r"(sw2_632[_pair]));
        }
        float2 v_633 = make_float2(_unpack_bf16_47[0], _unpack_bf16_47[1]);
        a_608[0] = fma_f32x2_rn_noftz(weight_631, v_633, a_608[0]);
        float2 v_634 = make_float2(_unpack_bf16_47[2], _unpack_bf16_47[3]);
        a_608[1] = fma_f32x2_rn_noftz(weight_631, v_634, a_608[1]);
        float2 v_635 = make_float2(_unpack_bf16_47[4], _unpack_bf16_47[5]);
        a_608[2] = fma_f32x2_rn_noftz(weight_631, v_635, a_608[2]);
        float2 v_636 = make_float2(_unpack_bf16_47[6], _unpack_bf16_47[7]);
        a_608[3] = fma_f32x2_rn_noftz(weight_631, v_636, a_608[3]);
        acc[16] = a_608[0].x;
        acc[17] = a_608[0].y;
        acc[18] = a_608[1].x;
        acc[19] = a_608[1].y;
        acc[20] = a_608[2].x;
        acc[21] = a_608[2].y;
        acc[22] = a_608[3].x;
        acc[23] = a_608[3].y;
        const int woff_637 = 12;
        int base_638 = 6144 + group * 512 + thread * 4;
        float2 a_639[4];
        float2 previous_640 = make_float2(acc[24], acc[25]);
        a_639[0] = mul_f32x2_noftz(previous_640, corr_543);
        float2 previous_641 = make_float2(acc[26], acc[27]);
        a_639[1] = mul_f32x2_noftz(previous_641, corr_543);
        float2 weight_642 = make_float2(weights_541[4], weights_541[4]);
        unsigned int sw2_643[4];
        sw2_643[0] = words[56 + woff_637];
        sw2_643[1] = words[56 + woff_637 + 1];
        float _unpack_bf16_48[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_48[_pair * 2])[0]), "=f"((&_unpack_bf16_48[_pair * 2])[1])
                : "r"(sw2_643[_pair]));
        }
        float2 v_644 = make_float2(_unpack_bf16_48[0], _unpack_bf16_48[1]);
        a_639[0] = fma_f32x2_rn_noftz(weight_642, v_644, a_639[0]);
        float2 v_645 = make_float2(_unpack_bf16_48[2], _unpack_bf16_48[3]);
        a_639[1] = fma_f32x2_rn_noftz(weight_642, v_645, a_639[1]);
        float2 weight_646 = make_float2(weights_541[5], weights_541[5]);
        unsigned int sw2_647[4];
        sw2_647[0] = words[70 + woff_637];
        sw2_647[1] = words[70 + woff_637 + 1];
        float _unpack_bf16_49[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_49[_pair * 2])[0]), "=f"((&_unpack_bf16_49[_pair * 2])[1])
                : "r"(sw2_647[_pair]));
        }
        float2 v_648 = make_float2(_unpack_bf16_49[0], _unpack_bf16_49[1]);
        a_639[0] = fma_f32x2_rn_noftz(weight_646, v_648, a_639[0]);
        float2 v_649 = make_float2(_unpack_bf16_49[2], _unpack_bf16_49[3]);
        a_639[1] = fma_f32x2_rn_noftz(weight_646, v_649, a_639[1]);
        float2 weight_650 = make_float2(weights_541[6], weights_541[6]);
        unsigned int sw2_651[4];
        sw2_651[0] = words[84 + woff_637];
        sw2_651[1] = words[84 + woff_637 + 1];
        float _unpack_bf16_50[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_50[_pair * 2])[0]), "=f"((&_unpack_bf16_50[_pair * 2])[1])
                : "r"(sw2_651[_pair]));
        }
        float2 v_652 = make_float2(_unpack_bf16_50[0], _unpack_bf16_50[1]);
        a_639[0] = fma_f32x2_rn_noftz(weight_650, v_652, a_639[0]);
        float2 v_653 = make_float2(_unpack_bf16_50[2], _unpack_bf16_50[3]);
        a_639[1] = fma_f32x2_rn_noftz(weight_650, v_653, a_639[1]);
        float2 weight_654 = make_float2(weights_541[7], weights_541[7]);
        unsigned int sw2_655[4];
        sw2_655[0] = words[98 + woff_637];
        sw2_655[1] = words[98 + woff_637 + 1];
        float _unpack_bf16_51[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_51[_pair * 2])[0]), "=f"((&_unpack_bf16_51[_pair * 2])[1])
                : "r"(sw2_655[_pair]));
        }
        float2 v_656 = make_float2(_unpack_bf16_51[0], _unpack_bf16_51[1]);
        a_639[0] = fma_f32x2_rn_noftz(weight_654, v_656, a_639[0]);
        float2 v_657 = make_float2(_unpack_bf16_51[2], _unpack_bf16_51[3]);
        a_639[1] = fma_f32x2_rn_noftz(weight_654, v_657, a_639[1]);
        acc[24] = a_639[0].x;
        acc[25] = a_639[0].y;
        acc[26] = a_639[1].x;
        acc[27] = a_639[1].y;
        sum_running = __fmaf_rn(sum_running, correction_540, sum_weights_542);
        max_running = max_new_539;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_658 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_658, v_658, output_sq_pair);
        float2 v_659 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_659, v_659, output_sq_pair);
        float2 v_660 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_660, v_660, output_sq_pair);
        float2 v_661 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_661, v_661, output_sq_pair);
        float2 v_662 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_662, v_662, output_sq_pair);
        float2 v_663 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_663, v_663, output_sq_pair);
        float2 v_664 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_664, v_664, output_sq_pair);
        float2 v_665 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_665, v_665, output_sq_pair);
        float2 v_666 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_666, v_666, output_sq_pair);
        float2 v_667 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_667, v_667, output_sq_pair);
        float2 v_668 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_668, v_668, output_sq_pair);
        float2 v_669 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_669, v_669, output_sq_pair);
        float2 v_670 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_670, v_670, output_sq_pair);
        float2 v_671 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_671, v_671, output_sq_pair);
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
        int base_672 = group * 1024 + thread * 8;
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
        const int value_idx_673 = 2;
        const int acc_idx_674 = value_idx_673;
        float2 _f32x2_mul_a_118 = make_float2(acc[acc_idx_674], acc[acc_idx_674 + 1]);
        float2 scaled_pair_675;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_675) : "l"(*(const unsigned long long*)&_f32x2_mul_a_118), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_119 = make_float2(wout[acc_idx_674], wout[acc_idx_674 + 1]);
        float2 normalized_pair_676;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_676) : "l"(*(const unsigned long long*)&scaled_pair_675), "l"(*(const unsigned long long*)&_f32x2_mul_b_119));
        output_values[value_idx_673] = normalized_pair_676.x;
        output_values[value_idx_673 + 1] = normalized_pair_676.y;
        const int value_idx_677 = 4;
        const int acc_idx_678 = value_idx_677;
        float2 _f32x2_mul_a_120 = make_float2(acc[acc_idx_678], acc[acc_idx_678 + 1]);
        float2 scaled_pair_679;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_679) : "l"(*(const unsigned long long*)&_f32x2_mul_a_120), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_121 = make_float2(wout[acc_idx_678], wout[acc_idx_678 + 1]);
        float2 normalized_pair_680;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_680) : "l"(*(const unsigned long long*)&scaled_pair_679), "l"(*(const unsigned long long*)&_f32x2_mul_b_121));
        output_values[value_idx_677] = normalized_pair_680.x;
        output_values[value_idx_677 + 1] = normalized_pair_680.y;
        const int value_idx_681 = 6;
        const int acc_idx_682 = value_idx_681;
        float2 _f32x2_mul_a_122 = make_float2(acc[acc_idx_682], acc[acc_idx_682 + 1]);
        float2 scaled_pair_683;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_683) : "l"(*(const unsigned long long*)&_f32x2_mul_a_122), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_123 = make_float2(wout[acc_idx_682], wout[acc_idx_682 + 1]);
        float2 normalized_pair_684;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_684) : "l"(*(const unsigned long long*)&scaled_pair_683), "l"(*(const unsigned long long*)&_f32x2_mul_b_123));
        output_values[value_idx_681] = normalized_pair_684.x;
        output_values[value_idx_681 + 1] = normalized_pair_684.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_672))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_685 = (2 + group) * 1024 + thread * 8;
        float output_values_686[8];
        const int value_idx_687 = 0;
        const int acc_idx_688 = 8 + value_idx_687;
        float2 _f32x2_mul_a_124 = make_float2(acc[acc_idx_688], acc[acc_idx_688 + 1]);
        float2 scaled_pair_689;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_689) : "l"(*(const unsigned long long*)&_f32x2_mul_a_124), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_125 = make_float2(wout[acc_idx_688], wout[acc_idx_688 + 1]);
        float2 normalized_pair_690;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_690) : "l"(*(const unsigned long long*)&scaled_pair_689), "l"(*(const unsigned long long*)&_f32x2_mul_b_125));
        output_values_686[value_idx_687] = normalized_pair_690.x;
        output_values_686[value_idx_687 + 1] = normalized_pair_690.y;
        const int value_idx_691 = 2;
        const int acc_idx_692 = 8 + value_idx_691;
        float2 _f32x2_mul_a_126 = make_float2(acc[acc_idx_692], acc[acc_idx_692 + 1]);
        float2 scaled_pair_693;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_693) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_127 = make_float2(wout[acc_idx_692], wout[acc_idx_692 + 1]);
        float2 normalized_pair_694;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_694) : "l"(*(const unsigned long long*)&scaled_pair_693), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
        output_values_686[value_idx_691] = normalized_pair_694.x;
        output_values_686[value_idx_691 + 1] = normalized_pair_694.y;
        const int value_idx_695 = 4;
        const int acc_idx_696 = 8 + value_idx_695;
        float2 _f32x2_mul_a_128 = make_float2(acc[acc_idx_696], acc[acc_idx_696 + 1]);
        float2 scaled_pair_697;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_697) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_129 = make_float2(wout[acc_idx_696], wout[acc_idx_696 + 1]);
        float2 normalized_pair_698;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_698) : "l"(*(const unsigned long long*)&scaled_pair_697), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
        output_values_686[value_idx_695] = normalized_pair_698.x;
        output_values_686[value_idx_695 + 1] = normalized_pair_698.y;
        const int value_idx_699 = 6;
        const int acc_idx_700 = 8 + value_idx_699;
        float2 _f32x2_mul_a_130 = make_float2(acc[acc_idx_700], acc[acc_idx_700 + 1]);
        float2 scaled_pair_701;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_701) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_131 = make_float2(wout[acc_idx_700], wout[acc_idx_700 + 1]);
        float2 normalized_pair_702;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_702) : "l"(*(const unsigned long long*)&scaled_pair_701), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
        output_values_686[value_idx_699] = normalized_pair_702.x;
        output_values_686[value_idx_699 + 1] = normalized_pair_702.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_686[_lp*2 + 0], output_values_686[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_686[_lp*2 + 2], output_values_686[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_686[_lp*2 + 4], output_values_686[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_686[_lp*2 + 6], output_values_686[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_685))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_703 = (4 + group) * 1024 + thread * 8;
        float output_values_704[8];
        const int value_idx_705 = 0;
        const int acc_idx_706 = 16 + value_idx_705;
        float2 _f32x2_mul_a_132 = make_float2(acc[acc_idx_706], acc[acc_idx_706 + 1]);
        float2 scaled_pair_707;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_707) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_133 = make_float2(wout[acc_idx_706], wout[acc_idx_706 + 1]);
        float2 normalized_pair_708;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_708) : "l"(*(const unsigned long long*)&scaled_pair_707), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
        output_values_704[value_idx_705] = normalized_pair_708.x;
        output_values_704[value_idx_705 + 1] = normalized_pair_708.y;
        const int value_idx_709 = 2;
        const int acc_idx_710 = 16 + value_idx_709;
        float2 _f32x2_mul_a_134 = make_float2(acc[acc_idx_710], acc[acc_idx_710 + 1]);
        float2 scaled_pair_711;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_711) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_135 = make_float2(wout[acc_idx_710], wout[acc_idx_710 + 1]);
        float2 normalized_pair_712;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_712) : "l"(*(const unsigned long long*)&scaled_pair_711), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
        output_values_704[value_idx_709] = normalized_pair_712.x;
        output_values_704[value_idx_709 + 1] = normalized_pair_712.y;
        const int value_idx_713 = 4;
        const int acc_idx_714 = 16 + value_idx_713;
        float2 _f32x2_mul_a_136 = make_float2(acc[acc_idx_714], acc[acc_idx_714 + 1]);
        float2 scaled_pair_715;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_715) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_137 = make_float2(wout[acc_idx_714], wout[acc_idx_714 + 1]);
        float2 normalized_pair_716;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_716) : "l"(*(const unsigned long long*)&scaled_pair_715), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
        output_values_704[value_idx_713] = normalized_pair_716.x;
        output_values_704[value_idx_713 + 1] = normalized_pair_716.y;
        const int value_idx_717 = 6;
        const int acc_idx_718 = 16 + value_idx_717;
        float2 _f32x2_mul_a_138 = make_float2(acc[acc_idx_718], acc[acc_idx_718 + 1]);
        float2 scaled_pair_719;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_719) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_139 = make_float2(wout[acc_idx_718], wout[acc_idx_718 + 1]);
        float2 normalized_pair_720;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_720) : "l"(*(const unsigned long long*)&scaled_pair_719), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
        output_values_704[value_idx_717] = normalized_pair_720.x;
        output_values_704[value_idx_717 + 1] = normalized_pair_720.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_704[_lp*2 + 0], output_values_704[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_704[_lp*2 + 2], output_values_704[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_704[_lp*2 + 4], output_values_704[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_704[_lp*2 + 6], output_values_704[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_703))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_721 = 6144 + group * 512 + thread * 4;
        float output_values_722[8];
        const int value_idx_723 = 0;
        const int acc_idx_724 = 24 + value_idx_723;
        float2 _f32x2_mul_a_140 = make_float2(acc[acc_idx_724], acc[acc_idx_724 + 1]);
        float2 scaled_pair_725;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_725) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_141 = make_float2(wout[acc_idx_724], wout[acc_idx_724 + 1]);
        float2 normalized_pair_726;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_726) : "l"(*(const unsigned long long*)&scaled_pair_725), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
        output_values_722[value_idx_723] = normalized_pair_726.x;
        output_values_722[value_idx_723 + 1] = normalized_pair_726.y;
        const int value_idx_727 = 2;
        const int acc_idx_728 = 24 + value_idx_727;
        float2 _f32x2_mul_a_142 = make_float2(acc[acc_idx_728], acc[acc_idx_728 + 1]);
        float2 scaled_pair_729;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_729) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_143 = make_float2(wout[acc_idx_728], wout[acc_idx_728 + 1]);
        float2 normalized_pair_730;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_730) : "l"(*(const unsigned long long*)&scaled_pair_729), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
        output_values_722[value_idx_727] = normalized_pair_730.x;
        output_values_722[value_idx_727 + 1] = normalized_pair_730.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_722[_lp*2 + 0], output_values_722[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_722[_lp*2 + 2], output_values_722[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_721))[0] = reinterpret_cast<int2*>(_packed_3)[0];
    }
}

} // extern "C"
