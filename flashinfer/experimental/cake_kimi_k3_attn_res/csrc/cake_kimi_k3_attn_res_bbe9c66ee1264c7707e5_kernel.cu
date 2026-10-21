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
kernel_cake_kimi_k3_attn_res_bbe9c66ee1264c7707e5(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_4[0 + 0] = _uv4_4.x;
            _vec_load_4[0 + 1] = _uv4_4.y;
            _vec_load_4[0 + 2] = _uv4_4.z;
            _vec_load_4[0 + 3] = _uv4_4.w;
        }
        dwords[woff] = _vec_load_4[0];
        dwords[woff + 1] = _vec_load_4[1];
        dwords[woff + 2] = _vec_load_4[2];
        dwords[woff + 3] = _vec_load_4[3];
        float _vec_load_5[8];
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
                        : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        float _vec_load_6[8];
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
                        : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        q[0] = _vec_load_5[0] * _vec_load_6[0];
        q[1] = _vec_load_5[1] * _vec_load_6[1];
        q[2] = _vec_load_5[2] * _vec_load_6[2];
        q[3] = _vec_load_5[3] * _vec_load_6[3];
        q[4] = _vec_load_5[4] * _vec_load_6[4];
        q[5] = _vec_load_5[5] * _vec_load_6[5];
        q[6] = _vec_load_5[6] * _vec_load_6[6];
        q[7] = _vec_load_5[7] * _vec_load_6[7];
        float _vec_load_7[8];
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
                        : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
        }
        wout[0] = _vec_load_7[0];
        wout[1] = _vec_load_7[1];
        wout[2] = _vec_load_7[2];
        wout[3] = _vec_load_7[3];
        wout[4] = _vec_load_7[4];
        wout[5] = _vec_load_7[5];
        wout[6] = _vec_load_7[6];
        wout[7] = _vec_load_7[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_8[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_8[0 + 0] = _uv4_8.x;
            _vec_load_8[0 + 1] = _uv4_8.y;
            _vec_load_8[0 + 2] = _uv4_8.z;
            _vec_load_8[0 + 3] = _uv4_8.w;
        }
        words[woff_1] = _vec_load_8[0];
        words[woff_1 + 1] = _vec_load_8[1];
        words[woff_1 + 2] = _vec_load_8[2];
        words[woff_1 + 3] = _vec_load_8[3];
        unsigned int _vec_load_9[4];
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_9[0 + 0] = _uv4_9.x;
            _vec_load_9[0 + 1] = _uv4_9.y;
            _vec_load_9[0 + 2] = _uv4_9.z;
            _vec_load_9[0 + 3] = _uv4_9.w;
        }
        words[14 + woff_1] = _vec_load_9[0];
        words[14 + woff_1 + 1] = _vec_load_9[1];
        words[14 + woff_1 + 2] = _vec_load_9[2];
        words[14 + woff_1 + 3] = _vec_load_9[3];
        unsigned int _vec_load_10[4];
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_10[0 + 0] = _uv4_10.x;
            _vec_load_10[0 + 1] = _uv4_10.y;
            _vec_load_10[0 + 2] = _uv4_10.z;
            _vec_load_10[0 + 3] = _uv4_10.w;
        }
        words[28 + woff_1] = _vec_load_10[0];
        words[28 + woff_1 + 1] = _vec_load_10[1];
        words[28 + woff_1 + 2] = _vec_load_10[2];
        words[28 + woff_1 + 3] = _vec_load_10[3];
        unsigned int _vec_load_11[4];
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_11[0 + 0] = _uv4_11.x;
            _vec_load_11[0 + 1] = _uv4_11.y;
            _vec_load_11[0 + 2] = _uv4_11.z;
            _vec_load_11[0 + 3] = _uv4_11.w;
        }
        words[42 + woff_1] = _vec_load_11[0];
        words[42 + woff_1 + 1] = _vec_load_11[1];
        words[42 + woff_1 + 2] = _vec_load_11[2];
        words[42 + woff_1 + 3] = _vec_load_11[3];
        unsigned int _vec_load_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_12[0 + 0] = _uv4_12.x;
            _vec_load_12[0 + 1] = _uv4_12.y;
            _vec_load_12[0 + 2] = _uv4_12.z;
            _vec_load_12[0 + 3] = _uv4_12.w;
        }
        dwords[woff_1] = _vec_load_12[0];
        dwords[woff_1 + 1] = _vec_load_12[1];
        dwords[woff_1 + 2] = _vec_load_12[2];
        dwords[woff_1 + 3] = _vec_load_12[3];
        float _vec_load_13[8];
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
                        : "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_13[_pair]));
                }
            }
        }
        float _vec_load_14[8];
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
                        : "=f"((&_vec_load_14[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_14[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_14[_pair]));
                }
            }
        }
        q[8] = _vec_load_13[0] * _vec_load_14[0];
        q[9] = _vec_load_13[1] * _vec_load_14[1];
        q[10] = _vec_load_13[2] * _vec_load_14[2];
        q[11] = _vec_load_13[3] * _vec_load_14[3];
        q[12] = _vec_load_13[4] * _vec_load_14[4];
        q[13] = _vec_load_13[5] * _vec_load_14[5];
        q[14] = _vec_load_13[6] * _vec_load_14[6];
        q[15] = _vec_load_13[7] * _vec_load_14[7];
        float _vec_load_15[8];
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
                        : "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
        }
        wout[8] = _vec_load_15[0];
        wout[9] = _vec_load_15[1];
        wout[10] = _vec_load_15[2];
        wout[11] = _vec_load_15[3];
        wout[12] = _vec_load_15[4];
        wout[13] = _vec_load_15[5];
        wout[14] = _vec_load_15[6];
        wout[15] = _vec_load_15[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_16[0 + 0] = _uv4_16.x;
            _vec_load_16[0 + 1] = _uv4_16.y;
            _vec_load_16[0 + 2] = _uv4_16.z;
            _vec_load_16[0 + 3] = _uv4_16.w;
        }
        words[woff_3] = _vec_load_16[0];
        words[woff_3 + 1] = _vec_load_16[1];
        words[woff_3 + 2] = _vec_load_16[2];
        words[woff_3 + 3] = _vec_load_16[3];
        unsigned int _vec_load_17[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_17[0 + 0] = _uv4_17.x;
            _vec_load_17[0 + 1] = _uv4_17.y;
            _vec_load_17[0 + 2] = _uv4_17.z;
            _vec_load_17[0 + 3] = _uv4_17.w;
        }
        words[14 + woff_3] = _vec_load_17[0];
        words[14 + woff_3 + 1] = _vec_load_17[1];
        words[14 + woff_3 + 2] = _vec_load_17[2];
        words[14 + woff_3 + 3] = _vec_load_17[3];
        unsigned int _vec_load_18[4];
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_18[0 + 0] = _uv4_18.x;
            _vec_load_18[0 + 1] = _uv4_18.y;
            _vec_load_18[0 + 2] = _uv4_18.z;
            _vec_load_18[0 + 3] = _uv4_18.w;
        }
        words[28 + woff_3] = _vec_load_18[0];
        words[28 + woff_3 + 1] = _vec_load_18[1];
        words[28 + woff_3 + 2] = _vec_load_18[2];
        words[28 + woff_3 + 3] = _vec_load_18[3];
        unsigned int _vec_load_19[4];
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_19[0 + 0] = _uv4_19.x;
            _vec_load_19[0 + 1] = _uv4_19.y;
            _vec_load_19[0 + 2] = _uv4_19.z;
            _vec_load_19[0 + 3] = _uv4_19.w;
        }
        words[42 + woff_3] = _vec_load_19[0];
        words[42 + woff_3 + 1] = _vec_load_19[1];
        words[42 + woff_3 + 2] = _vec_load_19[2];
        words[42 + woff_3 + 3] = _vec_load_19[3];
        unsigned int _vec_load_20[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_20[0 + 0] = _uv4_20.x;
            _vec_load_20[0 + 1] = _uv4_20.y;
            _vec_load_20[0 + 2] = _uv4_20.z;
            _vec_load_20[0 + 3] = _uv4_20.w;
        }
        dwords[woff_3] = _vec_load_20[0];
        dwords[woff_3 + 1] = _vec_load_20[1];
        dwords[woff_3 + 2] = _vec_load_20[2];
        dwords[woff_3 + 3] = _vec_load_20[3];
        float _vec_load_21[8];
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
                        : "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_21[_pair]));
                }
            }
        }
        float _vec_load_22[8];
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
                        : "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_22[_pair]));
                }
            }
        }
        q[16] = _vec_load_21[0] * _vec_load_22[0];
        q[17] = _vec_load_21[1] * _vec_load_22[1];
        q[18] = _vec_load_21[2] * _vec_load_22[2];
        q[19] = _vec_load_21[3] * _vec_load_22[3];
        q[20] = _vec_load_21[4] * _vec_load_22[4];
        q[21] = _vec_load_21[5] * _vec_load_22[5];
        q[22] = _vec_load_21[6] * _vec_load_22[6];
        q[23] = _vec_load_21[7] * _vec_load_22[7];
        float _vec_load_23[8];
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
                        : "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_23[_pair]));
                }
            }
        }
        wout[16] = _vec_load_23[0];
        wout[17] = _vec_load_23[1];
        wout[18] = _vec_load_23[2];
        wout[19] = _vec_load_23[3];
        wout[20] = _vec_load_23[4];
        wout[21] = _vec_load_23[5];
        wout[22] = _vec_load_23[6];
        wout[23] = _vec_load_23[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_24[1];
        {
            _vec_load_24[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_24[0];
        unsigned int _vec_load_25[1];
        {
            _vec_load_25[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_25[0];
        unsigned int _vec_load_26[1];
        {
            _vec_load_26[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_26[0];
        unsigned int _vec_load_27[1];
        {
            _vec_load_27[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_27[0];
        unsigned int _vec_load_28[1];
        {
            _vec_load_28[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_28[0];
        unsigned int _vec_load_29[1];
        {
            _vec_load_29[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_29[0];
        unsigned int _vec_load_30[1];
        {
            _vec_load_30[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[42 + woff_5] = _vec_load_30[0];
        unsigned int _vec_load_31[1];
        {
            _vec_load_31[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[42 + woff_5 + 1] = _vec_load_31[0];
        unsigned int _vec_load_32[1];
        {
            _vec_load_32[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_32[0];
        unsigned int _vec_load_33[1];
        {
            _vec_load_33[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_33[0];
        float _vec_load_34[4];
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
                    : "=f"((&_vec_load_34[0 + _pair * 2])[0]), "=f"((&_vec_load_34[0 + _pair * 2])[1])
                    : "r"(_vpairs_24[_pair]));
            }
        }
        float _vec_load_35[4];
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
                    : "=f"((&_vec_load_35[0 + _pair * 2])[0]), "=f"((&_vec_load_35[0 + _pair * 2])[1])
                    : "r"(_vpairs_25[_pair]));
            }
        }
        q[24] = _vec_load_34[0] * _vec_load_35[0];
        q[25] = _vec_load_34[1] * _vec_load_35[1];
        q[26] = _vec_load_34[2] * _vec_load_35[2];
        q[27] = _vec_load_34[3] * _vec_load_35[3];
        float _vec_load_36[4];
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
                    : "=f"((&_vec_load_36[0 + _pair * 2])[0]), "=f"((&_vec_load_36[0 + _pair * 2])[1])
                    : "r"(_vpairs_26[_pair]));
            }
        }
        wout[24] = _vec_load_36[0];
        wout[25] = _vec_load_36[1];
        wout[26] = _vec_load_36[2];
        wout[27] = _vec_load_36[3];
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
        float fsrc[84];
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
        float2 _f2_8 = make_float2(_unpack_bf16_0[0], _unpack_bf16_0[1]);
        float2 v = _f2_8;
        float2 _f2_9 = make_float2(q[0], q[1]);
        float2 qp = _f2_9;
        sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
        float2 _f2_10 = make_float2(_unpack_bf16_0[2], _unpack_bf16_0[3]);
        float2 v_8 = _f2_10;
        float2 _f2_11 = make_float2(q[2], q[3]);
        float2 qp_9 = _f2_11;
        sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_8, qp_9, dot[0]);
        float2 _f2_12 = make_float2(_unpack_bf16_0[4], _unpack_bf16_0[5]);
        float2 v_10 = _f2_12;
        float2 _f2_13 = make_float2(q[4], q[5]);
        float2 qp_11 = _f2_13;
        sq[0] = fma_f32x2_rn_noftz(v_10, v_10, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_10, qp_11, dot[0]);
        float2 _f2_14 = make_float2(_unpack_bf16_0[6], _unpack_bf16_0[7]);
        float2 v_12 = _f2_14;
        float2 _f2_15 = make_float2(q[6], q[7]);
        float2 qp_13 = _f2_15;
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
        float2 _f2_16 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
        float2 v_15 = _f2_16;
        float2 _f2_17 = make_float2(q[0], q[1]);
        float2 qp_16 = _f2_17;
        sq[1] = fma_f32x2_rn_noftz(v_15, v_15, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_15, qp_16, dot[1]);
        float2 _f2_18 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
        float2 v_17 = _f2_18;
        float2 _f2_19 = make_float2(q[2], q[3]);
        float2 qp_18 = _f2_19;
        sq[1] = fma_f32x2_rn_noftz(v_17, v_17, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_17, qp_18, dot[1]);
        float2 _f2_20 = make_float2(_unpack_bf16_1[4], _unpack_bf16_1[5]);
        float2 v_19 = _f2_20;
        float2 _f2_21 = make_float2(q[4], q[5]);
        float2 qp_20 = _f2_21;
        sq[1] = fma_f32x2_rn_noftz(v_19, v_19, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_19, qp_20, dot[1]);
        float2 _f2_22 = make_float2(_unpack_bf16_1[6], _unpack_bf16_1[7]);
        float2 v_21 = _f2_22;
        float2 _f2_23 = make_float2(q[6], q[7]);
        float2 qp_22 = _f2_23;
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
        float2 _f2_24 = make_float2(_unpack_bf16_2[0], _unpack_bf16_2[1]);
        float2 v_24 = _f2_24;
        float2 _f2_25 = make_float2(q[0], q[1]);
        float2 qp_25 = _f2_25;
        sq[2] = fma_f32x2_rn_noftz(v_24, v_24, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_24, qp_25, dot[2]);
        float2 _f2_26 = make_float2(_unpack_bf16_2[2], _unpack_bf16_2[3]);
        float2 v_26 = _f2_26;
        float2 _f2_27 = make_float2(q[2], q[3]);
        float2 qp_27 = _f2_27;
        sq[2] = fma_f32x2_rn_noftz(v_26, v_26, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_26, qp_27, dot[2]);
        float2 _f2_28 = make_float2(_unpack_bf16_2[4], _unpack_bf16_2[5]);
        float2 v_28 = _f2_28;
        float2 _f2_29 = make_float2(q[4], q[5]);
        float2 qp_29 = _f2_29;
        sq[2] = fma_f32x2_rn_noftz(v_28, v_28, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_28, qp_29, dot[2]);
        float2 _f2_30 = make_float2(_unpack_bf16_2[6], _unpack_bf16_2[7]);
        float2 v_30 = _f2_30;
        float2 _f2_31 = make_float2(q[6], q[7]);
        float2 qp_31 = _f2_31;
        sq[2] = fma_f32x2_rn_noftz(v_30, v_30, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_30, qp_31, dot[2]);
        unsigned int sw_32[4];
        sw_32[0] = words[42 + woff_7];
        sw_32[1] = words[42 + woff_7 + 1];
        sw_32[2] = words[42 + woff_7 + 2];
        sw_32[3] = words[42 + woff_7 + 3];
        __nv_bfloat162 a = __as_bf16x2(sw_32[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_32[0] = __as_u32(mixed);
        words[42 + woff_7] = sw_32[0];
        __nv_bfloat162 a_33 = __as_bf16x2(sw_32[1]);
        __nv_bfloat162 d_34 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_35 = a_33 + d_34;
        sw_32[1] = __as_u32(mixed_35);
        words[42 + woff_7 + 1] = sw_32[1];
        __nv_bfloat162 a_36 = __as_bf16x2(sw_32[2]);
        __nv_bfloat162 d_37 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_38 = a_36 + d_37;
        sw_32[2] = __as_u32(mixed_38);
        words[42 + woff_7 + 2] = sw_32[2];
        __nv_bfloat162 a_39 = __as_bf16x2(sw_32[3]);
        __nv_bfloat162 d_40 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_41 = a_39 + d_40;
        sw_32[3] = __as_u32(mixed_41);
        words[42 + woff_7 + 3] = sw_32[3];
        {
            int4 _iv4 = make_int4(sw_32[0 + 0], sw_32[0 + 1], sw_32[0 + 2], sw_32[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
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
                : "r"(sw_32[_pair]));
        }
        float2 _f2_32 = make_float2(_unpack_bf16_3[0], _unpack_bf16_3[1]);
        float2 v_42 = _f2_32;
        float2 _f2_33 = make_float2(q[0], q[1]);
        float2 qp_43 = _f2_33;
        sq[3] = fma_f32x2_rn_noftz(v_42, v_42, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_42, qp_43, dot[3]);
        float2 _f2_34 = make_float2(_unpack_bf16_3[2], _unpack_bf16_3[3]);
        float2 v_44 = _f2_34;
        float2 _f2_35 = make_float2(q[2], q[3]);
        float2 qp_45 = _f2_35;
        sq[3] = fma_f32x2_rn_noftz(v_44, v_44, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_44, qp_45, dot[3]);
        float2 _f2_36 = make_float2(_unpack_bf16_3[4], _unpack_bf16_3[5]);
        float2 v_46 = _f2_36;
        float2 _f2_37 = make_float2(q[4], q[5]);
        float2 qp_47 = _f2_37;
        sq[3] = fma_f32x2_rn_noftz(v_46, v_46, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_46, qp_47, dot[3]);
        float2 _f2_38 = make_float2(_unpack_bf16_3[6], _unpack_bf16_3[7]);
        float2 v_48 = _f2_38;
        float2 _f2_39 = make_float2(q[6], q[7]);
        float2 qp_49 = _f2_39;
        sq[3] = fma_f32x2_rn_noftz(v_48, v_48, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_48, qp_49, dot[3]);
        int base_50 = (2 + group) * 1024 + thread * 8;
        const int woff_51 = 4;
        unsigned int sw_52[4];
        sw_52[0] = words[woff_51];
        sw_52[1] = words[woff_51 + 1];
        sw_52[2] = words[woff_51 + 2];
        sw_52[3] = words[woff_51 + 3];
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                : "r"(sw_52[_pair]));
        }
        fsrc[8] = _unpack_bf16_4[0];
        fsrc[9] = _unpack_bf16_4[1];
        fsrc[10] = _unpack_bf16_4[2];
        fsrc[11] = _unpack_bf16_4[3];
        fsrc[12] = _unpack_bf16_4[4];
        fsrc[13] = _unpack_bf16_4[5];
        fsrc[14] = _unpack_bf16_4[6];
        fsrc[15] = _unpack_bf16_4[7];
        float2 _f2_40 = make_float2(_unpack_bf16_4[0], _unpack_bf16_4[1]);
        float2 v_53 = _f2_40;
        float2 _f2_41 = make_float2(q[8], q[9]);
        float2 qp_54 = _f2_41;
        sq[0] = fma_f32x2_rn_noftz(v_53, v_53, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_53, qp_54, dot[0]);
        float2 _f2_42 = make_float2(_unpack_bf16_4[2], _unpack_bf16_4[3]);
        float2 v_55 = _f2_42;
        float2 _f2_43 = make_float2(q[10], q[11]);
        float2 qp_56 = _f2_43;
        sq[0] = fma_f32x2_rn_noftz(v_55, v_55, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_55, qp_56, dot[0]);
        float2 _f2_44 = make_float2(_unpack_bf16_4[4], _unpack_bf16_4[5]);
        float2 v_57 = _f2_44;
        float2 _f2_45 = make_float2(q[12], q[13]);
        float2 qp_58 = _f2_45;
        sq[0] = fma_f32x2_rn_noftz(v_57, v_57, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_57, qp_58, dot[0]);
        float2 _f2_46 = make_float2(_unpack_bf16_4[6], _unpack_bf16_4[7]);
        float2 v_59 = _f2_46;
        float2 _f2_47 = make_float2(q[14], q[15]);
        float2 qp_60 = _f2_47;
        sq[0] = fma_f32x2_rn_noftz(v_59, v_59, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_59, qp_60, dot[0]);
        unsigned int sw_61[4];
        sw_61[0] = words[14 + woff_51];
        sw_61[1] = words[14 + woff_51 + 1];
        sw_61[2] = words[14 + woff_51 + 2];
        sw_61[3] = words[14 + woff_51 + 3];
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                : "r"(sw_61[_pair]));
        }
        fsrc[36] = _unpack_bf16_5[0];
        fsrc[37] = _unpack_bf16_5[1];
        fsrc[38] = _unpack_bf16_5[2];
        fsrc[39] = _unpack_bf16_5[3];
        fsrc[40] = _unpack_bf16_5[4];
        fsrc[41] = _unpack_bf16_5[5];
        fsrc[42] = _unpack_bf16_5[6];
        fsrc[43] = _unpack_bf16_5[7];
        float2 _f2_48 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 v_62 = _f2_48;
        float2 _f2_49 = make_float2(q[8], q[9]);
        float2 qp_63 = _f2_49;
        sq[1] = fma_f32x2_rn_noftz(v_62, v_62, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_62, qp_63, dot[1]);
        float2 _f2_50 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 v_64 = _f2_50;
        float2 _f2_51 = make_float2(q[10], q[11]);
        float2 qp_65 = _f2_51;
        sq[1] = fma_f32x2_rn_noftz(v_64, v_64, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_64, qp_65, dot[1]);
        float2 _f2_52 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 v_66 = _f2_52;
        float2 _f2_53 = make_float2(q[12], q[13]);
        float2 qp_67 = _f2_53;
        sq[1] = fma_f32x2_rn_noftz(v_66, v_66, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_66, qp_67, dot[1]);
        float2 _f2_54 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 v_68 = _f2_54;
        float2 _f2_55 = make_float2(q[14], q[15]);
        float2 qp_69 = _f2_55;
        sq[1] = fma_f32x2_rn_noftz(v_68, v_68, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_68, qp_69, dot[1]);
        unsigned int sw_70[4];
        sw_70[0] = words[28 + woff_51];
        sw_70[1] = words[28 + woff_51 + 1];
        sw_70[2] = words[28 + woff_51 + 2];
        sw_70[3] = words[28 + woff_51 + 3];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(sw_70[_pair]));
        }
        fsrc[64] = _unpack_bf16_6[0];
        fsrc[65] = _unpack_bf16_6[1];
        fsrc[66] = _unpack_bf16_6[2];
        fsrc[67] = _unpack_bf16_6[3];
        fsrc[68] = _unpack_bf16_6[4];
        fsrc[69] = _unpack_bf16_6[5];
        fsrc[70] = _unpack_bf16_6[6];
        fsrc[71] = _unpack_bf16_6[7];
        float2 _f2_56 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 v_71 = _f2_56;
        float2 _f2_57 = make_float2(q[8], q[9]);
        float2 qp_72 = _f2_57;
        sq[2] = fma_f32x2_rn_noftz(v_71, v_71, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_71, qp_72, dot[2]);
        float2 _f2_58 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 v_73 = _f2_58;
        float2 _f2_59 = make_float2(q[10], q[11]);
        float2 qp_74 = _f2_59;
        sq[2] = fma_f32x2_rn_noftz(v_73, v_73, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_73, qp_74, dot[2]);
        float2 _f2_60 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 v_75 = _f2_60;
        float2 _f2_61 = make_float2(q[12], q[13]);
        float2 qp_76 = _f2_61;
        sq[2] = fma_f32x2_rn_noftz(v_75, v_75, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_75, qp_76, dot[2]);
        float2 _f2_62 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 v_77 = _f2_62;
        float2 _f2_63 = make_float2(q[14], q[15]);
        float2 qp_78 = _f2_63;
        sq[2] = fma_f32x2_rn_noftz(v_77, v_77, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_77, qp_78, dot[2]);
        unsigned int sw_79[4];
        sw_79[0] = words[42 + woff_51];
        sw_79[1] = words[42 + woff_51 + 1];
        sw_79[2] = words[42 + woff_51 + 2];
        sw_79[3] = words[42 + woff_51 + 3];
        __nv_bfloat162 a_80 = __as_bf16x2(sw_79[0]);
        __nv_bfloat162 d_81 = __as_bf16x2(dwords[woff_51]);
        __nv_bfloat162 mixed_82 = a_80 + d_81;
        sw_79[0] = __as_u32(mixed_82);
        words[42 + woff_51] = sw_79[0];
        __nv_bfloat162 a_83 = __as_bf16x2(sw_79[1]);
        __nv_bfloat162 d_84 = __as_bf16x2(dwords[woff_51 + 1]);
        __nv_bfloat162 mixed_85 = a_83 + d_84;
        sw_79[1] = __as_u32(mixed_85);
        words[42 + woff_51 + 1] = sw_79[1];
        __nv_bfloat162 a_86 = __as_bf16x2(sw_79[2]);
        __nv_bfloat162 d_87 = __as_bf16x2(dwords[woff_51 + 2]);
        __nv_bfloat162 mixed_88 = a_86 + d_87;
        sw_79[2] = __as_u32(mixed_88);
        words[42 + woff_51 + 2] = sw_79[2];
        __nv_bfloat162 a_89 = __as_bf16x2(sw_79[3]);
        __nv_bfloat162 d_90 = __as_bf16x2(dwords[woff_51 + 3]);
        __nv_bfloat162 mixed_91 = a_89 + d_90;
        sw_79[3] = __as_u32(mixed_91);
        words[42 + woff_51 + 3] = sw_79[3];
        {
            int4 _iv4 = make_int4(sw_79[0 + 0], sw_79[0 + 1], sw_79[0 + 2], sw_79[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_50)) + 0) = _iv4;
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
                : "r"(sw_79[_pair]));
        }
        float2 _f2_64 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 v_92 = _f2_64;
        float2 _f2_65 = make_float2(q[8], q[9]);
        float2 qp_93 = _f2_65;
        sq[3] = fma_f32x2_rn_noftz(v_92, v_92, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_92, qp_93, dot[3]);
        float2 _f2_66 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 v_94 = _f2_66;
        float2 _f2_67 = make_float2(q[10], q[11]);
        float2 qp_95 = _f2_67;
        sq[3] = fma_f32x2_rn_noftz(v_94, v_94, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_94, qp_95, dot[3]);
        float2 _f2_68 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 v_96 = _f2_68;
        float2 _f2_69 = make_float2(q[12], q[13]);
        float2 qp_97 = _f2_69;
        sq[3] = fma_f32x2_rn_noftz(v_96, v_96, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_96, qp_97, dot[3]);
        float2 _f2_70 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 v_98 = _f2_70;
        float2 _f2_71 = make_float2(q[14], q[15]);
        float2 qp_99 = _f2_71;
        sq[3] = fma_f32x2_rn_noftz(v_98, v_98, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_98, qp_99, dot[3]);
        int base_100 = (4 + group) * 1024 + thread * 8;
        const int woff_101 = 8;
        unsigned int sw_102[4];
        sw_102[0] = words[woff_101];
        sw_102[1] = words[woff_101 + 1];
        sw_102[2] = words[woff_101 + 2];
        sw_102[3] = words[woff_101 + 3];
        float _unpack_bf16_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_8[_pair * 2])[0]), "=f"((&_unpack_bf16_8[_pair * 2])[1])
                : "r"(sw_102[_pair]));
        }
        fsrc[16] = _unpack_bf16_8[0];
        fsrc[17] = _unpack_bf16_8[1];
        fsrc[18] = _unpack_bf16_8[2];
        fsrc[19] = _unpack_bf16_8[3];
        fsrc[20] = _unpack_bf16_8[4];
        fsrc[21] = _unpack_bf16_8[5];
        fsrc[22] = _unpack_bf16_8[6];
        fsrc[23] = _unpack_bf16_8[7];
        float2 _f2_72 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 v_103 = _f2_72;
        float2 _f2_73 = make_float2(q[16], q[17]);
        float2 qp_104 = _f2_73;
        sq[0] = fma_f32x2_rn_noftz(v_103, v_103, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_103, qp_104, dot[0]);
        float2 _f2_74 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 v_105 = _f2_74;
        float2 _f2_75 = make_float2(q[18], q[19]);
        float2 qp_106 = _f2_75;
        sq[0] = fma_f32x2_rn_noftz(v_105, v_105, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_105, qp_106, dot[0]);
        float2 _f2_76 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 v_107 = _f2_76;
        float2 _f2_77 = make_float2(q[20], q[21]);
        float2 qp_108 = _f2_77;
        sq[0] = fma_f32x2_rn_noftz(v_107, v_107, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_107, qp_108, dot[0]);
        float2 _f2_78 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 v_109 = _f2_78;
        float2 _f2_79 = make_float2(q[22], q[23]);
        float2 qp_110 = _f2_79;
        sq[0] = fma_f32x2_rn_noftz(v_109, v_109, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_109, qp_110, dot[0]);
        unsigned int sw_111[4];
        sw_111[0] = words[14 + woff_101];
        sw_111[1] = words[14 + woff_101 + 1];
        sw_111[2] = words[14 + woff_101 + 2];
        sw_111[3] = words[14 + woff_101 + 3];
        float _unpack_bf16_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                : "r"(sw_111[_pair]));
        }
        fsrc[44] = _unpack_bf16_9[0];
        fsrc[45] = _unpack_bf16_9[1];
        fsrc[46] = _unpack_bf16_9[2];
        fsrc[47] = _unpack_bf16_9[3];
        fsrc[48] = _unpack_bf16_9[4];
        fsrc[49] = _unpack_bf16_9[5];
        fsrc[50] = _unpack_bf16_9[6];
        fsrc[51] = _unpack_bf16_9[7];
        float2 _f2_80 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 v_112 = _f2_80;
        float2 _f2_81 = make_float2(q[16], q[17]);
        float2 qp_113 = _f2_81;
        sq[1] = fma_f32x2_rn_noftz(v_112, v_112, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_112, qp_113, dot[1]);
        float2 _f2_82 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 v_114 = _f2_82;
        float2 _f2_83 = make_float2(q[18], q[19]);
        float2 qp_115 = _f2_83;
        sq[1] = fma_f32x2_rn_noftz(v_114, v_114, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_114, qp_115, dot[1]);
        float2 _f2_84 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 v_116 = _f2_84;
        float2 _f2_85 = make_float2(q[20], q[21]);
        float2 qp_117 = _f2_85;
        sq[1] = fma_f32x2_rn_noftz(v_116, v_116, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_116, qp_117, dot[1]);
        float2 _f2_86 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 v_118 = _f2_86;
        float2 _f2_87 = make_float2(q[22], q[23]);
        float2 qp_119 = _f2_87;
        sq[1] = fma_f32x2_rn_noftz(v_118, v_118, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_118, qp_119, dot[1]);
        unsigned int sw_120[4];
        sw_120[0] = words[28 + woff_101];
        sw_120[1] = words[28 + woff_101 + 1];
        sw_120[2] = words[28 + woff_101 + 2];
        sw_120[3] = words[28 + woff_101 + 3];
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                : "r"(sw_120[_pair]));
        }
        fsrc[72] = _unpack_bf16_10[0];
        fsrc[73] = _unpack_bf16_10[1];
        fsrc[74] = _unpack_bf16_10[2];
        fsrc[75] = _unpack_bf16_10[3];
        fsrc[76] = _unpack_bf16_10[4];
        fsrc[77] = _unpack_bf16_10[5];
        fsrc[78] = _unpack_bf16_10[6];
        fsrc[79] = _unpack_bf16_10[7];
        float2 _f2_88 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 v_121 = _f2_88;
        float2 _f2_89 = make_float2(q[16], q[17]);
        float2 qp_122 = _f2_89;
        sq[2] = fma_f32x2_rn_noftz(v_121, v_121, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_121, qp_122, dot[2]);
        float2 _f2_90 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 v_123 = _f2_90;
        float2 _f2_91 = make_float2(q[18], q[19]);
        float2 qp_124 = _f2_91;
        sq[2] = fma_f32x2_rn_noftz(v_123, v_123, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_123, qp_124, dot[2]);
        float2 _f2_92 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 v_125 = _f2_92;
        float2 _f2_93 = make_float2(q[20], q[21]);
        float2 qp_126 = _f2_93;
        sq[2] = fma_f32x2_rn_noftz(v_125, v_125, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_125, qp_126, dot[2]);
        float2 _f2_94 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 v_127 = _f2_94;
        float2 _f2_95 = make_float2(q[22], q[23]);
        float2 qp_128 = _f2_95;
        sq[2] = fma_f32x2_rn_noftz(v_127, v_127, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_127, qp_128, dot[2]);
        unsigned int sw_129[4];
        sw_129[0] = words[42 + woff_101];
        sw_129[1] = words[42 + woff_101 + 1];
        sw_129[2] = words[42 + woff_101 + 2];
        sw_129[3] = words[42 + woff_101 + 3];
        __nv_bfloat162 a_130 = __as_bf16x2(sw_129[0]);
        __nv_bfloat162 d_131 = __as_bf16x2(dwords[woff_101]);
        __nv_bfloat162 mixed_132 = a_130 + d_131;
        sw_129[0] = __as_u32(mixed_132);
        words[42 + woff_101] = sw_129[0];
        __nv_bfloat162 a_133 = __as_bf16x2(sw_129[1]);
        __nv_bfloat162 d_134 = __as_bf16x2(dwords[woff_101 + 1]);
        __nv_bfloat162 mixed_135 = a_133 + d_134;
        sw_129[1] = __as_u32(mixed_135);
        words[42 + woff_101 + 1] = sw_129[1];
        __nv_bfloat162 a_136 = __as_bf16x2(sw_129[2]);
        __nv_bfloat162 d_137 = __as_bf16x2(dwords[woff_101 + 2]);
        __nv_bfloat162 mixed_138 = a_136 + d_137;
        sw_129[2] = __as_u32(mixed_138);
        words[42 + woff_101 + 2] = sw_129[2];
        __nv_bfloat162 a_139 = __as_bf16x2(sw_129[3]);
        __nv_bfloat162 d_140 = __as_bf16x2(dwords[woff_101 + 3]);
        __nv_bfloat162 mixed_141 = a_139 + d_140;
        sw_129[3] = __as_u32(mixed_141);
        words[42 + woff_101 + 3] = sw_129[3];
        {
            int4 _iv4 = make_int4(sw_129[0 + 0], sw_129[0 + 1], sw_129[0 + 2], sw_129[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_100)) + 0) = _iv4;
        }
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                : "r"(sw_129[_pair]));
        }
        float2 _f2_96 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 v_142 = _f2_96;
        float2 _f2_97 = make_float2(q[16], q[17]);
        float2 qp_143 = _f2_97;
        sq[3] = fma_f32x2_rn_noftz(v_142, v_142, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_142, qp_143, dot[3]);
        float2 _f2_98 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 v_144 = _f2_98;
        float2 _f2_99 = make_float2(q[18], q[19]);
        float2 qp_145 = _f2_99;
        sq[3] = fma_f32x2_rn_noftz(v_144, v_144, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_144, qp_145, dot[3]);
        float2 _f2_100 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 v_146 = _f2_100;
        float2 _f2_101 = make_float2(q[20], q[21]);
        float2 qp_147 = _f2_101;
        sq[3] = fma_f32x2_rn_noftz(v_146, v_146, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_146, qp_147, dot[3]);
        float2 _f2_102 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 v_148 = _f2_102;
        float2 _f2_103 = make_float2(q[22], q[23]);
        float2 qp_149 = _f2_103;
        sq[3] = fma_f32x2_rn_noftz(v_148, v_148, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_148, qp_149, dot[3]);
        int base_150 = 6144 + group * 512 + thread * 4;
        const int woff_151 = 12;
        unsigned int sw_152[4];
        sw_152[0] = words[woff_151];
        sw_152[1] = words[woff_151 + 1];
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                : "r"(sw_152[_pair]));
        }
        fsrc[24] = _unpack_bf16_12[0];
        fsrc[25] = _unpack_bf16_12[1];
        fsrc[26] = _unpack_bf16_12[2];
        fsrc[27] = _unpack_bf16_12[3];
        float2 _f2_104 = make_float2(_unpack_bf16_12[0], _unpack_bf16_12[1]);
        float2 v_153 = _f2_104;
        sq[0] = fma_f32x2_rn_noftz(v_153, v_153, sq[0]);
        float2 _f2_105 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 v_154 = _f2_105;
        sq[0] = fma_f32x2_rn_noftz(v_154, v_154, sq[0]);
        float2 _f2_106 = make_float2(_unpack_bf16_12[0], _unpack_bf16_12[1]);
        float2 v_155 = _f2_106;
        float2 _f2_107 = make_float2(q[24], q[25]);
        float2 qp_156 = _f2_107;
        dot[0] = fma_f32x2_rn_noftz(v_155, qp_156, dot[0]);
        float2 _f2_108 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 v_157 = _f2_108;
        float2 _f2_109 = make_float2(q[26], q[27]);
        float2 qp_158 = _f2_109;
        dot[0] = fma_f32x2_rn_noftz(v_157, qp_158, dot[0]);
        unsigned int sw_159[4];
        sw_159[0] = words[14 + woff_151];
        sw_159[1] = words[14 + woff_151 + 1];
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                : "r"(sw_159[_pair]));
        }
        fsrc[52] = _unpack_bf16_13[0];
        fsrc[53] = _unpack_bf16_13[1];
        fsrc[54] = _unpack_bf16_13[2];
        fsrc[55] = _unpack_bf16_13[3];
        float2 _f2_110 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 v_160 = _f2_110;
        sq[1] = fma_f32x2_rn_noftz(v_160, v_160, sq[1]);
        float2 _f2_111 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 v_161 = _f2_111;
        sq[1] = fma_f32x2_rn_noftz(v_161, v_161, sq[1]);
        float2 _f2_112 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 v_162 = _f2_112;
        float2 _f2_113 = make_float2(q[24], q[25]);
        float2 qp_163 = _f2_113;
        dot[1] = fma_f32x2_rn_noftz(v_162, qp_163, dot[1]);
        float2 _f2_114 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 v_164 = _f2_114;
        float2 _f2_115 = make_float2(q[26], q[27]);
        float2 qp_165 = _f2_115;
        dot[1] = fma_f32x2_rn_noftz(v_164, qp_165, dot[1]);
        unsigned int sw_166[4];
        sw_166[0] = words[28 + woff_151];
        sw_166[1] = words[28 + woff_151 + 1];
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                : "r"(sw_166[_pair]));
        }
        fsrc[80] = _unpack_bf16_14[0];
        fsrc[81] = _unpack_bf16_14[1];
        fsrc[82] = _unpack_bf16_14[2];
        fsrc[83] = _unpack_bf16_14[3];
        float2 _f2_116 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 v_167 = _f2_116;
        sq[2] = fma_f32x2_rn_noftz(v_167, v_167, sq[2]);
        float2 _f2_117 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 v_168 = _f2_117;
        sq[2] = fma_f32x2_rn_noftz(v_168, v_168, sq[2]);
        float2 _f2_118 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 v_169 = _f2_118;
        float2 _f2_119 = make_float2(q[24], q[25]);
        float2 qp_170 = _f2_119;
        dot[2] = fma_f32x2_rn_noftz(v_169, qp_170, dot[2]);
        float2 _f2_120 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 v_171 = _f2_120;
        float2 _f2_121 = make_float2(q[26], q[27]);
        float2 qp_172 = _f2_121;
        dot[2] = fma_f32x2_rn_noftz(v_171, qp_172, dot[2]);
        unsigned int sw_173[4];
        sw_173[0] = words[42 + woff_151];
        sw_173[1] = words[42 + woff_151 + 1];
        __nv_bfloat162 a_174 = __as_bf16x2(sw_173[0]);
        __nv_bfloat162 d_175 = __as_bf16x2(dwords[woff_151]);
        __nv_bfloat162 mixed_176 = a_174 + d_175;
        sw_173[0] = __as_u32(mixed_176);
        words[42 + woff_151] = sw_173[0];
        __nv_bfloat162 a_177 = __as_bf16x2(sw_173[1]);
        __nv_bfloat162 d_178 = __as_bf16x2(dwords[woff_151 + 1]);
        __nv_bfloat162 mixed_179 = a_177 + d_178;
        sw_173[1] = __as_u32(mixed_179);
        words[42 + woff_151 + 1] = sw_173[1];
        {
            int2 _iv2 = make_int2(sw_173[0 + 0], sw_173[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_150)) + 0) = _iv2;
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
                : "r"(sw_173[_pair]));
        }
        float2 _f2_122 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 v_180 = _f2_122;
        sq[3] = fma_f32x2_rn_noftz(v_180, v_180, sq[3]);
        float2 _f2_123 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 v_181 = _f2_123;
        sq[3] = fma_f32x2_rn_noftz(v_181, v_181, sq[3]);
        float2 _f2_124 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 v_182 = _f2_124;
        float2 _f2_125 = make_float2(q[24], q[25]);
        float2 qp_183 = _f2_125;
        dot[3] = fma_f32x2_rn_noftz(v_182, qp_183, dot[3]);
        float2 _f2_126 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 v_184 = _f2_126;
        float2 _f2_127 = make_float2(q[26], q[27]);
        float2 qp_185 = _f2_127;
        dot[3] = fma_f32x2_rn_noftz(v_184, qp_185, dot[3]);
        float2 pairs[4];
        float2 _f2_128 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_128;
        float2 _f2_129 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_129;
        float2 _f2_130 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_130;
        float2 _f2_131 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_131;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_132 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_132;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_186 = 0;
        bits_186 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_186, 16);
        unsigned long long peerbits_187 = _shfl_xor_1;
        float2 _f2_133 = make_float2(0.0f, 0.0f);
        float2 peer_188 = _f2_133;
        peer_188 = reinterpret_cast<float2*>(&peerbits_187)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_188);
        unsigned long long bits_189 = 0;
        bits_189 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_189, 16);
        unsigned long long peerbits_190 = _shfl_xor_2;
        float2 _f2_134 = make_float2(0.0f, 0.0f);
        float2 peer_191 = _f2_134;
        peer_191 = reinterpret_cast<float2*>(&peerbits_190)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_191);
        unsigned long long bits_192 = 0;
        bits_192 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_192, 16);
        unsigned long long peerbits_193 = _shfl_xor_3;
        float2 _f2_135 = make_float2(0.0f, 0.0f);
        float2 peer_194 = _f2_135;
        peer_194 = reinterpret_cast<float2*>(&peerbits_193)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_194);
        unsigned long long bits_195 = 0;
        bits_195 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_195, 8);
        unsigned long long peerbits_196 = _shfl_xor_4;
        float2 _f2_136 = make_float2(0.0f, 0.0f);
        float2 peer_197 = _f2_136;
        peer_197 = reinterpret_cast<float2*>(&peerbits_196)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_197);
        unsigned long long bits_198 = 0;
        bits_198 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_198, 8);
        unsigned long long peerbits_199 = _shfl_xor_5;
        float2 _f2_137 = make_float2(0.0f, 0.0f);
        float2 peer_200 = _f2_137;
        peer_200 = reinterpret_cast<float2*>(&peerbits_199)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_200);
        unsigned long long bits_201 = 0;
        bits_201 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_201, 8);
        unsigned long long peerbits_202 = _shfl_xor_6;
        float2 _f2_138 = make_float2(0.0f, 0.0f);
        float2 peer_203 = _f2_138;
        peer_203 = reinterpret_cast<float2*>(&peerbits_202)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_203);
        unsigned long long bits_204 = 0;
        bits_204 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_204, 8);
        unsigned long long peerbits_205 = _shfl_xor_7;
        float2 _f2_139 = make_float2(0.0f, 0.0f);
        float2 peer_206 = _f2_139;
        peer_206 = reinterpret_cast<float2*>(&peerbits_205)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_206);
        unsigned long long bits_207 = 0;
        bits_207 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_207, 4);
        unsigned long long peerbits_208 = _shfl_xor_8;
        float2 _f2_140 = make_float2(0.0f, 0.0f);
        float2 peer_209 = _f2_140;
        peer_209 = reinterpret_cast<float2*>(&peerbits_208)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_209);
        unsigned long long bits_210 = 0;
        bits_210 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_210, 4);
        unsigned long long peerbits_211 = _shfl_xor_9;
        float2 _f2_141 = make_float2(0.0f, 0.0f);
        float2 peer_212 = _f2_141;
        peer_212 = reinterpret_cast<float2*>(&peerbits_211)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_212);
        unsigned long long bits_213 = 0;
        bits_213 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_213, 4);
        unsigned long long peerbits_214 = _shfl_xor_10;
        float2 _f2_142 = make_float2(0.0f, 0.0f);
        float2 peer_215 = _f2_142;
        peer_215 = reinterpret_cast<float2*>(&peerbits_214)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_215);
        unsigned long long bits_216 = 0;
        bits_216 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_216, 4);
        unsigned long long peerbits_217 = _shfl_xor_11;
        float2 _f2_143 = make_float2(0.0f, 0.0f);
        float2 peer_218 = _f2_143;
        peer_218 = reinterpret_cast<float2*>(&peerbits_217)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_218);
        unsigned long long bits_219 = 0;
        bits_219 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_219, 2);
        unsigned long long peerbits_220 = _shfl_xor_12;
        float2 _f2_144 = make_float2(0.0f, 0.0f);
        float2 peer_221 = _f2_144;
        peer_221 = reinterpret_cast<float2*>(&peerbits_220)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_221);
        unsigned long long bits_222 = 0;
        bits_222 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_222, 2);
        unsigned long long peerbits_223 = _shfl_xor_13;
        float2 _f2_145 = make_float2(0.0f, 0.0f);
        float2 peer_224 = _f2_145;
        peer_224 = reinterpret_cast<float2*>(&peerbits_223)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_224);
        unsigned long long bits_225 = 0;
        bits_225 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_225, 2);
        unsigned long long peerbits_226 = _shfl_xor_14;
        float2 _f2_146 = make_float2(0.0f, 0.0f);
        float2 peer_227 = _f2_146;
        peer_227 = reinterpret_cast<float2*>(&peerbits_226)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_227);
        unsigned long long bits_228 = 0;
        bits_228 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_228, 2);
        unsigned long long peerbits_229 = _shfl_xor_15;
        float2 _f2_147 = make_float2(0.0f, 0.0f);
        float2 peer_230 = _f2_147;
        peer_230 = reinterpret_cast<float2*>(&peerbits_229)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_230);
        unsigned long long bits_231 = 0;
        bits_231 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_231, 1);
        unsigned long long peerbits_232 = _shfl_xor_16;
        float2 _f2_148 = make_float2(0.0f, 0.0f);
        float2 peer_233 = _f2_148;
        peer_233 = reinterpret_cast<float2*>(&peerbits_232)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_233);
        unsigned long long bits_234 = 0;
        bits_234 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_234, 1);
        unsigned long long peerbits_235 = _shfl_xor_17;
        float2 _f2_149 = make_float2(0.0f, 0.0f);
        float2 peer_236 = _f2_149;
        peer_236 = reinterpret_cast<float2*>(&peerbits_235)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_236);
        unsigned long long bits_237 = 0;
        bits_237 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_237, 1);
        unsigned long long peerbits_238 = _shfl_xor_18;
        float2 _f2_150 = make_float2(0.0f, 0.0f);
        float2 peer_239 = _f2_150;
        peer_239 = reinterpret_cast<float2*>(&peerbits_238)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_239);
        unsigned long long bits_240 = 0;
        bits_240 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_240, 1);
        unsigned long long peerbits_241 = _shfl_xor_19;
        float2 _f2_151 = make_float2(0.0f, 0.0f);
        float2 peer_242 = _f2_151;
        peer_242 = reinterpret_cast<float2*>(&peerbits_241)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_242);
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
        float2 _f2_152 = make_float2(correction, correction);
        float2 corr = _f2_152;
        const int woff_243 = 0;
        int base_244 = group * 1024 + thread * 8;
        float2 a_245[4];
        float2 _f2_153 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_153;
        a_245[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_154 = make_float2(acc[2], acc[3]);
        float2 previous_246 = _f2_154;
        a_245[1] = mul_f32x2_noftz(previous_246, corr);
        float2 _f2_155 = make_float2(acc[4], acc[5]);
        float2 previous_247 = _f2_155;
        a_245[2] = mul_f32x2_noftz(previous_247, corr);
        float2 _f2_156 = make_float2(acc[6], acc[7]);
        float2 previous_248 = _f2_156;
        a_245[3] = mul_f32x2_noftz(previous_248, corr);
        float2 _f2_157 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_157;
        float2 _f2_158 = make_float2(fsrc[0], fsrc[1]);
        float2 v_249 = _f2_158;
        a_245[0] = fma_f32x2_rn_noftz(weight, v_249, a_245[0]);
        float2 _f2_159 = make_float2(fsrc[2], fsrc[3]);
        float2 v_250 = _f2_159;
        a_245[1] = fma_f32x2_rn_noftz(weight, v_250, a_245[1]);
        float2 _f2_160 = make_float2(fsrc[4], fsrc[5]);
        float2 v_251 = _f2_160;
        a_245[2] = fma_f32x2_rn_noftz(weight, v_251, a_245[2]);
        float2 _f2_161 = make_float2(fsrc[6], fsrc[7]);
        float2 v_252 = _f2_161;
        a_245[3] = fma_f32x2_rn_noftz(weight, v_252, a_245[3]);
        float2 _f2_162 = make_float2(weights[1], weights[1]);
        float2 weight_253 = _f2_162;
        float2 _f2_163 = make_float2(fsrc[28], fsrc[29]);
        float2 v_254 = _f2_163;
        a_245[0] = fma_f32x2_rn_noftz(weight_253, v_254, a_245[0]);
        float2 _f2_164 = make_float2(fsrc[30], fsrc[31]);
        float2 v_255 = _f2_164;
        a_245[1] = fma_f32x2_rn_noftz(weight_253, v_255, a_245[1]);
        float2 _f2_165 = make_float2(fsrc[32], fsrc[33]);
        float2 v_256 = _f2_165;
        a_245[2] = fma_f32x2_rn_noftz(weight_253, v_256, a_245[2]);
        float2 _f2_166 = make_float2(fsrc[34], fsrc[35]);
        float2 v_257 = _f2_166;
        a_245[3] = fma_f32x2_rn_noftz(weight_253, v_257, a_245[3]);
        float2 _f2_167 = make_float2(weights[2], weights[2]);
        float2 weight_258 = _f2_167;
        float2 _f2_168 = make_float2(fsrc[56], fsrc[57]);
        float2 v_259 = _f2_168;
        a_245[0] = fma_f32x2_rn_noftz(weight_258, v_259, a_245[0]);
        float2 _f2_169 = make_float2(fsrc[58], fsrc[59]);
        float2 v_260 = _f2_169;
        a_245[1] = fma_f32x2_rn_noftz(weight_258, v_260, a_245[1]);
        float2 _f2_170 = make_float2(fsrc[60], fsrc[61]);
        float2 v_261 = _f2_170;
        a_245[2] = fma_f32x2_rn_noftz(weight_258, v_261, a_245[2]);
        float2 _f2_171 = make_float2(fsrc[62], fsrc[63]);
        float2 v_262 = _f2_171;
        a_245[3] = fma_f32x2_rn_noftz(weight_258, v_262, a_245[3]);
        float2 _f2_172 = make_float2(weights[3], weights[3]);
        float2 weight_263 = _f2_172;
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_243];
        sw2[1] = words[42 + woff_243 + 1];
        sw2[2] = words[42 + woff_243 + 2];
        sw2[3] = words[42 + woff_243 + 3];
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(sw2[_pair]));
        }
        float2 _f2_173 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 v_264 = _f2_173;
        a_245[0] = fma_f32x2_rn_noftz(weight_263, v_264, a_245[0]);
        float2 _f2_174 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 v_265 = _f2_174;
        a_245[1] = fma_f32x2_rn_noftz(weight_263, v_265, a_245[1]);
        float2 _f2_175 = make_float2(_unpack_bf16_16[4], _unpack_bf16_16[5]);
        float2 v_266 = _f2_175;
        a_245[2] = fma_f32x2_rn_noftz(weight_263, v_266, a_245[2]);
        float2 _f2_176 = make_float2(_unpack_bf16_16[6], _unpack_bf16_16[7]);
        float2 v_267 = _f2_176;
        a_245[3] = fma_f32x2_rn_noftz(weight_263, v_267, a_245[3]);
        acc[0] = a_245[0].x;
        acc[1] = a_245[0].y;
        acc[2] = a_245[1].x;
        acc[3] = a_245[1].y;
        acc[4] = a_245[2].x;
        acc[5] = a_245[2].y;
        acc[6] = a_245[3].x;
        acc[7] = a_245[3].y;
        const int woff_268 = 4;
        int base_269 = (2 + group) * 1024 + thread * 8;
        float2 a_270[4];
        float2 _f2_177 = make_float2(acc[8], acc[9]);
        float2 previous_271 = _f2_177;
        a_270[0] = mul_f32x2_noftz(previous_271, corr);
        float2 _f2_178 = make_float2(acc[10], acc[11]);
        float2 previous_272 = _f2_178;
        a_270[1] = mul_f32x2_noftz(previous_272, corr);
        float2 _f2_179 = make_float2(acc[12], acc[13]);
        float2 previous_273 = _f2_179;
        a_270[2] = mul_f32x2_noftz(previous_273, corr);
        float2 _f2_180 = make_float2(acc[14], acc[15]);
        float2 previous_274 = _f2_180;
        a_270[3] = mul_f32x2_noftz(previous_274, corr);
        float2 _f2_181 = make_float2(weights[0], weights[0]);
        float2 weight_275 = _f2_181;
        float2 _f2_182 = make_float2(fsrc[8], fsrc[9]);
        float2 v_276 = _f2_182;
        a_270[0] = fma_f32x2_rn_noftz(weight_275, v_276, a_270[0]);
        float2 _f2_183 = make_float2(fsrc[10], fsrc[11]);
        float2 v_277 = _f2_183;
        a_270[1] = fma_f32x2_rn_noftz(weight_275, v_277, a_270[1]);
        float2 _f2_184 = make_float2(fsrc[12], fsrc[13]);
        float2 v_278 = _f2_184;
        a_270[2] = fma_f32x2_rn_noftz(weight_275, v_278, a_270[2]);
        float2 _f2_185 = make_float2(fsrc[14], fsrc[15]);
        float2 v_279 = _f2_185;
        a_270[3] = fma_f32x2_rn_noftz(weight_275, v_279, a_270[3]);
        float2 _f2_186 = make_float2(weights[1], weights[1]);
        float2 weight_280 = _f2_186;
        float2 _f2_187 = make_float2(fsrc[36], fsrc[37]);
        float2 v_281 = _f2_187;
        a_270[0] = fma_f32x2_rn_noftz(weight_280, v_281, a_270[0]);
        float2 _f2_188 = make_float2(fsrc[38], fsrc[39]);
        float2 v_282 = _f2_188;
        a_270[1] = fma_f32x2_rn_noftz(weight_280, v_282, a_270[1]);
        float2 _f2_189 = make_float2(fsrc[40], fsrc[41]);
        float2 v_283 = _f2_189;
        a_270[2] = fma_f32x2_rn_noftz(weight_280, v_283, a_270[2]);
        float2 _f2_190 = make_float2(fsrc[42], fsrc[43]);
        float2 v_284 = _f2_190;
        a_270[3] = fma_f32x2_rn_noftz(weight_280, v_284, a_270[3]);
        float2 _f2_191 = make_float2(weights[2], weights[2]);
        float2 weight_285 = _f2_191;
        float2 _f2_192 = make_float2(fsrc[64], fsrc[65]);
        float2 v_286 = _f2_192;
        a_270[0] = fma_f32x2_rn_noftz(weight_285, v_286, a_270[0]);
        float2 _f2_193 = make_float2(fsrc[66], fsrc[67]);
        float2 v_287 = _f2_193;
        a_270[1] = fma_f32x2_rn_noftz(weight_285, v_287, a_270[1]);
        float2 _f2_194 = make_float2(fsrc[68], fsrc[69]);
        float2 v_288 = _f2_194;
        a_270[2] = fma_f32x2_rn_noftz(weight_285, v_288, a_270[2]);
        float2 _f2_195 = make_float2(fsrc[70], fsrc[71]);
        float2 v_289 = _f2_195;
        a_270[3] = fma_f32x2_rn_noftz(weight_285, v_289, a_270[3]);
        float2 _f2_196 = make_float2(weights[3], weights[3]);
        float2 weight_290 = _f2_196;
        unsigned int sw2_291[4];
        sw2_291[0] = words[42 + woff_268];
        sw2_291[1] = words[42 + woff_268 + 1];
        sw2_291[2] = words[42 + woff_268 + 2];
        sw2_291[3] = words[42 + woff_268 + 3];
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(sw2_291[_pair]));
        }
        float2 _f2_197 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 v_292 = _f2_197;
        a_270[0] = fma_f32x2_rn_noftz(weight_290, v_292, a_270[0]);
        float2 _f2_198 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 v_293 = _f2_198;
        a_270[1] = fma_f32x2_rn_noftz(weight_290, v_293, a_270[1]);
        float2 _f2_199 = make_float2(_unpack_bf16_17[4], _unpack_bf16_17[5]);
        float2 v_294 = _f2_199;
        a_270[2] = fma_f32x2_rn_noftz(weight_290, v_294, a_270[2]);
        float2 _f2_200 = make_float2(_unpack_bf16_17[6], _unpack_bf16_17[7]);
        float2 v_295 = _f2_200;
        a_270[3] = fma_f32x2_rn_noftz(weight_290, v_295, a_270[3]);
        acc[8] = a_270[0].x;
        acc[9] = a_270[0].y;
        acc[10] = a_270[1].x;
        acc[11] = a_270[1].y;
        acc[12] = a_270[2].x;
        acc[13] = a_270[2].y;
        acc[14] = a_270[3].x;
        acc[15] = a_270[3].y;
        const int woff_296 = 8;
        int base_297 = (4 + group) * 1024 + thread * 8;
        float2 a_298[4];
        float2 _f2_201 = make_float2(acc[16], acc[17]);
        float2 previous_299 = _f2_201;
        a_298[0] = mul_f32x2_noftz(previous_299, corr);
        float2 _f2_202 = make_float2(acc[18], acc[19]);
        float2 previous_300 = _f2_202;
        a_298[1] = mul_f32x2_noftz(previous_300, corr);
        float2 _f2_203 = make_float2(acc[20], acc[21]);
        float2 previous_301 = _f2_203;
        a_298[2] = mul_f32x2_noftz(previous_301, corr);
        float2 _f2_204 = make_float2(acc[22], acc[23]);
        float2 previous_302 = _f2_204;
        a_298[3] = mul_f32x2_noftz(previous_302, corr);
        float2 _f2_205 = make_float2(weights[0], weights[0]);
        float2 weight_303 = _f2_205;
        float2 _f2_206 = make_float2(fsrc[16], fsrc[17]);
        float2 v_304 = _f2_206;
        a_298[0] = fma_f32x2_rn_noftz(weight_303, v_304, a_298[0]);
        float2 _f2_207 = make_float2(fsrc[18], fsrc[19]);
        float2 v_305 = _f2_207;
        a_298[1] = fma_f32x2_rn_noftz(weight_303, v_305, a_298[1]);
        float2 _f2_208 = make_float2(fsrc[20], fsrc[21]);
        float2 v_306 = _f2_208;
        a_298[2] = fma_f32x2_rn_noftz(weight_303, v_306, a_298[2]);
        float2 _f2_209 = make_float2(fsrc[22], fsrc[23]);
        float2 v_307 = _f2_209;
        a_298[3] = fma_f32x2_rn_noftz(weight_303, v_307, a_298[3]);
        float2 _f2_210 = make_float2(weights[1], weights[1]);
        float2 weight_308 = _f2_210;
        float2 _f2_211 = make_float2(fsrc[44], fsrc[45]);
        float2 v_309 = _f2_211;
        a_298[0] = fma_f32x2_rn_noftz(weight_308, v_309, a_298[0]);
        float2 _f2_212 = make_float2(fsrc[46], fsrc[47]);
        float2 v_310 = _f2_212;
        a_298[1] = fma_f32x2_rn_noftz(weight_308, v_310, a_298[1]);
        float2 _f2_213 = make_float2(fsrc[48], fsrc[49]);
        float2 v_311 = _f2_213;
        a_298[2] = fma_f32x2_rn_noftz(weight_308, v_311, a_298[2]);
        float2 _f2_214 = make_float2(fsrc[50], fsrc[51]);
        float2 v_312 = _f2_214;
        a_298[3] = fma_f32x2_rn_noftz(weight_308, v_312, a_298[3]);
        float2 _f2_215 = make_float2(weights[2], weights[2]);
        float2 weight_313 = _f2_215;
        float2 _f2_216 = make_float2(fsrc[72], fsrc[73]);
        float2 v_314 = _f2_216;
        a_298[0] = fma_f32x2_rn_noftz(weight_313, v_314, a_298[0]);
        float2 _f2_217 = make_float2(fsrc[74], fsrc[75]);
        float2 v_315 = _f2_217;
        a_298[1] = fma_f32x2_rn_noftz(weight_313, v_315, a_298[1]);
        float2 _f2_218 = make_float2(fsrc[76], fsrc[77]);
        float2 v_316 = _f2_218;
        a_298[2] = fma_f32x2_rn_noftz(weight_313, v_316, a_298[2]);
        float2 _f2_219 = make_float2(fsrc[78], fsrc[79]);
        float2 v_317 = _f2_219;
        a_298[3] = fma_f32x2_rn_noftz(weight_313, v_317, a_298[3]);
        float2 _f2_220 = make_float2(weights[3], weights[3]);
        float2 weight_318 = _f2_220;
        unsigned int sw2_319[4];
        sw2_319[0] = words[42 + woff_296];
        sw2_319[1] = words[42 + woff_296 + 1];
        sw2_319[2] = words[42 + woff_296 + 2];
        sw2_319[3] = words[42 + woff_296 + 3];
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                : "r"(sw2_319[_pair]));
        }
        float2 _f2_221 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 v_320 = _f2_221;
        a_298[0] = fma_f32x2_rn_noftz(weight_318, v_320, a_298[0]);
        float2 _f2_222 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 v_321 = _f2_222;
        a_298[1] = fma_f32x2_rn_noftz(weight_318, v_321, a_298[1]);
        float2 _f2_223 = make_float2(_unpack_bf16_18[4], _unpack_bf16_18[5]);
        float2 v_322 = _f2_223;
        a_298[2] = fma_f32x2_rn_noftz(weight_318, v_322, a_298[2]);
        float2 _f2_224 = make_float2(_unpack_bf16_18[6], _unpack_bf16_18[7]);
        float2 v_323 = _f2_224;
        a_298[3] = fma_f32x2_rn_noftz(weight_318, v_323, a_298[3]);
        acc[16] = a_298[0].x;
        acc[17] = a_298[0].y;
        acc[18] = a_298[1].x;
        acc[19] = a_298[1].y;
        acc[20] = a_298[2].x;
        acc[21] = a_298[2].y;
        acc[22] = a_298[3].x;
        acc[23] = a_298[3].y;
        const int woff_324 = 12;
        int base_325 = 6144 + group * 512 + thread * 4;
        float2 a_326[4];
        float2 _f2_225 = make_float2(acc[24], acc[25]);
        float2 previous_327 = _f2_225;
        a_326[0] = mul_f32x2_noftz(previous_327, corr);
        float2 _f2_226 = make_float2(acc[26], acc[27]);
        float2 previous_328 = _f2_226;
        a_326[1] = mul_f32x2_noftz(previous_328, corr);
        float2 _f2_227 = make_float2(weights[0], weights[0]);
        float2 weight_329 = _f2_227;
        float2 _f2_228 = make_float2(fsrc[24], fsrc[25]);
        float2 v_330 = _f2_228;
        a_326[0] = fma_f32x2_rn_noftz(weight_329, v_330, a_326[0]);
        float2 _f2_229 = make_float2(fsrc[26], fsrc[27]);
        float2 v_331 = _f2_229;
        a_326[1] = fma_f32x2_rn_noftz(weight_329, v_331, a_326[1]);
        float2 _f2_230 = make_float2(weights[1], weights[1]);
        float2 weight_332 = _f2_230;
        float2 _f2_231 = make_float2(fsrc[52], fsrc[53]);
        float2 v_333 = _f2_231;
        a_326[0] = fma_f32x2_rn_noftz(weight_332, v_333, a_326[0]);
        float2 _f2_232 = make_float2(fsrc[54], fsrc[55]);
        float2 v_334 = _f2_232;
        a_326[1] = fma_f32x2_rn_noftz(weight_332, v_334, a_326[1]);
        float2 _f2_233 = make_float2(weights[2], weights[2]);
        float2 weight_335 = _f2_233;
        float2 _f2_234 = make_float2(fsrc[80], fsrc[81]);
        float2 v_336 = _f2_234;
        a_326[0] = fma_f32x2_rn_noftz(weight_335, v_336, a_326[0]);
        float2 _f2_235 = make_float2(fsrc[82], fsrc[83]);
        float2 v_337 = _f2_235;
        a_326[1] = fma_f32x2_rn_noftz(weight_335, v_337, a_326[1]);
        float2 _f2_236 = make_float2(weights[3], weights[3]);
        float2 weight_338 = _f2_236;
        unsigned int sw2_339[4];
        sw2_339[0] = words[42 + woff_324];
        sw2_339[1] = words[42 + woff_324 + 1];
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                : "r"(sw2_339[_pair]));
        }
        float2 _f2_237 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 v_340 = _f2_237;
        a_326[0] = fma_f32x2_rn_noftz(weight_338, v_340, a_326[0]);
        float2 _f2_238 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 v_341 = _f2_238;
        a_326[1] = fma_f32x2_rn_noftz(weight_338, v_341, a_326[1]);
        acc[24] = a_326[0].x;
        acc[25] = a_326[0].y;
        acc[26] = a_326[1].x;
        acc[27] = a_326[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float2 _f2_239 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_239;
        float2 _f2_240 = make_float2(acc[0], acc[1]);
        float2 v_342 = _f2_240;
        output_sq_pair = fma_f32x2_rn_noftz(v_342, v_342, output_sq_pair);
        float2 _f2_241 = make_float2(acc[2], acc[3]);
        float2 v_343 = _f2_241;
        output_sq_pair = fma_f32x2_rn_noftz(v_343, v_343, output_sq_pair);
        float2 _f2_242 = make_float2(acc[4], acc[5]);
        float2 v_344 = _f2_242;
        output_sq_pair = fma_f32x2_rn_noftz(v_344, v_344, output_sq_pair);
        float2 _f2_243 = make_float2(acc[6], acc[7]);
        float2 v_345 = _f2_243;
        output_sq_pair = fma_f32x2_rn_noftz(v_345, v_345, output_sq_pair);
        float2 _f2_244 = make_float2(acc[8], acc[9]);
        float2 v_346 = _f2_244;
        output_sq_pair = fma_f32x2_rn_noftz(v_346, v_346, output_sq_pair);
        float2 _f2_245 = make_float2(acc[10], acc[11]);
        float2 v_347 = _f2_245;
        output_sq_pair = fma_f32x2_rn_noftz(v_347, v_347, output_sq_pair);
        float2 _f2_246 = make_float2(acc[12], acc[13]);
        float2 v_348 = _f2_246;
        output_sq_pair = fma_f32x2_rn_noftz(v_348, v_348, output_sq_pair);
        float2 _f2_247 = make_float2(acc[14], acc[15]);
        float2 v_349 = _f2_247;
        output_sq_pair = fma_f32x2_rn_noftz(v_349, v_349, output_sq_pair);
        float2 _f2_248 = make_float2(acc[16], acc[17]);
        float2 v_350 = _f2_248;
        output_sq_pair = fma_f32x2_rn_noftz(v_350, v_350, output_sq_pair);
        float2 _f2_249 = make_float2(acc[18], acc[19]);
        float2 v_351 = _f2_249;
        output_sq_pair = fma_f32x2_rn_noftz(v_351, v_351, output_sq_pair);
        float2 _f2_250 = make_float2(acc[20], acc[21]);
        float2 v_352 = _f2_250;
        output_sq_pair = fma_f32x2_rn_noftz(v_352, v_352, output_sq_pair);
        float2 _f2_251 = make_float2(acc[22], acc[23]);
        float2 v_353 = _f2_251;
        output_sq_pair = fma_f32x2_rn_noftz(v_353, v_353, output_sq_pair);
        float2 _f2_252 = make_float2(acc[24], acc[25]);
        float2 v_354 = _f2_252;
        output_sq_pair = fma_f32x2_rn_noftz(v_354, v_354, output_sq_pair);
        float2 _f2_253 = make_float2(acc[26], acc[27]);
        float2 v_355 = _f2_253;
        output_sq_pair = fma_f32x2_rn_noftz(v_355, v_355, output_sq_pair);
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
            float _rsqrt_1 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_1;
        }
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_4;
        float2 _f2_254 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_254;
        int base_356 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_255 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f2_255), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_256 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f2_256));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_357 = 2;
        const int acc_idx_358 = value_idx_357;
        float2 _f2_257 = make_float2(acc[acc_idx_358], acc[acc_idx_358 + 1]);
        float2 scaled_pair_359;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_359) : "l"(*(const unsigned long long*)&_f2_257), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_258 = make_float2(wout[acc_idx_358], wout[acc_idx_358 + 1]);
        float2 normalized_pair_360;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_360) : "l"(*(const unsigned long long*)&scaled_pair_359), "l"(*(const unsigned long long*)&_f2_258));
        output_values[value_idx_357] = normalized_pair_360.x;
        output_values[value_idx_357 + 1] = normalized_pair_360.y;
        const int value_idx_361 = 4;
        const int acc_idx_362 = value_idx_361;
        float2 _f2_259 = make_float2(acc[acc_idx_362], acc[acc_idx_362 + 1]);
        float2 scaled_pair_363;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_363) : "l"(*(const unsigned long long*)&_f2_259), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_260 = make_float2(wout[acc_idx_362], wout[acc_idx_362 + 1]);
        float2 normalized_pair_364;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_364) : "l"(*(const unsigned long long*)&scaled_pair_363), "l"(*(const unsigned long long*)&_f2_260));
        output_values[value_idx_361] = normalized_pair_364.x;
        output_values[value_idx_361 + 1] = normalized_pair_364.y;
        const int value_idx_365 = 6;
        const int acc_idx_366 = value_idx_365;
        float2 _f2_261 = make_float2(acc[acc_idx_366], acc[acc_idx_366 + 1]);
        float2 scaled_pair_367;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_367) : "l"(*(const unsigned long long*)&_f2_261), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_262 = make_float2(wout[acc_idx_366], wout[acc_idx_366 + 1]);
        float2 normalized_pair_368;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_368) : "l"(*(const unsigned long long*)&scaled_pair_367), "l"(*(const unsigned long long*)&_f2_262));
        output_values[value_idx_365] = normalized_pair_368.x;
        output_values[value_idx_365 + 1] = normalized_pair_368.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_356 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_369 = (2 + group) * 1024 + thread * 8;
        float output_values_370[8];
        const int value_idx_371 = 0;
        const int acc_idx_372 = 8 + value_idx_371;
        float2 _f2_263 = make_float2(acc[acc_idx_372], acc[acc_idx_372 + 1]);
        float2 scaled_pair_373;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_373) : "l"(*(const unsigned long long*)&_f2_263), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_264 = make_float2(wout[acc_idx_372], wout[acc_idx_372 + 1]);
        float2 normalized_pair_374;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_374) : "l"(*(const unsigned long long*)&scaled_pair_373), "l"(*(const unsigned long long*)&_f2_264));
        output_values_370[value_idx_371] = normalized_pair_374.x;
        output_values_370[value_idx_371 + 1] = normalized_pair_374.y;
        const int value_idx_375 = 2;
        const int acc_idx_376 = 8 + value_idx_375;
        float2 _f2_265 = make_float2(acc[acc_idx_376], acc[acc_idx_376 + 1]);
        float2 scaled_pair_377;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_377) : "l"(*(const unsigned long long*)&_f2_265), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_266 = make_float2(wout[acc_idx_376], wout[acc_idx_376 + 1]);
        float2 normalized_pair_378;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_378) : "l"(*(const unsigned long long*)&scaled_pair_377), "l"(*(const unsigned long long*)&_f2_266));
        output_values_370[value_idx_375] = normalized_pair_378.x;
        output_values_370[value_idx_375 + 1] = normalized_pair_378.y;
        const int value_idx_379 = 4;
        const int acc_idx_380 = 8 + value_idx_379;
        float2 _f2_267 = make_float2(acc[acc_idx_380], acc[acc_idx_380 + 1]);
        float2 scaled_pair_381;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_381) : "l"(*(const unsigned long long*)&_f2_267), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_268 = make_float2(wout[acc_idx_380], wout[acc_idx_380 + 1]);
        float2 normalized_pair_382;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_382) : "l"(*(const unsigned long long*)&scaled_pair_381), "l"(*(const unsigned long long*)&_f2_268));
        output_values_370[value_idx_379] = normalized_pair_382.x;
        output_values_370[value_idx_379 + 1] = normalized_pair_382.y;
        const int value_idx_383 = 6;
        const int acc_idx_384 = 8 + value_idx_383;
        float2 _f2_269 = make_float2(acc[acc_idx_384], acc[acc_idx_384 + 1]);
        float2 scaled_pair_385;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_385) : "l"(*(const unsigned long long*)&_f2_269), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_270 = make_float2(wout[acc_idx_384], wout[acc_idx_384 + 1]);
        float2 normalized_pair_386;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_386) : "l"(*(const unsigned long long*)&scaled_pair_385), "l"(*(const unsigned long long*)&_f2_270));
        output_values_370[value_idx_383] = normalized_pair_386.x;
        output_values_370[value_idx_383 + 1] = normalized_pair_386.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_370[0 + 0], output_values_370[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_370[0 + 2], output_values_370[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_370[0 + 4], output_values_370[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_370[0 + 6], output_values_370[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_369 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_387 = (4 + group) * 1024 + thread * 8;
        float output_values_388[8];
        const int value_idx_389 = 0;
        const int acc_idx_390 = 16 + value_idx_389;
        float2 _f2_271 = make_float2(acc[acc_idx_390], acc[acc_idx_390 + 1]);
        float2 scaled_pair_391;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_391) : "l"(*(const unsigned long long*)&_f2_271), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_272 = make_float2(wout[acc_idx_390], wout[acc_idx_390 + 1]);
        float2 normalized_pair_392;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_392) : "l"(*(const unsigned long long*)&scaled_pair_391), "l"(*(const unsigned long long*)&_f2_272));
        output_values_388[value_idx_389] = normalized_pair_392.x;
        output_values_388[value_idx_389 + 1] = normalized_pair_392.y;
        const int value_idx_393 = 2;
        const int acc_idx_394 = 16 + value_idx_393;
        float2 _f2_273 = make_float2(acc[acc_idx_394], acc[acc_idx_394 + 1]);
        float2 scaled_pair_395;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_395) : "l"(*(const unsigned long long*)&_f2_273), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_274 = make_float2(wout[acc_idx_394], wout[acc_idx_394 + 1]);
        float2 normalized_pair_396;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_396) : "l"(*(const unsigned long long*)&scaled_pair_395), "l"(*(const unsigned long long*)&_f2_274));
        output_values_388[value_idx_393] = normalized_pair_396.x;
        output_values_388[value_idx_393 + 1] = normalized_pair_396.y;
        const int value_idx_397 = 4;
        const int acc_idx_398 = 16 + value_idx_397;
        float2 _f2_275 = make_float2(acc[acc_idx_398], acc[acc_idx_398 + 1]);
        float2 scaled_pair_399;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_399) : "l"(*(const unsigned long long*)&_f2_275), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_276 = make_float2(wout[acc_idx_398], wout[acc_idx_398 + 1]);
        float2 normalized_pair_400;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_400) : "l"(*(const unsigned long long*)&scaled_pair_399), "l"(*(const unsigned long long*)&_f2_276));
        output_values_388[value_idx_397] = normalized_pair_400.x;
        output_values_388[value_idx_397 + 1] = normalized_pair_400.y;
        const int value_idx_401 = 6;
        const int acc_idx_402 = 16 + value_idx_401;
        float2 _f2_277 = make_float2(acc[acc_idx_402], acc[acc_idx_402 + 1]);
        float2 scaled_pair_403;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_403) : "l"(*(const unsigned long long*)&_f2_277), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_278 = make_float2(wout[acc_idx_402], wout[acc_idx_402 + 1]);
        float2 normalized_pair_404;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_404) : "l"(*(const unsigned long long*)&scaled_pair_403), "l"(*(const unsigned long long*)&_f2_278));
        output_values_388[value_idx_401] = normalized_pair_404.x;
        output_values_388[value_idx_401 + 1] = normalized_pair_404.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_388[0 + 0], output_values_388[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_388[0 + 2], output_values_388[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_388[0 + 4], output_values_388[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_388[0 + 6], output_values_388[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_387 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_405 = 6144 + group * 512 + thread * 4;
        float output_values_406[8];
        const int value_idx_407 = 0;
        const int acc_idx_408 = 24 + value_idx_407;
        float2 _f2_279 = make_float2(acc[acc_idx_408], acc[acc_idx_408 + 1]);
        float2 scaled_pair_409;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_409) : "l"(*(const unsigned long long*)&_f2_279), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_280 = make_float2(wout[acc_idx_408], wout[acc_idx_408 + 1]);
        float2 normalized_pair_410;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_410) : "l"(*(const unsigned long long*)&scaled_pair_409), "l"(*(const unsigned long long*)&_f2_280));
        output_values_406[value_idx_407] = normalized_pair_410.x;
        output_values_406[value_idx_407 + 1] = normalized_pair_410.y;
        const int value_idx_411 = 2;
        const int acc_idx_412 = 24 + value_idx_411;
        float2 _f2_281 = make_float2(acc[acc_idx_412], acc[acc_idx_412 + 1]);
        float2 scaled_pair_413;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_413) : "l"(*(const unsigned long long*)&_f2_281), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_282 = make_float2(wout[acc_idx_412], wout[acc_idx_412 + 1]);
        float2 normalized_pair_414;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_414) : "l"(*(const unsigned long long*)&scaled_pair_413), "l"(*(const unsigned long long*)&_f2_282));
        output_values_406[value_idx_411] = normalized_pair_414.x;
        output_values_406[value_idx_411 + 1] = normalized_pair_414.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_406[0 + 0], output_values_406[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_406[0 + 2], output_values_406[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_405]) = _pk2;
        }
    }
}

} // extern "C"
