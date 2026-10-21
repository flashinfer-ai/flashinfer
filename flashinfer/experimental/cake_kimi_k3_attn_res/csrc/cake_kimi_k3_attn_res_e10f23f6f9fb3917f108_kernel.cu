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
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_3[0 + 0] = _uv4_3.x;
            _vec_load_3[0 + 1] = _uv4_3.y;
            _vec_load_3[0 + 2] = _uv4_3.z;
            _vec_load_3[0 + 3] = _uv4_3.w;
        }
        dwords[woff] = _vec_load_3[0];
        dwords[woff + 1] = _vec_load_3[1];
        dwords[woff + 2] = _vec_load_3[2];
        dwords[woff + 3] = _vec_load_3[3];
        float _vec_load_4[8];
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
                        : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        float _vec_load_5[8];
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
                        : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        q[0] = _vec_load_4[0] * _vec_load_5[0];
        q[1] = _vec_load_4[1] * _vec_load_5[1];
        q[2] = _vec_load_4[2] * _vec_load_5[2];
        q[3] = _vec_load_4[3] * _vec_load_5[3];
        q[4] = _vec_load_4[4] * _vec_load_5[4];
        q[5] = _vec_load_4[5] * _vec_load_5[5];
        q[6] = _vec_load_4[6] * _vec_load_5[6];
        q[7] = _vec_load_4[7] * _vec_load_5[7];
        float _vec_load_6[8];
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
                        : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        wout[0] = _vec_load_6[0];
        wout[1] = _vec_load_6[1];
        wout[2] = _vec_load_6[2];
        wout[3] = _vec_load_6[3];
        wout[4] = _vec_load_6[4];
        wout[5] = _vec_load_6[5];
        wout[6] = _vec_load_6[6];
        wout[7] = _vec_load_6[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_7[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_7[0 + 0] = _uv4_7.x;
            _vec_load_7[0 + 1] = _uv4_7.y;
            _vec_load_7[0 + 2] = _uv4_7.z;
            _vec_load_7[0 + 3] = _uv4_7.w;
        }
        words[woff_1] = _vec_load_7[0];
        words[woff_1 + 1] = _vec_load_7[1];
        words[woff_1 + 2] = _vec_load_7[2];
        words[woff_1 + 3] = _vec_load_7[3];
        unsigned int _vec_load_8[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_8[0 + 0] = _uv4_8.x;
            _vec_load_8[0 + 1] = _uv4_8.y;
            _vec_load_8[0 + 2] = _uv4_8.z;
            _vec_load_8[0 + 3] = _uv4_8.w;
        }
        words[14 + woff_1] = _vec_load_8[0];
        words[14 + woff_1 + 1] = _vec_load_8[1];
        words[14 + woff_1 + 2] = _vec_load_8[2];
        words[14 + woff_1 + 3] = _vec_load_8[3];
        unsigned int _vec_load_9[4];
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_9[0 + 0] = _uv4_9.x;
            _vec_load_9[0 + 1] = _uv4_9.y;
            _vec_load_9[0 + 2] = _uv4_9.z;
            _vec_load_9[0 + 3] = _uv4_9.w;
        }
        words[28 + woff_1] = _vec_load_9[0];
        words[28 + woff_1 + 1] = _vec_load_9[1];
        words[28 + woff_1 + 2] = _vec_load_9[2];
        words[28 + woff_1 + 3] = _vec_load_9[3];
        unsigned int _vec_load_10[4];
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_10[0 + 0] = _uv4_10.x;
            _vec_load_10[0 + 1] = _uv4_10.y;
            _vec_load_10[0 + 2] = _uv4_10.z;
            _vec_load_10[0 + 3] = _uv4_10.w;
        }
        dwords[woff_1] = _vec_load_10[0];
        dwords[woff_1 + 1] = _vec_load_10[1];
        dwords[woff_1 + 2] = _vec_load_10[2];
        dwords[woff_1 + 3] = _vec_load_10[3];
        float _vec_load_11[8];
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
                        : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        float _vec_load_12[8];
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
                        : "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_12[_pair]));
                }
            }
        }
        q[8] = _vec_load_11[0] * _vec_load_12[0];
        q[9] = _vec_load_11[1] * _vec_load_12[1];
        q[10] = _vec_load_11[2] * _vec_load_12[2];
        q[11] = _vec_load_11[3] * _vec_load_12[3];
        q[12] = _vec_load_11[4] * _vec_load_12[4];
        q[13] = _vec_load_11[5] * _vec_load_12[5];
        q[14] = _vec_load_11[6] * _vec_load_12[6];
        q[15] = _vec_load_11[7] * _vec_load_12[7];
        float _vec_load_13[8];
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
                        : "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_13[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_13[_pair]));
                }
            }
        }
        wout[8] = _vec_load_13[0];
        wout[9] = _vec_load_13[1];
        wout[10] = _vec_load_13[2];
        wout[11] = _vec_load_13[3];
        wout[12] = _vec_load_13[4];
        wout[13] = _vec_load_13[5];
        wout[14] = _vec_load_13[6];
        wout[15] = _vec_load_13[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        words[woff_3] = _vec_load_14[0];
        words[woff_3 + 1] = _vec_load_14[1];
        words[woff_3 + 2] = _vec_load_14[2];
        words[woff_3 + 3] = _vec_load_14[3];
        unsigned int _vec_load_15[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_15[0 + 0] = _uv4_15.x;
            _vec_load_15[0 + 1] = _uv4_15.y;
            _vec_load_15[0 + 2] = _uv4_15.z;
            _vec_load_15[0 + 3] = _uv4_15.w;
        }
        words[14 + woff_3] = _vec_load_15[0];
        words[14 + woff_3 + 1] = _vec_load_15[1];
        words[14 + woff_3 + 2] = _vec_load_15[2];
        words[14 + woff_3 + 3] = _vec_load_15[3];
        unsigned int _vec_load_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_16[0 + 0] = _uv4_16.x;
            _vec_load_16[0 + 1] = _uv4_16.y;
            _vec_load_16[0 + 2] = _uv4_16.z;
            _vec_load_16[0 + 3] = _uv4_16.w;
        }
        words[28 + woff_3] = _vec_load_16[0];
        words[28 + woff_3 + 1] = _vec_load_16[1];
        words[28 + woff_3 + 2] = _vec_load_16[2];
        words[28 + woff_3 + 3] = _vec_load_16[3];
        unsigned int _vec_load_17[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_17[0 + 0] = _uv4_17.x;
            _vec_load_17[0 + 1] = _uv4_17.y;
            _vec_load_17[0 + 2] = _uv4_17.z;
            _vec_load_17[0 + 3] = _uv4_17.w;
        }
        dwords[woff_3] = _vec_load_17[0];
        dwords[woff_3 + 1] = _vec_load_17[1];
        dwords[woff_3 + 2] = _vec_load_17[2];
        dwords[woff_3 + 3] = _vec_load_17[3];
        float _vec_load_18[8];
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
                        : "=f"((&_vec_load_18[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_18[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_18[_pair]));
                }
            }
        }
        float _vec_load_19[8];
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
                        : "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        q[16] = _vec_load_18[0] * _vec_load_19[0];
        q[17] = _vec_load_18[1] * _vec_load_19[1];
        q[18] = _vec_load_18[2] * _vec_load_19[2];
        q[19] = _vec_load_18[3] * _vec_load_19[3];
        q[20] = _vec_load_18[4] * _vec_load_19[4];
        q[21] = _vec_load_18[5] * _vec_load_19[5];
        q[22] = _vec_load_18[6] * _vec_load_19[6];
        q[23] = _vec_load_18[7] * _vec_load_19[7];
        float _vec_load_20[8];
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
                        : "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_20[_pair]));
                }
            }
        }
        wout[16] = _vec_load_20[0];
        wout[17] = _vec_load_20[1];
        wout[18] = _vec_load_20[2];
        wout[19] = _vec_load_20[3];
        wout[20] = _vec_load_20[4];
        wout[21] = _vec_load_20[5];
        wout[22] = _vec_load_20[6];
        wout[23] = _vec_load_20[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_21[1];
        {
            _vec_load_21[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_21[0];
        unsigned int _vec_load_22[1];
        {
            _vec_load_22[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_22[0];
        unsigned int _vec_load_23[1];
        {
            _vec_load_23[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_23[0];
        unsigned int _vec_load_24[1];
        {
            _vec_load_24[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_24[0];
        unsigned int _vec_load_25[1];
        {
            _vec_load_25[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_25[0];
        unsigned int _vec_load_26[1];
        {
            _vec_load_26[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_26[0];
        unsigned int _vec_load_27[1];
        {
            _vec_load_27[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_27[0];
        unsigned int _vec_load_28[1];
        {
            _vec_load_28[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_28[0];
        float _vec_load_29[4];
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
                    : "=f"((&_vec_load_29[0 + _pair * 2])[0]), "=f"((&_vec_load_29[0 + _pair * 2])[1])
                    : "r"(_vpairs_21[_pair]));
            }
        }
        float _vec_load_30[4];
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
                    : "=f"((&_vec_load_30[0 + _pair * 2])[0]), "=f"((&_vec_load_30[0 + _pair * 2])[1])
                    : "r"(_vpairs_22[_pair]));
            }
        }
        q[24] = _vec_load_29[0] * _vec_load_30[0];
        q[25] = _vec_load_29[1] * _vec_load_30[1];
        q[26] = _vec_load_29[2] * _vec_load_30[2];
        q[27] = _vec_load_29[3] * _vec_load_30[3];
        float _vec_load_31[4];
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
                    : "=f"((&_vec_load_31[0 + _pair * 2])[0]), "=f"((&_vec_load_31[0 + _pair * 2])[1])
                    : "r"(_vpairs_23[_pair]));
            }
        }
        wout[24] = _vec_load_31[0];
        wout[25] = _vec_load_31[1];
        wout[26] = _vec_load_31[2];
        wout[27] = _vec_load_31[3];
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
        float2 _f2_6 = make_float2(_unpack_bf16_0[0], _unpack_bf16_0[1]);
        float2 v = _f2_6;
        float2 _f2_7 = make_float2(q[0], q[1]);
        float2 qp = _f2_7;
        sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
        float2 _f2_8 = make_float2(_unpack_bf16_0[2], _unpack_bf16_0[3]);
        float2 v_8 = _f2_8;
        float2 _f2_9 = make_float2(q[2], q[3]);
        float2 qp_9 = _f2_9;
        sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_8, qp_9, dot[0]);
        float2 _f2_10 = make_float2(_unpack_bf16_0[4], _unpack_bf16_0[5]);
        float2 v_10 = _f2_10;
        float2 _f2_11 = make_float2(q[4], q[5]);
        float2 qp_11 = _f2_11;
        sq[0] = fma_f32x2_rn_noftz(v_10, v_10, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_10, qp_11, dot[0]);
        float2 _f2_12 = make_float2(_unpack_bf16_0[6], _unpack_bf16_0[7]);
        float2 v_12 = _f2_12;
        float2 _f2_13 = make_float2(q[6], q[7]);
        float2 qp_13 = _f2_13;
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
        float2 _f2_14 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
        float2 v_15 = _f2_14;
        float2 _f2_15 = make_float2(q[0], q[1]);
        float2 qp_16 = _f2_15;
        sq[1] = fma_f32x2_rn_noftz(v_15, v_15, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_15, qp_16, dot[1]);
        float2 _f2_16 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
        float2 v_17 = _f2_16;
        float2 _f2_17 = make_float2(q[2], q[3]);
        float2 qp_18 = _f2_17;
        sq[1] = fma_f32x2_rn_noftz(v_17, v_17, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_17, qp_18, dot[1]);
        float2 _f2_18 = make_float2(_unpack_bf16_1[4], _unpack_bf16_1[5]);
        float2 v_19 = _f2_18;
        float2 _f2_19 = make_float2(q[4], q[5]);
        float2 qp_20 = _f2_19;
        sq[1] = fma_f32x2_rn_noftz(v_19, v_19, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_19, qp_20, dot[1]);
        float2 _f2_20 = make_float2(_unpack_bf16_1[6], _unpack_bf16_1[7]);
        float2 v_21 = _f2_20;
        float2 _f2_21 = make_float2(q[6], q[7]);
        float2 qp_22 = _f2_21;
        sq[1] = fma_f32x2_rn_noftz(v_21, v_21, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_21, qp_22, dot[1]);
        unsigned int sw_23[4];
        sw_23[0] = words[28 + woff_7];
        sw_23[1] = words[28 + woff_7 + 1];
        sw_23[2] = words[28 + woff_7 + 2];
        sw_23[3] = words[28 + woff_7 + 3];
        __nv_bfloat162 a = __as_bf16x2(sw_23[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_23[0] = __as_u32(mixed);
        words[28 + woff_7] = sw_23[0];
        __nv_bfloat162 a_24 = __as_bf16x2(sw_23[1]);
        __nv_bfloat162 d_25 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_26 = a_24 + d_25;
        sw_23[1] = __as_u32(mixed_26);
        words[28 + woff_7 + 1] = sw_23[1];
        __nv_bfloat162 a_27 = __as_bf16x2(sw_23[2]);
        __nv_bfloat162 d_28 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_29 = a_27 + d_28;
        sw_23[2] = __as_u32(mixed_29);
        words[28 + woff_7 + 2] = sw_23[2];
        __nv_bfloat162 a_30 = __as_bf16x2(sw_23[3]);
        __nv_bfloat162 d_31 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_32 = a_30 + d_31;
        sw_23[3] = __as_u32(mixed_32);
        words[28 + woff_7 + 3] = sw_23[3];
        {
            int4 _iv4 = make_int4(sw_23[0 + 0], sw_23[0 + 1], sw_23[0 + 2], sw_23[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
        }
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
        float2 _f2_22 = make_float2(_unpack_bf16_2[0], _unpack_bf16_2[1]);
        float2 v_33 = _f2_22;
        float2 _f2_23 = make_float2(q[0], q[1]);
        float2 qp_34 = _f2_23;
        sq[2] = fma_f32x2_rn_noftz(v_33, v_33, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_33, qp_34, dot[2]);
        float2 _f2_24 = make_float2(_unpack_bf16_2[2], _unpack_bf16_2[3]);
        float2 v_35 = _f2_24;
        float2 _f2_25 = make_float2(q[2], q[3]);
        float2 qp_36 = _f2_25;
        sq[2] = fma_f32x2_rn_noftz(v_35, v_35, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_35, qp_36, dot[2]);
        float2 _f2_26 = make_float2(_unpack_bf16_2[4], _unpack_bf16_2[5]);
        float2 v_37 = _f2_26;
        float2 _f2_27 = make_float2(q[4], q[5]);
        float2 qp_38 = _f2_27;
        sq[2] = fma_f32x2_rn_noftz(v_37, v_37, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_37, qp_38, dot[2]);
        float2 _f2_28 = make_float2(_unpack_bf16_2[6], _unpack_bf16_2[7]);
        float2 v_39 = _f2_28;
        float2 _f2_29 = make_float2(q[6], q[7]);
        float2 qp_40 = _f2_29;
        sq[2] = fma_f32x2_rn_noftz(v_39, v_39, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_39, qp_40, dot[2]);
        int base_41 = (2 + group) * 1024 + thread * 8;
        const int woff_42 = 4;
        unsigned int sw_43[4];
        sw_43[0] = words[woff_42];
        sw_43[1] = words[woff_42 + 1];
        sw_43[2] = words[woff_42 + 2];
        sw_43[3] = words[woff_42 + 3];
        float _unpack_bf16_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                : "r"(sw_43[_pair]));
        }
        fsrc[8] = _unpack_bf16_3[0];
        fsrc[9] = _unpack_bf16_3[1];
        fsrc[10] = _unpack_bf16_3[2];
        fsrc[11] = _unpack_bf16_3[3];
        fsrc[12] = _unpack_bf16_3[4];
        fsrc[13] = _unpack_bf16_3[5];
        fsrc[14] = _unpack_bf16_3[6];
        fsrc[15] = _unpack_bf16_3[7];
        float2 _f2_30 = make_float2(_unpack_bf16_3[0], _unpack_bf16_3[1]);
        float2 v_44 = _f2_30;
        float2 _f2_31 = make_float2(q[8], q[9]);
        float2 qp_45 = _f2_31;
        sq[0] = fma_f32x2_rn_noftz(v_44, v_44, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_44, qp_45, dot[0]);
        float2 _f2_32 = make_float2(_unpack_bf16_3[2], _unpack_bf16_3[3]);
        float2 v_46 = _f2_32;
        float2 _f2_33 = make_float2(q[10], q[11]);
        float2 qp_47 = _f2_33;
        sq[0] = fma_f32x2_rn_noftz(v_46, v_46, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_46, qp_47, dot[0]);
        float2 _f2_34 = make_float2(_unpack_bf16_3[4], _unpack_bf16_3[5]);
        float2 v_48 = _f2_34;
        float2 _f2_35 = make_float2(q[12], q[13]);
        float2 qp_49 = _f2_35;
        sq[0] = fma_f32x2_rn_noftz(v_48, v_48, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_48, qp_49, dot[0]);
        float2 _f2_36 = make_float2(_unpack_bf16_3[6], _unpack_bf16_3[7]);
        float2 v_50 = _f2_36;
        float2 _f2_37 = make_float2(q[14], q[15]);
        float2 qp_51 = _f2_37;
        sq[0] = fma_f32x2_rn_noftz(v_50, v_50, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_50, qp_51, dot[0]);
        unsigned int sw_52[4];
        sw_52[0] = words[14 + woff_42];
        sw_52[1] = words[14 + woff_42 + 1];
        sw_52[2] = words[14 + woff_42 + 2];
        sw_52[3] = words[14 + woff_42 + 3];
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
        fsrc[36] = _unpack_bf16_4[0];
        fsrc[37] = _unpack_bf16_4[1];
        fsrc[38] = _unpack_bf16_4[2];
        fsrc[39] = _unpack_bf16_4[3];
        fsrc[40] = _unpack_bf16_4[4];
        fsrc[41] = _unpack_bf16_4[5];
        fsrc[42] = _unpack_bf16_4[6];
        fsrc[43] = _unpack_bf16_4[7];
        float2 _f2_38 = make_float2(_unpack_bf16_4[0], _unpack_bf16_4[1]);
        float2 v_53 = _f2_38;
        float2 _f2_39 = make_float2(q[8], q[9]);
        float2 qp_54 = _f2_39;
        sq[1] = fma_f32x2_rn_noftz(v_53, v_53, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_53, qp_54, dot[1]);
        float2 _f2_40 = make_float2(_unpack_bf16_4[2], _unpack_bf16_4[3]);
        float2 v_55 = _f2_40;
        float2 _f2_41 = make_float2(q[10], q[11]);
        float2 qp_56 = _f2_41;
        sq[1] = fma_f32x2_rn_noftz(v_55, v_55, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_55, qp_56, dot[1]);
        float2 _f2_42 = make_float2(_unpack_bf16_4[4], _unpack_bf16_4[5]);
        float2 v_57 = _f2_42;
        float2 _f2_43 = make_float2(q[12], q[13]);
        float2 qp_58 = _f2_43;
        sq[1] = fma_f32x2_rn_noftz(v_57, v_57, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_57, qp_58, dot[1]);
        float2 _f2_44 = make_float2(_unpack_bf16_4[6], _unpack_bf16_4[7]);
        float2 v_59 = _f2_44;
        float2 _f2_45 = make_float2(q[14], q[15]);
        float2 qp_60 = _f2_45;
        sq[1] = fma_f32x2_rn_noftz(v_59, v_59, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_59, qp_60, dot[1]);
        unsigned int sw_61[4];
        sw_61[0] = words[28 + woff_42];
        sw_61[1] = words[28 + woff_42 + 1];
        sw_61[2] = words[28 + woff_42 + 2];
        sw_61[3] = words[28 + woff_42 + 3];
        __nv_bfloat162 a_62 = __as_bf16x2(sw_61[0]);
        __nv_bfloat162 d_63 = __as_bf16x2(dwords[woff_42]);
        __nv_bfloat162 mixed_64 = a_62 + d_63;
        sw_61[0] = __as_u32(mixed_64);
        words[28 + woff_42] = sw_61[0];
        __nv_bfloat162 a_65 = __as_bf16x2(sw_61[1]);
        __nv_bfloat162 d_66 = __as_bf16x2(dwords[woff_42 + 1]);
        __nv_bfloat162 mixed_67 = a_65 + d_66;
        sw_61[1] = __as_u32(mixed_67);
        words[28 + woff_42 + 1] = sw_61[1];
        __nv_bfloat162 a_68 = __as_bf16x2(sw_61[2]);
        __nv_bfloat162 d_69 = __as_bf16x2(dwords[woff_42 + 2]);
        __nv_bfloat162 mixed_70 = a_68 + d_69;
        sw_61[2] = __as_u32(mixed_70);
        words[28 + woff_42 + 2] = sw_61[2];
        __nv_bfloat162 a_71 = __as_bf16x2(sw_61[3]);
        __nv_bfloat162 d_72 = __as_bf16x2(dwords[woff_42 + 3]);
        __nv_bfloat162 mixed_73 = a_71 + d_72;
        sw_61[3] = __as_u32(mixed_73);
        words[28 + woff_42 + 3] = sw_61[3];
        {
            int4 _iv4 = make_int4(sw_61[0 + 0], sw_61[0 + 1], sw_61[0 + 2], sw_61[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_41)) + 0) = _iv4;
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
                : "r"(sw_61[_pair]));
        }
        fsrc[64] = _unpack_bf16_5[0];
        fsrc[65] = _unpack_bf16_5[1];
        fsrc[66] = _unpack_bf16_5[2];
        fsrc[67] = _unpack_bf16_5[3];
        fsrc[68] = _unpack_bf16_5[4];
        fsrc[69] = _unpack_bf16_5[5];
        fsrc[70] = _unpack_bf16_5[6];
        fsrc[71] = _unpack_bf16_5[7];
        float2 _f2_46 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 v_74 = _f2_46;
        float2 _f2_47 = make_float2(q[8], q[9]);
        float2 qp_75 = _f2_47;
        sq[2] = fma_f32x2_rn_noftz(v_74, v_74, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_74, qp_75, dot[2]);
        float2 _f2_48 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 v_76 = _f2_48;
        float2 _f2_49 = make_float2(q[10], q[11]);
        float2 qp_77 = _f2_49;
        sq[2] = fma_f32x2_rn_noftz(v_76, v_76, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_76, qp_77, dot[2]);
        float2 _f2_50 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 v_78 = _f2_50;
        float2 _f2_51 = make_float2(q[12], q[13]);
        float2 qp_79 = _f2_51;
        sq[2] = fma_f32x2_rn_noftz(v_78, v_78, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_78, qp_79, dot[2]);
        float2 _f2_52 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 v_80 = _f2_52;
        float2 _f2_53 = make_float2(q[14], q[15]);
        float2 qp_81 = _f2_53;
        sq[2] = fma_f32x2_rn_noftz(v_80, v_80, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_80, qp_81, dot[2]);
        int base_82 = (4 + group) * 1024 + thread * 8;
        const int woff_83 = 8;
        unsigned int sw_84[4];
        sw_84[0] = words[woff_83];
        sw_84[1] = words[woff_83 + 1];
        sw_84[2] = words[woff_83 + 2];
        sw_84[3] = words[woff_83 + 3];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(sw_84[_pair]));
        }
        fsrc[16] = _unpack_bf16_6[0];
        fsrc[17] = _unpack_bf16_6[1];
        fsrc[18] = _unpack_bf16_6[2];
        fsrc[19] = _unpack_bf16_6[3];
        fsrc[20] = _unpack_bf16_6[4];
        fsrc[21] = _unpack_bf16_6[5];
        fsrc[22] = _unpack_bf16_6[6];
        fsrc[23] = _unpack_bf16_6[7];
        float2 _f2_54 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 v_85 = _f2_54;
        float2 _f2_55 = make_float2(q[16], q[17]);
        float2 qp_86 = _f2_55;
        sq[0] = fma_f32x2_rn_noftz(v_85, v_85, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_85, qp_86, dot[0]);
        float2 _f2_56 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 v_87 = _f2_56;
        float2 _f2_57 = make_float2(q[18], q[19]);
        float2 qp_88 = _f2_57;
        sq[0] = fma_f32x2_rn_noftz(v_87, v_87, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_87, qp_88, dot[0]);
        float2 _f2_58 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 v_89 = _f2_58;
        float2 _f2_59 = make_float2(q[20], q[21]);
        float2 qp_90 = _f2_59;
        sq[0] = fma_f32x2_rn_noftz(v_89, v_89, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_89, qp_90, dot[0]);
        float2 _f2_60 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 v_91 = _f2_60;
        float2 _f2_61 = make_float2(q[22], q[23]);
        float2 qp_92 = _f2_61;
        sq[0] = fma_f32x2_rn_noftz(v_91, v_91, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_91, qp_92, dot[0]);
        unsigned int sw_93[4];
        sw_93[0] = words[14 + woff_83];
        sw_93[1] = words[14 + woff_83 + 1];
        sw_93[2] = words[14 + woff_83 + 2];
        sw_93[3] = words[14 + woff_83 + 3];
        float _unpack_bf16_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                : "r"(sw_93[_pair]));
        }
        fsrc[44] = _unpack_bf16_7[0];
        fsrc[45] = _unpack_bf16_7[1];
        fsrc[46] = _unpack_bf16_7[2];
        fsrc[47] = _unpack_bf16_7[3];
        fsrc[48] = _unpack_bf16_7[4];
        fsrc[49] = _unpack_bf16_7[5];
        fsrc[50] = _unpack_bf16_7[6];
        fsrc[51] = _unpack_bf16_7[7];
        float2 _f2_62 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 v_94 = _f2_62;
        float2 _f2_63 = make_float2(q[16], q[17]);
        float2 qp_95 = _f2_63;
        sq[1] = fma_f32x2_rn_noftz(v_94, v_94, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_94, qp_95, dot[1]);
        float2 _f2_64 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 v_96 = _f2_64;
        float2 _f2_65 = make_float2(q[18], q[19]);
        float2 qp_97 = _f2_65;
        sq[1] = fma_f32x2_rn_noftz(v_96, v_96, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_96, qp_97, dot[1]);
        float2 _f2_66 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 v_98 = _f2_66;
        float2 _f2_67 = make_float2(q[20], q[21]);
        float2 qp_99 = _f2_67;
        sq[1] = fma_f32x2_rn_noftz(v_98, v_98, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_98, qp_99, dot[1]);
        float2 _f2_68 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 v_100 = _f2_68;
        float2 _f2_69 = make_float2(q[22], q[23]);
        float2 qp_101 = _f2_69;
        sq[1] = fma_f32x2_rn_noftz(v_100, v_100, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_100, qp_101, dot[1]);
        unsigned int sw_102[4];
        sw_102[0] = words[28 + woff_83];
        sw_102[1] = words[28 + woff_83 + 1];
        sw_102[2] = words[28 + woff_83 + 2];
        sw_102[3] = words[28 + woff_83 + 3];
        __nv_bfloat162 a_103 = __as_bf16x2(sw_102[0]);
        __nv_bfloat162 d_104 = __as_bf16x2(dwords[woff_83]);
        __nv_bfloat162 mixed_105 = a_103 + d_104;
        sw_102[0] = __as_u32(mixed_105);
        words[28 + woff_83] = sw_102[0];
        __nv_bfloat162 a_106 = __as_bf16x2(sw_102[1]);
        __nv_bfloat162 d_107 = __as_bf16x2(dwords[woff_83 + 1]);
        __nv_bfloat162 mixed_108 = a_106 + d_107;
        sw_102[1] = __as_u32(mixed_108);
        words[28 + woff_83 + 1] = sw_102[1];
        __nv_bfloat162 a_109 = __as_bf16x2(sw_102[2]);
        __nv_bfloat162 d_110 = __as_bf16x2(dwords[woff_83 + 2]);
        __nv_bfloat162 mixed_111 = a_109 + d_110;
        sw_102[2] = __as_u32(mixed_111);
        words[28 + woff_83 + 2] = sw_102[2];
        __nv_bfloat162 a_112 = __as_bf16x2(sw_102[3]);
        __nv_bfloat162 d_113 = __as_bf16x2(dwords[woff_83 + 3]);
        __nv_bfloat162 mixed_114 = a_112 + d_113;
        sw_102[3] = __as_u32(mixed_114);
        words[28 + woff_83 + 3] = sw_102[3];
        {
            int4 _iv4 = make_int4(sw_102[0 + 0], sw_102[0 + 1], sw_102[0 + 2], sw_102[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_82)) + 0) = _iv4;
        }
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
        fsrc[72] = _unpack_bf16_8[0];
        fsrc[73] = _unpack_bf16_8[1];
        fsrc[74] = _unpack_bf16_8[2];
        fsrc[75] = _unpack_bf16_8[3];
        fsrc[76] = _unpack_bf16_8[4];
        fsrc[77] = _unpack_bf16_8[5];
        fsrc[78] = _unpack_bf16_8[6];
        fsrc[79] = _unpack_bf16_8[7];
        float2 _f2_70 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 v_115 = _f2_70;
        float2 _f2_71 = make_float2(q[16], q[17]);
        float2 qp_116 = _f2_71;
        sq[2] = fma_f32x2_rn_noftz(v_115, v_115, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_115, qp_116, dot[2]);
        float2 _f2_72 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 v_117 = _f2_72;
        float2 _f2_73 = make_float2(q[18], q[19]);
        float2 qp_118 = _f2_73;
        sq[2] = fma_f32x2_rn_noftz(v_117, v_117, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_117, qp_118, dot[2]);
        float2 _f2_74 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 v_119 = _f2_74;
        float2 _f2_75 = make_float2(q[20], q[21]);
        float2 qp_120 = _f2_75;
        sq[2] = fma_f32x2_rn_noftz(v_119, v_119, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_119, qp_120, dot[2]);
        float2 _f2_76 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 v_121 = _f2_76;
        float2 _f2_77 = make_float2(q[22], q[23]);
        float2 qp_122 = _f2_77;
        sq[2] = fma_f32x2_rn_noftz(v_121, v_121, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_121, qp_122, dot[2]);
        int base_123 = 6144 + group * 512 + thread * 4;
        const int woff_124 = 12;
        unsigned int sw_125[4];
        sw_125[0] = words[woff_124];
        sw_125[1] = words[woff_124 + 1];
        float _unpack_bf16_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_9[_pair * 2])[0]), "=f"((&_unpack_bf16_9[_pair * 2])[1])
                : "r"(sw_125[_pair]));
        }
        fsrc[24] = _unpack_bf16_9[0];
        fsrc[25] = _unpack_bf16_9[1];
        fsrc[26] = _unpack_bf16_9[2];
        fsrc[27] = _unpack_bf16_9[3];
        float2 _f2_78 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 v_126 = _f2_78;
        sq[0] = fma_f32x2_rn_noftz(v_126, v_126, sq[0]);
        float2 _f2_79 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 v_127 = _f2_79;
        sq[0] = fma_f32x2_rn_noftz(v_127, v_127, sq[0]);
        float2 _f2_80 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 v_128 = _f2_80;
        float2 _f2_81 = make_float2(q[24], q[25]);
        float2 qp_129 = _f2_81;
        dot[0] = fma_f32x2_rn_noftz(v_128, qp_129, dot[0]);
        float2 _f2_82 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 v_130 = _f2_82;
        float2 _f2_83 = make_float2(q[26], q[27]);
        float2 qp_131 = _f2_83;
        dot[0] = fma_f32x2_rn_noftz(v_130, qp_131, dot[0]);
        unsigned int sw_132[4];
        sw_132[0] = words[14 + woff_124];
        sw_132[1] = words[14 + woff_124 + 1];
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                : "r"(sw_132[_pair]));
        }
        fsrc[52] = _unpack_bf16_10[0];
        fsrc[53] = _unpack_bf16_10[1];
        fsrc[54] = _unpack_bf16_10[2];
        fsrc[55] = _unpack_bf16_10[3];
        float2 _f2_84 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 v_133 = _f2_84;
        sq[1] = fma_f32x2_rn_noftz(v_133, v_133, sq[1]);
        float2 _f2_85 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 v_134 = _f2_85;
        sq[1] = fma_f32x2_rn_noftz(v_134, v_134, sq[1]);
        float2 _f2_86 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 v_135 = _f2_86;
        float2 _f2_87 = make_float2(q[24], q[25]);
        float2 qp_136 = _f2_87;
        dot[1] = fma_f32x2_rn_noftz(v_135, qp_136, dot[1]);
        float2 _f2_88 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 v_137 = _f2_88;
        float2 _f2_89 = make_float2(q[26], q[27]);
        float2 qp_138 = _f2_89;
        dot[1] = fma_f32x2_rn_noftz(v_137, qp_138, dot[1]);
        unsigned int sw_139[4];
        sw_139[0] = words[28 + woff_124];
        sw_139[1] = words[28 + woff_124 + 1];
        __nv_bfloat162 a_140 = __as_bf16x2(sw_139[0]);
        __nv_bfloat162 d_141 = __as_bf16x2(dwords[woff_124]);
        __nv_bfloat162 mixed_142 = a_140 + d_141;
        sw_139[0] = __as_u32(mixed_142);
        words[28 + woff_124] = sw_139[0];
        __nv_bfloat162 a_143 = __as_bf16x2(sw_139[1]);
        __nv_bfloat162 d_144 = __as_bf16x2(dwords[woff_124 + 1]);
        __nv_bfloat162 mixed_145 = a_143 + d_144;
        sw_139[1] = __as_u32(mixed_145);
        words[28 + woff_124 + 1] = sw_139[1];
        {
            int2 _iv2 = make_int2(sw_139[0 + 0], sw_139[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_123)) + 0) = _iv2;
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
                : "r"(sw_139[_pair]));
        }
        fsrc[80] = _unpack_bf16_11[0];
        fsrc[81] = _unpack_bf16_11[1];
        fsrc[82] = _unpack_bf16_11[2];
        fsrc[83] = _unpack_bf16_11[3];
        float2 _f2_90 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 v_146 = _f2_90;
        sq[2] = fma_f32x2_rn_noftz(v_146, v_146, sq[2]);
        float2 _f2_91 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 v_147 = _f2_91;
        sq[2] = fma_f32x2_rn_noftz(v_147, v_147, sq[2]);
        float2 _f2_92 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 v_148 = _f2_92;
        float2 _f2_93 = make_float2(q[24], q[25]);
        float2 qp_149 = _f2_93;
        dot[2] = fma_f32x2_rn_noftz(v_148, qp_149, dot[2]);
        float2 _f2_94 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 v_150 = _f2_94;
        float2 _f2_95 = make_float2(q[26], q[27]);
        float2 qp_151 = _f2_95;
        dot[2] = fma_f32x2_rn_noftz(v_150, qp_151, dot[2]);
        float2 pairs[3];
        float2 _f2_96 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_96;
        float2 _f2_97 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_97;
        float2 _f2_98 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_98;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_99 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_99;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_152 = 0;
        bits_152 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_152, 16);
        unsigned long long peerbits_153 = _shfl_xor_1;
        float2 _f2_100 = make_float2(0.0f, 0.0f);
        float2 peer_154 = _f2_100;
        peer_154 = reinterpret_cast<float2*>(&peerbits_153)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_154);
        unsigned long long bits_155 = 0;
        bits_155 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_155, 16);
        unsigned long long peerbits_156 = _shfl_xor_2;
        float2 _f2_101 = make_float2(0.0f, 0.0f);
        float2 peer_157 = _f2_101;
        peer_157 = reinterpret_cast<float2*>(&peerbits_156)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_157);
        unsigned long long bits_158 = 0;
        bits_158 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_158, 8);
        unsigned long long peerbits_159 = _shfl_xor_3;
        float2 _f2_102 = make_float2(0.0f, 0.0f);
        float2 peer_160 = _f2_102;
        peer_160 = reinterpret_cast<float2*>(&peerbits_159)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_160);
        unsigned long long bits_161 = 0;
        bits_161 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_161, 8);
        unsigned long long peerbits_162 = _shfl_xor_4;
        float2 _f2_103 = make_float2(0.0f, 0.0f);
        float2 peer_163 = _f2_103;
        peer_163 = reinterpret_cast<float2*>(&peerbits_162)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_163);
        unsigned long long bits_164 = 0;
        bits_164 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_164, 8);
        unsigned long long peerbits_165 = _shfl_xor_5;
        float2 _f2_104 = make_float2(0.0f, 0.0f);
        float2 peer_166 = _f2_104;
        peer_166 = reinterpret_cast<float2*>(&peerbits_165)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_166);
        unsigned long long bits_167 = 0;
        bits_167 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_167, 4);
        unsigned long long peerbits_168 = _shfl_xor_6;
        float2 _f2_105 = make_float2(0.0f, 0.0f);
        float2 peer_169 = _f2_105;
        peer_169 = reinterpret_cast<float2*>(&peerbits_168)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_169);
        unsigned long long bits_170 = 0;
        bits_170 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_170, 4);
        unsigned long long peerbits_171 = _shfl_xor_7;
        float2 _f2_106 = make_float2(0.0f, 0.0f);
        float2 peer_172 = _f2_106;
        peer_172 = reinterpret_cast<float2*>(&peerbits_171)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_172);
        unsigned long long bits_173 = 0;
        bits_173 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_173, 4);
        unsigned long long peerbits_174 = _shfl_xor_8;
        float2 _f2_107 = make_float2(0.0f, 0.0f);
        float2 peer_175 = _f2_107;
        peer_175 = reinterpret_cast<float2*>(&peerbits_174)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_175);
        unsigned long long bits_176 = 0;
        bits_176 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_176, 2);
        unsigned long long peerbits_177 = _shfl_xor_9;
        float2 _f2_108 = make_float2(0.0f, 0.0f);
        float2 peer_178 = _f2_108;
        peer_178 = reinterpret_cast<float2*>(&peerbits_177)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_178);
        unsigned long long bits_179 = 0;
        bits_179 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_179, 2);
        unsigned long long peerbits_180 = _shfl_xor_10;
        float2 _f2_109 = make_float2(0.0f, 0.0f);
        float2 peer_181 = _f2_109;
        peer_181 = reinterpret_cast<float2*>(&peerbits_180)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_181);
        unsigned long long bits_182 = 0;
        bits_182 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_182, 2);
        unsigned long long peerbits_183 = _shfl_xor_11;
        float2 _f2_110 = make_float2(0.0f, 0.0f);
        float2 peer_184 = _f2_110;
        peer_184 = reinterpret_cast<float2*>(&peerbits_183)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_184);
        unsigned long long bits_185 = 0;
        bits_185 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_185, 1);
        unsigned long long peerbits_186 = _shfl_xor_12;
        float2 _f2_111 = make_float2(0.0f, 0.0f);
        float2 peer_187 = _f2_111;
        peer_187 = reinterpret_cast<float2*>(&peerbits_186)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_187);
        unsigned long long bits_188 = 0;
        bits_188 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_188, 1);
        unsigned long long peerbits_189 = _shfl_xor_13;
        float2 _f2_112 = make_float2(0.0f, 0.0f);
        float2 peer_190 = _f2_112;
        peer_190 = reinterpret_cast<float2*>(&peerbits_189)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_190);
        unsigned long long bits_191 = 0;
        bits_191 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_191, 1);
        unsigned long long peerbits_192 = _shfl_xor_14;
        float2 _f2_113 = make_float2(0.0f, 0.0f);
        float2 peer_193 = _f2_113;
        peer_193 = reinterpret_cast<float2*>(&peerbits_192)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_193);
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
        float2 _f2_114 = make_float2(correction, correction);
        float2 corr = _f2_114;
        const int woff_194 = 0;
        int base_195 = group * 1024 + thread * 8;
        float2 a_196[4];
        float2 _f2_115 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_115;
        a_196[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_116 = make_float2(acc[2], acc[3]);
        float2 previous_197 = _f2_116;
        a_196[1] = mul_f32x2_noftz(previous_197, corr);
        float2 _f2_117 = make_float2(acc[4], acc[5]);
        float2 previous_198 = _f2_117;
        a_196[2] = mul_f32x2_noftz(previous_198, corr);
        float2 _f2_118 = make_float2(acc[6], acc[7]);
        float2 previous_199 = _f2_118;
        a_196[3] = mul_f32x2_noftz(previous_199, corr);
        float2 _f2_119 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_119;
        float2 _f2_120 = make_float2(fsrc[0], fsrc[1]);
        float2 v_200 = _f2_120;
        a_196[0] = fma_f32x2_rn_noftz(weight, v_200, a_196[0]);
        float2 _f2_121 = make_float2(fsrc[2], fsrc[3]);
        float2 v_201 = _f2_121;
        a_196[1] = fma_f32x2_rn_noftz(weight, v_201, a_196[1]);
        float2 _f2_122 = make_float2(fsrc[4], fsrc[5]);
        float2 v_202 = _f2_122;
        a_196[2] = fma_f32x2_rn_noftz(weight, v_202, a_196[2]);
        float2 _f2_123 = make_float2(fsrc[6], fsrc[7]);
        float2 v_203 = _f2_123;
        a_196[3] = fma_f32x2_rn_noftz(weight, v_203, a_196[3]);
        float2 _f2_124 = make_float2(weights[1], weights[1]);
        float2 weight_204 = _f2_124;
        float2 _f2_125 = make_float2(fsrc[28], fsrc[29]);
        float2 v_205 = _f2_125;
        a_196[0] = fma_f32x2_rn_noftz(weight_204, v_205, a_196[0]);
        float2 _f2_126 = make_float2(fsrc[30], fsrc[31]);
        float2 v_206 = _f2_126;
        a_196[1] = fma_f32x2_rn_noftz(weight_204, v_206, a_196[1]);
        float2 _f2_127 = make_float2(fsrc[32], fsrc[33]);
        float2 v_207 = _f2_127;
        a_196[2] = fma_f32x2_rn_noftz(weight_204, v_207, a_196[2]);
        float2 _f2_128 = make_float2(fsrc[34], fsrc[35]);
        float2 v_208 = _f2_128;
        a_196[3] = fma_f32x2_rn_noftz(weight_204, v_208, a_196[3]);
        float2 _f2_129 = make_float2(weights[2], weights[2]);
        float2 weight_209 = _f2_129;
        float2 _f2_130 = make_float2(fsrc[56], fsrc[57]);
        float2 v_210 = _f2_130;
        a_196[0] = fma_f32x2_rn_noftz(weight_209, v_210, a_196[0]);
        float2 _f2_131 = make_float2(fsrc[58], fsrc[59]);
        float2 v_211 = _f2_131;
        a_196[1] = fma_f32x2_rn_noftz(weight_209, v_211, a_196[1]);
        float2 _f2_132 = make_float2(fsrc[60], fsrc[61]);
        float2 v_212 = _f2_132;
        a_196[2] = fma_f32x2_rn_noftz(weight_209, v_212, a_196[2]);
        float2 _f2_133 = make_float2(fsrc[62], fsrc[63]);
        float2 v_213 = _f2_133;
        a_196[3] = fma_f32x2_rn_noftz(weight_209, v_213, a_196[3]);
        acc[0] = a_196[0].x;
        acc[1] = a_196[0].y;
        acc[2] = a_196[1].x;
        acc[3] = a_196[1].y;
        acc[4] = a_196[2].x;
        acc[5] = a_196[2].y;
        acc[6] = a_196[3].x;
        acc[7] = a_196[3].y;
        const int woff_214 = 4;
        int base_215 = (2 + group) * 1024 + thread * 8;
        float2 a_216[4];
        float2 _f2_134 = make_float2(acc[8], acc[9]);
        float2 previous_217 = _f2_134;
        a_216[0] = mul_f32x2_noftz(previous_217, corr);
        float2 _f2_135 = make_float2(acc[10], acc[11]);
        float2 previous_218 = _f2_135;
        a_216[1] = mul_f32x2_noftz(previous_218, corr);
        float2 _f2_136 = make_float2(acc[12], acc[13]);
        float2 previous_219 = _f2_136;
        a_216[2] = mul_f32x2_noftz(previous_219, corr);
        float2 _f2_137 = make_float2(acc[14], acc[15]);
        float2 previous_220 = _f2_137;
        a_216[3] = mul_f32x2_noftz(previous_220, corr);
        float2 _f2_138 = make_float2(weights[0], weights[0]);
        float2 weight_221 = _f2_138;
        float2 _f2_139 = make_float2(fsrc[8], fsrc[9]);
        float2 v_222 = _f2_139;
        a_216[0] = fma_f32x2_rn_noftz(weight_221, v_222, a_216[0]);
        float2 _f2_140 = make_float2(fsrc[10], fsrc[11]);
        float2 v_223 = _f2_140;
        a_216[1] = fma_f32x2_rn_noftz(weight_221, v_223, a_216[1]);
        float2 _f2_141 = make_float2(fsrc[12], fsrc[13]);
        float2 v_224 = _f2_141;
        a_216[2] = fma_f32x2_rn_noftz(weight_221, v_224, a_216[2]);
        float2 _f2_142 = make_float2(fsrc[14], fsrc[15]);
        float2 v_225 = _f2_142;
        a_216[3] = fma_f32x2_rn_noftz(weight_221, v_225, a_216[3]);
        float2 _f2_143 = make_float2(weights[1], weights[1]);
        float2 weight_226 = _f2_143;
        float2 _f2_144 = make_float2(fsrc[36], fsrc[37]);
        float2 v_227 = _f2_144;
        a_216[0] = fma_f32x2_rn_noftz(weight_226, v_227, a_216[0]);
        float2 _f2_145 = make_float2(fsrc[38], fsrc[39]);
        float2 v_228 = _f2_145;
        a_216[1] = fma_f32x2_rn_noftz(weight_226, v_228, a_216[1]);
        float2 _f2_146 = make_float2(fsrc[40], fsrc[41]);
        float2 v_229 = _f2_146;
        a_216[2] = fma_f32x2_rn_noftz(weight_226, v_229, a_216[2]);
        float2 _f2_147 = make_float2(fsrc[42], fsrc[43]);
        float2 v_230 = _f2_147;
        a_216[3] = fma_f32x2_rn_noftz(weight_226, v_230, a_216[3]);
        float2 _f2_148 = make_float2(weights[2], weights[2]);
        float2 weight_231 = _f2_148;
        float2 _f2_149 = make_float2(fsrc[64], fsrc[65]);
        float2 v_232 = _f2_149;
        a_216[0] = fma_f32x2_rn_noftz(weight_231, v_232, a_216[0]);
        float2 _f2_150 = make_float2(fsrc[66], fsrc[67]);
        float2 v_233 = _f2_150;
        a_216[1] = fma_f32x2_rn_noftz(weight_231, v_233, a_216[1]);
        float2 _f2_151 = make_float2(fsrc[68], fsrc[69]);
        float2 v_234 = _f2_151;
        a_216[2] = fma_f32x2_rn_noftz(weight_231, v_234, a_216[2]);
        float2 _f2_152 = make_float2(fsrc[70], fsrc[71]);
        float2 v_235 = _f2_152;
        a_216[3] = fma_f32x2_rn_noftz(weight_231, v_235, a_216[3]);
        acc[8] = a_216[0].x;
        acc[9] = a_216[0].y;
        acc[10] = a_216[1].x;
        acc[11] = a_216[1].y;
        acc[12] = a_216[2].x;
        acc[13] = a_216[2].y;
        acc[14] = a_216[3].x;
        acc[15] = a_216[3].y;
        const int woff_236 = 8;
        int base_237 = (4 + group) * 1024 + thread * 8;
        float2 a_238[4];
        float2 _f2_153 = make_float2(acc[16], acc[17]);
        float2 previous_239 = _f2_153;
        a_238[0] = mul_f32x2_noftz(previous_239, corr);
        float2 _f2_154 = make_float2(acc[18], acc[19]);
        float2 previous_240 = _f2_154;
        a_238[1] = mul_f32x2_noftz(previous_240, corr);
        float2 _f2_155 = make_float2(acc[20], acc[21]);
        float2 previous_241 = _f2_155;
        a_238[2] = mul_f32x2_noftz(previous_241, corr);
        float2 _f2_156 = make_float2(acc[22], acc[23]);
        float2 previous_242 = _f2_156;
        a_238[3] = mul_f32x2_noftz(previous_242, corr);
        float2 _f2_157 = make_float2(weights[0], weights[0]);
        float2 weight_243 = _f2_157;
        float2 _f2_158 = make_float2(fsrc[16], fsrc[17]);
        float2 v_244 = _f2_158;
        a_238[0] = fma_f32x2_rn_noftz(weight_243, v_244, a_238[0]);
        float2 _f2_159 = make_float2(fsrc[18], fsrc[19]);
        float2 v_245 = _f2_159;
        a_238[1] = fma_f32x2_rn_noftz(weight_243, v_245, a_238[1]);
        float2 _f2_160 = make_float2(fsrc[20], fsrc[21]);
        float2 v_246 = _f2_160;
        a_238[2] = fma_f32x2_rn_noftz(weight_243, v_246, a_238[2]);
        float2 _f2_161 = make_float2(fsrc[22], fsrc[23]);
        float2 v_247 = _f2_161;
        a_238[3] = fma_f32x2_rn_noftz(weight_243, v_247, a_238[3]);
        float2 _f2_162 = make_float2(weights[1], weights[1]);
        float2 weight_248 = _f2_162;
        float2 _f2_163 = make_float2(fsrc[44], fsrc[45]);
        float2 v_249 = _f2_163;
        a_238[0] = fma_f32x2_rn_noftz(weight_248, v_249, a_238[0]);
        float2 _f2_164 = make_float2(fsrc[46], fsrc[47]);
        float2 v_250 = _f2_164;
        a_238[1] = fma_f32x2_rn_noftz(weight_248, v_250, a_238[1]);
        float2 _f2_165 = make_float2(fsrc[48], fsrc[49]);
        float2 v_251 = _f2_165;
        a_238[2] = fma_f32x2_rn_noftz(weight_248, v_251, a_238[2]);
        float2 _f2_166 = make_float2(fsrc[50], fsrc[51]);
        float2 v_252 = _f2_166;
        a_238[3] = fma_f32x2_rn_noftz(weight_248, v_252, a_238[3]);
        float2 _f2_167 = make_float2(weights[2], weights[2]);
        float2 weight_253 = _f2_167;
        float2 _f2_168 = make_float2(fsrc[72], fsrc[73]);
        float2 v_254 = _f2_168;
        a_238[0] = fma_f32x2_rn_noftz(weight_253, v_254, a_238[0]);
        float2 _f2_169 = make_float2(fsrc[74], fsrc[75]);
        float2 v_255 = _f2_169;
        a_238[1] = fma_f32x2_rn_noftz(weight_253, v_255, a_238[1]);
        float2 _f2_170 = make_float2(fsrc[76], fsrc[77]);
        float2 v_256 = _f2_170;
        a_238[2] = fma_f32x2_rn_noftz(weight_253, v_256, a_238[2]);
        float2 _f2_171 = make_float2(fsrc[78], fsrc[79]);
        float2 v_257 = _f2_171;
        a_238[3] = fma_f32x2_rn_noftz(weight_253, v_257, a_238[3]);
        acc[16] = a_238[0].x;
        acc[17] = a_238[0].y;
        acc[18] = a_238[1].x;
        acc[19] = a_238[1].y;
        acc[20] = a_238[2].x;
        acc[21] = a_238[2].y;
        acc[22] = a_238[3].x;
        acc[23] = a_238[3].y;
        const int woff_258 = 12;
        int base_259 = 6144 + group * 512 + thread * 4;
        float2 a_260[4];
        float2 _f2_172 = make_float2(acc[24], acc[25]);
        float2 previous_261 = _f2_172;
        a_260[0] = mul_f32x2_noftz(previous_261, corr);
        float2 _f2_173 = make_float2(acc[26], acc[27]);
        float2 previous_262 = _f2_173;
        a_260[1] = mul_f32x2_noftz(previous_262, corr);
        float2 _f2_174 = make_float2(weights[0], weights[0]);
        float2 weight_263 = _f2_174;
        float2 _f2_175 = make_float2(fsrc[24], fsrc[25]);
        float2 v_264 = _f2_175;
        a_260[0] = fma_f32x2_rn_noftz(weight_263, v_264, a_260[0]);
        float2 _f2_176 = make_float2(fsrc[26], fsrc[27]);
        float2 v_265 = _f2_176;
        a_260[1] = fma_f32x2_rn_noftz(weight_263, v_265, a_260[1]);
        float2 _f2_177 = make_float2(weights[1], weights[1]);
        float2 weight_266 = _f2_177;
        float2 _f2_178 = make_float2(fsrc[52], fsrc[53]);
        float2 v_267 = _f2_178;
        a_260[0] = fma_f32x2_rn_noftz(weight_266, v_267, a_260[0]);
        float2 _f2_179 = make_float2(fsrc[54], fsrc[55]);
        float2 v_268 = _f2_179;
        a_260[1] = fma_f32x2_rn_noftz(weight_266, v_268, a_260[1]);
        float2 _f2_180 = make_float2(weights[2], weights[2]);
        float2 weight_269 = _f2_180;
        float2 _f2_181 = make_float2(fsrc[80], fsrc[81]);
        float2 v_270 = _f2_181;
        a_260[0] = fma_f32x2_rn_noftz(weight_269, v_270, a_260[0]);
        float2 _f2_182 = make_float2(fsrc[82], fsrc[83]);
        float2 v_271 = _f2_182;
        a_260[1] = fma_f32x2_rn_noftz(weight_269, v_271, a_260[1]);
        acc[24] = a_260[0].x;
        acc[25] = a_260[0].y;
        acc[26] = a_260[1].x;
        acc[27] = a_260[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float2 _f2_183 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_183;
        float2 _f2_184 = make_float2(acc[0], acc[1]);
        float2 v_272 = _f2_184;
        output_sq_pair = fma_f32x2_rn_noftz(v_272, v_272, output_sq_pair);
        float2 _f2_185 = make_float2(acc[2], acc[3]);
        float2 v_273 = _f2_185;
        output_sq_pair = fma_f32x2_rn_noftz(v_273, v_273, output_sq_pair);
        float2 _f2_186 = make_float2(acc[4], acc[5]);
        float2 v_274 = _f2_186;
        output_sq_pair = fma_f32x2_rn_noftz(v_274, v_274, output_sq_pair);
        float2 _f2_187 = make_float2(acc[6], acc[7]);
        float2 v_275 = _f2_187;
        output_sq_pair = fma_f32x2_rn_noftz(v_275, v_275, output_sq_pair);
        float2 _f2_188 = make_float2(acc[8], acc[9]);
        float2 v_276 = _f2_188;
        output_sq_pair = fma_f32x2_rn_noftz(v_276, v_276, output_sq_pair);
        float2 _f2_189 = make_float2(acc[10], acc[11]);
        float2 v_277 = _f2_189;
        output_sq_pair = fma_f32x2_rn_noftz(v_277, v_277, output_sq_pair);
        float2 _f2_190 = make_float2(acc[12], acc[13]);
        float2 v_278 = _f2_190;
        output_sq_pair = fma_f32x2_rn_noftz(v_278, v_278, output_sq_pair);
        float2 _f2_191 = make_float2(acc[14], acc[15]);
        float2 v_279 = _f2_191;
        output_sq_pair = fma_f32x2_rn_noftz(v_279, v_279, output_sq_pair);
        float2 _f2_192 = make_float2(acc[16], acc[17]);
        float2 v_280 = _f2_192;
        output_sq_pair = fma_f32x2_rn_noftz(v_280, v_280, output_sq_pair);
        float2 _f2_193 = make_float2(acc[18], acc[19]);
        float2 v_281 = _f2_193;
        output_sq_pair = fma_f32x2_rn_noftz(v_281, v_281, output_sq_pair);
        float2 _f2_194 = make_float2(acc[20], acc[21]);
        float2 v_282 = _f2_194;
        output_sq_pair = fma_f32x2_rn_noftz(v_282, v_282, output_sq_pair);
        float2 _f2_195 = make_float2(acc[22], acc[23]);
        float2 v_283 = _f2_195;
        output_sq_pair = fma_f32x2_rn_noftz(v_283, v_283, output_sq_pair);
        float2 _f2_196 = make_float2(acc[24], acc[25]);
        float2 v_284 = _f2_196;
        output_sq_pair = fma_f32x2_rn_noftz(v_284, v_284, output_sq_pair);
        float2 _f2_197 = make_float2(acc[26], acc[27]);
        float2 v_285 = _f2_197;
        output_sq_pair = fma_f32x2_rn_noftz(v_285, v_285, output_sq_pair);
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
            float _rsqrt_1 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_1;
        }
        float _shfl_3 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_3;
        float2 _f2_198 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_198;
        int base_286 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_199 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f2_199), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_200 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f2_200));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_287 = 2;
        const int acc_idx_288 = value_idx_287;
        float2 _f2_201 = make_float2(acc[acc_idx_288], acc[acc_idx_288 + 1]);
        float2 scaled_pair_289;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_289) : "l"(*(const unsigned long long*)&_f2_201), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_202 = make_float2(wout[acc_idx_288], wout[acc_idx_288 + 1]);
        float2 normalized_pair_290;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_290) : "l"(*(const unsigned long long*)&scaled_pair_289), "l"(*(const unsigned long long*)&_f2_202));
        output_values[value_idx_287] = normalized_pair_290.x;
        output_values[value_idx_287 + 1] = normalized_pair_290.y;
        const int value_idx_291 = 4;
        const int acc_idx_292 = value_idx_291;
        float2 _f2_203 = make_float2(acc[acc_idx_292], acc[acc_idx_292 + 1]);
        float2 scaled_pair_293;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_293) : "l"(*(const unsigned long long*)&_f2_203), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_204 = make_float2(wout[acc_idx_292], wout[acc_idx_292 + 1]);
        float2 normalized_pair_294;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_294) : "l"(*(const unsigned long long*)&scaled_pair_293), "l"(*(const unsigned long long*)&_f2_204));
        output_values[value_idx_291] = normalized_pair_294.x;
        output_values[value_idx_291 + 1] = normalized_pair_294.y;
        const int value_idx_295 = 6;
        const int acc_idx_296 = value_idx_295;
        float2 _f2_205 = make_float2(acc[acc_idx_296], acc[acc_idx_296 + 1]);
        float2 scaled_pair_297;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_297) : "l"(*(const unsigned long long*)&_f2_205), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_206 = make_float2(wout[acc_idx_296], wout[acc_idx_296 + 1]);
        float2 normalized_pair_298;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_298) : "l"(*(const unsigned long long*)&scaled_pair_297), "l"(*(const unsigned long long*)&_f2_206));
        output_values[value_idx_295] = normalized_pair_298.x;
        output_values[value_idx_295 + 1] = normalized_pair_298.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_286 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_299 = (2 + group) * 1024 + thread * 8;
        float output_values_300[8];
        const int value_idx_301 = 0;
        const int acc_idx_302 = 8 + value_idx_301;
        float2 _f2_207 = make_float2(acc[acc_idx_302], acc[acc_idx_302 + 1]);
        float2 scaled_pair_303;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_303) : "l"(*(const unsigned long long*)&_f2_207), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_208 = make_float2(wout[acc_idx_302], wout[acc_idx_302 + 1]);
        float2 normalized_pair_304;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_304) : "l"(*(const unsigned long long*)&scaled_pair_303), "l"(*(const unsigned long long*)&_f2_208));
        output_values_300[value_idx_301] = normalized_pair_304.x;
        output_values_300[value_idx_301 + 1] = normalized_pair_304.y;
        const int value_idx_305 = 2;
        const int acc_idx_306 = 8 + value_idx_305;
        float2 _f2_209 = make_float2(acc[acc_idx_306], acc[acc_idx_306 + 1]);
        float2 scaled_pair_307;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_307) : "l"(*(const unsigned long long*)&_f2_209), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_210 = make_float2(wout[acc_idx_306], wout[acc_idx_306 + 1]);
        float2 normalized_pair_308;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_308) : "l"(*(const unsigned long long*)&scaled_pair_307), "l"(*(const unsigned long long*)&_f2_210));
        output_values_300[value_idx_305] = normalized_pair_308.x;
        output_values_300[value_idx_305 + 1] = normalized_pair_308.y;
        const int value_idx_309 = 4;
        const int acc_idx_310 = 8 + value_idx_309;
        float2 _f2_211 = make_float2(acc[acc_idx_310], acc[acc_idx_310 + 1]);
        float2 scaled_pair_311;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_311) : "l"(*(const unsigned long long*)&_f2_211), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_212 = make_float2(wout[acc_idx_310], wout[acc_idx_310 + 1]);
        float2 normalized_pair_312;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_312) : "l"(*(const unsigned long long*)&scaled_pair_311), "l"(*(const unsigned long long*)&_f2_212));
        output_values_300[value_idx_309] = normalized_pair_312.x;
        output_values_300[value_idx_309 + 1] = normalized_pair_312.y;
        const int value_idx_313 = 6;
        const int acc_idx_314 = 8 + value_idx_313;
        float2 _f2_213 = make_float2(acc[acc_idx_314], acc[acc_idx_314 + 1]);
        float2 scaled_pair_315;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_315) : "l"(*(const unsigned long long*)&_f2_213), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_214 = make_float2(wout[acc_idx_314], wout[acc_idx_314 + 1]);
        float2 normalized_pair_316;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_316) : "l"(*(const unsigned long long*)&scaled_pair_315), "l"(*(const unsigned long long*)&_f2_214));
        output_values_300[value_idx_313] = normalized_pair_316.x;
        output_values_300[value_idx_313 + 1] = normalized_pair_316.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_300[0 + 0], output_values_300[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_300[0 + 2], output_values_300[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_300[0 + 4], output_values_300[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_300[0 + 6], output_values_300[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_299 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_317 = (4 + group) * 1024 + thread * 8;
        float output_values_318[8];
        const int value_idx_319 = 0;
        const int acc_idx_320 = 16 + value_idx_319;
        float2 _f2_215 = make_float2(acc[acc_idx_320], acc[acc_idx_320 + 1]);
        float2 scaled_pair_321;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_321) : "l"(*(const unsigned long long*)&_f2_215), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_216 = make_float2(wout[acc_idx_320], wout[acc_idx_320 + 1]);
        float2 normalized_pair_322;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_322) : "l"(*(const unsigned long long*)&scaled_pair_321), "l"(*(const unsigned long long*)&_f2_216));
        output_values_318[value_idx_319] = normalized_pair_322.x;
        output_values_318[value_idx_319 + 1] = normalized_pair_322.y;
        const int value_idx_323 = 2;
        const int acc_idx_324 = 16 + value_idx_323;
        float2 _f2_217 = make_float2(acc[acc_idx_324], acc[acc_idx_324 + 1]);
        float2 scaled_pair_325;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_325) : "l"(*(const unsigned long long*)&_f2_217), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_218 = make_float2(wout[acc_idx_324], wout[acc_idx_324 + 1]);
        float2 normalized_pair_326;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_326) : "l"(*(const unsigned long long*)&scaled_pair_325), "l"(*(const unsigned long long*)&_f2_218));
        output_values_318[value_idx_323] = normalized_pair_326.x;
        output_values_318[value_idx_323 + 1] = normalized_pair_326.y;
        const int value_idx_327 = 4;
        const int acc_idx_328 = 16 + value_idx_327;
        float2 _f2_219 = make_float2(acc[acc_idx_328], acc[acc_idx_328 + 1]);
        float2 scaled_pair_329;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_329) : "l"(*(const unsigned long long*)&_f2_219), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_220 = make_float2(wout[acc_idx_328], wout[acc_idx_328 + 1]);
        float2 normalized_pair_330;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_330) : "l"(*(const unsigned long long*)&scaled_pair_329), "l"(*(const unsigned long long*)&_f2_220));
        output_values_318[value_idx_327] = normalized_pair_330.x;
        output_values_318[value_idx_327 + 1] = normalized_pair_330.y;
        const int value_idx_331 = 6;
        const int acc_idx_332 = 16 + value_idx_331;
        float2 _f2_221 = make_float2(acc[acc_idx_332], acc[acc_idx_332 + 1]);
        float2 scaled_pair_333;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_333) : "l"(*(const unsigned long long*)&_f2_221), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_222 = make_float2(wout[acc_idx_332], wout[acc_idx_332 + 1]);
        float2 normalized_pair_334;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_334) : "l"(*(const unsigned long long*)&scaled_pair_333), "l"(*(const unsigned long long*)&_f2_222));
        output_values_318[value_idx_331] = normalized_pair_334.x;
        output_values_318[value_idx_331 + 1] = normalized_pair_334.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_318[0 + 0], output_values_318[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_318[0 + 2], output_values_318[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_318[0 + 4], output_values_318[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_318[0 + 6], output_values_318[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_317 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_335 = 6144 + group * 512 + thread * 4;
        float output_values_336[8];
        const int value_idx_337 = 0;
        const int acc_idx_338 = 24 + value_idx_337;
        float2 _f2_223 = make_float2(acc[acc_idx_338], acc[acc_idx_338 + 1]);
        float2 scaled_pair_339;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_339) : "l"(*(const unsigned long long*)&_f2_223), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_224 = make_float2(wout[acc_idx_338], wout[acc_idx_338 + 1]);
        float2 normalized_pair_340;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_340) : "l"(*(const unsigned long long*)&scaled_pair_339), "l"(*(const unsigned long long*)&_f2_224));
        output_values_336[value_idx_337] = normalized_pair_340.x;
        output_values_336[value_idx_337 + 1] = normalized_pair_340.y;
        const int value_idx_341 = 2;
        const int acc_idx_342 = 24 + value_idx_341;
        float2 _f2_225 = make_float2(acc[acc_idx_342], acc[acc_idx_342 + 1]);
        float2 scaled_pair_343;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_343) : "l"(*(const unsigned long long*)&_f2_225), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_226 = make_float2(wout[acc_idx_342], wout[acc_idx_342 + 1]);
        float2 normalized_pair_344;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_344) : "l"(*(const unsigned long long*)&scaled_pair_343), "l"(*(const unsigned long long*)&_f2_226));
        output_values_336[value_idx_341] = normalized_pair_344.x;
        output_values_336[value_idx_341 + 1] = normalized_pair_344.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_336[0 + 0], output_values_336[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_336[0 + 2], output_values_336[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_335]) = _pk2;
        }
    }
}

} // extern "C"
