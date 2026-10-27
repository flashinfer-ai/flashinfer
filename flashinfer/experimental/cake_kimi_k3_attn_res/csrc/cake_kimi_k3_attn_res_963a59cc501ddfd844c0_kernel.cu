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
kernel_cake_kimi_k3_attn_res_963a59cc501ddfd844c0(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
        unsigned int words[28];
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
            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_2[0 + 0] = _uv4_2.x;
            _vec_load_2[0 + 1] = _uv4_2.y;
            _vec_load_2[0 + 2] = _uv4_2.z;
            _vec_load_2[0 + 3] = _uv4_2.w;
        }
        dwords[woff] = _vec_load_2[0];
        dwords[woff + 1] = _vec_load_2[1];
        dwords[woff + 2] = _vec_load_2[2];
        dwords[woff + 3] = _vec_load_2[3];
        float _vec_load_3[8];
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
                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_3[_pair]));
                }
            }
        }
        float _vec_load_4[8];
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
                        : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        q[0] = _vec_load_3[0] * _vec_load_4[0];
        q[1] = _vec_load_3[1] * _vec_load_4[1];
        q[2] = _vec_load_3[2] * _vec_load_4[2];
        q[3] = _vec_load_3[3] * _vec_load_4[3];
        q[4] = _vec_load_3[4] * _vec_load_4[4];
        q[5] = _vec_load_3[5] * _vec_load_4[5];
        q[6] = _vec_load_3[6] * _vec_load_4[6];
        q[7] = _vec_load_3[7] * _vec_load_4[7];
        float _vec_load_5[8];
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
                        : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        wout[0] = _vec_load_5[0];
        wout[1] = _vec_load_5[1];
        wout[2] = _vec_load_5[2];
        wout[3] = _vec_load_5[3];
        wout[4] = _vec_load_5[4];
        wout[5] = _vec_load_5[5];
        wout[6] = _vec_load_5[6];
        wout[7] = _vec_load_5[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_6[4];
        {
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_6[0 + 0] = _uv4_6.x;
            _vec_load_6[0 + 1] = _uv4_6.y;
            _vec_load_6[0 + 2] = _uv4_6.z;
            _vec_load_6[0 + 3] = _uv4_6.w;
        }
        words[woff_1] = _vec_load_6[0];
        words[woff_1 + 1] = _vec_load_6[1];
        words[woff_1 + 2] = _vec_load_6[2];
        words[woff_1 + 3] = _vec_load_6[3];
        unsigned int _vec_load_7[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_7[0 + 0] = _uv4_7.x;
            _vec_load_7[0 + 1] = _uv4_7.y;
            _vec_load_7[0 + 2] = _uv4_7.z;
            _vec_load_7[0 + 3] = _uv4_7.w;
        }
        words[14 + woff_1] = _vec_load_7[0];
        words[14 + woff_1 + 1] = _vec_load_7[1];
        words[14 + woff_1 + 2] = _vec_load_7[2];
        words[14 + woff_1 + 3] = _vec_load_7[3];
        unsigned int _vec_load_8[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_8[0 + 0] = _uv4_8.x;
            _vec_load_8[0 + 1] = _uv4_8.y;
            _vec_load_8[0 + 2] = _uv4_8.z;
            _vec_load_8[0 + 3] = _uv4_8.w;
        }
        dwords[woff_1] = _vec_load_8[0];
        dwords[woff_1 + 1] = _vec_load_8[1];
        dwords[woff_1 + 2] = _vec_load_8[2];
        dwords[woff_1 + 3] = _vec_load_8[3];
        float _vec_load_9[8];
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
                        : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        float _vec_load_10[8];
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
                        : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        q[8] = _vec_load_9[0] * _vec_load_10[0];
        q[9] = _vec_load_9[1] * _vec_load_10[1];
        q[10] = _vec_load_9[2] * _vec_load_10[2];
        q[11] = _vec_load_9[3] * _vec_load_10[3];
        q[12] = _vec_load_9[4] * _vec_load_10[4];
        q[13] = _vec_load_9[5] * _vec_load_10[5];
        q[14] = _vec_load_9[6] * _vec_load_10[6];
        q[15] = _vec_load_9[7] * _vec_load_10[7];
        float _vec_load_11[8];
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
                        : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        wout[8] = _vec_load_11[0];
        wout[9] = _vec_load_11[1];
        wout[10] = _vec_load_11[2];
        wout[11] = _vec_load_11[3];
        wout[12] = _vec_load_11[4];
        wout[13] = _vec_load_11[5];
        wout[14] = _vec_load_11[6];
        wout[15] = _vec_load_11[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_12[0 + 0] = _uv4_12.x;
            _vec_load_12[0 + 1] = _uv4_12.y;
            _vec_load_12[0 + 2] = _uv4_12.z;
            _vec_load_12[0 + 3] = _uv4_12.w;
        }
        words[woff_3] = _vec_load_12[0];
        words[woff_3 + 1] = _vec_load_12[1];
        words[woff_3 + 2] = _vec_load_12[2];
        words[woff_3 + 3] = _vec_load_12[3];
        unsigned int _vec_load_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_13[0 + 0] = _uv4_13.x;
            _vec_load_13[0 + 1] = _uv4_13.y;
            _vec_load_13[0 + 2] = _uv4_13.z;
            _vec_load_13[0 + 3] = _uv4_13.w;
        }
        words[14 + woff_3] = _vec_load_13[0];
        words[14 + woff_3 + 1] = _vec_load_13[1];
        words[14 + woff_3 + 2] = _vec_load_13[2];
        words[14 + woff_3 + 3] = _vec_load_13[3];
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        dwords[woff_3] = _vec_load_14[0];
        dwords[woff_3 + 1] = _vec_load_14[1];
        dwords[woff_3 + 2] = _vec_load_14[2];
        dwords[woff_3 + 3] = _vec_load_14[3];
        float _vec_load_15[8];
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
                        : "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
        }
        float _vec_load_16[8];
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
                        : "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_16[_pair]));
                }
            }
        }
        q[16] = _vec_load_15[0] * _vec_load_16[0];
        q[17] = _vec_load_15[1] * _vec_load_16[1];
        q[18] = _vec_load_15[2] * _vec_load_16[2];
        q[19] = _vec_load_15[3] * _vec_load_16[3];
        q[20] = _vec_load_15[4] * _vec_load_16[4];
        q[21] = _vec_load_15[5] * _vec_load_16[5];
        q[22] = _vec_load_15[6] * _vec_load_16[6];
        q[23] = _vec_load_15[7] * _vec_load_16[7];
        float _vec_load_17[8];
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
                        : "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_17[_pair]));
                }
            }
        }
        wout[16] = _vec_load_17[0];
        wout[17] = _vec_load_17[1];
        wout[18] = _vec_load_17[2];
        wout[19] = _vec_load_17[3];
        wout[20] = _vec_load_17[4];
        wout[21] = _vec_load_17[5];
        wout[22] = _vec_load_17[6];
        wout[23] = _vec_load_17[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_18[1];
        {
            _vec_load_18[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_18[0];
        unsigned int _vec_load_19[1];
        {
            _vec_load_19[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_19[0];
        unsigned int _vec_load_20[1];
        {
            _vec_load_20[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_20[0];
        unsigned int _vec_load_21[1];
        {
            _vec_load_21[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_21[0];
        unsigned int _vec_load_22[1];
        {
            _vec_load_22[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_22[0];
        unsigned int _vec_load_23[1];
        {
            _vec_load_23[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_23[0];
        float _vec_load_24[4];
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
                    : "=f"((&_vec_load_24[0 + _pair * 2])[0]), "=f"((&_vec_load_24[0 + _pair * 2])[1])
                    : "r"(_vpairs_18[_pair]));
            }
        }
        float _vec_load_25[4];
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
                    : "=f"((&_vec_load_25[0 + _pair * 2])[0]), "=f"((&_vec_load_25[0 + _pair * 2])[1])
                    : "r"(_vpairs_19[_pair]));
            }
        }
        q[24] = _vec_load_24[0] * _vec_load_25[0];
        q[25] = _vec_load_24[1] * _vec_load_25[1];
        q[26] = _vec_load_24[2] * _vec_load_25[2];
        q[27] = _vec_load_24[3] * _vec_load_25[3];
        float _vec_load_26[4];
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
                    : "=f"((&_vec_load_26[0 + _pair * 2])[0]), "=f"((&_vec_load_26[0 + _pair * 2])[1])
                    : "r"(_vpairs_20[_pair]));
            }
        }
        wout[24] = _vec_load_26[0];
        wout[25] = _vec_load_26[1];
        wout[26] = _vec_load_26[2];
        wout[27] = _vec_load_26[3];
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
        float fsrc[56];
        sq[0] = make_float2(0.0f, 0.0f);
        dot[0] = make_float2(0.0f, 0.0f);
        sq[1] = make_float2(0.0f, 0.0f);
        dot[1] = make_float2(0.0f, 0.0f);
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
        __nv_bfloat162 a = __as_bf16x2(sw_14[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_14[0] = __as_u32(mixed);
        words[14 + woff_7] = sw_14[0];
        __nv_bfloat162 a_15 = __as_bf16x2(sw_14[1]);
        __nv_bfloat162 d_16 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_17 = a_15 + d_16;
        sw_14[1] = __as_u32(mixed_17);
        words[14 + woff_7 + 1] = sw_14[1];
        __nv_bfloat162 a_18 = __as_bf16x2(sw_14[2]);
        __nv_bfloat162 d_19 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_20 = a_18 + d_19;
        sw_14[2] = __as_u32(mixed_20);
        words[14 + woff_7 + 2] = sw_14[2];
        __nv_bfloat162 a_21 = __as_bf16x2(sw_14[3]);
        __nv_bfloat162 d_22 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_23 = a_21 + d_22;
        sw_14[3] = __as_u32(mixed_23);
        words[14 + woff_7 + 3] = sw_14[3];
        {
            int4 _iv4 = make_int4(sw_14[0 + 0], sw_14[0 + 1], sw_14[0 + 2], sw_14[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
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
        float2 v_24 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
        float2 qp_25 = make_float2(q[0], q[1]);
        sq[1] = fma_f32x2_rn_noftz(v_24, v_24, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_24, qp_25, dot[1]);
        float2 v_26 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
        float2 qp_27 = make_float2(q[2], q[3]);
        sq[1] = fma_f32x2_rn_noftz(v_26, v_26, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_26, qp_27, dot[1]);
        float2 v_28 = make_float2(_unpack_bf16_1[4], _unpack_bf16_1[5]);
        float2 qp_29 = make_float2(q[4], q[5]);
        sq[1] = fma_f32x2_rn_noftz(v_28, v_28, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_28, qp_29, dot[1]);
        float2 v_30 = make_float2(_unpack_bf16_1[6], _unpack_bf16_1[7]);
        float2 qp_31 = make_float2(q[6], q[7]);
        sq[1] = fma_f32x2_rn_noftz(v_30, v_30, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_30, qp_31, dot[1]);
        int base_32 = (2 + group) * 1024 + thread * 8;
        const int woff_33 = 4;
        unsigned int sw_34[4];
        sw_34[0] = words[woff_33];
        sw_34[1] = words[woff_33 + 1];
        sw_34[2] = words[woff_33 + 2];
        sw_34[3] = words[woff_33 + 3];
        float _unpack_bf16_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                : "r"(sw_34[_pair]));
        }
        fsrc[8] = _unpack_bf16_2[0];
        fsrc[9] = _unpack_bf16_2[1];
        fsrc[10] = _unpack_bf16_2[2];
        fsrc[11] = _unpack_bf16_2[3];
        fsrc[12] = _unpack_bf16_2[4];
        fsrc[13] = _unpack_bf16_2[5];
        fsrc[14] = _unpack_bf16_2[6];
        fsrc[15] = _unpack_bf16_2[7];
        float2 v_35 = make_float2(_unpack_bf16_2[0], _unpack_bf16_2[1]);
        float2 qp_36 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_35, v_35, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_35, qp_36, dot[0]);
        float2 v_37 = make_float2(_unpack_bf16_2[2], _unpack_bf16_2[3]);
        float2 qp_38 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_37, v_37, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_37, qp_38, dot[0]);
        float2 v_39 = make_float2(_unpack_bf16_2[4], _unpack_bf16_2[5]);
        float2 qp_40 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_39, v_39, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_39, qp_40, dot[0]);
        float2 v_41 = make_float2(_unpack_bf16_2[6], _unpack_bf16_2[7]);
        float2 qp_42 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_41, v_41, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_41, qp_42, dot[0]);
        unsigned int sw_43[4];
        sw_43[0] = words[14 + woff_33];
        sw_43[1] = words[14 + woff_33 + 1];
        sw_43[2] = words[14 + woff_33 + 2];
        sw_43[3] = words[14 + woff_33 + 3];
        __nv_bfloat162 a_44 = __as_bf16x2(sw_43[0]);
        __nv_bfloat162 d_45 = __as_bf16x2(dwords[woff_33]);
        __nv_bfloat162 mixed_46 = a_44 + d_45;
        sw_43[0] = __as_u32(mixed_46);
        words[14 + woff_33] = sw_43[0];
        __nv_bfloat162 a_47 = __as_bf16x2(sw_43[1]);
        __nv_bfloat162 d_48 = __as_bf16x2(dwords[woff_33 + 1]);
        __nv_bfloat162 mixed_49 = a_47 + d_48;
        sw_43[1] = __as_u32(mixed_49);
        words[14 + woff_33 + 1] = sw_43[1];
        __nv_bfloat162 a_50 = __as_bf16x2(sw_43[2]);
        __nv_bfloat162 d_51 = __as_bf16x2(dwords[woff_33 + 2]);
        __nv_bfloat162 mixed_52 = a_50 + d_51;
        sw_43[2] = __as_u32(mixed_52);
        words[14 + woff_33 + 2] = sw_43[2];
        __nv_bfloat162 a_53 = __as_bf16x2(sw_43[3]);
        __nv_bfloat162 d_54 = __as_bf16x2(dwords[woff_33 + 3]);
        __nv_bfloat162 mixed_55 = a_53 + d_54;
        sw_43[3] = __as_u32(mixed_55);
        words[14 + woff_33 + 3] = sw_43[3];
        {
            int4 _iv4 = make_int4(sw_43[0 + 0], sw_43[0 + 1], sw_43[0 + 2], sw_43[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_32)) + 0) = _iv4;
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
                : "r"(sw_43[_pair]));
        }
        fsrc[36] = _unpack_bf16_3[0];
        fsrc[37] = _unpack_bf16_3[1];
        fsrc[38] = _unpack_bf16_3[2];
        fsrc[39] = _unpack_bf16_3[3];
        fsrc[40] = _unpack_bf16_3[4];
        fsrc[41] = _unpack_bf16_3[5];
        fsrc[42] = _unpack_bf16_3[6];
        fsrc[43] = _unpack_bf16_3[7];
        float2 v_56 = make_float2(_unpack_bf16_3[0], _unpack_bf16_3[1]);
        float2 qp_57 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_56, v_56, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_56, qp_57, dot[1]);
        float2 v_58 = make_float2(_unpack_bf16_3[2], _unpack_bf16_3[3]);
        float2 qp_59 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_58, v_58, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_58, qp_59, dot[1]);
        float2 v_60 = make_float2(_unpack_bf16_3[4], _unpack_bf16_3[5]);
        float2 qp_61 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_60, v_60, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_60, qp_61, dot[1]);
        float2 v_62 = make_float2(_unpack_bf16_3[6], _unpack_bf16_3[7]);
        float2 qp_63 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_62, v_62, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_62, qp_63, dot[1]);
        int base_64 = (4 + group) * 1024 + thread * 8;
        const int woff_65 = 8;
        unsigned int sw_66[4];
        sw_66[0] = words[woff_65];
        sw_66[1] = words[woff_65 + 1];
        sw_66[2] = words[woff_65 + 2];
        sw_66[3] = words[woff_65 + 3];
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                : "r"(sw_66[_pair]));
        }
        fsrc[16] = _unpack_bf16_4[0];
        fsrc[17] = _unpack_bf16_4[1];
        fsrc[18] = _unpack_bf16_4[2];
        fsrc[19] = _unpack_bf16_4[3];
        fsrc[20] = _unpack_bf16_4[4];
        fsrc[21] = _unpack_bf16_4[5];
        fsrc[22] = _unpack_bf16_4[6];
        fsrc[23] = _unpack_bf16_4[7];
        float2 v_67 = make_float2(_unpack_bf16_4[0], _unpack_bf16_4[1]);
        float2 qp_68 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_67, v_67, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_67, qp_68, dot[0]);
        float2 v_69 = make_float2(_unpack_bf16_4[2], _unpack_bf16_4[3]);
        float2 qp_70 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_69, v_69, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_69, qp_70, dot[0]);
        float2 v_71 = make_float2(_unpack_bf16_4[4], _unpack_bf16_4[5]);
        float2 qp_72 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_71, v_71, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_71, qp_72, dot[0]);
        float2 v_73 = make_float2(_unpack_bf16_4[6], _unpack_bf16_4[7]);
        float2 qp_74 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_73, v_73, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_73, qp_74, dot[0]);
        unsigned int sw_75[4];
        sw_75[0] = words[14 + woff_65];
        sw_75[1] = words[14 + woff_65 + 1];
        sw_75[2] = words[14 + woff_65 + 2];
        sw_75[3] = words[14 + woff_65 + 3];
        __nv_bfloat162 a_76 = __as_bf16x2(sw_75[0]);
        __nv_bfloat162 d_77 = __as_bf16x2(dwords[woff_65]);
        __nv_bfloat162 mixed_78 = a_76 + d_77;
        sw_75[0] = __as_u32(mixed_78);
        words[14 + woff_65] = sw_75[0];
        __nv_bfloat162 a_79 = __as_bf16x2(sw_75[1]);
        __nv_bfloat162 d_80 = __as_bf16x2(dwords[woff_65 + 1]);
        __nv_bfloat162 mixed_81 = a_79 + d_80;
        sw_75[1] = __as_u32(mixed_81);
        words[14 + woff_65 + 1] = sw_75[1];
        __nv_bfloat162 a_82 = __as_bf16x2(sw_75[2]);
        __nv_bfloat162 d_83 = __as_bf16x2(dwords[woff_65 + 2]);
        __nv_bfloat162 mixed_84 = a_82 + d_83;
        sw_75[2] = __as_u32(mixed_84);
        words[14 + woff_65 + 2] = sw_75[2];
        __nv_bfloat162 a_85 = __as_bf16x2(sw_75[3]);
        __nv_bfloat162 d_86 = __as_bf16x2(dwords[woff_65 + 3]);
        __nv_bfloat162 mixed_87 = a_85 + d_86;
        sw_75[3] = __as_u32(mixed_87);
        words[14 + woff_65 + 3] = sw_75[3];
        {
            int4 _iv4 = make_int4(sw_75[0 + 0], sw_75[0 + 1], sw_75[0 + 2], sw_75[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_64)) + 0) = _iv4;
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
                : "r"(sw_75[_pair]));
        }
        fsrc[44] = _unpack_bf16_5[0];
        fsrc[45] = _unpack_bf16_5[1];
        fsrc[46] = _unpack_bf16_5[2];
        fsrc[47] = _unpack_bf16_5[3];
        fsrc[48] = _unpack_bf16_5[4];
        fsrc[49] = _unpack_bf16_5[5];
        fsrc[50] = _unpack_bf16_5[6];
        fsrc[51] = _unpack_bf16_5[7];
        float2 v_88 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 qp_89 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_88, v_88, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_88, qp_89, dot[1]);
        float2 v_90 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 qp_91 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_90, v_90, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_90, qp_91, dot[1]);
        float2 v_92 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 qp_93 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_92, v_92, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_92, qp_93, dot[1]);
        float2 v_94 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 qp_95 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_94, v_94, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_94, qp_95, dot[1]);
        int base_96 = 6144 + group * 512 + thread * 4;
        const int woff_97 = 12;
        unsigned int sw_98[4];
        sw_98[0] = words[woff_97];
        sw_98[1] = words[woff_97 + 1];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(sw_98[_pair]));
        }
        fsrc[24] = _unpack_bf16_6[0];
        fsrc[25] = _unpack_bf16_6[1];
        fsrc[26] = _unpack_bf16_6[2];
        fsrc[27] = _unpack_bf16_6[3];
        float2 v_99 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        sq[0] = fma_f32x2_rn_noftz(v_99, v_99, sq[0]);
        float2 v_100 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        sq[0] = fma_f32x2_rn_noftz(v_100, v_100, sq[0]);
        float2 v_101 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 qp_102 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_101, qp_102, dot[0]);
        float2 v_103 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 qp_104 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_103, qp_104, dot[0]);
        unsigned int sw_105[4];
        sw_105[0] = words[14 + woff_97];
        sw_105[1] = words[14 + woff_97 + 1];
        __nv_bfloat162 a_106 = __as_bf16x2(sw_105[0]);
        __nv_bfloat162 d_107 = __as_bf16x2(dwords[woff_97]);
        __nv_bfloat162 mixed_108 = a_106 + d_107;
        sw_105[0] = __as_u32(mixed_108);
        words[14 + woff_97] = sw_105[0];
        __nv_bfloat162 a_109 = __as_bf16x2(sw_105[1]);
        __nv_bfloat162 d_110 = __as_bf16x2(dwords[woff_97 + 1]);
        __nv_bfloat162 mixed_111 = a_109 + d_110;
        sw_105[1] = __as_u32(mixed_111);
        words[14 + woff_97 + 1] = sw_105[1];
        {
            int2 _iv2 = make_int2(sw_105[0 + 0], sw_105[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_96)) + 0) = _iv2;
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
                : "r"(sw_105[_pair]));
        }
        fsrc[52] = _unpack_bf16_7[0];
        fsrc[53] = _unpack_bf16_7[1];
        fsrc[54] = _unpack_bf16_7[2];
        fsrc[55] = _unpack_bf16_7[3];
        float2 v_112 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        sq[1] = fma_f32x2_rn_noftz(v_112, v_112, sq[1]);
        float2 v_113 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        sq[1] = fma_f32x2_rn_noftz(v_113, v_113, sq[1]);
        float2 v_114 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 qp_115 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_114, qp_115, dot[1]);
        float2 v_116 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 qp_117 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_116, qp_117, dot[1]);
        float2 pairs[2];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        unsigned long long bits = 0;
        float2 _bit_cast_src_21 = pairs[0];
        bits = reinterpret_cast<unsigned long long*>(&_bit_cast_src_21)[0];
        unsigned long long peerbits = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        float2 peer = make_float2(0.0f, 0.0f);
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_118 = 0;
        float2 _bit_cast_src_22 = pairs[1];
        bits_118 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_22)[0];
        unsigned long long peerbits_119 = __shfl_xor_sync(0xFFFFFFFF, bits_118, 16);
        float2 peer_120 = make_float2(0.0f, 0.0f);
        peer_120 = reinterpret_cast<float2*>(&peerbits_119)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_120);
        unsigned long long bits_121 = 0;
        float2 _bit_cast_src_23 = pairs[0];
        bits_121 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_23)[0];
        unsigned long long peerbits_122 = __shfl_xor_sync(0xFFFFFFFF, bits_121, 8);
        float2 peer_123 = make_float2(0.0f, 0.0f);
        peer_123 = reinterpret_cast<float2*>(&peerbits_122)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_123);
        unsigned long long bits_124 = 0;
        float2 _bit_cast_src_24 = pairs[1];
        bits_124 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_24)[0];
        unsigned long long peerbits_125 = __shfl_xor_sync(0xFFFFFFFF, bits_124, 8);
        float2 peer_126 = make_float2(0.0f, 0.0f);
        peer_126 = reinterpret_cast<float2*>(&peerbits_125)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_126);
        unsigned long long bits_127 = 0;
        float2 _bit_cast_src_25 = pairs[0];
        bits_127 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_25)[0];
        unsigned long long peerbits_128 = __shfl_xor_sync(0xFFFFFFFF, bits_127, 4);
        float2 peer_129 = make_float2(0.0f, 0.0f);
        peer_129 = reinterpret_cast<float2*>(&peerbits_128)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_129);
        unsigned long long bits_130 = 0;
        float2 _bit_cast_src_26 = pairs[1];
        bits_130 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_26)[0];
        unsigned long long peerbits_131 = __shfl_xor_sync(0xFFFFFFFF, bits_130, 4);
        float2 peer_132 = make_float2(0.0f, 0.0f);
        peer_132 = reinterpret_cast<float2*>(&peerbits_131)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_132);
        unsigned long long bits_133 = 0;
        float2 _bit_cast_src_27 = pairs[0];
        bits_133 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_27)[0];
        unsigned long long peerbits_134 = __shfl_xor_sync(0xFFFFFFFF, bits_133, 2);
        float2 peer_135 = make_float2(0.0f, 0.0f);
        peer_135 = reinterpret_cast<float2*>(&peerbits_134)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_135);
        unsigned long long bits_136 = 0;
        float2 _bit_cast_src_28 = pairs[1];
        bits_136 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_28)[0];
        unsigned long long peerbits_137 = __shfl_xor_sync(0xFFFFFFFF, bits_136, 2);
        float2 peer_138 = make_float2(0.0f, 0.0f);
        peer_138 = reinterpret_cast<float2*>(&peerbits_137)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_138);
        unsigned long long bits_139 = 0;
        float2 _bit_cast_src_29 = pairs[0];
        bits_139 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_29)[0];
        unsigned long long peerbits_140 = __shfl_xor_sync(0xFFFFFFFF, bits_139, 1);
        float2 peer_141 = make_float2(0.0f, 0.0f);
        peer_141 = reinterpret_cast<float2*>(&peerbits_140)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_141);
        unsigned long long bits_142 = 0;
        float2 _bit_cast_src_30 = pairs[1];
        bits_142 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_30)[0];
        unsigned long long peerbits_143 = __shfl_xor_sync(0xFFFFFFFF, bits_142, 1);
        float2 peer_144 = make_float2(0.0f, 0.0f);
        peer_144 = reinterpret_cast<float2*>(&peerbits_143)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_144);
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
        float _shfl_down_sync_31 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_31;
        float _shfl_down_sync_32 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_32;
        float _shfl_down_sync_33 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_33;
        float _shfl_down_sync_34 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_34;
        float _shfl_down_sync_35 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_35;
        float _shfl_down_sync_36 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_36;
        float logit = 0.0f;
        if (stat_n < 2 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[2];
        float _shfl_sync_37 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_37;
        float _shfl_sync_38 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_38;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028235e+38f;
        max_chunk = fmaxf(max_chunk, logits[0]);
        max_chunk = fmaxf(max_chunk, logits[1]);
        float max_new = fmaxf(max_running, max_chunk);
        float correction = approx_exp2((max_running - max_new) * 1.442695f);
        float weights[2];
        float sum_weights = 0.0f;
        float _exp2_39 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_39;
        sum_weights += weights[0];
        float _exp2_40 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_40;
        sum_weights += weights[1];
        float2 corr = make_float2(correction, correction);
        const int woff_145 = 0;
        int base_146 = group * 1024 + thread * 8;
        float2 a_147[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_147[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_148 = make_float2(acc[2], acc[3]);
        a_147[1] = mul_f32x2_noftz(previous_148, corr);
        float2 previous_149 = make_float2(acc[4], acc[5]);
        a_147[2] = mul_f32x2_noftz(previous_149, corr);
        float2 previous_150 = make_float2(acc[6], acc[7]);
        a_147[3] = mul_f32x2_noftz(previous_150, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_151 = make_float2(fsrc[0], fsrc[1]);
        a_147[0] = fma_f32x2_rn_noftz(weight, v_151, a_147[0]);
        float2 v_152 = make_float2(fsrc[2], fsrc[3]);
        a_147[1] = fma_f32x2_rn_noftz(weight, v_152, a_147[1]);
        float2 v_153 = make_float2(fsrc[4], fsrc[5]);
        a_147[2] = fma_f32x2_rn_noftz(weight, v_153, a_147[2]);
        float2 v_154 = make_float2(fsrc[6], fsrc[7]);
        a_147[3] = fma_f32x2_rn_noftz(weight, v_154, a_147[3]);
        float2 weight_155 = make_float2(weights[1], weights[1]);
        float2 v_156 = make_float2(fsrc[28], fsrc[29]);
        a_147[0] = fma_f32x2_rn_noftz(weight_155, v_156, a_147[0]);
        float2 v_157 = make_float2(fsrc[30], fsrc[31]);
        a_147[1] = fma_f32x2_rn_noftz(weight_155, v_157, a_147[1]);
        float2 v_158 = make_float2(fsrc[32], fsrc[33]);
        a_147[2] = fma_f32x2_rn_noftz(weight_155, v_158, a_147[2]);
        float2 v_159 = make_float2(fsrc[34], fsrc[35]);
        a_147[3] = fma_f32x2_rn_noftz(weight_155, v_159, a_147[3]);
        acc[0] = a_147[0].x;
        acc[1] = a_147[0].y;
        acc[2] = a_147[1].x;
        acc[3] = a_147[1].y;
        acc[4] = a_147[2].x;
        acc[5] = a_147[2].y;
        acc[6] = a_147[3].x;
        acc[7] = a_147[3].y;
        const int woff_160 = 4;
        int base_161 = (2 + group) * 1024 + thread * 8;
        float2 a_162[4];
        float2 previous_163 = make_float2(acc[8], acc[9]);
        a_162[0] = mul_f32x2_noftz(previous_163, corr);
        float2 previous_164 = make_float2(acc[10], acc[11]);
        a_162[1] = mul_f32x2_noftz(previous_164, corr);
        float2 previous_165 = make_float2(acc[12], acc[13]);
        a_162[2] = mul_f32x2_noftz(previous_165, corr);
        float2 previous_166 = make_float2(acc[14], acc[15]);
        a_162[3] = mul_f32x2_noftz(previous_166, corr);
        float2 weight_167 = make_float2(weights[0], weights[0]);
        float2 v_168 = make_float2(fsrc[8], fsrc[9]);
        a_162[0] = fma_f32x2_rn_noftz(weight_167, v_168, a_162[0]);
        float2 v_169 = make_float2(fsrc[10], fsrc[11]);
        a_162[1] = fma_f32x2_rn_noftz(weight_167, v_169, a_162[1]);
        float2 v_170 = make_float2(fsrc[12], fsrc[13]);
        a_162[2] = fma_f32x2_rn_noftz(weight_167, v_170, a_162[2]);
        float2 v_171 = make_float2(fsrc[14], fsrc[15]);
        a_162[3] = fma_f32x2_rn_noftz(weight_167, v_171, a_162[3]);
        float2 weight_172 = make_float2(weights[1], weights[1]);
        float2 v_173 = make_float2(fsrc[36], fsrc[37]);
        a_162[0] = fma_f32x2_rn_noftz(weight_172, v_173, a_162[0]);
        float2 v_174 = make_float2(fsrc[38], fsrc[39]);
        a_162[1] = fma_f32x2_rn_noftz(weight_172, v_174, a_162[1]);
        float2 v_175 = make_float2(fsrc[40], fsrc[41]);
        a_162[2] = fma_f32x2_rn_noftz(weight_172, v_175, a_162[2]);
        float2 v_176 = make_float2(fsrc[42], fsrc[43]);
        a_162[3] = fma_f32x2_rn_noftz(weight_172, v_176, a_162[3]);
        acc[8] = a_162[0].x;
        acc[9] = a_162[0].y;
        acc[10] = a_162[1].x;
        acc[11] = a_162[1].y;
        acc[12] = a_162[2].x;
        acc[13] = a_162[2].y;
        acc[14] = a_162[3].x;
        acc[15] = a_162[3].y;
        const int woff_177 = 8;
        int base_178 = (4 + group) * 1024 + thread * 8;
        float2 a_179[4];
        float2 previous_180 = make_float2(acc[16], acc[17]);
        a_179[0] = mul_f32x2_noftz(previous_180, corr);
        float2 previous_181 = make_float2(acc[18], acc[19]);
        a_179[1] = mul_f32x2_noftz(previous_181, corr);
        float2 previous_182 = make_float2(acc[20], acc[21]);
        a_179[2] = mul_f32x2_noftz(previous_182, corr);
        float2 previous_183 = make_float2(acc[22], acc[23]);
        a_179[3] = mul_f32x2_noftz(previous_183, corr);
        float2 weight_184 = make_float2(weights[0], weights[0]);
        float2 v_185 = make_float2(fsrc[16], fsrc[17]);
        a_179[0] = fma_f32x2_rn_noftz(weight_184, v_185, a_179[0]);
        float2 v_186 = make_float2(fsrc[18], fsrc[19]);
        a_179[1] = fma_f32x2_rn_noftz(weight_184, v_186, a_179[1]);
        float2 v_187 = make_float2(fsrc[20], fsrc[21]);
        a_179[2] = fma_f32x2_rn_noftz(weight_184, v_187, a_179[2]);
        float2 v_188 = make_float2(fsrc[22], fsrc[23]);
        a_179[3] = fma_f32x2_rn_noftz(weight_184, v_188, a_179[3]);
        float2 weight_189 = make_float2(weights[1], weights[1]);
        float2 v_190 = make_float2(fsrc[44], fsrc[45]);
        a_179[0] = fma_f32x2_rn_noftz(weight_189, v_190, a_179[0]);
        float2 v_191 = make_float2(fsrc[46], fsrc[47]);
        a_179[1] = fma_f32x2_rn_noftz(weight_189, v_191, a_179[1]);
        float2 v_192 = make_float2(fsrc[48], fsrc[49]);
        a_179[2] = fma_f32x2_rn_noftz(weight_189, v_192, a_179[2]);
        float2 v_193 = make_float2(fsrc[50], fsrc[51]);
        a_179[3] = fma_f32x2_rn_noftz(weight_189, v_193, a_179[3]);
        acc[16] = a_179[0].x;
        acc[17] = a_179[0].y;
        acc[18] = a_179[1].x;
        acc[19] = a_179[1].y;
        acc[20] = a_179[2].x;
        acc[21] = a_179[2].y;
        acc[22] = a_179[3].x;
        acc[23] = a_179[3].y;
        const int woff_194 = 12;
        int base_195 = 6144 + group * 512 + thread * 4;
        float2 a_196[4];
        float2 previous_197 = make_float2(acc[24], acc[25]);
        a_196[0] = mul_f32x2_noftz(previous_197, corr);
        float2 previous_198 = make_float2(acc[26], acc[27]);
        a_196[1] = mul_f32x2_noftz(previous_198, corr);
        float2 weight_199 = make_float2(weights[0], weights[0]);
        float2 v_200 = make_float2(fsrc[24], fsrc[25]);
        a_196[0] = fma_f32x2_rn_noftz(weight_199, v_200, a_196[0]);
        float2 v_201 = make_float2(fsrc[26], fsrc[27]);
        a_196[1] = fma_f32x2_rn_noftz(weight_199, v_201, a_196[1]);
        float2 weight_202 = make_float2(weights[1], weights[1]);
        float2 v_203 = make_float2(fsrc[52], fsrc[53]);
        a_196[0] = fma_f32x2_rn_noftz(weight_202, v_203, a_196[0]);
        float2 v_204 = make_float2(fsrc[54], fsrc[55]);
        a_196[1] = fma_f32x2_rn_noftz(weight_202, v_204, a_196[1]);
        acc[24] = a_196[0].x;
        acc[25] = a_196[0].y;
        acc[26] = a_196[1].x;
        acc[27] = a_196[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_205 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_205, v_205, output_sq_pair);
        float2 v_206 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_206, v_206, output_sq_pair);
        float2 v_207 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_207, v_207, output_sq_pair);
        float2 v_208 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_208, v_208, output_sq_pair);
        float2 v_209 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_209, v_209, output_sq_pair);
        float2 v_210 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_210, v_210, output_sq_pair);
        float2 v_211 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_211, v_211, output_sq_pair);
        float2 v_212 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_212, v_212, output_sq_pair);
        float2 v_213 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_213, v_213, output_sq_pair);
        float2 v_214 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_214, v_214, output_sq_pair);
        float2 v_215 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_215, v_215, output_sq_pair);
        float2 v_216 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_216, v_216, output_sq_pair);
        float2 v_217 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_217, v_217, output_sq_pair);
        float2 v_218 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_218, v_218, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_41;
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_42;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_44;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_45;
        if (lane == 0) {
            out_stats[warp_0] = output_sq;
        }
        __syncthreads();
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_46 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_46;
        float _shfl_down_sync_47 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_47;
        float _shfl_down_sync_48 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_48;
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_49 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_49;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_219 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_50 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_50), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_51 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_51));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_220 = 2;
        const int acc_idx_221 = value_idx_220;
        float2 _f32x2_mul_a_52 = make_float2(acc[acc_idx_221], acc[acc_idx_221 + 1]);
        float2 scaled_pair_222;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_222) : "l"(*(const unsigned long long*)&_f32x2_mul_a_52), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_53 = make_float2(wout[acc_idx_221], wout[acc_idx_221 + 1]);
        float2 normalized_pair_223;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_223) : "l"(*(const unsigned long long*)&scaled_pair_222), "l"(*(const unsigned long long*)&_f32x2_mul_b_53));
        output_values[value_idx_220] = normalized_pair_223.x;
        output_values[value_idx_220 + 1] = normalized_pair_223.y;
        const int value_idx_224 = 4;
        const int acc_idx_225 = value_idx_224;
        float2 _f32x2_mul_a_54 = make_float2(acc[acc_idx_225], acc[acc_idx_225 + 1]);
        float2 scaled_pair_226;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_226) : "l"(*(const unsigned long long*)&_f32x2_mul_a_54), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_55 = make_float2(wout[acc_idx_225], wout[acc_idx_225 + 1]);
        float2 normalized_pair_227;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_227) : "l"(*(const unsigned long long*)&scaled_pair_226), "l"(*(const unsigned long long*)&_f32x2_mul_b_55));
        output_values[value_idx_224] = normalized_pair_227.x;
        output_values[value_idx_224 + 1] = normalized_pair_227.y;
        const int value_idx_228 = 6;
        const int acc_idx_229 = value_idx_228;
        float2 _f32x2_mul_a_56 = make_float2(acc[acc_idx_229], acc[acc_idx_229 + 1]);
        float2 scaled_pair_230;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_230) : "l"(*(const unsigned long long*)&_f32x2_mul_a_56), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_57 = make_float2(wout[acc_idx_229], wout[acc_idx_229 + 1]);
        float2 normalized_pair_231;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_231) : "l"(*(const unsigned long long*)&scaled_pair_230), "l"(*(const unsigned long long*)&_f32x2_mul_b_57));
        output_values[value_idx_228] = normalized_pair_231.x;
        output_values[value_idx_228 + 1] = normalized_pair_231.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_219))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_232 = (2 + group) * 1024 + thread * 8;
        float output_values_233[8];
        const int value_idx_234 = 0;
        const int acc_idx_235 = 8 + value_idx_234;
        float2 _f32x2_mul_a_58 = make_float2(acc[acc_idx_235], acc[acc_idx_235 + 1]);
        float2 scaled_pair_236;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_236) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_59 = make_float2(wout[acc_idx_235], wout[acc_idx_235 + 1]);
        float2 normalized_pair_237;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_237) : "l"(*(const unsigned long long*)&scaled_pair_236), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
        output_values_233[value_idx_234] = normalized_pair_237.x;
        output_values_233[value_idx_234 + 1] = normalized_pair_237.y;
        const int value_idx_238 = 2;
        const int acc_idx_239 = 8 + value_idx_238;
        float2 _f32x2_mul_a_60 = make_float2(acc[acc_idx_239], acc[acc_idx_239 + 1]);
        float2 scaled_pair_240;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_240) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_61 = make_float2(wout[acc_idx_239], wout[acc_idx_239 + 1]);
        float2 normalized_pair_241;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_241) : "l"(*(const unsigned long long*)&scaled_pair_240), "l"(*(const unsigned long long*)&_f32x2_mul_b_61));
        output_values_233[value_idx_238] = normalized_pair_241.x;
        output_values_233[value_idx_238 + 1] = normalized_pair_241.y;
        const int value_idx_242 = 4;
        const int acc_idx_243 = 8 + value_idx_242;
        float2 _f32x2_mul_a_62 = make_float2(acc[acc_idx_243], acc[acc_idx_243 + 1]);
        float2 scaled_pair_244;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_244) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_63 = make_float2(wout[acc_idx_243], wout[acc_idx_243 + 1]);
        float2 normalized_pair_245;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_245) : "l"(*(const unsigned long long*)&scaled_pair_244), "l"(*(const unsigned long long*)&_f32x2_mul_b_63));
        output_values_233[value_idx_242] = normalized_pair_245.x;
        output_values_233[value_idx_242 + 1] = normalized_pair_245.y;
        const int value_idx_246 = 6;
        const int acc_idx_247 = 8 + value_idx_246;
        float2 _f32x2_mul_a_64 = make_float2(acc[acc_idx_247], acc[acc_idx_247 + 1]);
        float2 scaled_pair_248;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_248) : "l"(*(const unsigned long long*)&_f32x2_mul_a_64), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_65 = make_float2(wout[acc_idx_247], wout[acc_idx_247 + 1]);
        float2 normalized_pair_249;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_249) : "l"(*(const unsigned long long*)&scaled_pair_248), "l"(*(const unsigned long long*)&_f32x2_mul_b_65));
        output_values_233[value_idx_246] = normalized_pair_249.x;
        output_values_233[value_idx_246 + 1] = normalized_pair_249.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_233[_lp*2 + 0], output_values_233[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_233[_lp*2 + 2], output_values_233[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_233[_lp*2 + 4], output_values_233[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_233[_lp*2 + 6], output_values_233[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_232))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_250 = (4 + group) * 1024 + thread * 8;
        float output_values_251[8];
        const int value_idx_252 = 0;
        const int acc_idx_253 = 16 + value_idx_252;
        float2 _f32x2_mul_a_66 = make_float2(acc[acc_idx_253], acc[acc_idx_253 + 1]);
        float2 scaled_pair_254;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_254) : "l"(*(const unsigned long long*)&_f32x2_mul_a_66), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_67 = make_float2(wout[acc_idx_253], wout[acc_idx_253 + 1]);
        float2 normalized_pair_255;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_255) : "l"(*(const unsigned long long*)&scaled_pair_254), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
        output_values_251[value_idx_252] = normalized_pair_255.x;
        output_values_251[value_idx_252 + 1] = normalized_pair_255.y;
        const int value_idx_256 = 2;
        const int acc_idx_257 = 16 + value_idx_256;
        float2 _f32x2_mul_a_68 = make_float2(acc[acc_idx_257], acc[acc_idx_257 + 1]);
        float2 scaled_pair_258;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_258) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_69 = make_float2(wout[acc_idx_257], wout[acc_idx_257 + 1]);
        float2 normalized_pair_259;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_259) : "l"(*(const unsigned long long*)&scaled_pair_258), "l"(*(const unsigned long long*)&_f32x2_mul_b_69));
        output_values_251[value_idx_256] = normalized_pair_259.x;
        output_values_251[value_idx_256 + 1] = normalized_pair_259.y;
        const int value_idx_260 = 4;
        const int acc_idx_261 = 16 + value_idx_260;
        float2 _f32x2_mul_a_70 = make_float2(acc[acc_idx_261], acc[acc_idx_261 + 1]);
        float2 scaled_pair_262;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_262) : "l"(*(const unsigned long long*)&_f32x2_mul_a_70), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_71 = make_float2(wout[acc_idx_261], wout[acc_idx_261 + 1]);
        float2 normalized_pair_263;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_263) : "l"(*(const unsigned long long*)&scaled_pair_262), "l"(*(const unsigned long long*)&_f32x2_mul_b_71));
        output_values_251[value_idx_260] = normalized_pair_263.x;
        output_values_251[value_idx_260 + 1] = normalized_pair_263.y;
        const int value_idx_264 = 6;
        const int acc_idx_265 = 16 + value_idx_264;
        float2 _f32x2_mul_a_72 = make_float2(acc[acc_idx_265], acc[acc_idx_265 + 1]);
        float2 scaled_pair_266;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_266) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_73 = make_float2(wout[acc_idx_265], wout[acc_idx_265 + 1]);
        float2 normalized_pair_267;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_267) : "l"(*(const unsigned long long*)&scaled_pair_266), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
        output_values_251[value_idx_264] = normalized_pair_267.x;
        output_values_251[value_idx_264 + 1] = normalized_pair_267.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_251[_lp*2 + 0], output_values_251[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_251[_lp*2 + 2], output_values_251[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_251[_lp*2 + 4], output_values_251[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_251[_lp*2 + 6], output_values_251[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_250))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_268 = 6144 + group * 512 + thread * 4;
        float output_values_269[8];
        const int value_idx_270 = 0;
        const int acc_idx_271 = 24 + value_idx_270;
        float2 _f32x2_mul_a_74 = make_float2(acc[acc_idx_271], acc[acc_idx_271 + 1]);
        float2 scaled_pair_272;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_272) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_75 = make_float2(wout[acc_idx_271], wout[acc_idx_271 + 1]);
        float2 normalized_pair_273;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_273) : "l"(*(const unsigned long long*)&scaled_pair_272), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
        output_values_269[value_idx_270] = normalized_pair_273.x;
        output_values_269[value_idx_270 + 1] = normalized_pair_273.y;
        const int value_idx_274 = 2;
        const int acc_idx_275 = 24 + value_idx_274;
        float2 _f32x2_mul_a_76 = make_float2(acc[acc_idx_275], acc[acc_idx_275 + 1]);
        float2 scaled_pair_276;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_276) : "l"(*(const unsigned long long*)&_f32x2_mul_a_76), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_77 = make_float2(wout[acc_idx_275], wout[acc_idx_275 + 1]);
        float2 normalized_pair_277;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_277) : "l"(*(const unsigned long long*)&scaled_pair_276), "l"(*(const unsigned long long*)&_f32x2_mul_b_77));
        output_values_269[value_idx_274] = normalized_pair_277.x;
        output_values_269[value_idx_274 + 1] = normalized_pair_277.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_269[_lp*2 + 0], output_values_269[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_269[_lp*2 + 2], output_values_269[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_268))[0] = reinterpret_cast<int2*>(_packed_3)[0];
    }
}

} // extern "C"
