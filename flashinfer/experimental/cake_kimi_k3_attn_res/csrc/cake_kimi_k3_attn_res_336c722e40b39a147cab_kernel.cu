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
kernel_cake_kimi_k3_attn_res_336c722e40b39a147cab(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
        float2 _f2_0 = make_float2(0.0f, 0.0f);
        sq[0] = _f2_0;
        float2 _f2_1 = make_float2(0.0f, 0.0f);
        dot[0] = _f2_1;
        float2 _f2_2 = make_float2(0.0f, 0.0f);
        sq[1] = _f2_2;
        float2 _f2_3 = make_float2(0.0f, 0.0f);
        dot[1] = _f2_3;
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
        float2 _f2_4 = make_float2(_unpack_bf16_0[0], _unpack_bf16_0[1]);
        float2 v = _f2_4;
        float2 _f2_5 = make_float2(q[0], q[1]);
        float2 qp = _f2_5;
        sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
        float2 _f2_6 = make_float2(_unpack_bf16_0[2], _unpack_bf16_0[3]);
        float2 v_8 = _f2_6;
        float2 _f2_7 = make_float2(q[2], q[3]);
        float2 qp_9 = _f2_7;
        sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_8, qp_9, dot[0]);
        float2 _f2_8 = make_float2(_unpack_bf16_0[4], _unpack_bf16_0[5]);
        float2 v_10 = _f2_8;
        float2 _f2_9 = make_float2(q[4], q[5]);
        float2 qp_11 = _f2_9;
        sq[0] = fma_f32x2_rn_noftz(v_10, v_10, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_10, qp_11, dot[0]);
        float2 _f2_10 = make_float2(_unpack_bf16_0[6], _unpack_bf16_0[7]);
        float2 v_12 = _f2_10;
        float2 _f2_11 = make_float2(q[6], q[7]);
        float2 qp_13 = _f2_11;
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
        unsigned long long write_base = block_base + blocks_k_stride + (unsigned long long)base_6;
        {
            int4 _iv4 = make_int4(sw_14[0 + 0], sw_14[0 + 1], sw_14[0 + 2], sw_14[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base) + 0) = _iv4;
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
        float2 _f2_12 = make_float2(_unpack_bf16_1[0], _unpack_bf16_1[1]);
        float2 v_24 = _f2_12;
        float2 _f2_13 = make_float2(q[0], q[1]);
        float2 qp_25 = _f2_13;
        sq[1] = fma_f32x2_rn_noftz(v_24, v_24, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_24, qp_25, dot[1]);
        float2 _f2_14 = make_float2(_unpack_bf16_1[2], _unpack_bf16_1[3]);
        float2 v_26 = _f2_14;
        float2 _f2_15 = make_float2(q[2], q[3]);
        float2 qp_27 = _f2_15;
        sq[1] = fma_f32x2_rn_noftz(v_26, v_26, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_26, qp_27, dot[1]);
        float2 _f2_16 = make_float2(_unpack_bf16_1[4], _unpack_bf16_1[5]);
        float2 v_28 = _f2_16;
        float2 _f2_17 = make_float2(q[4], q[5]);
        float2 qp_29 = _f2_17;
        sq[1] = fma_f32x2_rn_noftz(v_28, v_28, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_28, qp_29, dot[1]);
        float2 _f2_18 = make_float2(_unpack_bf16_1[6], _unpack_bf16_1[7]);
        float2 v_30 = _f2_18;
        float2 _f2_19 = make_float2(q[6], q[7]);
        float2 qp_31 = _f2_19;
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
        float2 _f2_20 = make_float2(_unpack_bf16_2[0], _unpack_bf16_2[1]);
        float2 v_35 = _f2_20;
        float2 _f2_21 = make_float2(q[8], q[9]);
        float2 qp_36 = _f2_21;
        sq[0] = fma_f32x2_rn_noftz(v_35, v_35, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_35, qp_36, dot[0]);
        float2 _f2_22 = make_float2(_unpack_bf16_2[2], _unpack_bf16_2[3]);
        float2 v_37 = _f2_22;
        float2 _f2_23 = make_float2(q[10], q[11]);
        float2 qp_38 = _f2_23;
        sq[0] = fma_f32x2_rn_noftz(v_37, v_37, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_37, qp_38, dot[0]);
        float2 _f2_24 = make_float2(_unpack_bf16_2[4], _unpack_bf16_2[5]);
        float2 v_39 = _f2_24;
        float2 _f2_25 = make_float2(q[12], q[13]);
        float2 qp_40 = _f2_25;
        sq[0] = fma_f32x2_rn_noftz(v_39, v_39, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_39, qp_40, dot[0]);
        float2 _f2_26 = make_float2(_unpack_bf16_2[6], _unpack_bf16_2[7]);
        float2 v_41 = _f2_26;
        float2 _f2_27 = make_float2(q[14], q[15]);
        float2 qp_42 = _f2_27;
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
        unsigned long long write_base_56 = block_base + blocks_k_stride + (unsigned long long)base_32;
        {
            int4 _iv4 = make_int4(sw_43[0 + 0], sw_43[0 + 1], sw_43[0 + 2], sw_43[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_56) + 0) = _iv4;
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
        float2 _f2_28 = make_float2(_unpack_bf16_3[0], _unpack_bf16_3[1]);
        float2 v_57 = _f2_28;
        float2 _f2_29 = make_float2(q[8], q[9]);
        float2 qp_58 = _f2_29;
        sq[1] = fma_f32x2_rn_noftz(v_57, v_57, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_57, qp_58, dot[1]);
        float2 _f2_30 = make_float2(_unpack_bf16_3[2], _unpack_bf16_3[3]);
        float2 v_59 = _f2_30;
        float2 _f2_31 = make_float2(q[10], q[11]);
        float2 qp_60 = _f2_31;
        sq[1] = fma_f32x2_rn_noftz(v_59, v_59, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_59, qp_60, dot[1]);
        float2 _f2_32 = make_float2(_unpack_bf16_3[4], _unpack_bf16_3[5]);
        float2 v_61 = _f2_32;
        float2 _f2_33 = make_float2(q[12], q[13]);
        float2 qp_62 = _f2_33;
        sq[1] = fma_f32x2_rn_noftz(v_61, v_61, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_61, qp_62, dot[1]);
        float2 _f2_34 = make_float2(_unpack_bf16_3[6], _unpack_bf16_3[7]);
        float2 v_63 = _f2_34;
        float2 _f2_35 = make_float2(q[14], q[15]);
        float2 qp_64 = _f2_35;
        sq[1] = fma_f32x2_rn_noftz(v_63, v_63, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_63, qp_64, dot[1]);
        int base_65 = (4 + group) * 1024 + thread * 8;
        const int woff_66 = 8;
        unsigned int sw_67[4];
        sw_67[0] = words[woff_66];
        sw_67[1] = words[woff_66 + 1];
        sw_67[2] = words[woff_66 + 2];
        sw_67[3] = words[woff_66 + 3];
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                : "r"(sw_67[_pair]));
        }
        fsrc[16] = _unpack_bf16_4[0];
        fsrc[17] = _unpack_bf16_4[1];
        fsrc[18] = _unpack_bf16_4[2];
        fsrc[19] = _unpack_bf16_4[3];
        fsrc[20] = _unpack_bf16_4[4];
        fsrc[21] = _unpack_bf16_4[5];
        fsrc[22] = _unpack_bf16_4[6];
        fsrc[23] = _unpack_bf16_4[7];
        float2 _f2_36 = make_float2(_unpack_bf16_4[0], _unpack_bf16_4[1]);
        float2 v_68 = _f2_36;
        float2 _f2_37 = make_float2(q[16], q[17]);
        float2 qp_69 = _f2_37;
        sq[0] = fma_f32x2_rn_noftz(v_68, v_68, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_68, qp_69, dot[0]);
        float2 _f2_38 = make_float2(_unpack_bf16_4[2], _unpack_bf16_4[3]);
        float2 v_70 = _f2_38;
        float2 _f2_39 = make_float2(q[18], q[19]);
        float2 qp_71 = _f2_39;
        sq[0] = fma_f32x2_rn_noftz(v_70, v_70, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_70, qp_71, dot[0]);
        float2 _f2_40 = make_float2(_unpack_bf16_4[4], _unpack_bf16_4[5]);
        float2 v_72 = _f2_40;
        float2 _f2_41 = make_float2(q[20], q[21]);
        float2 qp_73 = _f2_41;
        sq[0] = fma_f32x2_rn_noftz(v_72, v_72, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_72, qp_73, dot[0]);
        float2 _f2_42 = make_float2(_unpack_bf16_4[6], _unpack_bf16_4[7]);
        float2 v_74 = _f2_42;
        float2 _f2_43 = make_float2(q[22], q[23]);
        float2 qp_75 = _f2_43;
        sq[0] = fma_f32x2_rn_noftz(v_74, v_74, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_74, qp_75, dot[0]);
        unsigned int sw_76[4];
        sw_76[0] = words[14 + woff_66];
        sw_76[1] = words[14 + woff_66 + 1];
        sw_76[2] = words[14 + woff_66 + 2];
        sw_76[3] = words[14 + woff_66 + 3];
        __nv_bfloat162 a_77 = __as_bf16x2(sw_76[0]);
        __nv_bfloat162 d_78 = __as_bf16x2(dwords[woff_66]);
        __nv_bfloat162 mixed_79 = a_77 + d_78;
        sw_76[0] = __as_u32(mixed_79);
        words[14 + woff_66] = sw_76[0];
        __nv_bfloat162 a_80 = __as_bf16x2(sw_76[1]);
        __nv_bfloat162 d_81 = __as_bf16x2(dwords[woff_66 + 1]);
        __nv_bfloat162 mixed_82 = a_80 + d_81;
        sw_76[1] = __as_u32(mixed_82);
        words[14 + woff_66 + 1] = sw_76[1];
        __nv_bfloat162 a_83 = __as_bf16x2(sw_76[2]);
        __nv_bfloat162 d_84 = __as_bf16x2(dwords[woff_66 + 2]);
        __nv_bfloat162 mixed_85 = a_83 + d_84;
        sw_76[2] = __as_u32(mixed_85);
        words[14 + woff_66 + 2] = sw_76[2];
        __nv_bfloat162 a_86 = __as_bf16x2(sw_76[3]);
        __nv_bfloat162 d_87 = __as_bf16x2(dwords[woff_66 + 3]);
        __nv_bfloat162 mixed_88 = a_86 + d_87;
        sw_76[3] = __as_u32(mixed_88);
        words[14 + woff_66 + 3] = sw_76[3];
        {
            int4 _iv4 = make_int4(sw_76[0 + 0], sw_76[0 + 1], sw_76[0 + 2], sw_76[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_65)) + 0) = _iv4;
        }
        unsigned long long write_base_89 = block_base + blocks_k_stride + (unsigned long long)base_65;
        {
            int4 _iv4 = make_int4(sw_76[0 + 0], sw_76[0 + 1], sw_76[0 + 2], sw_76[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_89) + 0) = _iv4;
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
                : "r"(sw_76[_pair]));
        }
        fsrc[44] = _unpack_bf16_5[0];
        fsrc[45] = _unpack_bf16_5[1];
        fsrc[46] = _unpack_bf16_5[2];
        fsrc[47] = _unpack_bf16_5[3];
        fsrc[48] = _unpack_bf16_5[4];
        fsrc[49] = _unpack_bf16_5[5];
        fsrc[50] = _unpack_bf16_5[6];
        fsrc[51] = _unpack_bf16_5[7];
        float2 _f2_44 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 v_90 = _f2_44;
        float2 _f2_45 = make_float2(q[16], q[17]);
        float2 qp_91 = _f2_45;
        sq[1] = fma_f32x2_rn_noftz(v_90, v_90, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_90, qp_91, dot[1]);
        float2 _f2_46 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 v_92 = _f2_46;
        float2 _f2_47 = make_float2(q[18], q[19]);
        float2 qp_93 = _f2_47;
        sq[1] = fma_f32x2_rn_noftz(v_92, v_92, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_92, qp_93, dot[1]);
        float2 _f2_48 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 v_94 = _f2_48;
        float2 _f2_49 = make_float2(q[20], q[21]);
        float2 qp_95 = _f2_49;
        sq[1] = fma_f32x2_rn_noftz(v_94, v_94, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_94, qp_95, dot[1]);
        float2 _f2_50 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 v_96 = _f2_50;
        float2 _f2_51 = make_float2(q[22], q[23]);
        float2 qp_97 = _f2_51;
        sq[1] = fma_f32x2_rn_noftz(v_96, v_96, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_96, qp_97, dot[1]);
        int base_98 = 6144 + group * 512 + thread * 4;
        const int woff_99 = 12;
        unsigned int sw_100[4];
        sw_100[0] = words[woff_99];
        sw_100[1] = words[woff_99 + 1];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(sw_100[_pair]));
        }
        fsrc[24] = _unpack_bf16_6[0];
        fsrc[25] = _unpack_bf16_6[1];
        fsrc[26] = _unpack_bf16_6[2];
        fsrc[27] = _unpack_bf16_6[3];
        float2 _f2_52 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 v_101 = _f2_52;
        sq[0] = fma_f32x2_rn_noftz(v_101, v_101, sq[0]);
        float2 _f2_53 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 v_102 = _f2_53;
        sq[0] = fma_f32x2_rn_noftz(v_102, v_102, sq[0]);
        float2 _f2_54 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 v_103 = _f2_54;
        float2 _f2_55 = make_float2(q[24], q[25]);
        float2 qp_104 = _f2_55;
        dot[0] = fma_f32x2_rn_noftz(v_103, qp_104, dot[0]);
        float2 _f2_56 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 v_105 = _f2_56;
        float2 _f2_57 = make_float2(q[26], q[27]);
        float2 qp_106 = _f2_57;
        dot[0] = fma_f32x2_rn_noftz(v_105, qp_106, dot[0]);
        unsigned int sw_107[4];
        sw_107[0] = words[14 + woff_99];
        sw_107[1] = words[14 + woff_99 + 1];
        __nv_bfloat162 a_108 = __as_bf16x2(sw_107[0]);
        __nv_bfloat162 d_109 = __as_bf16x2(dwords[woff_99]);
        __nv_bfloat162 mixed_110 = a_108 + d_109;
        sw_107[0] = __as_u32(mixed_110);
        words[14 + woff_99] = sw_107[0];
        __nv_bfloat162 a_111 = __as_bf16x2(sw_107[1]);
        __nv_bfloat162 d_112 = __as_bf16x2(dwords[woff_99 + 1]);
        __nv_bfloat162 mixed_113 = a_111 + d_112;
        sw_107[1] = __as_u32(mixed_113);
        words[14 + woff_99 + 1] = sw_107[1];
        {
            int2 _iv2 = make_int2(sw_107[0 + 0], sw_107[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_98)) + 0) = _iv2;
        }
        unsigned long long write_base_114 = block_base + blocks_k_stride + (unsigned long long)base_98;
        {
            int2 _iv2 = make_int2(sw_107[0 + 0], sw_107[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(blocks + write_base_114) + 0) = _iv2;
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
                : "r"(sw_107[_pair]));
        }
        fsrc[52] = _unpack_bf16_7[0];
        fsrc[53] = _unpack_bf16_7[1];
        fsrc[54] = _unpack_bf16_7[2];
        fsrc[55] = _unpack_bf16_7[3];
        float2 _f2_58 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 v_115 = _f2_58;
        sq[1] = fma_f32x2_rn_noftz(v_115, v_115, sq[1]);
        float2 _f2_59 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 v_116 = _f2_59;
        sq[1] = fma_f32x2_rn_noftz(v_116, v_116, sq[1]);
        float2 _f2_60 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 v_117 = _f2_60;
        float2 _f2_61 = make_float2(q[24], q[25]);
        float2 qp_118 = _f2_61;
        dot[1] = fma_f32x2_rn_noftz(v_117, qp_118, dot[1]);
        float2 _f2_62 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 v_119 = _f2_62;
        float2 _f2_63 = make_float2(q[26], q[27]);
        float2 qp_120 = _f2_63;
        dot[1] = fma_f32x2_rn_noftz(v_119, qp_120, dot[1]);
        float2 pairs[2];
        float2 _f2_64 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_64;
        float2 _f2_65 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_65;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_66 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_66;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_121 = 0;
        bits_121 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_121, 16);
        unsigned long long peerbits_122 = _shfl_xor_1;
        float2 _f2_67 = make_float2(0.0f, 0.0f);
        float2 peer_123 = _f2_67;
        peer_123 = reinterpret_cast<float2*>(&peerbits_122)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_123);
        unsigned long long bits_124 = 0;
        bits_124 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_124, 8);
        unsigned long long peerbits_125 = _shfl_xor_2;
        float2 _f2_68 = make_float2(0.0f, 0.0f);
        float2 peer_126 = _f2_68;
        peer_126 = reinterpret_cast<float2*>(&peerbits_125)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_126);
        unsigned long long bits_127 = 0;
        bits_127 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_127, 8);
        unsigned long long peerbits_128 = _shfl_xor_3;
        float2 _f2_69 = make_float2(0.0f, 0.0f);
        float2 peer_129 = _f2_69;
        peer_129 = reinterpret_cast<float2*>(&peerbits_128)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_129);
        unsigned long long bits_130 = 0;
        bits_130 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_130, 4);
        unsigned long long peerbits_131 = _shfl_xor_4;
        float2 _f2_70 = make_float2(0.0f, 0.0f);
        float2 peer_132 = _f2_70;
        peer_132 = reinterpret_cast<float2*>(&peerbits_131)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_132);
        unsigned long long bits_133 = 0;
        bits_133 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_133, 4);
        unsigned long long peerbits_134 = _shfl_xor_5;
        float2 _f2_71 = make_float2(0.0f, 0.0f);
        float2 peer_135 = _f2_71;
        peer_135 = reinterpret_cast<float2*>(&peerbits_134)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_135);
        unsigned long long bits_136 = 0;
        bits_136 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_136, 2);
        unsigned long long peerbits_137 = _shfl_xor_6;
        float2 _f2_72 = make_float2(0.0f, 0.0f);
        float2 peer_138 = _f2_72;
        peer_138 = reinterpret_cast<float2*>(&peerbits_137)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_138);
        unsigned long long bits_139 = 0;
        bits_139 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_139, 2);
        unsigned long long peerbits_140 = _shfl_xor_7;
        float2 _f2_73 = make_float2(0.0f, 0.0f);
        float2 peer_141 = _f2_73;
        peer_141 = reinterpret_cast<float2*>(&peerbits_140)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_141);
        unsigned long long bits_142 = 0;
        bits_142 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_142, 1);
        unsigned long long peerbits_143 = _shfl_xor_8;
        float2 _f2_74 = make_float2(0.0f, 0.0f);
        float2 peer_144 = _f2_74;
        peer_144 = reinterpret_cast<float2*>(&peerbits_143)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_144);
        unsigned long long bits_145 = 0;
        bits_145 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_145, 1);
        unsigned long long peerbits_146 = _shfl_xor_9;
        float2 _f2_75 = make_float2(0.0f, 0.0f);
        float2 peer_147 = _f2_75;
        peer_147 = reinterpret_cast<float2*>(&peerbits_146)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_147);
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
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_0;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, logit, 8);
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
        float2 _f2_76 = make_float2(correction, correction);
        float2 corr = _f2_76;
        const int woff_148 = 0;
        int base_149 = group * 1024 + thread * 8;
        float2 a_150[4];
        float2 _f2_77 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_77;
        a_150[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_78 = make_float2(acc[2], acc[3]);
        float2 previous_151 = _f2_78;
        a_150[1] = mul_f32x2_noftz(previous_151, corr);
        float2 _f2_79 = make_float2(acc[4], acc[5]);
        float2 previous_152 = _f2_79;
        a_150[2] = mul_f32x2_noftz(previous_152, corr);
        float2 _f2_80 = make_float2(acc[6], acc[7]);
        float2 previous_153 = _f2_80;
        a_150[3] = mul_f32x2_noftz(previous_153, corr);
        float2 _f2_81 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_81;
        float2 _f2_82 = make_float2(fsrc[0], fsrc[1]);
        float2 v_154 = _f2_82;
        a_150[0] = fma_f32x2_rn_noftz(weight, v_154, a_150[0]);
        float2 _f2_83 = make_float2(fsrc[2], fsrc[3]);
        float2 v_155 = _f2_83;
        a_150[1] = fma_f32x2_rn_noftz(weight, v_155, a_150[1]);
        float2 _f2_84 = make_float2(fsrc[4], fsrc[5]);
        float2 v_156 = _f2_84;
        a_150[2] = fma_f32x2_rn_noftz(weight, v_156, a_150[2]);
        float2 _f2_85 = make_float2(fsrc[6], fsrc[7]);
        float2 v_157 = _f2_85;
        a_150[3] = fma_f32x2_rn_noftz(weight, v_157, a_150[3]);
        float2 _f2_86 = make_float2(weights[1], weights[1]);
        float2 weight_158 = _f2_86;
        float2 _f2_87 = make_float2(fsrc[28], fsrc[29]);
        float2 v_159 = _f2_87;
        a_150[0] = fma_f32x2_rn_noftz(weight_158, v_159, a_150[0]);
        float2 _f2_88 = make_float2(fsrc[30], fsrc[31]);
        float2 v_160 = _f2_88;
        a_150[1] = fma_f32x2_rn_noftz(weight_158, v_160, a_150[1]);
        float2 _f2_89 = make_float2(fsrc[32], fsrc[33]);
        float2 v_161 = _f2_89;
        a_150[2] = fma_f32x2_rn_noftz(weight_158, v_161, a_150[2]);
        float2 _f2_90 = make_float2(fsrc[34], fsrc[35]);
        float2 v_162 = _f2_90;
        a_150[3] = fma_f32x2_rn_noftz(weight_158, v_162, a_150[3]);
        acc[0] = a_150[0].x;
        acc[1] = a_150[0].y;
        acc[2] = a_150[1].x;
        acc[3] = a_150[1].y;
        acc[4] = a_150[2].x;
        acc[5] = a_150[2].y;
        acc[6] = a_150[3].x;
        acc[7] = a_150[3].y;
        const int woff_163 = 4;
        int base_164 = (2 + group) * 1024 + thread * 8;
        float2 a_165[4];
        float2 _f2_91 = make_float2(acc[8], acc[9]);
        float2 previous_166 = _f2_91;
        a_165[0] = mul_f32x2_noftz(previous_166, corr);
        float2 _f2_92 = make_float2(acc[10], acc[11]);
        float2 previous_167 = _f2_92;
        a_165[1] = mul_f32x2_noftz(previous_167, corr);
        float2 _f2_93 = make_float2(acc[12], acc[13]);
        float2 previous_168 = _f2_93;
        a_165[2] = mul_f32x2_noftz(previous_168, corr);
        float2 _f2_94 = make_float2(acc[14], acc[15]);
        float2 previous_169 = _f2_94;
        a_165[3] = mul_f32x2_noftz(previous_169, corr);
        float2 _f2_95 = make_float2(weights[0], weights[0]);
        float2 weight_170 = _f2_95;
        float2 _f2_96 = make_float2(fsrc[8], fsrc[9]);
        float2 v_171 = _f2_96;
        a_165[0] = fma_f32x2_rn_noftz(weight_170, v_171, a_165[0]);
        float2 _f2_97 = make_float2(fsrc[10], fsrc[11]);
        float2 v_172 = _f2_97;
        a_165[1] = fma_f32x2_rn_noftz(weight_170, v_172, a_165[1]);
        float2 _f2_98 = make_float2(fsrc[12], fsrc[13]);
        float2 v_173 = _f2_98;
        a_165[2] = fma_f32x2_rn_noftz(weight_170, v_173, a_165[2]);
        float2 _f2_99 = make_float2(fsrc[14], fsrc[15]);
        float2 v_174 = _f2_99;
        a_165[3] = fma_f32x2_rn_noftz(weight_170, v_174, a_165[3]);
        float2 _f2_100 = make_float2(weights[1], weights[1]);
        float2 weight_175 = _f2_100;
        float2 _f2_101 = make_float2(fsrc[36], fsrc[37]);
        float2 v_176 = _f2_101;
        a_165[0] = fma_f32x2_rn_noftz(weight_175, v_176, a_165[0]);
        float2 _f2_102 = make_float2(fsrc[38], fsrc[39]);
        float2 v_177 = _f2_102;
        a_165[1] = fma_f32x2_rn_noftz(weight_175, v_177, a_165[1]);
        float2 _f2_103 = make_float2(fsrc[40], fsrc[41]);
        float2 v_178 = _f2_103;
        a_165[2] = fma_f32x2_rn_noftz(weight_175, v_178, a_165[2]);
        float2 _f2_104 = make_float2(fsrc[42], fsrc[43]);
        float2 v_179 = _f2_104;
        a_165[3] = fma_f32x2_rn_noftz(weight_175, v_179, a_165[3]);
        acc[8] = a_165[0].x;
        acc[9] = a_165[0].y;
        acc[10] = a_165[1].x;
        acc[11] = a_165[1].y;
        acc[12] = a_165[2].x;
        acc[13] = a_165[2].y;
        acc[14] = a_165[3].x;
        acc[15] = a_165[3].y;
        const int woff_180 = 8;
        int base_181 = (4 + group) * 1024 + thread * 8;
        float2 a_182[4];
        float2 _f2_105 = make_float2(acc[16], acc[17]);
        float2 previous_183 = _f2_105;
        a_182[0] = mul_f32x2_noftz(previous_183, corr);
        float2 _f2_106 = make_float2(acc[18], acc[19]);
        float2 previous_184 = _f2_106;
        a_182[1] = mul_f32x2_noftz(previous_184, corr);
        float2 _f2_107 = make_float2(acc[20], acc[21]);
        float2 previous_185 = _f2_107;
        a_182[2] = mul_f32x2_noftz(previous_185, corr);
        float2 _f2_108 = make_float2(acc[22], acc[23]);
        float2 previous_186 = _f2_108;
        a_182[3] = mul_f32x2_noftz(previous_186, corr);
        float2 _f2_109 = make_float2(weights[0], weights[0]);
        float2 weight_187 = _f2_109;
        float2 _f2_110 = make_float2(fsrc[16], fsrc[17]);
        float2 v_188 = _f2_110;
        a_182[0] = fma_f32x2_rn_noftz(weight_187, v_188, a_182[0]);
        float2 _f2_111 = make_float2(fsrc[18], fsrc[19]);
        float2 v_189 = _f2_111;
        a_182[1] = fma_f32x2_rn_noftz(weight_187, v_189, a_182[1]);
        float2 _f2_112 = make_float2(fsrc[20], fsrc[21]);
        float2 v_190 = _f2_112;
        a_182[2] = fma_f32x2_rn_noftz(weight_187, v_190, a_182[2]);
        float2 _f2_113 = make_float2(fsrc[22], fsrc[23]);
        float2 v_191 = _f2_113;
        a_182[3] = fma_f32x2_rn_noftz(weight_187, v_191, a_182[3]);
        float2 _f2_114 = make_float2(weights[1], weights[1]);
        float2 weight_192 = _f2_114;
        float2 _f2_115 = make_float2(fsrc[44], fsrc[45]);
        float2 v_193 = _f2_115;
        a_182[0] = fma_f32x2_rn_noftz(weight_192, v_193, a_182[0]);
        float2 _f2_116 = make_float2(fsrc[46], fsrc[47]);
        float2 v_194 = _f2_116;
        a_182[1] = fma_f32x2_rn_noftz(weight_192, v_194, a_182[1]);
        float2 _f2_117 = make_float2(fsrc[48], fsrc[49]);
        float2 v_195 = _f2_117;
        a_182[2] = fma_f32x2_rn_noftz(weight_192, v_195, a_182[2]);
        float2 _f2_118 = make_float2(fsrc[50], fsrc[51]);
        float2 v_196 = _f2_118;
        a_182[3] = fma_f32x2_rn_noftz(weight_192, v_196, a_182[3]);
        acc[16] = a_182[0].x;
        acc[17] = a_182[0].y;
        acc[18] = a_182[1].x;
        acc[19] = a_182[1].y;
        acc[20] = a_182[2].x;
        acc[21] = a_182[2].y;
        acc[22] = a_182[3].x;
        acc[23] = a_182[3].y;
        const int woff_197 = 12;
        int base_198 = 6144 + group * 512 + thread * 4;
        float2 a_199[4];
        float2 _f2_119 = make_float2(acc[24], acc[25]);
        float2 previous_200 = _f2_119;
        a_199[0] = mul_f32x2_noftz(previous_200, corr);
        float2 _f2_120 = make_float2(acc[26], acc[27]);
        float2 previous_201 = _f2_120;
        a_199[1] = mul_f32x2_noftz(previous_201, corr);
        float2 _f2_121 = make_float2(weights[0], weights[0]);
        float2 weight_202 = _f2_121;
        float2 _f2_122 = make_float2(fsrc[24], fsrc[25]);
        float2 v_203 = _f2_122;
        a_199[0] = fma_f32x2_rn_noftz(weight_202, v_203, a_199[0]);
        float2 _f2_123 = make_float2(fsrc[26], fsrc[27]);
        float2 v_204 = _f2_123;
        a_199[1] = fma_f32x2_rn_noftz(weight_202, v_204, a_199[1]);
        float2 _f2_124 = make_float2(weights[1], weights[1]);
        float2 weight_205 = _f2_124;
        float2 _f2_125 = make_float2(fsrc[52], fsrc[53]);
        float2 v_206 = _f2_125;
        a_199[0] = fma_f32x2_rn_noftz(weight_205, v_206, a_199[0]);
        float2 _f2_126 = make_float2(fsrc[54], fsrc[55]);
        float2 v_207 = _f2_126;
        a_199[1] = fma_f32x2_rn_noftz(weight_205, v_207, a_199[1]);
        acc[24] = a_199[0].x;
        acc[25] = a_199[0].y;
        acc[26] = a_199[1].x;
        acc[27] = a_199[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float2 _f2_127 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_127;
        float2 _f2_128 = make_float2(acc[0], acc[1]);
        float2 v_208 = _f2_128;
        output_sq_pair = fma_f32x2_rn_noftz(v_208, v_208, output_sq_pair);
        float2 _f2_129 = make_float2(acc[2], acc[3]);
        float2 v_209 = _f2_129;
        output_sq_pair = fma_f32x2_rn_noftz(v_209, v_209, output_sq_pair);
        float2 _f2_130 = make_float2(acc[4], acc[5]);
        float2 v_210 = _f2_130;
        output_sq_pair = fma_f32x2_rn_noftz(v_210, v_210, output_sq_pair);
        float2 _f2_131 = make_float2(acc[6], acc[7]);
        float2 v_211 = _f2_131;
        output_sq_pair = fma_f32x2_rn_noftz(v_211, v_211, output_sq_pair);
        float2 _f2_132 = make_float2(acc[8], acc[9]);
        float2 v_212 = _f2_132;
        output_sq_pair = fma_f32x2_rn_noftz(v_212, v_212, output_sq_pair);
        float2 _f2_133 = make_float2(acc[10], acc[11]);
        float2 v_213 = _f2_133;
        output_sq_pair = fma_f32x2_rn_noftz(v_213, v_213, output_sq_pair);
        float2 _f2_134 = make_float2(acc[12], acc[13]);
        float2 v_214 = _f2_134;
        output_sq_pair = fma_f32x2_rn_noftz(v_214, v_214, output_sq_pair);
        float2 _f2_135 = make_float2(acc[14], acc[15]);
        float2 v_215 = _f2_135;
        output_sq_pair = fma_f32x2_rn_noftz(v_215, v_215, output_sq_pair);
        float2 _f2_136 = make_float2(acc[16], acc[17]);
        float2 v_216 = _f2_136;
        output_sq_pair = fma_f32x2_rn_noftz(v_216, v_216, output_sq_pair);
        float2 _f2_137 = make_float2(acc[18], acc[19]);
        float2 v_217 = _f2_137;
        output_sq_pair = fma_f32x2_rn_noftz(v_217, v_217, output_sq_pair);
        float2 _f2_138 = make_float2(acc[20], acc[21]);
        float2 v_218 = _f2_138;
        output_sq_pair = fma_f32x2_rn_noftz(v_218, v_218, output_sq_pair);
        float2 _f2_139 = make_float2(acc[22], acc[23]);
        float2 v_219 = _f2_139;
        output_sq_pair = fma_f32x2_rn_noftz(v_219, v_219, output_sq_pair);
        float2 _f2_140 = make_float2(acc[24], acc[25]);
        float2 v_220 = _f2_140;
        output_sq_pair = fma_f32x2_rn_noftz(v_220, v_220, output_sq_pair);
        float2 _f2_141 = make_float2(acc[26], acc[27]);
        float2 v_221 = _f2_141;
        output_sq_pair = fma_f32x2_rn_noftz(v_221, v_221, output_sq_pair);
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
            float _rsqrt_1 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_1;
        }
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_2;
        float2 _f2_142 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_142;
        int base_222 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_143 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f2_143), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_144 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f2_144));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_223 = 2;
        const int acc_idx_224 = value_idx_223;
        float2 _f2_145 = make_float2(acc[acc_idx_224], acc[acc_idx_224 + 1]);
        float2 scaled_pair_225;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_225) : "l"(*(const unsigned long long*)&_f2_145), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_146 = make_float2(wout[acc_idx_224], wout[acc_idx_224 + 1]);
        float2 normalized_pair_226;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_226) : "l"(*(const unsigned long long*)&scaled_pair_225), "l"(*(const unsigned long long*)&_f2_146));
        output_values[value_idx_223] = normalized_pair_226.x;
        output_values[value_idx_223 + 1] = normalized_pair_226.y;
        const int value_idx_227 = 4;
        const int acc_idx_228 = value_idx_227;
        float2 _f2_147 = make_float2(acc[acc_idx_228], acc[acc_idx_228 + 1]);
        float2 scaled_pair_229;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_229) : "l"(*(const unsigned long long*)&_f2_147), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_148 = make_float2(wout[acc_idx_228], wout[acc_idx_228 + 1]);
        float2 normalized_pair_230;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_230) : "l"(*(const unsigned long long*)&scaled_pair_229), "l"(*(const unsigned long long*)&_f2_148));
        output_values[value_idx_227] = normalized_pair_230.x;
        output_values[value_idx_227 + 1] = normalized_pair_230.y;
        const int value_idx_231 = 6;
        const int acc_idx_232 = value_idx_231;
        float2 _f2_149 = make_float2(acc[acc_idx_232], acc[acc_idx_232 + 1]);
        float2 scaled_pair_233;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_233) : "l"(*(const unsigned long long*)&_f2_149), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_150 = make_float2(wout[acc_idx_232], wout[acc_idx_232 + 1]);
        float2 normalized_pair_234;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_234) : "l"(*(const unsigned long long*)&scaled_pair_233), "l"(*(const unsigned long long*)&_f2_150));
        output_values[value_idx_231] = normalized_pair_234.x;
        output_values[value_idx_231 + 1] = normalized_pair_234.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_222 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_235 = (2 + group) * 1024 + thread * 8;
        float output_values_236[8];
        const int value_idx_237 = 0;
        const int acc_idx_238 = 8 + value_idx_237;
        float2 _f2_151 = make_float2(acc[acc_idx_238], acc[acc_idx_238 + 1]);
        float2 scaled_pair_239;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_239) : "l"(*(const unsigned long long*)&_f2_151), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_152 = make_float2(wout[acc_idx_238], wout[acc_idx_238 + 1]);
        float2 normalized_pair_240;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_240) : "l"(*(const unsigned long long*)&scaled_pair_239), "l"(*(const unsigned long long*)&_f2_152));
        output_values_236[value_idx_237] = normalized_pair_240.x;
        output_values_236[value_idx_237 + 1] = normalized_pair_240.y;
        const int value_idx_241 = 2;
        const int acc_idx_242 = 8 + value_idx_241;
        float2 _f2_153 = make_float2(acc[acc_idx_242], acc[acc_idx_242 + 1]);
        float2 scaled_pair_243;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_243) : "l"(*(const unsigned long long*)&_f2_153), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_154 = make_float2(wout[acc_idx_242], wout[acc_idx_242 + 1]);
        float2 normalized_pair_244;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_244) : "l"(*(const unsigned long long*)&scaled_pair_243), "l"(*(const unsigned long long*)&_f2_154));
        output_values_236[value_idx_241] = normalized_pair_244.x;
        output_values_236[value_idx_241 + 1] = normalized_pair_244.y;
        const int value_idx_245 = 4;
        const int acc_idx_246 = 8 + value_idx_245;
        float2 _f2_155 = make_float2(acc[acc_idx_246], acc[acc_idx_246 + 1]);
        float2 scaled_pair_247;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_247) : "l"(*(const unsigned long long*)&_f2_155), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_156 = make_float2(wout[acc_idx_246], wout[acc_idx_246 + 1]);
        float2 normalized_pair_248;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_248) : "l"(*(const unsigned long long*)&scaled_pair_247), "l"(*(const unsigned long long*)&_f2_156));
        output_values_236[value_idx_245] = normalized_pair_248.x;
        output_values_236[value_idx_245 + 1] = normalized_pair_248.y;
        const int value_idx_249 = 6;
        const int acc_idx_250 = 8 + value_idx_249;
        float2 _f2_157 = make_float2(acc[acc_idx_250], acc[acc_idx_250 + 1]);
        float2 scaled_pair_251;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_251) : "l"(*(const unsigned long long*)&_f2_157), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_158 = make_float2(wout[acc_idx_250], wout[acc_idx_250 + 1]);
        float2 normalized_pair_252;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_252) : "l"(*(const unsigned long long*)&scaled_pair_251), "l"(*(const unsigned long long*)&_f2_158));
        output_values_236[value_idx_249] = normalized_pair_252.x;
        output_values_236[value_idx_249 + 1] = normalized_pair_252.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_236[0 + 0], output_values_236[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_236[0 + 2], output_values_236[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_236[0 + 4], output_values_236[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_236[0 + 6], output_values_236[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_235 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_253 = (4 + group) * 1024 + thread * 8;
        float output_values_254[8];
        const int value_idx_255 = 0;
        const int acc_idx_256 = 16 + value_idx_255;
        float2 _f2_159 = make_float2(acc[acc_idx_256], acc[acc_idx_256 + 1]);
        float2 scaled_pair_257;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_257) : "l"(*(const unsigned long long*)&_f2_159), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_160 = make_float2(wout[acc_idx_256], wout[acc_idx_256 + 1]);
        float2 normalized_pair_258;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_258) : "l"(*(const unsigned long long*)&scaled_pair_257), "l"(*(const unsigned long long*)&_f2_160));
        output_values_254[value_idx_255] = normalized_pair_258.x;
        output_values_254[value_idx_255 + 1] = normalized_pair_258.y;
        const int value_idx_259 = 2;
        const int acc_idx_260 = 16 + value_idx_259;
        float2 _f2_161 = make_float2(acc[acc_idx_260], acc[acc_idx_260 + 1]);
        float2 scaled_pair_261;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_261) : "l"(*(const unsigned long long*)&_f2_161), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_162 = make_float2(wout[acc_idx_260], wout[acc_idx_260 + 1]);
        float2 normalized_pair_262;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_262) : "l"(*(const unsigned long long*)&scaled_pair_261), "l"(*(const unsigned long long*)&_f2_162));
        output_values_254[value_idx_259] = normalized_pair_262.x;
        output_values_254[value_idx_259 + 1] = normalized_pair_262.y;
        const int value_idx_263 = 4;
        const int acc_idx_264 = 16 + value_idx_263;
        float2 _f2_163 = make_float2(acc[acc_idx_264], acc[acc_idx_264 + 1]);
        float2 scaled_pair_265;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_265) : "l"(*(const unsigned long long*)&_f2_163), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_164 = make_float2(wout[acc_idx_264], wout[acc_idx_264 + 1]);
        float2 normalized_pair_266;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_266) : "l"(*(const unsigned long long*)&scaled_pair_265), "l"(*(const unsigned long long*)&_f2_164));
        output_values_254[value_idx_263] = normalized_pair_266.x;
        output_values_254[value_idx_263 + 1] = normalized_pair_266.y;
        const int value_idx_267 = 6;
        const int acc_idx_268 = 16 + value_idx_267;
        float2 _f2_165 = make_float2(acc[acc_idx_268], acc[acc_idx_268 + 1]);
        float2 scaled_pair_269;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_269) : "l"(*(const unsigned long long*)&_f2_165), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_166 = make_float2(wout[acc_idx_268], wout[acc_idx_268 + 1]);
        float2 normalized_pair_270;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_270) : "l"(*(const unsigned long long*)&scaled_pair_269), "l"(*(const unsigned long long*)&_f2_166));
        output_values_254[value_idx_267] = normalized_pair_270.x;
        output_values_254[value_idx_267 + 1] = normalized_pair_270.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_254[0 + 0], output_values_254[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_254[0 + 2], output_values_254[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_254[0 + 4], output_values_254[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_254[0 + 6], output_values_254[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_253 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_271 = 6144 + group * 512 + thread * 4;
        float output_values_272[8];
        const int value_idx_273 = 0;
        const int acc_idx_274 = 24 + value_idx_273;
        float2 _f2_167 = make_float2(acc[acc_idx_274], acc[acc_idx_274 + 1]);
        float2 scaled_pair_275;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_275) : "l"(*(const unsigned long long*)&_f2_167), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_168 = make_float2(wout[acc_idx_274], wout[acc_idx_274 + 1]);
        float2 normalized_pair_276;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_276) : "l"(*(const unsigned long long*)&scaled_pair_275), "l"(*(const unsigned long long*)&_f2_168));
        output_values_272[value_idx_273] = normalized_pair_276.x;
        output_values_272[value_idx_273 + 1] = normalized_pair_276.y;
        const int value_idx_277 = 2;
        const int acc_idx_278 = 24 + value_idx_277;
        float2 _f2_169 = make_float2(acc[acc_idx_278], acc[acc_idx_278 + 1]);
        float2 scaled_pair_279;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_279) : "l"(*(const unsigned long long*)&_f2_169), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f2_170 = make_float2(wout[acc_idx_278], wout[acc_idx_278 + 1]);
        float2 normalized_pair_280;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_280) : "l"(*(const unsigned long long*)&scaled_pair_279), "l"(*(const unsigned long long*)&_f2_170));
        output_values_272[value_idx_277] = normalized_pair_280.x;
        output_values_272[value_idx_277 + 1] = normalized_pair_280.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_272[0 + 0], output_values_272[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_272[0 + 2], output_values_272[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_271]) = _pk2;
        }
    }
}

} // extern "C"
