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
#define SMEM_STATS_STAGE_BYTES 320
#define SMEM_STATS_STRIDE 320
#define SMEM_OUT_STATS_OFF 320
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 384
#define THREADS 128

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

__global__ __launch_bounds__(128) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_attn_res_c0ab67a53d992f2fdf17(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* stats = reinterpret_cast<float*>(smem_raw + 0);
    const int stats_addr = smem + 0;
    float* out_stats = reinterpret_cast<float*>(smem_raw + 320);
    const int out_stats_addr = smem + 320;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = warp;
    int token = bid;
    warp_0 = cta_rank * 4 + warp;
    token = cluster_id;
    int group = warp_0 / 4;
    int thread = (warp_0 * 32 + lane) % 128;
    if (token < M) {
        unsigned long long token64 = (unsigned long long)token;
        unsigned long long row_base = token64 * 7168;
        unsigned long long block_base = token64 * blocks_m_stride;
        unsigned int words[70];
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
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_12[0 + 0] = _uv4_4.x;
                _vec_load_12[0 + 1] = _uv4_4.y;
                _vec_load_12[0 + 2] = _uv4_4.z;
                _vec_load_12[0 + 3] = _uv4_4.w;
            }
            words[56 + woff] = _vec_load_12[0];
            words[56 + woff + 1] = _vec_load_12[1];
            words[56 + woff + 2] = _vec_load_12[2];
            words[56 + woff + 3] = _vec_load_12[3];
        }
        {
            unsigned int _vec_load_15[4];
            {
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_15[0 + 0] = _uv4_5.x;
                _vec_load_15[0 + 1] = _uv4_5.y;
                _vec_load_15[0 + 2] = _uv4_5.z;
                _vec_load_15[0 + 3] = _uv4_5.w;
            }
            dwords[woff] = _vec_load_15[0];
            dwords[woff + 1] = _vec_load_15[1];
            dwords[woff + 2] = _vec_load_15[2];
            dwords[woff + 3] = _vec_load_15[3];
        }
        float _vec_load_18[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_18[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_18[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        float _vec_load_19[8];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
        }
        q[0] = _vec_load_18[0] * _vec_load_19[0];
        q[1] = _vec_load_18[1] * _vec_load_19[1];
        q[2] = _vec_load_18[2] * _vec_load_19[2];
        q[3] = _vec_load_18[3] * _vec_load_19[3];
        q[4] = _vec_load_18[4] * _vec_load_19[4];
        q[5] = _vec_load_18[5] * _vec_load_19[5];
        q[6] = _vec_load_18[6] * _vec_load_19[6];
        q[7] = _vec_load_18[7] * _vec_load_19[7];
        float _vec_load_20[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(output_norm_weight + base);
            uint4 _vld_8[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_8[_blk] = _vptr_8[_blk];
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        wout[0] = _vec_load_20[0];
        wout[1] = _vec_load_20[1];
        wout[2] = _vec_load_20[2];
        wout[3] = _vec_load_20[3];
        wout[4] = _vec_load_20[4];
        wout[5] = _vec_load_20[5];
        wout[6] = _vec_load_20[6];
        wout[7] = _vec_load_20[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_21[4];
            {
                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_21[0 + 0] = _uv4_9.x;
                _vec_load_21[0 + 1] = _uv4_9.y;
                _vec_load_21[0 + 2] = _uv4_9.z;
                _vec_load_21[0 + 3] = _uv4_9.w;
            }
            words[woff_1] = _vec_load_21[0];
            words[woff_1 + 1] = _vec_load_21[1];
            words[woff_1 + 2] = _vec_load_21[2];
            words[woff_1 + 3] = _vec_load_21[3];
        }
        {
            unsigned int _vec_load_24[4];
            {
                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_24[0 + 0] = _uv4_10.x;
                _vec_load_24[0 + 1] = _uv4_10.y;
                _vec_load_24[0 + 2] = _uv4_10.z;
                _vec_load_24[0 + 3] = _uv4_10.w;
            }
            words[14 + woff_1] = _vec_load_24[0];
            words[14 + woff_1 + 1] = _vec_load_24[1];
            words[14 + woff_1 + 2] = _vec_load_24[2];
            words[14 + woff_1 + 3] = _vec_load_24[3];
        }
        {
            unsigned int _vec_load_27[4];
            {
                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_27[0 + 0] = _uv4_11.x;
                _vec_load_27[0 + 1] = _uv4_11.y;
                _vec_load_27[0 + 2] = _uv4_11.z;
                _vec_load_27[0 + 3] = _uv4_11.w;
            }
            words[28 + woff_1] = _vec_load_27[0];
            words[28 + woff_1 + 1] = _vec_load_27[1];
            words[28 + woff_1 + 2] = _vec_load_27[2];
            words[28 + woff_1 + 3] = _vec_load_27[3];
        }
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_30[0 + 0] = _uv4_12.x;
                _vec_load_30[0 + 1] = _uv4_12.y;
                _vec_load_30[0 + 2] = _uv4_12.z;
                _vec_load_30[0 + 3] = _uv4_12.w;
            }
            words[42 + woff_1] = _vec_load_30[0];
            words[42 + woff_1 + 1] = _vec_load_30[1];
            words[42 + woff_1 + 2] = _vec_load_30[2];
            words[42 + woff_1 + 3] = _vec_load_30[3];
        }
        {
            unsigned int _vec_load_33[4];
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_33[0 + 0] = _uv4_13.x;
                _vec_load_33[0 + 1] = _uv4_13.y;
                _vec_load_33[0 + 2] = _uv4_13.z;
                _vec_load_33[0 + 3] = _uv4_13.w;
            }
            words[56 + woff_1] = _vec_load_33[0];
            words[56 + woff_1 + 1] = _vec_load_33[1];
            words[56 + woff_1 + 2] = _vec_load_33[2];
            words[56 + woff_1 + 3] = _vec_load_33[3];
        }
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_36[0 + 0] = _uv4_14.x;
                _vec_load_36[0 + 1] = _uv4_14.y;
                _vec_load_36[0 + 2] = _uv4_14.z;
                _vec_load_36[0 + 3] = _uv4_14.w;
            }
            dwords[woff_1] = _vec_load_36[0];
            dwords[woff_1 + 1] = _vec_load_36[1];
            dwords[woff_1 + 2] = _vec_load_36[2];
            dwords[woff_1 + 3] = _vec_load_36[3];
        }
        float _vec_load_39[8];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_39[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_39[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
        }
        float _vec_load_40[8];
        {
            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(qk_weight + base_0);
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
                        : "=f"((&_vec_load_40[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_40[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_16[_pair]));
                }
            }
        }
        q[8] = _vec_load_39[0] * _vec_load_40[0];
        q[9] = _vec_load_39[1] * _vec_load_40[1];
        q[10] = _vec_load_39[2] * _vec_load_40[2];
        q[11] = _vec_load_39[3] * _vec_load_40[3];
        q[12] = _vec_load_39[4] * _vec_load_40[4];
        q[13] = _vec_load_39[5] * _vec_load_40[5];
        q[14] = _vec_load_39[6] * _vec_load_40[6];
        q[15] = _vec_load_39[7] * _vec_load_40[7];
        float _vec_load_41[8];
        {
            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_41[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_41[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_17[_pair]));
                }
            }
        }
        wout[8] = _vec_load_41[0];
        wout[9] = _vec_load_41[1];
        wout[10] = _vec_load_41[2];
        wout[11] = _vec_load_41[3];
        wout[12] = _vec_load_41[4];
        wout[13] = _vec_load_41[5];
        wout[14] = _vec_load_41[6];
        wout[15] = _vec_load_41[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_42[4];
            {
                uint4 _uv4_18 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_42[0 + 0] = _uv4_18.x;
                _vec_load_42[0 + 1] = _uv4_18.y;
                _vec_load_42[0 + 2] = _uv4_18.z;
                _vec_load_42[0 + 3] = _uv4_18.w;
            }
            words[woff_3] = _vec_load_42[0];
            words[woff_3 + 1] = _vec_load_42[1];
            words[woff_3 + 2] = _vec_load_42[2];
            words[woff_3 + 3] = _vec_load_42[3];
        }
        {
            unsigned int _vec_load_45[4];
            {
                uint4 _uv4_19 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_45[0 + 0] = _uv4_19.x;
                _vec_load_45[0 + 1] = _uv4_19.y;
                _vec_load_45[0 + 2] = _uv4_19.z;
                _vec_load_45[0 + 3] = _uv4_19.w;
            }
            words[14 + woff_3] = _vec_load_45[0];
            words[14 + woff_3 + 1] = _vec_load_45[1];
            words[14 + woff_3 + 2] = _vec_load_45[2];
            words[14 + woff_3 + 3] = _vec_load_45[3];
        }
        {
            unsigned int _vec_load_48[4];
            {
                uint4 _uv4_20 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_48[0 + 0] = _uv4_20.x;
                _vec_load_48[0 + 1] = _uv4_20.y;
                _vec_load_48[0 + 2] = _uv4_20.z;
                _vec_load_48[0 + 3] = _uv4_20.w;
            }
            words[28 + woff_3] = _vec_load_48[0];
            words[28 + woff_3 + 1] = _vec_load_48[1];
            words[28 + woff_3 + 2] = _vec_load_48[2];
            words[28 + woff_3 + 3] = _vec_load_48[3];
        }
        {
            unsigned int _vec_load_51[4];
            {
                uint4 _uv4_21 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_51[0 + 0] = _uv4_21.x;
                _vec_load_51[0 + 1] = _uv4_21.y;
                _vec_load_51[0 + 2] = _uv4_21.z;
                _vec_load_51[0 + 3] = _uv4_21.w;
            }
            words[42 + woff_3] = _vec_load_51[0];
            words[42 + woff_3 + 1] = _vec_load_51[1];
            words[42 + woff_3 + 2] = _vec_load_51[2];
            words[42 + woff_3 + 3] = _vec_load_51[3];
        }
        {
            unsigned int _vec_load_54[4];
            {
                uint4 _uv4_22 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_54[0 + 0] = _uv4_22.x;
                _vec_load_54[0 + 1] = _uv4_22.y;
                _vec_load_54[0 + 2] = _uv4_22.z;
                _vec_load_54[0 + 3] = _uv4_22.w;
            }
            words[56 + woff_3] = _vec_load_54[0];
            words[56 + woff_3 + 1] = _vec_load_54[1];
            words[56 + woff_3 + 2] = _vec_load_54[2];
            words[56 + woff_3 + 3] = _vec_load_54[3];
        }
        {
            unsigned int _vec_load_57[4];
            {
                uint4 _uv4_23 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_57[0 + 0] = _uv4_23.x;
                _vec_load_57[0 + 1] = _uv4_23.y;
                _vec_load_57[0 + 2] = _uv4_23.z;
                _vec_load_57[0 + 3] = _uv4_23.w;
            }
            dwords[woff_3] = _vec_load_57[0];
            dwords[woff_3 + 1] = _vec_load_57[1];
            dwords[woff_3 + 2] = _vec_load_57[2];
            dwords[woff_3 + 3] = _vec_load_57[3];
        }
        float _vec_load_60[8];
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_24[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_24[_blk] = _vptr_24[_blk];
                uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_60[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_60[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_24[_pair]));
                }
            }
        }
        float _vec_load_61[8];
        {
            const uint4* _vptr_25 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_25[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_25[_blk] = _vptr_25[_blk];
                uint32_t* _vpairs_25 = reinterpret_cast<uint32_t*>(&_vld_25[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_61[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_61[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_25[_pair]));
                }
            }
        }
        q[16] = _vec_load_60[0] * _vec_load_61[0];
        q[17] = _vec_load_60[1] * _vec_load_61[1];
        q[18] = _vec_load_60[2] * _vec_load_61[2];
        q[19] = _vec_load_60[3] * _vec_load_61[3];
        q[20] = _vec_load_60[4] * _vec_load_61[4];
        q[21] = _vec_load_60[5] * _vec_load_61[5];
        q[22] = _vec_load_60[6] * _vec_load_61[6];
        q[23] = _vec_load_60[7] * _vec_load_61[7];
        float _vec_load_62[8];
        {
            const uint4* _vptr_26 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_26[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_26[_blk] = _vptr_26[_blk];
                uint32_t* _vpairs_26 = reinterpret_cast<uint32_t*>(&_vld_26[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_62[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_62[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_26[_pair]));
                }
            }
        }
        wout[16] = _vec_load_62[0];
        wout[17] = _vec_load_62[1];
        wout[18] = _vec_load_62[2];
        wout[19] = _vec_load_62[3];
        wout[20] = _vec_load_62[4];
        wout[21] = _vec_load_62[5];
        wout[22] = _vec_load_62[6];
        wout[23] = _vec_load_62[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_64[1];
            {
                _vec_load_64[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_64[0];
            unsigned int _vec_load_65[1];
            {
                _vec_load_65[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_65[0];
        }
        {
            unsigned int _vec_load_67[1];
            {
                _vec_load_67[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_67[0];
            unsigned int _vec_load_68[1];
            {
                _vec_load_68[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_68[0];
        }
        {
            unsigned int _vec_load_70[1];
            {
                _vec_load_70[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_70[0];
            unsigned int _vec_load_71[1];
            {
                _vec_load_71[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_71[0];
        }
        {
            unsigned int _vec_load_73[1];
            {
                _vec_load_73[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[42 + woff_5] = _vec_load_73[0];
            unsigned int _vec_load_74[1];
            {
                _vec_load_74[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[42 + woff_5 + 1] = _vec_load_74[0];
        }
        {
            unsigned int _vec_load_76[1];
            {
                _vec_load_76[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[56 + woff_5] = _vec_load_76[0];
            unsigned int _vec_load_77[1];
            {
                _vec_load_77[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[56 + woff_5 + 1] = _vec_load_77[0];
        }
        {
            unsigned int _vec_load_79[1];
            {
                _vec_load_79[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_79[0];
            unsigned int _vec_load_80[1];
            {
                _vec_load_80[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_80[0];
        }
        float _vec_load_81[4];
        {
            uint2 _vld_27;
            _vld_27 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_81[0 + _pair * 2])[0]), "=f"((&_vec_load_81[0 + _pair * 2])[1])
                    : "r"(_vpairs_27[_pair]));
            }
        }
        float _vec_load_82[4];
        {
            uint2 _vld_28;
            _vld_28 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_82[0 + _pair * 2])[0]), "=f"((&_vec_load_82[0 + _pair * 2])[1])
                    : "r"(_vpairs_28[_pair]));
            }
        }
        q[24] = _vec_load_81[0] * _vec_load_82[0];
        q[25] = _vec_load_81[1] * _vec_load_82[1];
        q[26] = _vec_load_81[2] * _vec_load_82[2];
        q[27] = _vec_load_81[3] * _vec_load_82[3];
        float _vec_load_83[4];
        {
            uint2 _vld_29;
            _vld_29 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_83[0 + _pair * 2])[0]), "=f"((&_vec_load_83[0 + _pair * 2])[1])
                    : "r"(_vpairs_29[_pair]));
            }
        }
        wout[24] = _vec_load_83[0];
        wout[25] = _vec_load_83[1];
        wout[26] = _vec_load_83[2];
        wout[27] = _vec_load_83[3];
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
        float2 sq[5];
        float2 dot[5];
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
        float2 _f2_8 = make_float2(0.0f, 0.0f);
        sq[4] = _f2_8;
        float2 _f2_9 = make_float2(0.0f, 0.0f);
        dot[4] = _f2_9;
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
            float2 _f2_16 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_16;
            float2 _f2_17 = make_float2(q[0], q[1]);
            float2 qp = _f2_17;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_18 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_18;
            float2 _f2_19 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_19;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_20 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_20;
            float2 _f2_21 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_21;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_22 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_22;
            float2 _f2_23 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_23;
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
            float2 _f2_30 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_30;
            float2 _f2_31 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_31;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_32 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_32;
            float2 _f2_33 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_33;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_34 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_34;
            float2 _f2_35 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_35;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_36 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_36;
            float2 _f2_37 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_37;
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
            float2 _f2_44 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_44;
            float2 _f2_45 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_45;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_46 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_46;
            float2 _f2_47 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_47;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_48 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_48;
            float2 _f2_49 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_49;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_50 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_50;
            float2 _f2_51 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_51;
            sq[2] = fma_f32x2_rn_noftz(v_4_2, v_4_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_2, qp_5_2, dot[2]);
        }
        unsigned int sw_10[4];
        sw_10[0] = words[42 + woff_7];
        sw_10[1] = words[42 + woff_7 + 1];
        sw_10[2] = words[42 + woff_7 + 2];
        sw_10[3] = words[42 + woff_7 + 3];
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
            float2 _f2_58 = make_float2(sw_10_f32[0], sw_10_f32[1]);
            float2 v_5 = _f2_58;
            float2 _f2_59 = make_float2(q[0], q[1]);
            float2 qp_6 = _f2_59;
            sq[3] = fma_f32x2_rn_noftz(v_5, v_5, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_5, qp_6, dot[3]);
            float2 _f2_60 = make_float2(sw_10_f32[2], sw_10_f32[3]);
            float2 v_0_3 = _f2_60;
            float2 _f2_61 = make_float2(q[2], q[3]);
            float2 qp_1_3 = _f2_61;
            sq[3] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[3]);
            float2 _f2_62 = make_float2(sw_10_f32[4], sw_10_f32[5]);
            float2 v_2_3 = _f2_62;
            float2 _f2_63 = make_float2(q[4], q[5]);
            float2 qp_3_3 = _f2_63;
            sq[3] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[3]);
            float2 _f2_64 = make_float2(sw_10_f32[6], sw_10_f32[7]);
            float2 v_4_3 = _f2_64;
            float2 _f2_65 = make_float2(q[6], q[7]);
            float2 qp_5_3 = _f2_65;
            sq[3] = fma_f32x2_rn_noftz(v_4_3, v_4_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_3, qp_5_3, dot[3]);
        }
        unsigned int sw_11[4];
        sw_11[0] = words[56 + woff_7];
        sw_11[1] = words[56 + woff_7 + 1];
        sw_11[2] = words[56 + woff_7 + 2];
        sw_11[3] = words[56 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_11[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_11[0] = __as_u32(mixed);
            words[56 + woff_7] = sw_11[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_11[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_11[1] = __as_u32(mixed_2);
            words[56 + woff_7 + 1] = sw_11[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_11[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_11[2] = __as_u32(mixed_5);
            words[56 + woff_7 + 2] = sw_11[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_11[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_11[3] = __as_u32(mixed_8);
            words[56 + woff_7 + 3] = sw_11[3];
            {
                int4 _iv4 = make_int4(sw_11[0 + 0], sw_11[0 + 1], sw_11[0 + 2], sw_11[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
            unsigned long long write_base = block_base + 4 * blocks_k_stride + (unsigned long long)base_6;
            {
                int4 _iv4 = make_int4(sw_11[0 + 0], sw_11[0 + 1], sw_11[0 + 2], sw_11[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base) + 0) = _iv4;
            }
        }
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
            float2 _f2_72 = make_float2(sw_11_f32[0], sw_11_f32[1]);
            float2 v_6 = _f2_72;
            float2 _f2_73 = make_float2(q[0], q[1]);
            float2 qp_7 = _f2_73;
            sq[4] = fma_f32x2_rn_noftz(v_6, v_6, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_6, qp_7, dot[4]);
            float2 _f2_74 = make_float2(sw_11_f32[2], sw_11_f32[3]);
            float2 v_0_4 = _f2_74;
            float2 _f2_75 = make_float2(q[2], q[3]);
            float2 qp_1_4 = _f2_75;
            sq[4] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[4]);
            float2 _f2_76 = make_float2(sw_11_f32[4], sw_11_f32[5]);
            float2 v_2_4 = _f2_76;
            float2 _f2_77 = make_float2(q[4], q[5]);
            float2 qp_3_4 = _f2_77;
            sq[4] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[4]);
            float2 _f2_78 = make_float2(sw_11_f32[6], sw_11_f32[7]);
            float2 v_4_4 = _f2_78;
            float2 _f2_79 = make_float2(q[6], q[7]);
            float2 qp_5_4 = _f2_79;
            sq[4] = fma_f32x2_rn_noftz(v_4_4, v_4_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_4, qp_5_4, dot[4]);
        }
        int base_12 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_13 = 4;
        unsigned int sw_14[4];
        sw_14[0] = words[woff_13];
        sw_14[1] = words[woff_13 + 1];
        sw_14[2] = words[woff_13 + 2];
        sw_14[3] = words[woff_13 + 3];
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
            fsrc[8] = sw_14_f32[0];
            fsrc[9] = sw_14_f32[1];
            fsrc[10] = sw_14_f32[2];
            fsrc[11] = sw_14_f32[3];
            fsrc[12] = sw_14_f32[4];
            fsrc[13] = sw_14_f32[5];
            fsrc[14] = sw_14_f32[6];
            fsrc[15] = sw_14_f32[7];
        }
        {
            float2 _f2_86 = make_float2(sw_14_f32[0], sw_14_f32[1]);
            float2 v_7 = _f2_86;
            float2 _f2_87 = make_float2(q[8], q[9]);
            float2 qp_8 = _f2_87;
            sq[0] = fma_f32x2_rn_noftz(v_7, v_7, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_7, qp_8, dot[0]);
            float2 _f2_88 = make_float2(sw_14_f32[2], sw_14_f32[3]);
            float2 v_0_5 = _f2_88;
            float2 _f2_89 = make_float2(q[10], q[11]);
            float2 qp_1_5 = _f2_89;
            sq[0] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[0]);
            float2 _f2_90 = make_float2(sw_14_f32[4], sw_14_f32[5]);
            float2 v_2_5 = _f2_90;
            float2 _f2_91 = make_float2(q[12], q[13]);
            float2 qp_3_5 = _f2_91;
            sq[0] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[0]);
            float2 _f2_92 = make_float2(sw_14_f32[6], sw_14_f32[7]);
            float2 v_4_5 = _f2_92;
            float2 _f2_93 = make_float2(q[14], q[15]);
            float2 qp_5_5 = _f2_93;
            sq[0] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[0]);
        }
        unsigned int sw_15[4];
        sw_15[0] = words[14 + woff_13];
        sw_15[1] = words[14 + woff_13 + 1];
        sw_15[2] = words[14 + woff_13 + 2];
        sw_15[3] = words[14 + woff_13 + 3];
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
            fsrc[36] = sw_15_f32[0];
            fsrc[37] = sw_15_f32[1];
            fsrc[38] = sw_15_f32[2];
            fsrc[39] = sw_15_f32[3];
            fsrc[40] = sw_15_f32[4];
            fsrc[41] = sw_15_f32[5];
            fsrc[42] = sw_15_f32[6];
            fsrc[43] = sw_15_f32[7];
        }
        {
            float2 _f2_100 = make_float2(sw_15_f32[0], sw_15_f32[1]);
            float2 v_8 = _f2_100;
            float2 _f2_101 = make_float2(q[8], q[9]);
            float2 qp_9 = _f2_101;
            sq[1] = fma_f32x2_rn_noftz(v_8, v_8, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_8, qp_9, dot[1]);
            float2 _f2_102 = make_float2(sw_15_f32[2], sw_15_f32[3]);
            float2 v_0_6 = _f2_102;
            float2 _f2_103 = make_float2(q[10], q[11]);
            float2 qp_1_6 = _f2_103;
            sq[1] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[1]);
            float2 _f2_104 = make_float2(sw_15_f32[4], sw_15_f32[5]);
            float2 v_2_6 = _f2_104;
            float2 _f2_105 = make_float2(q[12], q[13]);
            float2 qp_3_6 = _f2_105;
            sq[1] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[1]);
            float2 _f2_106 = make_float2(sw_15_f32[6], sw_15_f32[7]);
            float2 v_4_6 = _f2_106;
            float2 _f2_107 = make_float2(q[14], q[15]);
            float2 qp_5_6 = _f2_107;
            sq[1] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[1]);
        }
        unsigned int sw_16[4];
        sw_16[0] = words[28 + woff_13];
        sw_16[1] = words[28 + woff_13 + 1];
        sw_16[2] = words[28 + woff_13 + 2];
        sw_16[3] = words[28 + woff_13 + 3];
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
            fsrc[64] = sw_16_f32[0];
            fsrc[65] = sw_16_f32[1];
            fsrc[66] = sw_16_f32[2];
            fsrc[67] = sw_16_f32[3];
            fsrc[68] = sw_16_f32[4];
            fsrc[69] = sw_16_f32[5];
            fsrc[70] = sw_16_f32[6];
            fsrc[71] = sw_16_f32[7];
        }
        {
            float2 _f2_114 = make_float2(sw_16_f32[0], sw_16_f32[1]);
            float2 v_9 = _f2_114;
            float2 _f2_115 = make_float2(q[8], q[9]);
            float2 qp_10 = _f2_115;
            sq[2] = fma_f32x2_rn_noftz(v_9, v_9, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_9, qp_10, dot[2]);
            float2 _f2_116 = make_float2(sw_16_f32[2], sw_16_f32[3]);
            float2 v_0_7 = _f2_116;
            float2 _f2_117 = make_float2(q[10], q[11]);
            float2 qp_1_7 = _f2_117;
            sq[2] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[2]);
            float2 _f2_118 = make_float2(sw_16_f32[4], sw_16_f32[5]);
            float2 v_2_7 = _f2_118;
            float2 _f2_119 = make_float2(q[12], q[13]);
            float2 qp_3_7 = _f2_119;
            sq[2] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[2]);
            float2 _f2_120 = make_float2(sw_16_f32[6], sw_16_f32[7]);
            float2 v_4_7 = _f2_120;
            float2 _f2_121 = make_float2(q[14], q[15]);
            float2 qp_5_7 = _f2_121;
            sq[2] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[2]);
        }
        unsigned int sw_17[4];
        sw_17[0] = words[42 + woff_13];
        sw_17[1] = words[42 + woff_13 + 1];
        sw_17[2] = words[42 + woff_13 + 2];
        sw_17[3] = words[42 + woff_13 + 3];
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
            float2 _f2_128 = make_float2(sw_17_f32[0], sw_17_f32[1]);
            float2 v_10 = _f2_128;
            float2 _f2_129 = make_float2(q[8], q[9]);
            float2 qp_11 = _f2_129;
            sq[3] = fma_f32x2_rn_noftz(v_10, v_10, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_10, qp_11, dot[3]);
            float2 _f2_130 = make_float2(sw_17_f32[2], sw_17_f32[3]);
            float2 v_0_8 = _f2_130;
            float2 _f2_131 = make_float2(q[10], q[11]);
            float2 qp_1_8 = _f2_131;
            sq[3] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[3]);
            float2 _f2_132 = make_float2(sw_17_f32[4], sw_17_f32[5]);
            float2 v_2_8 = _f2_132;
            float2 _f2_133 = make_float2(q[12], q[13]);
            float2 qp_3_8 = _f2_133;
            sq[3] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[3]);
            float2 _f2_134 = make_float2(sw_17_f32[6], sw_17_f32[7]);
            float2 v_4_8 = _f2_134;
            float2 _f2_135 = make_float2(q[14], q[15]);
            float2 qp_5_8 = _f2_135;
            sq[3] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[3]);
        }
        unsigned int sw_18[4];
        sw_18[0] = words[56 + woff_13];
        sw_18[1] = words[56 + woff_13 + 1];
        sw_18[2] = words[56 + woff_13 + 2];
        sw_18[3] = words[56 + woff_13 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_18[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_13]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_18[0] = __as_u32(mixed_1);
            words[56 + woff_13] = sw_18[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_18[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_13 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_18[1] = __as_u32(mixed_2_1);
            words[56 + woff_13 + 1] = sw_18[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_18[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_13 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_18[2] = __as_u32(mixed_5_1);
            words[56 + woff_13 + 2] = sw_18[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_18[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_13 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_18[3] = __as_u32(mixed_8_1);
            words[56 + woff_13 + 3] = sw_18[3];
            {
                int4 _iv4 = make_int4(sw_18[0 + 0], sw_18[0 + 1], sw_18[0 + 2], sw_18[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_12)) + 0) = _iv4;
            }
            unsigned long long write_base_1 = block_base + 4 * blocks_k_stride + (unsigned long long)base_12;
            {
                int4 _iv4 = make_int4(sw_18[0 + 0], sw_18[0 + 1], sw_18[0 + 2], sw_18[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_1) + 0) = _iv4;
            }
        }
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
            float2 _f2_142 = make_float2(sw_18_f32[0], sw_18_f32[1]);
            float2 v_11 = _f2_142;
            float2 _f2_143 = make_float2(q[8], q[9]);
            float2 qp_12 = _f2_143;
            sq[4] = fma_f32x2_rn_noftz(v_11, v_11, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_11, qp_12, dot[4]);
            float2 _f2_144 = make_float2(sw_18_f32[2], sw_18_f32[3]);
            float2 v_0_9 = _f2_144;
            float2 _f2_145 = make_float2(q[10], q[11]);
            float2 qp_1_9 = _f2_145;
            sq[4] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_9, qp_1_9, dot[4]);
            float2 _f2_146 = make_float2(sw_18_f32[4], sw_18_f32[5]);
            float2 v_2_9 = _f2_146;
            float2 _f2_147 = make_float2(q[12], q[13]);
            float2 qp_3_9 = _f2_147;
            sq[4] = fma_f32x2_rn_noftz(v_2_9, v_2_9, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[4]);
            float2 _f2_148 = make_float2(sw_18_f32[6], sw_18_f32[7]);
            float2 v_4_9 = _f2_148;
            float2 _f2_149 = make_float2(q[14], q[15]);
            float2 qp_5_9 = _f2_149;
            sq[4] = fma_f32x2_rn_noftz(v_4_9, v_4_9, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_9, qp_5_9, dot[4]);
        }
        int base_19 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_20 = 8;
        unsigned int sw_21[4];
        sw_21[0] = words[woff_20];
        sw_21[1] = words[woff_20 + 1];
        sw_21[2] = words[woff_20 + 2];
        sw_21[3] = words[woff_20 + 3];
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
            fsrc[16] = sw_21_f32[0];
            fsrc[17] = sw_21_f32[1];
            fsrc[18] = sw_21_f32[2];
            fsrc[19] = sw_21_f32[3];
            fsrc[20] = sw_21_f32[4];
            fsrc[21] = sw_21_f32[5];
            fsrc[22] = sw_21_f32[6];
            fsrc[23] = sw_21_f32[7];
        }
        {
            float2 _f2_156 = make_float2(sw_21_f32[0], sw_21_f32[1]);
            float2 v_12 = _f2_156;
            float2 _f2_157 = make_float2(q[16], q[17]);
            float2 qp_13 = _f2_157;
            sq[0] = fma_f32x2_rn_noftz(v_12, v_12, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_12, qp_13, dot[0]);
            float2 _f2_158 = make_float2(sw_21_f32[2], sw_21_f32[3]);
            float2 v_0_10 = _f2_158;
            float2 _f2_159 = make_float2(q[18], q[19]);
            float2 qp_1_10 = _f2_159;
            sq[0] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_10, qp_1_10, dot[0]);
            float2 _f2_160 = make_float2(sw_21_f32[4], sw_21_f32[5]);
            float2 v_2_10 = _f2_160;
            float2 _f2_161 = make_float2(q[20], q[21]);
            float2 qp_3_10 = _f2_161;
            sq[0] = fma_f32x2_rn_noftz(v_2_10, v_2_10, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[0]);
            float2 _f2_162 = make_float2(sw_21_f32[6], sw_21_f32[7]);
            float2 v_4_10 = _f2_162;
            float2 _f2_163 = make_float2(q[22], q[23]);
            float2 qp_5_10 = _f2_163;
            sq[0] = fma_f32x2_rn_noftz(v_4_10, v_4_10, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_10, qp_5_10, dot[0]);
        }
        unsigned int sw_22[4];
        sw_22[0] = words[14 + woff_20];
        sw_22[1] = words[14 + woff_20 + 1];
        sw_22[2] = words[14 + woff_20 + 2];
        sw_22[3] = words[14 + woff_20 + 3];
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
            fsrc[44] = sw_22_f32[0];
            fsrc[45] = sw_22_f32[1];
            fsrc[46] = sw_22_f32[2];
            fsrc[47] = sw_22_f32[3];
            fsrc[48] = sw_22_f32[4];
            fsrc[49] = sw_22_f32[5];
            fsrc[50] = sw_22_f32[6];
            fsrc[51] = sw_22_f32[7];
        }
        {
            float2 _f2_170 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_13 = _f2_170;
            float2 _f2_171 = make_float2(q[16], q[17]);
            float2 qp_14 = _f2_171;
            sq[1] = fma_f32x2_rn_noftz(v_13, v_13, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_13, qp_14, dot[1]);
            float2 _f2_172 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_0_11 = _f2_172;
            float2 _f2_173 = make_float2(q[18], q[19]);
            float2 qp_1_11 = _f2_173;
            sq[1] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_11, qp_1_11, dot[1]);
            float2 _f2_174 = make_float2(sw_22_f32[4], sw_22_f32[5]);
            float2 v_2_11 = _f2_174;
            float2 _f2_175 = make_float2(q[20], q[21]);
            float2 qp_3_11 = _f2_175;
            sq[1] = fma_f32x2_rn_noftz(v_2_11, v_2_11, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[1]);
            float2 _f2_176 = make_float2(sw_22_f32[6], sw_22_f32[7]);
            float2 v_4_11 = _f2_176;
            float2 _f2_177 = make_float2(q[22], q[23]);
            float2 qp_5_11 = _f2_177;
            sq[1] = fma_f32x2_rn_noftz(v_4_11, v_4_11, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_11, qp_5_11, dot[1]);
        }
        unsigned int sw_23[4];
        sw_23[0] = words[28 + woff_20];
        sw_23[1] = words[28 + woff_20 + 1];
        sw_23[2] = words[28 + woff_20 + 2];
        sw_23[3] = words[28 + woff_20 + 3];
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
            fsrc[72] = sw_23_f32[0];
            fsrc[73] = sw_23_f32[1];
            fsrc[74] = sw_23_f32[2];
            fsrc[75] = sw_23_f32[3];
            fsrc[76] = sw_23_f32[4];
            fsrc[77] = sw_23_f32[5];
            fsrc[78] = sw_23_f32[6];
            fsrc[79] = sw_23_f32[7];
        }
        {
            float2 _f2_184 = make_float2(sw_23_f32[0], sw_23_f32[1]);
            float2 v_14 = _f2_184;
            float2 _f2_185 = make_float2(q[16], q[17]);
            float2 qp_15 = _f2_185;
            sq[2] = fma_f32x2_rn_noftz(v_14, v_14, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_14, qp_15, dot[2]);
            float2 _f2_186 = make_float2(sw_23_f32[2], sw_23_f32[3]);
            float2 v_0_12 = _f2_186;
            float2 _f2_187 = make_float2(q[18], q[19]);
            float2 qp_1_12 = _f2_187;
            sq[2] = fma_f32x2_rn_noftz(v_0_12, v_0_12, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_12, qp_1_12, dot[2]);
            float2 _f2_188 = make_float2(sw_23_f32[4], sw_23_f32[5]);
            float2 v_2_12 = _f2_188;
            float2 _f2_189 = make_float2(q[20], q[21]);
            float2 qp_3_12 = _f2_189;
            sq[2] = fma_f32x2_rn_noftz(v_2_12, v_2_12, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_12, qp_3_12, dot[2]);
            float2 _f2_190 = make_float2(sw_23_f32[6], sw_23_f32[7]);
            float2 v_4_12 = _f2_190;
            float2 _f2_191 = make_float2(q[22], q[23]);
            float2 qp_5_12 = _f2_191;
            sq[2] = fma_f32x2_rn_noftz(v_4_12, v_4_12, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_12, qp_5_12, dot[2]);
        }
        unsigned int sw_24[4];
        sw_24[0] = words[42 + woff_20];
        sw_24[1] = words[42 + woff_20 + 1];
        sw_24[2] = words[42 + woff_20 + 2];
        sw_24[3] = words[42 + woff_20 + 3];
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
            float2 _f2_198 = make_float2(sw_24_f32[0], sw_24_f32[1]);
            float2 v_15 = _f2_198;
            float2 _f2_199 = make_float2(q[16], q[17]);
            float2 qp_16 = _f2_199;
            sq[3] = fma_f32x2_rn_noftz(v_15, v_15, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_15, qp_16, dot[3]);
            float2 _f2_200 = make_float2(sw_24_f32[2], sw_24_f32[3]);
            float2 v_0_13 = _f2_200;
            float2 _f2_201 = make_float2(q[18], q[19]);
            float2 qp_1_13 = _f2_201;
            sq[3] = fma_f32x2_rn_noftz(v_0_13, v_0_13, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_13, qp_1_13, dot[3]);
            float2 _f2_202 = make_float2(sw_24_f32[4], sw_24_f32[5]);
            float2 v_2_13 = _f2_202;
            float2 _f2_203 = make_float2(q[20], q[21]);
            float2 qp_3_13 = _f2_203;
            sq[3] = fma_f32x2_rn_noftz(v_2_13, v_2_13, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_13, qp_3_13, dot[3]);
            float2 _f2_204 = make_float2(sw_24_f32[6], sw_24_f32[7]);
            float2 v_4_13 = _f2_204;
            float2 _f2_205 = make_float2(q[22], q[23]);
            float2 qp_5_13 = _f2_205;
            sq[3] = fma_f32x2_rn_noftz(v_4_13, v_4_13, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_13, qp_5_13, dot[3]);
        }
        unsigned int sw_25[4];
        sw_25[0] = words[56 + woff_20];
        sw_25[1] = words[56 + woff_20 + 1];
        sw_25[2] = words[56 + woff_20 + 2];
        sw_25[3] = words[56 + woff_20 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_25[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_20]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_25[0] = __as_u32(mixed_3);
            words[56 + woff_20] = sw_25[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_25[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_20 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_25[1] = __as_u32(mixed_2_2);
            words[56 + woff_20 + 1] = sw_25[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_25[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_20 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_25[2] = __as_u32(mixed_5_2);
            words[56 + woff_20 + 2] = sw_25[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_25[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_20 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_25[3] = __as_u32(mixed_8_2);
            words[56 + woff_20 + 3] = sw_25[3];
            {
                int4 _iv4 = make_int4(sw_25[0 + 0], sw_25[0 + 1], sw_25[0 + 2], sw_25[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_19)) + 0) = _iv4;
            }
            unsigned long long write_base_2 = block_base + 4 * blocks_k_stride + (unsigned long long)base_19;
            {
                int4 _iv4 = make_int4(sw_25[0 + 0], sw_25[0 + 1], sw_25[0 + 2], sw_25[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_2) + 0) = _iv4;
            }
        }
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
            float2 _f2_212 = make_float2(sw_25_f32[0], sw_25_f32[1]);
            float2 v_16 = _f2_212;
            float2 _f2_213 = make_float2(q[16], q[17]);
            float2 qp_17 = _f2_213;
            sq[4] = fma_f32x2_rn_noftz(v_16, v_16, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_16, qp_17, dot[4]);
            float2 _f2_214 = make_float2(sw_25_f32[2], sw_25_f32[3]);
            float2 v_0_14 = _f2_214;
            float2 _f2_215 = make_float2(q[18], q[19]);
            float2 qp_1_14 = _f2_215;
            sq[4] = fma_f32x2_rn_noftz(v_0_14, v_0_14, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_14, qp_1_14, dot[4]);
            float2 _f2_216 = make_float2(sw_25_f32[4], sw_25_f32[5]);
            float2 v_2_14 = _f2_216;
            float2 _f2_217 = make_float2(q[20], q[21]);
            float2 qp_3_14 = _f2_217;
            sq[4] = fma_f32x2_rn_noftz(v_2_14, v_2_14, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_14, qp_3_14, dot[4]);
            float2 _f2_218 = make_float2(sw_25_f32[6], sw_25_f32[7]);
            float2 v_4_14 = _f2_218;
            float2 _f2_219 = make_float2(q[22], q[23]);
            float2 qp_5_14 = _f2_219;
            sq[4] = fma_f32x2_rn_noftz(v_4_14, v_4_14, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_14, qp_5_14, dot[4]);
        }
        int base_26 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_27 = 12;
        unsigned int sw_28[4];
        sw_28[0] = words[woff_27];
        sw_28[1] = words[woff_27 + 1];
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
            fsrc[24] = sw_28_f32[0];
            fsrc[25] = sw_28_f32[1];
            fsrc[26] = sw_28_f32[2];
            fsrc[27] = sw_28_f32[3];
        }
        {
            float2 _f2_220 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_17 = _f2_220;
            sq[0] = fma_f32x2_rn_noftz(v_17, v_17, sq[0]);
            float2 _f2_221 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_0_15 = _f2_221;
            sq[0] = fma_f32x2_rn_noftz(v_0_15, v_0_15, sq[0]);
            float2 _f2_222 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_1_1 = _f2_222;
            float2 _f2_223 = make_float2(q[24], q[25]);
            float2 qp_18 = _f2_223;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_18, dot[0]);
            float2 _f2_224 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_2_15 = _f2_224;
            float2 _f2_225 = make_float2(q[26], q[27]);
            float2 qp_3_15 = _f2_225;
            dot[0] = fma_f32x2_rn_noftz(v_2_15, qp_3_15, dot[0]);
        }
        unsigned int sw_29[4];
        sw_29[0] = words[14 + woff_27];
        sw_29[1] = words[14 + woff_27 + 1];
        float sw_29_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_29_f32[_pair * 2])[0]), "=f"((&sw_29_f32[_pair * 2])[1])
                : "r"(sw_29[_pair]));
        }
        {
            fsrc[52] = sw_29_f32[0];
            fsrc[53] = sw_29_f32[1];
            fsrc[54] = sw_29_f32[2];
            fsrc[55] = sw_29_f32[3];
        }
        {
            float2 _f2_234 = make_float2(sw_29_f32[0], sw_29_f32[1]);
            float2 v_18 = _f2_234;
            sq[1] = fma_f32x2_rn_noftz(v_18, v_18, sq[1]);
            float2 _f2_235 = make_float2(sw_29_f32[2], sw_29_f32[3]);
            float2 v_0_16 = _f2_235;
            sq[1] = fma_f32x2_rn_noftz(v_0_16, v_0_16, sq[1]);
            float2 _f2_236 = make_float2(sw_29_f32[0], sw_29_f32[1]);
            float2 v_1_2 = _f2_236;
            float2 _f2_237 = make_float2(q[24], q[25]);
            float2 qp_19 = _f2_237;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_19, dot[1]);
            float2 _f2_238 = make_float2(sw_29_f32[2], sw_29_f32[3]);
            float2 v_2_16 = _f2_238;
            float2 _f2_239 = make_float2(q[26], q[27]);
            float2 qp_3_16 = _f2_239;
            dot[1] = fma_f32x2_rn_noftz(v_2_16, qp_3_16, dot[1]);
        }
        unsigned int sw_30[4];
        sw_30[0] = words[28 + woff_27];
        sw_30[1] = words[28 + woff_27 + 1];
        float sw_30_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_30_f32[_pair * 2])[0]), "=f"((&sw_30_f32[_pair * 2])[1])
                : "r"(sw_30[_pair]));
        }
        {
            fsrc[80] = sw_30_f32[0];
            fsrc[81] = sw_30_f32[1];
            fsrc[82] = sw_30_f32[2];
            fsrc[83] = sw_30_f32[3];
        }
        {
            float2 _f2_248 = make_float2(sw_30_f32[0], sw_30_f32[1]);
            float2 v_19 = _f2_248;
            sq[2] = fma_f32x2_rn_noftz(v_19, v_19, sq[2]);
            float2 _f2_249 = make_float2(sw_30_f32[2], sw_30_f32[3]);
            float2 v_0_17 = _f2_249;
            sq[2] = fma_f32x2_rn_noftz(v_0_17, v_0_17, sq[2]);
            float2 _f2_250 = make_float2(sw_30_f32[0], sw_30_f32[1]);
            float2 v_1_3 = _f2_250;
            float2 _f2_251 = make_float2(q[24], q[25]);
            float2 qp_20 = _f2_251;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_20, dot[2]);
            float2 _f2_252 = make_float2(sw_30_f32[2], sw_30_f32[3]);
            float2 v_2_17 = _f2_252;
            float2 _f2_253 = make_float2(q[26], q[27]);
            float2 qp_3_17 = _f2_253;
            dot[2] = fma_f32x2_rn_noftz(v_2_17, qp_3_17, dot[2]);
        }
        unsigned int sw_31[4];
        sw_31[0] = words[42 + woff_27];
        sw_31[1] = words[42 + woff_27 + 1];
        float sw_31_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_31_f32[_pair * 2])[0]), "=f"((&sw_31_f32[_pair * 2])[1])
                : "r"(sw_31[_pair]));
        }
        {
            float2 _f2_262 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_20 = _f2_262;
            sq[3] = fma_f32x2_rn_noftz(v_20, v_20, sq[3]);
            float2 _f2_263 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_0_18 = _f2_263;
            sq[3] = fma_f32x2_rn_noftz(v_0_18, v_0_18, sq[3]);
            float2 _f2_264 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_1_4 = _f2_264;
            float2 _f2_265 = make_float2(q[24], q[25]);
            float2 qp_21 = _f2_265;
            dot[3] = fma_f32x2_rn_noftz(v_1_4, qp_21, dot[3]);
            float2 _f2_266 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_2_18 = _f2_266;
            float2 _f2_267 = make_float2(q[26], q[27]);
            float2 qp_3_18 = _f2_267;
            dot[3] = fma_f32x2_rn_noftz(v_2_18, qp_3_18, dot[3]);
        }
        unsigned int sw_32[4];
        sw_32[0] = words[56 + woff_27];
        sw_32[1] = words[56 + woff_27 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_32[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_27]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_32[0] = __as_u32(mixed_4);
            words[56 + woff_27] = sw_32[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_32[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_27 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_32[1] = __as_u32(mixed_2_3);
            words[56 + woff_27 + 1] = sw_32[1];
            {
                int2 _iv2 = make_int2(sw_32[0 + 0], sw_32[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_26)) + 0) = _iv2;
            }
            unsigned long long write_base_3 = block_base + 4 * blocks_k_stride + (unsigned long long)base_26;
            {
                int2 _iv2 = make_int2(sw_32[0 + 0], sw_32[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(blocks + write_base_3) + 0) = _iv2;
            }
        }
        float sw_32_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_32_f32[_pair * 2])[0]), "=f"((&sw_32_f32[_pair * 2])[1])
                : "r"(sw_32[_pair]));
        }
        {
            float2 _f2_276 = make_float2(sw_32_f32[0], sw_32_f32[1]);
            float2 v_21 = _f2_276;
            sq[4] = fma_f32x2_rn_noftz(v_21, v_21, sq[4]);
            float2 _f2_277 = make_float2(sw_32_f32[2], sw_32_f32[3]);
            float2 v_0_19 = _f2_277;
            sq[4] = fma_f32x2_rn_noftz(v_0_19, v_0_19, sq[4]);
            float2 _f2_278 = make_float2(sw_32_f32[0], sw_32_f32[1]);
            float2 v_1_5 = _f2_278;
            float2 _f2_279 = make_float2(q[24], q[25]);
            float2 qp_22 = _f2_279;
            dot[4] = fma_f32x2_rn_noftz(v_1_5, qp_22, dot[4]);
            float2 _f2_280 = make_float2(sw_32_f32[2], sw_32_f32[3]);
            float2 v_2_19 = _f2_280;
            float2 _f2_281 = make_float2(q[26], q[27]);
            float2 qp_3_19 = _f2_281;
            dot[4] = fma_f32x2_rn_noftz(v_2_19, qp_3_19, dot[4]);
        }
        float2 pairs[5];
        float2 _f2_290 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_290;
        float2 _f2_291 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_291;
        float2 _f2_292 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_292;
        float2 _f2_293 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_293;
        float2 _f2_294 = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[4] = _f2_294;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_295 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_295;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_33 = 0;
        bits_33 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_33, 16);
        unsigned long long peerbits_34 = _shfl_xor_1;
        float2 _f2_296 = make_float2(0.0f, 0.0f);
        float2 peer_35 = _f2_296;
        peer_35 = reinterpret_cast<float2*>(&peerbits_34)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_35);
        unsigned long long bits_36 = 0;
        bits_36 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_36, 16);
        unsigned long long peerbits_37 = _shfl_xor_2;
        float2 _f2_297 = make_float2(0.0f, 0.0f);
        float2 peer_38 = _f2_297;
        peer_38 = reinterpret_cast<float2*>(&peerbits_37)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_38);
        unsigned long long bits_39 = 0;
        bits_39 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_39, 16);
        unsigned long long peerbits_40 = _shfl_xor_3;
        float2 _f2_298 = make_float2(0.0f, 0.0f);
        float2 peer_41 = _f2_298;
        peer_41 = reinterpret_cast<float2*>(&peerbits_40)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_41);
        unsigned long long bits_42 = 0;
        bits_42 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_42, 16);
        unsigned long long peerbits_43 = _shfl_xor_4;
        float2 _f2_299 = make_float2(0.0f, 0.0f);
        float2 peer_44 = _f2_299;
        peer_44 = reinterpret_cast<float2*>(&peerbits_43)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_44);
        unsigned long long bits_45 = 0;
        bits_45 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_45, 8);
        unsigned long long peerbits_46 = _shfl_xor_5;
        float2 _f2_300 = make_float2(0.0f, 0.0f);
        float2 peer_47 = _f2_300;
        peer_47 = reinterpret_cast<float2*>(&peerbits_46)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_47);
        unsigned long long bits_48 = 0;
        bits_48 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_48, 8);
        unsigned long long peerbits_49 = _shfl_xor_6;
        float2 _f2_301 = make_float2(0.0f, 0.0f);
        float2 peer_50 = _f2_301;
        peer_50 = reinterpret_cast<float2*>(&peerbits_49)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_50);
        unsigned long long bits_51 = 0;
        bits_51 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_51, 8);
        unsigned long long peerbits_52 = _shfl_xor_7;
        float2 _f2_302 = make_float2(0.0f, 0.0f);
        float2 peer_53 = _f2_302;
        peer_53 = reinterpret_cast<float2*>(&peerbits_52)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_53);
        unsigned long long bits_54 = 0;
        bits_54 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_54, 8);
        unsigned long long peerbits_55 = _shfl_xor_8;
        float2 _f2_303 = make_float2(0.0f, 0.0f);
        float2 peer_56 = _f2_303;
        peer_56 = reinterpret_cast<float2*>(&peerbits_55)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_56);
        unsigned long long bits_57 = 0;
        bits_57 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_57, 8);
        unsigned long long peerbits_58 = _shfl_xor_9;
        float2 _f2_304 = make_float2(0.0f, 0.0f);
        float2 peer_59 = _f2_304;
        peer_59 = reinterpret_cast<float2*>(&peerbits_58)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_59);
        unsigned long long bits_60 = 0;
        bits_60 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_60, 4);
        unsigned long long peerbits_61 = _shfl_xor_10;
        float2 _f2_305 = make_float2(0.0f, 0.0f);
        float2 peer_62 = _f2_305;
        peer_62 = reinterpret_cast<float2*>(&peerbits_61)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_62);
        unsigned long long bits_63 = 0;
        bits_63 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_63, 4);
        unsigned long long peerbits_64 = _shfl_xor_11;
        float2 _f2_306 = make_float2(0.0f, 0.0f);
        float2 peer_65 = _f2_306;
        peer_65 = reinterpret_cast<float2*>(&peerbits_64)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_65);
        unsigned long long bits_66 = 0;
        bits_66 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_66, 4);
        unsigned long long peerbits_67 = _shfl_xor_12;
        float2 _f2_307 = make_float2(0.0f, 0.0f);
        float2 peer_68 = _f2_307;
        peer_68 = reinterpret_cast<float2*>(&peerbits_67)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_68);
        unsigned long long bits_69 = 0;
        bits_69 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_69, 4);
        unsigned long long peerbits_70 = _shfl_xor_13;
        float2 _f2_308 = make_float2(0.0f, 0.0f);
        float2 peer_71 = _f2_308;
        peer_71 = reinterpret_cast<float2*>(&peerbits_70)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_71);
        unsigned long long bits_72 = 0;
        bits_72 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_72, 4);
        unsigned long long peerbits_73 = _shfl_xor_14;
        float2 _f2_309 = make_float2(0.0f, 0.0f);
        float2 peer_74 = _f2_309;
        peer_74 = reinterpret_cast<float2*>(&peerbits_73)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_74);
        unsigned long long bits_75 = 0;
        bits_75 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_75, 2);
        unsigned long long peerbits_76 = _shfl_xor_15;
        float2 _f2_310 = make_float2(0.0f, 0.0f);
        float2 peer_77 = _f2_310;
        peer_77 = reinterpret_cast<float2*>(&peerbits_76)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_77);
        unsigned long long bits_78 = 0;
        bits_78 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_78, 2);
        unsigned long long peerbits_79 = _shfl_xor_16;
        float2 _f2_311 = make_float2(0.0f, 0.0f);
        float2 peer_80 = _f2_311;
        peer_80 = reinterpret_cast<float2*>(&peerbits_79)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_80);
        unsigned long long bits_81 = 0;
        bits_81 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_81, 2);
        unsigned long long peerbits_82 = _shfl_xor_17;
        float2 _f2_312 = make_float2(0.0f, 0.0f);
        float2 peer_83 = _f2_312;
        peer_83 = reinterpret_cast<float2*>(&peerbits_82)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_83);
        unsigned long long bits_84 = 0;
        bits_84 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_84, 2);
        unsigned long long peerbits_85 = _shfl_xor_18;
        float2 _f2_313 = make_float2(0.0f, 0.0f);
        float2 peer_86 = _f2_313;
        peer_86 = reinterpret_cast<float2*>(&peerbits_85)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_86);
        unsigned long long bits_87 = 0;
        bits_87 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_87, 2);
        unsigned long long peerbits_88 = _shfl_xor_19;
        float2 _f2_314 = make_float2(0.0f, 0.0f);
        float2 peer_89 = _f2_314;
        peer_89 = reinterpret_cast<float2*>(&peerbits_88)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_89);
        unsigned long long bits_90 = 0;
        bits_90 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, bits_90, 1);
        unsigned long long peerbits_91 = _shfl_xor_20;
        float2 _f2_315 = make_float2(0.0f, 0.0f);
        float2 peer_92 = _f2_315;
        peer_92 = reinterpret_cast<float2*>(&peerbits_91)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_92);
        unsigned long long bits_93 = 0;
        bits_93 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, bits_93, 1);
        unsigned long long peerbits_94 = _shfl_xor_21;
        float2 _f2_316 = make_float2(0.0f, 0.0f);
        float2 peer_95 = _f2_316;
        peer_95 = reinterpret_cast<float2*>(&peerbits_94)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_95);
        unsigned long long bits_96 = 0;
        bits_96 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, bits_96, 1);
        unsigned long long peerbits_97 = _shfl_xor_22;
        float2 _f2_317 = make_float2(0.0f, 0.0f);
        float2 peer_98 = _f2_317;
        peer_98 = reinterpret_cast<float2*>(&peerbits_97)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_98);
        unsigned long long bits_99 = 0;
        bits_99 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, bits_99, 1);
        unsigned long long peerbits_100 = _shfl_xor_23;
        float2 _f2_318 = make_float2(0.0f, 0.0f);
        float2 peer_101 = _f2_318;
        peer_101 = reinterpret_cast<float2*>(&peerbits_100)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_101);
        unsigned long long bits_102 = 0;
        bits_102 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, bits_102, 1);
        unsigned long long peerbits_103 = _shfl_xor_24;
        float2 _f2_319 = make_float2(0.0f, 0.0f);
        float2 peer_104 = _f2_319;
        peer_104 = reinterpret_cast<float2*>(&peerbits_103)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_104);
        if (lane == 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(stats_addr + (unsigned int)(warp_0 * 5 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_0), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(stats_addr + (unsigned int)((warp_0 * 5 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_1), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_2), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_3), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_4) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_4), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_5;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_5) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_5), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_6) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_6), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_7;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_7) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_7), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_8) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_8), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_9) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_9), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_10) : "r"(stats_addr + (unsigned int)(warp_0 * 5 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_10), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_11) : "r"(stats_addr + (unsigned int)((warp_0 * 5 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_11), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_12;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_12) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_12), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_13;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_13) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_13), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_14;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_14) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_14), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_15;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_15) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_15), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_16) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_16), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_17;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_17) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_17), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_18) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_18), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_19;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_19) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_19), "f"(pairs[4].y) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 5) {
            total_sq = stats[(stat_w * 5 + stat_n) * 2];
            total_dot = stats[(stat_w * 5 + stat_n) * 2 + 1];
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
        if (stat_n < 5 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[5];
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
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 1) {
            total_sq2 = stats[(stat_w * 5 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 5 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_6;
        float _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_7;
        float _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_8;
        float _shfl_down_9 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_9;
        float _shfl_down_10 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_10;
        float _shfl_down_11 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_11;
        logit2 = 0.0f;
        if (stat_n < 1 && stat_w == 0) {
            float _rsqrt_1 = rsqrtf(total_sq2 / 7168.0f + eps);
            float sigma2 = _rsqrt_1;
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_4;
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
        float weights[5];
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
        float2 _f2_320 = make_float2(correction, correction);
        float2 corr = _f2_320;
        const int woff_105 = 0;
        int base_106 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_321 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_321;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_322 = make_float2(acc[2], acc[3]);
        float2 previous_107 = _f2_322;
        a_5[1] = mul_f32x2_noftz(previous_107, corr);
        float2 _f2_323 = make_float2(acc[4], acc[5]);
        float2 previous_108 = _f2_323;
        a_5[2] = mul_f32x2_noftz(previous_108, corr);
        float2 _f2_324 = make_float2(acc[6], acc[7]);
        float2 previous_109 = _f2_324;
        a_5[3] = mul_f32x2_noftz(previous_109, corr);
        float2 _f2_325 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_325;
        {
            float2 _f2_326 = make_float2(fsrc[0], fsrc[1]);
            float2 v_22 = _f2_326;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_22, a_5[0]);
            float2 _f2_327 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_20 = _f2_327;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_20, a_5[1]);
            float2 _f2_328 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_6 = _f2_328;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_6, a_5[2]);
            float2 _f2_329 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_20 = _f2_329;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_20, a_5[3]);
        }
        float2 _f2_334 = make_float2(weights[1], weights[1]);
        float2 weight_110 = _f2_334;
        {
            float2 _f2_335 = make_float2(fsrc[28], fsrc[29]);
            float2 v_23 = _f2_335;
            a_5[0] = fma_f32x2_rn_noftz(weight_110, v_23, a_5[0]);
            float2 _f2_336 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_21 = _f2_336;
            a_5[1] = fma_f32x2_rn_noftz(weight_110, v_0_21, a_5[1]);
            float2 _f2_337 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_7 = _f2_337;
            a_5[2] = fma_f32x2_rn_noftz(weight_110, v_1_7, a_5[2]);
            float2 _f2_338 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_21 = _f2_338;
            a_5[3] = fma_f32x2_rn_noftz(weight_110, v_2_21, a_5[3]);
        }
        float2 _f2_343 = make_float2(weights[2], weights[2]);
        float2 weight_111 = _f2_343;
        {
            float2 _f2_344 = make_float2(fsrc[56], fsrc[57]);
            float2 v_24 = _f2_344;
            a_5[0] = fma_f32x2_rn_noftz(weight_111, v_24, a_5[0]);
            float2 _f2_345 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_22 = _f2_345;
            a_5[1] = fma_f32x2_rn_noftz(weight_111, v_0_22, a_5[1]);
            float2 _f2_346 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_8 = _f2_346;
            a_5[2] = fma_f32x2_rn_noftz(weight_111, v_1_8, a_5[2]);
            float2 _f2_347 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_22 = _f2_347;
            a_5[3] = fma_f32x2_rn_noftz(weight_111, v_2_22, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_112 = 4;
        int base_113 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_114[4];
        float2 _f2_352 = make_float2(acc[8], acc[9]);
        float2 previous_115 = _f2_352;
        a_114[0] = mul_f32x2_noftz(previous_115, corr);
        float2 _f2_353 = make_float2(acc[10], acc[11]);
        float2 previous_116 = _f2_353;
        a_114[1] = mul_f32x2_noftz(previous_116, corr);
        float2 _f2_354 = make_float2(acc[12], acc[13]);
        float2 previous_117 = _f2_354;
        a_114[2] = mul_f32x2_noftz(previous_117, corr);
        float2 _f2_355 = make_float2(acc[14], acc[15]);
        float2 previous_118 = _f2_355;
        a_114[3] = mul_f32x2_noftz(previous_118, corr);
        float2 _f2_356 = make_float2(weights[0], weights[0]);
        float2 weight_119 = _f2_356;
        {
            float2 _f2_357 = make_float2(fsrc[8], fsrc[9]);
            float2 v_25 = _f2_357;
            a_114[0] = fma_f32x2_rn_noftz(weight_119, v_25, a_114[0]);
            float2 _f2_358 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_23 = _f2_358;
            a_114[1] = fma_f32x2_rn_noftz(weight_119, v_0_23, a_114[1]);
            float2 _f2_359 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_9 = _f2_359;
            a_114[2] = fma_f32x2_rn_noftz(weight_119, v_1_9, a_114[2]);
            float2 _f2_360 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_23 = _f2_360;
            a_114[3] = fma_f32x2_rn_noftz(weight_119, v_2_23, a_114[3]);
        }
        float2 _f2_365 = make_float2(weights[1], weights[1]);
        float2 weight_120 = _f2_365;
        {
            float2 _f2_366 = make_float2(fsrc[36], fsrc[37]);
            float2 v_26 = _f2_366;
            a_114[0] = fma_f32x2_rn_noftz(weight_120, v_26, a_114[0]);
            float2 _f2_367 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_24 = _f2_367;
            a_114[1] = fma_f32x2_rn_noftz(weight_120, v_0_24, a_114[1]);
            float2 _f2_368 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_10 = _f2_368;
            a_114[2] = fma_f32x2_rn_noftz(weight_120, v_1_10, a_114[2]);
            float2 _f2_369 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_24 = _f2_369;
            a_114[3] = fma_f32x2_rn_noftz(weight_120, v_2_24, a_114[3]);
        }
        float2 _f2_374 = make_float2(weights[2], weights[2]);
        float2 weight_121 = _f2_374;
        {
            float2 _f2_375 = make_float2(fsrc[64], fsrc[65]);
            float2 v_27 = _f2_375;
            a_114[0] = fma_f32x2_rn_noftz(weight_121, v_27, a_114[0]);
            float2 _f2_376 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_25 = _f2_376;
            a_114[1] = fma_f32x2_rn_noftz(weight_121, v_0_25, a_114[1]);
            float2 _f2_377 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_11 = _f2_377;
            a_114[2] = fma_f32x2_rn_noftz(weight_121, v_1_11, a_114[2]);
            float2 _f2_378 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_25 = _f2_378;
            a_114[3] = fma_f32x2_rn_noftz(weight_121, v_2_25, a_114[3]);
        }
        acc[8] = a_114[0].x;
        acc[9] = a_114[0].y;
        acc[10] = a_114[1].x;
        acc[11] = a_114[1].y;
        acc[12] = a_114[2].x;
        acc[13] = a_114[2].y;
        acc[14] = a_114[3].x;
        acc[15] = a_114[3].y;
        const int woff_122 = 8;
        int base_123 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_124[4];
        float2 _f2_383 = make_float2(acc[16], acc[17]);
        float2 previous_125 = _f2_383;
        a_124[0] = mul_f32x2_noftz(previous_125, corr);
        float2 _f2_384 = make_float2(acc[18], acc[19]);
        float2 previous_126 = _f2_384;
        a_124[1] = mul_f32x2_noftz(previous_126, corr);
        float2 _f2_385 = make_float2(acc[20], acc[21]);
        float2 previous_127 = _f2_385;
        a_124[2] = mul_f32x2_noftz(previous_127, corr);
        float2 _f2_386 = make_float2(acc[22], acc[23]);
        float2 previous_128 = _f2_386;
        a_124[3] = mul_f32x2_noftz(previous_128, corr);
        float2 _f2_387 = make_float2(weights[0], weights[0]);
        float2 weight_129 = _f2_387;
        {
            float2 _f2_388 = make_float2(fsrc[16], fsrc[17]);
            float2 v_28 = _f2_388;
            a_124[0] = fma_f32x2_rn_noftz(weight_129, v_28, a_124[0]);
            float2 _f2_389 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_26 = _f2_389;
            a_124[1] = fma_f32x2_rn_noftz(weight_129, v_0_26, a_124[1]);
            float2 _f2_390 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_12 = _f2_390;
            a_124[2] = fma_f32x2_rn_noftz(weight_129, v_1_12, a_124[2]);
            float2 _f2_391 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_26 = _f2_391;
            a_124[3] = fma_f32x2_rn_noftz(weight_129, v_2_26, a_124[3]);
        }
        float2 _f2_396 = make_float2(weights[1], weights[1]);
        float2 weight_130 = _f2_396;
        {
            float2 _f2_397 = make_float2(fsrc[44], fsrc[45]);
            float2 v_29 = _f2_397;
            a_124[0] = fma_f32x2_rn_noftz(weight_130, v_29, a_124[0]);
            float2 _f2_398 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_27 = _f2_398;
            a_124[1] = fma_f32x2_rn_noftz(weight_130, v_0_27, a_124[1]);
            float2 _f2_399 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_13 = _f2_399;
            a_124[2] = fma_f32x2_rn_noftz(weight_130, v_1_13, a_124[2]);
            float2 _f2_400 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_27 = _f2_400;
            a_124[3] = fma_f32x2_rn_noftz(weight_130, v_2_27, a_124[3]);
        }
        float2 _f2_405 = make_float2(weights[2], weights[2]);
        float2 weight_131 = _f2_405;
        {
            float2 _f2_406 = make_float2(fsrc[72], fsrc[73]);
            float2 v_30 = _f2_406;
            a_124[0] = fma_f32x2_rn_noftz(weight_131, v_30, a_124[0]);
            float2 _f2_407 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_28 = _f2_407;
            a_124[1] = fma_f32x2_rn_noftz(weight_131, v_0_28, a_124[1]);
            float2 _f2_408 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_14 = _f2_408;
            a_124[2] = fma_f32x2_rn_noftz(weight_131, v_1_14, a_124[2]);
            float2 _f2_409 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_28 = _f2_409;
            a_124[3] = fma_f32x2_rn_noftz(weight_131, v_2_28, a_124[3]);
        }
        acc[16] = a_124[0].x;
        acc[17] = a_124[0].y;
        acc[18] = a_124[1].x;
        acc[19] = a_124[1].y;
        acc[20] = a_124[2].x;
        acc[21] = a_124[2].y;
        acc[22] = a_124[3].x;
        acc[23] = a_124[3].y;
        const int woff_132 = 12;
        int base_133 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_134[4];
        float2 _f2_414 = make_float2(acc[24], acc[25]);
        float2 previous_135 = _f2_414;
        a_134[0] = mul_f32x2_noftz(previous_135, corr);
        float2 _f2_415 = make_float2(acc[26], acc[27]);
        float2 previous_136 = _f2_415;
        a_134[1] = mul_f32x2_noftz(previous_136, corr);
        float2 _f2_416 = make_float2(weights[0], weights[0]);
        float2 weight_137 = _f2_416;
        {
            float2 _f2_417 = make_float2(fsrc[24], fsrc[25]);
            float2 v_31 = _f2_417;
            a_134[0] = fma_f32x2_rn_noftz(weight_137, v_31, a_134[0]);
            float2 _f2_418 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_29 = _f2_418;
            a_134[1] = fma_f32x2_rn_noftz(weight_137, v_0_29, a_134[1]);
        }
        float2 _f2_421 = make_float2(weights[1], weights[1]);
        float2 weight_138 = _f2_421;
        {
            float2 _f2_422 = make_float2(fsrc[52], fsrc[53]);
            float2 v_32 = _f2_422;
            a_134[0] = fma_f32x2_rn_noftz(weight_138, v_32, a_134[0]);
            float2 _f2_423 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_30 = _f2_423;
            a_134[1] = fma_f32x2_rn_noftz(weight_138, v_0_30, a_134[1]);
        }
        float2 _f2_426 = make_float2(weights[2], weights[2]);
        float2 weight_139 = _f2_426;
        {
            float2 _f2_427 = make_float2(fsrc[80], fsrc[81]);
            float2 v_33 = _f2_427;
            a_134[0] = fma_f32x2_rn_noftz(weight_139, v_33, a_134[0]);
            float2 _f2_428 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_31 = _f2_428;
            a_134[1] = fma_f32x2_rn_noftz(weight_139, v_0_31, a_134[1]);
        }
        acc[24] = a_134[0].x;
        acc[25] = a_134[0].y;
        acc[26] = a_134[1].x;
        acc[27] = a_134[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float max_chunk_140 = -3.4028234663852886e+38f;
        float _fmax_4 = fmaxf(max_chunk_140, logits[3]);
        max_chunk_140 = _fmax_4;
        float _fmax_5 = fmaxf(max_chunk_140, logits[4]);
        max_chunk_140 = _fmax_5;
        float _fmax_6 = fmaxf(max_running, max_chunk_140);
        float max_new_141 = _fmax_6;
        float _exp2_4 = approx_exp2((max_running - max_new_141) * 1.4426950408889634f);
        float correction_142 = _exp2_4;
        float weights_143[5];
        float sum_weights_144 = 0.0f;
        float _exp2_5 = approx_exp2((logits[3] - max_new_141) * 1.4426950408889634f);
        weights_143[3] = _exp2_5;
        sum_weights_144 += weights_143[3];
        float _exp2_6 = approx_exp2((logits[4] - max_new_141) * 1.4426950408889634f);
        weights_143[4] = _exp2_6;
        sum_weights_144 += weights_143[4];
        float2 _f2_431 = make_float2(correction_142, correction_142);
        float2 corr_145 = _f2_431;
        const int woff_146 = 0;
        int base_147 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_148[4];
        float2 _f2_432 = make_float2(acc[0], acc[1]);
        float2 previous_149 = _f2_432;
        a_148[0] = mul_f32x2_noftz(previous_149, corr_145);
        float2 _f2_433 = make_float2(acc[2], acc[3]);
        float2 previous_150 = _f2_433;
        a_148[1] = mul_f32x2_noftz(previous_150, corr_145);
        float2 _f2_434 = make_float2(acc[4], acc[5]);
        float2 previous_151 = _f2_434;
        a_148[2] = mul_f32x2_noftz(previous_151, corr_145);
        float2 _f2_435 = make_float2(acc[6], acc[7]);
        float2 previous_152 = _f2_435;
        a_148[3] = mul_f32x2_noftz(previous_152, corr_145);
        float2 _f2_436 = make_float2(weights_143[3], weights_143[3]);
        float2 weight_153 = _f2_436;
        {
            unsigned int sw2[4];
            sw2[0] = words[42 + woff_146];
            sw2[1] = words[42 + woff_146 + 1];
            sw2[2] = words[42 + woff_146 + 2];
            sw2[3] = words[42 + woff_146 + 3];
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
            float2 _f2_441 = make_float2(sw2_f32[0], sw2_f32[1]);
            float2 v_34 = _f2_441;
            a_148[0] = fma_f32x2_rn_noftz(weight_153, v_34, a_148[0]);
            float2 _f2_442 = make_float2(sw2_f32[2], sw2_f32[3]);
            float2 v_0_32 = _f2_442;
            a_148[1] = fma_f32x2_rn_noftz(weight_153, v_0_32, a_148[1]);
            float2 _f2_443 = make_float2(sw2_f32[4], sw2_f32[5]);
            float2 v_1_15 = _f2_443;
            a_148[2] = fma_f32x2_rn_noftz(weight_153, v_1_15, a_148[2]);
            float2 _f2_444 = make_float2(sw2_f32[6], sw2_f32[7]);
            float2 v_2_29 = _f2_444;
            a_148[3] = fma_f32x2_rn_noftz(weight_153, v_2_29, a_148[3]);
        }
        float2 _f2_445 = make_float2(weights_143[4], weights_143[4]);
        float2 weight_154 = _f2_445;
        {
            unsigned int sw2_1[4];
            sw2_1[0] = words[56 + woff_146];
            sw2_1[1] = words[56 + woff_146 + 1];
            sw2_1[2] = words[56 + woff_146 + 2];
            sw2_1[3] = words[56 + woff_146 + 3];
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
            float2 _f2_450 = make_float2(sw2_f32_1[0], sw2_f32_1[1]);
            float2 v_35 = _f2_450;
            a_148[0] = fma_f32x2_rn_noftz(weight_154, v_35, a_148[0]);
            float2 _f2_451 = make_float2(sw2_f32_1[2], sw2_f32_1[3]);
            float2 v_0_33 = _f2_451;
            a_148[1] = fma_f32x2_rn_noftz(weight_154, v_0_33, a_148[1]);
            float2 _f2_452 = make_float2(sw2_f32_1[4], sw2_f32_1[5]);
            float2 v_1_16 = _f2_452;
            a_148[2] = fma_f32x2_rn_noftz(weight_154, v_1_16, a_148[2]);
            float2 _f2_453 = make_float2(sw2_f32_1[6], sw2_f32_1[7]);
            float2 v_2_30 = _f2_453;
            a_148[3] = fma_f32x2_rn_noftz(weight_154, v_2_30, a_148[3]);
        }
        acc[0] = a_148[0].x;
        acc[1] = a_148[0].y;
        acc[2] = a_148[1].x;
        acc[3] = a_148[1].y;
        acc[4] = a_148[2].x;
        acc[5] = a_148[2].y;
        acc[6] = a_148[3].x;
        acc[7] = a_148[3].y;
        const int woff_155 = 4;
        int base_156 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_157[4];
        float2 _f2_454 = make_float2(acc[8], acc[9]);
        float2 previous_158 = _f2_454;
        a_157[0] = mul_f32x2_noftz(previous_158, corr_145);
        float2 _f2_455 = make_float2(acc[10], acc[11]);
        float2 previous_159 = _f2_455;
        a_157[1] = mul_f32x2_noftz(previous_159, corr_145);
        float2 _f2_456 = make_float2(acc[12], acc[13]);
        float2 previous_160 = _f2_456;
        a_157[2] = mul_f32x2_noftz(previous_160, corr_145);
        float2 _f2_457 = make_float2(acc[14], acc[15]);
        float2 previous_161 = _f2_457;
        a_157[3] = mul_f32x2_noftz(previous_161, corr_145);
        float2 _f2_458 = make_float2(weights_143[3], weights_143[3]);
        float2 weight_162 = _f2_458;
        {
            unsigned int sw2_2[4];
            sw2_2[0] = words[42 + woff_155];
            sw2_2[1] = words[42 + woff_155 + 1];
            sw2_2[2] = words[42 + woff_155 + 2];
            sw2_2[3] = words[42 + woff_155 + 3];
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
            float2 _f2_463 = make_float2(sw2_f32_2[0], sw2_f32_2[1]);
            float2 v_36 = _f2_463;
            a_157[0] = fma_f32x2_rn_noftz(weight_162, v_36, a_157[0]);
            float2 _f2_464 = make_float2(sw2_f32_2[2], sw2_f32_2[3]);
            float2 v_0_34 = _f2_464;
            a_157[1] = fma_f32x2_rn_noftz(weight_162, v_0_34, a_157[1]);
            float2 _f2_465 = make_float2(sw2_f32_2[4], sw2_f32_2[5]);
            float2 v_1_17 = _f2_465;
            a_157[2] = fma_f32x2_rn_noftz(weight_162, v_1_17, a_157[2]);
            float2 _f2_466 = make_float2(sw2_f32_2[6], sw2_f32_2[7]);
            float2 v_2_31 = _f2_466;
            a_157[3] = fma_f32x2_rn_noftz(weight_162, v_2_31, a_157[3]);
        }
        float2 _f2_467 = make_float2(weights_143[4], weights_143[4]);
        float2 weight_163 = _f2_467;
        {
            unsigned int sw2_3[4];
            sw2_3[0] = words[56 + woff_155];
            sw2_3[1] = words[56 + woff_155 + 1];
            sw2_3[2] = words[56 + woff_155 + 2];
            sw2_3[3] = words[56 + woff_155 + 3];
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
            float2 _f2_472 = make_float2(sw2_f32_3[0], sw2_f32_3[1]);
            float2 v_37 = _f2_472;
            a_157[0] = fma_f32x2_rn_noftz(weight_163, v_37, a_157[0]);
            float2 _f2_473 = make_float2(sw2_f32_3[2], sw2_f32_3[3]);
            float2 v_0_35 = _f2_473;
            a_157[1] = fma_f32x2_rn_noftz(weight_163, v_0_35, a_157[1]);
            float2 _f2_474 = make_float2(sw2_f32_3[4], sw2_f32_3[5]);
            float2 v_1_18 = _f2_474;
            a_157[2] = fma_f32x2_rn_noftz(weight_163, v_1_18, a_157[2]);
            float2 _f2_475 = make_float2(sw2_f32_3[6], sw2_f32_3[7]);
            float2 v_2_32 = _f2_475;
            a_157[3] = fma_f32x2_rn_noftz(weight_163, v_2_32, a_157[3]);
        }
        acc[8] = a_157[0].x;
        acc[9] = a_157[0].y;
        acc[10] = a_157[1].x;
        acc[11] = a_157[1].y;
        acc[12] = a_157[2].x;
        acc[13] = a_157[2].y;
        acc[14] = a_157[3].x;
        acc[15] = a_157[3].y;
        const int woff_164 = 8;
        int base_165 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_166[4];
        float2 _f2_476 = make_float2(acc[16], acc[17]);
        float2 previous_167 = _f2_476;
        a_166[0] = mul_f32x2_noftz(previous_167, corr_145);
        float2 _f2_477 = make_float2(acc[18], acc[19]);
        float2 previous_168 = _f2_477;
        a_166[1] = mul_f32x2_noftz(previous_168, corr_145);
        float2 _f2_478 = make_float2(acc[20], acc[21]);
        float2 previous_169 = _f2_478;
        a_166[2] = mul_f32x2_noftz(previous_169, corr_145);
        float2 _f2_479 = make_float2(acc[22], acc[23]);
        float2 previous_170 = _f2_479;
        a_166[3] = mul_f32x2_noftz(previous_170, corr_145);
        float2 _f2_480 = make_float2(weights_143[3], weights_143[3]);
        float2 weight_171 = _f2_480;
        {
            unsigned int sw2_4[4];
            sw2_4[0] = words[42 + woff_164];
            sw2_4[1] = words[42 + woff_164 + 1];
            sw2_4[2] = words[42 + woff_164 + 2];
            sw2_4[3] = words[42 + woff_164 + 3];
            float sw2_f32_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_4[_pair * 2])[0]), "=f"((&sw2_f32_4[_pair * 2])[1])
                    : "r"(sw2_4[_pair]));
            }
            float2 _f2_485 = make_float2(sw2_f32_4[0], sw2_f32_4[1]);
            float2 v_38 = _f2_485;
            a_166[0] = fma_f32x2_rn_noftz(weight_171, v_38, a_166[0]);
            float2 _f2_486 = make_float2(sw2_f32_4[2], sw2_f32_4[3]);
            float2 v_0_36 = _f2_486;
            a_166[1] = fma_f32x2_rn_noftz(weight_171, v_0_36, a_166[1]);
            float2 _f2_487 = make_float2(sw2_f32_4[4], sw2_f32_4[5]);
            float2 v_1_19 = _f2_487;
            a_166[2] = fma_f32x2_rn_noftz(weight_171, v_1_19, a_166[2]);
            float2 _f2_488 = make_float2(sw2_f32_4[6], sw2_f32_4[7]);
            float2 v_2_33 = _f2_488;
            a_166[3] = fma_f32x2_rn_noftz(weight_171, v_2_33, a_166[3]);
        }
        float2 _f2_489 = make_float2(weights_143[4], weights_143[4]);
        float2 weight_172 = _f2_489;
        {
            unsigned int sw2_5[4];
            sw2_5[0] = words[56 + woff_164];
            sw2_5[1] = words[56 + woff_164 + 1];
            sw2_5[2] = words[56 + woff_164 + 2];
            sw2_5[3] = words[56 + woff_164 + 3];
            float sw2_f32_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_5[_pair * 2])[0]), "=f"((&sw2_f32_5[_pair * 2])[1])
                    : "r"(sw2_5[_pair]));
            }
            float2 _f2_494 = make_float2(sw2_f32_5[0], sw2_f32_5[1]);
            float2 v_39 = _f2_494;
            a_166[0] = fma_f32x2_rn_noftz(weight_172, v_39, a_166[0]);
            float2 _f2_495 = make_float2(sw2_f32_5[2], sw2_f32_5[3]);
            float2 v_0_37 = _f2_495;
            a_166[1] = fma_f32x2_rn_noftz(weight_172, v_0_37, a_166[1]);
            float2 _f2_496 = make_float2(sw2_f32_5[4], sw2_f32_5[5]);
            float2 v_1_20 = _f2_496;
            a_166[2] = fma_f32x2_rn_noftz(weight_172, v_1_20, a_166[2]);
            float2 _f2_497 = make_float2(sw2_f32_5[6], sw2_f32_5[7]);
            float2 v_2_34 = _f2_497;
            a_166[3] = fma_f32x2_rn_noftz(weight_172, v_2_34, a_166[3]);
        }
        acc[16] = a_166[0].x;
        acc[17] = a_166[0].y;
        acc[18] = a_166[1].x;
        acc[19] = a_166[1].y;
        acc[20] = a_166[2].x;
        acc[21] = a_166[2].y;
        acc[22] = a_166[3].x;
        acc[23] = a_166[3].y;
        const int woff_173 = 12;
        int base_174 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_175[4];
        float2 _f2_498 = make_float2(acc[24], acc[25]);
        float2 previous_176 = _f2_498;
        a_175[0] = mul_f32x2_noftz(previous_176, corr_145);
        float2 _f2_499 = make_float2(acc[26], acc[27]);
        float2 previous_177 = _f2_499;
        a_175[1] = mul_f32x2_noftz(previous_177, corr_145);
        float2 _f2_500 = make_float2(weights_143[3], weights_143[3]);
        float2 weight_178 = _f2_500;
        {
            unsigned int sw2_6[4];
            sw2_6[0] = words[42 + woff_173];
            sw2_6[1] = words[42 + woff_173 + 1];
            float sw2_f32_6[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_6[_pair * 2])[0]), "=f"((&sw2_f32_6[_pair * 2])[1])
                    : "r"(sw2_6[_pair]));
            }
            float2 _f2_503 = make_float2(sw2_f32_6[0], sw2_f32_6[1]);
            float2 v_40 = _f2_503;
            a_175[0] = fma_f32x2_rn_noftz(weight_178, v_40, a_175[0]);
            float2 _f2_504 = make_float2(sw2_f32_6[2], sw2_f32_6[3]);
            float2 v_0_38 = _f2_504;
            a_175[1] = fma_f32x2_rn_noftz(weight_178, v_0_38, a_175[1]);
        }
        float2 _f2_505 = make_float2(weights_143[4], weights_143[4]);
        float2 weight_179 = _f2_505;
        {
            unsigned int sw2_7[4];
            sw2_7[0] = words[56 + woff_173];
            sw2_7[1] = words[56 + woff_173 + 1];
            float sw2_f32_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_7[_pair * 2])[0]), "=f"((&sw2_f32_7[_pair * 2])[1])
                    : "r"(sw2_7[_pair]));
            }
            float2 _f2_508 = make_float2(sw2_f32_7[0], sw2_f32_7[1]);
            float2 v_41 = _f2_508;
            a_175[0] = fma_f32x2_rn_noftz(weight_179, v_41, a_175[0]);
            float2 _f2_509 = make_float2(sw2_f32_7[2], sw2_f32_7[3]);
            float2 v_0_39 = _f2_509;
            a_175[1] = fma_f32x2_rn_noftz(weight_179, v_0_39, a_175[1]);
        }
        acc[24] = a_175[0].x;
        acc[25] = a_175[0].y;
        acc[26] = a_175[1].x;
        acc[27] = a_175[1].y;
        sum_running = sum_running * correction_142 + sum_weights_144;
        max_running = max_new_141;
        float2 _f2_510 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_510;
        float2 _f2_511 = make_float2(acc[0], acc[1]);
        float2 v_42 = _f2_511;
        output_sq_pair = fma_f32x2_rn_noftz(v_42, v_42, output_sq_pair);
        float2 _f2_512 = make_float2(acc[2], acc[3]);
        float2 v_180 = _f2_512;
        output_sq_pair = fma_f32x2_rn_noftz(v_180, v_180, output_sq_pair);
        float2 _f2_513 = make_float2(acc[4], acc[5]);
        float2 v_181 = _f2_513;
        output_sq_pair = fma_f32x2_rn_noftz(v_181, v_181, output_sq_pair);
        float2 _f2_514 = make_float2(acc[6], acc[7]);
        float2 v_182 = _f2_514;
        output_sq_pair = fma_f32x2_rn_noftz(v_182, v_182, output_sq_pair);
        float2 _f2_515 = make_float2(acc[8], acc[9]);
        float2 v_183 = _f2_515;
        output_sq_pair = fma_f32x2_rn_noftz(v_183, v_183, output_sq_pair);
        float2 _f2_516 = make_float2(acc[10], acc[11]);
        float2 v_184 = _f2_516;
        output_sq_pair = fma_f32x2_rn_noftz(v_184, v_184, output_sq_pair);
        float2 _f2_517 = make_float2(acc[12], acc[13]);
        float2 v_185 = _f2_517;
        output_sq_pair = fma_f32x2_rn_noftz(v_185, v_185, output_sq_pair);
        float2 _f2_518 = make_float2(acc[14], acc[15]);
        float2 v_186 = _f2_518;
        output_sq_pair = fma_f32x2_rn_noftz(v_186, v_186, output_sq_pair);
        float2 _f2_519 = make_float2(acc[16], acc[17]);
        float2 v_187 = _f2_519;
        output_sq_pair = fma_f32x2_rn_noftz(v_187, v_187, output_sq_pair);
        float2 _f2_520 = make_float2(acc[18], acc[19]);
        float2 v_188 = _f2_520;
        output_sq_pair = fma_f32x2_rn_noftz(v_188, v_188, output_sq_pair);
        float2 _f2_521 = make_float2(acc[20], acc[21]);
        float2 v_189 = _f2_521;
        output_sq_pair = fma_f32x2_rn_noftz(v_189, v_189, output_sq_pair);
        float2 _f2_522 = make_float2(acc[22], acc[23]);
        float2 v_190 = _f2_522;
        output_sq_pair = fma_f32x2_rn_noftz(v_190, v_190, output_sq_pair);
        float2 _f2_523 = make_float2(acc[24], acc[25]);
        float2 v_191 = _f2_523;
        output_sq_pair = fma_f32x2_rn_noftz(v_191, v_191, output_sq_pair);
        float2 _f2_524 = make_float2(acc[26], acc[27]);
        float2 v_192 = _f2_524;
        output_sq_pair = fma_f32x2_rn_noftz(v_192, v_192, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_25;
        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_26;
        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_27;
        float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_28;
        float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_29;
        if (lane == 0) {
            uint32_t _mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_20) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_20), "f"(output_sq) : "memory");
            uint32_t _mapa_21;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_21) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_21), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_12;
        float _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_13;
        float _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_14;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_2 = rsqrtf(output_total / 7168.0f + output_norm_eps * sum_running * sum_running);
            rsigma_lane = _rsqrt_2;
        }
        float _shfl_5 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_5;
        float2 _f2_525 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_525;
        int base_193 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_526 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_526, rsigma_pair);
        float2 _f2_527 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_527);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_194 = 2;
        const int acc_idx_195 = value_idx_194;
        float2 _f2_528 = make_float2(acc[acc_idx_195], acc[acc_idx_195 + 1]);
        float2 scaled_pair_196 = mul_f32x2_noftz(_f2_528, rsigma_pair);
        float2 _f2_529 = make_float2(wout[acc_idx_195], wout[acc_idx_195 + 1]);
        float2 normalized_pair_197 = mul_f32x2_noftz(scaled_pair_196, _f2_529);
        output_values[value_idx_194] = normalized_pair_197.x;
        output_values[value_idx_194 + 1] = normalized_pair_197.y;
        const int value_idx_198 = 4;
        const int acc_idx_199 = value_idx_198;
        float2 _f2_530 = make_float2(acc[acc_idx_199], acc[acc_idx_199 + 1]);
        float2 scaled_pair_200 = mul_f32x2_noftz(_f2_530, rsigma_pair);
        float2 _f2_531 = make_float2(wout[acc_idx_199], wout[acc_idx_199 + 1]);
        float2 normalized_pair_201 = mul_f32x2_noftz(scaled_pair_200, _f2_531);
        output_values[value_idx_198] = normalized_pair_201.x;
        output_values[value_idx_198 + 1] = normalized_pair_201.y;
        const int value_idx_202 = 6;
        const int acc_idx_203 = value_idx_202;
        float2 _f2_532 = make_float2(acc[acc_idx_203], acc[acc_idx_203 + 1]);
        float2 scaled_pair_204 = mul_f32x2_noftz(_f2_532, rsigma_pair);
        float2 _f2_533 = make_float2(wout[acc_idx_203], wout[acc_idx_203 + 1]);
        float2 normalized_pair_205 = mul_f32x2_noftz(scaled_pair_204, _f2_533);
        output_values[value_idx_202] = normalized_pair_205.x;
        output_values[value_idx_202 + 1] = normalized_pair_205.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_193 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_206 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_207[8];
        const int value_idx_208 = 0;
        const int acc_idx_209 = 8 + value_idx_208;
        float2 _f2_534 = make_float2(acc[acc_idx_209], acc[acc_idx_209 + 1]);
        float2 scaled_pair_210 = mul_f32x2_noftz(_f2_534, rsigma_pair);
        float2 _f2_535 = make_float2(wout[acc_idx_209], wout[acc_idx_209 + 1]);
        float2 normalized_pair_211 = mul_f32x2_noftz(scaled_pair_210, _f2_535);
        output_values_207[value_idx_208] = normalized_pair_211.x;
        output_values_207[value_idx_208 + 1] = normalized_pair_211.y;
        const int value_idx_212 = 2;
        const int acc_idx_213 = 8 + value_idx_212;
        float2 _f2_536 = make_float2(acc[acc_idx_213], acc[acc_idx_213 + 1]);
        float2 scaled_pair_214 = mul_f32x2_noftz(_f2_536, rsigma_pair);
        float2 _f2_537 = make_float2(wout[acc_idx_213], wout[acc_idx_213 + 1]);
        float2 normalized_pair_215 = mul_f32x2_noftz(scaled_pair_214, _f2_537);
        output_values_207[value_idx_212] = normalized_pair_215.x;
        output_values_207[value_idx_212 + 1] = normalized_pair_215.y;
        const int value_idx_216 = 4;
        const int acc_idx_217 = 8 + value_idx_216;
        float2 _f2_538 = make_float2(acc[acc_idx_217], acc[acc_idx_217 + 1]);
        float2 scaled_pair_218 = mul_f32x2_noftz(_f2_538, rsigma_pair);
        float2 _f2_539 = make_float2(wout[acc_idx_217], wout[acc_idx_217 + 1]);
        float2 normalized_pair_219 = mul_f32x2_noftz(scaled_pair_218, _f2_539);
        output_values_207[value_idx_216] = normalized_pair_219.x;
        output_values_207[value_idx_216 + 1] = normalized_pair_219.y;
        const int value_idx_220 = 6;
        const int acc_idx_221 = 8 + value_idx_220;
        float2 _f2_540 = make_float2(acc[acc_idx_221], acc[acc_idx_221 + 1]);
        float2 scaled_pair_222 = mul_f32x2_noftz(_f2_540, rsigma_pair);
        float2 _f2_541 = make_float2(wout[acc_idx_221], wout[acc_idx_221 + 1]);
        float2 normalized_pair_223 = mul_f32x2_noftz(scaled_pair_222, _f2_541);
        output_values_207[value_idx_220] = normalized_pair_223.x;
        output_values_207[value_idx_220 + 1] = normalized_pair_223.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_207[0 + 0], output_values_207[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_207[0 + 2], output_values_207[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_207[0 + 4], output_values_207[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_207[0 + 6], output_values_207[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_206 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_224 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_225[8];
        const int value_idx_226 = 0;
        const int acc_idx_227 = 16 + value_idx_226;
        float2 _f2_542 = make_float2(acc[acc_idx_227], acc[acc_idx_227 + 1]);
        float2 scaled_pair_228 = mul_f32x2_noftz(_f2_542, rsigma_pair);
        float2 _f2_543 = make_float2(wout[acc_idx_227], wout[acc_idx_227 + 1]);
        float2 normalized_pair_229 = mul_f32x2_noftz(scaled_pair_228, _f2_543);
        output_values_225[value_idx_226] = normalized_pair_229.x;
        output_values_225[value_idx_226 + 1] = normalized_pair_229.y;
        const int value_idx_230 = 2;
        const int acc_idx_231 = 16 + value_idx_230;
        float2 _f2_544 = make_float2(acc[acc_idx_231], acc[acc_idx_231 + 1]);
        float2 scaled_pair_232 = mul_f32x2_noftz(_f2_544, rsigma_pair);
        float2 _f2_545 = make_float2(wout[acc_idx_231], wout[acc_idx_231 + 1]);
        float2 normalized_pair_233 = mul_f32x2_noftz(scaled_pair_232, _f2_545);
        output_values_225[value_idx_230] = normalized_pair_233.x;
        output_values_225[value_idx_230 + 1] = normalized_pair_233.y;
        const int value_idx_234 = 4;
        const int acc_idx_235 = 16 + value_idx_234;
        float2 _f2_546 = make_float2(acc[acc_idx_235], acc[acc_idx_235 + 1]);
        float2 scaled_pair_236 = mul_f32x2_noftz(_f2_546, rsigma_pair);
        float2 _f2_547 = make_float2(wout[acc_idx_235], wout[acc_idx_235 + 1]);
        float2 normalized_pair_237 = mul_f32x2_noftz(scaled_pair_236, _f2_547);
        output_values_225[value_idx_234] = normalized_pair_237.x;
        output_values_225[value_idx_234 + 1] = normalized_pair_237.y;
        const int value_idx_238 = 6;
        const int acc_idx_239 = 16 + value_idx_238;
        float2 _f2_548 = make_float2(acc[acc_idx_239], acc[acc_idx_239 + 1]);
        float2 scaled_pair_240 = mul_f32x2_noftz(_f2_548, rsigma_pair);
        float2 _f2_549 = make_float2(wout[acc_idx_239], wout[acc_idx_239 + 1]);
        float2 normalized_pair_241 = mul_f32x2_noftz(scaled_pair_240, _f2_549);
        output_values_225[value_idx_238] = normalized_pair_241.x;
        output_values_225[value_idx_238 + 1] = normalized_pair_241.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_225[0 + 0], output_values_225[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_225[0 + 2], output_values_225[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_225[0 + 4], output_values_225[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_225[0 + 6], output_values_225[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_224 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_242 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_243[8];
        const int value_idx_244 = 0;
        const int acc_idx_245 = 24 + value_idx_244;
        float2 _f2_550 = make_float2(acc[acc_idx_245], acc[acc_idx_245 + 1]);
        float2 scaled_pair_246 = mul_f32x2_noftz(_f2_550, rsigma_pair);
        float2 _f2_551 = make_float2(wout[acc_idx_245], wout[acc_idx_245 + 1]);
        float2 normalized_pair_247 = mul_f32x2_noftz(scaled_pair_246, _f2_551);
        output_values_243[value_idx_244] = normalized_pair_247.x;
        output_values_243[value_idx_244 + 1] = normalized_pair_247.y;
        const int value_idx_248 = 2;
        const int acc_idx_249 = 24 + value_idx_248;
        float2 _f2_552 = make_float2(acc[acc_idx_249], acc[acc_idx_249 + 1]);
        float2 scaled_pair_250 = mul_f32x2_noftz(_f2_552, rsigma_pair);
        float2 _f2_553 = make_float2(wout[acc_idx_249], wout[acc_idx_249 + 1]);
        float2 normalized_pair_251 = mul_f32x2_noftz(scaled_pair_250, _f2_553);
        output_values_243[value_idx_248] = normalized_pair_251.x;
        output_values_243[value_idx_248 + 1] = normalized_pair_251.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_243[0 + 0], output_values_243[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_243[0 + 2], output_values_243[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_242]) = _pk2;
        }
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
