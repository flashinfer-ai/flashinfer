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
#define SMEM_STATS_STAGE_BYTES 576
#define SMEM_STATS_STRIDE 576
#define SMEM_OUT_STATS_OFF 576
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 640
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
kernel_cake_kimi_k3_attn_res_c84b13f3de176fff0611(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 576);
    const int out_stats_addr = smem + 576;

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
        unsigned int words[126];
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
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_15[0 + 0] = _uv4_5.x;
                _vec_load_15[0 + 1] = _uv4_5.y;
                _vec_load_15[0 + 2] = _uv4_5.z;
                _vec_load_15[0 + 3] = _uv4_5.w;
            }
            words[70 + woff] = _vec_load_15[0];
            words[70 + woff + 1] = _vec_load_15[1];
            words[70 + woff + 2] = _vec_load_15[2];
            words[70 + woff + 3] = _vec_load_15[3];
        }
        {
            unsigned int _vec_load_18[4];
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_18[0 + 0] = _uv4_6.x;
                _vec_load_18[0 + 1] = _uv4_6.y;
                _vec_load_18[0 + 2] = _uv4_6.z;
                _vec_load_18[0 + 3] = _uv4_6.w;
            }
            words[84 + woff] = _vec_load_18[0];
            words[84 + woff + 1] = _vec_load_18[1];
            words[84 + woff + 2] = _vec_load_18[2];
            words[84 + woff + 3] = _vec_load_18[3];
        }
        {
            unsigned int _vec_load_21[4];
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_21[0 + 0] = _uv4_7.x;
                _vec_load_21[0 + 1] = _uv4_7.y;
                _vec_load_21[0 + 2] = _uv4_7.z;
                _vec_load_21[0 + 3] = _uv4_7.w;
            }
            words[98 + woff] = _vec_load_21[0];
            words[98 + woff + 1] = _vec_load_21[1];
            words[98 + woff + 2] = _vec_load_21[2];
            words[98 + woff + 3] = _vec_load_21[3];
        }
        {
            unsigned int _vec_load_24[4];
            {
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 8 * blocks_k_stride + (unsigned long long)base))) + 0);
                _vec_load_24[0 + 0] = _uv4_8.x;
                _vec_load_24[0 + 1] = _uv4_8.y;
                _vec_load_24[0 + 2] = _uv4_8.z;
                _vec_load_24[0 + 3] = _uv4_8.w;
            }
            words[112 + woff] = _vec_load_24[0];
            words[112 + woff + 1] = _vec_load_24[1];
            words[112 + woff + 2] = _vec_load_24[2];
            words[112 + woff + 3] = _vec_load_24[3];
        }
        {
            unsigned int _vec_load_27[4];
            {
                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_27[0 + 0] = _uv4_9.x;
                _vec_load_27[0 + 1] = _uv4_9.y;
                _vec_load_27[0 + 2] = _uv4_9.z;
                _vec_load_27[0 + 3] = _uv4_9.w;
            }
            dwords[woff] = _vec_load_27[0];
            dwords[woff + 1] = _vec_load_27[1];
            dwords[woff + 2] = _vec_load_27[2];
            dwords[woff + 3] = _vec_load_27[3];
        }
        float _vec_load_30[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_30[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_30[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        float _vec_load_31[8];
        {
            const uint4* _vptr_11 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_31[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_31[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        q[0] = _vec_load_30[0] * _vec_load_31[0];
        q[1] = _vec_load_30[1] * _vec_load_31[1];
        q[2] = _vec_load_30[2] * _vec_load_31[2];
        q[3] = _vec_load_30[3] * _vec_load_31[3];
        q[4] = _vec_load_30[4] * _vec_load_31[4];
        q[5] = _vec_load_30[5] * _vec_load_31[5];
        q[6] = _vec_load_30[6] * _vec_load_31[6];
        q[7] = _vec_load_30[7] * _vec_load_31[7];
        float _vec_load_32[8];
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(output_norm_weight + base);
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
                        : "=f"((&_vec_load_32[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_32[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_12[_pair]));
                }
            }
        }
        wout[0] = _vec_load_32[0];
        wout[1] = _vec_load_32[1];
        wout[2] = _vec_load_32[2];
        wout[3] = _vec_load_32[3];
        wout[4] = _vec_load_32[4];
        wout[5] = _vec_load_32[5];
        wout[6] = _vec_load_32[6];
        wout[7] = _vec_load_32[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_33[4];
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_33[0 + 0] = _uv4_13.x;
                _vec_load_33[0 + 1] = _uv4_13.y;
                _vec_load_33[0 + 2] = _uv4_13.z;
                _vec_load_33[0 + 3] = _uv4_13.w;
            }
            words[woff_1] = _vec_load_33[0];
            words[woff_1 + 1] = _vec_load_33[1];
            words[woff_1 + 2] = _vec_load_33[2];
            words[woff_1 + 3] = _vec_load_33[3];
        }
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_36[0 + 0] = _uv4_14.x;
                _vec_load_36[0 + 1] = _uv4_14.y;
                _vec_load_36[0 + 2] = _uv4_14.z;
                _vec_load_36[0 + 3] = _uv4_14.w;
            }
            words[14 + woff_1] = _vec_load_36[0];
            words[14 + woff_1 + 1] = _vec_load_36[1];
            words[14 + woff_1 + 2] = _vec_load_36[2];
            words[14 + woff_1 + 3] = _vec_load_36[3];
        }
        {
            unsigned int _vec_load_39[4];
            {
                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_39[0 + 0] = _uv4_15.x;
                _vec_load_39[0 + 1] = _uv4_15.y;
                _vec_load_39[0 + 2] = _uv4_15.z;
                _vec_load_39[0 + 3] = _uv4_15.w;
            }
            words[28 + woff_1] = _vec_load_39[0];
            words[28 + woff_1 + 1] = _vec_load_39[1];
            words[28 + woff_1 + 2] = _vec_load_39[2];
            words[28 + woff_1 + 3] = _vec_load_39[3];
        }
        {
            unsigned int _vec_load_42[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_42[0 + 0] = _uv4_16.x;
                _vec_load_42[0 + 1] = _uv4_16.y;
                _vec_load_42[0 + 2] = _uv4_16.z;
                _vec_load_42[0 + 3] = _uv4_16.w;
            }
            words[42 + woff_1] = _vec_load_42[0];
            words[42 + woff_1 + 1] = _vec_load_42[1];
            words[42 + woff_1 + 2] = _vec_load_42[2];
            words[42 + woff_1 + 3] = _vec_load_42[3];
        }
        {
            unsigned int _vec_load_45[4];
            {
                uint4 _uv4_17 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_45[0 + 0] = _uv4_17.x;
                _vec_load_45[0 + 1] = _uv4_17.y;
                _vec_load_45[0 + 2] = _uv4_17.z;
                _vec_load_45[0 + 3] = _uv4_17.w;
            }
            words[56 + woff_1] = _vec_load_45[0];
            words[56 + woff_1 + 1] = _vec_load_45[1];
            words[56 + woff_1 + 2] = _vec_load_45[2];
            words[56 + woff_1 + 3] = _vec_load_45[3];
        }
        {
            unsigned int _vec_load_48[4];
            {
                uint4 _uv4_18 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_48[0 + 0] = _uv4_18.x;
                _vec_load_48[0 + 1] = _uv4_18.y;
                _vec_load_48[0 + 2] = _uv4_18.z;
                _vec_load_48[0 + 3] = _uv4_18.w;
            }
            words[70 + woff_1] = _vec_load_48[0];
            words[70 + woff_1 + 1] = _vec_load_48[1];
            words[70 + woff_1 + 2] = _vec_load_48[2];
            words[70 + woff_1 + 3] = _vec_load_48[3];
        }
        {
            unsigned int _vec_load_51[4];
            {
                uint4 _uv4_19 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_51[0 + 0] = _uv4_19.x;
                _vec_load_51[0 + 1] = _uv4_19.y;
                _vec_load_51[0 + 2] = _uv4_19.z;
                _vec_load_51[0 + 3] = _uv4_19.w;
            }
            words[84 + woff_1] = _vec_load_51[0];
            words[84 + woff_1 + 1] = _vec_load_51[1];
            words[84 + woff_1 + 2] = _vec_load_51[2];
            words[84 + woff_1 + 3] = _vec_load_51[3];
        }
        {
            unsigned int _vec_load_54[4];
            {
                uint4 _uv4_20 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_54[0 + 0] = _uv4_20.x;
                _vec_load_54[0 + 1] = _uv4_20.y;
                _vec_load_54[0 + 2] = _uv4_20.z;
                _vec_load_54[0 + 3] = _uv4_20.w;
            }
            words[98 + woff_1] = _vec_load_54[0];
            words[98 + woff_1 + 1] = _vec_load_54[1];
            words[98 + woff_1 + 2] = _vec_load_54[2];
            words[98 + woff_1 + 3] = _vec_load_54[3];
        }
        {
            unsigned int _vec_load_57[4];
            {
                uint4 _uv4_21 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 8 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_57[0 + 0] = _uv4_21.x;
                _vec_load_57[0 + 1] = _uv4_21.y;
                _vec_load_57[0 + 2] = _uv4_21.z;
                _vec_load_57[0 + 3] = _uv4_21.w;
            }
            words[112 + woff_1] = _vec_load_57[0];
            words[112 + woff_1 + 1] = _vec_load_57[1];
            words[112 + woff_1 + 2] = _vec_load_57[2];
            words[112 + woff_1 + 3] = _vec_load_57[3];
        }
        {
            unsigned int _vec_load_60[4];
            {
                uint4 _uv4_22 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_60[0 + 0] = _uv4_22.x;
                _vec_load_60[0 + 1] = _uv4_22.y;
                _vec_load_60[0 + 2] = _uv4_22.z;
                _vec_load_60[0 + 3] = _uv4_22.w;
            }
            dwords[woff_1] = _vec_load_60[0];
            dwords[woff_1 + 1] = _vec_load_60[1];
            dwords[woff_1 + 2] = _vec_load_60[2];
            dwords[woff_1 + 3] = _vec_load_60[3];
        }
        float _vec_load_63[8];
        {
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_63[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_63[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_23[_pair]));
                }
            }
        }
        float _vec_load_64[8];
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(qk_weight + base_0);
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
                        : "=f"((&_vec_load_64[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_64[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_24[_pair]));
                }
            }
        }
        q[8] = _vec_load_63[0] * _vec_load_64[0];
        q[9] = _vec_load_63[1] * _vec_load_64[1];
        q[10] = _vec_load_63[2] * _vec_load_64[2];
        q[11] = _vec_load_63[3] * _vec_load_64[3];
        q[12] = _vec_load_63[4] * _vec_load_64[4];
        q[13] = _vec_load_63[5] * _vec_load_64[5];
        q[14] = _vec_load_63[6] * _vec_load_64[6];
        q[15] = _vec_load_63[7] * _vec_load_64[7];
        float _vec_load_65[8];
        {
            const uint4* _vptr_25 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_65[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_65[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_25[_pair]));
                }
            }
        }
        wout[8] = _vec_load_65[0];
        wout[9] = _vec_load_65[1];
        wout[10] = _vec_load_65[2];
        wout[11] = _vec_load_65[3];
        wout[12] = _vec_load_65[4];
        wout[13] = _vec_load_65[5];
        wout[14] = _vec_load_65[6];
        wout[15] = _vec_load_65[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_66[4];
            {
                uint4 _uv4_26 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_66[0 + 0] = _uv4_26.x;
                _vec_load_66[0 + 1] = _uv4_26.y;
                _vec_load_66[0 + 2] = _uv4_26.z;
                _vec_load_66[0 + 3] = _uv4_26.w;
            }
            words[woff_3] = _vec_load_66[0];
            words[woff_3 + 1] = _vec_load_66[1];
            words[woff_3 + 2] = _vec_load_66[2];
            words[woff_3 + 3] = _vec_load_66[3];
        }
        {
            unsigned int _vec_load_69[4];
            {
                uint4 _uv4_27 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_69[0 + 0] = _uv4_27.x;
                _vec_load_69[0 + 1] = _uv4_27.y;
                _vec_load_69[0 + 2] = _uv4_27.z;
                _vec_load_69[0 + 3] = _uv4_27.w;
            }
            words[14 + woff_3] = _vec_load_69[0];
            words[14 + woff_3 + 1] = _vec_load_69[1];
            words[14 + woff_3 + 2] = _vec_load_69[2];
            words[14 + woff_3 + 3] = _vec_load_69[3];
        }
        {
            unsigned int _vec_load_72[4];
            {
                uint4 _uv4_28 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_72[0 + 0] = _uv4_28.x;
                _vec_load_72[0 + 1] = _uv4_28.y;
                _vec_load_72[0 + 2] = _uv4_28.z;
                _vec_load_72[0 + 3] = _uv4_28.w;
            }
            words[28 + woff_3] = _vec_load_72[0];
            words[28 + woff_3 + 1] = _vec_load_72[1];
            words[28 + woff_3 + 2] = _vec_load_72[2];
            words[28 + woff_3 + 3] = _vec_load_72[3];
        }
        {
            unsigned int _vec_load_75[4];
            {
                uint4 _uv4_29 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_75[0 + 0] = _uv4_29.x;
                _vec_load_75[0 + 1] = _uv4_29.y;
                _vec_load_75[0 + 2] = _uv4_29.z;
                _vec_load_75[0 + 3] = _uv4_29.w;
            }
            words[42 + woff_3] = _vec_load_75[0];
            words[42 + woff_3 + 1] = _vec_load_75[1];
            words[42 + woff_3 + 2] = _vec_load_75[2];
            words[42 + woff_3 + 3] = _vec_load_75[3];
        }
        {
            unsigned int _vec_load_78[4];
            {
                uint4 _uv4_30 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_78[0 + 0] = _uv4_30.x;
                _vec_load_78[0 + 1] = _uv4_30.y;
                _vec_load_78[0 + 2] = _uv4_30.z;
                _vec_load_78[0 + 3] = _uv4_30.w;
            }
            words[56 + woff_3] = _vec_load_78[0];
            words[56 + woff_3 + 1] = _vec_load_78[1];
            words[56 + woff_3 + 2] = _vec_load_78[2];
            words[56 + woff_3 + 3] = _vec_load_78[3];
        }
        {
            unsigned int _vec_load_81[4];
            {
                uint4 _uv4_31 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_81[0 + 0] = _uv4_31.x;
                _vec_load_81[0 + 1] = _uv4_31.y;
                _vec_load_81[0 + 2] = _uv4_31.z;
                _vec_load_81[0 + 3] = _uv4_31.w;
            }
            words[70 + woff_3] = _vec_load_81[0];
            words[70 + woff_3 + 1] = _vec_load_81[1];
            words[70 + woff_3 + 2] = _vec_load_81[2];
            words[70 + woff_3 + 3] = _vec_load_81[3];
        }
        {
            unsigned int _vec_load_84[4];
            {
                uint4 _uv4_32 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_84[0 + 0] = _uv4_32.x;
                _vec_load_84[0 + 1] = _uv4_32.y;
                _vec_load_84[0 + 2] = _uv4_32.z;
                _vec_load_84[0 + 3] = _uv4_32.w;
            }
            words[84 + woff_3] = _vec_load_84[0];
            words[84 + woff_3 + 1] = _vec_load_84[1];
            words[84 + woff_3 + 2] = _vec_load_84[2];
            words[84 + woff_3 + 3] = _vec_load_84[3];
        }
        {
            unsigned int _vec_load_87[4];
            {
                uint4 _uv4_33 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_87[0 + 0] = _uv4_33.x;
                _vec_load_87[0 + 1] = _uv4_33.y;
                _vec_load_87[0 + 2] = _uv4_33.z;
                _vec_load_87[0 + 3] = _uv4_33.w;
            }
            words[98 + woff_3] = _vec_load_87[0];
            words[98 + woff_3 + 1] = _vec_load_87[1];
            words[98 + woff_3 + 2] = _vec_load_87[2];
            words[98 + woff_3 + 3] = _vec_load_87[3];
        }
        {
            unsigned int _vec_load_90[4];
            {
                uint4 _uv4_34 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 8 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_90[0 + 0] = _uv4_34.x;
                _vec_load_90[0 + 1] = _uv4_34.y;
                _vec_load_90[0 + 2] = _uv4_34.z;
                _vec_load_90[0 + 3] = _uv4_34.w;
            }
            words[112 + woff_3] = _vec_load_90[0];
            words[112 + woff_3 + 1] = _vec_load_90[1];
            words[112 + woff_3 + 2] = _vec_load_90[2];
            words[112 + woff_3 + 3] = _vec_load_90[3];
        }
        {
            unsigned int _vec_load_93[4];
            {
                uint4 _uv4_35 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_93[0 + 0] = _uv4_35.x;
                _vec_load_93[0 + 1] = _uv4_35.y;
                _vec_load_93[0 + 2] = _uv4_35.z;
                _vec_load_93[0 + 3] = _uv4_35.w;
            }
            dwords[woff_3] = _vec_load_93[0];
            dwords[woff_3 + 1] = _vec_load_93[1];
            dwords[woff_3 + 2] = _vec_load_93[2];
            dwords[woff_3 + 3] = _vec_load_93[3];
        }
        float _vec_load_96[8];
        {
            const uint4* _vptr_36 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_36[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_36[_blk] = _vptr_36[_blk];
                uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_96[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_96[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_36[_pair]));
                }
            }
        }
        float _vec_load_97[8];
        {
            const uint4* _vptr_37 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_37[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_37[_blk] = _vptr_37[_blk];
                uint32_t* _vpairs_37 = reinterpret_cast<uint32_t*>(&_vld_37[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_97[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_97[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_37[_pair]));
                }
            }
        }
        q[16] = _vec_load_96[0] * _vec_load_97[0];
        q[17] = _vec_load_96[1] * _vec_load_97[1];
        q[18] = _vec_load_96[2] * _vec_load_97[2];
        q[19] = _vec_load_96[3] * _vec_load_97[3];
        q[20] = _vec_load_96[4] * _vec_load_97[4];
        q[21] = _vec_load_96[5] * _vec_load_97[5];
        q[22] = _vec_load_96[6] * _vec_load_97[6];
        q[23] = _vec_load_96[7] * _vec_load_97[7];
        float _vec_load_98[8];
        {
            const uint4* _vptr_38 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_38[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_38[_blk] = _vptr_38[_blk];
                uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_98[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_98[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_38[_pair]));
                }
            }
        }
        wout[16] = _vec_load_98[0];
        wout[17] = _vec_load_98[1];
        wout[18] = _vec_load_98[2];
        wout[19] = _vec_load_98[3];
        wout[20] = _vec_load_98[4];
        wout[21] = _vec_load_98[5];
        wout[22] = _vec_load_98[6];
        wout[23] = _vec_load_98[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_100[1];
            {
                _vec_load_100[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_100[0];
            unsigned int _vec_load_101[1];
            {
                _vec_load_101[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_101[0];
        }
        {
            unsigned int _vec_load_103[1];
            {
                _vec_load_103[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_103[0];
            unsigned int _vec_load_104[1];
            {
                _vec_load_104[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_104[0];
        }
        {
            unsigned int _vec_load_106[1];
            {
                _vec_load_106[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_106[0];
            unsigned int _vec_load_107[1];
            {
                _vec_load_107[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_107[0];
        }
        {
            unsigned int _vec_load_109[1];
            {
                _vec_load_109[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[42 + woff_5] = _vec_load_109[0];
            unsigned int _vec_load_110[1];
            {
                _vec_load_110[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[42 + woff_5 + 1] = _vec_load_110[0];
        }
        {
            unsigned int _vec_load_112[1];
            {
                _vec_load_112[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[56 + woff_5] = _vec_load_112[0];
            unsigned int _vec_load_113[1];
            {
                _vec_load_113[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[56 + woff_5 + 1] = _vec_load_113[0];
        }
        {
            unsigned int _vec_load_115[1];
            {
                _vec_load_115[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[70 + woff_5] = _vec_load_115[0];
            unsigned int _vec_load_116[1];
            {
                _vec_load_116[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[70 + woff_5 + 1] = _vec_load_116[0];
        }
        {
            unsigned int _vec_load_118[1];
            {
                _vec_load_118[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[84 + woff_5] = _vec_load_118[0];
            unsigned int _vec_load_119[1];
            {
                _vec_load_119[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[84 + woff_5 + 1] = _vec_load_119[0];
        }
        {
            unsigned int _vec_load_121[1];
            {
                _vec_load_121[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[98 + woff_5] = _vec_load_121[0];
            unsigned int _vec_load_122[1];
            {
                _vec_load_122[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[98 + woff_5 + 1] = _vec_load_122[0];
        }
        {
            unsigned int _vec_load_124[1];
            {
                _vec_load_124[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 8 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[112 + woff_5] = _vec_load_124[0];
            unsigned int _vec_load_125[1];
            {
                _vec_load_125[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 8 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[112 + woff_5 + 1] = _vec_load_125[0];
        }
        {
            unsigned int _vec_load_127[1];
            {
                _vec_load_127[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_127[0];
            unsigned int _vec_load_128[1];
            {
                _vec_load_128[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_128[0];
        }
        float _vec_load_129[4];
        {
            uint2 _vld_39;
            _vld_39 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_39 = reinterpret_cast<uint32_t*>(&_vld_39);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_129[0 + _pair * 2])[0]), "=f"((&_vec_load_129[0 + _pair * 2])[1])
                    : "r"(_vpairs_39[_pair]));
            }
        }
        float _vec_load_130[4];
        {
            uint2 _vld_40;
            _vld_40 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_40 = reinterpret_cast<uint32_t*>(&_vld_40);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_130[0 + _pair * 2])[0]), "=f"((&_vec_load_130[0 + _pair * 2])[1])
                    : "r"(_vpairs_40[_pair]));
            }
        }
        q[24] = _vec_load_129[0] * _vec_load_130[0];
        q[25] = _vec_load_129[1] * _vec_load_130[1];
        q[26] = _vec_load_129[2] * _vec_load_130[2];
        q[27] = _vec_load_129[3] * _vec_load_130[3];
        float _vec_load_131[4];
        {
            uint2 _vld_41;
            _vld_41 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_131[0 + _pair * 2])[0]), "=f"((&_vec_load_131[0 + _pair * 2])[1])
                    : "r"(_vpairs_41[_pair]));
            }
        }
        wout[24] = _vec_load_131[0];
        wout[25] = _vec_load_131[1];
        wout[26] = _vec_load_131[2];
        wout[27] = _vec_load_131[3];
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
        float2 sq[9];
        float2 dot[9];
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
        float2 _f2_10 = make_float2(0.0f, 0.0f);
        sq[5] = _f2_10;
        float2 _f2_11 = make_float2(0.0f, 0.0f);
        dot[5] = _f2_11;
        float2 _f2_12 = make_float2(0.0f, 0.0f);
        sq[6] = _f2_12;
        float2 _f2_13 = make_float2(0.0f, 0.0f);
        dot[6] = _f2_13;
        float2 _f2_14 = make_float2(0.0f, 0.0f);
        sq[7] = _f2_14;
        float2 _f2_15 = make_float2(0.0f, 0.0f);
        dot[7] = _f2_15;
        float2 _f2_16 = make_float2(0.0f, 0.0f);
        sq[8] = _f2_16;
        float2 _f2_17 = make_float2(0.0f, 0.0f);
        dot[8] = _f2_17;
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
            float2 _f2_24 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_24;
            float2 _f2_25 = make_float2(q[0], q[1]);
            float2 qp = _f2_25;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_26 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_26;
            float2 _f2_27 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_27;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_28 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_28;
            float2 _f2_29 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_29;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_30 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_30;
            float2 _f2_31 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_31;
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
            float2 _f2_38 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_38;
            float2 _f2_39 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_39;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_40 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_40;
            float2 _f2_41 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_41;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_42 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_42;
            float2 _f2_43 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_43;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_44 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_44;
            float2 _f2_45 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_45;
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
            float2 _f2_52 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_52;
            float2 _f2_53 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_53;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_54 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_54;
            float2 _f2_55 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_55;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_56 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_56;
            float2 _f2_57 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_57;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_58 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_58;
            float2 _f2_59 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_59;
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
            float2 _f2_66 = make_float2(sw_10_f32[0], sw_10_f32[1]);
            float2 v_5 = _f2_66;
            float2 _f2_67 = make_float2(q[0], q[1]);
            float2 qp_6 = _f2_67;
            sq[3] = fma_f32x2_rn_noftz(v_5, v_5, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_5, qp_6, dot[3]);
            float2 _f2_68 = make_float2(sw_10_f32[2], sw_10_f32[3]);
            float2 v_0_3 = _f2_68;
            float2 _f2_69 = make_float2(q[2], q[3]);
            float2 qp_1_3 = _f2_69;
            sq[3] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[3]);
            float2 _f2_70 = make_float2(sw_10_f32[4], sw_10_f32[5]);
            float2 v_2_3 = _f2_70;
            float2 _f2_71 = make_float2(q[4], q[5]);
            float2 qp_3_3 = _f2_71;
            sq[3] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[3]);
            float2 _f2_72 = make_float2(sw_10_f32[6], sw_10_f32[7]);
            float2 v_4_3 = _f2_72;
            float2 _f2_73 = make_float2(q[6], q[7]);
            float2 qp_5_3 = _f2_73;
            sq[3] = fma_f32x2_rn_noftz(v_4_3, v_4_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_3, qp_5_3, dot[3]);
        }
        unsigned int sw_11[4];
        sw_11[0] = words[56 + woff_7];
        sw_11[1] = words[56 + woff_7 + 1];
        sw_11[2] = words[56 + woff_7 + 2];
        sw_11[3] = words[56 + woff_7 + 3];
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
            float2 _f2_80 = make_float2(sw_11_f32[0], sw_11_f32[1]);
            float2 v_6 = _f2_80;
            float2 _f2_81 = make_float2(q[0], q[1]);
            float2 qp_7 = _f2_81;
            sq[4] = fma_f32x2_rn_noftz(v_6, v_6, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_6, qp_7, dot[4]);
            float2 _f2_82 = make_float2(sw_11_f32[2], sw_11_f32[3]);
            float2 v_0_4 = _f2_82;
            float2 _f2_83 = make_float2(q[2], q[3]);
            float2 qp_1_4 = _f2_83;
            sq[4] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[4]);
            float2 _f2_84 = make_float2(sw_11_f32[4], sw_11_f32[5]);
            float2 v_2_4 = _f2_84;
            float2 _f2_85 = make_float2(q[4], q[5]);
            float2 qp_3_4 = _f2_85;
            sq[4] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[4]);
            float2 _f2_86 = make_float2(sw_11_f32[6], sw_11_f32[7]);
            float2 v_4_4 = _f2_86;
            float2 _f2_87 = make_float2(q[6], q[7]);
            float2 qp_5_4 = _f2_87;
            sq[4] = fma_f32x2_rn_noftz(v_4_4, v_4_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_4, qp_5_4, dot[4]);
        }
        unsigned int sw_12[4];
        sw_12[0] = words[70 + woff_7];
        sw_12[1] = words[70 + woff_7 + 1];
        sw_12[2] = words[70 + woff_7 + 2];
        sw_12[3] = words[70 + woff_7 + 3];
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
            float2 _f2_94 = make_float2(sw_12_f32[0], sw_12_f32[1]);
            float2 v_7 = _f2_94;
            float2 _f2_95 = make_float2(q[0], q[1]);
            float2 qp_8 = _f2_95;
            sq[5] = fma_f32x2_rn_noftz(v_7, v_7, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_7, qp_8, dot[5]);
            float2 _f2_96 = make_float2(sw_12_f32[2], sw_12_f32[3]);
            float2 v_0_5 = _f2_96;
            float2 _f2_97 = make_float2(q[2], q[3]);
            float2 qp_1_5 = _f2_97;
            sq[5] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[5]);
            float2 _f2_98 = make_float2(sw_12_f32[4], sw_12_f32[5]);
            float2 v_2_5 = _f2_98;
            float2 _f2_99 = make_float2(q[4], q[5]);
            float2 qp_3_5 = _f2_99;
            sq[5] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[5]);
            float2 _f2_100 = make_float2(sw_12_f32[6], sw_12_f32[7]);
            float2 v_4_5 = _f2_100;
            float2 _f2_101 = make_float2(q[6], q[7]);
            float2 qp_5_5 = _f2_101;
            sq[5] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[5]);
        }
        unsigned int sw_13[4];
        sw_13[0] = words[84 + woff_7];
        sw_13[1] = words[84 + woff_7 + 1];
        sw_13[2] = words[84 + woff_7 + 2];
        sw_13[3] = words[84 + woff_7 + 3];
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
            float2 _f2_108 = make_float2(sw_13_f32[0], sw_13_f32[1]);
            float2 v_8 = _f2_108;
            float2 _f2_109 = make_float2(q[0], q[1]);
            float2 qp_9 = _f2_109;
            sq[6] = fma_f32x2_rn_noftz(v_8, v_8, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_8, qp_9, dot[6]);
            float2 _f2_110 = make_float2(sw_13_f32[2], sw_13_f32[3]);
            float2 v_0_6 = _f2_110;
            float2 _f2_111 = make_float2(q[2], q[3]);
            float2 qp_1_6 = _f2_111;
            sq[6] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[6]);
            float2 _f2_112 = make_float2(sw_13_f32[4], sw_13_f32[5]);
            float2 v_2_6 = _f2_112;
            float2 _f2_113 = make_float2(q[4], q[5]);
            float2 qp_3_6 = _f2_113;
            sq[6] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[6]);
            float2 _f2_114 = make_float2(sw_13_f32[6], sw_13_f32[7]);
            float2 v_4_6 = _f2_114;
            float2 _f2_115 = make_float2(q[6], q[7]);
            float2 qp_5_6 = _f2_115;
            sq[6] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[6]);
        }
        unsigned int sw_14[4];
        sw_14[0] = words[98 + woff_7];
        sw_14[1] = words[98 + woff_7 + 1];
        sw_14[2] = words[98 + woff_7 + 2];
        sw_14[3] = words[98 + woff_7 + 3];
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
            float2 _f2_122 = make_float2(sw_14_f32[0], sw_14_f32[1]);
            float2 v_9 = _f2_122;
            float2 _f2_123 = make_float2(q[0], q[1]);
            float2 qp_10 = _f2_123;
            sq[7] = fma_f32x2_rn_noftz(v_9, v_9, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_9, qp_10, dot[7]);
            float2 _f2_124 = make_float2(sw_14_f32[2], sw_14_f32[3]);
            float2 v_0_7 = _f2_124;
            float2 _f2_125 = make_float2(q[2], q[3]);
            float2 qp_1_7 = _f2_125;
            sq[7] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[7]);
            float2 _f2_126 = make_float2(sw_14_f32[4], sw_14_f32[5]);
            float2 v_2_7 = _f2_126;
            float2 _f2_127 = make_float2(q[4], q[5]);
            float2 qp_3_7 = _f2_127;
            sq[7] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[7]);
            float2 _f2_128 = make_float2(sw_14_f32[6], sw_14_f32[7]);
            float2 v_4_7 = _f2_128;
            float2 _f2_129 = make_float2(q[6], q[7]);
            float2 qp_5_7 = _f2_129;
            sq[7] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[7]);
        }
        unsigned int sw_15[4];
        sw_15[0] = words[112 + woff_7];
        sw_15[1] = words[112 + woff_7 + 1];
        sw_15[2] = words[112 + woff_7 + 2];
        sw_15[3] = words[112 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_15[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_15[0] = __as_u32(mixed);
            words[112 + woff_7] = sw_15[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_15[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_15[1] = __as_u32(mixed_2);
            words[112 + woff_7 + 1] = sw_15[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_15[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_15[2] = __as_u32(mixed_5);
            words[112 + woff_7 + 2] = sw_15[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_15[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_15[3] = __as_u32(mixed_8);
            words[112 + woff_7 + 3] = sw_15[3];
            {
                int4 _iv4 = make_int4(sw_15[0 + 0], sw_15[0 + 1], sw_15[0 + 2], sw_15[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
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
            float2 _f2_136 = make_float2(sw_15_f32[0], sw_15_f32[1]);
            float2 v_10 = _f2_136;
            float2 _f2_137 = make_float2(q[0], q[1]);
            float2 qp_11 = _f2_137;
            sq[8] = fma_f32x2_rn_noftz(v_10, v_10, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_10, qp_11, dot[8]);
            float2 _f2_138 = make_float2(sw_15_f32[2], sw_15_f32[3]);
            float2 v_0_8 = _f2_138;
            float2 _f2_139 = make_float2(q[2], q[3]);
            float2 qp_1_8 = _f2_139;
            sq[8] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[8]);
            float2 _f2_140 = make_float2(sw_15_f32[4], sw_15_f32[5]);
            float2 v_2_8 = _f2_140;
            float2 _f2_141 = make_float2(q[4], q[5]);
            float2 qp_3_8 = _f2_141;
            sq[8] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[8]);
            float2 _f2_142 = make_float2(sw_15_f32[6], sw_15_f32[7]);
            float2 v_4_8 = _f2_142;
            float2 _f2_143 = make_float2(q[6], q[7]);
            float2 qp_5_8 = _f2_143;
            sq[8] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[8]);
        }
        int base_16 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_17 = 4;
        unsigned int sw_18[4];
        sw_18[0] = words[woff_17];
        sw_18[1] = words[woff_17 + 1];
        sw_18[2] = words[woff_17 + 2];
        sw_18[3] = words[woff_17 + 3];
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
            fsrc[8] = sw_18_f32[0];
            fsrc[9] = sw_18_f32[1];
            fsrc[10] = sw_18_f32[2];
            fsrc[11] = sw_18_f32[3];
            fsrc[12] = sw_18_f32[4];
            fsrc[13] = sw_18_f32[5];
            fsrc[14] = sw_18_f32[6];
            fsrc[15] = sw_18_f32[7];
        }
        {
            float2 _f2_150 = make_float2(sw_18_f32[0], sw_18_f32[1]);
            float2 v_11 = _f2_150;
            float2 _f2_151 = make_float2(q[8], q[9]);
            float2 qp_12 = _f2_151;
            sq[0] = fma_f32x2_rn_noftz(v_11, v_11, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_11, qp_12, dot[0]);
            float2 _f2_152 = make_float2(sw_18_f32[2], sw_18_f32[3]);
            float2 v_0_9 = _f2_152;
            float2 _f2_153 = make_float2(q[10], q[11]);
            float2 qp_1_9 = _f2_153;
            sq[0] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_9, qp_1_9, dot[0]);
            float2 _f2_154 = make_float2(sw_18_f32[4], sw_18_f32[5]);
            float2 v_2_9 = _f2_154;
            float2 _f2_155 = make_float2(q[12], q[13]);
            float2 qp_3_9 = _f2_155;
            sq[0] = fma_f32x2_rn_noftz(v_2_9, v_2_9, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[0]);
            float2 _f2_156 = make_float2(sw_18_f32[6], sw_18_f32[7]);
            float2 v_4_9 = _f2_156;
            float2 _f2_157 = make_float2(q[14], q[15]);
            float2 qp_5_9 = _f2_157;
            sq[0] = fma_f32x2_rn_noftz(v_4_9, v_4_9, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_9, qp_5_9, dot[0]);
        }
        unsigned int sw_19[4];
        sw_19[0] = words[14 + woff_17];
        sw_19[1] = words[14 + woff_17 + 1];
        sw_19[2] = words[14 + woff_17 + 2];
        sw_19[3] = words[14 + woff_17 + 3];
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
            fsrc[36] = sw_19_f32[0];
            fsrc[37] = sw_19_f32[1];
            fsrc[38] = sw_19_f32[2];
            fsrc[39] = sw_19_f32[3];
            fsrc[40] = sw_19_f32[4];
            fsrc[41] = sw_19_f32[5];
            fsrc[42] = sw_19_f32[6];
            fsrc[43] = sw_19_f32[7];
        }
        {
            float2 _f2_164 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_12 = _f2_164;
            float2 _f2_165 = make_float2(q[8], q[9]);
            float2 qp_13 = _f2_165;
            sq[1] = fma_f32x2_rn_noftz(v_12, v_12, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_12, qp_13, dot[1]);
            float2 _f2_166 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_10 = _f2_166;
            float2 _f2_167 = make_float2(q[10], q[11]);
            float2 qp_1_10 = _f2_167;
            sq[1] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_10, qp_1_10, dot[1]);
            float2 _f2_168 = make_float2(sw_19_f32[4], sw_19_f32[5]);
            float2 v_2_10 = _f2_168;
            float2 _f2_169 = make_float2(q[12], q[13]);
            float2 qp_3_10 = _f2_169;
            sq[1] = fma_f32x2_rn_noftz(v_2_10, v_2_10, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[1]);
            float2 _f2_170 = make_float2(sw_19_f32[6], sw_19_f32[7]);
            float2 v_4_10 = _f2_170;
            float2 _f2_171 = make_float2(q[14], q[15]);
            float2 qp_5_10 = _f2_171;
            sq[1] = fma_f32x2_rn_noftz(v_4_10, v_4_10, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_10, qp_5_10, dot[1]);
        }
        unsigned int sw_20[4];
        sw_20[0] = words[28 + woff_17];
        sw_20[1] = words[28 + woff_17 + 1];
        sw_20[2] = words[28 + woff_17 + 2];
        sw_20[3] = words[28 + woff_17 + 3];
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
            fsrc[64] = sw_20_f32[0];
            fsrc[65] = sw_20_f32[1];
            fsrc[66] = sw_20_f32[2];
            fsrc[67] = sw_20_f32[3];
            fsrc[68] = sw_20_f32[4];
            fsrc[69] = sw_20_f32[5];
            fsrc[70] = sw_20_f32[6];
            fsrc[71] = sw_20_f32[7];
        }
        {
            float2 _f2_178 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_13 = _f2_178;
            float2 _f2_179 = make_float2(q[8], q[9]);
            float2 qp_14 = _f2_179;
            sq[2] = fma_f32x2_rn_noftz(v_13, v_13, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_13, qp_14, dot[2]);
            float2 _f2_180 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_0_11 = _f2_180;
            float2 _f2_181 = make_float2(q[10], q[11]);
            float2 qp_1_11 = _f2_181;
            sq[2] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_11, qp_1_11, dot[2]);
            float2 _f2_182 = make_float2(sw_20_f32[4], sw_20_f32[5]);
            float2 v_2_11 = _f2_182;
            float2 _f2_183 = make_float2(q[12], q[13]);
            float2 qp_3_11 = _f2_183;
            sq[2] = fma_f32x2_rn_noftz(v_2_11, v_2_11, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[2]);
            float2 _f2_184 = make_float2(sw_20_f32[6], sw_20_f32[7]);
            float2 v_4_11 = _f2_184;
            float2 _f2_185 = make_float2(q[14], q[15]);
            float2 qp_5_11 = _f2_185;
            sq[2] = fma_f32x2_rn_noftz(v_4_11, v_4_11, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_11, qp_5_11, dot[2]);
        }
        unsigned int sw_21[4];
        sw_21[0] = words[42 + woff_17];
        sw_21[1] = words[42 + woff_17 + 1];
        sw_21[2] = words[42 + woff_17 + 2];
        sw_21[3] = words[42 + woff_17 + 3];
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
            float2 _f2_192 = make_float2(sw_21_f32[0], sw_21_f32[1]);
            float2 v_14 = _f2_192;
            float2 _f2_193 = make_float2(q[8], q[9]);
            float2 qp_15 = _f2_193;
            sq[3] = fma_f32x2_rn_noftz(v_14, v_14, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_14, qp_15, dot[3]);
            float2 _f2_194 = make_float2(sw_21_f32[2], sw_21_f32[3]);
            float2 v_0_12 = _f2_194;
            float2 _f2_195 = make_float2(q[10], q[11]);
            float2 qp_1_12 = _f2_195;
            sq[3] = fma_f32x2_rn_noftz(v_0_12, v_0_12, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_12, qp_1_12, dot[3]);
            float2 _f2_196 = make_float2(sw_21_f32[4], sw_21_f32[5]);
            float2 v_2_12 = _f2_196;
            float2 _f2_197 = make_float2(q[12], q[13]);
            float2 qp_3_12 = _f2_197;
            sq[3] = fma_f32x2_rn_noftz(v_2_12, v_2_12, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_12, qp_3_12, dot[3]);
            float2 _f2_198 = make_float2(sw_21_f32[6], sw_21_f32[7]);
            float2 v_4_12 = _f2_198;
            float2 _f2_199 = make_float2(q[14], q[15]);
            float2 qp_5_12 = _f2_199;
            sq[3] = fma_f32x2_rn_noftz(v_4_12, v_4_12, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_12, qp_5_12, dot[3]);
        }
        unsigned int sw_22[4];
        sw_22[0] = words[56 + woff_17];
        sw_22[1] = words[56 + woff_17 + 1];
        sw_22[2] = words[56 + woff_17 + 2];
        sw_22[3] = words[56 + woff_17 + 3];
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
            float2 _f2_206 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_15 = _f2_206;
            float2 _f2_207 = make_float2(q[8], q[9]);
            float2 qp_16 = _f2_207;
            sq[4] = fma_f32x2_rn_noftz(v_15, v_15, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_15, qp_16, dot[4]);
            float2 _f2_208 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_0_13 = _f2_208;
            float2 _f2_209 = make_float2(q[10], q[11]);
            float2 qp_1_13 = _f2_209;
            sq[4] = fma_f32x2_rn_noftz(v_0_13, v_0_13, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_13, qp_1_13, dot[4]);
            float2 _f2_210 = make_float2(sw_22_f32[4], sw_22_f32[5]);
            float2 v_2_13 = _f2_210;
            float2 _f2_211 = make_float2(q[12], q[13]);
            float2 qp_3_13 = _f2_211;
            sq[4] = fma_f32x2_rn_noftz(v_2_13, v_2_13, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_13, qp_3_13, dot[4]);
            float2 _f2_212 = make_float2(sw_22_f32[6], sw_22_f32[7]);
            float2 v_4_13 = _f2_212;
            float2 _f2_213 = make_float2(q[14], q[15]);
            float2 qp_5_13 = _f2_213;
            sq[4] = fma_f32x2_rn_noftz(v_4_13, v_4_13, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_13, qp_5_13, dot[4]);
        }
        unsigned int sw_23[4];
        sw_23[0] = words[70 + woff_17];
        sw_23[1] = words[70 + woff_17 + 1];
        sw_23[2] = words[70 + woff_17 + 2];
        sw_23[3] = words[70 + woff_17 + 3];
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
            float2 _f2_220 = make_float2(sw_23_f32[0], sw_23_f32[1]);
            float2 v_16 = _f2_220;
            float2 _f2_221 = make_float2(q[8], q[9]);
            float2 qp_17 = _f2_221;
            sq[5] = fma_f32x2_rn_noftz(v_16, v_16, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_16, qp_17, dot[5]);
            float2 _f2_222 = make_float2(sw_23_f32[2], sw_23_f32[3]);
            float2 v_0_14 = _f2_222;
            float2 _f2_223 = make_float2(q[10], q[11]);
            float2 qp_1_14 = _f2_223;
            sq[5] = fma_f32x2_rn_noftz(v_0_14, v_0_14, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_14, qp_1_14, dot[5]);
            float2 _f2_224 = make_float2(sw_23_f32[4], sw_23_f32[5]);
            float2 v_2_14 = _f2_224;
            float2 _f2_225 = make_float2(q[12], q[13]);
            float2 qp_3_14 = _f2_225;
            sq[5] = fma_f32x2_rn_noftz(v_2_14, v_2_14, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_14, qp_3_14, dot[5]);
            float2 _f2_226 = make_float2(sw_23_f32[6], sw_23_f32[7]);
            float2 v_4_14 = _f2_226;
            float2 _f2_227 = make_float2(q[14], q[15]);
            float2 qp_5_14 = _f2_227;
            sq[5] = fma_f32x2_rn_noftz(v_4_14, v_4_14, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_14, qp_5_14, dot[5]);
        }
        unsigned int sw_24[4];
        sw_24[0] = words[84 + woff_17];
        sw_24[1] = words[84 + woff_17 + 1];
        sw_24[2] = words[84 + woff_17 + 2];
        sw_24[3] = words[84 + woff_17 + 3];
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
            float2 _f2_234 = make_float2(sw_24_f32[0], sw_24_f32[1]);
            float2 v_17 = _f2_234;
            float2 _f2_235 = make_float2(q[8], q[9]);
            float2 qp_18 = _f2_235;
            sq[6] = fma_f32x2_rn_noftz(v_17, v_17, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_17, qp_18, dot[6]);
            float2 _f2_236 = make_float2(sw_24_f32[2], sw_24_f32[3]);
            float2 v_0_15 = _f2_236;
            float2 _f2_237 = make_float2(q[10], q[11]);
            float2 qp_1_15 = _f2_237;
            sq[6] = fma_f32x2_rn_noftz(v_0_15, v_0_15, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_15, qp_1_15, dot[6]);
            float2 _f2_238 = make_float2(sw_24_f32[4], sw_24_f32[5]);
            float2 v_2_15 = _f2_238;
            float2 _f2_239 = make_float2(q[12], q[13]);
            float2 qp_3_15 = _f2_239;
            sq[6] = fma_f32x2_rn_noftz(v_2_15, v_2_15, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_15, qp_3_15, dot[6]);
            float2 _f2_240 = make_float2(sw_24_f32[6], sw_24_f32[7]);
            float2 v_4_15 = _f2_240;
            float2 _f2_241 = make_float2(q[14], q[15]);
            float2 qp_5_15 = _f2_241;
            sq[6] = fma_f32x2_rn_noftz(v_4_15, v_4_15, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_15, qp_5_15, dot[6]);
        }
        unsigned int sw_25[4];
        sw_25[0] = words[98 + woff_17];
        sw_25[1] = words[98 + woff_17 + 1];
        sw_25[2] = words[98 + woff_17 + 2];
        sw_25[3] = words[98 + woff_17 + 3];
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
            float2 _f2_248 = make_float2(sw_25_f32[0], sw_25_f32[1]);
            float2 v_18 = _f2_248;
            float2 _f2_249 = make_float2(q[8], q[9]);
            float2 qp_19 = _f2_249;
            sq[7] = fma_f32x2_rn_noftz(v_18, v_18, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_18, qp_19, dot[7]);
            float2 _f2_250 = make_float2(sw_25_f32[2], sw_25_f32[3]);
            float2 v_0_16 = _f2_250;
            float2 _f2_251 = make_float2(q[10], q[11]);
            float2 qp_1_16 = _f2_251;
            sq[7] = fma_f32x2_rn_noftz(v_0_16, v_0_16, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_0_16, qp_1_16, dot[7]);
            float2 _f2_252 = make_float2(sw_25_f32[4], sw_25_f32[5]);
            float2 v_2_16 = _f2_252;
            float2 _f2_253 = make_float2(q[12], q[13]);
            float2 qp_3_16 = _f2_253;
            sq[7] = fma_f32x2_rn_noftz(v_2_16, v_2_16, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_2_16, qp_3_16, dot[7]);
            float2 _f2_254 = make_float2(sw_25_f32[6], sw_25_f32[7]);
            float2 v_4_16 = _f2_254;
            float2 _f2_255 = make_float2(q[14], q[15]);
            float2 qp_5_16 = _f2_255;
            sq[7] = fma_f32x2_rn_noftz(v_4_16, v_4_16, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_4_16, qp_5_16, dot[7]);
        }
        unsigned int sw_26[4];
        sw_26[0] = words[112 + woff_17];
        sw_26[1] = words[112 + woff_17 + 1];
        sw_26[2] = words[112 + woff_17 + 2];
        sw_26[3] = words[112 + woff_17 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_26[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_17]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_26[0] = __as_u32(mixed_1);
            words[112 + woff_17] = sw_26[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_26[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_17 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_26[1] = __as_u32(mixed_2_1);
            words[112 + woff_17 + 1] = sw_26[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_26[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_17 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_26[2] = __as_u32(mixed_5_1);
            words[112 + woff_17 + 2] = sw_26[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_26[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_17 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_26[3] = __as_u32(mixed_8_1);
            words[112 + woff_17 + 3] = sw_26[3];
            {
                int4 _iv4 = make_int4(sw_26[0 + 0], sw_26[0 + 1], sw_26[0 + 2], sw_26[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_16)) + 0) = _iv4;
            }
        }
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
            float2 _f2_262 = make_float2(sw_26_f32[0], sw_26_f32[1]);
            float2 v_19 = _f2_262;
            float2 _f2_263 = make_float2(q[8], q[9]);
            float2 qp_20 = _f2_263;
            sq[8] = fma_f32x2_rn_noftz(v_19, v_19, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_19, qp_20, dot[8]);
            float2 _f2_264 = make_float2(sw_26_f32[2], sw_26_f32[3]);
            float2 v_0_17 = _f2_264;
            float2 _f2_265 = make_float2(q[10], q[11]);
            float2 qp_1_17 = _f2_265;
            sq[8] = fma_f32x2_rn_noftz(v_0_17, v_0_17, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_0_17, qp_1_17, dot[8]);
            float2 _f2_266 = make_float2(sw_26_f32[4], sw_26_f32[5]);
            float2 v_2_17 = _f2_266;
            float2 _f2_267 = make_float2(q[12], q[13]);
            float2 qp_3_17 = _f2_267;
            sq[8] = fma_f32x2_rn_noftz(v_2_17, v_2_17, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_2_17, qp_3_17, dot[8]);
            float2 _f2_268 = make_float2(sw_26_f32[6], sw_26_f32[7]);
            float2 v_4_17 = _f2_268;
            float2 _f2_269 = make_float2(q[14], q[15]);
            float2 qp_5_17 = _f2_269;
            sq[8] = fma_f32x2_rn_noftz(v_4_17, v_4_17, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_4_17, qp_5_17, dot[8]);
        }
        int base_27 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_28 = 8;
        unsigned int sw_29[4];
        sw_29[0] = words[woff_28];
        sw_29[1] = words[woff_28 + 1];
        sw_29[2] = words[woff_28 + 2];
        sw_29[3] = words[woff_28 + 3];
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
            fsrc[16] = sw_29_f32[0];
            fsrc[17] = sw_29_f32[1];
            fsrc[18] = sw_29_f32[2];
            fsrc[19] = sw_29_f32[3];
            fsrc[20] = sw_29_f32[4];
            fsrc[21] = sw_29_f32[5];
            fsrc[22] = sw_29_f32[6];
            fsrc[23] = sw_29_f32[7];
        }
        {
            float2 _f2_276 = make_float2(sw_29_f32[0], sw_29_f32[1]);
            float2 v_20 = _f2_276;
            float2 _f2_277 = make_float2(q[16], q[17]);
            float2 qp_21 = _f2_277;
            sq[0] = fma_f32x2_rn_noftz(v_20, v_20, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_20, qp_21, dot[0]);
            float2 _f2_278 = make_float2(sw_29_f32[2], sw_29_f32[3]);
            float2 v_0_18 = _f2_278;
            float2 _f2_279 = make_float2(q[18], q[19]);
            float2 qp_1_18 = _f2_279;
            sq[0] = fma_f32x2_rn_noftz(v_0_18, v_0_18, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_18, qp_1_18, dot[0]);
            float2 _f2_280 = make_float2(sw_29_f32[4], sw_29_f32[5]);
            float2 v_2_18 = _f2_280;
            float2 _f2_281 = make_float2(q[20], q[21]);
            float2 qp_3_18 = _f2_281;
            sq[0] = fma_f32x2_rn_noftz(v_2_18, v_2_18, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_18, qp_3_18, dot[0]);
            float2 _f2_282 = make_float2(sw_29_f32[6], sw_29_f32[7]);
            float2 v_4_18 = _f2_282;
            float2 _f2_283 = make_float2(q[22], q[23]);
            float2 qp_5_18 = _f2_283;
            sq[0] = fma_f32x2_rn_noftz(v_4_18, v_4_18, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_18, qp_5_18, dot[0]);
        }
        unsigned int sw_30[4];
        sw_30[0] = words[14 + woff_28];
        sw_30[1] = words[14 + woff_28 + 1];
        sw_30[2] = words[14 + woff_28 + 2];
        sw_30[3] = words[14 + woff_28 + 3];
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
            fsrc[44] = sw_30_f32[0];
            fsrc[45] = sw_30_f32[1];
            fsrc[46] = sw_30_f32[2];
            fsrc[47] = sw_30_f32[3];
            fsrc[48] = sw_30_f32[4];
            fsrc[49] = sw_30_f32[5];
            fsrc[50] = sw_30_f32[6];
            fsrc[51] = sw_30_f32[7];
        }
        {
            float2 _f2_290 = make_float2(sw_30_f32[0], sw_30_f32[1]);
            float2 v_21 = _f2_290;
            float2 _f2_291 = make_float2(q[16], q[17]);
            float2 qp_22 = _f2_291;
            sq[1] = fma_f32x2_rn_noftz(v_21, v_21, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_21, qp_22, dot[1]);
            float2 _f2_292 = make_float2(sw_30_f32[2], sw_30_f32[3]);
            float2 v_0_19 = _f2_292;
            float2 _f2_293 = make_float2(q[18], q[19]);
            float2 qp_1_19 = _f2_293;
            sq[1] = fma_f32x2_rn_noftz(v_0_19, v_0_19, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_19, qp_1_19, dot[1]);
            float2 _f2_294 = make_float2(sw_30_f32[4], sw_30_f32[5]);
            float2 v_2_19 = _f2_294;
            float2 _f2_295 = make_float2(q[20], q[21]);
            float2 qp_3_19 = _f2_295;
            sq[1] = fma_f32x2_rn_noftz(v_2_19, v_2_19, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_19, qp_3_19, dot[1]);
            float2 _f2_296 = make_float2(sw_30_f32[6], sw_30_f32[7]);
            float2 v_4_19 = _f2_296;
            float2 _f2_297 = make_float2(q[22], q[23]);
            float2 qp_5_19 = _f2_297;
            sq[1] = fma_f32x2_rn_noftz(v_4_19, v_4_19, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_19, qp_5_19, dot[1]);
        }
        unsigned int sw_31[4];
        sw_31[0] = words[28 + woff_28];
        sw_31[1] = words[28 + woff_28 + 1];
        sw_31[2] = words[28 + woff_28 + 2];
        sw_31[3] = words[28 + woff_28 + 3];
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
            fsrc[72] = sw_31_f32[0];
            fsrc[73] = sw_31_f32[1];
            fsrc[74] = sw_31_f32[2];
            fsrc[75] = sw_31_f32[3];
            fsrc[76] = sw_31_f32[4];
            fsrc[77] = sw_31_f32[5];
            fsrc[78] = sw_31_f32[6];
            fsrc[79] = sw_31_f32[7];
        }
        {
            float2 _f2_304 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_22 = _f2_304;
            float2 _f2_305 = make_float2(q[16], q[17]);
            float2 qp_23 = _f2_305;
            sq[2] = fma_f32x2_rn_noftz(v_22, v_22, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_22, qp_23, dot[2]);
            float2 _f2_306 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_0_20 = _f2_306;
            float2 _f2_307 = make_float2(q[18], q[19]);
            float2 qp_1_20 = _f2_307;
            sq[2] = fma_f32x2_rn_noftz(v_0_20, v_0_20, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_20, qp_1_20, dot[2]);
            float2 _f2_308 = make_float2(sw_31_f32[4], sw_31_f32[5]);
            float2 v_2_20 = _f2_308;
            float2 _f2_309 = make_float2(q[20], q[21]);
            float2 qp_3_20 = _f2_309;
            sq[2] = fma_f32x2_rn_noftz(v_2_20, v_2_20, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_20, qp_3_20, dot[2]);
            float2 _f2_310 = make_float2(sw_31_f32[6], sw_31_f32[7]);
            float2 v_4_20 = _f2_310;
            float2 _f2_311 = make_float2(q[22], q[23]);
            float2 qp_5_20 = _f2_311;
            sq[2] = fma_f32x2_rn_noftz(v_4_20, v_4_20, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_20, qp_5_20, dot[2]);
        }
        unsigned int sw_32[4];
        sw_32[0] = words[42 + woff_28];
        sw_32[1] = words[42 + woff_28 + 1];
        sw_32[2] = words[42 + woff_28 + 2];
        sw_32[3] = words[42 + woff_28 + 3];
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
            float2 _f2_318 = make_float2(sw_32_f32[0], sw_32_f32[1]);
            float2 v_23 = _f2_318;
            float2 _f2_319 = make_float2(q[16], q[17]);
            float2 qp_24 = _f2_319;
            sq[3] = fma_f32x2_rn_noftz(v_23, v_23, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_23, qp_24, dot[3]);
            float2 _f2_320 = make_float2(sw_32_f32[2], sw_32_f32[3]);
            float2 v_0_21 = _f2_320;
            float2 _f2_321 = make_float2(q[18], q[19]);
            float2 qp_1_21 = _f2_321;
            sq[3] = fma_f32x2_rn_noftz(v_0_21, v_0_21, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_21, qp_1_21, dot[3]);
            float2 _f2_322 = make_float2(sw_32_f32[4], sw_32_f32[5]);
            float2 v_2_21 = _f2_322;
            float2 _f2_323 = make_float2(q[20], q[21]);
            float2 qp_3_21 = _f2_323;
            sq[3] = fma_f32x2_rn_noftz(v_2_21, v_2_21, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_21, qp_3_21, dot[3]);
            float2 _f2_324 = make_float2(sw_32_f32[6], sw_32_f32[7]);
            float2 v_4_21 = _f2_324;
            float2 _f2_325 = make_float2(q[22], q[23]);
            float2 qp_5_21 = _f2_325;
            sq[3] = fma_f32x2_rn_noftz(v_4_21, v_4_21, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_21, qp_5_21, dot[3]);
        }
        unsigned int sw_33[4];
        sw_33[0] = words[56 + woff_28];
        sw_33[1] = words[56 + woff_28 + 1];
        sw_33[2] = words[56 + woff_28 + 2];
        sw_33[3] = words[56 + woff_28 + 3];
        float sw_33_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_33_f32[_pair * 2])[0]), "=f"((&sw_33_f32[_pair * 2])[1])
                : "r"(sw_33[_pair]));
        }
        {
            float2 _f2_332 = make_float2(sw_33_f32[0], sw_33_f32[1]);
            float2 v_24 = _f2_332;
            float2 _f2_333 = make_float2(q[16], q[17]);
            float2 qp_25 = _f2_333;
            sq[4] = fma_f32x2_rn_noftz(v_24, v_24, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_24, qp_25, dot[4]);
            float2 _f2_334 = make_float2(sw_33_f32[2], sw_33_f32[3]);
            float2 v_0_22 = _f2_334;
            float2 _f2_335 = make_float2(q[18], q[19]);
            float2 qp_1_22 = _f2_335;
            sq[4] = fma_f32x2_rn_noftz(v_0_22, v_0_22, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_22, qp_1_22, dot[4]);
            float2 _f2_336 = make_float2(sw_33_f32[4], sw_33_f32[5]);
            float2 v_2_22 = _f2_336;
            float2 _f2_337 = make_float2(q[20], q[21]);
            float2 qp_3_22 = _f2_337;
            sq[4] = fma_f32x2_rn_noftz(v_2_22, v_2_22, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_22, qp_3_22, dot[4]);
            float2 _f2_338 = make_float2(sw_33_f32[6], sw_33_f32[7]);
            float2 v_4_22 = _f2_338;
            float2 _f2_339 = make_float2(q[22], q[23]);
            float2 qp_5_22 = _f2_339;
            sq[4] = fma_f32x2_rn_noftz(v_4_22, v_4_22, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_22, qp_5_22, dot[4]);
        }
        unsigned int sw_34[4];
        sw_34[0] = words[70 + woff_28];
        sw_34[1] = words[70 + woff_28 + 1];
        sw_34[2] = words[70 + woff_28 + 2];
        sw_34[3] = words[70 + woff_28 + 3];
        float sw_34_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_34_f32[_pair * 2])[0]), "=f"((&sw_34_f32[_pair * 2])[1])
                : "r"(sw_34[_pair]));
        }
        {
            float2 _f2_346 = make_float2(sw_34_f32[0], sw_34_f32[1]);
            float2 v_25 = _f2_346;
            float2 _f2_347 = make_float2(q[16], q[17]);
            float2 qp_26 = _f2_347;
            sq[5] = fma_f32x2_rn_noftz(v_25, v_25, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_25, qp_26, dot[5]);
            float2 _f2_348 = make_float2(sw_34_f32[2], sw_34_f32[3]);
            float2 v_0_23 = _f2_348;
            float2 _f2_349 = make_float2(q[18], q[19]);
            float2 qp_1_23 = _f2_349;
            sq[5] = fma_f32x2_rn_noftz(v_0_23, v_0_23, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_23, qp_1_23, dot[5]);
            float2 _f2_350 = make_float2(sw_34_f32[4], sw_34_f32[5]);
            float2 v_2_23 = _f2_350;
            float2 _f2_351 = make_float2(q[20], q[21]);
            float2 qp_3_23 = _f2_351;
            sq[5] = fma_f32x2_rn_noftz(v_2_23, v_2_23, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_23, qp_3_23, dot[5]);
            float2 _f2_352 = make_float2(sw_34_f32[6], sw_34_f32[7]);
            float2 v_4_23 = _f2_352;
            float2 _f2_353 = make_float2(q[22], q[23]);
            float2 qp_5_23 = _f2_353;
            sq[5] = fma_f32x2_rn_noftz(v_4_23, v_4_23, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_23, qp_5_23, dot[5]);
        }
        unsigned int sw_35[4];
        sw_35[0] = words[84 + woff_28];
        sw_35[1] = words[84 + woff_28 + 1];
        sw_35[2] = words[84 + woff_28 + 2];
        sw_35[3] = words[84 + woff_28 + 3];
        float sw_35_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_35_f32[_pair * 2])[0]), "=f"((&sw_35_f32[_pair * 2])[1])
                : "r"(sw_35[_pair]));
        }
        {
            float2 _f2_360 = make_float2(sw_35_f32[0], sw_35_f32[1]);
            float2 v_26 = _f2_360;
            float2 _f2_361 = make_float2(q[16], q[17]);
            float2 qp_27 = _f2_361;
            sq[6] = fma_f32x2_rn_noftz(v_26, v_26, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_26, qp_27, dot[6]);
            float2 _f2_362 = make_float2(sw_35_f32[2], sw_35_f32[3]);
            float2 v_0_24 = _f2_362;
            float2 _f2_363 = make_float2(q[18], q[19]);
            float2 qp_1_24 = _f2_363;
            sq[6] = fma_f32x2_rn_noftz(v_0_24, v_0_24, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_24, qp_1_24, dot[6]);
            float2 _f2_364 = make_float2(sw_35_f32[4], sw_35_f32[5]);
            float2 v_2_24 = _f2_364;
            float2 _f2_365 = make_float2(q[20], q[21]);
            float2 qp_3_24 = _f2_365;
            sq[6] = fma_f32x2_rn_noftz(v_2_24, v_2_24, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_24, qp_3_24, dot[6]);
            float2 _f2_366 = make_float2(sw_35_f32[6], sw_35_f32[7]);
            float2 v_4_24 = _f2_366;
            float2 _f2_367 = make_float2(q[22], q[23]);
            float2 qp_5_24 = _f2_367;
            sq[6] = fma_f32x2_rn_noftz(v_4_24, v_4_24, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_24, qp_5_24, dot[6]);
        }
        unsigned int sw_36[4];
        sw_36[0] = words[98 + woff_28];
        sw_36[1] = words[98 + woff_28 + 1];
        sw_36[2] = words[98 + woff_28 + 2];
        sw_36[3] = words[98 + woff_28 + 3];
        float sw_36_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_36_f32[_pair * 2])[0]), "=f"((&sw_36_f32[_pair * 2])[1])
                : "r"(sw_36[_pair]));
        }
        {
            float2 _f2_374 = make_float2(sw_36_f32[0], sw_36_f32[1]);
            float2 v_27 = _f2_374;
            float2 _f2_375 = make_float2(q[16], q[17]);
            float2 qp_28 = _f2_375;
            sq[7] = fma_f32x2_rn_noftz(v_27, v_27, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_27, qp_28, dot[7]);
            float2 _f2_376 = make_float2(sw_36_f32[2], sw_36_f32[3]);
            float2 v_0_25 = _f2_376;
            float2 _f2_377 = make_float2(q[18], q[19]);
            float2 qp_1_25 = _f2_377;
            sq[7] = fma_f32x2_rn_noftz(v_0_25, v_0_25, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_0_25, qp_1_25, dot[7]);
            float2 _f2_378 = make_float2(sw_36_f32[4], sw_36_f32[5]);
            float2 v_2_25 = _f2_378;
            float2 _f2_379 = make_float2(q[20], q[21]);
            float2 qp_3_25 = _f2_379;
            sq[7] = fma_f32x2_rn_noftz(v_2_25, v_2_25, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_2_25, qp_3_25, dot[7]);
            float2 _f2_380 = make_float2(sw_36_f32[6], sw_36_f32[7]);
            float2 v_4_25 = _f2_380;
            float2 _f2_381 = make_float2(q[22], q[23]);
            float2 qp_5_25 = _f2_381;
            sq[7] = fma_f32x2_rn_noftz(v_4_25, v_4_25, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_4_25, qp_5_25, dot[7]);
        }
        unsigned int sw_37[4];
        sw_37[0] = words[112 + woff_28];
        sw_37[1] = words[112 + woff_28 + 1];
        sw_37[2] = words[112 + woff_28 + 2];
        sw_37[3] = words[112 + woff_28 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_37[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_28]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_37[0] = __as_u32(mixed_3);
            words[112 + woff_28] = sw_37[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_37[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_28 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_37[1] = __as_u32(mixed_2_2);
            words[112 + woff_28 + 1] = sw_37[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_37[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_28 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_37[2] = __as_u32(mixed_5_2);
            words[112 + woff_28 + 2] = sw_37[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_37[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_28 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_37[3] = __as_u32(mixed_8_2);
            words[112 + woff_28 + 3] = sw_37[3];
            {
                int4 _iv4 = make_int4(sw_37[0 + 0], sw_37[0 + 1], sw_37[0 + 2], sw_37[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_27)) + 0) = _iv4;
            }
        }
        float sw_37_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_37_f32[_pair * 2])[0]), "=f"((&sw_37_f32[_pair * 2])[1])
                : "r"(sw_37[_pair]));
        }
        {
            float2 _f2_388 = make_float2(sw_37_f32[0], sw_37_f32[1]);
            float2 v_28 = _f2_388;
            float2 _f2_389 = make_float2(q[16], q[17]);
            float2 qp_29 = _f2_389;
            sq[8] = fma_f32x2_rn_noftz(v_28, v_28, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_28, qp_29, dot[8]);
            float2 _f2_390 = make_float2(sw_37_f32[2], sw_37_f32[3]);
            float2 v_0_26 = _f2_390;
            float2 _f2_391 = make_float2(q[18], q[19]);
            float2 qp_1_26 = _f2_391;
            sq[8] = fma_f32x2_rn_noftz(v_0_26, v_0_26, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_0_26, qp_1_26, dot[8]);
            float2 _f2_392 = make_float2(sw_37_f32[4], sw_37_f32[5]);
            float2 v_2_26 = _f2_392;
            float2 _f2_393 = make_float2(q[20], q[21]);
            float2 qp_3_26 = _f2_393;
            sq[8] = fma_f32x2_rn_noftz(v_2_26, v_2_26, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_2_26, qp_3_26, dot[8]);
            float2 _f2_394 = make_float2(sw_37_f32[6], sw_37_f32[7]);
            float2 v_4_26 = _f2_394;
            float2 _f2_395 = make_float2(q[22], q[23]);
            float2 qp_5_26 = _f2_395;
            sq[8] = fma_f32x2_rn_noftz(v_4_26, v_4_26, sq[8]);
            dot[8] = fma_f32x2_rn_noftz(v_4_26, qp_5_26, dot[8]);
        }
        int base_38 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_39 = 12;
        unsigned int sw_40[4];
        sw_40[0] = words[woff_39];
        sw_40[1] = words[woff_39 + 1];
        float sw_40_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_40_f32[_pair * 2])[0]), "=f"((&sw_40_f32[_pair * 2])[1])
                : "r"(sw_40[_pair]));
        }
        {
            fsrc[24] = sw_40_f32[0];
            fsrc[25] = sw_40_f32[1];
            fsrc[26] = sw_40_f32[2];
            fsrc[27] = sw_40_f32[3];
        }
        {
            float2 _f2_396 = make_float2(sw_40_f32[0], sw_40_f32[1]);
            float2 v_29 = _f2_396;
            sq[0] = fma_f32x2_rn_noftz(v_29, v_29, sq[0]);
            float2 _f2_397 = make_float2(sw_40_f32[2], sw_40_f32[3]);
            float2 v_0_27 = _f2_397;
            sq[0] = fma_f32x2_rn_noftz(v_0_27, v_0_27, sq[0]);
            float2 _f2_398 = make_float2(sw_40_f32[0], sw_40_f32[1]);
            float2 v_1_1 = _f2_398;
            float2 _f2_399 = make_float2(q[24], q[25]);
            float2 qp_30 = _f2_399;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_30, dot[0]);
            float2 _f2_400 = make_float2(sw_40_f32[2], sw_40_f32[3]);
            float2 v_2_27 = _f2_400;
            float2 _f2_401 = make_float2(q[26], q[27]);
            float2 qp_3_27 = _f2_401;
            dot[0] = fma_f32x2_rn_noftz(v_2_27, qp_3_27, dot[0]);
        }
        unsigned int sw_41[4];
        sw_41[0] = words[14 + woff_39];
        sw_41[1] = words[14 + woff_39 + 1];
        float sw_41_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_41_f32[_pair * 2])[0]), "=f"((&sw_41_f32[_pair * 2])[1])
                : "r"(sw_41[_pair]));
        }
        {
            fsrc[52] = sw_41_f32[0];
            fsrc[53] = sw_41_f32[1];
            fsrc[54] = sw_41_f32[2];
            fsrc[55] = sw_41_f32[3];
        }
        {
            float2 _f2_410 = make_float2(sw_41_f32[0], sw_41_f32[1]);
            float2 v_30 = _f2_410;
            sq[1] = fma_f32x2_rn_noftz(v_30, v_30, sq[1]);
            float2 _f2_411 = make_float2(sw_41_f32[2], sw_41_f32[3]);
            float2 v_0_28 = _f2_411;
            sq[1] = fma_f32x2_rn_noftz(v_0_28, v_0_28, sq[1]);
            float2 _f2_412 = make_float2(sw_41_f32[0], sw_41_f32[1]);
            float2 v_1_2 = _f2_412;
            float2 _f2_413 = make_float2(q[24], q[25]);
            float2 qp_31 = _f2_413;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_31, dot[1]);
            float2 _f2_414 = make_float2(sw_41_f32[2], sw_41_f32[3]);
            float2 v_2_28 = _f2_414;
            float2 _f2_415 = make_float2(q[26], q[27]);
            float2 qp_3_28 = _f2_415;
            dot[1] = fma_f32x2_rn_noftz(v_2_28, qp_3_28, dot[1]);
        }
        unsigned int sw_42[4];
        sw_42[0] = words[28 + woff_39];
        sw_42[1] = words[28 + woff_39 + 1];
        float sw_42_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_42_f32[_pair * 2])[0]), "=f"((&sw_42_f32[_pair * 2])[1])
                : "r"(sw_42[_pair]));
        }
        {
            fsrc[80] = sw_42_f32[0];
            fsrc[81] = sw_42_f32[1];
            fsrc[82] = sw_42_f32[2];
            fsrc[83] = sw_42_f32[3];
        }
        {
            float2 _f2_424 = make_float2(sw_42_f32[0], sw_42_f32[1]);
            float2 v_31 = _f2_424;
            sq[2] = fma_f32x2_rn_noftz(v_31, v_31, sq[2]);
            float2 _f2_425 = make_float2(sw_42_f32[2], sw_42_f32[3]);
            float2 v_0_29 = _f2_425;
            sq[2] = fma_f32x2_rn_noftz(v_0_29, v_0_29, sq[2]);
            float2 _f2_426 = make_float2(sw_42_f32[0], sw_42_f32[1]);
            float2 v_1_3 = _f2_426;
            float2 _f2_427 = make_float2(q[24], q[25]);
            float2 qp_32 = _f2_427;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_32, dot[2]);
            float2 _f2_428 = make_float2(sw_42_f32[2], sw_42_f32[3]);
            float2 v_2_29 = _f2_428;
            float2 _f2_429 = make_float2(q[26], q[27]);
            float2 qp_3_29 = _f2_429;
            dot[2] = fma_f32x2_rn_noftz(v_2_29, qp_3_29, dot[2]);
        }
        unsigned int sw_43[4];
        sw_43[0] = words[42 + woff_39];
        sw_43[1] = words[42 + woff_39 + 1];
        float sw_43_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_43_f32[_pair * 2])[0]), "=f"((&sw_43_f32[_pair * 2])[1])
                : "r"(sw_43[_pair]));
        }
        {
            float2 _f2_438 = make_float2(sw_43_f32[0], sw_43_f32[1]);
            float2 v_32 = _f2_438;
            sq[3] = fma_f32x2_rn_noftz(v_32, v_32, sq[3]);
            float2 _f2_439 = make_float2(sw_43_f32[2], sw_43_f32[3]);
            float2 v_0_30 = _f2_439;
            sq[3] = fma_f32x2_rn_noftz(v_0_30, v_0_30, sq[3]);
            float2 _f2_440 = make_float2(sw_43_f32[0], sw_43_f32[1]);
            float2 v_1_4 = _f2_440;
            float2 _f2_441 = make_float2(q[24], q[25]);
            float2 qp_33 = _f2_441;
            dot[3] = fma_f32x2_rn_noftz(v_1_4, qp_33, dot[3]);
            float2 _f2_442 = make_float2(sw_43_f32[2], sw_43_f32[3]);
            float2 v_2_30 = _f2_442;
            float2 _f2_443 = make_float2(q[26], q[27]);
            float2 qp_3_30 = _f2_443;
            dot[3] = fma_f32x2_rn_noftz(v_2_30, qp_3_30, dot[3]);
        }
        unsigned int sw_44[4];
        sw_44[0] = words[56 + woff_39];
        sw_44[1] = words[56 + woff_39 + 1];
        float sw_44_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_44_f32[_pair * 2])[0]), "=f"((&sw_44_f32[_pair * 2])[1])
                : "r"(sw_44[_pair]));
        }
        {
            float2 _f2_452 = make_float2(sw_44_f32[0], sw_44_f32[1]);
            float2 v_33 = _f2_452;
            sq[4] = fma_f32x2_rn_noftz(v_33, v_33, sq[4]);
            float2 _f2_453 = make_float2(sw_44_f32[2], sw_44_f32[3]);
            float2 v_0_31 = _f2_453;
            sq[4] = fma_f32x2_rn_noftz(v_0_31, v_0_31, sq[4]);
            float2 _f2_454 = make_float2(sw_44_f32[0], sw_44_f32[1]);
            float2 v_1_5 = _f2_454;
            float2 _f2_455 = make_float2(q[24], q[25]);
            float2 qp_34 = _f2_455;
            dot[4] = fma_f32x2_rn_noftz(v_1_5, qp_34, dot[4]);
            float2 _f2_456 = make_float2(sw_44_f32[2], sw_44_f32[3]);
            float2 v_2_31 = _f2_456;
            float2 _f2_457 = make_float2(q[26], q[27]);
            float2 qp_3_31 = _f2_457;
            dot[4] = fma_f32x2_rn_noftz(v_2_31, qp_3_31, dot[4]);
        }
        unsigned int sw_45[4];
        sw_45[0] = words[70 + woff_39];
        sw_45[1] = words[70 + woff_39 + 1];
        float sw_45_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_45_f32[_pair * 2])[0]), "=f"((&sw_45_f32[_pair * 2])[1])
                : "r"(sw_45[_pair]));
        }
        {
            float2 _f2_466 = make_float2(sw_45_f32[0], sw_45_f32[1]);
            float2 v_34 = _f2_466;
            sq[5] = fma_f32x2_rn_noftz(v_34, v_34, sq[5]);
            float2 _f2_467 = make_float2(sw_45_f32[2], sw_45_f32[3]);
            float2 v_0_32 = _f2_467;
            sq[5] = fma_f32x2_rn_noftz(v_0_32, v_0_32, sq[5]);
            float2 _f2_468 = make_float2(sw_45_f32[0], sw_45_f32[1]);
            float2 v_1_6 = _f2_468;
            float2 _f2_469 = make_float2(q[24], q[25]);
            float2 qp_35 = _f2_469;
            dot[5] = fma_f32x2_rn_noftz(v_1_6, qp_35, dot[5]);
            float2 _f2_470 = make_float2(sw_45_f32[2], sw_45_f32[3]);
            float2 v_2_32 = _f2_470;
            float2 _f2_471 = make_float2(q[26], q[27]);
            float2 qp_3_32 = _f2_471;
            dot[5] = fma_f32x2_rn_noftz(v_2_32, qp_3_32, dot[5]);
        }
        unsigned int sw_46[4];
        sw_46[0] = words[84 + woff_39];
        sw_46[1] = words[84 + woff_39 + 1];
        float sw_46_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_46_f32[_pair * 2])[0]), "=f"((&sw_46_f32[_pair * 2])[1])
                : "r"(sw_46[_pair]));
        }
        {
            float2 _f2_480 = make_float2(sw_46_f32[0], sw_46_f32[1]);
            float2 v_35 = _f2_480;
            sq[6] = fma_f32x2_rn_noftz(v_35, v_35, sq[6]);
            float2 _f2_481 = make_float2(sw_46_f32[2], sw_46_f32[3]);
            float2 v_0_33 = _f2_481;
            sq[6] = fma_f32x2_rn_noftz(v_0_33, v_0_33, sq[6]);
            float2 _f2_482 = make_float2(sw_46_f32[0], sw_46_f32[1]);
            float2 v_1_7 = _f2_482;
            float2 _f2_483 = make_float2(q[24], q[25]);
            float2 qp_36 = _f2_483;
            dot[6] = fma_f32x2_rn_noftz(v_1_7, qp_36, dot[6]);
            float2 _f2_484 = make_float2(sw_46_f32[2], sw_46_f32[3]);
            float2 v_2_33 = _f2_484;
            float2 _f2_485 = make_float2(q[26], q[27]);
            float2 qp_3_33 = _f2_485;
            dot[6] = fma_f32x2_rn_noftz(v_2_33, qp_3_33, dot[6]);
        }
        unsigned int sw_47[4];
        sw_47[0] = words[98 + woff_39];
        sw_47[1] = words[98 + woff_39 + 1];
        float sw_47_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_47_f32[_pair * 2])[0]), "=f"((&sw_47_f32[_pair * 2])[1])
                : "r"(sw_47[_pair]));
        }
        {
            float2 _f2_494 = make_float2(sw_47_f32[0], sw_47_f32[1]);
            float2 v_36 = _f2_494;
            sq[7] = fma_f32x2_rn_noftz(v_36, v_36, sq[7]);
            float2 _f2_495 = make_float2(sw_47_f32[2], sw_47_f32[3]);
            float2 v_0_34 = _f2_495;
            sq[7] = fma_f32x2_rn_noftz(v_0_34, v_0_34, sq[7]);
            float2 _f2_496 = make_float2(sw_47_f32[0], sw_47_f32[1]);
            float2 v_1_8 = _f2_496;
            float2 _f2_497 = make_float2(q[24], q[25]);
            float2 qp_37 = _f2_497;
            dot[7] = fma_f32x2_rn_noftz(v_1_8, qp_37, dot[7]);
            float2 _f2_498 = make_float2(sw_47_f32[2], sw_47_f32[3]);
            float2 v_2_34 = _f2_498;
            float2 _f2_499 = make_float2(q[26], q[27]);
            float2 qp_3_34 = _f2_499;
            dot[7] = fma_f32x2_rn_noftz(v_2_34, qp_3_34, dot[7]);
        }
        unsigned int sw_48[4];
        sw_48[0] = words[112 + woff_39];
        sw_48[1] = words[112 + woff_39 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_48[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_39]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_48[0] = __as_u32(mixed_4);
            words[112 + woff_39] = sw_48[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_48[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_39 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_48[1] = __as_u32(mixed_2_3);
            words[112 + woff_39 + 1] = sw_48[1];
            {
                int2 _iv2 = make_int2(sw_48[0 + 0], sw_48[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_38)) + 0) = _iv2;
            }
        }
        float sw_48_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_48_f32[_pair * 2])[0]), "=f"((&sw_48_f32[_pair * 2])[1])
                : "r"(sw_48[_pair]));
        }
        {
            float2 _f2_508 = make_float2(sw_48_f32[0], sw_48_f32[1]);
            float2 v_37 = _f2_508;
            sq[8] = fma_f32x2_rn_noftz(v_37, v_37, sq[8]);
            float2 _f2_509 = make_float2(sw_48_f32[2], sw_48_f32[3]);
            float2 v_0_35 = _f2_509;
            sq[8] = fma_f32x2_rn_noftz(v_0_35, v_0_35, sq[8]);
            float2 _f2_510 = make_float2(sw_48_f32[0], sw_48_f32[1]);
            float2 v_1_9 = _f2_510;
            float2 _f2_511 = make_float2(q[24], q[25]);
            float2 qp_38 = _f2_511;
            dot[8] = fma_f32x2_rn_noftz(v_1_9, qp_38, dot[8]);
            float2 _f2_512 = make_float2(sw_48_f32[2], sw_48_f32[3]);
            float2 v_2_35 = _f2_512;
            float2 _f2_513 = make_float2(q[26], q[27]);
            float2 qp_3_35 = _f2_513;
            dot[8] = fma_f32x2_rn_noftz(v_2_35, qp_3_35, dot[8]);
        }
        float2 pairs[9];
        float2 _f2_522 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_522;
        float2 _f2_523 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_523;
        float2 _f2_524 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_524;
        float2 _f2_525 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_525;
        float2 _f2_526 = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[4] = _f2_526;
        float2 _f2_527 = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[5] = _f2_527;
        float2 _f2_528 = make_float2(sq[6].x + sq[6].y, dot[6].x + dot[6].y);
        pairs[6] = _f2_528;
        float2 _f2_529 = make_float2(sq[7].x + sq[7].y, dot[7].x + dot[7].y);
        pairs[7] = _f2_529;
        float2 _f2_530 = make_float2(sq[8].x + sq[8].y, dot[8].x + dot[8].y);
        pairs[8] = _f2_530;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_531 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_531;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_49 = 0;
        bits_49 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_49, 16);
        unsigned long long peerbits_50 = _shfl_xor_1;
        float2 _f2_532 = make_float2(0.0f, 0.0f);
        float2 peer_51 = _f2_532;
        peer_51 = reinterpret_cast<float2*>(&peerbits_50)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_51);
        unsigned long long bits_52 = 0;
        bits_52 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_52, 16);
        unsigned long long peerbits_53 = _shfl_xor_2;
        float2 _f2_533 = make_float2(0.0f, 0.0f);
        float2 peer_54 = _f2_533;
        peer_54 = reinterpret_cast<float2*>(&peerbits_53)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_54);
        unsigned long long bits_55 = 0;
        bits_55 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_55, 16);
        unsigned long long peerbits_56 = _shfl_xor_3;
        float2 _f2_534 = make_float2(0.0f, 0.0f);
        float2 peer_57 = _f2_534;
        peer_57 = reinterpret_cast<float2*>(&peerbits_56)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_57);
        unsigned long long bits_58 = 0;
        bits_58 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_58, 16);
        unsigned long long peerbits_59 = _shfl_xor_4;
        float2 _f2_535 = make_float2(0.0f, 0.0f);
        float2 peer_60 = _f2_535;
        peer_60 = reinterpret_cast<float2*>(&peerbits_59)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_60);
        unsigned long long bits_61 = 0;
        bits_61 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_61, 16);
        unsigned long long peerbits_62 = _shfl_xor_5;
        float2 _f2_536 = make_float2(0.0f, 0.0f);
        float2 peer_63 = _f2_536;
        peer_63 = reinterpret_cast<float2*>(&peerbits_62)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_63);
        unsigned long long bits_64 = 0;
        bits_64 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_64, 16);
        unsigned long long peerbits_65 = _shfl_xor_6;
        float2 _f2_537 = make_float2(0.0f, 0.0f);
        float2 peer_66 = _f2_537;
        peer_66 = reinterpret_cast<float2*>(&peerbits_65)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_66);
        unsigned long long bits_67 = 0;
        bits_67 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_67, 16);
        unsigned long long peerbits_68 = _shfl_xor_7;
        float2 _f2_538 = make_float2(0.0f, 0.0f);
        float2 peer_69 = _f2_538;
        peer_69 = reinterpret_cast<float2*>(&peerbits_68)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_69);
        unsigned long long bits_70 = 0;
        bits_70 = reinterpret_cast<unsigned long long*>(&pairs[8])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_70, 16);
        unsigned long long peerbits_71 = _shfl_xor_8;
        float2 _f2_539 = make_float2(0.0f, 0.0f);
        float2 peer_72 = _f2_539;
        peer_72 = reinterpret_cast<float2*>(&peerbits_71)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_72);
        unsigned long long bits_73 = 0;
        bits_73 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_73, 8);
        unsigned long long peerbits_74 = _shfl_xor_9;
        float2 _f2_540 = make_float2(0.0f, 0.0f);
        float2 peer_75 = _f2_540;
        peer_75 = reinterpret_cast<float2*>(&peerbits_74)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_75);
        unsigned long long bits_76 = 0;
        bits_76 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_76, 8);
        unsigned long long peerbits_77 = _shfl_xor_10;
        float2 _f2_541 = make_float2(0.0f, 0.0f);
        float2 peer_78 = _f2_541;
        peer_78 = reinterpret_cast<float2*>(&peerbits_77)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_78);
        unsigned long long bits_79 = 0;
        bits_79 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_79, 8);
        unsigned long long peerbits_80 = _shfl_xor_11;
        float2 _f2_542 = make_float2(0.0f, 0.0f);
        float2 peer_81 = _f2_542;
        peer_81 = reinterpret_cast<float2*>(&peerbits_80)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_81);
        unsigned long long bits_82 = 0;
        bits_82 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_82, 8);
        unsigned long long peerbits_83 = _shfl_xor_12;
        float2 _f2_543 = make_float2(0.0f, 0.0f);
        float2 peer_84 = _f2_543;
        peer_84 = reinterpret_cast<float2*>(&peerbits_83)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_84);
        unsigned long long bits_85 = 0;
        bits_85 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_85, 8);
        unsigned long long peerbits_86 = _shfl_xor_13;
        float2 _f2_544 = make_float2(0.0f, 0.0f);
        float2 peer_87 = _f2_544;
        peer_87 = reinterpret_cast<float2*>(&peerbits_86)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_87);
        unsigned long long bits_88 = 0;
        bits_88 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_88, 8);
        unsigned long long peerbits_89 = _shfl_xor_14;
        float2 _f2_545 = make_float2(0.0f, 0.0f);
        float2 peer_90 = _f2_545;
        peer_90 = reinterpret_cast<float2*>(&peerbits_89)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_90);
        unsigned long long bits_91 = 0;
        bits_91 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_91, 8);
        unsigned long long peerbits_92 = _shfl_xor_15;
        float2 _f2_546 = make_float2(0.0f, 0.0f);
        float2 peer_93 = _f2_546;
        peer_93 = reinterpret_cast<float2*>(&peerbits_92)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_93);
        unsigned long long bits_94 = 0;
        bits_94 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_94, 8);
        unsigned long long peerbits_95 = _shfl_xor_16;
        float2 _f2_547 = make_float2(0.0f, 0.0f);
        float2 peer_96 = _f2_547;
        peer_96 = reinterpret_cast<float2*>(&peerbits_95)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_96);
        unsigned long long bits_97 = 0;
        bits_97 = reinterpret_cast<unsigned long long*>(&pairs[8])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_97, 8);
        unsigned long long peerbits_98 = _shfl_xor_17;
        float2 _f2_548 = make_float2(0.0f, 0.0f);
        float2 peer_99 = _f2_548;
        peer_99 = reinterpret_cast<float2*>(&peerbits_98)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_99);
        unsigned long long bits_100 = 0;
        bits_100 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_100, 4);
        unsigned long long peerbits_101 = _shfl_xor_18;
        float2 _f2_549 = make_float2(0.0f, 0.0f);
        float2 peer_102 = _f2_549;
        peer_102 = reinterpret_cast<float2*>(&peerbits_101)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_102);
        unsigned long long bits_103 = 0;
        bits_103 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_103, 4);
        unsigned long long peerbits_104 = _shfl_xor_19;
        float2 _f2_550 = make_float2(0.0f, 0.0f);
        float2 peer_105 = _f2_550;
        peer_105 = reinterpret_cast<float2*>(&peerbits_104)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_105);
        unsigned long long bits_106 = 0;
        bits_106 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, bits_106, 4);
        unsigned long long peerbits_107 = _shfl_xor_20;
        float2 _f2_551 = make_float2(0.0f, 0.0f);
        float2 peer_108 = _f2_551;
        peer_108 = reinterpret_cast<float2*>(&peerbits_107)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_108);
        unsigned long long bits_109 = 0;
        bits_109 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, bits_109, 4);
        unsigned long long peerbits_110 = _shfl_xor_21;
        float2 _f2_552 = make_float2(0.0f, 0.0f);
        float2 peer_111 = _f2_552;
        peer_111 = reinterpret_cast<float2*>(&peerbits_110)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_111);
        unsigned long long bits_112 = 0;
        bits_112 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, bits_112, 4);
        unsigned long long peerbits_113 = _shfl_xor_22;
        float2 _f2_553 = make_float2(0.0f, 0.0f);
        float2 peer_114 = _f2_553;
        peer_114 = reinterpret_cast<float2*>(&peerbits_113)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_114);
        unsigned long long bits_115 = 0;
        bits_115 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, bits_115, 4);
        unsigned long long peerbits_116 = _shfl_xor_23;
        float2 _f2_554 = make_float2(0.0f, 0.0f);
        float2 peer_117 = _f2_554;
        peer_117 = reinterpret_cast<float2*>(&peerbits_116)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_117);
        unsigned long long bits_118 = 0;
        bits_118 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, bits_118, 4);
        unsigned long long peerbits_119 = _shfl_xor_24;
        float2 _f2_555 = make_float2(0.0f, 0.0f);
        float2 peer_120 = _f2_555;
        peer_120 = reinterpret_cast<float2*>(&peerbits_119)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_120);
        unsigned long long bits_121 = 0;
        bits_121 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, bits_121, 4);
        unsigned long long peerbits_122 = _shfl_xor_25;
        float2 _f2_556 = make_float2(0.0f, 0.0f);
        float2 peer_123 = _f2_556;
        peer_123 = reinterpret_cast<float2*>(&peerbits_122)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_123);
        unsigned long long bits_124 = 0;
        bits_124 = reinterpret_cast<unsigned long long*>(&pairs[8])[0];
        unsigned long long _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, bits_124, 4);
        unsigned long long peerbits_125 = _shfl_xor_26;
        float2 _f2_557 = make_float2(0.0f, 0.0f);
        float2 peer_126 = _f2_557;
        peer_126 = reinterpret_cast<float2*>(&peerbits_125)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_126);
        unsigned long long bits_127 = 0;
        bits_127 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, bits_127, 2);
        unsigned long long peerbits_128 = _shfl_xor_27;
        float2 _f2_558 = make_float2(0.0f, 0.0f);
        float2 peer_129 = _f2_558;
        peer_129 = reinterpret_cast<float2*>(&peerbits_128)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_129);
        unsigned long long bits_130 = 0;
        bits_130 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, bits_130, 2);
        unsigned long long peerbits_131 = _shfl_xor_28;
        float2 _f2_559 = make_float2(0.0f, 0.0f);
        float2 peer_132 = _f2_559;
        peer_132 = reinterpret_cast<float2*>(&peerbits_131)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_132);
        unsigned long long bits_133 = 0;
        bits_133 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, bits_133, 2);
        unsigned long long peerbits_134 = _shfl_xor_29;
        float2 _f2_560 = make_float2(0.0f, 0.0f);
        float2 peer_135 = _f2_560;
        peer_135 = reinterpret_cast<float2*>(&peerbits_134)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_135);
        unsigned long long bits_136 = 0;
        bits_136 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, bits_136, 2);
        unsigned long long peerbits_137 = _shfl_xor_30;
        float2 _f2_561 = make_float2(0.0f, 0.0f);
        float2 peer_138 = _f2_561;
        peer_138 = reinterpret_cast<float2*>(&peerbits_137)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_138);
        unsigned long long bits_139 = 0;
        bits_139 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, bits_139, 2);
        unsigned long long peerbits_140 = _shfl_xor_31;
        float2 _f2_562 = make_float2(0.0f, 0.0f);
        float2 peer_141 = _f2_562;
        peer_141 = reinterpret_cast<float2*>(&peerbits_140)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_141);
        unsigned long long bits_142 = 0;
        bits_142 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, bits_142, 2);
        unsigned long long peerbits_143 = _shfl_xor_32;
        float2 _f2_563 = make_float2(0.0f, 0.0f);
        float2 peer_144 = _f2_563;
        peer_144 = reinterpret_cast<float2*>(&peerbits_143)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_144);
        unsigned long long bits_145 = 0;
        bits_145 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, bits_145, 2);
        unsigned long long peerbits_146 = _shfl_xor_33;
        float2 _f2_564 = make_float2(0.0f, 0.0f);
        float2 peer_147 = _f2_564;
        peer_147 = reinterpret_cast<float2*>(&peerbits_146)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_147);
        unsigned long long bits_148 = 0;
        bits_148 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, bits_148, 2);
        unsigned long long peerbits_149 = _shfl_xor_34;
        float2 _f2_565 = make_float2(0.0f, 0.0f);
        float2 peer_150 = _f2_565;
        peer_150 = reinterpret_cast<float2*>(&peerbits_149)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_150);
        unsigned long long bits_151 = 0;
        bits_151 = reinterpret_cast<unsigned long long*>(&pairs[8])[0];
        unsigned long long _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, bits_151, 2);
        unsigned long long peerbits_152 = _shfl_xor_35;
        float2 _f2_566 = make_float2(0.0f, 0.0f);
        float2 peer_153 = _f2_566;
        peer_153 = reinterpret_cast<float2*>(&peerbits_152)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_153);
        unsigned long long bits_154 = 0;
        bits_154 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, bits_154, 1);
        unsigned long long peerbits_155 = _shfl_xor_36;
        float2 _f2_567 = make_float2(0.0f, 0.0f);
        float2 peer_156 = _f2_567;
        peer_156 = reinterpret_cast<float2*>(&peerbits_155)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_156);
        unsigned long long bits_157 = 0;
        bits_157 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, bits_157, 1);
        unsigned long long peerbits_158 = _shfl_xor_37;
        float2 _f2_568 = make_float2(0.0f, 0.0f);
        float2 peer_159 = _f2_568;
        peer_159 = reinterpret_cast<float2*>(&peerbits_158)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_159);
        unsigned long long bits_160 = 0;
        bits_160 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, bits_160, 1);
        unsigned long long peerbits_161 = _shfl_xor_38;
        float2 _f2_569 = make_float2(0.0f, 0.0f);
        float2 peer_162 = _f2_569;
        peer_162 = reinterpret_cast<float2*>(&peerbits_161)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_162);
        unsigned long long bits_163 = 0;
        bits_163 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, bits_163, 1);
        unsigned long long peerbits_164 = _shfl_xor_39;
        float2 _f2_570 = make_float2(0.0f, 0.0f);
        float2 peer_165 = _f2_570;
        peer_165 = reinterpret_cast<float2*>(&peerbits_164)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_165);
        unsigned long long bits_166 = 0;
        bits_166 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, bits_166, 1);
        unsigned long long peerbits_167 = _shfl_xor_40;
        float2 _f2_571 = make_float2(0.0f, 0.0f);
        float2 peer_168 = _f2_571;
        peer_168 = reinterpret_cast<float2*>(&peerbits_167)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_168);
        unsigned long long bits_169 = 0;
        bits_169 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, bits_169, 1);
        unsigned long long peerbits_170 = _shfl_xor_41;
        float2 _f2_572 = make_float2(0.0f, 0.0f);
        float2 peer_171 = _f2_572;
        peer_171 = reinterpret_cast<float2*>(&peerbits_170)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_171);
        unsigned long long bits_172 = 0;
        bits_172 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, bits_172, 1);
        unsigned long long peerbits_173 = _shfl_xor_42;
        float2 _f2_573 = make_float2(0.0f, 0.0f);
        float2 peer_174 = _f2_573;
        peer_174 = reinterpret_cast<float2*>(&peerbits_173)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_174);
        unsigned long long bits_175 = 0;
        bits_175 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, bits_175, 1);
        unsigned long long peerbits_176 = _shfl_xor_43;
        float2 _f2_574 = make_float2(0.0f, 0.0f);
        float2 peer_177 = _f2_574;
        peer_177 = reinterpret_cast<float2*>(&peerbits_176)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_177);
        unsigned long long bits_178 = 0;
        bits_178 = reinterpret_cast<unsigned long long*>(&pairs[8])[0];
        unsigned long long _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, bits_178, 1);
        unsigned long long peerbits_179 = _shfl_xor_44;
        float2 _f2_575 = make_float2(0.0f, 0.0f);
        float2 peer_180 = _f2_575;
        peer_180 = reinterpret_cast<float2*>(&peerbits_179)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_180);
        if (lane == 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(stats_addr + (unsigned int)(warp_0 * 9 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_0), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(stats_addr + (unsigned int)((warp_0 * 9 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_1), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_2), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_3), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_4) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_4), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_5;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_5) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_5), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_6) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_6), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_7;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_7) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_7), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_8) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_8), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_9) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_9), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_10) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_10), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_11) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_11), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_12;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_12) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 6) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_12), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_13;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_13) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 6) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_13), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_14;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_14) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 7) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_14), "f"(pairs[7].x) : "memory");
            uint32_t _mapa_15;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_15) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 7) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_15), "f"(pairs[7].y) : "memory");
            uint32_t _mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_16) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 8) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_16), "f"(pairs[8].x) : "memory");
            uint32_t _mapa_17;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_17) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 8) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_17), "f"(pairs[8].y) : "memory");
            uint32_t _mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_18) : "r"(stats_addr + (unsigned int)(warp_0 * 9 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_18), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_19;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_19) : "r"(stats_addr + (unsigned int)((warp_0 * 9 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_19), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_20) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_20), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_21;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_21) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_21), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_22) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_22), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_23;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_23) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_23), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_24;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_24) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_24), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_25;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_25) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_25), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_26;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_26) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_26), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_27;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_27) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_27), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_28;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_28) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_28), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_29;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_29) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_29), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_30;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_30) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 6) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_30), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_31;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_31) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 6) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_31), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_32;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_32) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 7) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_32), "f"(pairs[7].x) : "memory");
            uint32_t _mapa_33;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_33) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 7) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_33), "f"(pairs[7].y) : "memory");
            uint32_t _mapa_34;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_34) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 8) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_34), "f"(pairs[8].x) : "memory");
            uint32_t _mapa_35;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_35) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 8) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_35), "f"(pairs[8].y) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 9) {
            total_sq = stats[(stat_w * 9 + stat_n) * 2];
            total_dot = stats[(stat_w * 9 + stat_n) * 2 + 1];
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
        if (stat_n < 9 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[9];
        float _shfl_0;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(logit), "r"(0));
        logits[0] = _shfl_0;
        float _shfl_1;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(logit), "r"(8));
        logits[1] = _shfl_1;
        float _shfl_2;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(logit), "r"(16));
        logits[2] = _shfl_2;
        float _shfl_3;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_3) : "f"(logit), "r"(24));
        logits[3] = _shfl_3;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 4) {
            total_sq2 = stats[(stat_w * 9 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 9 + 4 + stat_n) * 2 + 1];
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
        if (stat_n < 4 && stat_w == 0) {
            float _rsqrt_1 = rsqrtf(total_sq2 / 7168.0f + eps);
            float sigma2 = _rsqrt_1;
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_4;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_4) : "f"(logit2), "r"(0));
        logits[4] = _shfl_4;
        float _shfl_5;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_5) : "f"(logit2), "r"(8));
        logits[5] = _shfl_5;
        float _shfl_6;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_6) : "f"(logit2), "r"(16));
        logits[6] = _shfl_6;
        float _shfl_7;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_7) : "f"(logit2), "r"(24));
        logits[7] = _shfl_7;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 1) {
            total_sq2 = stats[(stat_w * 9 + 8 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 9 + 8 + stat_n) * 2 + 1];
        }
        float _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_12;
        float _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_13;
        float _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_14;
        float _shfl_down_15 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_15;
        float _shfl_down_16 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_16;
        float _shfl_down_17 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_17;
        logit2 = 0.0f;
        if (stat_n < 1 && stat_w == 0) {
            float _rsqrt_2 = rsqrtf(total_sq2 / 7168.0f + eps);
            float sigma2_1 = _rsqrt_2;
            logit2 = total_dot2 * sigma2_1;
        }
        float _shfl_8;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_8) : "f"(logit2), "r"(0));
        logits[8] = _shfl_8;
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
        float weights[9];
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
        float2 _f2_576 = make_float2(correction, correction);
        float2 corr = _f2_576;
        const int woff_181 = 0;
        int base_182 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_577 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_577;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_578 = make_float2(acc[2], acc[3]);
        float2 previous_183 = _f2_578;
        a_5[1] = mul_f32x2_noftz(previous_183, corr);
        float2 _f2_579 = make_float2(acc[4], acc[5]);
        float2 previous_184 = _f2_579;
        a_5[2] = mul_f32x2_noftz(previous_184, corr);
        float2 _f2_580 = make_float2(acc[6], acc[7]);
        float2 previous_185 = _f2_580;
        a_5[3] = mul_f32x2_noftz(previous_185, corr);
        float2 _f2_581 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_581;
        {
            float2 _f2_582 = make_float2(fsrc[0], fsrc[1]);
            float2 v_38 = _f2_582;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_38, a_5[0]);
            float2 _f2_583 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_36 = _f2_583;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_36, a_5[1]);
            float2 _f2_584 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_10 = _f2_584;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_10, a_5[2]);
            float2 _f2_585 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_36 = _f2_585;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_36, a_5[3]);
        }
        float2 _f2_590 = make_float2(weights[1], weights[1]);
        float2 weight_186 = _f2_590;
        {
            float2 _f2_591 = make_float2(fsrc[28], fsrc[29]);
            float2 v_39 = _f2_591;
            a_5[0] = fma_f32x2_rn_noftz(weight_186, v_39, a_5[0]);
            float2 _f2_592 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_37 = _f2_592;
            a_5[1] = fma_f32x2_rn_noftz(weight_186, v_0_37, a_5[1]);
            float2 _f2_593 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_11 = _f2_593;
            a_5[2] = fma_f32x2_rn_noftz(weight_186, v_1_11, a_5[2]);
            float2 _f2_594 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_37 = _f2_594;
            a_5[3] = fma_f32x2_rn_noftz(weight_186, v_2_37, a_5[3]);
        }
        float2 _f2_599 = make_float2(weights[2], weights[2]);
        float2 weight_187 = _f2_599;
        {
            float2 _f2_600 = make_float2(fsrc[56], fsrc[57]);
            float2 v_40 = _f2_600;
            a_5[0] = fma_f32x2_rn_noftz(weight_187, v_40, a_5[0]);
            float2 _f2_601 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_38 = _f2_601;
            a_5[1] = fma_f32x2_rn_noftz(weight_187, v_0_38, a_5[1]);
            float2 _f2_602 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_12 = _f2_602;
            a_5[2] = fma_f32x2_rn_noftz(weight_187, v_1_12, a_5[2]);
            float2 _f2_603 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_38 = _f2_603;
            a_5[3] = fma_f32x2_rn_noftz(weight_187, v_2_38, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_188 = 4;
        int base_189 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_190[4];
        float2 _f2_608 = make_float2(acc[8], acc[9]);
        float2 previous_191 = _f2_608;
        a_190[0] = mul_f32x2_noftz(previous_191, corr);
        float2 _f2_609 = make_float2(acc[10], acc[11]);
        float2 previous_192 = _f2_609;
        a_190[1] = mul_f32x2_noftz(previous_192, corr);
        float2 _f2_610 = make_float2(acc[12], acc[13]);
        float2 previous_193 = _f2_610;
        a_190[2] = mul_f32x2_noftz(previous_193, corr);
        float2 _f2_611 = make_float2(acc[14], acc[15]);
        float2 previous_194 = _f2_611;
        a_190[3] = mul_f32x2_noftz(previous_194, corr);
        float2 _f2_612 = make_float2(weights[0], weights[0]);
        float2 weight_195 = _f2_612;
        {
            float2 _f2_613 = make_float2(fsrc[8], fsrc[9]);
            float2 v_41 = _f2_613;
            a_190[0] = fma_f32x2_rn_noftz(weight_195, v_41, a_190[0]);
            float2 _f2_614 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_39 = _f2_614;
            a_190[1] = fma_f32x2_rn_noftz(weight_195, v_0_39, a_190[1]);
            float2 _f2_615 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_13 = _f2_615;
            a_190[2] = fma_f32x2_rn_noftz(weight_195, v_1_13, a_190[2]);
            float2 _f2_616 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_39 = _f2_616;
            a_190[3] = fma_f32x2_rn_noftz(weight_195, v_2_39, a_190[3]);
        }
        float2 _f2_621 = make_float2(weights[1], weights[1]);
        float2 weight_196 = _f2_621;
        {
            float2 _f2_622 = make_float2(fsrc[36], fsrc[37]);
            float2 v_42 = _f2_622;
            a_190[0] = fma_f32x2_rn_noftz(weight_196, v_42, a_190[0]);
            float2 _f2_623 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_40 = _f2_623;
            a_190[1] = fma_f32x2_rn_noftz(weight_196, v_0_40, a_190[1]);
            float2 _f2_624 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_14 = _f2_624;
            a_190[2] = fma_f32x2_rn_noftz(weight_196, v_1_14, a_190[2]);
            float2 _f2_625 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_40 = _f2_625;
            a_190[3] = fma_f32x2_rn_noftz(weight_196, v_2_40, a_190[3]);
        }
        float2 _f2_630 = make_float2(weights[2], weights[2]);
        float2 weight_197 = _f2_630;
        {
            float2 _f2_631 = make_float2(fsrc[64], fsrc[65]);
            float2 v_43 = _f2_631;
            a_190[0] = fma_f32x2_rn_noftz(weight_197, v_43, a_190[0]);
            float2 _f2_632 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_41 = _f2_632;
            a_190[1] = fma_f32x2_rn_noftz(weight_197, v_0_41, a_190[1]);
            float2 _f2_633 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_15 = _f2_633;
            a_190[2] = fma_f32x2_rn_noftz(weight_197, v_1_15, a_190[2]);
            float2 _f2_634 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_41 = _f2_634;
            a_190[3] = fma_f32x2_rn_noftz(weight_197, v_2_41, a_190[3]);
        }
        acc[8] = a_190[0].x;
        acc[9] = a_190[0].y;
        acc[10] = a_190[1].x;
        acc[11] = a_190[1].y;
        acc[12] = a_190[2].x;
        acc[13] = a_190[2].y;
        acc[14] = a_190[3].x;
        acc[15] = a_190[3].y;
        const int woff_198 = 8;
        int base_199 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_200[4];
        float2 _f2_639 = make_float2(acc[16], acc[17]);
        float2 previous_201 = _f2_639;
        a_200[0] = mul_f32x2_noftz(previous_201, corr);
        float2 _f2_640 = make_float2(acc[18], acc[19]);
        float2 previous_202 = _f2_640;
        a_200[1] = mul_f32x2_noftz(previous_202, corr);
        float2 _f2_641 = make_float2(acc[20], acc[21]);
        float2 previous_203 = _f2_641;
        a_200[2] = mul_f32x2_noftz(previous_203, corr);
        float2 _f2_642 = make_float2(acc[22], acc[23]);
        float2 previous_204 = _f2_642;
        a_200[3] = mul_f32x2_noftz(previous_204, corr);
        float2 _f2_643 = make_float2(weights[0], weights[0]);
        float2 weight_205 = _f2_643;
        {
            float2 _f2_644 = make_float2(fsrc[16], fsrc[17]);
            float2 v_44 = _f2_644;
            a_200[0] = fma_f32x2_rn_noftz(weight_205, v_44, a_200[0]);
            float2 _f2_645 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_42 = _f2_645;
            a_200[1] = fma_f32x2_rn_noftz(weight_205, v_0_42, a_200[1]);
            float2 _f2_646 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_16 = _f2_646;
            a_200[2] = fma_f32x2_rn_noftz(weight_205, v_1_16, a_200[2]);
            float2 _f2_647 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_42 = _f2_647;
            a_200[3] = fma_f32x2_rn_noftz(weight_205, v_2_42, a_200[3]);
        }
        float2 _f2_652 = make_float2(weights[1], weights[1]);
        float2 weight_206 = _f2_652;
        {
            float2 _f2_653 = make_float2(fsrc[44], fsrc[45]);
            float2 v_45 = _f2_653;
            a_200[0] = fma_f32x2_rn_noftz(weight_206, v_45, a_200[0]);
            float2 _f2_654 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_43 = _f2_654;
            a_200[1] = fma_f32x2_rn_noftz(weight_206, v_0_43, a_200[1]);
            float2 _f2_655 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_17 = _f2_655;
            a_200[2] = fma_f32x2_rn_noftz(weight_206, v_1_17, a_200[2]);
            float2 _f2_656 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_43 = _f2_656;
            a_200[3] = fma_f32x2_rn_noftz(weight_206, v_2_43, a_200[3]);
        }
        float2 _f2_661 = make_float2(weights[2], weights[2]);
        float2 weight_207 = _f2_661;
        {
            float2 _f2_662 = make_float2(fsrc[72], fsrc[73]);
            float2 v_46 = _f2_662;
            a_200[0] = fma_f32x2_rn_noftz(weight_207, v_46, a_200[0]);
            float2 _f2_663 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_44 = _f2_663;
            a_200[1] = fma_f32x2_rn_noftz(weight_207, v_0_44, a_200[1]);
            float2 _f2_664 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_18 = _f2_664;
            a_200[2] = fma_f32x2_rn_noftz(weight_207, v_1_18, a_200[2]);
            float2 _f2_665 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_44 = _f2_665;
            a_200[3] = fma_f32x2_rn_noftz(weight_207, v_2_44, a_200[3]);
        }
        acc[16] = a_200[0].x;
        acc[17] = a_200[0].y;
        acc[18] = a_200[1].x;
        acc[19] = a_200[1].y;
        acc[20] = a_200[2].x;
        acc[21] = a_200[2].y;
        acc[22] = a_200[3].x;
        acc[23] = a_200[3].y;
        const int woff_208 = 12;
        int base_209 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_210[4];
        float2 _f2_670 = make_float2(acc[24], acc[25]);
        float2 previous_211 = _f2_670;
        a_210[0] = mul_f32x2_noftz(previous_211, corr);
        float2 _f2_671 = make_float2(acc[26], acc[27]);
        float2 previous_212 = _f2_671;
        a_210[1] = mul_f32x2_noftz(previous_212, corr);
        float2 _f2_672 = make_float2(weights[0], weights[0]);
        float2 weight_213 = _f2_672;
        {
            float2 _f2_673 = make_float2(fsrc[24], fsrc[25]);
            float2 v_47 = _f2_673;
            a_210[0] = fma_f32x2_rn_noftz(weight_213, v_47, a_210[0]);
            float2 _f2_674 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_45 = _f2_674;
            a_210[1] = fma_f32x2_rn_noftz(weight_213, v_0_45, a_210[1]);
        }
        float2 _f2_677 = make_float2(weights[1], weights[1]);
        float2 weight_214 = _f2_677;
        {
            float2 _f2_678 = make_float2(fsrc[52], fsrc[53]);
            float2 v_48 = _f2_678;
            a_210[0] = fma_f32x2_rn_noftz(weight_214, v_48, a_210[0]);
            float2 _f2_679 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_46 = _f2_679;
            a_210[1] = fma_f32x2_rn_noftz(weight_214, v_0_46, a_210[1]);
        }
        float2 _f2_682 = make_float2(weights[2], weights[2]);
        float2 weight_215 = _f2_682;
        {
            float2 _f2_683 = make_float2(fsrc[80], fsrc[81]);
            float2 v_49 = _f2_683;
            a_210[0] = fma_f32x2_rn_noftz(weight_215, v_49, a_210[0]);
            float2 _f2_684 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_47 = _f2_684;
            a_210[1] = fma_f32x2_rn_noftz(weight_215, v_0_47, a_210[1]);
        }
        acc[24] = a_210[0].x;
        acc[25] = a_210[0].y;
        acc[26] = a_210[1].x;
        acc[27] = a_210[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float max_chunk_216 = -3.4028234663852886e+38f;
        float _fmax_4 = fmaxf(max_chunk_216, logits[3]);
        max_chunk_216 = _fmax_4;
        float _fmax_5 = fmaxf(max_chunk_216, logits[4]);
        max_chunk_216 = _fmax_5;
        float _fmax_6 = fmaxf(max_chunk_216, logits[5]);
        max_chunk_216 = _fmax_6;
        float _fmax_7 = fmaxf(max_running, max_chunk_216);
        float max_new_217 = _fmax_7;
        float _exp2_4 = approx_exp2((max_running - max_new_217) * 1.4426950408889634f);
        float correction_218 = _exp2_4;
        float weights_219[9];
        float sum_weights_220 = 0.0f;
        float _exp2_5 = approx_exp2((logits[3] - max_new_217) * 1.4426950408889634f);
        weights_219[3] = _exp2_5;
        sum_weights_220 += weights_219[3];
        float _exp2_6 = approx_exp2((logits[4] - max_new_217) * 1.4426950408889634f);
        weights_219[4] = _exp2_6;
        sum_weights_220 += weights_219[4];
        float _exp2_7 = approx_exp2((logits[5] - max_new_217) * 1.4426950408889634f);
        weights_219[5] = _exp2_7;
        sum_weights_220 += weights_219[5];
        float2 _f2_687 = make_float2(correction_218, correction_218);
        float2 corr_221 = _f2_687;
        const int woff_222 = 0;
        int base_223 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_224[4];
        float2 _f2_688 = make_float2(acc[0], acc[1]);
        float2 previous_225 = _f2_688;
        a_224[0] = mul_f32x2_noftz(previous_225, corr_221);
        float2 _f2_689 = make_float2(acc[2], acc[3]);
        float2 previous_226 = _f2_689;
        a_224[1] = mul_f32x2_noftz(previous_226, corr_221);
        float2 _f2_690 = make_float2(acc[4], acc[5]);
        float2 previous_227 = _f2_690;
        a_224[2] = mul_f32x2_noftz(previous_227, corr_221);
        float2 _f2_691 = make_float2(acc[6], acc[7]);
        float2 previous_228 = _f2_691;
        a_224[3] = mul_f32x2_noftz(previous_228, corr_221);
        float2 _f2_692 = make_float2(weights_219[3], weights_219[3]);
        float2 weight_229 = _f2_692;
        {
            unsigned int sw2[4];
            sw2[0] = words[42 + woff_222];
            sw2[1] = words[42 + woff_222 + 1];
            sw2[2] = words[42 + woff_222 + 2];
            sw2[3] = words[42 + woff_222 + 3];
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
            float2 _f2_697 = make_float2(sw2_f32[0], sw2_f32[1]);
            float2 v_50 = _f2_697;
            a_224[0] = fma_f32x2_rn_noftz(weight_229, v_50, a_224[0]);
            float2 _f2_698 = make_float2(sw2_f32[2], sw2_f32[3]);
            float2 v_0_48 = _f2_698;
            a_224[1] = fma_f32x2_rn_noftz(weight_229, v_0_48, a_224[1]);
            float2 _f2_699 = make_float2(sw2_f32[4], sw2_f32[5]);
            float2 v_1_19 = _f2_699;
            a_224[2] = fma_f32x2_rn_noftz(weight_229, v_1_19, a_224[2]);
            float2 _f2_700 = make_float2(sw2_f32[6], sw2_f32[7]);
            float2 v_2_45 = _f2_700;
            a_224[3] = fma_f32x2_rn_noftz(weight_229, v_2_45, a_224[3]);
        }
        float2 _f2_701 = make_float2(weights_219[4], weights_219[4]);
        float2 weight_230 = _f2_701;
        {
            unsigned int sw2_1[4];
            sw2_1[0] = words[56 + woff_222];
            sw2_1[1] = words[56 + woff_222 + 1];
            sw2_1[2] = words[56 + woff_222 + 2];
            sw2_1[3] = words[56 + woff_222 + 3];
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
            float2 _f2_706 = make_float2(sw2_f32_1[0], sw2_f32_1[1]);
            float2 v_51 = _f2_706;
            a_224[0] = fma_f32x2_rn_noftz(weight_230, v_51, a_224[0]);
            float2 _f2_707 = make_float2(sw2_f32_1[2], sw2_f32_1[3]);
            float2 v_0_49 = _f2_707;
            a_224[1] = fma_f32x2_rn_noftz(weight_230, v_0_49, a_224[1]);
            float2 _f2_708 = make_float2(sw2_f32_1[4], sw2_f32_1[5]);
            float2 v_1_20 = _f2_708;
            a_224[2] = fma_f32x2_rn_noftz(weight_230, v_1_20, a_224[2]);
            float2 _f2_709 = make_float2(sw2_f32_1[6], sw2_f32_1[7]);
            float2 v_2_46 = _f2_709;
            a_224[3] = fma_f32x2_rn_noftz(weight_230, v_2_46, a_224[3]);
        }
        float2 _f2_710 = make_float2(weights_219[5], weights_219[5]);
        float2 weight_231 = _f2_710;
        {
            unsigned int sw2_2[4];
            sw2_2[0] = words[70 + woff_222];
            sw2_2[1] = words[70 + woff_222 + 1];
            sw2_2[2] = words[70 + woff_222 + 2];
            sw2_2[3] = words[70 + woff_222 + 3];
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
            float2 _f2_715 = make_float2(sw2_f32_2[0], sw2_f32_2[1]);
            float2 v_52 = _f2_715;
            a_224[0] = fma_f32x2_rn_noftz(weight_231, v_52, a_224[0]);
            float2 _f2_716 = make_float2(sw2_f32_2[2], sw2_f32_2[3]);
            float2 v_0_50 = _f2_716;
            a_224[1] = fma_f32x2_rn_noftz(weight_231, v_0_50, a_224[1]);
            float2 _f2_717 = make_float2(sw2_f32_2[4], sw2_f32_2[5]);
            float2 v_1_21 = _f2_717;
            a_224[2] = fma_f32x2_rn_noftz(weight_231, v_1_21, a_224[2]);
            float2 _f2_718 = make_float2(sw2_f32_2[6], sw2_f32_2[7]);
            float2 v_2_47 = _f2_718;
            a_224[3] = fma_f32x2_rn_noftz(weight_231, v_2_47, a_224[3]);
        }
        acc[0] = a_224[0].x;
        acc[1] = a_224[0].y;
        acc[2] = a_224[1].x;
        acc[3] = a_224[1].y;
        acc[4] = a_224[2].x;
        acc[5] = a_224[2].y;
        acc[6] = a_224[3].x;
        acc[7] = a_224[3].y;
        const int woff_232 = 4;
        int base_233 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_234[4];
        float2 _f2_719 = make_float2(acc[8], acc[9]);
        float2 previous_235 = _f2_719;
        a_234[0] = mul_f32x2_noftz(previous_235, corr_221);
        float2 _f2_720 = make_float2(acc[10], acc[11]);
        float2 previous_236 = _f2_720;
        a_234[1] = mul_f32x2_noftz(previous_236, corr_221);
        float2 _f2_721 = make_float2(acc[12], acc[13]);
        float2 previous_237 = _f2_721;
        a_234[2] = mul_f32x2_noftz(previous_237, corr_221);
        float2 _f2_722 = make_float2(acc[14], acc[15]);
        float2 previous_238 = _f2_722;
        a_234[3] = mul_f32x2_noftz(previous_238, corr_221);
        float2 _f2_723 = make_float2(weights_219[3], weights_219[3]);
        float2 weight_239 = _f2_723;
        {
            unsigned int sw2_3[4];
            sw2_3[0] = words[42 + woff_232];
            sw2_3[1] = words[42 + woff_232 + 1];
            sw2_3[2] = words[42 + woff_232 + 2];
            sw2_3[3] = words[42 + woff_232 + 3];
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
            float2 _f2_728 = make_float2(sw2_f32_3[0], sw2_f32_3[1]);
            float2 v_53 = _f2_728;
            a_234[0] = fma_f32x2_rn_noftz(weight_239, v_53, a_234[0]);
            float2 _f2_729 = make_float2(sw2_f32_3[2], sw2_f32_3[3]);
            float2 v_0_51 = _f2_729;
            a_234[1] = fma_f32x2_rn_noftz(weight_239, v_0_51, a_234[1]);
            float2 _f2_730 = make_float2(sw2_f32_3[4], sw2_f32_3[5]);
            float2 v_1_22 = _f2_730;
            a_234[2] = fma_f32x2_rn_noftz(weight_239, v_1_22, a_234[2]);
            float2 _f2_731 = make_float2(sw2_f32_3[6], sw2_f32_3[7]);
            float2 v_2_48 = _f2_731;
            a_234[3] = fma_f32x2_rn_noftz(weight_239, v_2_48, a_234[3]);
        }
        float2 _f2_732 = make_float2(weights_219[4], weights_219[4]);
        float2 weight_240 = _f2_732;
        {
            unsigned int sw2_4[4];
            sw2_4[0] = words[56 + woff_232];
            sw2_4[1] = words[56 + woff_232 + 1];
            sw2_4[2] = words[56 + woff_232 + 2];
            sw2_4[3] = words[56 + woff_232 + 3];
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
            float2 _f2_737 = make_float2(sw2_f32_4[0], sw2_f32_4[1]);
            float2 v_54 = _f2_737;
            a_234[0] = fma_f32x2_rn_noftz(weight_240, v_54, a_234[0]);
            float2 _f2_738 = make_float2(sw2_f32_4[2], sw2_f32_4[3]);
            float2 v_0_52 = _f2_738;
            a_234[1] = fma_f32x2_rn_noftz(weight_240, v_0_52, a_234[1]);
            float2 _f2_739 = make_float2(sw2_f32_4[4], sw2_f32_4[5]);
            float2 v_1_23 = _f2_739;
            a_234[2] = fma_f32x2_rn_noftz(weight_240, v_1_23, a_234[2]);
            float2 _f2_740 = make_float2(sw2_f32_4[6], sw2_f32_4[7]);
            float2 v_2_49 = _f2_740;
            a_234[3] = fma_f32x2_rn_noftz(weight_240, v_2_49, a_234[3]);
        }
        float2 _f2_741 = make_float2(weights_219[5], weights_219[5]);
        float2 weight_241 = _f2_741;
        {
            unsigned int sw2_5[4];
            sw2_5[0] = words[70 + woff_232];
            sw2_5[1] = words[70 + woff_232 + 1];
            sw2_5[2] = words[70 + woff_232 + 2];
            sw2_5[3] = words[70 + woff_232 + 3];
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
            float2 _f2_746 = make_float2(sw2_f32_5[0], sw2_f32_5[1]);
            float2 v_55 = _f2_746;
            a_234[0] = fma_f32x2_rn_noftz(weight_241, v_55, a_234[0]);
            float2 _f2_747 = make_float2(sw2_f32_5[2], sw2_f32_5[3]);
            float2 v_0_53 = _f2_747;
            a_234[1] = fma_f32x2_rn_noftz(weight_241, v_0_53, a_234[1]);
            float2 _f2_748 = make_float2(sw2_f32_5[4], sw2_f32_5[5]);
            float2 v_1_24 = _f2_748;
            a_234[2] = fma_f32x2_rn_noftz(weight_241, v_1_24, a_234[2]);
            float2 _f2_749 = make_float2(sw2_f32_5[6], sw2_f32_5[7]);
            float2 v_2_50 = _f2_749;
            a_234[3] = fma_f32x2_rn_noftz(weight_241, v_2_50, a_234[3]);
        }
        acc[8] = a_234[0].x;
        acc[9] = a_234[0].y;
        acc[10] = a_234[1].x;
        acc[11] = a_234[1].y;
        acc[12] = a_234[2].x;
        acc[13] = a_234[2].y;
        acc[14] = a_234[3].x;
        acc[15] = a_234[3].y;
        const int woff_242 = 8;
        int base_243 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_244[4];
        float2 _f2_750 = make_float2(acc[16], acc[17]);
        float2 previous_245 = _f2_750;
        a_244[0] = mul_f32x2_noftz(previous_245, corr_221);
        float2 _f2_751 = make_float2(acc[18], acc[19]);
        float2 previous_246 = _f2_751;
        a_244[1] = mul_f32x2_noftz(previous_246, corr_221);
        float2 _f2_752 = make_float2(acc[20], acc[21]);
        float2 previous_247 = _f2_752;
        a_244[2] = mul_f32x2_noftz(previous_247, corr_221);
        float2 _f2_753 = make_float2(acc[22], acc[23]);
        float2 previous_248 = _f2_753;
        a_244[3] = mul_f32x2_noftz(previous_248, corr_221);
        float2 _f2_754 = make_float2(weights_219[3], weights_219[3]);
        float2 weight_249 = _f2_754;
        {
            unsigned int sw2_6[4];
            sw2_6[0] = words[42 + woff_242];
            sw2_6[1] = words[42 + woff_242 + 1];
            sw2_6[2] = words[42 + woff_242 + 2];
            sw2_6[3] = words[42 + woff_242 + 3];
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
            float2 _f2_759 = make_float2(sw2_f32_6[0], sw2_f32_6[1]);
            float2 v_56 = _f2_759;
            a_244[0] = fma_f32x2_rn_noftz(weight_249, v_56, a_244[0]);
            float2 _f2_760 = make_float2(sw2_f32_6[2], sw2_f32_6[3]);
            float2 v_0_54 = _f2_760;
            a_244[1] = fma_f32x2_rn_noftz(weight_249, v_0_54, a_244[1]);
            float2 _f2_761 = make_float2(sw2_f32_6[4], sw2_f32_6[5]);
            float2 v_1_25 = _f2_761;
            a_244[2] = fma_f32x2_rn_noftz(weight_249, v_1_25, a_244[2]);
            float2 _f2_762 = make_float2(sw2_f32_6[6], sw2_f32_6[7]);
            float2 v_2_51 = _f2_762;
            a_244[3] = fma_f32x2_rn_noftz(weight_249, v_2_51, a_244[3]);
        }
        float2 _f2_763 = make_float2(weights_219[4], weights_219[4]);
        float2 weight_250 = _f2_763;
        {
            unsigned int sw2_7[4];
            sw2_7[0] = words[56 + woff_242];
            sw2_7[1] = words[56 + woff_242 + 1];
            sw2_7[2] = words[56 + woff_242 + 2];
            sw2_7[3] = words[56 + woff_242 + 3];
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
            float2 _f2_768 = make_float2(sw2_f32_7[0], sw2_f32_7[1]);
            float2 v_57 = _f2_768;
            a_244[0] = fma_f32x2_rn_noftz(weight_250, v_57, a_244[0]);
            float2 _f2_769 = make_float2(sw2_f32_7[2], sw2_f32_7[3]);
            float2 v_0_55 = _f2_769;
            a_244[1] = fma_f32x2_rn_noftz(weight_250, v_0_55, a_244[1]);
            float2 _f2_770 = make_float2(sw2_f32_7[4], sw2_f32_7[5]);
            float2 v_1_26 = _f2_770;
            a_244[2] = fma_f32x2_rn_noftz(weight_250, v_1_26, a_244[2]);
            float2 _f2_771 = make_float2(sw2_f32_7[6], sw2_f32_7[7]);
            float2 v_2_52 = _f2_771;
            a_244[3] = fma_f32x2_rn_noftz(weight_250, v_2_52, a_244[3]);
        }
        float2 _f2_772 = make_float2(weights_219[5], weights_219[5]);
        float2 weight_251 = _f2_772;
        {
            unsigned int sw2_8[4];
            sw2_8[0] = words[70 + woff_242];
            sw2_8[1] = words[70 + woff_242 + 1];
            sw2_8[2] = words[70 + woff_242 + 2];
            sw2_8[3] = words[70 + woff_242 + 3];
            float sw2_f32_8[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_8[_pair * 2])[0]), "=f"((&sw2_f32_8[_pair * 2])[1])
                    : "r"(sw2_8[_pair]));
            }
            float2 _f2_777 = make_float2(sw2_f32_8[0], sw2_f32_8[1]);
            float2 v_58 = _f2_777;
            a_244[0] = fma_f32x2_rn_noftz(weight_251, v_58, a_244[0]);
            float2 _f2_778 = make_float2(sw2_f32_8[2], sw2_f32_8[3]);
            float2 v_0_56 = _f2_778;
            a_244[1] = fma_f32x2_rn_noftz(weight_251, v_0_56, a_244[1]);
            float2 _f2_779 = make_float2(sw2_f32_8[4], sw2_f32_8[5]);
            float2 v_1_27 = _f2_779;
            a_244[2] = fma_f32x2_rn_noftz(weight_251, v_1_27, a_244[2]);
            float2 _f2_780 = make_float2(sw2_f32_8[6], sw2_f32_8[7]);
            float2 v_2_53 = _f2_780;
            a_244[3] = fma_f32x2_rn_noftz(weight_251, v_2_53, a_244[3]);
        }
        acc[16] = a_244[0].x;
        acc[17] = a_244[0].y;
        acc[18] = a_244[1].x;
        acc[19] = a_244[1].y;
        acc[20] = a_244[2].x;
        acc[21] = a_244[2].y;
        acc[22] = a_244[3].x;
        acc[23] = a_244[3].y;
        const int woff_252 = 12;
        int base_253 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_254[4];
        float2 _f2_781 = make_float2(acc[24], acc[25]);
        float2 previous_255 = _f2_781;
        a_254[0] = mul_f32x2_noftz(previous_255, corr_221);
        float2 _f2_782 = make_float2(acc[26], acc[27]);
        float2 previous_256 = _f2_782;
        a_254[1] = mul_f32x2_noftz(previous_256, corr_221);
        float2 _f2_783 = make_float2(weights_219[3], weights_219[3]);
        float2 weight_257 = _f2_783;
        {
            unsigned int sw2_9[4];
            sw2_9[0] = words[42 + woff_252];
            sw2_9[1] = words[42 + woff_252 + 1];
            float sw2_f32_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_9[_pair * 2])[0]), "=f"((&sw2_f32_9[_pair * 2])[1])
                    : "r"(sw2_9[_pair]));
            }
            float2 _f2_786 = make_float2(sw2_f32_9[0], sw2_f32_9[1]);
            float2 v_59 = _f2_786;
            a_254[0] = fma_f32x2_rn_noftz(weight_257, v_59, a_254[0]);
            float2 _f2_787 = make_float2(sw2_f32_9[2], sw2_f32_9[3]);
            float2 v_0_57 = _f2_787;
            a_254[1] = fma_f32x2_rn_noftz(weight_257, v_0_57, a_254[1]);
        }
        float2 _f2_788 = make_float2(weights_219[4], weights_219[4]);
        float2 weight_258 = _f2_788;
        {
            unsigned int sw2_10[4];
            sw2_10[0] = words[56 + woff_252];
            sw2_10[1] = words[56 + woff_252 + 1];
            float sw2_f32_10[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_10[_pair * 2])[0]), "=f"((&sw2_f32_10[_pair * 2])[1])
                    : "r"(sw2_10[_pair]));
            }
            float2 _f2_791 = make_float2(sw2_f32_10[0], sw2_f32_10[1]);
            float2 v_60 = _f2_791;
            a_254[0] = fma_f32x2_rn_noftz(weight_258, v_60, a_254[0]);
            float2 _f2_792 = make_float2(sw2_f32_10[2], sw2_f32_10[3]);
            float2 v_0_58 = _f2_792;
            a_254[1] = fma_f32x2_rn_noftz(weight_258, v_0_58, a_254[1]);
        }
        float2 _f2_793 = make_float2(weights_219[5], weights_219[5]);
        float2 weight_259 = _f2_793;
        {
            unsigned int sw2_11[4];
            sw2_11[0] = words[70 + woff_252];
            sw2_11[1] = words[70 + woff_252 + 1];
            float sw2_f32_11[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_11[_pair * 2])[0]), "=f"((&sw2_f32_11[_pair * 2])[1])
                    : "r"(sw2_11[_pair]));
            }
            float2 _f2_796 = make_float2(sw2_f32_11[0], sw2_f32_11[1]);
            float2 v_61 = _f2_796;
            a_254[0] = fma_f32x2_rn_noftz(weight_259, v_61, a_254[0]);
            float2 _f2_797 = make_float2(sw2_f32_11[2], sw2_f32_11[3]);
            float2 v_0_59 = _f2_797;
            a_254[1] = fma_f32x2_rn_noftz(weight_259, v_0_59, a_254[1]);
        }
        acc[24] = a_254[0].x;
        acc[25] = a_254[0].y;
        acc[26] = a_254[1].x;
        acc[27] = a_254[1].y;
        sum_running = sum_running * correction_218 + sum_weights_220;
        max_running = max_new_217;
        float max_chunk_260 = -3.4028234663852886e+38f;
        float _fmax_8 = fmaxf(max_chunk_260, logits[6]);
        max_chunk_260 = _fmax_8;
        float _fmax_9 = fmaxf(max_chunk_260, logits[7]);
        max_chunk_260 = _fmax_9;
        float _fmax_10 = fmaxf(max_chunk_260, logits[8]);
        max_chunk_260 = _fmax_10;
        float _fmax_11 = fmaxf(max_running, max_chunk_260);
        float max_new_261 = _fmax_11;
        float _exp2_8 = approx_exp2((max_running - max_new_261) * 1.4426950408889634f);
        float correction_262 = _exp2_8;
        float weights_263[9];
        float sum_weights_264 = 0.0f;
        float _exp2_9 = approx_exp2((logits[6] - max_new_261) * 1.4426950408889634f);
        weights_263[6] = _exp2_9;
        sum_weights_264 += weights_263[6];
        float _exp2_10 = approx_exp2((logits[7] - max_new_261) * 1.4426950408889634f);
        weights_263[7] = _exp2_10;
        sum_weights_264 += weights_263[7];
        float _exp2_11 = approx_exp2((logits[8] - max_new_261) * 1.4426950408889634f);
        weights_263[8] = _exp2_11;
        sum_weights_264 += weights_263[8];
        float2 _f2_798 = make_float2(correction_262, correction_262);
        float2 corr_265 = _f2_798;
        const int woff_266 = 0;
        int base_267 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_268[4];
        float2 _f2_799 = make_float2(acc[0], acc[1]);
        float2 previous_269 = _f2_799;
        a_268[0] = mul_f32x2_noftz(previous_269, corr_265);
        float2 _f2_800 = make_float2(acc[2], acc[3]);
        float2 previous_270 = _f2_800;
        a_268[1] = mul_f32x2_noftz(previous_270, corr_265);
        float2 _f2_801 = make_float2(acc[4], acc[5]);
        float2 previous_271 = _f2_801;
        a_268[2] = mul_f32x2_noftz(previous_271, corr_265);
        float2 _f2_802 = make_float2(acc[6], acc[7]);
        float2 previous_272 = _f2_802;
        a_268[3] = mul_f32x2_noftz(previous_272, corr_265);
        float2 _f2_803 = make_float2(weights_263[6], weights_263[6]);
        float2 weight_273 = _f2_803;
        {
            unsigned int sw2_12[4];
            sw2_12[0] = words[84 + woff_266];
            sw2_12[1] = words[84 + woff_266 + 1];
            sw2_12[2] = words[84 + woff_266 + 2];
            sw2_12[3] = words[84 + woff_266 + 3];
            float sw2_f32_12[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_12[_pair * 2])[0]), "=f"((&sw2_f32_12[_pair * 2])[1])
                    : "r"(sw2_12[_pair]));
            }
            float2 _f2_808 = make_float2(sw2_f32_12[0], sw2_f32_12[1]);
            float2 v_62 = _f2_808;
            a_268[0] = fma_f32x2_rn_noftz(weight_273, v_62, a_268[0]);
            float2 _f2_809 = make_float2(sw2_f32_12[2], sw2_f32_12[3]);
            float2 v_0_60 = _f2_809;
            a_268[1] = fma_f32x2_rn_noftz(weight_273, v_0_60, a_268[1]);
            float2 _f2_810 = make_float2(sw2_f32_12[4], sw2_f32_12[5]);
            float2 v_1_28 = _f2_810;
            a_268[2] = fma_f32x2_rn_noftz(weight_273, v_1_28, a_268[2]);
            float2 _f2_811 = make_float2(sw2_f32_12[6], sw2_f32_12[7]);
            float2 v_2_54 = _f2_811;
            a_268[3] = fma_f32x2_rn_noftz(weight_273, v_2_54, a_268[3]);
        }
        float2 _f2_812 = make_float2(weights_263[7], weights_263[7]);
        float2 weight_274 = _f2_812;
        {
            unsigned int sw2_13[4];
            sw2_13[0] = words[98 + woff_266];
            sw2_13[1] = words[98 + woff_266 + 1];
            sw2_13[2] = words[98 + woff_266 + 2];
            sw2_13[3] = words[98 + woff_266 + 3];
            float sw2_f32_13[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_13[_pair * 2])[0]), "=f"((&sw2_f32_13[_pair * 2])[1])
                    : "r"(sw2_13[_pair]));
            }
            float2 _f2_817 = make_float2(sw2_f32_13[0], sw2_f32_13[1]);
            float2 v_63 = _f2_817;
            a_268[0] = fma_f32x2_rn_noftz(weight_274, v_63, a_268[0]);
            float2 _f2_818 = make_float2(sw2_f32_13[2], sw2_f32_13[3]);
            float2 v_0_61 = _f2_818;
            a_268[1] = fma_f32x2_rn_noftz(weight_274, v_0_61, a_268[1]);
            float2 _f2_819 = make_float2(sw2_f32_13[4], sw2_f32_13[5]);
            float2 v_1_29 = _f2_819;
            a_268[2] = fma_f32x2_rn_noftz(weight_274, v_1_29, a_268[2]);
            float2 _f2_820 = make_float2(sw2_f32_13[6], sw2_f32_13[7]);
            float2 v_2_55 = _f2_820;
            a_268[3] = fma_f32x2_rn_noftz(weight_274, v_2_55, a_268[3]);
        }
        float2 _f2_821 = make_float2(weights_263[8], weights_263[8]);
        float2 weight_275 = _f2_821;
        {
            unsigned int sw2_14[4];
            sw2_14[0] = words[112 + woff_266];
            sw2_14[1] = words[112 + woff_266 + 1];
            sw2_14[2] = words[112 + woff_266 + 2];
            sw2_14[3] = words[112 + woff_266 + 3];
            float sw2_f32_14[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_14[_pair * 2])[0]), "=f"((&sw2_f32_14[_pair * 2])[1])
                    : "r"(sw2_14[_pair]));
            }
            float2 _f2_826 = make_float2(sw2_f32_14[0], sw2_f32_14[1]);
            float2 v_64 = _f2_826;
            a_268[0] = fma_f32x2_rn_noftz(weight_275, v_64, a_268[0]);
            float2 _f2_827 = make_float2(sw2_f32_14[2], sw2_f32_14[3]);
            float2 v_0_62 = _f2_827;
            a_268[1] = fma_f32x2_rn_noftz(weight_275, v_0_62, a_268[1]);
            float2 _f2_828 = make_float2(sw2_f32_14[4], sw2_f32_14[5]);
            float2 v_1_30 = _f2_828;
            a_268[2] = fma_f32x2_rn_noftz(weight_275, v_1_30, a_268[2]);
            float2 _f2_829 = make_float2(sw2_f32_14[6], sw2_f32_14[7]);
            float2 v_2_56 = _f2_829;
            a_268[3] = fma_f32x2_rn_noftz(weight_275, v_2_56, a_268[3]);
        }
        acc[0] = a_268[0].x;
        acc[1] = a_268[0].y;
        acc[2] = a_268[1].x;
        acc[3] = a_268[1].y;
        acc[4] = a_268[2].x;
        acc[5] = a_268[2].y;
        acc[6] = a_268[3].x;
        acc[7] = a_268[3].y;
        const int woff_276 = 4;
        int base_277 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_278[4];
        float2 _f2_830 = make_float2(acc[8], acc[9]);
        float2 previous_279 = _f2_830;
        a_278[0] = mul_f32x2_noftz(previous_279, corr_265);
        float2 _f2_831 = make_float2(acc[10], acc[11]);
        float2 previous_280 = _f2_831;
        a_278[1] = mul_f32x2_noftz(previous_280, corr_265);
        float2 _f2_832 = make_float2(acc[12], acc[13]);
        float2 previous_281 = _f2_832;
        a_278[2] = mul_f32x2_noftz(previous_281, corr_265);
        float2 _f2_833 = make_float2(acc[14], acc[15]);
        float2 previous_282 = _f2_833;
        a_278[3] = mul_f32x2_noftz(previous_282, corr_265);
        float2 _f2_834 = make_float2(weights_263[6], weights_263[6]);
        float2 weight_283 = _f2_834;
        {
            unsigned int sw2_15[4];
            sw2_15[0] = words[84 + woff_276];
            sw2_15[1] = words[84 + woff_276 + 1];
            sw2_15[2] = words[84 + woff_276 + 2];
            sw2_15[3] = words[84 + woff_276 + 3];
            float sw2_f32_15[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_15[_pair * 2])[0]), "=f"((&sw2_f32_15[_pair * 2])[1])
                    : "r"(sw2_15[_pair]));
            }
            float2 _f2_839 = make_float2(sw2_f32_15[0], sw2_f32_15[1]);
            float2 v_65 = _f2_839;
            a_278[0] = fma_f32x2_rn_noftz(weight_283, v_65, a_278[0]);
            float2 _f2_840 = make_float2(sw2_f32_15[2], sw2_f32_15[3]);
            float2 v_0_63 = _f2_840;
            a_278[1] = fma_f32x2_rn_noftz(weight_283, v_0_63, a_278[1]);
            float2 _f2_841 = make_float2(sw2_f32_15[4], sw2_f32_15[5]);
            float2 v_1_31 = _f2_841;
            a_278[2] = fma_f32x2_rn_noftz(weight_283, v_1_31, a_278[2]);
            float2 _f2_842 = make_float2(sw2_f32_15[6], sw2_f32_15[7]);
            float2 v_2_57 = _f2_842;
            a_278[3] = fma_f32x2_rn_noftz(weight_283, v_2_57, a_278[3]);
        }
        float2 _f2_843 = make_float2(weights_263[7], weights_263[7]);
        float2 weight_284 = _f2_843;
        {
            unsigned int sw2_16[4];
            sw2_16[0] = words[98 + woff_276];
            sw2_16[1] = words[98 + woff_276 + 1];
            sw2_16[2] = words[98 + woff_276 + 2];
            sw2_16[3] = words[98 + woff_276 + 3];
            float sw2_f32_16[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_16[_pair * 2])[0]), "=f"((&sw2_f32_16[_pair * 2])[1])
                    : "r"(sw2_16[_pair]));
            }
            float2 _f2_848 = make_float2(sw2_f32_16[0], sw2_f32_16[1]);
            float2 v_66 = _f2_848;
            a_278[0] = fma_f32x2_rn_noftz(weight_284, v_66, a_278[0]);
            float2 _f2_849 = make_float2(sw2_f32_16[2], sw2_f32_16[3]);
            float2 v_0_64 = _f2_849;
            a_278[1] = fma_f32x2_rn_noftz(weight_284, v_0_64, a_278[1]);
            float2 _f2_850 = make_float2(sw2_f32_16[4], sw2_f32_16[5]);
            float2 v_1_32 = _f2_850;
            a_278[2] = fma_f32x2_rn_noftz(weight_284, v_1_32, a_278[2]);
            float2 _f2_851 = make_float2(sw2_f32_16[6], sw2_f32_16[7]);
            float2 v_2_58 = _f2_851;
            a_278[3] = fma_f32x2_rn_noftz(weight_284, v_2_58, a_278[3]);
        }
        float2 _f2_852 = make_float2(weights_263[8], weights_263[8]);
        float2 weight_285 = _f2_852;
        {
            unsigned int sw2_17[4];
            sw2_17[0] = words[112 + woff_276];
            sw2_17[1] = words[112 + woff_276 + 1];
            sw2_17[2] = words[112 + woff_276 + 2];
            sw2_17[3] = words[112 + woff_276 + 3];
            float sw2_f32_17[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_17[_pair * 2])[0]), "=f"((&sw2_f32_17[_pair * 2])[1])
                    : "r"(sw2_17[_pair]));
            }
            float2 _f2_857 = make_float2(sw2_f32_17[0], sw2_f32_17[1]);
            float2 v_67 = _f2_857;
            a_278[0] = fma_f32x2_rn_noftz(weight_285, v_67, a_278[0]);
            float2 _f2_858 = make_float2(sw2_f32_17[2], sw2_f32_17[3]);
            float2 v_0_65 = _f2_858;
            a_278[1] = fma_f32x2_rn_noftz(weight_285, v_0_65, a_278[1]);
            float2 _f2_859 = make_float2(sw2_f32_17[4], sw2_f32_17[5]);
            float2 v_1_33 = _f2_859;
            a_278[2] = fma_f32x2_rn_noftz(weight_285, v_1_33, a_278[2]);
            float2 _f2_860 = make_float2(sw2_f32_17[6], sw2_f32_17[7]);
            float2 v_2_59 = _f2_860;
            a_278[3] = fma_f32x2_rn_noftz(weight_285, v_2_59, a_278[3]);
        }
        acc[8] = a_278[0].x;
        acc[9] = a_278[0].y;
        acc[10] = a_278[1].x;
        acc[11] = a_278[1].y;
        acc[12] = a_278[2].x;
        acc[13] = a_278[2].y;
        acc[14] = a_278[3].x;
        acc[15] = a_278[3].y;
        const int woff_286 = 8;
        int base_287 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_288[4];
        float2 _f2_861 = make_float2(acc[16], acc[17]);
        float2 previous_289 = _f2_861;
        a_288[0] = mul_f32x2_noftz(previous_289, corr_265);
        float2 _f2_862 = make_float2(acc[18], acc[19]);
        float2 previous_290 = _f2_862;
        a_288[1] = mul_f32x2_noftz(previous_290, corr_265);
        float2 _f2_863 = make_float2(acc[20], acc[21]);
        float2 previous_291 = _f2_863;
        a_288[2] = mul_f32x2_noftz(previous_291, corr_265);
        float2 _f2_864 = make_float2(acc[22], acc[23]);
        float2 previous_292 = _f2_864;
        a_288[3] = mul_f32x2_noftz(previous_292, corr_265);
        float2 _f2_865 = make_float2(weights_263[6], weights_263[6]);
        float2 weight_293 = _f2_865;
        {
            unsigned int sw2_18[4];
            sw2_18[0] = words[84 + woff_286];
            sw2_18[1] = words[84 + woff_286 + 1];
            sw2_18[2] = words[84 + woff_286 + 2];
            sw2_18[3] = words[84 + woff_286 + 3];
            float sw2_f32_18[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_18[_pair * 2])[0]), "=f"((&sw2_f32_18[_pair * 2])[1])
                    : "r"(sw2_18[_pair]));
            }
            float2 _f2_870 = make_float2(sw2_f32_18[0], sw2_f32_18[1]);
            float2 v_68 = _f2_870;
            a_288[0] = fma_f32x2_rn_noftz(weight_293, v_68, a_288[0]);
            float2 _f2_871 = make_float2(sw2_f32_18[2], sw2_f32_18[3]);
            float2 v_0_66 = _f2_871;
            a_288[1] = fma_f32x2_rn_noftz(weight_293, v_0_66, a_288[1]);
            float2 _f2_872 = make_float2(sw2_f32_18[4], sw2_f32_18[5]);
            float2 v_1_34 = _f2_872;
            a_288[2] = fma_f32x2_rn_noftz(weight_293, v_1_34, a_288[2]);
            float2 _f2_873 = make_float2(sw2_f32_18[6], sw2_f32_18[7]);
            float2 v_2_60 = _f2_873;
            a_288[3] = fma_f32x2_rn_noftz(weight_293, v_2_60, a_288[3]);
        }
        float2 _f2_874 = make_float2(weights_263[7], weights_263[7]);
        float2 weight_294 = _f2_874;
        {
            unsigned int sw2_19[4];
            sw2_19[0] = words[98 + woff_286];
            sw2_19[1] = words[98 + woff_286 + 1];
            sw2_19[2] = words[98 + woff_286 + 2];
            sw2_19[3] = words[98 + woff_286 + 3];
            float sw2_f32_19[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_19[_pair * 2])[0]), "=f"((&sw2_f32_19[_pair * 2])[1])
                    : "r"(sw2_19[_pair]));
            }
            float2 _f2_879 = make_float2(sw2_f32_19[0], sw2_f32_19[1]);
            float2 v_69 = _f2_879;
            a_288[0] = fma_f32x2_rn_noftz(weight_294, v_69, a_288[0]);
            float2 _f2_880 = make_float2(sw2_f32_19[2], sw2_f32_19[3]);
            float2 v_0_67 = _f2_880;
            a_288[1] = fma_f32x2_rn_noftz(weight_294, v_0_67, a_288[1]);
            float2 _f2_881 = make_float2(sw2_f32_19[4], sw2_f32_19[5]);
            float2 v_1_35 = _f2_881;
            a_288[2] = fma_f32x2_rn_noftz(weight_294, v_1_35, a_288[2]);
            float2 _f2_882 = make_float2(sw2_f32_19[6], sw2_f32_19[7]);
            float2 v_2_61 = _f2_882;
            a_288[3] = fma_f32x2_rn_noftz(weight_294, v_2_61, a_288[3]);
        }
        float2 _f2_883 = make_float2(weights_263[8], weights_263[8]);
        float2 weight_295 = _f2_883;
        {
            unsigned int sw2_20[4];
            sw2_20[0] = words[112 + woff_286];
            sw2_20[1] = words[112 + woff_286 + 1];
            sw2_20[2] = words[112 + woff_286 + 2];
            sw2_20[3] = words[112 + woff_286 + 3];
            float sw2_f32_20[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_20[_pair * 2])[0]), "=f"((&sw2_f32_20[_pair * 2])[1])
                    : "r"(sw2_20[_pair]));
            }
            float2 _f2_888 = make_float2(sw2_f32_20[0], sw2_f32_20[1]);
            float2 v_70 = _f2_888;
            a_288[0] = fma_f32x2_rn_noftz(weight_295, v_70, a_288[0]);
            float2 _f2_889 = make_float2(sw2_f32_20[2], sw2_f32_20[3]);
            float2 v_0_68 = _f2_889;
            a_288[1] = fma_f32x2_rn_noftz(weight_295, v_0_68, a_288[1]);
            float2 _f2_890 = make_float2(sw2_f32_20[4], sw2_f32_20[5]);
            float2 v_1_36 = _f2_890;
            a_288[2] = fma_f32x2_rn_noftz(weight_295, v_1_36, a_288[2]);
            float2 _f2_891 = make_float2(sw2_f32_20[6], sw2_f32_20[7]);
            float2 v_2_62 = _f2_891;
            a_288[3] = fma_f32x2_rn_noftz(weight_295, v_2_62, a_288[3]);
        }
        acc[16] = a_288[0].x;
        acc[17] = a_288[0].y;
        acc[18] = a_288[1].x;
        acc[19] = a_288[1].y;
        acc[20] = a_288[2].x;
        acc[21] = a_288[2].y;
        acc[22] = a_288[3].x;
        acc[23] = a_288[3].y;
        const int woff_296 = 12;
        int base_297 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_298[4];
        float2 _f2_892 = make_float2(acc[24], acc[25]);
        float2 previous_299 = _f2_892;
        a_298[0] = mul_f32x2_noftz(previous_299, corr_265);
        float2 _f2_893 = make_float2(acc[26], acc[27]);
        float2 previous_300 = _f2_893;
        a_298[1] = mul_f32x2_noftz(previous_300, corr_265);
        float2 _f2_894 = make_float2(weights_263[6], weights_263[6]);
        float2 weight_301 = _f2_894;
        {
            unsigned int sw2_21[4];
            sw2_21[0] = words[84 + woff_296];
            sw2_21[1] = words[84 + woff_296 + 1];
            float sw2_f32_21[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_21[_pair * 2])[0]), "=f"((&sw2_f32_21[_pair * 2])[1])
                    : "r"(sw2_21[_pair]));
            }
            float2 _f2_897 = make_float2(sw2_f32_21[0], sw2_f32_21[1]);
            float2 v_71 = _f2_897;
            a_298[0] = fma_f32x2_rn_noftz(weight_301, v_71, a_298[0]);
            float2 _f2_898 = make_float2(sw2_f32_21[2], sw2_f32_21[3]);
            float2 v_0_69 = _f2_898;
            a_298[1] = fma_f32x2_rn_noftz(weight_301, v_0_69, a_298[1]);
        }
        float2 _f2_899 = make_float2(weights_263[7], weights_263[7]);
        float2 weight_302 = _f2_899;
        {
            unsigned int sw2_22[4];
            sw2_22[0] = words[98 + woff_296];
            sw2_22[1] = words[98 + woff_296 + 1];
            float sw2_f32_22[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_22[_pair * 2])[0]), "=f"((&sw2_f32_22[_pair * 2])[1])
                    : "r"(sw2_22[_pair]));
            }
            float2 _f2_902 = make_float2(sw2_f32_22[0], sw2_f32_22[1]);
            float2 v_72 = _f2_902;
            a_298[0] = fma_f32x2_rn_noftz(weight_302, v_72, a_298[0]);
            float2 _f2_903 = make_float2(sw2_f32_22[2], sw2_f32_22[3]);
            float2 v_0_70 = _f2_903;
            a_298[1] = fma_f32x2_rn_noftz(weight_302, v_0_70, a_298[1]);
        }
        float2 _f2_904 = make_float2(weights_263[8], weights_263[8]);
        float2 weight_303 = _f2_904;
        {
            unsigned int sw2_23[4];
            sw2_23[0] = words[112 + woff_296];
            sw2_23[1] = words[112 + woff_296 + 1];
            float sw2_f32_23[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sw2_f32_23[_pair * 2])[0]), "=f"((&sw2_f32_23[_pair * 2])[1])
                    : "r"(sw2_23[_pair]));
            }
            float2 _f2_907 = make_float2(sw2_f32_23[0], sw2_f32_23[1]);
            float2 v_73 = _f2_907;
            a_298[0] = fma_f32x2_rn_noftz(weight_303, v_73, a_298[0]);
            float2 _f2_908 = make_float2(sw2_f32_23[2], sw2_f32_23[3]);
            float2 v_0_71 = _f2_908;
            a_298[1] = fma_f32x2_rn_noftz(weight_303, v_0_71, a_298[1]);
        }
        acc[24] = a_298[0].x;
        acc[25] = a_298[0].y;
        acc[26] = a_298[1].x;
        acc[27] = a_298[1].y;
        sum_running = sum_running * correction_262 + sum_weights_264;
        max_running = max_new_261;
        float2 _f2_909 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_909;
        float2 _f2_910 = make_float2(acc[0], acc[1]);
        float2 v_74 = _f2_910;
        output_sq_pair = fma_f32x2_rn_noftz(v_74, v_74, output_sq_pair);
        float2 _f2_911 = make_float2(acc[2], acc[3]);
        float2 v_304 = _f2_911;
        output_sq_pair = fma_f32x2_rn_noftz(v_304, v_304, output_sq_pair);
        float2 _f2_912 = make_float2(acc[4], acc[5]);
        float2 v_305 = _f2_912;
        output_sq_pair = fma_f32x2_rn_noftz(v_305, v_305, output_sq_pair);
        float2 _f2_913 = make_float2(acc[6], acc[7]);
        float2 v_306 = _f2_913;
        output_sq_pair = fma_f32x2_rn_noftz(v_306, v_306, output_sq_pair);
        float2 _f2_914 = make_float2(acc[8], acc[9]);
        float2 v_307 = _f2_914;
        output_sq_pair = fma_f32x2_rn_noftz(v_307, v_307, output_sq_pair);
        float2 _f2_915 = make_float2(acc[10], acc[11]);
        float2 v_308 = _f2_915;
        output_sq_pair = fma_f32x2_rn_noftz(v_308, v_308, output_sq_pair);
        float2 _f2_916 = make_float2(acc[12], acc[13]);
        float2 v_309 = _f2_916;
        output_sq_pair = fma_f32x2_rn_noftz(v_309, v_309, output_sq_pair);
        float2 _f2_917 = make_float2(acc[14], acc[15]);
        float2 v_310 = _f2_917;
        output_sq_pair = fma_f32x2_rn_noftz(v_310, v_310, output_sq_pair);
        float2 _f2_918 = make_float2(acc[16], acc[17]);
        float2 v_311 = _f2_918;
        output_sq_pair = fma_f32x2_rn_noftz(v_311, v_311, output_sq_pair);
        float2 _f2_919 = make_float2(acc[18], acc[19]);
        float2 v_312 = _f2_919;
        output_sq_pair = fma_f32x2_rn_noftz(v_312, v_312, output_sq_pair);
        float2 _f2_920 = make_float2(acc[20], acc[21]);
        float2 v_313 = _f2_920;
        output_sq_pair = fma_f32x2_rn_noftz(v_313, v_313, output_sq_pair);
        float2 _f2_921 = make_float2(acc[22], acc[23]);
        float2 v_314 = _f2_921;
        output_sq_pair = fma_f32x2_rn_noftz(v_314, v_314, output_sq_pair);
        float2 _f2_922 = make_float2(acc[24], acc[25]);
        float2 v_315 = _f2_922;
        output_sq_pair = fma_f32x2_rn_noftz(v_315, v_315, output_sq_pair);
        float2 _f2_923 = make_float2(acc[26], acc[27]);
        float2 v_316 = _f2_923;
        output_sq_pair = fma_f32x2_rn_noftz(v_316, v_316, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_45;
        float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_46;
        float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_47;
        float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_48;
        float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_49;
        if (lane == 0) {
            uint32_t _mapa_36;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_36) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_36), "f"(output_sq) : "memory");
            uint32_t _mapa_37;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_37) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_37), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_18 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_18;
        float _shfl_down_19 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_19;
        float _shfl_down_20 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_20;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_3 = rsqrtf(output_total / 7168.0f + output_norm_eps * sum_running * sum_running);
            rsigma_lane = _rsqrt_3;
        }
        float _shfl_9;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_9) : "f"(rsigma_lane), "r"(0));
        float rsigma = _shfl_9;
        float2 _f2_924 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_924;
        int base_317 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_925 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_925, rsigma_pair);
        float2 _f2_926 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_926);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_318 = 2;
        const int acc_idx_319 = value_idx_318;
        float2 _f2_927 = make_float2(acc[acc_idx_319], acc[acc_idx_319 + 1]);
        float2 scaled_pair_320 = mul_f32x2_noftz(_f2_927, rsigma_pair);
        float2 _f2_928 = make_float2(wout[acc_idx_319], wout[acc_idx_319 + 1]);
        float2 normalized_pair_321 = mul_f32x2_noftz(scaled_pair_320, _f2_928);
        output_values[value_idx_318] = normalized_pair_321.x;
        output_values[value_idx_318 + 1] = normalized_pair_321.y;
        const int value_idx_322 = 4;
        const int acc_idx_323 = value_idx_322;
        float2 _f2_929 = make_float2(acc[acc_idx_323], acc[acc_idx_323 + 1]);
        float2 scaled_pair_324 = mul_f32x2_noftz(_f2_929, rsigma_pair);
        float2 _f2_930 = make_float2(wout[acc_idx_323], wout[acc_idx_323 + 1]);
        float2 normalized_pair_325 = mul_f32x2_noftz(scaled_pair_324, _f2_930);
        output_values[value_idx_322] = normalized_pair_325.x;
        output_values[value_idx_322 + 1] = normalized_pair_325.y;
        const int value_idx_326 = 6;
        const int acc_idx_327 = value_idx_326;
        float2 _f2_931 = make_float2(acc[acc_idx_327], acc[acc_idx_327 + 1]);
        float2 scaled_pair_328 = mul_f32x2_noftz(_f2_931, rsigma_pair);
        float2 _f2_932 = make_float2(wout[acc_idx_327], wout[acc_idx_327 + 1]);
        float2 normalized_pair_329 = mul_f32x2_noftz(scaled_pair_328, _f2_932);
        output_values[value_idx_326] = normalized_pair_329.x;
        output_values[value_idx_326 + 1] = normalized_pair_329.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_317 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_330 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_331[8];
        const int value_idx_332 = 0;
        const int acc_idx_333 = 8 + value_idx_332;
        float2 _f2_933 = make_float2(acc[acc_idx_333], acc[acc_idx_333 + 1]);
        float2 scaled_pair_334 = mul_f32x2_noftz(_f2_933, rsigma_pair);
        float2 _f2_934 = make_float2(wout[acc_idx_333], wout[acc_idx_333 + 1]);
        float2 normalized_pair_335 = mul_f32x2_noftz(scaled_pair_334, _f2_934);
        output_values_331[value_idx_332] = normalized_pair_335.x;
        output_values_331[value_idx_332 + 1] = normalized_pair_335.y;
        const int value_idx_336 = 2;
        const int acc_idx_337 = 8 + value_idx_336;
        float2 _f2_935 = make_float2(acc[acc_idx_337], acc[acc_idx_337 + 1]);
        float2 scaled_pair_338 = mul_f32x2_noftz(_f2_935, rsigma_pair);
        float2 _f2_936 = make_float2(wout[acc_idx_337], wout[acc_idx_337 + 1]);
        float2 normalized_pair_339 = mul_f32x2_noftz(scaled_pair_338, _f2_936);
        output_values_331[value_idx_336] = normalized_pair_339.x;
        output_values_331[value_idx_336 + 1] = normalized_pair_339.y;
        const int value_idx_340 = 4;
        const int acc_idx_341 = 8 + value_idx_340;
        float2 _f2_937 = make_float2(acc[acc_idx_341], acc[acc_idx_341 + 1]);
        float2 scaled_pair_342 = mul_f32x2_noftz(_f2_937, rsigma_pair);
        float2 _f2_938 = make_float2(wout[acc_idx_341], wout[acc_idx_341 + 1]);
        float2 normalized_pair_343 = mul_f32x2_noftz(scaled_pair_342, _f2_938);
        output_values_331[value_idx_340] = normalized_pair_343.x;
        output_values_331[value_idx_340 + 1] = normalized_pair_343.y;
        const int value_idx_344 = 6;
        const int acc_idx_345 = 8 + value_idx_344;
        float2 _f2_939 = make_float2(acc[acc_idx_345], acc[acc_idx_345 + 1]);
        float2 scaled_pair_346 = mul_f32x2_noftz(_f2_939, rsigma_pair);
        float2 _f2_940 = make_float2(wout[acc_idx_345], wout[acc_idx_345 + 1]);
        float2 normalized_pair_347 = mul_f32x2_noftz(scaled_pair_346, _f2_940);
        output_values_331[value_idx_344] = normalized_pair_347.x;
        output_values_331[value_idx_344 + 1] = normalized_pair_347.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_331[0 + 0], output_values_331[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_331[0 + 2], output_values_331[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_331[0 + 4], output_values_331[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_331[0 + 6], output_values_331[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_330 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_348 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_349[8];
        const int value_idx_350 = 0;
        const int acc_idx_351 = 16 + value_idx_350;
        float2 _f2_941 = make_float2(acc[acc_idx_351], acc[acc_idx_351 + 1]);
        float2 scaled_pair_352 = mul_f32x2_noftz(_f2_941, rsigma_pair);
        float2 _f2_942 = make_float2(wout[acc_idx_351], wout[acc_idx_351 + 1]);
        float2 normalized_pair_353 = mul_f32x2_noftz(scaled_pair_352, _f2_942);
        output_values_349[value_idx_350] = normalized_pair_353.x;
        output_values_349[value_idx_350 + 1] = normalized_pair_353.y;
        const int value_idx_354 = 2;
        const int acc_idx_355 = 16 + value_idx_354;
        float2 _f2_943 = make_float2(acc[acc_idx_355], acc[acc_idx_355 + 1]);
        float2 scaled_pair_356 = mul_f32x2_noftz(_f2_943, rsigma_pair);
        float2 _f2_944 = make_float2(wout[acc_idx_355], wout[acc_idx_355 + 1]);
        float2 normalized_pair_357 = mul_f32x2_noftz(scaled_pair_356, _f2_944);
        output_values_349[value_idx_354] = normalized_pair_357.x;
        output_values_349[value_idx_354 + 1] = normalized_pair_357.y;
        const int value_idx_358 = 4;
        const int acc_idx_359 = 16 + value_idx_358;
        float2 _f2_945 = make_float2(acc[acc_idx_359], acc[acc_idx_359 + 1]);
        float2 scaled_pair_360 = mul_f32x2_noftz(_f2_945, rsigma_pair);
        float2 _f2_946 = make_float2(wout[acc_idx_359], wout[acc_idx_359 + 1]);
        float2 normalized_pair_361 = mul_f32x2_noftz(scaled_pair_360, _f2_946);
        output_values_349[value_idx_358] = normalized_pair_361.x;
        output_values_349[value_idx_358 + 1] = normalized_pair_361.y;
        const int value_idx_362 = 6;
        const int acc_idx_363 = 16 + value_idx_362;
        float2 _f2_947 = make_float2(acc[acc_idx_363], acc[acc_idx_363 + 1]);
        float2 scaled_pair_364 = mul_f32x2_noftz(_f2_947, rsigma_pair);
        float2 _f2_948 = make_float2(wout[acc_idx_363], wout[acc_idx_363 + 1]);
        float2 normalized_pair_365 = mul_f32x2_noftz(scaled_pair_364, _f2_948);
        output_values_349[value_idx_362] = normalized_pair_365.x;
        output_values_349[value_idx_362 + 1] = normalized_pair_365.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_349[0 + 0], output_values_349[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_349[0 + 2], output_values_349[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_349[0 + 4], output_values_349[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_349[0 + 6], output_values_349[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_348 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_366 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_367[8];
        const int value_idx_368 = 0;
        const int acc_idx_369 = 24 + value_idx_368;
        float2 _f2_949 = make_float2(acc[acc_idx_369], acc[acc_idx_369 + 1]);
        float2 scaled_pair_370 = mul_f32x2_noftz(_f2_949, rsigma_pair);
        float2 _f2_950 = make_float2(wout[acc_idx_369], wout[acc_idx_369 + 1]);
        float2 normalized_pair_371 = mul_f32x2_noftz(scaled_pair_370, _f2_950);
        output_values_367[value_idx_368] = normalized_pair_371.x;
        output_values_367[value_idx_368 + 1] = normalized_pair_371.y;
        const int value_idx_372 = 2;
        const int acc_idx_373 = 24 + value_idx_372;
        float2 _f2_951 = make_float2(acc[acc_idx_373], acc[acc_idx_373 + 1]);
        float2 scaled_pair_374 = mul_f32x2_noftz(_f2_951, rsigma_pair);
        float2 _f2_952 = make_float2(wout[acc_idx_373], wout[acc_idx_373 + 1]);
        float2 normalized_pair_375 = mul_f32x2_noftz(scaled_pair_374, _f2_952);
        output_values_367[value_idx_372] = normalized_pair_375.x;
        output_values_367[value_idx_372 + 1] = normalized_pair_375.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_367[0 + 0], output_values_367[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_367[0 + 2], output_values_367[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_366]) = _pk2;
        }
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
