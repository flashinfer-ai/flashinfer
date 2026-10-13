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
#define SMEM_STATS_STAGE_BYTES 384
#define SMEM_STATS_STRIDE 384
#define SMEM_OUT_STATS_OFF 384
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 512
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
kernel_cake_kimi_k3_attn_res_b0f25d64586277608a9b(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 384);
    const int out_stats_addr = smem + 384;

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
        unsigned int words[84];
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
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_18[0 + 0] = _uv4_6.x;
                _vec_load_18[0 + 1] = _uv4_6.y;
                _vec_load_18[0 + 2] = _uv4_6.z;
                _vec_load_18[0 + 3] = _uv4_6.w;
            }
            dwords[woff] = _vec_load_18[0];
            dwords[woff + 1] = _vec_load_18[1];
            dwords[woff + 2] = _vec_load_18[2];
            dwords[woff + 3] = _vec_load_18[3];
        }
        float _vec_load_21[8];
        {
            const uint4* _vptr_7 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
        }
        float _vec_load_22[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_22[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        q[0] = _vec_load_21[0] * _vec_load_22[0];
        q[1] = _vec_load_21[1] * _vec_load_22[1];
        q[2] = _vec_load_21[2] * _vec_load_22[2];
        q[3] = _vec_load_21[3] * _vec_load_22[3];
        q[4] = _vec_load_21[4] * _vec_load_22[4];
        q[5] = _vec_load_21[5] * _vec_load_22[5];
        q[6] = _vec_load_21[6] * _vec_load_22[6];
        q[7] = _vec_load_21[7] * _vec_load_22[7];
        float _vec_load_23[8];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(output_norm_weight + base);
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
                        : "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        wout[0] = _vec_load_23[0];
        wout[1] = _vec_load_23[1];
        wout[2] = _vec_load_23[2];
        wout[3] = _vec_load_23[3];
        wout[4] = _vec_load_23[4];
        wout[5] = _vec_load_23[5];
        wout[6] = _vec_load_23[6];
        wout[7] = _vec_load_23[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_24[4];
            {
                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_24[0 + 0] = _uv4_10.x;
                _vec_load_24[0 + 1] = _uv4_10.y;
                _vec_load_24[0 + 2] = _uv4_10.z;
                _vec_load_24[0 + 3] = _uv4_10.w;
            }
            words[woff_1] = _vec_load_24[0];
            words[woff_1 + 1] = _vec_load_24[1];
            words[woff_1 + 2] = _vec_load_24[2];
            words[woff_1 + 3] = _vec_load_24[3];
        }
        {
            unsigned int _vec_load_27[4];
            {
                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_27[0 + 0] = _uv4_11.x;
                _vec_load_27[0 + 1] = _uv4_11.y;
                _vec_load_27[0 + 2] = _uv4_11.z;
                _vec_load_27[0 + 3] = _uv4_11.w;
            }
            words[14 + woff_1] = _vec_load_27[0];
            words[14 + woff_1 + 1] = _vec_load_27[1];
            words[14 + woff_1 + 2] = _vec_load_27[2];
            words[14 + woff_1 + 3] = _vec_load_27[3];
        }
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_30[0 + 0] = _uv4_12.x;
                _vec_load_30[0 + 1] = _uv4_12.y;
                _vec_load_30[0 + 2] = _uv4_12.z;
                _vec_load_30[0 + 3] = _uv4_12.w;
            }
            words[28 + woff_1] = _vec_load_30[0];
            words[28 + woff_1 + 1] = _vec_load_30[1];
            words[28 + woff_1 + 2] = _vec_load_30[2];
            words[28 + woff_1 + 3] = _vec_load_30[3];
        }
        {
            unsigned int _vec_load_33[4];
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_33[0 + 0] = _uv4_13.x;
                _vec_load_33[0 + 1] = _uv4_13.y;
                _vec_load_33[0 + 2] = _uv4_13.z;
                _vec_load_33[0 + 3] = _uv4_13.w;
            }
            words[42 + woff_1] = _vec_load_33[0];
            words[42 + woff_1 + 1] = _vec_load_33[1];
            words[42 + woff_1 + 2] = _vec_load_33[2];
            words[42 + woff_1 + 3] = _vec_load_33[3];
        }
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_36[0 + 0] = _uv4_14.x;
                _vec_load_36[0 + 1] = _uv4_14.y;
                _vec_load_36[0 + 2] = _uv4_14.z;
                _vec_load_36[0 + 3] = _uv4_14.w;
            }
            words[56 + woff_1] = _vec_load_36[0];
            words[56 + woff_1 + 1] = _vec_load_36[1];
            words[56 + woff_1 + 2] = _vec_load_36[2];
            words[56 + woff_1 + 3] = _vec_load_36[3];
        }
        {
            unsigned int _vec_load_39[4];
            {
                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_39[0 + 0] = _uv4_15.x;
                _vec_load_39[0 + 1] = _uv4_15.y;
                _vec_load_39[0 + 2] = _uv4_15.z;
                _vec_load_39[0 + 3] = _uv4_15.w;
            }
            words[70 + woff_1] = _vec_load_39[0];
            words[70 + woff_1 + 1] = _vec_load_39[1];
            words[70 + woff_1 + 2] = _vec_load_39[2];
            words[70 + woff_1 + 3] = _vec_load_39[3];
        }
        {
            unsigned int _vec_load_42[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_42[0 + 0] = _uv4_16.x;
                _vec_load_42[0 + 1] = _uv4_16.y;
                _vec_load_42[0 + 2] = _uv4_16.z;
                _vec_load_42[0 + 3] = _uv4_16.w;
            }
            dwords[woff_1] = _vec_load_42[0];
            dwords[woff_1 + 1] = _vec_load_42[1];
            dwords[woff_1 + 2] = _vec_load_42[2];
            dwords[woff_1 + 3] = _vec_load_42[3];
        }
        float _vec_load_45[8];
        {
            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_45[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_45[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_17[_pair]));
                }
            }
        }
        float _vec_load_46[8];
        {
            const uint4* _vptr_18 = reinterpret_cast<const uint4*>(qk_weight + base_0);
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
                        : "=f"((&_vec_load_46[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_46[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_18[_pair]));
                }
            }
        }
        q[8] = _vec_load_45[0] * _vec_load_46[0];
        q[9] = _vec_load_45[1] * _vec_load_46[1];
        q[10] = _vec_load_45[2] * _vec_load_46[2];
        q[11] = _vec_load_45[3] * _vec_load_46[3];
        q[12] = _vec_load_45[4] * _vec_load_46[4];
        q[13] = _vec_load_45[5] * _vec_load_46[5];
        q[14] = _vec_load_45[6] * _vec_load_46[6];
        q[15] = _vec_load_45[7] * _vec_load_46[7];
        float _vec_load_47[8];
        {
            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_47[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_47[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        wout[8] = _vec_load_47[0];
        wout[9] = _vec_load_47[1];
        wout[10] = _vec_load_47[2];
        wout[11] = _vec_load_47[3];
        wout[12] = _vec_load_47[4];
        wout[13] = _vec_load_47[5];
        wout[14] = _vec_load_47[6];
        wout[15] = _vec_load_47[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_48[4];
            {
                uint4 _uv4_20 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_48[0 + 0] = _uv4_20.x;
                _vec_load_48[0 + 1] = _uv4_20.y;
                _vec_load_48[0 + 2] = _uv4_20.z;
                _vec_load_48[0 + 3] = _uv4_20.w;
            }
            words[woff_3] = _vec_load_48[0];
            words[woff_3 + 1] = _vec_load_48[1];
            words[woff_3 + 2] = _vec_load_48[2];
            words[woff_3 + 3] = _vec_load_48[3];
        }
        {
            unsigned int _vec_load_51[4];
            {
                uint4 _uv4_21 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_51[0 + 0] = _uv4_21.x;
                _vec_load_51[0 + 1] = _uv4_21.y;
                _vec_load_51[0 + 2] = _uv4_21.z;
                _vec_load_51[0 + 3] = _uv4_21.w;
            }
            words[14 + woff_3] = _vec_load_51[0];
            words[14 + woff_3 + 1] = _vec_load_51[1];
            words[14 + woff_3 + 2] = _vec_load_51[2];
            words[14 + woff_3 + 3] = _vec_load_51[3];
        }
        {
            unsigned int _vec_load_54[4];
            {
                uint4 _uv4_22 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_54[0 + 0] = _uv4_22.x;
                _vec_load_54[0 + 1] = _uv4_22.y;
                _vec_load_54[0 + 2] = _uv4_22.z;
                _vec_load_54[0 + 3] = _uv4_22.w;
            }
            words[28 + woff_3] = _vec_load_54[0];
            words[28 + woff_3 + 1] = _vec_load_54[1];
            words[28 + woff_3 + 2] = _vec_load_54[2];
            words[28 + woff_3 + 3] = _vec_load_54[3];
        }
        {
            unsigned int _vec_load_57[4];
            {
                uint4 _uv4_23 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_57[0 + 0] = _uv4_23.x;
                _vec_load_57[0 + 1] = _uv4_23.y;
                _vec_load_57[0 + 2] = _uv4_23.z;
                _vec_load_57[0 + 3] = _uv4_23.w;
            }
            words[42 + woff_3] = _vec_load_57[0];
            words[42 + woff_3 + 1] = _vec_load_57[1];
            words[42 + woff_3 + 2] = _vec_load_57[2];
            words[42 + woff_3 + 3] = _vec_load_57[3];
        }
        {
            unsigned int _vec_load_60[4];
            {
                uint4 _uv4_24 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_60[0 + 0] = _uv4_24.x;
                _vec_load_60[0 + 1] = _uv4_24.y;
                _vec_load_60[0 + 2] = _uv4_24.z;
                _vec_load_60[0 + 3] = _uv4_24.w;
            }
            words[56 + woff_3] = _vec_load_60[0];
            words[56 + woff_3 + 1] = _vec_load_60[1];
            words[56 + woff_3 + 2] = _vec_load_60[2];
            words[56 + woff_3 + 3] = _vec_load_60[3];
        }
        {
            unsigned int _vec_load_63[4];
            {
                uint4 _uv4_25 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_63[0 + 0] = _uv4_25.x;
                _vec_load_63[0 + 1] = _uv4_25.y;
                _vec_load_63[0 + 2] = _uv4_25.z;
                _vec_load_63[0 + 3] = _uv4_25.w;
            }
            words[70 + woff_3] = _vec_load_63[0];
            words[70 + woff_3 + 1] = _vec_load_63[1];
            words[70 + woff_3 + 2] = _vec_load_63[2];
            words[70 + woff_3 + 3] = _vec_load_63[3];
        }
        {
            unsigned int _vec_load_66[4];
            {
                uint4 _uv4_26 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_66[0 + 0] = _uv4_26.x;
                _vec_load_66[0 + 1] = _uv4_26.y;
                _vec_load_66[0 + 2] = _uv4_26.z;
                _vec_load_66[0 + 3] = _uv4_26.w;
            }
            dwords[woff_3] = _vec_load_66[0];
            dwords[woff_3 + 1] = _vec_load_66[1];
            dwords[woff_3 + 2] = _vec_load_66[2];
            dwords[woff_3 + 3] = _vec_load_66[3];
        }
        float _vec_load_69[8];
        {
            const uint4* _vptr_27 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_27[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_27[_blk] = _vptr_27[_blk];
                uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_69[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_69[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_27[_pair]));
                }
            }
        }
        float _vec_load_70[8];
        {
            const uint4* _vptr_28 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_28[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_28[_blk] = _vptr_28[_blk];
                uint32_t* _vpairs_28 = reinterpret_cast<uint32_t*>(&_vld_28[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_70[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_70[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_28[_pair]));
                }
            }
        }
        q[16] = _vec_load_69[0] * _vec_load_70[0];
        q[17] = _vec_load_69[1] * _vec_load_70[1];
        q[18] = _vec_load_69[2] * _vec_load_70[2];
        q[19] = _vec_load_69[3] * _vec_load_70[3];
        q[20] = _vec_load_69[4] * _vec_load_70[4];
        q[21] = _vec_load_69[5] * _vec_load_70[5];
        q[22] = _vec_load_69[6] * _vec_load_70[6];
        q[23] = _vec_load_69[7] * _vec_load_70[7];
        float _vec_load_71[8];
        {
            const uint4* _vptr_29 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_29[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_29[_blk] = _vptr_29[_blk];
                uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_71[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_71[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_29[_pair]));
                }
            }
        }
        wout[16] = _vec_load_71[0];
        wout[17] = _vec_load_71[1];
        wout[18] = _vec_load_71[2];
        wout[19] = _vec_load_71[3];
        wout[20] = _vec_load_71[4];
        wout[21] = _vec_load_71[5];
        wout[22] = _vec_load_71[6];
        wout[23] = _vec_load_71[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_73[1];
            {
                _vec_load_73[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_73[0];
            unsigned int _vec_load_74[1];
            {
                _vec_load_74[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_74[0];
        }
        {
            unsigned int _vec_load_76[1];
            {
                _vec_load_76[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_76[0];
            unsigned int _vec_load_77[1];
            {
                _vec_load_77[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_77[0];
        }
        {
            unsigned int _vec_load_79[1];
            {
                _vec_load_79[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_79[0];
            unsigned int _vec_load_80[1];
            {
                _vec_load_80[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_80[0];
        }
        {
            unsigned int _vec_load_82[1];
            {
                _vec_load_82[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[42 + woff_5] = _vec_load_82[0];
            unsigned int _vec_load_83[1];
            {
                _vec_load_83[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[42 + woff_5 + 1] = _vec_load_83[0];
        }
        {
            unsigned int _vec_load_85[1];
            {
                _vec_load_85[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[56 + woff_5] = _vec_load_85[0];
            unsigned int _vec_load_86[1];
            {
                _vec_load_86[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[56 + woff_5 + 1] = _vec_load_86[0];
        }
        {
            unsigned int _vec_load_88[1];
            {
                _vec_load_88[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[70 + woff_5] = _vec_load_88[0];
            unsigned int _vec_load_89[1];
            {
                _vec_load_89[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[70 + woff_5 + 1] = _vec_load_89[0];
        }
        {
            unsigned int _vec_load_91[1];
            {
                _vec_load_91[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_91[0];
            unsigned int _vec_load_92[1];
            {
                _vec_load_92[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_92[0];
        }
        float _vec_load_93[4];
        {
            uint2 _vld_30;
            _vld_30 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_93[0 + _pair * 2])[0]), "=f"((&_vec_load_93[0 + _pair * 2])[1])
                    : "r"(_vpairs_30[_pair]));
            }
        }
        float _vec_load_94[4];
        {
            uint2 _vld_31;
            _vld_31 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_94[0 + _pair * 2])[0]), "=f"((&_vec_load_94[0 + _pair * 2])[1])
                    : "r"(_vpairs_31[_pair]));
            }
        }
        q[24] = _vec_load_93[0] * _vec_load_94[0];
        q[25] = _vec_load_93[1] * _vec_load_94[1];
        q[26] = _vec_load_93[2] * _vec_load_94[2];
        q[27] = _vec_load_93[3] * _vec_load_94[3];
        float _vec_load_95[4];
        {
            uint2 _vld_32;
            _vld_32 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_95[0 + _pair * 2])[0]), "=f"((&_vec_load_95[0 + _pair * 2])[1])
                    : "r"(_vpairs_32[_pair]));
            }
        }
        wout[24] = _vec_load_95[0];
        wout[25] = _vec_load_95[1];
        wout[26] = _vec_load_95[2];
        wout[27] = _vec_load_95[3];
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
        float2 sq[6];
        float2 dot[6];
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
            float2 _f2_18 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_18;
            float2 _f2_19 = make_float2(q[0], q[1]);
            float2 qp = _f2_19;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_20 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_20;
            float2 _f2_21 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_21;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_22 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_22;
            float2 _f2_23 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_23;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_24 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_24;
            float2 _f2_25 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_25;
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
            float2 _f2_32 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_32;
            float2 _f2_33 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_33;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_34 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_34;
            float2 _f2_35 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_35;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_36 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_36;
            float2 _f2_37 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_37;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_38 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_38;
            float2 _f2_39 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_39;
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
            float2 _f2_46 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_46;
            float2 _f2_47 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_47;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_48 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_48;
            float2 _f2_49 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_49;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_50 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_50;
            float2 _f2_51 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_51;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_52 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_52;
            float2 _f2_53 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_53;
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
            float2 _f2_60 = make_float2(sw_10_f32[0], sw_10_f32[1]);
            float2 v_5 = _f2_60;
            float2 _f2_61 = make_float2(q[0], q[1]);
            float2 qp_6 = _f2_61;
            sq[3] = fma_f32x2_rn_noftz(v_5, v_5, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_5, qp_6, dot[3]);
            float2 _f2_62 = make_float2(sw_10_f32[2], sw_10_f32[3]);
            float2 v_0_3 = _f2_62;
            float2 _f2_63 = make_float2(q[2], q[3]);
            float2 qp_1_3 = _f2_63;
            sq[3] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[3]);
            float2 _f2_64 = make_float2(sw_10_f32[4], sw_10_f32[5]);
            float2 v_2_3 = _f2_64;
            float2 _f2_65 = make_float2(q[4], q[5]);
            float2 qp_3_3 = _f2_65;
            sq[3] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[3]);
            float2 _f2_66 = make_float2(sw_10_f32[6], sw_10_f32[7]);
            float2 v_4_3 = _f2_66;
            float2 _f2_67 = make_float2(q[6], q[7]);
            float2 qp_5_3 = _f2_67;
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
            float2 _f2_74 = make_float2(sw_11_f32[0], sw_11_f32[1]);
            float2 v_6 = _f2_74;
            float2 _f2_75 = make_float2(q[0], q[1]);
            float2 qp_7 = _f2_75;
            sq[4] = fma_f32x2_rn_noftz(v_6, v_6, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_6, qp_7, dot[4]);
            float2 _f2_76 = make_float2(sw_11_f32[2], sw_11_f32[3]);
            float2 v_0_4 = _f2_76;
            float2 _f2_77 = make_float2(q[2], q[3]);
            float2 qp_1_4 = _f2_77;
            sq[4] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[4]);
            float2 _f2_78 = make_float2(sw_11_f32[4], sw_11_f32[5]);
            float2 v_2_4 = _f2_78;
            float2 _f2_79 = make_float2(q[4], q[5]);
            float2 qp_3_4 = _f2_79;
            sq[4] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[4]);
            float2 _f2_80 = make_float2(sw_11_f32[6], sw_11_f32[7]);
            float2 v_4_4 = _f2_80;
            float2 _f2_81 = make_float2(q[6], q[7]);
            float2 qp_5_4 = _f2_81;
            sq[4] = fma_f32x2_rn_noftz(v_4_4, v_4_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_4, qp_5_4, dot[4]);
        }
        unsigned int sw_12[4];
        sw_12[0] = words[70 + woff_7];
        sw_12[1] = words[70 + woff_7 + 1];
        sw_12[2] = words[70 + woff_7 + 2];
        sw_12[3] = words[70 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_12[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_12[0] = __as_u32(mixed);
            words[70 + woff_7] = sw_12[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_12[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_12[1] = __as_u32(mixed_2);
            words[70 + woff_7 + 1] = sw_12[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_12[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_12[2] = __as_u32(mixed_5);
            words[70 + woff_7 + 2] = sw_12[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_12[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_12[3] = __as_u32(mixed_8);
            words[70 + woff_7 + 3] = sw_12[3];
            {
                int4 _iv4 = make_int4(sw_12[0 + 0], sw_12[0 + 1], sw_12[0 + 2], sw_12[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
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
            float2 _f2_88 = make_float2(sw_12_f32[0], sw_12_f32[1]);
            float2 v_7 = _f2_88;
            float2 _f2_89 = make_float2(q[0], q[1]);
            float2 qp_8 = _f2_89;
            sq[5] = fma_f32x2_rn_noftz(v_7, v_7, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_7, qp_8, dot[5]);
            float2 _f2_90 = make_float2(sw_12_f32[2], sw_12_f32[3]);
            float2 v_0_5 = _f2_90;
            float2 _f2_91 = make_float2(q[2], q[3]);
            float2 qp_1_5 = _f2_91;
            sq[5] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[5]);
            float2 _f2_92 = make_float2(sw_12_f32[4], sw_12_f32[5]);
            float2 v_2_5 = _f2_92;
            float2 _f2_93 = make_float2(q[4], q[5]);
            float2 qp_3_5 = _f2_93;
            sq[5] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[5]);
            float2 _f2_94 = make_float2(sw_12_f32[6], sw_12_f32[7]);
            float2 v_4_5 = _f2_94;
            float2 _f2_95 = make_float2(q[6], q[7]);
            float2 qp_5_5 = _f2_95;
            sq[5] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[5]);
        }
        int base_13 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_14 = 4;
        unsigned int sw_15[4];
        sw_15[0] = words[woff_14];
        sw_15[1] = words[woff_14 + 1];
        sw_15[2] = words[woff_14 + 2];
        sw_15[3] = words[woff_14 + 3];
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
            fsrc[8] = sw_15_f32[0];
            fsrc[9] = sw_15_f32[1];
            fsrc[10] = sw_15_f32[2];
            fsrc[11] = sw_15_f32[3];
            fsrc[12] = sw_15_f32[4];
            fsrc[13] = sw_15_f32[5];
            fsrc[14] = sw_15_f32[6];
            fsrc[15] = sw_15_f32[7];
        }
        {
            float2 _f2_102 = make_float2(sw_15_f32[0], sw_15_f32[1]);
            float2 v_8 = _f2_102;
            float2 _f2_103 = make_float2(q[8], q[9]);
            float2 qp_9 = _f2_103;
            sq[0] = fma_f32x2_rn_noftz(v_8, v_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_8, qp_9, dot[0]);
            float2 _f2_104 = make_float2(sw_15_f32[2], sw_15_f32[3]);
            float2 v_0_6 = _f2_104;
            float2 _f2_105 = make_float2(q[10], q[11]);
            float2 qp_1_6 = _f2_105;
            sq[0] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[0]);
            float2 _f2_106 = make_float2(sw_15_f32[4], sw_15_f32[5]);
            float2 v_2_6 = _f2_106;
            float2 _f2_107 = make_float2(q[12], q[13]);
            float2 qp_3_6 = _f2_107;
            sq[0] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[0]);
            float2 _f2_108 = make_float2(sw_15_f32[6], sw_15_f32[7]);
            float2 v_4_6 = _f2_108;
            float2 _f2_109 = make_float2(q[14], q[15]);
            float2 qp_5_6 = _f2_109;
            sq[0] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[0]);
        }
        unsigned int sw_16[4];
        sw_16[0] = words[14 + woff_14];
        sw_16[1] = words[14 + woff_14 + 1];
        sw_16[2] = words[14 + woff_14 + 2];
        sw_16[3] = words[14 + woff_14 + 3];
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
            fsrc[36] = sw_16_f32[0];
            fsrc[37] = sw_16_f32[1];
            fsrc[38] = sw_16_f32[2];
            fsrc[39] = sw_16_f32[3];
            fsrc[40] = sw_16_f32[4];
            fsrc[41] = sw_16_f32[5];
            fsrc[42] = sw_16_f32[6];
            fsrc[43] = sw_16_f32[7];
        }
        {
            float2 _f2_116 = make_float2(sw_16_f32[0], sw_16_f32[1]);
            float2 v_9 = _f2_116;
            float2 _f2_117 = make_float2(q[8], q[9]);
            float2 qp_10 = _f2_117;
            sq[1] = fma_f32x2_rn_noftz(v_9, v_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_9, qp_10, dot[1]);
            float2 _f2_118 = make_float2(sw_16_f32[2], sw_16_f32[3]);
            float2 v_0_7 = _f2_118;
            float2 _f2_119 = make_float2(q[10], q[11]);
            float2 qp_1_7 = _f2_119;
            sq[1] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[1]);
            float2 _f2_120 = make_float2(sw_16_f32[4], sw_16_f32[5]);
            float2 v_2_7 = _f2_120;
            float2 _f2_121 = make_float2(q[12], q[13]);
            float2 qp_3_7 = _f2_121;
            sq[1] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[1]);
            float2 _f2_122 = make_float2(sw_16_f32[6], sw_16_f32[7]);
            float2 v_4_7 = _f2_122;
            float2 _f2_123 = make_float2(q[14], q[15]);
            float2 qp_5_7 = _f2_123;
            sq[1] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[1]);
        }
        unsigned int sw_17[4];
        sw_17[0] = words[28 + woff_14];
        sw_17[1] = words[28 + woff_14 + 1];
        sw_17[2] = words[28 + woff_14 + 2];
        sw_17[3] = words[28 + woff_14 + 3];
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
            fsrc[64] = sw_17_f32[0];
            fsrc[65] = sw_17_f32[1];
            fsrc[66] = sw_17_f32[2];
            fsrc[67] = sw_17_f32[3];
            fsrc[68] = sw_17_f32[4];
            fsrc[69] = sw_17_f32[5];
            fsrc[70] = sw_17_f32[6];
            fsrc[71] = sw_17_f32[7];
        }
        {
            float2 _f2_130 = make_float2(sw_17_f32[0], sw_17_f32[1]);
            float2 v_10 = _f2_130;
            float2 _f2_131 = make_float2(q[8], q[9]);
            float2 qp_11 = _f2_131;
            sq[2] = fma_f32x2_rn_noftz(v_10, v_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_10, qp_11, dot[2]);
            float2 _f2_132 = make_float2(sw_17_f32[2], sw_17_f32[3]);
            float2 v_0_8 = _f2_132;
            float2 _f2_133 = make_float2(q[10], q[11]);
            float2 qp_1_8 = _f2_133;
            sq[2] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[2]);
            float2 _f2_134 = make_float2(sw_17_f32[4], sw_17_f32[5]);
            float2 v_2_8 = _f2_134;
            float2 _f2_135 = make_float2(q[12], q[13]);
            float2 qp_3_8 = _f2_135;
            sq[2] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[2]);
            float2 _f2_136 = make_float2(sw_17_f32[6], sw_17_f32[7]);
            float2 v_4_8 = _f2_136;
            float2 _f2_137 = make_float2(q[14], q[15]);
            float2 qp_5_8 = _f2_137;
            sq[2] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[2]);
        }
        unsigned int sw_18[4];
        sw_18[0] = words[42 + woff_14];
        sw_18[1] = words[42 + woff_14 + 1];
        sw_18[2] = words[42 + woff_14 + 2];
        sw_18[3] = words[42 + woff_14 + 3];
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
            float2 _f2_144 = make_float2(sw_18_f32[0], sw_18_f32[1]);
            float2 v_11 = _f2_144;
            float2 _f2_145 = make_float2(q[8], q[9]);
            float2 qp_12 = _f2_145;
            sq[3] = fma_f32x2_rn_noftz(v_11, v_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_11, qp_12, dot[3]);
            float2 _f2_146 = make_float2(sw_18_f32[2], sw_18_f32[3]);
            float2 v_0_9 = _f2_146;
            float2 _f2_147 = make_float2(q[10], q[11]);
            float2 qp_1_9 = _f2_147;
            sq[3] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_9, qp_1_9, dot[3]);
            float2 _f2_148 = make_float2(sw_18_f32[4], sw_18_f32[5]);
            float2 v_2_9 = _f2_148;
            float2 _f2_149 = make_float2(q[12], q[13]);
            float2 qp_3_9 = _f2_149;
            sq[3] = fma_f32x2_rn_noftz(v_2_9, v_2_9, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[3]);
            float2 _f2_150 = make_float2(sw_18_f32[6], sw_18_f32[7]);
            float2 v_4_9 = _f2_150;
            float2 _f2_151 = make_float2(q[14], q[15]);
            float2 qp_5_9 = _f2_151;
            sq[3] = fma_f32x2_rn_noftz(v_4_9, v_4_9, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_9, qp_5_9, dot[3]);
        }
        unsigned int sw_19[4];
        sw_19[0] = words[56 + woff_14];
        sw_19[1] = words[56 + woff_14 + 1];
        sw_19[2] = words[56 + woff_14 + 2];
        sw_19[3] = words[56 + woff_14 + 3];
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
            float2 _f2_158 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_12 = _f2_158;
            float2 _f2_159 = make_float2(q[8], q[9]);
            float2 qp_13 = _f2_159;
            sq[4] = fma_f32x2_rn_noftz(v_12, v_12, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_12, qp_13, dot[4]);
            float2 _f2_160 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_10 = _f2_160;
            float2 _f2_161 = make_float2(q[10], q[11]);
            float2 qp_1_10 = _f2_161;
            sq[4] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_10, qp_1_10, dot[4]);
            float2 _f2_162 = make_float2(sw_19_f32[4], sw_19_f32[5]);
            float2 v_2_10 = _f2_162;
            float2 _f2_163 = make_float2(q[12], q[13]);
            float2 qp_3_10 = _f2_163;
            sq[4] = fma_f32x2_rn_noftz(v_2_10, v_2_10, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[4]);
            float2 _f2_164 = make_float2(sw_19_f32[6], sw_19_f32[7]);
            float2 v_4_10 = _f2_164;
            float2 _f2_165 = make_float2(q[14], q[15]);
            float2 qp_5_10 = _f2_165;
            sq[4] = fma_f32x2_rn_noftz(v_4_10, v_4_10, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_10, qp_5_10, dot[4]);
        }
        unsigned int sw_20[4];
        sw_20[0] = words[70 + woff_14];
        sw_20[1] = words[70 + woff_14 + 1];
        sw_20[2] = words[70 + woff_14 + 2];
        sw_20[3] = words[70 + woff_14 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_20[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_14]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_20[0] = __as_u32(mixed_1);
            words[70 + woff_14] = sw_20[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_20[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_14 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_20[1] = __as_u32(mixed_2_1);
            words[70 + woff_14 + 1] = sw_20[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_20[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_14 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_20[2] = __as_u32(mixed_5_1);
            words[70 + woff_14 + 2] = sw_20[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_20[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_14 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_20[3] = __as_u32(mixed_8_1);
            words[70 + woff_14 + 3] = sw_20[3];
            {
                int4 _iv4 = make_int4(sw_20[0 + 0], sw_20[0 + 1], sw_20[0 + 2], sw_20[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_13)) + 0) = _iv4;
            }
        }
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
            float2 _f2_172 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_13 = _f2_172;
            float2 _f2_173 = make_float2(q[8], q[9]);
            float2 qp_14 = _f2_173;
            sq[5] = fma_f32x2_rn_noftz(v_13, v_13, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_13, qp_14, dot[5]);
            float2 _f2_174 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_0_11 = _f2_174;
            float2 _f2_175 = make_float2(q[10], q[11]);
            float2 qp_1_11 = _f2_175;
            sq[5] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_11, qp_1_11, dot[5]);
            float2 _f2_176 = make_float2(sw_20_f32[4], sw_20_f32[5]);
            float2 v_2_11 = _f2_176;
            float2 _f2_177 = make_float2(q[12], q[13]);
            float2 qp_3_11 = _f2_177;
            sq[5] = fma_f32x2_rn_noftz(v_2_11, v_2_11, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[5]);
            float2 _f2_178 = make_float2(sw_20_f32[6], sw_20_f32[7]);
            float2 v_4_11 = _f2_178;
            float2 _f2_179 = make_float2(q[14], q[15]);
            float2 qp_5_11 = _f2_179;
            sq[5] = fma_f32x2_rn_noftz(v_4_11, v_4_11, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_11, qp_5_11, dot[5]);
        }
        int base_21 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_22 = 8;
        unsigned int sw_23[4];
        sw_23[0] = words[woff_22];
        sw_23[1] = words[woff_22 + 1];
        sw_23[2] = words[woff_22 + 2];
        sw_23[3] = words[woff_22 + 3];
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
            fsrc[16] = sw_23_f32[0];
            fsrc[17] = sw_23_f32[1];
            fsrc[18] = sw_23_f32[2];
            fsrc[19] = sw_23_f32[3];
            fsrc[20] = sw_23_f32[4];
            fsrc[21] = sw_23_f32[5];
            fsrc[22] = sw_23_f32[6];
            fsrc[23] = sw_23_f32[7];
        }
        {
            float2 _f2_186 = make_float2(sw_23_f32[0], sw_23_f32[1]);
            float2 v_14 = _f2_186;
            float2 _f2_187 = make_float2(q[16], q[17]);
            float2 qp_15 = _f2_187;
            sq[0] = fma_f32x2_rn_noftz(v_14, v_14, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_14, qp_15, dot[0]);
            float2 _f2_188 = make_float2(sw_23_f32[2], sw_23_f32[3]);
            float2 v_0_12 = _f2_188;
            float2 _f2_189 = make_float2(q[18], q[19]);
            float2 qp_1_12 = _f2_189;
            sq[0] = fma_f32x2_rn_noftz(v_0_12, v_0_12, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_12, qp_1_12, dot[0]);
            float2 _f2_190 = make_float2(sw_23_f32[4], sw_23_f32[5]);
            float2 v_2_12 = _f2_190;
            float2 _f2_191 = make_float2(q[20], q[21]);
            float2 qp_3_12 = _f2_191;
            sq[0] = fma_f32x2_rn_noftz(v_2_12, v_2_12, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_12, qp_3_12, dot[0]);
            float2 _f2_192 = make_float2(sw_23_f32[6], sw_23_f32[7]);
            float2 v_4_12 = _f2_192;
            float2 _f2_193 = make_float2(q[22], q[23]);
            float2 qp_5_12 = _f2_193;
            sq[0] = fma_f32x2_rn_noftz(v_4_12, v_4_12, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_12, qp_5_12, dot[0]);
        }
        unsigned int sw_24[4];
        sw_24[0] = words[14 + woff_22];
        sw_24[1] = words[14 + woff_22 + 1];
        sw_24[2] = words[14 + woff_22 + 2];
        sw_24[3] = words[14 + woff_22 + 3];
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
            fsrc[44] = sw_24_f32[0];
            fsrc[45] = sw_24_f32[1];
            fsrc[46] = sw_24_f32[2];
            fsrc[47] = sw_24_f32[3];
            fsrc[48] = sw_24_f32[4];
            fsrc[49] = sw_24_f32[5];
            fsrc[50] = sw_24_f32[6];
            fsrc[51] = sw_24_f32[7];
        }
        {
            float2 _f2_200 = make_float2(sw_24_f32[0], sw_24_f32[1]);
            float2 v_15 = _f2_200;
            float2 _f2_201 = make_float2(q[16], q[17]);
            float2 qp_16 = _f2_201;
            sq[1] = fma_f32x2_rn_noftz(v_15, v_15, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_15, qp_16, dot[1]);
            float2 _f2_202 = make_float2(sw_24_f32[2], sw_24_f32[3]);
            float2 v_0_13 = _f2_202;
            float2 _f2_203 = make_float2(q[18], q[19]);
            float2 qp_1_13 = _f2_203;
            sq[1] = fma_f32x2_rn_noftz(v_0_13, v_0_13, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_13, qp_1_13, dot[1]);
            float2 _f2_204 = make_float2(sw_24_f32[4], sw_24_f32[5]);
            float2 v_2_13 = _f2_204;
            float2 _f2_205 = make_float2(q[20], q[21]);
            float2 qp_3_13 = _f2_205;
            sq[1] = fma_f32x2_rn_noftz(v_2_13, v_2_13, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_13, qp_3_13, dot[1]);
            float2 _f2_206 = make_float2(sw_24_f32[6], sw_24_f32[7]);
            float2 v_4_13 = _f2_206;
            float2 _f2_207 = make_float2(q[22], q[23]);
            float2 qp_5_13 = _f2_207;
            sq[1] = fma_f32x2_rn_noftz(v_4_13, v_4_13, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_13, qp_5_13, dot[1]);
        }
        unsigned int sw_25[4];
        sw_25[0] = words[28 + woff_22];
        sw_25[1] = words[28 + woff_22 + 1];
        sw_25[2] = words[28 + woff_22 + 2];
        sw_25[3] = words[28 + woff_22 + 3];
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
            fsrc[72] = sw_25_f32[0];
            fsrc[73] = sw_25_f32[1];
            fsrc[74] = sw_25_f32[2];
            fsrc[75] = sw_25_f32[3];
            fsrc[76] = sw_25_f32[4];
            fsrc[77] = sw_25_f32[5];
            fsrc[78] = sw_25_f32[6];
            fsrc[79] = sw_25_f32[7];
        }
        {
            float2 _f2_214 = make_float2(sw_25_f32[0], sw_25_f32[1]);
            float2 v_16 = _f2_214;
            float2 _f2_215 = make_float2(q[16], q[17]);
            float2 qp_17 = _f2_215;
            sq[2] = fma_f32x2_rn_noftz(v_16, v_16, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_16, qp_17, dot[2]);
            float2 _f2_216 = make_float2(sw_25_f32[2], sw_25_f32[3]);
            float2 v_0_14 = _f2_216;
            float2 _f2_217 = make_float2(q[18], q[19]);
            float2 qp_1_14 = _f2_217;
            sq[2] = fma_f32x2_rn_noftz(v_0_14, v_0_14, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_14, qp_1_14, dot[2]);
            float2 _f2_218 = make_float2(sw_25_f32[4], sw_25_f32[5]);
            float2 v_2_14 = _f2_218;
            float2 _f2_219 = make_float2(q[20], q[21]);
            float2 qp_3_14 = _f2_219;
            sq[2] = fma_f32x2_rn_noftz(v_2_14, v_2_14, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_14, qp_3_14, dot[2]);
            float2 _f2_220 = make_float2(sw_25_f32[6], sw_25_f32[7]);
            float2 v_4_14 = _f2_220;
            float2 _f2_221 = make_float2(q[22], q[23]);
            float2 qp_5_14 = _f2_221;
            sq[2] = fma_f32x2_rn_noftz(v_4_14, v_4_14, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_14, qp_5_14, dot[2]);
        }
        unsigned int sw_26[4];
        sw_26[0] = words[42 + woff_22];
        sw_26[1] = words[42 + woff_22 + 1];
        sw_26[2] = words[42 + woff_22 + 2];
        sw_26[3] = words[42 + woff_22 + 3];
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
            float2 _f2_228 = make_float2(sw_26_f32[0], sw_26_f32[1]);
            float2 v_17 = _f2_228;
            float2 _f2_229 = make_float2(q[16], q[17]);
            float2 qp_18 = _f2_229;
            sq[3] = fma_f32x2_rn_noftz(v_17, v_17, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_17, qp_18, dot[3]);
            float2 _f2_230 = make_float2(sw_26_f32[2], sw_26_f32[3]);
            float2 v_0_15 = _f2_230;
            float2 _f2_231 = make_float2(q[18], q[19]);
            float2 qp_1_15 = _f2_231;
            sq[3] = fma_f32x2_rn_noftz(v_0_15, v_0_15, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_15, qp_1_15, dot[3]);
            float2 _f2_232 = make_float2(sw_26_f32[4], sw_26_f32[5]);
            float2 v_2_15 = _f2_232;
            float2 _f2_233 = make_float2(q[20], q[21]);
            float2 qp_3_15 = _f2_233;
            sq[3] = fma_f32x2_rn_noftz(v_2_15, v_2_15, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_15, qp_3_15, dot[3]);
            float2 _f2_234 = make_float2(sw_26_f32[6], sw_26_f32[7]);
            float2 v_4_15 = _f2_234;
            float2 _f2_235 = make_float2(q[22], q[23]);
            float2 qp_5_15 = _f2_235;
            sq[3] = fma_f32x2_rn_noftz(v_4_15, v_4_15, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_15, qp_5_15, dot[3]);
        }
        unsigned int sw_27[4];
        sw_27[0] = words[56 + woff_22];
        sw_27[1] = words[56 + woff_22 + 1];
        sw_27[2] = words[56 + woff_22 + 2];
        sw_27[3] = words[56 + woff_22 + 3];
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
            float2 _f2_242 = make_float2(sw_27_f32[0], sw_27_f32[1]);
            float2 v_18 = _f2_242;
            float2 _f2_243 = make_float2(q[16], q[17]);
            float2 qp_19 = _f2_243;
            sq[4] = fma_f32x2_rn_noftz(v_18, v_18, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_18, qp_19, dot[4]);
            float2 _f2_244 = make_float2(sw_27_f32[2], sw_27_f32[3]);
            float2 v_0_16 = _f2_244;
            float2 _f2_245 = make_float2(q[18], q[19]);
            float2 qp_1_16 = _f2_245;
            sq[4] = fma_f32x2_rn_noftz(v_0_16, v_0_16, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_16, qp_1_16, dot[4]);
            float2 _f2_246 = make_float2(sw_27_f32[4], sw_27_f32[5]);
            float2 v_2_16 = _f2_246;
            float2 _f2_247 = make_float2(q[20], q[21]);
            float2 qp_3_16 = _f2_247;
            sq[4] = fma_f32x2_rn_noftz(v_2_16, v_2_16, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_16, qp_3_16, dot[4]);
            float2 _f2_248 = make_float2(sw_27_f32[6], sw_27_f32[7]);
            float2 v_4_16 = _f2_248;
            float2 _f2_249 = make_float2(q[22], q[23]);
            float2 qp_5_16 = _f2_249;
            sq[4] = fma_f32x2_rn_noftz(v_4_16, v_4_16, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_16, qp_5_16, dot[4]);
        }
        unsigned int sw_28[4];
        sw_28[0] = words[70 + woff_22];
        sw_28[1] = words[70 + woff_22 + 1];
        sw_28[2] = words[70 + woff_22 + 2];
        sw_28[3] = words[70 + woff_22 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_28[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_22]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_28[0] = __as_u32(mixed_3);
            words[70 + woff_22] = sw_28[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_28[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_22 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_28[1] = __as_u32(mixed_2_2);
            words[70 + woff_22 + 1] = sw_28[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_28[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_22 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_28[2] = __as_u32(mixed_5_2);
            words[70 + woff_22 + 2] = sw_28[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_28[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_22 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_28[3] = __as_u32(mixed_8_2);
            words[70 + woff_22 + 3] = sw_28[3];
            {
                int4 _iv4 = make_int4(sw_28[0 + 0], sw_28[0 + 1], sw_28[0 + 2], sw_28[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_21)) + 0) = _iv4;
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
            float2 _f2_256 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_19 = _f2_256;
            float2 _f2_257 = make_float2(q[16], q[17]);
            float2 qp_20 = _f2_257;
            sq[5] = fma_f32x2_rn_noftz(v_19, v_19, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_19, qp_20, dot[5]);
            float2 _f2_258 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_0_17 = _f2_258;
            float2 _f2_259 = make_float2(q[18], q[19]);
            float2 qp_1_17 = _f2_259;
            sq[5] = fma_f32x2_rn_noftz(v_0_17, v_0_17, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_17, qp_1_17, dot[5]);
            float2 _f2_260 = make_float2(sw_28_f32[4], sw_28_f32[5]);
            float2 v_2_17 = _f2_260;
            float2 _f2_261 = make_float2(q[20], q[21]);
            float2 qp_3_17 = _f2_261;
            sq[5] = fma_f32x2_rn_noftz(v_2_17, v_2_17, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_17, qp_3_17, dot[5]);
            float2 _f2_262 = make_float2(sw_28_f32[6], sw_28_f32[7]);
            float2 v_4_17 = _f2_262;
            float2 _f2_263 = make_float2(q[22], q[23]);
            float2 qp_5_17 = _f2_263;
            sq[5] = fma_f32x2_rn_noftz(v_4_17, v_4_17, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_17, qp_5_17, dot[5]);
        }
        int base_29 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_30 = 12;
        unsigned int sw_31[4];
        sw_31[0] = words[woff_30];
        sw_31[1] = words[woff_30 + 1];
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
            fsrc[24] = sw_31_f32[0];
            fsrc[25] = sw_31_f32[1];
            fsrc[26] = sw_31_f32[2];
            fsrc[27] = sw_31_f32[3];
        }
        {
            float2 _f2_264 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_20 = _f2_264;
            sq[0] = fma_f32x2_rn_noftz(v_20, v_20, sq[0]);
            float2 _f2_265 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_0_18 = _f2_265;
            sq[0] = fma_f32x2_rn_noftz(v_0_18, v_0_18, sq[0]);
            float2 _f2_266 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_1_1 = _f2_266;
            float2 _f2_267 = make_float2(q[24], q[25]);
            float2 qp_21 = _f2_267;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_21, dot[0]);
            float2 _f2_268 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_2_18 = _f2_268;
            float2 _f2_269 = make_float2(q[26], q[27]);
            float2 qp_3_18 = _f2_269;
            dot[0] = fma_f32x2_rn_noftz(v_2_18, qp_3_18, dot[0]);
        }
        unsigned int sw_32[4];
        sw_32[0] = words[14 + woff_30];
        sw_32[1] = words[14 + woff_30 + 1];
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
            fsrc[52] = sw_32_f32[0];
            fsrc[53] = sw_32_f32[1];
            fsrc[54] = sw_32_f32[2];
            fsrc[55] = sw_32_f32[3];
        }
        {
            float2 _f2_278 = make_float2(sw_32_f32[0], sw_32_f32[1]);
            float2 v_21 = _f2_278;
            sq[1] = fma_f32x2_rn_noftz(v_21, v_21, sq[1]);
            float2 _f2_279 = make_float2(sw_32_f32[2], sw_32_f32[3]);
            float2 v_0_19 = _f2_279;
            sq[1] = fma_f32x2_rn_noftz(v_0_19, v_0_19, sq[1]);
            float2 _f2_280 = make_float2(sw_32_f32[0], sw_32_f32[1]);
            float2 v_1_2 = _f2_280;
            float2 _f2_281 = make_float2(q[24], q[25]);
            float2 qp_22 = _f2_281;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_22, dot[1]);
            float2 _f2_282 = make_float2(sw_32_f32[2], sw_32_f32[3]);
            float2 v_2_19 = _f2_282;
            float2 _f2_283 = make_float2(q[26], q[27]);
            float2 qp_3_19 = _f2_283;
            dot[1] = fma_f32x2_rn_noftz(v_2_19, qp_3_19, dot[1]);
        }
        unsigned int sw_33[4];
        sw_33[0] = words[28 + woff_30];
        sw_33[1] = words[28 + woff_30 + 1];
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
            fsrc[80] = sw_33_f32[0];
            fsrc[81] = sw_33_f32[1];
            fsrc[82] = sw_33_f32[2];
            fsrc[83] = sw_33_f32[3];
        }
        {
            float2 _f2_292 = make_float2(sw_33_f32[0], sw_33_f32[1]);
            float2 v_22 = _f2_292;
            sq[2] = fma_f32x2_rn_noftz(v_22, v_22, sq[2]);
            float2 _f2_293 = make_float2(sw_33_f32[2], sw_33_f32[3]);
            float2 v_0_20 = _f2_293;
            sq[2] = fma_f32x2_rn_noftz(v_0_20, v_0_20, sq[2]);
            float2 _f2_294 = make_float2(sw_33_f32[0], sw_33_f32[1]);
            float2 v_1_3 = _f2_294;
            float2 _f2_295 = make_float2(q[24], q[25]);
            float2 qp_23 = _f2_295;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_23, dot[2]);
            float2 _f2_296 = make_float2(sw_33_f32[2], sw_33_f32[3]);
            float2 v_2_20 = _f2_296;
            float2 _f2_297 = make_float2(q[26], q[27]);
            float2 qp_3_20 = _f2_297;
            dot[2] = fma_f32x2_rn_noftz(v_2_20, qp_3_20, dot[2]);
        }
        unsigned int sw_34[4];
        sw_34[0] = words[42 + woff_30];
        sw_34[1] = words[42 + woff_30 + 1];
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
            float2 _f2_306 = make_float2(sw_34_f32[0], sw_34_f32[1]);
            float2 v_23 = _f2_306;
            sq[3] = fma_f32x2_rn_noftz(v_23, v_23, sq[3]);
            float2 _f2_307 = make_float2(sw_34_f32[2], sw_34_f32[3]);
            float2 v_0_21 = _f2_307;
            sq[3] = fma_f32x2_rn_noftz(v_0_21, v_0_21, sq[3]);
            float2 _f2_308 = make_float2(sw_34_f32[0], sw_34_f32[1]);
            float2 v_1_4 = _f2_308;
            float2 _f2_309 = make_float2(q[24], q[25]);
            float2 qp_24 = _f2_309;
            dot[3] = fma_f32x2_rn_noftz(v_1_4, qp_24, dot[3]);
            float2 _f2_310 = make_float2(sw_34_f32[2], sw_34_f32[3]);
            float2 v_2_21 = _f2_310;
            float2 _f2_311 = make_float2(q[26], q[27]);
            float2 qp_3_21 = _f2_311;
            dot[3] = fma_f32x2_rn_noftz(v_2_21, qp_3_21, dot[3]);
        }
        unsigned int sw_35[4];
        sw_35[0] = words[56 + woff_30];
        sw_35[1] = words[56 + woff_30 + 1];
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
            float2 _f2_320 = make_float2(sw_35_f32[0], sw_35_f32[1]);
            float2 v_24 = _f2_320;
            sq[4] = fma_f32x2_rn_noftz(v_24, v_24, sq[4]);
            float2 _f2_321 = make_float2(sw_35_f32[2], sw_35_f32[3]);
            float2 v_0_22 = _f2_321;
            sq[4] = fma_f32x2_rn_noftz(v_0_22, v_0_22, sq[4]);
            float2 _f2_322 = make_float2(sw_35_f32[0], sw_35_f32[1]);
            float2 v_1_5 = _f2_322;
            float2 _f2_323 = make_float2(q[24], q[25]);
            float2 qp_25 = _f2_323;
            dot[4] = fma_f32x2_rn_noftz(v_1_5, qp_25, dot[4]);
            float2 _f2_324 = make_float2(sw_35_f32[2], sw_35_f32[3]);
            float2 v_2_22 = _f2_324;
            float2 _f2_325 = make_float2(q[26], q[27]);
            float2 qp_3_22 = _f2_325;
            dot[4] = fma_f32x2_rn_noftz(v_2_22, qp_3_22, dot[4]);
        }
        unsigned int sw_36[4];
        sw_36[0] = words[70 + woff_30];
        sw_36[1] = words[70 + woff_30 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_36[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_30]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_36[0] = __as_u32(mixed_4);
            words[70 + woff_30] = sw_36[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_36[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_30 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_36[1] = __as_u32(mixed_2_3);
            words[70 + woff_30 + 1] = sw_36[1];
            {
                int2 _iv2 = make_int2(sw_36[0 + 0], sw_36[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_29)) + 0) = _iv2;
            }
        }
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
            float2 _f2_334 = make_float2(sw_36_f32[0], sw_36_f32[1]);
            float2 v_25 = _f2_334;
            sq[5] = fma_f32x2_rn_noftz(v_25, v_25, sq[5]);
            float2 _f2_335 = make_float2(sw_36_f32[2], sw_36_f32[3]);
            float2 v_0_23 = _f2_335;
            sq[5] = fma_f32x2_rn_noftz(v_0_23, v_0_23, sq[5]);
            float2 _f2_336 = make_float2(sw_36_f32[0], sw_36_f32[1]);
            float2 v_1_6 = _f2_336;
            float2 _f2_337 = make_float2(q[24], q[25]);
            float2 qp_26 = _f2_337;
            dot[5] = fma_f32x2_rn_noftz(v_1_6, qp_26, dot[5]);
            float2 _f2_338 = make_float2(sw_36_f32[2], sw_36_f32[3]);
            float2 v_2_23 = _f2_338;
            float2 _f2_339 = make_float2(q[26], q[27]);
            float2 qp_3_23 = _f2_339;
            dot[5] = fma_f32x2_rn_noftz(v_2_23, qp_3_23, dot[5]);
        }
        float2 pairs[6];
        float2 _f2_348 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_348;
        float2 _f2_349 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_349;
        float2 _f2_350 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_350;
        float2 _f2_351 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_351;
        float2 _f2_352 = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[4] = _f2_352;
        float2 _f2_353 = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[5] = _f2_353;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_354 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_354;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_37 = 0;
        bits_37 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_37, 16);
        unsigned long long peerbits_38 = _shfl_xor_1;
        float2 _f2_355 = make_float2(0.0f, 0.0f);
        float2 peer_39 = _f2_355;
        peer_39 = reinterpret_cast<float2*>(&peerbits_38)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_39);
        unsigned long long bits_40 = 0;
        bits_40 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_40, 16);
        unsigned long long peerbits_41 = _shfl_xor_2;
        float2 _f2_356 = make_float2(0.0f, 0.0f);
        float2 peer_42 = _f2_356;
        peer_42 = reinterpret_cast<float2*>(&peerbits_41)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_42);
        unsigned long long bits_43 = 0;
        bits_43 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_43, 16);
        unsigned long long peerbits_44 = _shfl_xor_3;
        float2 _f2_357 = make_float2(0.0f, 0.0f);
        float2 peer_45 = _f2_357;
        peer_45 = reinterpret_cast<float2*>(&peerbits_44)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_45);
        unsigned long long bits_46 = 0;
        bits_46 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_46, 16);
        unsigned long long peerbits_47 = _shfl_xor_4;
        float2 _f2_358 = make_float2(0.0f, 0.0f);
        float2 peer_48 = _f2_358;
        peer_48 = reinterpret_cast<float2*>(&peerbits_47)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_48);
        unsigned long long bits_49 = 0;
        bits_49 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_49, 16);
        unsigned long long peerbits_50 = _shfl_xor_5;
        float2 _f2_359 = make_float2(0.0f, 0.0f);
        float2 peer_51 = _f2_359;
        peer_51 = reinterpret_cast<float2*>(&peerbits_50)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_51);
        unsigned long long bits_52 = 0;
        bits_52 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_52, 8);
        unsigned long long peerbits_53 = _shfl_xor_6;
        float2 _f2_360 = make_float2(0.0f, 0.0f);
        float2 peer_54 = _f2_360;
        peer_54 = reinterpret_cast<float2*>(&peerbits_53)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_54);
        unsigned long long bits_55 = 0;
        bits_55 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_55, 8);
        unsigned long long peerbits_56 = _shfl_xor_7;
        float2 _f2_361 = make_float2(0.0f, 0.0f);
        float2 peer_57 = _f2_361;
        peer_57 = reinterpret_cast<float2*>(&peerbits_56)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_57);
        unsigned long long bits_58 = 0;
        bits_58 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_58, 8);
        unsigned long long peerbits_59 = _shfl_xor_8;
        float2 _f2_362 = make_float2(0.0f, 0.0f);
        float2 peer_60 = _f2_362;
        peer_60 = reinterpret_cast<float2*>(&peerbits_59)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_60);
        unsigned long long bits_61 = 0;
        bits_61 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_61, 8);
        unsigned long long peerbits_62 = _shfl_xor_9;
        float2 _f2_363 = make_float2(0.0f, 0.0f);
        float2 peer_63 = _f2_363;
        peer_63 = reinterpret_cast<float2*>(&peerbits_62)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_63);
        unsigned long long bits_64 = 0;
        bits_64 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_64, 8);
        unsigned long long peerbits_65 = _shfl_xor_10;
        float2 _f2_364 = make_float2(0.0f, 0.0f);
        float2 peer_66 = _f2_364;
        peer_66 = reinterpret_cast<float2*>(&peerbits_65)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_66);
        unsigned long long bits_67 = 0;
        bits_67 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_67, 8);
        unsigned long long peerbits_68 = _shfl_xor_11;
        float2 _f2_365 = make_float2(0.0f, 0.0f);
        float2 peer_69 = _f2_365;
        peer_69 = reinterpret_cast<float2*>(&peerbits_68)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_69);
        unsigned long long bits_70 = 0;
        bits_70 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_70, 4);
        unsigned long long peerbits_71 = _shfl_xor_12;
        float2 _f2_366 = make_float2(0.0f, 0.0f);
        float2 peer_72 = _f2_366;
        peer_72 = reinterpret_cast<float2*>(&peerbits_71)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_72);
        unsigned long long bits_73 = 0;
        bits_73 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_73, 4);
        unsigned long long peerbits_74 = _shfl_xor_13;
        float2 _f2_367 = make_float2(0.0f, 0.0f);
        float2 peer_75 = _f2_367;
        peer_75 = reinterpret_cast<float2*>(&peerbits_74)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_75);
        unsigned long long bits_76 = 0;
        bits_76 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_76, 4);
        unsigned long long peerbits_77 = _shfl_xor_14;
        float2 _f2_368 = make_float2(0.0f, 0.0f);
        float2 peer_78 = _f2_368;
        peer_78 = reinterpret_cast<float2*>(&peerbits_77)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_78);
        unsigned long long bits_79 = 0;
        bits_79 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_79, 4);
        unsigned long long peerbits_80 = _shfl_xor_15;
        float2 _f2_369 = make_float2(0.0f, 0.0f);
        float2 peer_81 = _f2_369;
        peer_81 = reinterpret_cast<float2*>(&peerbits_80)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_81);
        unsigned long long bits_82 = 0;
        bits_82 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_82, 4);
        unsigned long long peerbits_83 = _shfl_xor_16;
        float2 _f2_370 = make_float2(0.0f, 0.0f);
        float2 peer_84 = _f2_370;
        peer_84 = reinterpret_cast<float2*>(&peerbits_83)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_84);
        unsigned long long bits_85 = 0;
        bits_85 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_85, 4);
        unsigned long long peerbits_86 = _shfl_xor_17;
        float2 _f2_371 = make_float2(0.0f, 0.0f);
        float2 peer_87 = _f2_371;
        peer_87 = reinterpret_cast<float2*>(&peerbits_86)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_87);
        unsigned long long bits_88 = 0;
        bits_88 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_88, 2);
        unsigned long long peerbits_89 = _shfl_xor_18;
        float2 _f2_372 = make_float2(0.0f, 0.0f);
        float2 peer_90 = _f2_372;
        peer_90 = reinterpret_cast<float2*>(&peerbits_89)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_90);
        unsigned long long bits_91 = 0;
        bits_91 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_91, 2);
        unsigned long long peerbits_92 = _shfl_xor_19;
        float2 _f2_373 = make_float2(0.0f, 0.0f);
        float2 peer_93 = _f2_373;
        peer_93 = reinterpret_cast<float2*>(&peerbits_92)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_93);
        unsigned long long bits_94 = 0;
        bits_94 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, bits_94, 2);
        unsigned long long peerbits_95 = _shfl_xor_20;
        float2 _f2_374 = make_float2(0.0f, 0.0f);
        float2 peer_96 = _f2_374;
        peer_96 = reinterpret_cast<float2*>(&peerbits_95)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_96);
        unsigned long long bits_97 = 0;
        bits_97 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, bits_97, 2);
        unsigned long long peerbits_98 = _shfl_xor_21;
        float2 _f2_375 = make_float2(0.0f, 0.0f);
        float2 peer_99 = _f2_375;
        peer_99 = reinterpret_cast<float2*>(&peerbits_98)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_99);
        unsigned long long bits_100 = 0;
        bits_100 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, bits_100, 2);
        unsigned long long peerbits_101 = _shfl_xor_22;
        float2 _f2_376 = make_float2(0.0f, 0.0f);
        float2 peer_102 = _f2_376;
        peer_102 = reinterpret_cast<float2*>(&peerbits_101)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_102);
        unsigned long long bits_103 = 0;
        bits_103 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, bits_103, 2);
        unsigned long long peerbits_104 = _shfl_xor_23;
        float2 _f2_377 = make_float2(0.0f, 0.0f);
        float2 peer_105 = _f2_377;
        peer_105 = reinterpret_cast<float2*>(&peerbits_104)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_105);
        unsigned long long bits_106 = 0;
        bits_106 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, bits_106, 1);
        unsigned long long peerbits_107 = _shfl_xor_24;
        float2 _f2_378 = make_float2(0.0f, 0.0f);
        float2 peer_108 = _f2_378;
        peer_108 = reinterpret_cast<float2*>(&peerbits_107)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_108);
        unsigned long long bits_109 = 0;
        bits_109 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, bits_109, 1);
        unsigned long long peerbits_110 = _shfl_xor_25;
        float2 _f2_379 = make_float2(0.0f, 0.0f);
        float2 peer_111 = _f2_379;
        peer_111 = reinterpret_cast<float2*>(&peerbits_110)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_111);
        unsigned long long bits_112 = 0;
        bits_112 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, bits_112, 1);
        unsigned long long peerbits_113 = _shfl_xor_26;
        float2 _f2_380 = make_float2(0.0f, 0.0f);
        float2 peer_114 = _f2_380;
        peer_114 = reinterpret_cast<float2*>(&peerbits_113)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_114);
        unsigned long long bits_115 = 0;
        bits_115 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, bits_115, 1);
        unsigned long long peerbits_116 = _shfl_xor_27;
        float2 _f2_381 = make_float2(0.0f, 0.0f);
        float2 peer_117 = _f2_381;
        peer_117 = reinterpret_cast<float2*>(&peerbits_116)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_117);
        unsigned long long bits_118 = 0;
        bits_118 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, bits_118, 1);
        unsigned long long peerbits_119 = _shfl_xor_28;
        float2 _f2_382 = make_float2(0.0f, 0.0f);
        float2 peer_120 = _f2_382;
        peer_120 = reinterpret_cast<float2*>(&peerbits_119)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_120);
        unsigned long long bits_121 = 0;
        bits_121 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, bits_121, 1);
        unsigned long long peerbits_122 = _shfl_xor_29;
        float2 _f2_383 = make_float2(0.0f, 0.0f);
        float2 peer_123 = _f2_383;
        peer_123 = reinterpret_cast<float2*>(&peerbits_122)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_123);
        if (lane == 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(stats_addr + (unsigned int)(warp_0 * 6 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_0), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(stats_addr + (unsigned int)((warp_0 * 6 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_1), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_2), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_3), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_4) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_4), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_5;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_5) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_5), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_6) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_6), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_7;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_7) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_7), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_8) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_8), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_9) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_9), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_10) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_10), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_11) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_11), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_12;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_12) : "r"(stats_addr + (unsigned int)(warp_0 * 6 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_12), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_13;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_13) : "r"(stats_addr + (unsigned int)((warp_0 * 6 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_13), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_14;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_14) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_14), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_15;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_15) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_15), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_16) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_16), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_17;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_17) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_17), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_18) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_18), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_19;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_19) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_19), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_20) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_20), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_21;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_21) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_21), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_22) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_22), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_23;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_23) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_23), "f"(pairs[5].y) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 6) {
            total_sq = stats[(stat_w * 6 + stat_n) * 2];
            total_dot = stats[(stat_w * 6 + stat_n) * 2 + 1];
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
        if (stat_n < 6 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[6];
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
        if (stat_n < 2) {
            total_sq2 = stats[(stat_w * 6 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 6 + 4 + stat_n) * 2 + 1];
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
        if (stat_n < 2 && stat_w == 0) {
            float _rsqrt_1 = rsqrtf(total_sq2 / 7168.0f + eps);
            float sigma2 = _rsqrt_1;
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_4;
        float _shfl_5 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_5;
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
        float weights[6];
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
        float2 _f2_384 = make_float2(correction, correction);
        float2 corr = _f2_384;
        const int woff_124 = 0;
        int base_125 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_385 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_385;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_386 = make_float2(acc[2], acc[3]);
        float2 previous_126 = _f2_386;
        a_5[1] = mul_f32x2_noftz(previous_126, corr);
        float2 _f2_387 = make_float2(acc[4], acc[5]);
        float2 previous_127 = _f2_387;
        a_5[2] = mul_f32x2_noftz(previous_127, corr);
        float2 _f2_388 = make_float2(acc[6], acc[7]);
        float2 previous_128 = _f2_388;
        a_5[3] = mul_f32x2_noftz(previous_128, corr);
        float2 _f2_389 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_389;
        {
            float2 _f2_390 = make_float2(fsrc[0], fsrc[1]);
            float2 v_26 = _f2_390;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_26, a_5[0]);
            float2 _f2_391 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_24 = _f2_391;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_24, a_5[1]);
            float2 _f2_392 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_7 = _f2_392;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_7, a_5[2]);
            float2 _f2_393 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_24 = _f2_393;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_24, a_5[3]);
        }
        float2 _f2_398 = make_float2(weights[1], weights[1]);
        float2 weight_129 = _f2_398;
        {
            float2 _f2_399 = make_float2(fsrc[28], fsrc[29]);
            float2 v_27 = _f2_399;
            a_5[0] = fma_f32x2_rn_noftz(weight_129, v_27, a_5[0]);
            float2 _f2_400 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_25 = _f2_400;
            a_5[1] = fma_f32x2_rn_noftz(weight_129, v_0_25, a_5[1]);
            float2 _f2_401 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_8 = _f2_401;
            a_5[2] = fma_f32x2_rn_noftz(weight_129, v_1_8, a_5[2]);
            float2 _f2_402 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_25 = _f2_402;
            a_5[3] = fma_f32x2_rn_noftz(weight_129, v_2_25, a_5[3]);
        }
        float2 _f2_407 = make_float2(weights[2], weights[2]);
        float2 weight_130 = _f2_407;
        {
            float2 _f2_408 = make_float2(fsrc[56], fsrc[57]);
            float2 v_28 = _f2_408;
            a_5[0] = fma_f32x2_rn_noftz(weight_130, v_28, a_5[0]);
            float2 _f2_409 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_26 = _f2_409;
            a_5[1] = fma_f32x2_rn_noftz(weight_130, v_0_26, a_5[1]);
            float2 _f2_410 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_9 = _f2_410;
            a_5[2] = fma_f32x2_rn_noftz(weight_130, v_1_9, a_5[2]);
            float2 _f2_411 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_26 = _f2_411;
            a_5[3] = fma_f32x2_rn_noftz(weight_130, v_2_26, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_131 = 4;
        int base_132 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_133[4];
        float2 _f2_416 = make_float2(acc[8], acc[9]);
        float2 previous_134 = _f2_416;
        a_133[0] = mul_f32x2_noftz(previous_134, corr);
        float2 _f2_417 = make_float2(acc[10], acc[11]);
        float2 previous_135 = _f2_417;
        a_133[1] = mul_f32x2_noftz(previous_135, corr);
        float2 _f2_418 = make_float2(acc[12], acc[13]);
        float2 previous_136 = _f2_418;
        a_133[2] = mul_f32x2_noftz(previous_136, corr);
        float2 _f2_419 = make_float2(acc[14], acc[15]);
        float2 previous_137 = _f2_419;
        a_133[3] = mul_f32x2_noftz(previous_137, corr);
        float2 _f2_420 = make_float2(weights[0], weights[0]);
        float2 weight_138 = _f2_420;
        {
            float2 _f2_421 = make_float2(fsrc[8], fsrc[9]);
            float2 v_29 = _f2_421;
            a_133[0] = fma_f32x2_rn_noftz(weight_138, v_29, a_133[0]);
            float2 _f2_422 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_27 = _f2_422;
            a_133[1] = fma_f32x2_rn_noftz(weight_138, v_0_27, a_133[1]);
            float2 _f2_423 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_10 = _f2_423;
            a_133[2] = fma_f32x2_rn_noftz(weight_138, v_1_10, a_133[2]);
            float2 _f2_424 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_27 = _f2_424;
            a_133[3] = fma_f32x2_rn_noftz(weight_138, v_2_27, a_133[3]);
        }
        float2 _f2_429 = make_float2(weights[1], weights[1]);
        float2 weight_139 = _f2_429;
        {
            float2 _f2_430 = make_float2(fsrc[36], fsrc[37]);
            float2 v_30 = _f2_430;
            a_133[0] = fma_f32x2_rn_noftz(weight_139, v_30, a_133[0]);
            float2 _f2_431 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_28 = _f2_431;
            a_133[1] = fma_f32x2_rn_noftz(weight_139, v_0_28, a_133[1]);
            float2 _f2_432 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_11 = _f2_432;
            a_133[2] = fma_f32x2_rn_noftz(weight_139, v_1_11, a_133[2]);
            float2 _f2_433 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_28 = _f2_433;
            a_133[3] = fma_f32x2_rn_noftz(weight_139, v_2_28, a_133[3]);
        }
        float2 _f2_438 = make_float2(weights[2], weights[2]);
        float2 weight_140 = _f2_438;
        {
            float2 _f2_439 = make_float2(fsrc[64], fsrc[65]);
            float2 v_31 = _f2_439;
            a_133[0] = fma_f32x2_rn_noftz(weight_140, v_31, a_133[0]);
            float2 _f2_440 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_29 = _f2_440;
            a_133[1] = fma_f32x2_rn_noftz(weight_140, v_0_29, a_133[1]);
            float2 _f2_441 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_12 = _f2_441;
            a_133[2] = fma_f32x2_rn_noftz(weight_140, v_1_12, a_133[2]);
            float2 _f2_442 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_29 = _f2_442;
            a_133[3] = fma_f32x2_rn_noftz(weight_140, v_2_29, a_133[3]);
        }
        acc[8] = a_133[0].x;
        acc[9] = a_133[0].y;
        acc[10] = a_133[1].x;
        acc[11] = a_133[1].y;
        acc[12] = a_133[2].x;
        acc[13] = a_133[2].y;
        acc[14] = a_133[3].x;
        acc[15] = a_133[3].y;
        const int woff_141 = 8;
        int base_142 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_143[4];
        float2 _f2_447 = make_float2(acc[16], acc[17]);
        float2 previous_144 = _f2_447;
        a_143[0] = mul_f32x2_noftz(previous_144, corr);
        float2 _f2_448 = make_float2(acc[18], acc[19]);
        float2 previous_145 = _f2_448;
        a_143[1] = mul_f32x2_noftz(previous_145, corr);
        float2 _f2_449 = make_float2(acc[20], acc[21]);
        float2 previous_146 = _f2_449;
        a_143[2] = mul_f32x2_noftz(previous_146, corr);
        float2 _f2_450 = make_float2(acc[22], acc[23]);
        float2 previous_147 = _f2_450;
        a_143[3] = mul_f32x2_noftz(previous_147, corr);
        float2 _f2_451 = make_float2(weights[0], weights[0]);
        float2 weight_148 = _f2_451;
        {
            float2 _f2_452 = make_float2(fsrc[16], fsrc[17]);
            float2 v_32 = _f2_452;
            a_143[0] = fma_f32x2_rn_noftz(weight_148, v_32, a_143[0]);
            float2 _f2_453 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_30 = _f2_453;
            a_143[1] = fma_f32x2_rn_noftz(weight_148, v_0_30, a_143[1]);
            float2 _f2_454 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_13 = _f2_454;
            a_143[2] = fma_f32x2_rn_noftz(weight_148, v_1_13, a_143[2]);
            float2 _f2_455 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_30 = _f2_455;
            a_143[3] = fma_f32x2_rn_noftz(weight_148, v_2_30, a_143[3]);
        }
        float2 _f2_460 = make_float2(weights[1], weights[1]);
        float2 weight_149 = _f2_460;
        {
            float2 _f2_461 = make_float2(fsrc[44], fsrc[45]);
            float2 v_33 = _f2_461;
            a_143[0] = fma_f32x2_rn_noftz(weight_149, v_33, a_143[0]);
            float2 _f2_462 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_31 = _f2_462;
            a_143[1] = fma_f32x2_rn_noftz(weight_149, v_0_31, a_143[1]);
            float2 _f2_463 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_14 = _f2_463;
            a_143[2] = fma_f32x2_rn_noftz(weight_149, v_1_14, a_143[2]);
            float2 _f2_464 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_31 = _f2_464;
            a_143[3] = fma_f32x2_rn_noftz(weight_149, v_2_31, a_143[3]);
        }
        float2 _f2_469 = make_float2(weights[2], weights[2]);
        float2 weight_150 = _f2_469;
        {
            float2 _f2_470 = make_float2(fsrc[72], fsrc[73]);
            float2 v_34 = _f2_470;
            a_143[0] = fma_f32x2_rn_noftz(weight_150, v_34, a_143[0]);
            float2 _f2_471 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_32 = _f2_471;
            a_143[1] = fma_f32x2_rn_noftz(weight_150, v_0_32, a_143[1]);
            float2 _f2_472 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_15 = _f2_472;
            a_143[2] = fma_f32x2_rn_noftz(weight_150, v_1_15, a_143[2]);
            float2 _f2_473 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_32 = _f2_473;
            a_143[3] = fma_f32x2_rn_noftz(weight_150, v_2_32, a_143[3]);
        }
        acc[16] = a_143[0].x;
        acc[17] = a_143[0].y;
        acc[18] = a_143[1].x;
        acc[19] = a_143[1].y;
        acc[20] = a_143[2].x;
        acc[21] = a_143[2].y;
        acc[22] = a_143[3].x;
        acc[23] = a_143[3].y;
        const int woff_151 = 12;
        int base_152 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_153[4];
        float2 _f2_478 = make_float2(acc[24], acc[25]);
        float2 previous_154 = _f2_478;
        a_153[0] = mul_f32x2_noftz(previous_154, corr);
        float2 _f2_479 = make_float2(acc[26], acc[27]);
        float2 previous_155 = _f2_479;
        a_153[1] = mul_f32x2_noftz(previous_155, corr);
        float2 _f2_480 = make_float2(weights[0], weights[0]);
        float2 weight_156 = _f2_480;
        {
            float2 _f2_481 = make_float2(fsrc[24], fsrc[25]);
            float2 v_35 = _f2_481;
            a_153[0] = fma_f32x2_rn_noftz(weight_156, v_35, a_153[0]);
            float2 _f2_482 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_33 = _f2_482;
            a_153[1] = fma_f32x2_rn_noftz(weight_156, v_0_33, a_153[1]);
        }
        float2 _f2_485 = make_float2(weights[1], weights[1]);
        float2 weight_157 = _f2_485;
        {
            float2 _f2_486 = make_float2(fsrc[52], fsrc[53]);
            float2 v_36 = _f2_486;
            a_153[0] = fma_f32x2_rn_noftz(weight_157, v_36, a_153[0]);
            float2 _f2_487 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_34 = _f2_487;
            a_153[1] = fma_f32x2_rn_noftz(weight_157, v_0_34, a_153[1]);
        }
        float2 _f2_490 = make_float2(weights[2], weights[2]);
        float2 weight_158 = _f2_490;
        {
            float2 _f2_491 = make_float2(fsrc[80], fsrc[81]);
            float2 v_37 = _f2_491;
            a_153[0] = fma_f32x2_rn_noftz(weight_158, v_37, a_153[0]);
            float2 _f2_492 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_35 = _f2_492;
            a_153[1] = fma_f32x2_rn_noftz(weight_158, v_0_35, a_153[1]);
        }
        acc[24] = a_153[0].x;
        acc[25] = a_153[0].y;
        acc[26] = a_153[1].x;
        acc[27] = a_153[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float max_chunk_159 = -3.4028234663852886e+38f;
        float _fmax_4 = fmaxf(max_chunk_159, logits[3]);
        max_chunk_159 = _fmax_4;
        float _fmax_5 = fmaxf(max_chunk_159, logits[4]);
        max_chunk_159 = _fmax_5;
        float _fmax_6 = fmaxf(max_chunk_159, logits[5]);
        max_chunk_159 = _fmax_6;
        float _fmax_7 = fmaxf(max_running, max_chunk_159);
        float max_new_160 = _fmax_7;
        float _exp2_4 = approx_exp2((max_running - max_new_160) * 1.4426950408889634f);
        float correction_161 = _exp2_4;
        float weights_162[6];
        float sum_weights_163 = 0.0f;
        float _exp2_5 = approx_exp2((logits[3] - max_new_160) * 1.4426950408889634f);
        weights_162[3] = _exp2_5;
        sum_weights_163 += weights_162[3];
        float _exp2_6 = approx_exp2((logits[4] - max_new_160) * 1.4426950408889634f);
        weights_162[4] = _exp2_6;
        sum_weights_163 += weights_162[4];
        float _exp2_7 = approx_exp2((logits[5] - max_new_160) * 1.4426950408889634f);
        weights_162[5] = _exp2_7;
        sum_weights_163 += weights_162[5];
        float2 _f2_495 = make_float2(correction_161, correction_161);
        float2 corr_164 = _f2_495;
        const int woff_165 = 0;
        int base_166 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_167[4];
        float2 _f2_496 = make_float2(acc[0], acc[1]);
        float2 previous_168 = _f2_496;
        a_167[0] = mul_f32x2_noftz(previous_168, corr_164);
        float2 _f2_497 = make_float2(acc[2], acc[3]);
        float2 previous_169 = _f2_497;
        a_167[1] = mul_f32x2_noftz(previous_169, corr_164);
        float2 _f2_498 = make_float2(acc[4], acc[5]);
        float2 previous_170 = _f2_498;
        a_167[2] = mul_f32x2_noftz(previous_170, corr_164);
        float2 _f2_499 = make_float2(acc[6], acc[7]);
        float2 previous_171 = _f2_499;
        a_167[3] = mul_f32x2_noftz(previous_171, corr_164);
        float2 _f2_500 = make_float2(weights_162[3], weights_162[3]);
        float2 weight_172 = _f2_500;
        {
            unsigned int sw2[4];
            sw2[0] = words[42 + woff_165];
            sw2[1] = words[42 + woff_165 + 1];
            sw2[2] = words[42 + woff_165 + 2];
            sw2[3] = words[42 + woff_165 + 3];
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
            float2 _f2_505 = make_float2(sw2_f32[0], sw2_f32[1]);
            float2 v_38 = _f2_505;
            a_167[0] = fma_f32x2_rn_noftz(weight_172, v_38, a_167[0]);
            float2 _f2_506 = make_float2(sw2_f32[2], sw2_f32[3]);
            float2 v_0_36 = _f2_506;
            a_167[1] = fma_f32x2_rn_noftz(weight_172, v_0_36, a_167[1]);
            float2 _f2_507 = make_float2(sw2_f32[4], sw2_f32[5]);
            float2 v_1_16 = _f2_507;
            a_167[2] = fma_f32x2_rn_noftz(weight_172, v_1_16, a_167[2]);
            float2 _f2_508 = make_float2(sw2_f32[6], sw2_f32[7]);
            float2 v_2_33 = _f2_508;
            a_167[3] = fma_f32x2_rn_noftz(weight_172, v_2_33, a_167[3]);
        }
        float2 _f2_509 = make_float2(weights_162[4], weights_162[4]);
        float2 weight_173 = _f2_509;
        {
            unsigned int sw2_1[4];
            sw2_1[0] = words[56 + woff_165];
            sw2_1[1] = words[56 + woff_165 + 1];
            sw2_1[2] = words[56 + woff_165 + 2];
            sw2_1[3] = words[56 + woff_165 + 3];
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
            float2 _f2_514 = make_float2(sw2_f32_1[0], sw2_f32_1[1]);
            float2 v_39 = _f2_514;
            a_167[0] = fma_f32x2_rn_noftz(weight_173, v_39, a_167[0]);
            float2 _f2_515 = make_float2(sw2_f32_1[2], sw2_f32_1[3]);
            float2 v_0_37 = _f2_515;
            a_167[1] = fma_f32x2_rn_noftz(weight_173, v_0_37, a_167[1]);
            float2 _f2_516 = make_float2(sw2_f32_1[4], sw2_f32_1[5]);
            float2 v_1_17 = _f2_516;
            a_167[2] = fma_f32x2_rn_noftz(weight_173, v_1_17, a_167[2]);
            float2 _f2_517 = make_float2(sw2_f32_1[6], sw2_f32_1[7]);
            float2 v_2_34 = _f2_517;
            a_167[3] = fma_f32x2_rn_noftz(weight_173, v_2_34, a_167[3]);
        }
        float2 _f2_518 = make_float2(weights_162[5], weights_162[5]);
        float2 weight_174 = _f2_518;
        {
            unsigned int sw2_2[4];
            sw2_2[0] = words[70 + woff_165];
            sw2_2[1] = words[70 + woff_165 + 1];
            sw2_2[2] = words[70 + woff_165 + 2];
            sw2_2[3] = words[70 + woff_165 + 3];
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
            float2 _f2_523 = make_float2(sw2_f32_2[0], sw2_f32_2[1]);
            float2 v_40 = _f2_523;
            a_167[0] = fma_f32x2_rn_noftz(weight_174, v_40, a_167[0]);
            float2 _f2_524 = make_float2(sw2_f32_2[2], sw2_f32_2[3]);
            float2 v_0_38 = _f2_524;
            a_167[1] = fma_f32x2_rn_noftz(weight_174, v_0_38, a_167[1]);
            float2 _f2_525 = make_float2(sw2_f32_2[4], sw2_f32_2[5]);
            float2 v_1_18 = _f2_525;
            a_167[2] = fma_f32x2_rn_noftz(weight_174, v_1_18, a_167[2]);
            float2 _f2_526 = make_float2(sw2_f32_2[6], sw2_f32_2[7]);
            float2 v_2_35 = _f2_526;
            a_167[3] = fma_f32x2_rn_noftz(weight_174, v_2_35, a_167[3]);
        }
        acc[0] = a_167[0].x;
        acc[1] = a_167[0].y;
        acc[2] = a_167[1].x;
        acc[3] = a_167[1].y;
        acc[4] = a_167[2].x;
        acc[5] = a_167[2].y;
        acc[6] = a_167[3].x;
        acc[7] = a_167[3].y;
        const int woff_175 = 4;
        int base_176 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_177[4];
        float2 _f2_527 = make_float2(acc[8], acc[9]);
        float2 previous_178 = _f2_527;
        a_177[0] = mul_f32x2_noftz(previous_178, corr_164);
        float2 _f2_528 = make_float2(acc[10], acc[11]);
        float2 previous_179 = _f2_528;
        a_177[1] = mul_f32x2_noftz(previous_179, corr_164);
        float2 _f2_529 = make_float2(acc[12], acc[13]);
        float2 previous_180 = _f2_529;
        a_177[2] = mul_f32x2_noftz(previous_180, corr_164);
        float2 _f2_530 = make_float2(acc[14], acc[15]);
        float2 previous_181 = _f2_530;
        a_177[3] = mul_f32x2_noftz(previous_181, corr_164);
        float2 _f2_531 = make_float2(weights_162[3], weights_162[3]);
        float2 weight_182 = _f2_531;
        {
            unsigned int sw2_3[4];
            sw2_3[0] = words[42 + woff_175];
            sw2_3[1] = words[42 + woff_175 + 1];
            sw2_3[2] = words[42 + woff_175 + 2];
            sw2_3[3] = words[42 + woff_175 + 3];
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
            float2 _f2_536 = make_float2(sw2_f32_3[0], sw2_f32_3[1]);
            float2 v_41 = _f2_536;
            a_177[0] = fma_f32x2_rn_noftz(weight_182, v_41, a_177[0]);
            float2 _f2_537 = make_float2(sw2_f32_3[2], sw2_f32_3[3]);
            float2 v_0_39 = _f2_537;
            a_177[1] = fma_f32x2_rn_noftz(weight_182, v_0_39, a_177[1]);
            float2 _f2_538 = make_float2(sw2_f32_3[4], sw2_f32_3[5]);
            float2 v_1_19 = _f2_538;
            a_177[2] = fma_f32x2_rn_noftz(weight_182, v_1_19, a_177[2]);
            float2 _f2_539 = make_float2(sw2_f32_3[6], sw2_f32_3[7]);
            float2 v_2_36 = _f2_539;
            a_177[3] = fma_f32x2_rn_noftz(weight_182, v_2_36, a_177[3]);
        }
        float2 _f2_540 = make_float2(weights_162[4], weights_162[4]);
        float2 weight_183 = _f2_540;
        {
            unsigned int sw2_4[4];
            sw2_4[0] = words[56 + woff_175];
            sw2_4[1] = words[56 + woff_175 + 1];
            sw2_4[2] = words[56 + woff_175 + 2];
            sw2_4[3] = words[56 + woff_175 + 3];
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
            float2 _f2_545 = make_float2(sw2_f32_4[0], sw2_f32_4[1]);
            float2 v_42 = _f2_545;
            a_177[0] = fma_f32x2_rn_noftz(weight_183, v_42, a_177[0]);
            float2 _f2_546 = make_float2(sw2_f32_4[2], sw2_f32_4[3]);
            float2 v_0_40 = _f2_546;
            a_177[1] = fma_f32x2_rn_noftz(weight_183, v_0_40, a_177[1]);
            float2 _f2_547 = make_float2(sw2_f32_4[4], sw2_f32_4[5]);
            float2 v_1_20 = _f2_547;
            a_177[2] = fma_f32x2_rn_noftz(weight_183, v_1_20, a_177[2]);
            float2 _f2_548 = make_float2(sw2_f32_4[6], sw2_f32_4[7]);
            float2 v_2_37 = _f2_548;
            a_177[3] = fma_f32x2_rn_noftz(weight_183, v_2_37, a_177[3]);
        }
        float2 _f2_549 = make_float2(weights_162[5], weights_162[5]);
        float2 weight_184 = _f2_549;
        {
            unsigned int sw2_5[4];
            sw2_5[0] = words[70 + woff_175];
            sw2_5[1] = words[70 + woff_175 + 1];
            sw2_5[2] = words[70 + woff_175 + 2];
            sw2_5[3] = words[70 + woff_175 + 3];
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
            float2 _f2_554 = make_float2(sw2_f32_5[0], sw2_f32_5[1]);
            float2 v_43 = _f2_554;
            a_177[0] = fma_f32x2_rn_noftz(weight_184, v_43, a_177[0]);
            float2 _f2_555 = make_float2(sw2_f32_5[2], sw2_f32_5[3]);
            float2 v_0_41 = _f2_555;
            a_177[1] = fma_f32x2_rn_noftz(weight_184, v_0_41, a_177[1]);
            float2 _f2_556 = make_float2(sw2_f32_5[4], sw2_f32_5[5]);
            float2 v_1_21 = _f2_556;
            a_177[2] = fma_f32x2_rn_noftz(weight_184, v_1_21, a_177[2]);
            float2 _f2_557 = make_float2(sw2_f32_5[6], sw2_f32_5[7]);
            float2 v_2_38 = _f2_557;
            a_177[3] = fma_f32x2_rn_noftz(weight_184, v_2_38, a_177[3]);
        }
        acc[8] = a_177[0].x;
        acc[9] = a_177[0].y;
        acc[10] = a_177[1].x;
        acc[11] = a_177[1].y;
        acc[12] = a_177[2].x;
        acc[13] = a_177[2].y;
        acc[14] = a_177[3].x;
        acc[15] = a_177[3].y;
        const int woff_185 = 8;
        int base_186 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_187[4];
        float2 _f2_558 = make_float2(acc[16], acc[17]);
        float2 previous_188 = _f2_558;
        a_187[0] = mul_f32x2_noftz(previous_188, corr_164);
        float2 _f2_559 = make_float2(acc[18], acc[19]);
        float2 previous_189 = _f2_559;
        a_187[1] = mul_f32x2_noftz(previous_189, corr_164);
        float2 _f2_560 = make_float2(acc[20], acc[21]);
        float2 previous_190 = _f2_560;
        a_187[2] = mul_f32x2_noftz(previous_190, corr_164);
        float2 _f2_561 = make_float2(acc[22], acc[23]);
        float2 previous_191 = _f2_561;
        a_187[3] = mul_f32x2_noftz(previous_191, corr_164);
        float2 _f2_562 = make_float2(weights_162[3], weights_162[3]);
        float2 weight_192 = _f2_562;
        {
            unsigned int sw2_6[4];
            sw2_6[0] = words[42 + woff_185];
            sw2_6[1] = words[42 + woff_185 + 1];
            sw2_6[2] = words[42 + woff_185 + 2];
            sw2_6[3] = words[42 + woff_185 + 3];
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
            float2 _f2_567 = make_float2(sw2_f32_6[0], sw2_f32_6[1]);
            float2 v_44 = _f2_567;
            a_187[0] = fma_f32x2_rn_noftz(weight_192, v_44, a_187[0]);
            float2 _f2_568 = make_float2(sw2_f32_6[2], sw2_f32_6[3]);
            float2 v_0_42 = _f2_568;
            a_187[1] = fma_f32x2_rn_noftz(weight_192, v_0_42, a_187[1]);
            float2 _f2_569 = make_float2(sw2_f32_6[4], sw2_f32_6[5]);
            float2 v_1_22 = _f2_569;
            a_187[2] = fma_f32x2_rn_noftz(weight_192, v_1_22, a_187[2]);
            float2 _f2_570 = make_float2(sw2_f32_6[6], sw2_f32_6[7]);
            float2 v_2_39 = _f2_570;
            a_187[3] = fma_f32x2_rn_noftz(weight_192, v_2_39, a_187[3]);
        }
        float2 _f2_571 = make_float2(weights_162[4], weights_162[4]);
        float2 weight_193 = _f2_571;
        {
            unsigned int sw2_7[4];
            sw2_7[0] = words[56 + woff_185];
            sw2_7[1] = words[56 + woff_185 + 1];
            sw2_7[2] = words[56 + woff_185 + 2];
            sw2_7[3] = words[56 + woff_185 + 3];
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
            float2 _f2_576 = make_float2(sw2_f32_7[0], sw2_f32_7[1]);
            float2 v_45 = _f2_576;
            a_187[0] = fma_f32x2_rn_noftz(weight_193, v_45, a_187[0]);
            float2 _f2_577 = make_float2(sw2_f32_7[2], sw2_f32_7[3]);
            float2 v_0_43 = _f2_577;
            a_187[1] = fma_f32x2_rn_noftz(weight_193, v_0_43, a_187[1]);
            float2 _f2_578 = make_float2(sw2_f32_7[4], sw2_f32_7[5]);
            float2 v_1_23 = _f2_578;
            a_187[2] = fma_f32x2_rn_noftz(weight_193, v_1_23, a_187[2]);
            float2 _f2_579 = make_float2(sw2_f32_7[6], sw2_f32_7[7]);
            float2 v_2_40 = _f2_579;
            a_187[3] = fma_f32x2_rn_noftz(weight_193, v_2_40, a_187[3]);
        }
        float2 _f2_580 = make_float2(weights_162[5], weights_162[5]);
        float2 weight_194 = _f2_580;
        {
            unsigned int sw2_8[4];
            sw2_8[0] = words[70 + woff_185];
            sw2_8[1] = words[70 + woff_185 + 1];
            sw2_8[2] = words[70 + woff_185 + 2];
            sw2_8[3] = words[70 + woff_185 + 3];
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
            float2 _f2_585 = make_float2(sw2_f32_8[0], sw2_f32_8[1]);
            float2 v_46 = _f2_585;
            a_187[0] = fma_f32x2_rn_noftz(weight_194, v_46, a_187[0]);
            float2 _f2_586 = make_float2(sw2_f32_8[2], sw2_f32_8[3]);
            float2 v_0_44 = _f2_586;
            a_187[1] = fma_f32x2_rn_noftz(weight_194, v_0_44, a_187[1]);
            float2 _f2_587 = make_float2(sw2_f32_8[4], sw2_f32_8[5]);
            float2 v_1_24 = _f2_587;
            a_187[2] = fma_f32x2_rn_noftz(weight_194, v_1_24, a_187[2]);
            float2 _f2_588 = make_float2(sw2_f32_8[6], sw2_f32_8[7]);
            float2 v_2_41 = _f2_588;
            a_187[3] = fma_f32x2_rn_noftz(weight_194, v_2_41, a_187[3]);
        }
        acc[16] = a_187[0].x;
        acc[17] = a_187[0].y;
        acc[18] = a_187[1].x;
        acc[19] = a_187[1].y;
        acc[20] = a_187[2].x;
        acc[21] = a_187[2].y;
        acc[22] = a_187[3].x;
        acc[23] = a_187[3].y;
        const int woff_195 = 12;
        int base_196 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_197[4];
        float2 _f2_589 = make_float2(acc[24], acc[25]);
        float2 previous_198 = _f2_589;
        a_197[0] = mul_f32x2_noftz(previous_198, corr_164);
        float2 _f2_590 = make_float2(acc[26], acc[27]);
        float2 previous_199 = _f2_590;
        a_197[1] = mul_f32x2_noftz(previous_199, corr_164);
        float2 _f2_591 = make_float2(weights_162[3], weights_162[3]);
        float2 weight_200 = _f2_591;
        {
            unsigned int sw2_9[4];
            sw2_9[0] = words[42 + woff_195];
            sw2_9[1] = words[42 + woff_195 + 1];
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
            float2 _f2_594 = make_float2(sw2_f32_9[0], sw2_f32_9[1]);
            float2 v_47 = _f2_594;
            a_197[0] = fma_f32x2_rn_noftz(weight_200, v_47, a_197[0]);
            float2 _f2_595 = make_float2(sw2_f32_9[2], sw2_f32_9[3]);
            float2 v_0_45 = _f2_595;
            a_197[1] = fma_f32x2_rn_noftz(weight_200, v_0_45, a_197[1]);
        }
        float2 _f2_596 = make_float2(weights_162[4], weights_162[4]);
        float2 weight_201 = _f2_596;
        {
            unsigned int sw2_10[4];
            sw2_10[0] = words[56 + woff_195];
            sw2_10[1] = words[56 + woff_195 + 1];
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
            float2 _f2_599 = make_float2(sw2_f32_10[0], sw2_f32_10[1]);
            float2 v_48 = _f2_599;
            a_197[0] = fma_f32x2_rn_noftz(weight_201, v_48, a_197[0]);
            float2 _f2_600 = make_float2(sw2_f32_10[2], sw2_f32_10[3]);
            float2 v_0_46 = _f2_600;
            a_197[1] = fma_f32x2_rn_noftz(weight_201, v_0_46, a_197[1]);
        }
        float2 _f2_601 = make_float2(weights_162[5], weights_162[5]);
        float2 weight_202 = _f2_601;
        {
            unsigned int sw2_11[4];
            sw2_11[0] = words[70 + woff_195];
            sw2_11[1] = words[70 + woff_195 + 1];
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
            float2 _f2_604 = make_float2(sw2_f32_11[0], sw2_f32_11[1]);
            float2 v_49 = _f2_604;
            a_197[0] = fma_f32x2_rn_noftz(weight_202, v_49, a_197[0]);
            float2 _f2_605 = make_float2(sw2_f32_11[2], sw2_f32_11[3]);
            float2 v_0_47 = _f2_605;
            a_197[1] = fma_f32x2_rn_noftz(weight_202, v_0_47, a_197[1]);
        }
        acc[24] = a_197[0].x;
        acc[25] = a_197[0].y;
        acc[26] = a_197[1].x;
        acc[27] = a_197[1].y;
        sum_running = sum_running * correction_161 + sum_weights_163;
        max_running = max_new_160;
        float2 _f2_606 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_606;
        float2 _f2_607 = make_float2(acc[0], acc[1]);
        float2 v_50 = _f2_607;
        output_sq_pair = fma_f32x2_rn_noftz(v_50, v_50, output_sq_pair);
        float2 _f2_608 = make_float2(acc[2], acc[3]);
        float2 v_203 = _f2_608;
        output_sq_pair = fma_f32x2_rn_noftz(v_203, v_203, output_sq_pair);
        float2 _f2_609 = make_float2(acc[4], acc[5]);
        float2 v_204 = _f2_609;
        output_sq_pair = fma_f32x2_rn_noftz(v_204, v_204, output_sq_pair);
        float2 _f2_610 = make_float2(acc[6], acc[7]);
        float2 v_205 = _f2_610;
        output_sq_pair = fma_f32x2_rn_noftz(v_205, v_205, output_sq_pair);
        float2 _f2_611 = make_float2(acc[8], acc[9]);
        float2 v_206 = _f2_611;
        output_sq_pair = fma_f32x2_rn_noftz(v_206, v_206, output_sq_pair);
        float2 _f2_612 = make_float2(acc[10], acc[11]);
        float2 v_207 = _f2_612;
        output_sq_pair = fma_f32x2_rn_noftz(v_207, v_207, output_sq_pair);
        float2 _f2_613 = make_float2(acc[12], acc[13]);
        float2 v_208 = _f2_613;
        output_sq_pair = fma_f32x2_rn_noftz(v_208, v_208, output_sq_pair);
        float2 _f2_614 = make_float2(acc[14], acc[15]);
        float2 v_209 = _f2_614;
        output_sq_pair = fma_f32x2_rn_noftz(v_209, v_209, output_sq_pair);
        float2 _f2_615 = make_float2(acc[16], acc[17]);
        float2 v_210 = _f2_615;
        output_sq_pair = fma_f32x2_rn_noftz(v_210, v_210, output_sq_pair);
        float2 _f2_616 = make_float2(acc[18], acc[19]);
        float2 v_211 = _f2_616;
        output_sq_pair = fma_f32x2_rn_noftz(v_211, v_211, output_sq_pair);
        float2 _f2_617 = make_float2(acc[20], acc[21]);
        float2 v_212 = _f2_617;
        output_sq_pair = fma_f32x2_rn_noftz(v_212, v_212, output_sq_pair);
        float2 _f2_618 = make_float2(acc[22], acc[23]);
        float2 v_213 = _f2_618;
        output_sq_pair = fma_f32x2_rn_noftz(v_213, v_213, output_sq_pair);
        float2 _f2_619 = make_float2(acc[24], acc[25]);
        float2 v_214 = _f2_619;
        output_sq_pair = fma_f32x2_rn_noftz(v_214, v_214, output_sq_pair);
        float2 _f2_620 = make_float2(acc[26], acc[27]);
        float2 v_215 = _f2_620;
        output_sq_pair = fma_f32x2_rn_noftz(v_215, v_215, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_30;
        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_31;
        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_32;
        float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_33;
        float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_34;
        if (lane == 0) {
            uint32_t _mapa_24;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_24) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_24), "f"(output_sq) : "memory");
            uint32_t _mapa_25;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_25) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_25), "f"(output_sq) : "memory");
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
        float _shfl_6 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_6;
        float2 _f2_621 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_621;
        int base_216 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_622 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_622, rsigma_pair);
        float2 _f2_623 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_623);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_217 = 2;
        const int acc_idx_218 = value_idx_217;
        float2 _f2_624 = make_float2(acc[acc_idx_218], acc[acc_idx_218 + 1]);
        float2 scaled_pair_219 = mul_f32x2_noftz(_f2_624, rsigma_pair);
        float2 _f2_625 = make_float2(wout[acc_idx_218], wout[acc_idx_218 + 1]);
        float2 normalized_pair_220 = mul_f32x2_noftz(scaled_pair_219, _f2_625);
        output_values[value_idx_217] = normalized_pair_220.x;
        output_values[value_idx_217 + 1] = normalized_pair_220.y;
        const int value_idx_221 = 4;
        const int acc_idx_222 = value_idx_221;
        float2 _f2_626 = make_float2(acc[acc_idx_222], acc[acc_idx_222 + 1]);
        float2 scaled_pair_223 = mul_f32x2_noftz(_f2_626, rsigma_pair);
        float2 _f2_627 = make_float2(wout[acc_idx_222], wout[acc_idx_222 + 1]);
        float2 normalized_pair_224 = mul_f32x2_noftz(scaled_pair_223, _f2_627);
        output_values[value_idx_221] = normalized_pair_224.x;
        output_values[value_idx_221 + 1] = normalized_pair_224.y;
        const int value_idx_225 = 6;
        const int acc_idx_226 = value_idx_225;
        float2 _f2_628 = make_float2(acc[acc_idx_226], acc[acc_idx_226 + 1]);
        float2 scaled_pair_227 = mul_f32x2_noftz(_f2_628, rsigma_pair);
        float2 _f2_629 = make_float2(wout[acc_idx_226], wout[acc_idx_226 + 1]);
        float2 normalized_pair_228 = mul_f32x2_noftz(scaled_pair_227, _f2_629);
        output_values[value_idx_225] = normalized_pair_228.x;
        output_values[value_idx_225 + 1] = normalized_pair_228.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_216 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_229 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_230[8];
        const int value_idx_231 = 0;
        const int acc_idx_232 = 8 + value_idx_231;
        float2 _f2_630 = make_float2(acc[acc_idx_232], acc[acc_idx_232 + 1]);
        float2 scaled_pair_233 = mul_f32x2_noftz(_f2_630, rsigma_pair);
        float2 _f2_631 = make_float2(wout[acc_idx_232], wout[acc_idx_232 + 1]);
        float2 normalized_pair_234 = mul_f32x2_noftz(scaled_pair_233, _f2_631);
        output_values_230[value_idx_231] = normalized_pair_234.x;
        output_values_230[value_idx_231 + 1] = normalized_pair_234.y;
        const int value_idx_235 = 2;
        const int acc_idx_236 = 8 + value_idx_235;
        float2 _f2_632 = make_float2(acc[acc_idx_236], acc[acc_idx_236 + 1]);
        float2 scaled_pair_237 = mul_f32x2_noftz(_f2_632, rsigma_pair);
        float2 _f2_633 = make_float2(wout[acc_idx_236], wout[acc_idx_236 + 1]);
        float2 normalized_pair_238 = mul_f32x2_noftz(scaled_pair_237, _f2_633);
        output_values_230[value_idx_235] = normalized_pair_238.x;
        output_values_230[value_idx_235 + 1] = normalized_pair_238.y;
        const int value_idx_239 = 4;
        const int acc_idx_240 = 8 + value_idx_239;
        float2 _f2_634 = make_float2(acc[acc_idx_240], acc[acc_idx_240 + 1]);
        float2 scaled_pair_241 = mul_f32x2_noftz(_f2_634, rsigma_pair);
        float2 _f2_635 = make_float2(wout[acc_idx_240], wout[acc_idx_240 + 1]);
        float2 normalized_pair_242 = mul_f32x2_noftz(scaled_pair_241, _f2_635);
        output_values_230[value_idx_239] = normalized_pair_242.x;
        output_values_230[value_idx_239 + 1] = normalized_pair_242.y;
        const int value_idx_243 = 6;
        const int acc_idx_244 = 8 + value_idx_243;
        float2 _f2_636 = make_float2(acc[acc_idx_244], acc[acc_idx_244 + 1]);
        float2 scaled_pair_245 = mul_f32x2_noftz(_f2_636, rsigma_pair);
        float2 _f2_637 = make_float2(wout[acc_idx_244], wout[acc_idx_244 + 1]);
        float2 normalized_pair_246 = mul_f32x2_noftz(scaled_pair_245, _f2_637);
        output_values_230[value_idx_243] = normalized_pair_246.x;
        output_values_230[value_idx_243 + 1] = normalized_pair_246.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_230[0 + 0], output_values_230[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_230[0 + 2], output_values_230[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_230[0 + 4], output_values_230[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_230[0 + 6], output_values_230[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_229 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_247 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_248[8];
        const int value_idx_249 = 0;
        const int acc_idx_250 = 16 + value_idx_249;
        float2 _f2_638 = make_float2(acc[acc_idx_250], acc[acc_idx_250 + 1]);
        float2 scaled_pair_251 = mul_f32x2_noftz(_f2_638, rsigma_pair);
        float2 _f2_639 = make_float2(wout[acc_idx_250], wout[acc_idx_250 + 1]);
        float2 normalized_pair_252 = mul_f32x2_noftz(scaled_pair_251, _f2_639);
        output_values_248[value_idx_249] = normalized_pair_252.x;
        output_values_248[value_idx_249 + 1] = normalized_pair_252.y;
        const int value_idx_253 = 2;
        const int acc_idx_254 = 16 + value_idx_253;
        float2 _f2_640 = make_float2(acc[acc_idx_254], acc[acc_idx_254 + 1]);
        float2 scaled_pair_255 = mul_f32x2_noftz(_f2_640, rsigma_pair);
        float2 _f2_641 = make_float2(wout[acc_idx_254], wout[acc_idx_254 + 1]);
        float2 normalized_pair_256 = mul_f32x2_noftz(scaled_pair_255, _f2_641);
        output_values_248[value_idx_253] = normalized_pair_256.x;
        output_values_248[value_idx_253 + 1] = normalized_pair_256.y;
        const int value_idx_257 = 4;
        const int acc_idx_258 = 16 + value_idx_257;
        float2 _f2_642 = make_float2(acc[acc_idx_258], acc[acc_idx_258 + 1]);
        float2 scaled_pair_259 = mul_f32x2_noftz(_f2_642, rsigma_pair);
        float2 _f2_643 = make_float2(wout[acc_idx_258], wout[acc_idx_258 + 1]);
        float2 normalized_pair_260 = mul_f32x2_noftz(scaled_pair_259, _f2_643);
        output_values_248[value_idx_257] = normalized_pair_260.x;
        output_values_248[value_idx_257 + 1] = normalized_pair_260.y;
        const int value_idx_261 = 6;
        const int acc_idx_262 = 16 + value_idx_261;
        float2 _f2_644 = make_float2(acc[acc_idx_262], acc[acc_idx_262 + 1]);
        float2 scaled_pair_263 = mul_f32x2_noftz(_f2_644, rsigma_pair);
        float2 _f2_645 = make_float2(wout[acc_idx_262], wout[acc_idx_262 + 1]);
        float2 normalized_pair_264 = mul_f32x2_noftz(scaled_pair_263, _f2_645);
        output_values_248[value_idx_261] = normalized_pair_264.x;
        output_values_248[value_idx_261 + 1] = normalized_pair_264.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_248[0 + 0], output_values_248[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_248[0 + 2], output_values_248[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_248[0 + 4], output_values_248[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_248[0 + 6], output_values_248[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_247 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_265 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_266[8];
        const int value_idx_267 = 0;
        const int acc_idx_268 = 24 + value_idx_267;
        float2 _f2_646 = make_float2(acc[acc_idx_268], acc[acc_idx_268 + 1]);
        float2 scaled_pair_269 = mul_f32x2_noftz(_f2_646, rsigma_pair);
        float2 _f2_647 = make_float2(wout[acc_idx_268], wout[acc_idx_268 + 1]);
        float2 normalized_pair_270 = mul_f32x2_noftz(scaled_pair_269, _f2_647);
        output_values_266[value_idx_267] = normalized_pair_270.x;
        output_values_266[value_idx_267 + 1] = normalized_pair_270.y;
        const int value_idx_271 = 2;
        const int acc_idx_272 = 24 + value_idx_271;
        float2 _f2_648 = make_float2(acc[acc_idx_272], acc[acc_idx_272 + 1]);
        float2 scaled_pair_273 = mul_f32x2_noftz(_f2_648, rsigma_pair);
        float2 _f2_649 = make_float2(wout[acc_idx_272], wout[acc_idx_272 + 1]);
        float2 normalized_pair_274 = mul_f32x2_noftz(scaled_pair_273, _f2_649);
        output_values_266[value_idx_271] = normalized_pair_274.x;
        output_values_266[value_idx_271 + 1] = normalized_pair_274.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_266[0 + 0], output_values_266[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_266[0 + 2], output_values_266[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_265]) = _pk2;
        }
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
