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
#define SMEM_STATS_STAGE_BYTES 448
#define SMEM_STATS_STRIDE 448
#define SMEM_OUT_STATS_OFF 448
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 512
#define THREADS 64

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

__global__ __launch_bounds__(64) __cluster_dims__(4,1,1) void
kernel_cake_kimi_k3_attn_res_767d23dd29a86915ae01(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* stats = reinterpret_cast<float*>(smem_raw + 0);
    const int stats_addr = smem + 0;
    float* out_stats = reinterpret_cast<float*>(smem_raw + 448);
    const int out_stats_addr = smem + 448;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int warp_0 = warp;
    int token = bid;
    warp_0 = cta_rank * 2 + warp;
    token = cluster_id;
    int group = warp_0 / 4;
    int thread = (warp_0 * 32 + lane) % 128;
    if (token < M) {
        unsigned long long token64 = (unsigned long long)token;
        unsigned long long row_base = token64 * 7168;
        unsigned long long block_base = token64 * blocks_m_stride;
        unsigned int words[98];
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
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_21[0 + 0] = _uv4_7.x;
                _vec_load_21[0 + 1] = _uv4_7.y;
                _vec_load_21[0 + 2] = _uv4_7.z;
                _vec_load_21[0 + 3] = _uv4_7.w;
            }
            dwords[woff] = _vec_load_21[0];
            dwords[woff + 1] = _vec_load_21[1];
            dwords[woff + 2] = _vec_load_21[2];
            dwords[woff + 3] = _vec_load_21[3];
        }
        float _vec_load_24[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(norm_weight + base);
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
                        : "=f"((&_vec_load_24[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_24[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        float _vec_load_25[8];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(qk_weight + base);
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
                        : "=f"((&_vec_load_25[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_25[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        q[0] = _vec_load_24[0] * _vec_load_25[0];
        q[1] = _vec_load_24[1] * _vec_load_25[1];
        q[2] = _vec_load_24[2] * _vec_load_25[2];
        q[3] = _vec_load_24[3] * _vec_load_25[3];
        q[4] = _vec_load_24[4] * _vec_load_25[4];
        q[5] = _vec_load_24[5] * _vec_load_25[5];
        q[6] = _vec_load_24[6] * _vec_load_25[6];
        q[7] = _vec_load_24[7] * _vec_load_25[7];
        float _vec_load_26[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(output_norm_weight + base);
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
                        : "=f"((&_vec_load_26[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_26[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        wout[0] = _vec_load_26[0];
        wout[1] = _vec_load_26[1];
        wout[2] = _vec_load_26[2];
        wout[3] = _vec_load_26[3];
        wout[4] = _vec_load_26[4];
        wout[5] = _vec_load_26[5];
        wout[6] = _vec_load_26[6];
        wout[7] = _vec_load_26[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_27[4];
            {
                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_27[0 + 0] = _uv4_11.x;
                _vec_load_27[0 + 1] = _uv4_11.y;
                _vec_load_27[0 + 2] = _uv4_11.z;
                _vec_load_27[0 + 3] = _uv4_11.w;
            }
            words[woff_1] = _vec_load_27[0];
            words[woff_1 + 1] = _vec_load_27[1];
            words[woff_1 + 2] = _vec_load_27[2];
            words[woff_1 + 3] = _vec_load_27[3];
        }
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_30[0 + 0] = _uv4_12.x;
                _vec_load_30[0 + 1] = _uv4_12.y;
                _vec_load_30[0 + 2] = _uv4_12.z;
                _vec_load_30[0 + 3] = _uv4_12.w;
            }
            words[14 + woff_1] = _vec_load_30[0];
            words[14 + woff_1 + 1] = _vec_load_30[1];
            words[14 + woff_1 + 2] = _vec_load_30[2];
            words[14 + woff_1 + 3] = _vec_load_30[3];
        }
        {
            unsigned int _vec_load_33[4];
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_33[0 + 0] = _uv4_13.x;
                _vec_load_33[0 + 1] = _uv4_13.y;
                _vec_load_33[0 + 2] = _uv4_13.z;
                _vec_load_33[0 + 3] = _uv4_13.w;
            }
            words[28 + woff_1] = _vec_load_33[0];
            words[28 + woff_1 + 1] = _vec_load_33[1];
            words[28 + woff_1 + 2] = _vec_load_33[2];
            words[28 + woff_1 + 3] = _vec_load_33[3];
        }
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_36[0 + 0] = _uv4_14.x;
                _vec_load_36[0 + 1] = _uv4_14.y;
                _vec_load_36[0 + 2] = _uv4_14.z;
                _vec_load_36[0 + 3] = _uv4_14.w;
            }
            words[42 + woff_1] = _vec_load_36[0];
            words[42 + woff_1 + 1] = _vec_load_36[1];
            words[42 + woff_1 + 2] = _vec_load_36[2];
            words[42 + woff_1 + 3] = _vec_load_36[3];
        }
        {
            unsigned int _vec_load_39[4];
            {
                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_39[0 + 0] = _uv4_15.x;
                _vec_load_39[0 + 1] = _uv4_15.y;
                _vec_load_39[0 + 2] = _uv4_15.z;
                _vec_load_39[0 + 3] = _uv4_15.w;
            }
            words[56 + woff_1] = _vec_load_39[0];
            words[56 + woff_1 + 1] = _vec_load_39[1];
            words[56 + woff_1 + 2] = _vec_load_39[2];
            words[56 + woff_1 + 3] = _vec_load_39[3];
        }
        {
            unsigned int _vec_load_42[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_42[0 + 0] = _uv4_16.x;
                _vec_load_42[0 + 1] = _uv4_16.y;
                _vec_load_42[0 + 2] = _uv4_16.z;
                _vec_load_42[0 + 3] = _uv4_16.w;
            }
            words[70 + woff_1] = _vec_load_42[0];
            words[70 + woff_1 + 1] = _vec_load_42[1];
            words[70 + woff_1 + 2] = _vec_load_42[2];
            words[70 + woff_1 + 3] = _vec_load_42[3];
        }
        {
            unsigned int _vec_load_45[4];
            {
                uint4 _uv4_17 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_45[0 + 0] = _uv4_17.x;
                _vec_load_45[0 + 1] = _uv4_17.y;
                _vec_load_45[0 + 2] = _uv4_17.z;
                _vec_load_45[0 + 3] = _uv4_17.w;
            }
            words[84 + woff_1] = _vec_load_45[0];
            words[84 + woff_1 + 1] = _vec_load_45[1];
            words[84 + woff_1 + 2] = _vec_load_45[2];
            words[84 + woff_1 + 3] = _vec_load_45[3];
        }
        {
            unsigned int _vec_load_48[4];
            {
                uint4 _uv4_18 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_48[0 + 0] = _uv4_18.x;
                _vec_load_48[0 + 1] = _uv4_18.y;
                _vec_load_48[0 + 2] = _uv4_18.z;
                _vec_load_48[0 + 3] = _uv4_18.w;
            }
            dwords[woff_1] = _vec_load_48[0];
            dwords[woff_1 + 1] = _vec_load_48[1];
            dwords[woff_1 + 2] = _vec_load_48[2];
            dwords[woff_1 + 3] = _vec_load_48[3];
        }
        float _vec_load_51[8];
        {
            const uint4* _vptr_19 = reinterpret_cast<const uint4*>(norm_weight + base_0);
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
                        : "=f"((&_vec_load_51[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_51[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        float _vec_load_52[8];
        {
            const uint4* _vptr_20 = reinterpret_cast<const uint4*>(qk_weight + base_0);
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
                        : "=f"((&_vec_load_52[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_52[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_20[_pair]));
                }
            }
        }
        q[8] = _vec_load_51[0] * _vec_load_52[0];
        q[9] = _vec_load_51[1] * _vec_load_52[1];
        q[10] = _vec_load_51[2] * _vec_load_52[2];
        q[11] = _vec_load_51[3] * _vec_load_52[3];
        q[12] = _vec_load_51[4] * _vec_load_52[4];
        q[13] = _vec_load_51[5] * _vec_load_52[5];
        q[14] = _vec_load_51[6] * _vec_load_52[6];
        q[15] = _vec_load_51[7] * _vec_load_52[7];
        float _vec_load_53[8];
        {
            const uint4* _vptr_21 = reinterpret_cast<const uint4*>(output_norm_weight + base_0);
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
                        : "=f"((&_vec_load_53[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_53[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_21[_pair]));
                }
            }
        }
        wout[8] = _vec_load_53[0];
        wout[9] = _vec_load_53[1];
        wout[10] = _vec_load_53[2];
        wout[11] = _vec_load_53[3];
        wout[12] = _vec_load_53[4];
        wout[13] = _vec_load_53[5];
        wout[14] = _vec_load_53[6];
        wout[15] = _vec_load_53[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_54[4];
            {
                uint4 _uv4_22 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_54[0 + 0] = _uv4_22.x;
                _vec_load_54[0 + 1] = _uv4_22.y;
                _vec_load_54[0 + 2] = _uv4_22.z;
                _vec_load_54[0 + 3] = _uv4_22.w;
            }
            words[woff_3] = _vec_load_54[0];
            words[woff_3 + 1] = _vec_load_54[1];
            words[woff_3 + 2] = _vec_load_54[2];
            words[woff_3 + 3] = _vec_load_54[3];
        }
        {
            unsigned int _vec_load_57[4];
            {
                uint4 _uv4_23 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_57[0 + 0] = _uv4_23.x;
                _vec_load_57[0 + 1] = _uv4_23.y;
                _vec_load_57[0 + 2] = _uv4_23.z;
                _vec_load_57[0 + 3] = _uv4_23.w;
            }
            words[14 + woff_3] = _vec_load_57[0];
            words[14 + woff_3 + 1] = _vec_load_57[1];
            words[14 + woff_3 + 2] = _vec_load_57[2];
            words[14 + woff_3 + 3] = _vec_load_57[3];
        }
        {
            unsigned int _vec_load_60[4];
            {
                uint4 _uv4_24 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_60[0 + 0] = _uv4_24.x;
                _vec_load_60[0 + 1] = _uv4_24.y;
                _vec_load_60[0 + 2] = _uv4_24.z;
                _vec_load_60[0 + 3] = _uv4_24.w;
            }
            words[28 + woff_3] = _vec_load_60[0];
            words[28 + woff_3 + 1] = _vec_load_60[1];
            words[28 + woff_3 + 2] = _vec_load_60[2];
            words[28 + woff_3 + 3] = _vec_load_60[3];
        }
        {
            unsigned int _vec_load_63[4];
            {
                uint4 _uv4_25 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_63[0 + 0] = _uv4_25.x;
                _vec_load_63[0 + 1] = _uv4_25.y;
                _vec_load_63[0 + 2] = _uv4_25.z;
                _vec_load_63[0 + 3] = _uv4_25.w;
            }
            words[42 + woff_3] = _vec_load_63[0];
            words[42 + woff_3 + 1] = _vec_load_63[1];
            words[42 + woff_3 + 2] = _vec_load_63[2];
            words[42 + woff_3 + 3] = _vec_load_63[3];
        }
        {
            unsigned int _vec_load_66[4];
            {
                uint4 _uv4_26 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_66[0 + 0] = _uv4_26.x;
                _vec_load_66[0 + 1] = _uv4_26.y;
                _vec_load_66[0 + 2] = _uv4_26.z;
                _vec_load_66[0 + 3] = _uv4_26.w;
            }
            words[56 + woff_3] = _vec_load_66[0];
            words[56 + woff_3 + 1] = _vec_load_66[1];
            words[56 + woff_3 + 2] = _vec_load_66[2];
            words[56 + woff_3 + 3] = _vec_load_66[3];
        }
        {
            unsigned int _vec_load_69[4];
            {
                uint4 _uv4_27 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_69[0 + 0] = _uv4_27.x;
                _vec_load_69[0 + 1] = _uv4_27.y;
                _vec_load_69[0 + 2] = _uv4_27.z;
                _vec_load_69[0 + 3] = _uv4_27.w;
            }
            words[70 + woff_3] = _vec_load_69[0];
            words[70 + woff_3 + 1] = _vec_load_69[1];
            words[70 + woff_3 + 2] = _vec_load_69[2];
            words[70 + woff_3 + 3] = _vec_load_69[3];
        }
        {
            unsigned int _vec_load_72[4];
            {
                uint4 _uv4_28 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_72[0 + 0] = _uv4_28.x;
                _vec_load_72[0 + 1] = _uv4_28.y;
                _vec_load_72[0 + 2] = _uv4_28.z;
                _vec_load_72[0 + 3] = _uv4_28.w;
            }
            words[84 + woff_3] = _vec_load_72[0];
            words[84 + woff_3 + 1] = _vec_load_72[1];
            words[84 + woff_3 + 2] = _vec_load_72[2];
            words[84 + woff_3 + 3] = _vec_load_72[3];
        }
        {
            unsigned int _vec_load_75[4];
            {
                uint4 _uv4_29 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_75[0 + 0] = _uv4_29.x;
                _vec_load_75[0 + 1] = _uv4_29.y;
                _vec_load_75[0 + 2] = _uv4_29.z;
                _vec_load_75[0 + 3] = _uv4_29.w;
            }
            dwords[woff_3] = _vec_load_75[0];
            dwords[woff_3 + 1] = _vec_load_75[1];
            dwords[woff_3 + 2] = _vec_load_75[2];
            dwords[woff_3 + 3] = _vec_load_75[3];
        }
        float _vec_load_78[8];
        {
            const uint4* _vptr_30 = reinterpret_cast<const uint4*>(norm_weight + base_2);
            uint4 _vld_30[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_30[_blk] = _vptr_30[_blk];
                uint32_t* _vpairs_30 = reinterpret_cast<uint32_t*>(&_vld_30[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_78[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_78[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_30[_pair]));
                }
            }
        }
        float _vec_load_79[8];
        {
            const uint4* _vptr_31 = reinterpret_cast<const uint4*>(qk_weight + base_2);
            uint4 _vld_31[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_31[_blk] = _vptr_31[_blk];
                uint32_t* _vpairs_31 = reinterpret_cast<uint32_t*>(&_vld_31[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_79[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_79[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_31[_pair]));
                }
            }
        }
        q[16] = _vec_load_78[0] * _vec_load_79[0];
        q[17] = _vec_load_78[1] * _vec_load_79[1];
        q[18] = _vec_load_78[2] * _vec_load_79[2];
        q[19] = _vec_load_78[3] * _vec_load_79[3];
        q[20] = _vec_load_78[4] * _vec_load_79[4];
        q[21] = _vec_load_78[5] * _vec_load_79[5];
        q[22] = _vec_load_78[6] * _vec_load_79[6];
        q[23] = _vec_load_78[7] * _vec_load_79[7];
        float _vec_load_80[8];
        {
            const uint4* _vptr_32 = reinterpret_cast<const uint4*>(output_norm_weight + base_2);
            uint4 _vld_32[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_32[_blk] = _vptr_32[_blk];
                uint32_t* _vpairs_32 = reinterpret_cast<uint32_t*>(&_vld_32[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_80[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_80[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_32[_pair]));
                }
            }
        }
        wout[16] = _vec_load_80[0];
        wout[17] = _vec_load_80[1];
        wout[18] = _vec_load_80[2];
        wout[19] = _vec_load_80[3];
        wout[20] = _vec_load_80[4];
        wout[21] = _vec_load_80[5];
        wout[22] = _vec_load_80[6];
        wout[23] = _vec_load_80[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_82[1];
            {
                _vec_load_82[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_82[0];
            unsigned int _vec_load_83[1];
            {
                _vec_load_83[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_83[0];
        }
        {
            unsigned int _vec_load_85[1];
            {
                _vec_load_85[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_85[0];
            unsigned int _vec_load_86[1];
            {
                _vec_load_86[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_86[0];
        }
        {
            unsigned int _vec_load_88[1];
            {
                _vec_load_88[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_88[0];
            unsigned int _vec_load_89[1];
            {
                _vec_load_89[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_89[0];
        }
        {
            unsigned int _vec_load_91[1];
            {
                _vec_load_91[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[42 + woff_5] = _vec_load_91[0];
            unsigned int _vec_load_92[1];
            {
                _vec_load_92[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[42 + woff_5 + 1] = _vec_load_92[0];
        }
        {
            unsigned int _vec_load_94[1];
            {
                _vec_load_94[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[56 + woff_5] = _vec_load_94[0];
            unsigned int _vec_load_95[1];
            {
                _vec_load_95[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[56 + woff_5 + 1] = _vec_load_95[0];
        }
        {
            unsigned int _vec_load_97[1];
            {
                _vec_load_97[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[70 + woff_5] = _vec_load_97[0];
            unsigned int _vec_load_98[1];
            {
                _vec_load_98[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[70 + woff_5 + 1] = _vec_load_98[0];
        }
        {
            unsigned int _vec_load_100[1];
            {
                _vec_load_100[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[84 + woff_5] = _vec_load_100[0];
            unsigned int _vec_load_101[1];
            {
                _vec_load_101[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[84 + woff_5 + 1] = _vec_load_101[0];
        }
        {
            unsigned int _vec_load_103[1];
            {
                _vec_load_103[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_103[0];
            unsigned int _vec_load_104[1];
            {
                _vec_load_104[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_104[0];
        }
        float _vec_load_105[4];
        {
            uint2 _vld_33;
            _vld_33 = *reinterpret_cast<const uint2*>(norm_weight + base_4);
            uint32_t* _vpairs_33 = reinterpret_cast<uint32_t*>(&_vld_33);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_105[0 + _pair * 2])[0]), "=f"((&_vec_load_105[0 + _pair * 2])[1])
                    : "r"(_vpairs_33[_pair]));
            }
        }
        float _vec_load_106[4];
        {
            uint2 _vld_34;
            _vld_34 = *reinterpret_cast<const uint2*>(qk_weight + base_4);
            uint32_t* _vpairs_34 = reinterpret_cast<uint32_t*>(&_vld_34);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_106[0 + _pair * 2])[0]), "=f"((&_vec_load_106[0 + _pair * 2])[1])
                    : "r"(_vpairs_34[_pair]));
            }
        }
        q[24] = _vec_load_105[0] * _vec_load_106[0];
        q[25] = _vec_load_105[1] * _vec_load_106[1];
        q[26] = _vec_load_105[2] * _vec_load_106[2];
        q[27] = _vec_load_105[3] * _vec_load_106[3];
        float _vec_load_107[4];
        {
            uint2 _vld_35;
            _vld_35 = *reinterpret_cast<const uint2*>(output_norm_weight + base_4);
            uint32_t* _vpairs_35 = reinterpret_cast<uint32_t*>(&_vld_35);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_107[0 + _pair * 2])[0]), "=f"((&_vec_load_107[0 + _pair * 2])[1])
                    : "r"(_vpairs_35[_pair]));
            }
        }
        wout[24] = _vec_load_107[0];
        wout[25] = _vec_load_107[1];
        wout[26] = _vec_load_107[2];
        wout[27] = _vec_load_107[3];
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
        float2 sq[7];
        float2 dot[7];
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
            float2 _f2_20 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_20;
            float2 _f2_21 = make_float2(q[0], q[1]);
            float2 qp = _f2_21;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_22 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_22;
            float2 _f2_23 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_23;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_24 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_24;
            float2 _f2_25 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_25;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_26 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_26;
            float2 _f2_27 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_27;
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
            float2 _f2_34 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_34;
            float2 _f2_35 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_35;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_36 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_36;
            float2 _f2_37 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_37;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_38 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_38;
            float2 _f2_39 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_39;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_40 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_40;
            float2 _f2_41 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_41;
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
            float2 _f2_48 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_48;
            float2 _f2_49 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_49;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_50 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_50;
            float2 _f2_51 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_51;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_52 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_52;
            float2 _f2_53 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_53;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_54 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_54;
            float2 _f2_55 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_55;
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
            float2 _f2_62 = make_float2(sw_10_f32[0], sw_10_f32[1]);
            float2 v_5 = _f2_62;
            float2 _f2_63 = make_float2(q[0], q[1]);
            float2 qp_6 = _f2_63;
            sq[3] = fma_f32x2_rn_noftz(v_5, v_5, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_5, qp_6, dot[3]);
            float2 _f2_64 = make_float2(sw_10_f32[2], sw_10_f32[3]);
            float2 v_0_3 = _f2_64;
            float2 _f2_65 = make_float2(q[2], q[3]);
            float2 qp_1_3 = _f2_65;
            sq[3] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[3]);
            float2 _f2_66 = make_float2(sw_10_f32[4], sw_10_f32[5]);
            float2 v_2_3 = _f2_66;
            float2 _f2_67 = make_float2(q[4], q[5]);
            float2 qp_3_3 = _f2_67;
            sq[3] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[3]);
            float2 _f2_68 = make_float2(sw_10_f32[6], sw_10_f32[7]);
            float2 v_4_3 = _f2_68;
            float2 _f2_69 = make_float2(q[6], q[7]);
            float2 qp_5_3 = _f2_69;
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
            float2 _f2_76 = make_float2(sw_11_f32[0], sw_11_f32[1]);
            float2 v_6 = _f2_76;
            float2 _f2_77 = make_float2(q[0], q[1]);
            float2 qp_7 = _f2_77;
            sq[4] = fma_f32x2_rn_noftz(v_6, v_6, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_6, qp_7, dot[4]);
            float2 _f2_78 = make_float2(sw_11_f32[2], sw_11_f32[3]);
            float2 v_0_4 = _f2_78;
            float2 _f2_79 = make_float2(q[2], q[3]);
            float2 qp_1_4 = _f2_79;
            sq[4] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[4]);
            float2 _f2_80 = make_float2(sw_11_f32[4], sw_11_f32[5]);
            float2 v_2_4 = _f2_80;
            float2 _f2_81 = make_float2(q[4], q[5]);
            float2 qp_3_4 = _f2_81;
            sq[4] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[4]);
            float2 _f2_82 = make_float2(sw_11_f32[6], sw_11_f32[7]);
            float2 v_4_4 = _f2_82;
            float2 _f2_83 = make_float2(q[6], q[7]);
            float2 qp_5_4 = _f2_83;
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
            float2 _f2_90 = make_float2(sw_12_f32[0], sw_12_f32[1]);
            float2 v_7 = _f2_90;
            float2 _f2_91 = make_float2(q[0], q[1]);
            float2 qp_8 = _f2_91;
            sq[5] = fma_f32x2_rn_noftz(v_7, v_7, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_7, qp_8, dot[5]);
            float2 _f2_92 = make_float2(sw_12_f32[2], sw_12_f32[3]);
            float2 v_0_5 = _f2_92;
            float2 _f2_93 = make_float2(q[2], q[3]);
            float2 qp_1_5 = _f2_93;
            sq[5] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[5]);
            float2 _f2_94 = make_float2(sw_12_f32[4], sw_12_f32[5]);
            float2 v_2_5 = _f2_94;
            float2 _f2_95 = make_float2(q[4], q[5]);
            float2 qp_3_5 = _f2_95;
            sq[5] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[5]);
            float2 _f2_96 = make_float2(sw_12_f32[6], sw_12_f32[7]);
            float2 v_4_5 = _f2_96;
            float2 _f2_97 = make_float2(q[6], q[7]);
            float2 qp_5_5 = _f2_97;
            sq[5] = fma_f32x2_rn_noftz(v_4_5, v_4_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_5, qp_5_5, dot[5]);
        }
        unsigned int sw_13[4];
        sw_13[0] = words[84 + woff_7];
        sw_13[1] = words[84 + woff_7 + 1];
        sw_13[2] = words[84 + woff_7 + 2];
        sw_13[3] = words[84 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_13[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_13[0] = __as_u32(mixed);
            words[84 + woff_7] = sw_13[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_13[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_13[1] = __as_u32(mixed_2);
            words[84 + woff_7 + 1] = sw_13[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_13[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_13[2] = __as_u32(mixed_5);
            words[84 + woff_7 + 2] = sw_13[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_13[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_13[3] = __as_u32(mixed_8);
            words[84 + woff_7 + 3] = sw_13[3];
            {
                int4 _iv4 = make_int4(sw_13[0 + 0], sw_13[0 + 1], sw_13[0 + 2], sw_13[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
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
            float2 _f2_104 = make_float2(sw_13_f32[0], sw_13_f32[1]);
            float2 v_8 = _f2_104;
            float2 _f2_105 = make_float2(q[0], q[1]);
            float2 qp_9 = _f2_105;
            sq[6] = fma_f32x2_rn_noftz(v_8, v_8, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_8, qp_9, dot[6]);
            float2 _f2_106 = make_float2(sw_13_f32[2], sw_13_f32[3]);
            float2 v_0_6 = _f2_106;
            float2 _f2_107 = make_float2(q[2], q[3]);
            float2 qp_1_6 = _f2_107;
            sq[6] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[6]);
            float2 _f2_108 = make_float2(sw_13_f32[4], sw_13_f32[5]);
            float2 v_2_6 = _f2_108;
            float2 _f2_109 = make_float2(q[4], q[5]);
            float2 qp_3_6 = _f2_109;
            sq[6] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[6]);
            float2 _f2_110 = make_float2(sw_13_f32[6], sw_13_f32[7]);
            float2 v_4_6 = _f2_110;
            float2 _f2_111 = make_float2(q[6], q[7]);
            float2 qp_5_6 = _f2_111;
            sq[6] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[6]);
        }
        int base_14 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_15 = 4;
        unsigned int sw_16[4];
        sw_16[0] = words[woff_15];
        sw_16[1] = words[woff_15 + 1];
        sw_16[2] = words[woff_15 + 2];
        sw_16[3] = words[woff_15 + 3];
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
            fsrc[8] = sw_16_f32[0];
            fsrc[9] = sw_16_f32[1];
            fsrc[10] = sw_16_f32[2];
            fsrc[11] = sw_16_f32[3];
            fsrc[12] = sw_16_f32[4];
            fsrc[13] = sw_16_f32[5];
            fsrc[14] = sw_16_f32[6];
            fsrc[15] = sw_16_f32[7];
        }
        {
            float2 _f2_118 = make_float2(sw_16_f32[0], sw_16_f32[1]);
            float2 v_9 = _f2_118;
            float2 _f2_119 = make_float2(q[8], q[9]);
            float2 qp_10 = _f2_119;
            sq[0] = fma_f32x2_rn_noftz(v_9, v_9, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_9, qp_10, dot[0]);
            float2 _f2_120 = make_float2(sw_16_f32[2], sw_16_f32[3]);
            float2 v_0_7 = _f2_120;
            float2 _f2_121 = make_float2(q[10], q[11]);
            float2 qp_1_7 = _f2_121;
            sq[0] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[0]);
            float2 _f2_122 = make_float2(sw_16_f32[4], sw_16_f32[5]);
            float2 v_2_7 = _f2_122;
            float2 _f2_123 = make_float2(q[12], q[13]);
            float2 qp_3_7 = _f2_123;
            sq[0] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[0]);
            float2 _f2_124 = make_float2(sw_16_f32[6], sw_16_f32[7]);
            float2 v_4_7 = _f2_124;
            float2 _f2_125 = make_float2(q[14], q[15]);
            float2 qp_5_7 = _f2_125;
            sq[0] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[0]);
        }
        unsigned int sw_17[4];
        sw_17[0] = words[14 + woff_15];
        sw_17[1] = words[14 + woff_15 + 1];
        sw_17[2] = words[14 + woff_15 + 2];
        sw_17[3] = words[14 + woff_15 + 3];
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
            fsrc[36] = sw_17_f32[0];
            fsrc[37] = sw_17_f32[1];
            fsrc[38] = sw_17_f32[2];
            fsrc[39] = sw_17_f32[3];
            fsrc[40] = sw_17_f32[4];
            fsrc[41] = sw_17_f32[5];
            fsrc[42] = sw_17_f32[6];
            fsrc[43] = sw_17_f32[7];
        }
        {
            float2 _f2_132 = make_float2(sw_17_f32[0], sw_17_f32[1]);
            float2 v_10 = _f2_132;
            float2 _f2_133 = make_float2(q[8], q[9]);
            float2 qp_11 = _f2_133;
            sq[1] = fma_f32x2_rn_noftz(v_10, v_10, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_10, qp_11, dot[1]);
            float2 _f2_134 = make_float2(sw_17_f32[2], sw_17_f32[3]);
            float2 v_0_8 = _f2_134;
            float2 _f2_135 = make_float2(q[10], q[11]);
            float2 qp_1_8 = _f2_135;
            sq[1] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[1]);
            float2 _f2_136 = make_float2(sw_17_f32[4], sw_17_f32[5]);
            float2 v_2_8 = _f2_136;
            float2 _f2_137 = make_float2(q[12], q[13]);
            float2 qp_3_8 = _f2_137;
            sq[1] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[1]);
            float2 _f2_138 = make_float2(sw_17_f32[6], sw_17_f32[7]);
            float2 v_4_8 = _f2_138;
            float2 _f2_139 = make_float2(q[14], q[15]);
            float2 qp_5_8 = _f2_139;
            sq[1] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[1]);
        }
        unsigned int sw_18[4];
        sw_18[0] = words[28 + woff_15];
        sw_18[1] = words[28 + woff_15 + 1];
        sw_18[2] = words[28 + woff_15 + 2];
        sw_18[3] = words[28 + woff_15 + 3];
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
            fsrc[64] = sw_18_f32[0];
            fsrc[65] = sw_18_f32[1];
            fsrc[66] = sw_18_f32[2];
            fsrc[67] = sw_18_f32[3];
            fsrc[68] = sw_18_f32[4];
            fsrc[69] = sw_18_f32[5];
            fsrc[70] = sw_18_f32[6];
            fsrc[71] = sw_18_f32[7];
        }
        {
            float2 _f2_146 = make_float2(sw_18_f32[0], sw_18_f32[1]);
            float2 v_11 = _f2_146;
            float2 _f2_147 = make_float2(q[8], q[9]);
            float2 qp_12 = _f2_147;
            sq[2] = fma_f32x2_rn_noftz(v_11, v_11, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_11, qp_12, dot[2]);
            float2 _f2_148 = make_float2(sw_18_f32[2], sw_18_f32[3]);
            float2 v_0_9 = _f2_148;
            float2 _f2_149 = make_float2(q[10], q[11]);
            float2 qp_1_9 = _f2_149;
            sq[2] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_9, qp_1_9, dot[2]);
            float2 _f2_150 = make_float2(sw_18_f32[4], sw_18_f32[5]);
            float2 v_2_9 = _f2_150;
            float2 _f2_151 = make_float2(q[12], q[13]);
            float2 qp_3_9 = _f2_151;
            sq[2] = fma_f32x2_rn_noftz(v_2_9, v_2_9, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[2]);
            float2 _f2_152 = make_float2(sw_18_f32[6], sw_18_f32[7]);
            float2 v_4_9 = _f2_152;
            float2 _f2_153 = make_float2(q[14], q[15]);
            float2 qp_5_9 = _f2_153;
            sq[2] = fma_f32x2_rn_noftz(v_4_9, v_4_9, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_9, qp_5_9, dot[2]);
        }
        unsigned int sw_19[4];
        sw_19[0] = words[42 + woff_15];
        sw_19[1] = words[42 + woff_15 + 1];
        sw_19[2] = words[42 + woff_15 + 2];
        sw_19[3] = words[42 + woff_15 + 3];
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
            float2 _f2_160 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_12 = _f2_160;
            float2 _f2_161 = make_float2(q[8], q[9]);
            float2 qp_13 = _f2_161;
            sq[3] = fma_f32x2_rn_noftz(v_12, v_12, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_12, qp_13, dot[3]);
            float2 _f2_162 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_10 = _f2_162;
            float2 _f2_163 = make_float2(q[10], q[11]);
            float2 qp_1_10 = _f2_163;
            sq[3] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_10, qp_1_10, dot[3]);
            float2 _f2_164 = make_float2(sw_19_f32[4], sw_19_f32[5]);
            float2 v_2_10 = _f2_164;
            float2 _f2_165 = make_float2(q[12], q[13]);
            float2 qp_3_10 = _f2_165;
            sq[3] = fma_f32x2_rn_noftz(v_2_10, v_2_10, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[3]);
            float2 _f2_166 = make_float2(sw_19_f32[6], sw_19_f32[7]);
            float2 v_4_10 = _f2_166;
            float2 _f2_167 = make_float2(q[14], q[15]);
            float2 qp_5_10 = _f2_167;
            sq[3] = fma_f32x2_rn_noftz(v_4_10, v_4_10, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_10, qp_5_10, dot[3]);
        }
        unsigned int sw_20[4];
        sw_20[0] = words[56 + woff_15];
        sw_20[1] = words[56 + woff_15 + 1];
        sw_20[2] = words[56 + woff_15 + 2];
        sw_20[3] = words[56 + woff_15 + 3];
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
            float2 _f2_174 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_13 = _f2_174;
            float2 _f2_175 = make_float2(q[8], q[9]);
            float2 qp_14 = _f2_175;
            sq[4] = fma_f32x2_rn_noftz(v_13, v_13, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_13, qp_14, dot[4]);
            float2 _f2_176 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_0_11 = _f2_176;
            float2 _f2_177 = make_float2(q[10], q[11]);
            float2 qp_1_11 = _f2_177;
            sq[4] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_11, qp_1_11, dot[4]);
            float2 _f2_178 = make_float2(sw_20_f32[4], sw_20_f32[5]);
            float2 v_2_11 = _f2_178;
            float2 _f2_179 = make_float2(q[12], q[13]);
            float2 qp_3_11 = _f2_179;
            sq[4] = fma_f32x2_rn_noftz(v_2_11, v_2_11, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[4]);
            float2 _f2_180 = make_float2(sw_20_f32[6], sw_20_f32[7]);
            float2 v_4_11 = _f2_180;
            float2 _f2_181 = make_float2(q[14], q[15]);
            float2 qp_5_11 = _f2_181;
            sq[4] = fma_f32x2_rn_noftz(v_4_11, v_4_11, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_11, qp_5_11, dot[4]);
        }
        unsigned int sw_21[4];
        sw_21[0] = words[70 + woff_15];
        sw_21[1] = words[70 + woff_15 + 1];
        sw_21[2] = words[70 + woff_15 + 2];
        sw_21[3] = words[70 + woff_15 + 3];
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
            float2 _f2_188 = make_float2(sw_21_f32[0], sw_21_f32[1]);
            float2 v_14 = _f2_188;
            float2 _f2_189 = make_float2(q[8], q[9]);
            float2 qp_15 = _f2_189;
            sq[5] = fma_f32x2_rn_noftz(v_14, v_14, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_14, qp_15, dot[5]);
            float2 _f2_190 = make_float2(sw_21_f32[2], sw_21_f32[3]);
            float2 v_0_12 = _f2_190;
            float2 _f2_191 = make_float2(q[10], q[11]);
            float2 qp_1_12 = _f2_191;
            sq[5] = fma_f32x2_rn_noftz(v_0_12, v_0_12, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_12, qp_1_12, dot[5]);
            float2 _f2_192 = make_float2(sw_21_f32[4], sw_21_f32[5]);
            float2 v_2_12 = _f2_192;
            float2 _f2_193 = make_float2(q[12], q[13]);
            float2 qp_3_12 = _f2_193;
            sq[5] = fma_f32x2_rn_noftz(v_2_12, v_2_12, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_12, qp_3_12, dot[5]);
            float2 _f2_194 = make_float2(sw_21_f32[6], sw_21_f32[7]);
            float2 v_4_12 = _f2_194;
            float2 _f2_195 = make_float2(q[14], q[15]);
            float2 qp_5_12 = _f2_195;
            sq[5] = fma_f32x2_rn_noftz(v_4_12, v_4_12, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_12, qp_5_12, dot[5]);
        }
        unsigned int sw_22[4];
        sw_22[0] = words[84 + woff_15];
        sw_22[1] = words[84 + woff_15 + 1];
        sw_22[2] = words[84 + woff_15 + 2];
        sw_22[3] = words[84 + woff_15 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_22[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_15]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_22[0] = __as_u32(mixed_1);
            words[84 + woff_15] = sw_22[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_22[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_15 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_22[1] = __as_u32(mixed_2_1);
            words[84 + woff_15 + 1] = sw_22[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_22[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_15 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_22[2] = __as_u32(mixed_5_1);
            words[84 + woff_15 + 2] = sw_22[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_22[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_15 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_22[3] = __as_u32(mixed_8_1);
            words[84 + woff_15 + 3] = sw_22[3];
            {
                int4 _iv4 = make_int4(sw_22[0 + 0], sw_22[0 + 1], sw_22[0 + 2], sw_22[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_14)) + 0) = _iv4;
            }
        }
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
            float2 _f2_202 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_15 = _f2_202;
            float2 _f2_203 = make_float2(q[8], q[9]);
            float2 qp_16 = _f2_203;
            sq[6] = fma_f32x2_rn_noftz(v_15, v_15, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_15, qp_16, dot[6]);
            float2 _f2_204 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_0_13 = _f2_204;
            float2 _f2_205 = make_float2(q[10], q[11]);
            float2 qp_1_13 = _f2_205;
            sq[6] = fma_f32x2_rn_noftz(v_0_13, v_0_13, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_13, qp_1_13, dot[6]);
            float2 _f2_206 = make_float2(sw_22_f32[4], sw_22_f32[5]);
            float2 v_2_13 = _f2_206;
            float2 _f2_207 = make_float2(q[12], q[13]);
            float2 qp_3_13 = _f2_207;
            sq[6] = fma_f32x2_rn_noftz(v_2_13, v_2_13, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_13, qp_3_13, dot[6]);
            float2 _f2_208 = make_float2(sw_22_f32[6], sw_22_f32[7]);
            float2 v_4_13 = _f2_208;
            float2 _f2_209 = make_float2(q[14], q[15]);
            float2 qp_5_13 = _f2_209;
            sq[6] = fma_f32x2_rn_noftz(v_4_13, v_4_13, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_13, qp_5_13, dot[6]);
        }
        int base_23 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_24 = 8;
        unsigned int sw_25[4];
        sw_25[0] = words[woff_24];
        sw_25[1] = words[woff_24 + 1];
        sw_25[2] = words[woff_24 + 2];
        sw_25[3] = words[woff_24 + 3];
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
            fsrc[16] = sw_25_f32[0];
            fsrc[17] = sw_25_f32[1];
            fsrc[18] = sw_25_f32[2];
            fsrc[19] = sw_25_f32[3];
            fsrc[20] = sw_25_f32[4];
            fsrc[21] = sw_25_f32[5];
            fsrc[22] = sw_25_f32[6];
            fsrc[23] = sw_25_f32[7];
        }
        {
            float2 _f2_216 = make_float2(sw_25_f32[0], sw_25_f32[1]);
            float2 v_16 = _f2_216;
            float2 _f2_217 = make_float2(q[16], q[17]);
            float2 qp_17 = _f2_217;
            sq[0] = fma_f32x2_rn_noftz(v_16, v_16, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_16, qp_17, dot[0]);
            float2 _f2_218 = make_float2(sw_25_f32[2], sw_25_f32[3]);
            float2 v_0_14 = _f2_218;
            float2 _f2_219 = make_float2(q[18], q[19]);
            float2 qp_1_14 = _f2_219;
            sq[0] = fma_f32x2_rn_noftz(v_0_14, v_0_14, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_14, qp_1_14, dot[0]);
            float2 _f2_220 = make_float2(sw_25_f32[4], sw_25_f32[5]);
            float2 v_2_14 = _f2_220;
            float2 _f2_221 = make_float2(q[20], q[21]);
            float2 qp_3_14 = _f2_221;
            sq[0] = fma_f32x2_rn_noftz(v_2_14, v_2_14, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_14, qp_3_14, dot[0]);
            float2 _f2_222 = make_float2(sw_25_f32[6], sw_25_f32[7]);
            float2 v_4_14 = _f2_222;
            float2 _f2_223 = make_float2(q[22], q[23]);
            float2 qp_5_14 = _f2_223;
            sq[0] = fma_f32x2_rn_noftz(v_4_14, v_4_14, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_14, qp_5_14, dot[0]);
        }
        unsigned int sw_26[4];
        sw_26[0] = words[14 + woff_24];
        sw_26[1] = words[14 + woff_24 + 1];
        sw_26[2] = words[14 + woff_24 + 2];
        sw_26[3] = words[14 + woff_24 + 3];
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
            fsrc[44] = sw_26_f32[0];
            fsrc[45] = sw_26_f32[1];
            fsrc[46] = sw_26_f32[2];
            fsrc[47] = sw_26_f32[3];
            fsrc[48] = sw_26_f32[4];
            fsrc[49] = sw_26_f32[5];
            fsrc[50] = sw_26_f32[6];
            fsrc[51] = sw_26_f32[7];
        }
        {
            float2 _f2_230 = make_float2(sw_26_f32[0], sw_26_f32[1]);
            float2 v_17 = _f2_230;
            float2 _f2_231 = make_float2(q[16], q[17]);
            float2 qp_18 = _f2_231;
            sq[1] = fma_f32x2_rn_noftz(v_17, v_17, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_17, qp_18, dot[1]);
            float2 _f2_232 = make_float2(sw_26_f32[2], sw_26_f32[3]);
            float2 v_0_15 = _f2_232;
            float2 _f2_233 = make_float2(q[18], q[19]);
            float2 qp_1_15 = _f2_233;
            sq[1] = fma_f32x2_rn_noftz(v_0_15, v_0_15, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_15, qp_1_15, dot[1]);
            float2 _f2_234 = make_float2(sw_26_f32[4], sw_26_f32[5]);
            float2 v_2_15 = _f2_234;
            float2 _f2_235 = make_float2(q[20], q[21]);
            float2 qp_3_15 = _f2_235;
            sq[1] = fma_f32x2_rn_noftz(v_2_15, v_2_15, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_15, qp_3_15, dot[1]);
            float2 _f2_236 = make_float2(sw_26_f32[6], sw_26_f32[7]);
            float2 v_4_15 = _f2_236;
            float2 _f2_237 = make_float2(q[22], q[23]);
            float2 qp_5_15 = _f2_237;
            sq[1] = fma_f32x2_rn_noftz(v_4_15, v_4_15, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_15, qp_5_15, dot[1]);
        }
        unsigned int sw_27[4];
        sw_27[0] = words[28 + woff_24];
        sw_27[1] = words[28 + woff_24 + 1];
        sw_27[2] = words[28 + woff_24 + 2];
        sw_27[3] = words[28 + woff_24 + 3];
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
            fsrc[72] = sw_27_f32[0];
            fsrc[73] = sw_27_f32[1];
            fsrc[74] = sw_27_f32[2];
            fsrc[75] = sw_27_f32[3];
            fsrc[76] = sw_27_f32[4];
            fsrc[77] = sw_27_f32[5];
            fsrc[78] = sw_27_f32[6];
            fsrc[79] = sw_27_f32[7];
        }
        {
            float2 _f2_244 = make_float2(sw_27_f32[0], sw_27_f32[1]);
            float2 v_18 = _f2_244;
            float2 _f2_245 = make_float2(q[16], q[17]);
            float2 qp_19 = _f2_245;
            sq[2] = fma_f32x2_rn_noftz(v_18, v_18, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_18, qp_19, dot[2]);
            float2 _f2_246 = make_float2(sw_27_f32[2], sw_27_f32[3]);
            float2 v_0_16 = _f2_246;
            float2 _f2_247 = make_float2(q[18], q[19]);
            float2 qp_1_16 = _f2_247;
            sq[2] = fma_f32x2_rn_noftz(v_0_16, v_0_16, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_16, qp_1_16, dot[2]);
            float2 _f2_248 = make_float2(sw_27_f32[4], sw_27_f32[5]);
            float2 v_2_16 = _f2_248;
            float2 _f2_249 = make_float2(q[20], q[21]);
            float2 qp_3_16 = _f2_249;
            sq[2] = fma_f32x2_rn_noftz(v_2_16, v_2_16, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_16, qp_3_16, dot[2]);
            float2 _f2_250 = make_float2(sw_27_f32[6], sw_27_f32[7]);
            float2 v_4_16 = _f2_250;
            float2 _f2_251 = make_float2(q[22], q[23]);
            float2 qp_5_16 = _f2_251;
            sq[2] = fma_f32x2_rn_noftz(v_4_16, v_4_16, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_16, qp_5_16, dot[2]);
        }
        unsigned int sw_28[4];
        sw_28[0] = words[42 + woff_24];
        sw_28[1] = words[42 + woff_24 + 1];
        sw_28[2] = words[42 + woff_24 + 2];
        sw_28[3] = words[42 + woff_24 + 3];
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
            float2 _f2_258 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_19 = _f2_258;
            float2 _f2_259 = make_float2(q[16], q[17]);
            float2 qp_20 = _f2_259;
            sq[3] = fma_f32x2_rn_noftz(v_19, v_19, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_19, qp_20, dot[3]);
            float2 _f2_260 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_0_17 = _f2_260;
            float2 _f2_261 = make_float2(q[18], q[19]);
            float2 qp_1_17 = _f2_261;
            sq[3] = fma_f32x2_rn_noftz(v_0_17, v_0_17, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_17, qp_1_17, dot[3]);
            float2 _f2_262 = make_float2(sw_28_f32[4], sw_28_f32[5]);
            float2 v_2_17 = _f2_262;
            float2 _f2_263 = make_float2(q[20], q[21]);
            float2 qp_3_17 = _f2_263;
            sq[3] = fma_f32x2_rn_noftz(v_2_17, v_2_17, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_17, qp_3_17, dot[3]);
            float2 _f2_264 = make_float2(sw_28_f32[6], sw_28_f32[7]);
            float2 v_4_17 = _f2_264;
            float2 _f2_265 = make_float2(q[22], q[23]);
            float2 qp_5_17 = _f2_265;
            sq[3] = fma_f32x2_rn_noftz(v_4_17, v_4_17, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_17, qp_5_17, dot[3]);
        }
        unsigned int sw_29[4];
        sw_29[0] = words[56 + woff_24];
        sw_29[1] = words[56 + woff_24 + 1];
        sw_29[2] = words[56 + woff_24 + 2];
        sw_29[3] = words[56 + woff_24 + 3];
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
            float2 _f2_272 = make_float2(sw_29_f32[0], sw_29_f32[1]);
            float2 v_20 = _f2_272;
            float2 _f2_273 = make_float2(q[16], q[17]);
            float2 qp_21 = _f2_273;
            sq[4] = fma_f32x2_rn_noftz(v_20, v_20, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_20, qp_21, dot[4]);
            float2 _f2_274 = make_float2(sw_29_f32[2], sw_29_f32[3]);
            float2 v_0_18 = _f2_274;
            float2 _f2_275 = make_float2(q[18], q[19]);
            float2 qp_1_18 = _f2_275;
            sq[4] = fma_f32x2_rn_noftz(v_0_18, v_0_18, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_18, qp_1_18, dot[4]);
            float2 _f2_276 = make_float2(sw_29_f32[4], sw_29_f32[5]);
            float2 v_2_18 = _f2_276;
            float2 _f2_277 = make_float2(q[20], q[21]);
            float2 qp_3_18 = _f2_277;
            sq[4] = fma_f32x2_rn_noftz(v_2_18, v_2_18, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_18, qp_3_18, dot[4]);
            float2 _f2_278 = make_float2(sw_29_f32[6], sw_29_f32[7]);
            float2 v_4_18 = _f2_278;
            float2 _f2_279 = make_float2(q[22], q[23]);
            float2 qp_5_18 = _f2_279;
            sq[4] = fma_f32x2_rn_noftz(v_4_18, v_4_18, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_18, qp_5_18, dot[4]);
        }
        unsigned int sw_30[4];
        sw_30[0] = words[70 + woff_24];
        sw_30[1] = words[70 + woff_24 + 1];
        sw_30[2] = words[70 + woff_24 + 2];
        sw_30[3] = words[70 + woff_24 + 3];
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
            float2 _f2_286 = make_float2(sw_30_f32[0], sw_30_f32[1]);
            float2 v_21 = _f2_286;
            float2 _f2_287 = make_float2(q[16], q[17]);
            float2 qp_22 = _f2_287;
            sq[5] = fma_f32x2_rn_noftz(v_21, v_21, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_21, qp_22, dot[5]);
            float2 _f2_288 = make_float2(sw_30_f32[2], sw_30_f32[3]);
            float2 v_0_19 = _f2_288;
            float2 _f2_289 = make_float2(q[18], q[19]);
            float2 qp_1_19 = _f2_289;
            sq[5] = fma_f32x2_rn_noftz(v_0_19, v_0_19, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_19, qp_1_19, dot[5]);
            float2 _f2_290 = make_float2(sw_30_f32[4], sw_30_f32[5]);
            float2 v_2_19 = _f2_290;
            float2 _f2_291 = make_float2(q[20], q[21]);
            float2 qp_3_19 = _f2_291;
            sq[5] = fma_f32x2_rn_noftz(v_2_19, v_2_19, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_19, qp_3_19, dot[5]);
            float2 _f2_292 = make_float2(sw_30_f32[6], sw_30_f32[7]);
            float2 v_4_19 = _f2_292;
            float2 _f2_293 = make_float2(q[22], q[23]);
            float2 qp_5_19 = _f2_293;
            sq[5] = fma_f32x2_rn_noftz(v_4_19, v_4_19, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_19, qp_5_19, dot[5]);
        }
        unsigned int sw_31[4];
        sw_31[0] = words[84 + woff_24];
        sw_31[1] = words[84 + woff_24 + 1];
        sw_31[2] = words[84 + woff_24 + 2];
        sw_31[3] = words[84 + woff_24 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_31[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_24]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_31[0] = __as_u32(mixed_3);
            words[84 + woff_24] = sw_31[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_31[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_24 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_31[1] = __as_u32(mixed_2_2);
            words[84 + woff_24 + 1] = sw_31[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_31[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_24 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_31[2] = __as_u32(mixed_5_2);
            words[84 + woff_24 + 2] = sw_31[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_31[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_24 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_31[3] = __as_u32(mixed_8_2);
            words[84 + woff_24 + 3] = sw_31[3];
            {
                int4 _iv4 = make_int4(sw_31[0 + 0], sw_31[0 + 1], sw_31[0 + 2], sw_31[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_23)) + 0) = _iv4;
            }
        }
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
            float2 _f2_300 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_22 = _f2_300;
            float2 _f2_301 = make_float2(q[16], q[17]);
            float2 qp_23 = _f2_301;
            sq[6] = fma_f32x2_rn_noftz(v_22, v_22, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_22, qp_23, dot[6]);
            float2 _f2_302 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_0_20 = _f2_302;
            float2 _f2_303 = make_float2(q[18], q[19]);
            float2 qp_1_20 = _f2_303;
            sq[6] = fma_f32x2_rn_noftz(v_0_20, v_0_20, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_20, qp_1_20, dot[6]);
            float2 _f2_304 = make_float2(sw_31_f32[4], sw_31_f32[5]);
            float2 v_2_20 = _f2_304;
            float2 _f2_305 = make_float2(q[20], q[21]);
            float2 qp_3_20 = _f2_305;
            sq[6] = fma_f32x2_rn_noftz(v_2_20, v_2_20, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_20, qp_3_20, dot[6]);
            float2 _f2_306 = make_float2(sw_31_f32[6], sw_31_f32[7]);
            float2 v_4_20 = _f2_306;
            float2 _f2_307 = make_float2(q[22], q[23]);
            float2 qp_5_20 = _f2_307;
            sq[6] = fma_f32x2_rn_noftz(v_4_20, v_4_20, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_20, qp_5_20, dot[6]);
        }
        int base_32 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_33 = 12;
        unsigned int sw_34[4];
        sw_34[0] = words[woff_33];
        sw_34[1] = words[woff_33 + 1];
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
            fsrc[24] = sw_34_f32[0];
            fsrc[25] = sw_34_f32[1];
            fsrc[26] = sw_34_f32[2];
            fsrc[27] = sw_34_f32[3];
        }
        {
            float2 _f2_308 = make_float2(sw_34_f32[0], sw_34_f32[1]);
            float2 v_23 = _f2_308;
            sq[0] = fma_f32x2_rn_noftz(v_23, v_23, sq[0]);
            float2 _f2_309 = make_float2(sw_34_f32[2], sw_34_f32[3]);
            float2 v_0_21 = _f2_309;
            sq[0] = fma_f32x2_rn_noftz(v_0_21, v_0_21, sq[0]);
            float2 _f2_310 = make_float2(sw_34_f32[0], sw_34_f32[1]);
            float2 v_1_1 = _f2_310;
            float2 _f2_311 = make_float2(q[24], q[25]);
            float2 qp_24 = _f2_311;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_24, dot[0]);
            float2 _f2_312 = make_float2(sw_34_f32[2], sw_34_f32[3]);
            float2 v_2_21 = _f2_312;
            float2 _f2_313 = make_float2(q[26], q[27]);
            float2 qp_3_21 = _f2_313;
            dot[0] = fma_f32x2_rn_noftz(v_2_21, qp_3_21, dot[0]);
        }
        unsigned int sw_35[4];
        sw_35[0] = words[14 + woff_33];
        sw_35[1] = words[14 + woff_33 + 1];
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
            fsrc[52] = sw_35_f32[0];
            fsrc[53] = sw_35_f32[1];
            fsrc[54] = sw_35_f32[2];
            fsrc[55] = sw_35_f32[3];
        }
        {
            float2 _f2_322 = make_float2(sw_35_f32[0], sw_35_f32[1]);
            float2 v_24 = _f2_322;
            sq[1] = fma_f32x2_rn_noftz(v_24, v_24, sq[1]);
            float2 _f2_323 = make_float2(sw_35_f32[2], sw_35_f32[3]);
            float2 v_0_22 = _f2_323;
            sq[1] = fma_f32x2_rn_noftz(v_0_22, v_0_22, sq[1]);
            float2 _f2_324 = make_float2(sw_35_f32[0], sw_35_f32[1]);
            float2 v_1_2 = _f2_324;
            float2 _f2_325 = make_float2(q[24], q[25]);
            float2 qp_25 = _f2_325;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_25, dot[1]);
            float2 _f2_326 = make_float2(sw_35_f32[2], sw_35_f32[3]);
            float2 v_2_22 = _f2_326;
            float2 _f2_327 = make_float2(q[26], q[27]);
            float2 qp_3_22 = _f2_327;
            dot[1] = fma_f32x2_rn_noftz(v_2_22, qp_3_22, dot[1]);
        }
        unsigned int sw_36[4];
        sw_36[0] = words[28 + woff_33];
        sw_36[1] = words[28 + woff_33 + 1];
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
            fsrc[80] = sw_36_f32[0];
            fsrc[81] = sw_36_f32[1];
            fsrc[82] = sw_36_f32[2];
            fsrc[83] = sw_36_f32[3];
        }
        {
            float2 _f2_336 = make_float2(sw_36_f32[0], sw_36_f32[1]);
            float2 v_25 = _f2_336;
            sq[2] = fma_f32x2_rn_noftz(v_25, v_25, sq[2]);
            float2 _f2_337 = make_float2(sw_36_f32[2], sw_36_f32[3]);
            float2 v_0_23 = _f2_337;
            sq[2] = fma_f32x2_rn_noftz(v_0_23, v_0_23, sq[2]);
            float2 _f2_338 = make_float2(sw_36_f32[0], sw_36_f32[1]);
            float2 v_1_3 = _f2_338;
            float2 _f2_339 = make_float2(q[24], q[25]);
            float2 qp_26 = _f2_339;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_26, dot[2]);
            float2 _f2_340 = make_float2(sw_36_f32[2], sw_36_f32[3]);
            float2 v_2_23 = _f2_340;
            float2 _f2_341 = make_float2(q[26], q[27]);
            float2 qp_3_23 = _f2_341;
            dot[2] = fma_f32x2_rn_noftz(v_2_23, qp_3_23, dot[2]);
        }
        unsigned int sw_37[4];
        sw_37[0] = words[42 + woff_33];
        sw_37[1] = words[42 + woff_33 + 1];
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
            float2 _f2_350 = make_float2(sw_37_f32[0], sw_37_f32[1]);
            float2 v_26 = _f2_350;
            sq[3] = fma_f32x2_rn_noftz(v_26, v_26, sq[3]);
            float2 _f2_351 = make_float2(sw_37_f32[2], sw_37_f32[3]);
            float2 v_0_24 = _f2_351;
            sq[3] = fma_f32x2_rn_noftz(v_0_24, v_0_24, sq[3]);
            float2 _f2_352 = make_float2(sw_37_f32[0], sw_37_f32[1]);
            float2 v_1_4 = _f2_352;
            float2 _f2_353 = make_float2(q[24], q[25]);
            float2 qp_27 = _f2_353;
            dot[3] = fma_f32x2_rn_noftz(v_1_4, qp_27, dot[3]);
            float2 _f2_354 = make_float2(sw_37_f32[2], sw_37_f32[3]);
            float2 v_2_24 = _f2_354;
            float2 _f2_355 = make_float2(q[26], q[27]);
            float2 qp_3_24 = _f2_355;
            dot[3] = fma_f32x2_rn_noftz(v_2_24, qp_3_24, dot[3]);
        }
        unsigned int sw_38[4];
        sw_38[0] = words[56 + woff_33];
        sw_38[1] = words[56 + woff_33 + 1];
        float sw_38_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_38_f32[_pair * 2])[0]), "=f"((&sw_38_f32[_pair * 2])[1])
                : "r"(sw_38[_pair]));
        }
        {
            float2 _f2_364 = make_float2(sw_38_f32[0], sw_38_f32[1]);
            float2 v_27 = _f2_364;
            sq[4] = fma_f32x2_rn_noftz(v_27, v_27, sq[4]);
            float2 _f2_365 = make_float2(sw_38_f32[2], sw_38_f32[3]);
            float2 v_0_25 = _f2_365;
            sq[4] = fma_f32x2_rn_noftz(v_0_25, v_0_25, sq[4]);
            float2 _f2_366 = make_float2(sw_38_f32[0], sw_38_f32[1]);
            float2 v_1_5 = _f2_366;
            float2 _f2_367 = make_float2(q[24], q[25]);
            float2 qp_28 = _f2_367;
            dot[4] = fma_f32x2_rn_noftz(v_1_5, qp_28, dot[4]);
            float2 _f2_368 = make_float2(sw_38_f32[2], sw_38_f32[3]);
            float2 v_2_25 = _f2_368;
            float2 _f2_369 = make_float2(q[26], q[27]);
            float2 qp_3_25 = _f2_369;
            dot[4] = fma_f32x2_rn_noftz(v_2_25, qp_3_25, dot[4]);
        }
        unsigned int sw_39[4];
        sw_39[0] = words[70 + woff_33];
        sw_39[1] = words[70 + woff_33 + 1];
        float sw_39_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&sw_39_f32[_pair * 2])[0]), "=f"((&sw_39_f32[_pair * 2])[1])
                : "r"(sw_39[_pair]));
        }
        {
            float2 _f2_378 = make_float2(sw_39_f32[0], sw_39_f32[1]);
            float2 v_28 = _f2_378;
            sq[5] = fma_f32x2_rn_noftz(v_28, v_28, sq[5]);
            float2 _f2_379 = make_float2(sw_39_f32[2], sw_39_f32[3]);
            float2 v_0_26 = _f2_379;
            sq[5] = fma_f32x2_rn_noftz(v_0_26, v_0_26, sq[5]);
            float2 _f2_380 = make_float2(sw_39_f32[0], sw_39_f32[1]);
            float2 v_1_6 = _f2_380;
            float2 _f2_381 = make_float2(q[24], q[25]);
            float2 qp_29 = _f2_381;
            dot[5] = fma_f32x2_rn_noftz(v_1_6, qp_29, dot[5]);
            float2 _f2_382 = make_float2(sw_39_f32[2], sw_39_f32[3]);
            float2 v_2_26 = _f2_382;
            float2 _f2_383 = make_float2(q[26], q[27]);
            float2 qp_3_26 = _f2_383;
            dot[5] = fma_f32x2_rn_noftz(v_2_26, qp_3_26, dot[5]);
        }
        unsigned int sw_40[4];
        sw_40[0] = words[84 + woff_33];
        sw_40[1] = words[84 + woff_33 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_40[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_33]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_40[0] = __as_u32(mixed_4);
            words[84 + woff_33] = sw_40[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_40[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_33 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_40[1] = __as_u32(mixed_2_3);
            words[84 + woff_33 + 1] = sw_40[1];
            {
                int2 _iv2 = make_int2(sw_40[0 + 0], sw_40[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_32)) + 0) = _iv2;
            }
        }
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
            float2 _f2_392 = make_float2(sw_40_f32[0], sw_40_f32[1]);
            float2 v_29 = _f2_392;
            sq[6] = fma_f32x2_rn_noftz(v_29, v_29, sq[6]);
            float2 _f2_393 = make_float2(sw_40_f32[2], sw_40_f32[3]);
            float2 v_0_27 = _f2_393;
            sq[6] = fma_f32x2_rn_noftz(v_0_27, v_0_27, sq[6]);
            float2 _f2_394 = make_float2(sw_40_f32[0], sw_40_f32[1]);
            float2 v_1_7 = _f2_394;
            float2 _f2_395 = make_float2(q[24], q[25]);
            float2 qp_30 = _f2_395;
            dot[6] = fma_f32x2_rn_noftz(v_1_7, qp_30, dot[6]);
            float2 _f2_396 = make_float2(sw_40_f32[2], sw_40_f32[3]);
            float2 v_2_27 = _f2_396;
            float2 _f2_397 = make_float2(q[26], q[27]);
            float2 qp_3_27 = _f2_397;
            dot[6] = fma_f32x2_rn_noftz(v_2_27, qp_3_27, dot[6]);
        }
        float2 pairs[7];
        float2 _f2_406 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_406;
        float2 _f2_407 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_407;
        float2 _f2_408 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_408;
        float2 _f2_409 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_409;
        float2 _f2_410 = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[4] = _f2_410;
        float2 _f2_411 = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[5] = _f2_411;
        float2 _f2_412 = make_float2(sq[6].x + sq[6].y, dot[6].x + dot[6].y);
        pairs[6] = _f2_412;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_413 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_413;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_41 = 0;
        bits_41 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_41, 16);
        unsigned long long peerbits_42 = _shfl_xor_1;
        float2 _f2_414 = make_float2(0.0f, 0.0f);
        float2 peer_43 = _f2_414;
        peer_43 = reinterpret_cast<float2*>(&peerbits_42)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_43);
        unsigned long long bits_44 = 0;
        bits_44 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_44, 16);
        unsigned long long peerbits_45 = _shfl_xor_2;
        float2 _f2_415 = make_float2(0.0f, 0.0f);
        float2 peer_46 = _f2_415;
        peer_46 = reinterpret_cast<float2*>(&peerbits_45)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_46);
        unsigned long long bits_47 = 0;
        bits_47 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_47, 16);
        unsigned long long peerbits_48 = _shfl_xor_3;
        float2 _f2_416 = make_float2(0.0f, 0.0f);
        float2 peer_49 = _f2_416;
        peer_49 = reinterpret_cast<float2*>(&peerbits_48)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_49);
        unsigned long long bits_50 = 0;
        bits_50 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_50, 16);
        unsigned long long peerbits_51 = _shfl_xor_4;
        float2 _f2_417 = make_float2(0.0f, 0.0f);
        float2 peer_52 = _f2_417;
        peer_52 = reinterpret_cast<float2*>(&peerbits_51)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_52);
        unsigned long long bits_53 = 0;
        bits_53 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_53, 16);
        unsigned long long peerbits_54 = _shfl_xor_5;
        float2 _f2_418 = make_float2(0.0f, 0.0f);
        float2 peer_55 = _f2_418;
        peer_55 = reinterpret_cast<float2*>(&peerbits_54)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_55);
        unsigned long long bits_56 = 0;
        bits_56 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_56, 16);
        unsigned long long peerbits_57 = _shfl_xor_6;
        float2 _f2_419 = make_float2(0.0f, 0.0f);
        float2 peer_58 = _f2_419;
        peer_58 = reinterpret_cast<float2*>(&peerbits_57)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_58);
        unsigned long long bits_59 = 0;
        bits_59 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_59, 8);
        unsigned long long peerbits_60 = _shfl_xor_7;
        float2 _f2_420 = make_float2(0.0f, 0.0f);
        float2 peer_61 = _f2_420;
        peer_61 = reinterpret_cast<float2*>(&peerbits_60)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_61);
        unsigned long long bits_62 = 0;
        bits_62 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_62, 8);
        unsigned long long peerbits_63 = _shfl_xor_8;
        float2 _f2_421 = make_float2(0.0f, 0.0f);
        float2 peer_64 = _f2_421;
        peer_64 = reinterpret_cast<float2*>(&peerbits_63)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_64);
        unsigned long long bits_65 = 0;
        bits_65 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_65, 8);
        unsigned long long peerbits_66 = _shfl_xor_9;
        float2 _f2_422 = make_float2(0.0f, 0.0f);
        float2 peer_67 = _f2_422;
        peer_67 = reinterpret_cast<float2*>(&peerbits_66)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_67);
        unsigned long long bits_68 = 0;
        bits_68 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_68, 8);
        unsigned long long peerbits_69 = _shfl_xor_10;
        float2 _f2_423 = make_float2(0.0f, 0.0f);
        float2 peer_70 = _f2_423;
        peer_70 = reinterpret_cast<float2*>(&peerbits_69)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_70);
        unsigned long long bits_71 = 0;
        bits_71 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_71, 8);
        unsigned long long peerbits_72 = _shfl_xor_11;
        float2 _f2_424 = make_float2(0.0f, 0.0f);
        float2 peer_73 = _f2_424;
        peer_73 = reinterpret_cast<float2*>(&peerbits_72)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_73);
        unsigned long long bits_74 = 0;
        bits_74 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_74, 8);
        unsigned long long peerbits_75 = _shfl_xor_12;
        float2 _f2_425 = make_float2(0.0f, 0.0f);
        float2 peer_76 = _f2_425;
        peer_76 = reinterpret_cast<float2*>(&peerbits_75)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_76);
        unsigned long long bits_77 = 0;
        bits_77 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_77, 8);
        unsigned long long peerbits_78 = _shfl_xor_13;
        float2 _f2_426 = make_float2(0.0f, 0.0f);
        float2 peer_79 = _f2_426;
        peer_79 = reinterpret_cast<float2*>(&peerbits_78)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_79);
        unsigned long long bits_80 = 0;
        bits_80 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_80, 4);
        unsigned long long peerbits_81 = _shfl_xor_14;
        float2 _f2_427 = make_float2(0.0f, 0.0f);
        float2 peer_82 = _f2_427;
        peer_82 = reinterpret_cast<float2*>(&peerbits_81)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_82);
        unsigned long long bits_83 = 0;
        bits_83 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_83, 4);
        unsigned long long peerbits_84 = _shfl_xor_15;
        float2 _f2_428 = make_float2(0.0f, 0.0f);
        float2 peer_85 = _f2_428;
        peer_85 = reinterpret_cast<float2*>(&peerbits_84)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_85);
        unsigned long long bits_86 = 0;
        bits_86 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_86, 4);
        unsigned long long peerbits_87 = _shfl_xor_16;
        float2 _f2_429 = make_float2(0.0f, 0.0f);
        float2 peer_88 = _f2_429;
        peer_88 = reinterpret_cast<float2*>(&peerbits_87)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_88);
        unsigned long long bits_89 = 0;
        bits_89 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_89, 4);
        unsigned long long peerbits_90 = _shfl_xor_17;
        float2 _f2_430 = make_float2(0.0f, 0.0f);
        float2 peer_91 = _f2_430;
        peer_91 = reinterpret_cast<float2*>(&peerbits_90)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_91);
        unsigned long long bits_92 = 0;
        bits_92 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_92, 4);
        unsigned long long peerbits_93 = _shfl_xor_18;
        float2 _f2_431 = make_float2(0.0f, 0.0f);
        float2 peer_94 = _f2_431;
        peer_94 = reinterpret_cast<float2*>(&peerbits_93)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_94);
        unsigned long long bits_95 = 0;
        bits_95 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_95, 4);
        unsigned long long peerbits_96 = _shfl_xor_19;
        float2 _f2_432 = make_float2(0.0f, 0.0f);
        float2 peer_97 = _f2_432;
        peer_97 = reinterpret_cast<float2*>(&peerbits_96)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_97);
        unsigned long long bits_98 = 0;
        bits_98 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, bits_98, 4);
        unsigned long long peerbits_99 = _shfl_xor_20;
        float2 _f2_433 = make_float2(0.0f, 0.0f);
        float2 peer_100 = _f2_433;
        peer_100 = reinterpret_cast<float2*>(&peerbits_99)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_100);
        unsigned long long bits_101 = 0;
        bits_101 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, bits_101, 2);
        unsigned long long peerbits_102 = _shfl_xor_21;
        float2 _f2_434 = make_float2(0.0f, 0.0f);
        float2 peer_103 = _f2_434;
        peer_103 = reinterpret_cast<float2*>(&peerbits_102)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_103);
        unsigned long long bits_104 = 0;
        bits_104 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, bits_104, 2);
        unsigned long long peerbits_105 = _shfl_xor_22;
        float2 _f2_435 = make_float2(0.0f, 0.0f);
        float2 peer_106 = _f2_435;
        peer_106 = reinterpret_cast<float2*>(&peerbits_105)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_106);
        unsigned long long bits_107 = 0;
        bits_107 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, bits_107, 2);
        unsigned long long peerbits_108 = _shfl_xor_23;
        float2 _f2_436 = make_float2(0.0f, 0.0f);
        float2 peer_109 = _f2_436;
        peer_109 = reinterpret_cast<float2*>(&peerbits_108)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_109);
        unsigned long long bits_110 = 0;
        bits_110 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, bits_110, 2);
        unsigned long long peerbits_111 = _shfl_xor_24;
        float2 _f2_437 = make_float2(0.0f, 0.0f);
        float2 peer_112 = _f2_437;
        peer_112 = reinterpret_cast<float2*>(&peerbits_111)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_112);
        unsigned long long bits_113 = 0;
        bits_113 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, bits_113, 2);
        unsigned long long peerbits_114 = _shfl_xor_25;
        float2 _f2_438 = make_float2(0.0f, 0.0f);
        float2 peer_115 = _f2_438;
        peer_115 = reinterpret_cast<float2*>(&peerbits_114)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_115);
        unsigned long long bits_116 = 0;
        bits_116 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, bits_116, 2);
        unsigned long long peerbits_117 = _shfl_xor_26;
        float2 _f2_439 = make_float2(0.0f, 0.0f);
        float2 peer_118 = _f2_439;
        peer_118 = reinterpret_cast<float2*>(&peerbits_117)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_118);
        unsigned long long bits_119 = 0;
        bits_119 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, bits_119, 2);
        unsigned long long peerbits_120 = _shfl_xor_27;
        float2 _f2_440 = make_float2(0.0f, 0.0f);
        float2 peer_121 = _f2_440;
        peer_121 = reinterpret_cast<float2*>(&peerbits_120)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_121);
        unsigned long long bits_122 = 0;
        bits_122 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, bits_122, 1);
        unsigned long long peerbits_123 = _shfl_xor_28;
        float2 _f2_441 = make_float2(0.0f, 0.0f);
        float2 peer_124 = _f2_441;
        peer_124 = reinterpret_cast<float2*>(&peerbits_123)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_124);
        unsigned long long bits_125 = 0;
        bits_125 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, bits_125, 1);
        unsigned long long peerbits_126 = _shfl_xor_29;
        float2 _f2_442 = make_float2(0.0f, 0.0f);
        float2 peer_127 = _f2_442;
        peer_127 = reinterpret_cast<float2*>(&peerbits_126)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_127);
        unsigned long long bits_128 = 0;
        bits_128 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, bits_128, 1);
        unsigned long long peerbits_129 = _shfl_xor_30;
        float2 _f2_443 = make_float2(0.0f, 0.0f);
        float2 peer_130 = _f2_443;
        peer_130 = reinterpret_cast<float2*>(&peerbits_129)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_130);
        unsigned long long bits_131 = 0;
        bits_131 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, bits_131, 1);
        unsigned long long peerbits_132 = _shfl_xor_31;
        float2 _f2_444 = make_float2(0.0f, 0.0f);
        float2 peer_133 = _f2_444;
        peer_133 = reinterpret_cast<float2*>(&peerbits_132)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_133);
        unsigned long long bits_134 = 0;
        bits_134 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, bits_134, 1);
        unsigned long long peerbits_135 = _shfl_xor_32;
        float2 _f2_445 = make_float2(0.0f, 0.0f);
        float2 peer_136 = _f2_445;
        peer_136 = reinterpret_cast<float2*>(&peerbits_135)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_136);
        unsigned long long bits_137 = 0;
        bits_137 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, bits_137, 1);
        unsigned long long peerbits_138 = _shfl_xor_33;
        float2 _f2_446 = make_float2(0.0f, 0.0f);
        float2 peer_139 = _f2_446;
        peer_139 = reinterpret_cast<float2*>(&peerbits_138)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_139);
        unsigned long long bits_140 = 0;
        bits_140 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, bits_140, 1);
        unsigned long long peerbits_141 = _shfl_xor_34;
        float2 _f2_447 = make_float2(0.0f, 0.0f);
        float2 peer_142 = _f2_447;
        peer_142 = reinterpret_cast<float2*>(&peerbits_141)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_142);
        if (lane == 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(stats_addr + (unsigned int)(warp_0 * 7 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_0), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(stats_addr + (unsigned int)((warp_0 * 7 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_1), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_2), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_3), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_4) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_4), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_5;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_5) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_5), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_6) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_6), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_7;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_7) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_7), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_8) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_8), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_9) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_9), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_10) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_10), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_11) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_11), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_12;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_12) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 6) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_12), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_13;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_13) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 6) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_13), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_14;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_14) : "r"(stats_addr + (unsigned int)(warp_0 * 7 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_14), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_15;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_15) : "r"(stats_addr + (unsigned int)((warp_0 * 7 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_15), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_16) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_16), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_17;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_17) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_17), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_18) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_18), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_19;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_19) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_19), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_20) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_20), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_21;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_21) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_21), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_22) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_22), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_23;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_23) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_23), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_24;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_24) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_24), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_25;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_25) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_25), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_26;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_26) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 6) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_26), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_27;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_27) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 6) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_27), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_28;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_28) : "r"(stats_addr + (unsigned int)(warp_0 * 7 * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_28), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_29;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_29) : "r"(stats_addr + (unsigned int)((warp_0 * 7 * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_29), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_30;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_30) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 1) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_30), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_31;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_31) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 1) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_31), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_32;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_32) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 2) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_32), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_33;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_33) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 2) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_33), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_34;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_34) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 3) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_34), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_35;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_35) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 3) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_35), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_36;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_36) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 4) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_36), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_37;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_37) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 4) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_37), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_38;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_38) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 5) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_38), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_39;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_39) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 5) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_39), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_40;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_40) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 6) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_40), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_41;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_41) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 6) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_41), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_42;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_42) : "r"(stats_addr + (unsigned int)(warp_0 * 7 * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_42), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_43;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_43) : "r"(stats_addr + (unsigned int)((warp_0 * 7 * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_43), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_44;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_44) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 1) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_44), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_45;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_45) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 1) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_45), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_46;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_46) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 2) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_46), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_47;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_47) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 2) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_47), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_48;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_48) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 3) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_48), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_49;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_49) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 3) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_49), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_50;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_50) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 4) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_50), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_51;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_51) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 4) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_51), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_52;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_52) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 5) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_52), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_53;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_53) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 5) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_53), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_54;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_54) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 6) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_54), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_55;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_55) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 6) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_55), "f"(pairs[6].y) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 7) {
            total_sq = stats[(stat_w * 7 + stat_n) * 2];
            total_dot = stats[(stat_w * 7 + stat_n) * 2 + 1];
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
        if (stat_n < 7 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[7];
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
        if (stat_n < 3) {
            total_sq2 = stats[(stat_w * 7 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 7 + 4 + stat_n) * 2 + 1];
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
        if (stat_n < 3 && stat_w == 0) {
            float _rsqrt_1 = rsqrtf(total_sq2 / 7168.0f + eps);
            float sigma2 = _rsqrt_1;
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_4;
        float _shfl_5 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_5;
        float _shfl_6 = __shfl_sync(0xFFFFFFFF, logit2, 16);
        logits[6] = _shfl_6;
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
        float weights[7];
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
        float2 _f2_448 = make_float2(correction, correction);
        float2 corr = _f2_448;
        const int woff_143 = 0;
        int base_144 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_449 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_449;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_450 = make_float2(acc[2], acc[3]);
        float2 previous_145 = _f2_450;
        a_5[1] = mul_f32x2_noftz(previous_145, corr);
        float2 _f2_451 = make_float2(acc[4], acc[5]);
        float2 previous_146 = _f2_451;
        a_5[2] = mul_f32x2_noftz(previous_146, corr);
        float2 _f2_452 = make_float2(acc[6], acc[7]);
        float2 previous_147 = _f2_452;
        a_5[3] = mul_f32x2_noftz(previous_147, corr);
        float2 _f2_453 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_453;
        {
            float2 _f2_454 = make_float2(fsrc[0], fsrc[1]);
            float2 v_30 = _f2_454;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_30, a_5[0]);
            float2 _f2_455 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_28 = _f2_455;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_28, a_5[1]);
            float2 _f2_456 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_8 = _f2_456;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_8, a_5[2]);
            float2 _f2_457 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_28 = _f2_457;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_28, a_5[3]);
        }
        float2 _f2_462 = make_float2(weights[1], weights[1]);
        float2 weight_148 = _f2_462;
        {
            float2 _f2_463 = make_float2(fsrc[28], fsrc[29]);
            float2 v_31 = _f2_463;
            a_5[0] = fma_f32x2_rn_noftz(weight_148, v_31, a_5[0]);
            float2 _f2_464 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_29 = _f2_464;
            a_5[1] = fma_f32x2_rn_noftz(weight_148, v_0_29, a_5[1]);
            float2 _f2_465 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_9 = _f2_465;
            a_5[2] = fma_f32x2_rn_noftz(weight_148, v_1_9, a_5[2]);
            float2 _f2_466 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_29 = _f2_466;
            a_5[3] = fma_f32x2_rn_noftz(weight_148, v_2_29, a_5[3]);
        }
        float2 _f2_471 = make_float2(weights[2], weights[2]);
        float2 weight_149 = _f2_471;
        {
            float2 _f2_472 = make_float2(fsrc[56], fsrc[57]);
            float2 v_32 = _f2_472;
            a_5[0] = fma_f32x2_rn_noftz(weight_149, v_32, a_5[0]);
            float2 _f2_473 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_30 = _f2_473;
            a_5[1] = fma_f32x2_rn_noftz(weight_149, v_0_30, a_5[1]);
            float2 _f2_474 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_10 = _f2_474;
            a_5[2] = fma_f32x2_rn_noftz(weight_149, v_1_10, a_5[2]);
            float2 _f2_475 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_30 = _f2_475;
            a_5[3] = fma_f32x2_rn_noftz(weight_149, v_2_30, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_150 = 4;
        int base_151 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_152[4];
        float2 _f2_480 = make_float2(acc[8], acc[9]);
        float2 previous_153 = _f2_480;
        a_152[0] = mul_f32x2_noftz(previous_153, corr);
        float2 _f2_481 = make_float2(acc[10], acc[11]);
        float2 previous_154 = _f2_481;
        a_152[1] = mul_f32x2_noftz(previous_154, corr);
        float2 _f2_482 = make_float2(acc[12], acc[13]);
        float2 previous_155 = _f2_482;
        a_152[2] = mul_f32x2_noftz(previous_155, corr);
        float2 _f2_483 = make_float2(acc[14], acc[15]);
        float2 previous_156 = _f2_483;
        a_152[3] = mul_f32x2_noftz(previous_156, corr);
        float2 _f2_484 = make_float2(weights[0], weights[0]);
        float2 weight_157 = _f2_484;
        {
            float2 _f2_485 = make_float2(fsrc[8], fsrc[9]);
            float2 v_33 = _f2_485;
            a_152[0] = fma_f32x2_rn_noftz(weight_157, v_33, a_152[0]);
            float2 _f2_486 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_31 = _f2_486;
            a_152[1] = fma_f32x2_rn_noftz(weight_157, v_0_31, a_152[1]);
            float2 _f2_487 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_11 = _f2_487;
            a_152[2] = fma_f32x2_rn_noftz(weight_157, v_1_11, a_152[2]);
            float2 _f2_488 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_31 = _f2_488;
            a_152[3] = fma_f32x2_rn_noftz(weight_157, v_2_31, a_152[3]);
        }
        float2 _f2_493 = make_float2(weights[1], weights[1]);
        float2 weight_158 = _f2_493;
        {
            float2 _f2_494 = make_float2(fsrc[36], fsrc[37]);
            float2 v_34 = _f2_494;
            a_152[0] = fma_f32x2_rn_noftz(weight_158, v_34, a_152[0]);
            float2 _f2_495 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_32 = _f2_495;
            a_152[1] = fma_f32x2_rn_noftz(weight_158, v_0_32, a_152[1]);
            float2 _f2_496 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_12 = _f2_496;
            a_152[2] = fma_f32x2_rn_noftz(weight_158, v_1_12, a_152[2]);
            float2 _f2_497 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_32 = _f2_497;
            a_152[3] = fma_f32x2_rn_noftz(weight_158, v_2_32, a_152[3]);
        }
        float2 _f2_502 = make_float2(weights[2], weights[2]);
        float2 weight_159 = _f2_502;
        {
            float2 _f2_503 = make_float2(fsrc[64], fsrc[65]);
            float2 v_35 = _f2_503;
            a_152[0] = fma_f32x2_rn_noftz(weight_159, v_35, a_152[0]);
            float2 _f2_504 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_33 = _f2_504;
            a_152[1] = fma_f32x2_rn_noftz(weight_159, v_0_33, a_152[1]);
            float2 _f2_505 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_13 = _f2_505;
            a_152[2] = fma_f32x2_rn_noftz(weight_159, v_1_13, a_152[2]);
            float2 _f2_506 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_33 = _f2_506;
            a_152[3] = fma_f32x2_rn_noftz(weight_159, v_2_33, a_152[3]);
        }
        acc[8] = a_152[0].x;
        acc[9] = a_152[0].y;
        acc[10] = a_152[1].x;
        acc[11] = a_152[1].y;
        acc[12] = a_152[2].x;
        acc[13] = a_152[2].y;
        acc[14] = a_152[3].x;
        acc[15] = a_152[3].y;
        const int woff_160 = 8;
        int base_161 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_162[4];
        float2 _f2_511 = make_float2(acc[16], acc[17]);
        float2 previous_163 = _f2_511;
        a_162[0] = mul_f32x2_noftz(previous_163, corr);
        float2 _f2_512 = make_float2(acc[18], acc[19]);
        float2 previous_164 = _f2_512;
        a_162[1] = mul_f32x2_noftz(previous_164, corr);
        float2 _f2_513 = make_float2(acc[20], acc[21]);
        float2 previous_165 = _f2_513;
        a_162[2] = mul_f32x2_noftz(previous_165, corr);
        float2 _f2_514 = make_float2(acc[22], acc[23]);
        float2 previous_166 = _f2_514;
        a_162[3] = mul_f32x2_noftz(previous_166, corr);
        float2 _f2_515 = make_float2(weights[0], weights[0]);
        float2 weight_167 = _f2_515;
        {
            float2 _f2_516 = make_float2(fsrc[16], fsrc[17]);
            float2 v_36 = _f2_516;
            a_162[0] = fma_f32x2_rn_noftz(weight_167, v_36, a_162[0]);
            float2 _f2_517 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_34 = _f2_517;
            a_162[1] = fma_f32x2_rn_noftz(weight_167, v_0_34, a_162[1]);
            float2 _f2_518 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_14 = _f2_518;
            a_162[2] = fma_f32x2_rn_noftz(weight_167, v_1_14, a_162[2]);
            float2 _f2_519 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_34 = _f2_519;
            a_162[3] = fma_f32x2_rn_noftz(weight_167, v_2_34, a_162[3]);
        }
        float2 _f2_524 = make_float2(weights[1], weights[1]);
        float2 weight_168 = _f2_524;
        {
            float2 _f2_525 = make_float2(fsrc[44], fsrc[45]);
            float2 v_37 = _f2_525;
            a_162[0] = fma_f32x2_rn_noftz(weight_168, v_37, a_162[0]);
            float2 _f2_526 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_35 = _f2_526;
            a_162[1] = fma_f32x2_rn_noftz(weight_168, v_0_35, a_162[1]);
            float2 _f2_527 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_15 = _f2_527;
            a_162[2] = fma_f32x2_rn_noftz(weight_168, v_1_15, a_162[2]);
            float2 _f2_528 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_35 = _f2_528;
            a_162[3] = fma_f32x2_rn_noftz(weight_168, v_2_35, a_162[3]);
        }
        float2 _f2_533 = make_float2(weights[2], weights[2]);
        float2 weight_169 = _f2_533;
        {
            float2 _f2_534 = make_float2(fsrc[72], fsrc[73]);
            float2 v_38 = _f2_534;
            a_162[0] = fma_f32x2_rn_noftz(weight_169, v_38, a_162[0]);
            float2 _f2_535 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_36 = _f2_535;
            a_162[1] = fma_f32x2_rn_noftz(weight_169, v_0_36, a_162[1]);
            float2 _f2_536 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_16 = _f2_536;
            a_162[2] = fma_f32x2_rn_noftz(weight_169, v_1_16, a_162[2]);
            float2 _f2_537 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_36 = _f2_537;
            a_162[3] = fma_f32x2_rn_noftz(weight_169, v_2_36, a_162[3]);
        }
        acc[16] = a_162[0].x;
        acc[17] = a_162[0].y;
        acc[18] = a_162[1].x;
        acc[19] = a_162[1].y;
        acc[20] = a_162[2].x;
        acc[21] = a_162[2].y;
        acc[22] = a_162[3].x;
        acc[23] = a_162[3].y;
        const int woff_170 = 12;
        int base_171 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_172[4];
        float2 _f2_542 = make_float2(acc[24], acc[25]);
        float2 previous_173 = _f2_542;
        a_172[0] = mul_f32x2_noftz(previous_173, corr);
        float2 _f2_543 = make_float2(acc[26], acc[27]);
        float2 previous_174 = _f2_543;
        a_172[1] = mul_f32x2_noftz(previous_174, corr);
        float2 _f2_544 = make_float2(weights[0], weights[0]);
        float2 weight_175 = _f2_544;
        {
            float2 _f2_545 = make_float2(fsrc[24], fsrc[25]);
            float2 v_39 = _f2_545;
            a_172[0] = fma_f32x2_rn_noftz(weight_175, v_39, a_172[0]);
            float2 _f2_546 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_37 = _f2_546;
            a_172[1] = fma_f32x2_rn_noftz(weight_175, v_0_37, a_172[1]);
        }
        float2 _f2_549 = make_float2(weights[1], weights[1]);
        float2 weight_176 = _f2_549;
        {
            float2 _f2_550 = make_float2(fsrc[52], fsrc[53]);
            float2 v_40 = _f2_550;
            a_172[0] = fma_f32x2_rn_noftz(weight_176, v_40, a_172[0]);
            float2 _f2_551 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_38 = _f2_551;
            a_172[1] = fma_f32x2_rn_noftz(weight_176, v_0_38, a_172[1]);
        }
        float2 _f2_554 = make_float2(weights[2], weights[2]);
        float2 weight_177 = _f2_554;
        {
            float2 _f2_555 = make_float2(fsrc[80], fsrc[81]);
            float2 v_41 = _f2_555;
            a_172[0] = fma_f32x2_rn_noftz(weight_177, v_41, a_172[0]);
            float2 _f2_556 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_39 = _f2_556;
            a_172[1] = fma_f32x2_rn_noftz(weight_177, v_0_39, a_172[1]);
        }
        acc[24] = a_172[0].x;
        acc[25] = a_172[0].y;
        acc[26] = a_172[1].x;
        acc[27] = a_172[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float max_chunk_178 = -3.4028234663852886e+38f;
        float _fmax_4 = fmaxf(max_chunk_178, logits[3]);
        max_chunk_178 = _fmax_4;
        float _fmax_5 = fmaxf(max_chunk_178, logits[4]);
        max_chunk_178 = _fmax_5;
        float _fmax_6 = fmaxf(max_chunk_178, logits[5]);
        max_chunk_178 = _fmax_6;
        float _fmax_7 = fmaxf(max_running, max_chunk_178);
        float max_new_179 = _fmax_7;
        float _exp2_4 = approx_exp2((max_running - max_new_179) * 1.4426950408889634f);
        float correction_180 = _exp2_4;
        float weights_181[7];
        float sum_weights_182 = 0.0f;
        float _exp2_5 = approx_exp2((logits[3] - max_new_179) * 1.4426950408889634f);
        weights_181[3] = _exp2_5;
        sum_weights_182 += weights_181[3];
        float _exp2_6 = approx_exp2((logits[4] - max_new_179) * 1.4426950408889634f);
        weights_181[4] = _exp2_6;
        sum_weights_182 += weights_181[4];
        float _exp2_7 = approx_exp2((logits[5] - max_new_179) * 1.4426950408889634f);
        weights_181[5] = _exp2_7;
        sum_weights_182 += weights_181[5];
        float2 _f2_559 = make_float2(correction_180, correction_180);
        float2 corr_183 = _f2_559;
        const int woff_184 = 0;
        int base_185 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_186[4];
        float2 _f2_560 = make_float2(acc[0], acc[1]);
        float2 previous_187 = _f2_560;
        a_186[0] = mul_f32x2_noftz(previous_187, corr_183);
        float2 _f2_561 = make_float2(acc[2], acc[3]);
        float2 previous_188 = _f2_561;
        a_186[1] = mul_f32x2_noftz(previous_188, corr_183);
        float2 _f2_562 = make_float2(acc[4], acc[5]);
        float2 previous_189 = _f2_562;
        a_186[2] = mul_f32x2_noftz(previous_189, corr_183);
        float2 _f2_563 = make_float2(acc[6], acc[7]);
        float2 previous_190 = _f2_563;
        a_186[3] = mul_f32x2_noftz(previous_190, corr_183);
        float2 _f2_564 = make_float2(weights_181[3], weights_181[3]);
        float2 weight_191 = _f2_564;
        {
            unsigned int sw2[4];
            sw2[0] = words[42 + woff_184];
            sw2[1] = words[42 + woff_184 + 1];
            sw2[2] = words[42 + woff_184 + 2];
            sw2[3] = words[42 + woff_184 + 3];
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
            float2 _f2_569 = make_float2(sw2_f32[0], sw2_f32[1]);
            float2 v_42 = _f2_569;
            a_186[0] = fma_f32x2_rn_noftz(weight_191, v_42, a_186[0]);
            float2 _f2_570 = make_float2(sw2_f32[2], sw2_f32[3]);
            float2 v_0_40 = _f2_570;
            a_186[1] = fma_f32x2_rn_noftz(weight_191, v_0_40, a_186[1]);
            float2 _f2_571 = make_float2(sw2_f32[4], sw2_f32[5]);
            float2 v_1_17 = _f2_571;
            a_186[2] = fma_f32x2_rn_noftz(weight_191, v_1_17, a_186[2]);
            float2 _f2_572 = make_float2(sw2_f32[6], sw2_f32[7]);
            float2 v_2_37 = _f2_572;
            a_186[3] = fma_f32x2_rn_noftz(weight_191, v_2_37, a_186[3]);
        }
        float2 _f2_573 = make_float2(weights_181[4], weights_181[4]);
        float2 weight_192 = _f2_573;
        {
            unsigned int sw2_1[4];
            sw2_1[0] = words[56 + woff_184];
            sw2_1[1] = words[56 + woff_184 + 1];
            sw2_1[2] = words[56 + woff_184 + 2];
            sw2_1[3] = words[56 + woff_184 + 3];
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
            float2 _f2_578 = make_float2(sw2_f32_1[0], sw2_f32_1[1]);
            float2 v_43 = _f2_578;
            a_186[0] = fma_f32x2_rn_noftz(weight_192, v_43, a_186[0]);
            float2 _f2_579 = make_float2(sw2_f32_1[2], sw2_f32_1[3]);
            float2 v_0_41 = _f2_579;
            a_186[1] = fma_f32x2_rn_noftz(weight_192, v_0_41, a_186[1]);
            float2 _f2_580 = make_float2(sw2_f32_1[4], sw2_f32_1[5]);
            float2 v_1_18 = _f2_580;
            a_186[2] = fma_f32x2_rn_noftz(weight_192, v_1_18, a_186[2]);
            float2 _f2_581 = make_float2(sw2_f32_1[6], sw2_f32_1[7]);
            float2 v_2_38 = _f2_581;
            a_186[3] = fma_f32x2_rn_noftz(weight_192, v_2_38, a_186[3]);
        }
        float2 _f2_582 = make_float2(weights_181[5], weights_181[5]);
        float2 weight_193 = _f2_582;
        {
            unsigned int sw2_2[4];
            sw2_2[0] = words[70 + woff_184];
            sw2_2[1] = words[70 + woff_184 + 1];
            sw2_2[2] = words[70 + woff_184 + 2];
            sw2_2[3] = words[70 + woff_184 + 3];
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
            float2 _f2_587 = make_float2(sw2_f32_2[0], sw2_f32_2[1]);
            float2 v_44 = _f2_587;
            a_186[0] = fma_f32x2_rn_noftz(weight_193, v_44, a_186[0]);
            float2 _f2_588 = make_float2(sw2_f32_2[2], sw2_f32_2[3]);
            float2 v_0_42 = _f2_588;
            a_186[1] = fma_f32x2_rn_noftz(weight_193, v_0_42, a_186[1]);
            float2 _f2_589 = make_float2(sw2_f32_2[4], sw2_f32_2[5]);
            float2 v_1_19 = _f2_589;
            a_186[2] = fma_f32x2_rn_noftz(weight_193, v_1_19, a_186[2]);
            float2 _f2_590 = make_float2(sw2_f32_2[6], sw2_f32_2[7]);
            float2 v_2_39 = _f2_590;
            a_186[3] = fma_f32x2_rn_noftz(weight_193, v_2_39, a_186[3]);
        }
        acc[0] = a_186[0].x;
        acc[1] = a_186[0].y;
        acc[2] = a_186[1].x;
        acc[3] = a_186[1].y;
        acc[4] = a_186[2].x;
        acc[5] = a_186[2].y;
        acc[6] = a_186[3].x;
        acc[7] = a_186[3].y;
        const int woff_194 = 4;
        int base_195 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_196[4];
        float2 _f2_591 = make_float2(acc[8], acc[9]);
        float2 previous_197 = _f2_591;
        a_196[0] = mul_f32x2_noftz(previous_197, corr_183);
        float2 _f2_592 = make_float2(acc[10], acc[11]);
        float2 previous_198 = _f2_592;
        a_196[1] = mul_f32x2_noftz(previous_198, corr_183);
        float2 _f2_593 = make_float2(acc[12], acc[13]);
        float2 previous_199 = _f2_593;
        a_196[2] = mul_f32x2_noftz(previous_199, corr_183);
        float2 _f2_594 = make_float2(acc[14], acc[15]);
        float2 previous_200 = _f2_594;
        a_196[3] = mul_f32x2_noftz(previous_200, corr_183);
        float2 _f2_595 = make_float2(weights_181[3], weights_181[3]);
        float2 weight_201 = _f2_595;
        {
            unsigned int sw2_3[4];
            sw2_3[0] = words[42 + woff_194];
            sw2_3[1] = words[42 + woff_194 + 1];
            sw2_3[2] = words[42 + woff_194 + 2];
            sw2_3[3] = words[42 + woff_194 + 3];
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
            float2 _f2_600 = make_float2(sw2_f32_3[0], sw2_f32_3[1]);
            float2 v_45 = _f2_600;
            a_196[0] = fma_f32x2_rn_noftz(weight_201, v_45, a_196[0]);
            float2 _f2_601 = make_float2(sw2_f32_3[2], sw2_f32_3[3]);
            float2 v_0_43 = _f2_601;
            a_196[1] = fma_f32x2_rn_noftz(weight_201, v_0_43, a_196[1]);
            float2 _f2_602 = make_float2(sw2_f32_3[4], sw2_f32_3[5]);
            float2 v_1_20 = _f2_602;
            a_196[2] = fma_f32x2_rn_noftz(weight_201, v_1_20, a_196[2]);
            float2 _f2_603 = make_float2(sw2_f32_3[6], sw2_f32_3[7]);
            float2 v_2_40 = _f2_603;
            a_196[3] = fma_f32x2_rn_noftz(weight_201, v_2_40, a_196[3]);
        }
        float2 _f2_604 = make_float2(weights_181[4], weights_181[4]);
        float2 weight_202 = _f2_604;
        {
            unsigned int sw2_4[4];
            sw2_4[0] = words[56 + woff_194];
            sw2_4[1] = words[56 + woff_194 + 1];
            sw2_4[2] = words[56 + woff_194 + 2];
            sw2_4[3] = words[56 + woff_194 + 3];
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
            float2 _f2_609 = make_float2(sw2_f32_4[0], sw2_f32_4[1]);
            float2 v_46 = _f2_609;
            a_196[0] = fma_f32x2_rn_noftz(weight_202, v_46, a_196[0]);
            float2 _f2_610 = make_float2(sw2_f32_4[2], sw2_f32_4[3]);
            float2 v_0_44 = _f2_610;
            a_196[1] = fma_f32x2_rn_noftz(weight_202, v_0_44, a_196[1]);
            float2 _f2_611 = make_float2(sw2_f32_4[4], sw2_f32_4[5]);
            float2 v_1_21 = _f2_611;
            a_196[2] = fma_f32x2_rn_noftz(weight_202, v_1_21, a_196[2]);
            float2 _f2_612 = make_float2(sw2_f32_4[6], sw2_f32_4[7]);
            float2 v_2_41 = _f2_612;
            a_196[3] = fma_f32x2_rn_noftz(weight_202, v_2_41, a_196[3]);
        }
        float2 _f2_613 = make_float2(weights_181[5], weights_181[5]);
        float2 weight_203 = _f2_613;
        {
            unsigned int sw2_5[4];
            sw2_5[0] = words[70 + woff_194];
            sw2_5[1] = words[70 + woff_194 + 1];
            sw2_5[2] = words[70 + woff_194 + 2];
            sw2_5[3] = words[70 + woff_194 + 3];
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
            float2 _f2_618 = make_float2(sw2_f32_5[0], sw2_f32_5[1]);
            float2 v_47 = _f2_618;
            a_196[0] = fma_f32x2_rn_noftz(weight_203, v_47, a_196[0]);
            float2 _f2_619 = make_float2(sw2_f32_5[2], sw2_f32_5[3]);
            float2 v_0_45 = _f2_619;
            a_196[1] = fma_f32x2_rn_noftz(weight_203, v_0_45, a_196[1]);
            float2 _f2_620 = make_float2(sw2_f32_5[4], sw2_f32_5[5]);
            float2 v_1_22 = _f2_620;
            a_196[2] = fma_f32x2_rn_noftz(weight_203, v_1_22, a_196[2]);
            float2 _f2_621 = make_float2(sw2_f32_5[6], sw2_f32_5[7]);
            float2 v_2_42 = _f2_621;
            a_196[3] = fma_f32x2_rn_noftz(weight_203, v_2_42, a_196[3]);
        }
        acc[8] = a_196[0].x;
        acc[9] = a_196[0].y;
        acc[10] = a_196[1].x;
        acc[11] = a_196[1].y;
        acc[12] = a_196[2].x;
        acc[13] = a_196[2].y;
        acc[14] = a_196[3].x;
        acc[15] = a_196[3].y;
        const int woff_204 = 8;
        int base_205 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_206[4];
        float2 _f2_622 = make_float2(acc[16], acc[17]);
        float2 previous_207 = _f2_622;
        a_206[0] = mul_f32x2_noftz(previous_207, corr_183);
        float2 _f2_623 = make_float2(acc[18], acc[19]);
        float2 previous_208 = _f2_623;
        a_206[1] = mul_f32x2_noftz(previous_208, corr_183);
        float2 _f2_624 = make_float2(acc[20], acc[21]);
        float2 previous_209 = _f2_624;
        a_206[2] = mul_f32x2_noftz(previous_209, corr_183);
        float2 _f2_625 = make_float2(acc[22], acc[23]);
        float2 previous_210 = _f2_625;
        a_206[3] = mul_f32x2_noftz(previous_210, corr_183);
        float2 _f2_626 = make_float2(weights_181[3], weights_181[3]);
        float2 weight_211 = _f2_626;
        {
            unsigned int sw2_6[4];
            sw2_6[0] = words[42 + woff_204];
            sw2_6[1] = words[42 + woff_204 + 1];
            sw2_6[2] = words[42 + woff_204 + 2];
            sw2_6[3] = words[42 + woff_204 + 3];
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
            float2 _f2_631 = make_float2(sw2_f32_6[0], sw2_f32_6[1]);
            float2 v_48 = _f2_631;
            a_206[0] = fma_f32x2_rn_noftz(weight_211, v_48, a_206[0]);
            float2 _f2_632 = make_float2(sw2_f32_6[2], sw2_f32_6[3]);
            float2 v_0_46 = _f2_632;
            a_206[1] = fma_f32x2_rn_noftz(weight_211, v_0_46, a_206[1]);
            float2 _f2_633 = make_float2(sw2_f32_6[4], sw2_f32_6[5]);
            float2 v_1_23 = _f2_633;
            a_206[2] = fma_f32x2_rn_noftz(weight_211, v_1_23, a_206[2]);
            float2 _f2_634 = make_float2(sw2_f32_6[6], sw2_f32_6[7]);
            float2 v_2_43 = _f2_634;
            a_206[3] = fma_f32x2_rn_noftz(weight_211, v_2_43, a_206[3]);
        }
        float2 _f2_635 = make_float2(weights_181[4], weights_181[4]);
        float2 weight_212 = _f2_635;
        {
            unsigned int sw2_7[4];
            sw2_7[0] = words[56 + woff_204];
            sw2_7[1] = words[56 + woff_204 + 1];
            sw2_7[2] = words[56 + woff_204 + 2];
            sw2_7[3] = words[56 + woff_204 + 3];
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
            float2 _f2_640 = make_float2(sw2_f32_7[0], sw2_f32_7[1]);
            float2 v_49 = _f2_640;
            a_206[0] = fma_f32x2_rn_noftz(weight_212, v_49, a_206[0]);
            float2 _f2_641 = make_float2(sw2_f32_7[2], sw2_f32_7[3]);
            float2 v_0_47 = _f2_641;
            a_206[1] = fma_f32x2_rn_noftz(weight_212, v_0_47, a_206[1]);
            float2 _f2_642 = make_float2(sw2_f32_7[4], sw2_f32_7[5]);
            float2 v_1_24 = _f2_642;
            a_206[2] = fma_f32x2_rn_noftz(weight_212, v_1_24, a_206[2]);
            float2 _f2_643 = make_float2(sw2_f32_7[6], sw2_f32_7[7]);
            float2 v_2_44 = _f2_643;
            a_206[3] = fma_f32x2_rn_noftz(weight_212, v_2_44, a_206[3]);
        }
        float2 _f2_644 = make_float2(weights_181[5], weights_181[5]);
        float2 weight_213 = _f2_644;
        {
            unsigned int sw2_8[4];
            sw2_8[0] = words[70 + woff_204];
            sw2_8[1] = words[70 + woff_204 + 1];
            sw2_8[2] = words[70 + woff_204 + 2];
            sw2_8[3] = words[70 + woff_204 + 3];
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
            float2 _f2_649 = make_float2(sw2_f32_8[0], sw2_f32_8[1]);
            float2 v_50 = _f2_649;
            a_206[0] = fma_f32x2_rn_noftz(weight_213, v_50, a_206[0]);
            float2 _f2_650 = make_float2(sw2_f32_8[2], sw2_f32_8[3]);
            float2 v_0_48 = _f2_650;
            a_206[1] = fma_f32x2_rn_noftz(weight_213, v_0_48, a_206[1]);
            float2 _f2_651 = make_float2(sw2_f32_8[4], sw2_f32_8[5]);
            float2 v_1_25 = _f2_651;
            a_206[2] = fma_f32x2_rn_noftz(weight_213, v_1_25, a_206[2]);
            float2 _f2_652 = make_float2(sw2_f32_8[6], sw2_f32_8[7]);
            float2 v_2_45 = _f2_652;
            a_206[3] = fma_f32x2_rn_noftz(weight_213, v_2_45, a_206[3]);
        }
        acc[16] = a_206[0].x;
        acc[17] = a_206[0].y;
        acc[18] = a_206[1].x;
        acc[19] = a_206[1].y;
        acc[20] = a_206[2].x;
        acc[21] = a_206[2].y;
        acc[22] = a_206[3].x;
        acc[23] = a_206[3].y;
        const int woff_214 = 12;
        int base_215 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_216[4];
        float2 _f2_653 = make_float2(acc[24], acc[25]);
        float2 previous_217 = _f2_653;
        a_216[0] = mul_f32x2_noftz(previous_217, corr_183);
        float2 _f2_654 = make_float2(acc[26], acc[27]);
        float2 previous_218 = _f2_654;
        a_216[1] = mul_f32x2_noftz(previous_218, corr_183);
        float2 _f2_655 = make_float2(weights_181[3], weights_181[3]);
        float2 weight_219 = _f2_655;
        {
            unsigned int sw2_9[4];
            sw2_9[0] = words[42 + woff_214];
            sw2_9[1] = words[42 + woff_214 + 1];
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
            float2 _f2_658 = make_float2(sw2_f32_9[0], sw2_f32_9[1]);
            float2 v_51 = _f2_658;
            a_216[0] = fma_f32x2_rn_noftz(weight_219, v_51, a_216[0]);
            float2 _f2_659 = make_float2(sw2_f32_9[2], sw2_f32_9[3]);
            float2 v_0_49 = _f2_659;
            a_216[1] = fma_f32x2_rn_noftz(weight_219, v_0_49, a_216[1]);
        }
        float2 _f2_660 = make_float2(weights_181[4], weights_181[4]);
        float2 weight_220 = _f2_660;
        {
            unsigned int sw2_10[4];
            sw2_10[0] = words[56 + woff_214];
            sw2_10[1] = words[56 + woff_214 + 1];
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
            float2 _f2_663 = make_float2(sw2_f32_10[0], sw2_f32_10[1]);
            float2 v_52 = _f2_663;
            a_216[0] = fma_f32x2_rn_noftz(weight_220, v_52, a_216[0]);
            float2 _f2_664 = make_float2(sw2_f32_10[2], sw2_f32_10[3]);
            float2 v_0_50 = _f2_664;
            a_216[1] = fma_f32x2_rn_noftz(weight_220, v_0_50, a_216[1]);
        }
        float2 _f2_665 = make_float2(weights_181[5], weights_181[5]);
        float2 weight_221 = _f2_665;
        {
            unsigned int sw2_11[4];
            sw2_11[0] = words[70 + woff_214];
            sw2_11[1] = words[70 + woff_214 + 1];
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
            float2 _f2_668 = make_float2(sw2_f32_11[0], sw2_f32_11[1]);
            float2 v_53 = _f2_668;
            a_216[0] = fma_f32x2_rn_noftz(weight_221, v_53, a_216[0]);
            float2 _f2_669 = make_float2(sw2_f32_11[2], sw2_f32_11[3]);
            float2 v_0_51 = _f2_669;
            a_216[1] = fma_f32x2_rn_noftz(weight_221, v_0_51, a_216[1]);
        }
        acc[24] = a_216[0].x;
        acc[25] = a_216[0].y;
        acc[26] = a_216[1].x;
        acc[27] = a_216[1].y;
        sum_running = sum_running * correction_180 + sum_weights_182;
        max_running = max_new_179;
        float max_chunk_222 = -3.4028234663852886e+38f;
        float _fmax_8 = fmaxf(max_chunk_222, logits[6]);
        max_chunk_222 = _fmax_8;
        float _fmax_9 = fmaxf(max_running, max_chunk_222);
        float max_new_223 = _fmax_9;
        float _exp2_8 = approx_exp2((max_running - max_new_223) * 1.4426950408889634f);
        float correction_224 = _exp2_8;
        float weights_225[7];
        float sum_weights_226 = 0.0f;
        float _exp2_9 = approx_exp2((logits[6] - max_new_223) * 1.4426950408889634f);
        weights_225[6] = _exp2_9;
        sum_weights_226 += weights_225[6];
        float2 _f2_670 = make_float2(correction_224, correction_224);
        float2 corr_227 = _f2_670;
        const int woff_228 = 0;
        int base_229 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_230[4];
        float2 _f2_671 = make_float2(acc[0], acc[1]);
        float2 previous_231 = _f2_671;
        a_230[0] = mul_f32x2_noftz(previous_231, corr_227);
        float2 _f2_672 = make_float2(acc[2], acc[3]);
        float2 previous_232 = _f2_672;
        a_230[1] = mul_f32x2_noftz(previous_232, corr_227);
        float2 _f2_673 = make_float2(acc[4], acc[5]);
        float2 previous_233 = _f2_673;
        a_230[2] = mul_f32x2_noftz(previous_233, corr_227);
        float2 _f2_674 = make_float2(acc[6], acc[7]);
        float2 previous_234 = _f2_674;
        a_230[3] = mul_f32x2_noftz(previous_234, corr_227);
        float2 _f2_675 = make_float2(weights_225[6], weights_225[6]);
        float2 weight_235 = _f2_675;
        {
            unsigned int sw2_12[4];
            sw2_12[0] = words[84 + woff_228];
            sw2_12[1] = words[84 + woff_228 + 1];
            sw2_12[2] = words[84 + woff_228 + 2];
            sw2_12[3] = words[84 + woff_228 + 3];
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
            float2 _f2_680 = make_float2(sw2_f32_12[0], sw2_f32_12[1]);
            float2 v_54 = _f2_680;
            a_230[0] = fma_f32x2_rn_noftz(weight_235, v_54, a_230[0]);
            float2 _f2_681 = make_float2(sw2_f32_12[2], sw2_f32_12[3]);
            float2 v_0_52 = _f2_681;
            a_230[1] = fma_f32x2_rn_noftz(weight_235, v_0_52, a_230[1]);
            float2 _f2_682 = make_float2(sw2_f32_12[4], sw2_f32_12[5]);
            float2 v_1_26 = _f2_682;
            a_230[2] = fma_f32x2_rn_noftz(weight_235, v_1_26, a_230[2]);
            float2 _f2_683 = make_float2(sw2_f32_12[6], sw2_f32_12[7]);
            float2 v_2_46 = _f2_683;
            a_230[3] = fma_f32x2_rn_noftz(weight_235, v_2_46, a_230[3]);
        }
        acc[0] = a_230[0].x;
        acc[1] = a_230[0].y;
        acc[2] = a_230[1].x;
        acc[3] = a_230[1].y;
        acc[4] = a_230[2].x;
        acc[5] = a_230[2].y;
        acc[6] = a_230[3].x;
        acc[7] = a_230[3].y;
        const int woff_236 = 4;
        int base_237 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_238[4];
        float2 _f2_684 = make_float2(acc[8], acc[9]);
        float2 previous_239 = _f2_684;
        a_238[0] = mul_f32x2_noftz(previous_239, corr_227);
        float2 _f2_685 = make_float2(acc[10], acc[11]);
        float2 previous_240 = _f2_685;
        a_238[1] = mul_f32x2_noftz(previous_240, corr_227);
        float2 _f2_686 = make_float2(acc[12], acc[13]);
        float2 previous_241 = _f2_686;
        a_238[2] = mul_f32x2_noftz(previous_241, corr_227);
        float2 _f2_687 = make_float2(acc[14], acc[15]);
        float2 previous_242 = _f2_687;
        a_238[3] = mul_f32x2_noftz(previous_242, corr_227);
        float2 _f2_688 = make_float2(weights_225[6], weights_225[6]);
        float2 weight_243 = _f2_688;
        {
            unsigned int sw2_13[4];
            sw2_13[0] = words[84 + woff_236];
            sw2_13[1] = words[84 + woff_236 + 1];
            sw2_13[2] = words[84 + woff_236 + 2];
            sw2_13[3] = words[84 + woff_236 + 3];
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
            float2 _f2_693 = make_float2(sw2_f32_13[0], sw2_f32_13[1]);
            float2 v_55 = _f2_693;
            a_238[0] = fma_f32x2_rn_noftz(weight_243, v_55, a_238[0]);
            float2 _f2_694 = make_float2(sw2_f32_13[2], sw2_f32_13[3]);
            float2 v_0_53 = _f2_694;
            a_238[1] = fma_f32x2_rn_noftz(weight_243, v_0_53, a_238[1]);
            float2 _f2_695 = make_float2(sw2_f32_13[4], sw2_f32_13[5]);
            float2 v_1_27 = _f2_695;
            a_238[2] = fma_f32x2_rn_noftz(weight_243, v_1_27, a_238[2]);
            float2 _f2_696 = make_float2(sw2_f32_13[6], sw2_f32_13[7]);
            float2 v_2_47 = _f2_696;
            a_238[3] = fma_f32x2_rn_noftz(weight_243, v_2_47, a_238[3]);
        }
        acc[8] = a_238[0].x;
        acc[9] = a_238[0].y;
        acc[10] = a_238[1].x;
        acc[11] = a_238[1].y;
        acc[12] = a_238[2].x;
        acc[13] = a_238[2].y;
        acc[14] = a_238[3].x;
        acc[15] = a_238[3].y;
        const int woff_244 = 8;
        int base_245 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_246[4];
        float2 _f2_697 = make_float2(acc[16], acc[17]);
        float2 previous_247 = _f2_697;
        a_246[0] = mul_f32x2_noftz(previous_247, corr_227);
        float2 _f2_698 = make_float2(acc[18], acc[19]);
        float2 previous_248 = _f2_698;
        a_246[1] = mul_f32x2_noftz(previous_248, corr_227);
        float2 _f2_699 = make_float2(acc[20], acc[21]);
        float2 previous_249 = _f2_699;
        a_246[2] = mul_f32x2_noftz(previous_249, corr_227);
        float2 _f2_700 = make_float2(acc[22], acc[23]);
        float2 previous_250 = _f2_700;
        a_246[3] = mul_f32x2_noftz(previous_250, corr_227);
        float2 _f2_701 = make_float2(weights_225[6], weights_225[6]);
        float2 weight_251 = _f2_701;
        {
            unsigned int sw2_14[4];
            sw2_14[0] = words[84 + woff_244];
            sw2_14[1] = words[84 + woff_244 + 1];
            sw2_14[2] = words[84 + woff_244 + 2];
            sw2_14[3] = words[84 + woff_244 + 3];
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
            float2 _f2_706 = make_float2(sw2_f32_14[0], sw2_f32_14[1]);
            float2 v_56 = _f2_706;
            a_246[0] = fma_f32x2_rn_noftz(weight_251, v_56, a_246[0]);
            float2 _f2_707 = make_float2(sw2_f32_14[2], sw2_f32_14[3]);
            float2 v_0_54 = _f2_707;
            a_246[1] = fma_f32x2_rn_noftz(weight_251, v_0_54, a_246[1]);
            float2 _f2_708 = make_float2(sw2_f32_14[4], sw2_f32_14[5]);
            float2 v_1_28 = _f2_708;
            a_246[2] = fma_f32x2_rn_noftz(weight_251, v_1_28, a_246[2]);
            float2 _f2_709 = make_float2(sw2_f32_14[6], sw2_f32_14[7]);
            float2 v_2_48 = _f2_709;
            a_246[3] = fma_f32x2_rn_noftz(weight_251, v_2_48, a_246[3]);
        }
        acc[16] = a_246[0].x;
        acc[17] = a_246[0].y;
        acc[18] = a_246[1].x;
        acc[19] = a_246[1].y;
        acc[20] = a_246[2].x;
        acc[21] = a_246[2].y;
        acc[22] = a_246[3].x;
        acc[23] = a_246[3].y;
        const int woff_252 = 12;
        int base_253 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_254[4];
        float2 _f2_710 = make_float2(acc[24], acc[25]);
        float2 previous_255 = _f2_710;
        a_254[0] = mul_f32x2_noftz(previous_255, corr_227);
        float2 _f2_711 = make_float2(acc[26], acc[27]);
        float2 previous_256 = _f2_711;
        a_254[1] = mul_f32x2_noftz(previous_256, corr_227);
        float2 _f2_712 = make_float2(weights_225[6], weights_225[6]);
        float2 weight_257 = _f2_712;
        {
            unsigned int sw2_15[4];
            sw2_15[0] = words[84 + woff_252];
            sw2_15[1] = words[84 + woff_252 + 1];
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
            float2 _f2_715 = make_float2(sw2_f32_15[0], sw2_f32_15[1]);
            float2 v_57 = _f2_715;
            a_254[0] = fma_f32x2_rn_noftz(weight_257, v_57, a_254[0]);
            float2 _f2_716 = make_float2(sw2_f32_15[2], sw2_f32_15[3]);
            float2 v_0_55 = _f2_716;
            a_254[1] = fma_f32x2_rn_noftz(weight_257, v_0_55, a_254[1]);
        }
        acc[24] = a_254[0].x;
        acc[25] = a_254[0].y;
        acc[26] = a_254[1].x;
        acc[27] = a_254[1].y;
        sum_running = sum_running * correction_224 + sum_weights_226;
        max_running = max_new_223;
        float2 _f2_717 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_717;
        float2 _f2_718 = make_float2(acc[0], acc[1]);
        float2 v_58 = _f2_718;
        output_sq_pair = fma_f32x2_rn_noftz(v_58, v_58, output_sq_pair);
        float2 _f2_719 = make_float2(acc[2], acc[3]);
        float2 v_258 = _f2_719;
        output_sq_pair = fma_f32x2_rn_noftz(v_258, v_258, output_sq_pair);
        float2 _f2_720 = make_float2(acc[4], acc[5]);
        float2 v_259 = _f2_720;
        output_sq_pair = fma_f32x2_rn_noftz(v_259, v_259, output_sq_pair);
        float2 _f2_721 = make_float2(acc[6], acc[7]);
        float2 v_260 = _f2_721;
        output_sq_pair = fma_f32x2_rn_noftz(v_260, v_260, output_sq_pair);
        float2 _f2_722 = make_float2(acc[8], acc[9]);
        float2 v_261 = _f2_722;
        output_sq_pair = fma_f32x2_rn_noftz(v_261, v_261, output_sq_pair);
        float2 _f2_723 = make_float2(acc[10], acc[11]);
        float2 v_262 = _f2_723;
        output_sq_pair = fma_f32x2_rn_noftz(v_262, v_262, output_sq_pair);
        float2 _f2_724 = make_float2(acc[12], acc[13]);
        float2 v_263 = _f2_724;
        output_sq_pair = fma_f32x2_rn_noftz(v_263, v_263, output_sq_pair);
        float2 _f2_725 = make_float2(acc[14], acc[15]);
        float2 v_264 = _f2_725;
        output_sq_pair = fma_f32x2_rn_noftz(v_264, v_264, output_sq_pair);
        float2 _f2_726 = make_float2(acc[16], acc[17]);
        float2 v_265 = _f2_726;
        output_sq_pair = fma_f32x2_rn_noftz(v_265, v_265, output_sq_pair);
        float2 _f2_727 = make_float2(acc[18], acc[19]);
        float2 v_266 = _f2_727;
        output_sq_pair = fma_f32x2_rn_noftz(v_266, v_266, output_sq_pair);
        float2 _f2_728 = make_float2(acc[20], acc[21]);
        float2 v_267 = _f2_728;
        output_sq_pair = fma_f32x2_rn_noftz(v_267, v_267, output_sq_pair);
        float2 _f2_729 = make_float2(acc[22], acc[23]);
        float2 v_268 = _f2_729;
        output_sq_pair = fma_f32x2_rn_noftz(v_268, v_268, output_sq_pair);
        float2 _f2_730 = make_float2(acc[24], acc[25]);
        float2 v_269 = _f2_730;
        output_sq_pair = fma_f32x2_rn_noftz(v_269, v_269, output_sq_pair);
        float2 _f2_731 = make_float2(acc[26], acc[27]);
        float2 v_270 = _f2_731;
        output_sq_pair = fma_f32x2_rn_noftz(v_270, v_270, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_35;
        float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_36;
        float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_37;
        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_38;
        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_39;
        if (lane == 0) {
            uint32_t _mapa_56;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_56) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_56), "f"(output_sq) : "memory");
            uint32_t _mapa_57;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_57) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_57), "f"(output_sq) : "memory");
            uint32_t _mapa_58;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_58) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_58), "f"(output_sq) : "memory");
            uint32_t _mapa_59;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_59) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_59), "f"(output_sq) : "memory");
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
        float _shfl_7 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_7;
        float2 _f2_732 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_732;
        int base_271 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_733 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_733, rsigma_pair);
        float2 _f2_734 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_734);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_272 = 2;
        const int acc_idx_273 = value_idx_272;
        float2 _f2_735 = make_float2(acc[acc_idx_273], acc[acc_idx_273 + 1]);
        float2 scaled_pair_274 = mul_f32x2_noftz(_f2_735, rsigma_pair);
        float2 _f2_736 = make_float2(wout[acc_idx_273], wout[acc_idx_273 + 1]);
        float2 normalized_pair_275 = mul_f32x2_noftz(scaled_pair_274, _f2_736);
        output_values[value_idx_272] = normalized_pair_275.x;
        output_values[value_idx_272 + 1] = normalized_pair_275.y;
        const int value_idx_276 = 4;
        const int acc_idx_277 = value_idx_276;
        float2 _f2_737 = make_float2(acc[acc_idx_277], acc[acc_idx_277 + 1]);
        float2 scaled_pair_278 = mul_f32x2_noftz(_f2_737, rsigma_pair);
        float2 _f2_738 = make_float2(wout[acc_idx_277], wout[acc_idx_277 + 1]);
        float2 normalized_pair_279 = mul_f32x2_noftz(scaled_pair_278, _f2_738);
        output_values[value_idx_276] = normalized_pair_279.x;
        output_values[value_idx_276 + 1] = normalized_pair_279.y;
        const int value_idx_280 = 6;
        const int acc_idx_281 = value_idx_280;
        float2 _f2_739 = make_float2(acc[acc_idx_281], acc[acc_idx_281 + 1]);
        float2 scaled_pair_282 = mul_f32x2_noftz(_f2_739, rsigma_pair);
        float2 _f2_740 = make_float2(wout[acc_idx_281], wout[acc_idx_281 + 1]);
        float2 normalized_pair_283 = mul_f32x2_noftz(scaled_pair_282, _f2_740);
        output_values[value_idx_280] = normalized_pair_283.x;
        output_values[value_idx_280 + 1] = normalized_pair_283.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_271 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_284 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_285[8];
        const int value_idx_286 = 0;
        const int acc_idx_287 = 8 + value_idx_286;
        float2 _f2_741 = make_float2(acc[acc_idx_287], acc[acc_idx_287 + 1]);
        float2 scaled_pair_288 = mul_f32x2_noftz(_f2_741, rsigma_pair);
        float2 _f2_742 = make_float2(wout[acc_idx_287], wout[acc_idx_287 + 1]);
        float2 normalized_pair_289 = mul_f32x2_noftz(scaled_pair_288, _f2_742);
        output_values_285[value_idx_286] = normalized_pair_289.x;
        output_values_285[value_idx_286 + 1] = normalized_pair_289.y;
        const int value_idx_290 = 2;
        const int acc_idx_291 = 8 + value_idx_290;
        float2 _f2_743 = make_float2(acc[acc_idx_291], acc[acc_idx_291 + 1]);
        float2 scaled_pair_292 = mul_f32x2_noftz(_f2_743, rsigma_pair);
        float2 _f2_744 = make_float2(wout[acc_idx_291], wout[acc_idx_291 + 1]);
        float2 normalized_pair_293 = mul_f32x2_noftz(scaled_pair_292, _f2_744);
        output_values_285[value_idx_290] = normalized_pair_293.x;
        output_values_285[value_idx_290 + 1] = normalized_pair_293.y;
        const int value_idx_294 = 4;
        const int acc_idx_295 = 8 + value_idx_294;
        float2 _f2_745 = make_float2(acc[acc_idx_295], acc[acc_idx_295 + 1]);
        float2 scaled_pair_296 = mul_f32x2_noftz(_f2_745, rsigma_pair);
        float2 _f2_746 = make_float2(wout[acc_idx_295], wout[acc_idx_295 + 1]);
        float2 normalized_pair_297 = mul_f32x2_noftz(scaled_pair_296, _f2_746);
        output_values_285[value_idx_294] = normalized_pair_297.x;
        output_values_285[value_idx_294 + 1] = normalized_pair_297.y;
        const int value_idx_298 = 6;
        const int acc_idx_299 = 8 + value_idx_298;
        float2 _f2_747 = make_float2(acc[acc_idx_299], acc[acc_idx_299 + 1]);
        float2 scaled_pair_300 = mul_f32x2_noftz(_f2_747, rsigma_pair);
        float2 _f2_748 = make_float2(wout[acc_idx_299], wout[acc_idx_299 + 1]);
        float2 normalized_pair_301 = mul_f32x2_noftz(scaled_pair_300, _f2_748);
        output_values_285[value_idx_298] = normalized_pair_301.x;
        output_values_285[value_idx_298 + 1] = normalized_pair_301.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_285[0 + 0], output_values_285[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_285[0 + 2], output_values_285[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_285[0 + 4], output_values_285[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_285[0 + 6], output_values_285[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_284 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_302 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_303[8];
        const int value_idx_304 = 0;
        const int acc_idx_305 = 16 + value_idx_304;
        float2 _f2_749 = make_float2(acc[acc_idx_305], acc[acc_idx_305 + 1]);
        float2 scaled_pair_306 = mul_f32x2_noftz(_f2_749, rsigma_pair);
        float2 _f2_750 = make_float2(wout[acc_idx_305], wout[acc_idx_305 + 1]);
        float2 normalized_pair_307 = mul_f32x2_noftz(scaled_pair_306, _f2_750);
        output_values_303[value_idx_304] = normalized_pair_307.x;
        output_values_303[value_idx_304 + 1] = normalized_pair_307.y;
        const int value_idx_308 = 2;
        const int acc_idx_309 = 16 + value_idx_308;
        float2 _f2_751 = make_float2(acc[acc_idx_309], acc[acc_idx_309 + 1]);
        float2 scaled_pair_310 = mul_f32x2_noftz(_f2_751, rsigma_pair);
        float2 _f2_752 = make_float2(wout[acc_idx_309], wout[acc_idx_309 + 1]);
        float2 normalized_pair_311 = mul_f32x2_noftz(scaled_pair_310, _f2_752);
        output_values_303[value_idx_308] = normalized_pair_311.x;
        output_values_303[value_idx_308 + 1] = normalized_pair_311.y;
        const int value_idx_312 = 4;
        const int acc_idx_313 = 16 + value_idx_312;
        float2 _f2_753 = make_float2(acc[acc_idx_313], acc[acc_idx_313 + 1]);
        float2 scaled_pair_314 = mul_f32x2_noftz(_f2_753, rsigma_pair);
        float2 _f2_754 = make_float2(wout[acc_idx_313], wout[acc_idx_313 + 1]);
        float2 normalized_pair_315 = mul_f32x2_noftz(scaled_pair_314, _f2_754);
        output_values_303[value_idx_312] = normalized_pair_315.x;
        output_values_303[value_idx_312 + 1] = normalized_pair_315.y;
        const int value_idx_316 = 6;
        const int acc_idx_317 = 16 + value_idx_316;
        float2 _f2_755 = make_float2(acc[acc_idx_317], acc[acc_idx_317 + 1]);
        float2 scaled_pair_318 = mul_f32x2_noftz(_f2_755, rsigma_pair);
        float2 _f2_756 = make_float2(wout[acc_idx_317], wout[acc_idx_317 + 1]);
        float2 normalized_pair_319 = mul_f32x2_noftz(scaled_pair_318, _f2_756);
        output_values_303[value_idx_316] = normalized_pair_319.x;
        output_values_303[value_idx_316 + 1] = normalized_pair_319.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_303[0 + 0], output_values_303[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_303[0 + 2], output_values_303[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_303[0 + 4], output_values_303[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_303[0 + 6], output_values_303[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_302 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_320 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_321[8];
        const int value_idx_322 = 0;
        const int acc_idx_323 = 24 + value_idx_322;
        float2 _f2_757 = make_float2(acc[acc_idx_323], acc[acc_idx_323 + 1]);
        float2 scaled_pair_324 = mul_f32x2_noftz(_f2_757, rsigma_pair);
        float2 _f2_758 = make_float2(wout[acc_idx_323], wout[acc_idx_323 + 1]);
        float2 normalized_pair_325 = mul_f32x2_noftz(scaled_pair_324, _f2_758);
        output_values_321[value_idx_322] = normalized_pair_325.x;
        output_values_321[value_idx_322 + 1] = normalized_pair_325.y;
        const int value_idx_326 = 2;
        const int acc_idx_327 = 24 + value_idx_326;
        float2 _f2_759 = make_float2(acc[acc_idx_327], acc[acc_idx_327 + 1]);
        float2 scaled_pair_328 = mul_f32x2_noftz(_f2_759, rsigma_pair);
        float2 _f2_760 = make_float2(wout[acc_idx_327], wout[acc_idx_327 + 1]);
        float2 normalized_pair_329 = mul_f32x2_noftz(scaled_pair_328, _f2_760);
        output_values_321[value_idx_326] = normalized_pair_329.x;
        output_values_321[value_idx_326 + 1] = normalized_pair_329.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_321[0 + 0], output_values_321[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_321[0 + 2], output_values_321[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_320]) = _pk2;
        }
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
