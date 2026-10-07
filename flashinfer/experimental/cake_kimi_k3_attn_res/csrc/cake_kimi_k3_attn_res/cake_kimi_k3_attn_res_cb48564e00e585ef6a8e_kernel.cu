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
#define SMEM_STATS_STAGE_BYTES 512
#define SMEM_STATS_STRIDE 512
#define SMEM_OUT_STATS_OFF 512
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
kernel_cake_kimi_k3_attn_res_cb48564e00e585ef6a8e(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 512);
    const int out_stats_addr = smem + 512;

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
        unsigned int words[112];
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
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base))) + 0);
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
                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
                _vec_load_24[0 + 0] = _uv4_8.x;
                _vec_load_24[0 + 1] = _uv4_8.y;
                _vec_load_24[0 + 2] = _uv4_8.z;
                _vec_load_24[0 + 3] = _uv4_8.w;
            }
            dwords[woff] = _vec_load_24[0];
            dwords[woff + 1] = _vec_load_24[1];
            dwords[woff + 2] = _vec_load_24[2];
            dwords[woff + 3] = _vec_load_24[3];
        }
        float _vec_load_27[8];
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
                        : "=f"((&_vec_load_27[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_27[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        float _vec_load_28[8];
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
                        : "=f"((&_vec_load_28[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_28[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        q[0] = _vec_load_27[0] * _vec_load_28[0];
        q[1] = _vec_load_27[1] * _vec_load_28[1];
        q[2] = _vec_load_27[2] * _vec_load_28[2];
        q[3] = _vec_load_27[3] * _vec_load_28[3];
        q[4] = _vec_load_27[4] * _vec_load_28[4];
        q[5] = _vec_load_27[5] * _vec_load_28[5];
        q[6] = _vec_load_27[6] * _vec_load_28[6];
        q[7] = _vec_load_27[7] * _vec_load_28[7];
        float _vec_load_29[8];
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
                        : "=f"((&_vec_load_29[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_29[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        wout[0] = _vec_load_29[0];
        wout[1] = _vec_load_29[1];
        wout[2] = _vec_load_29[2];
        wout[3] = _vec_load_29[3];
        wout[4] = _vec_load_29[4];
        wout[5] = _vec_load_29[5];
        wout[6] = _vec_load_29[6];
        wout[7] = _vec_load_29[7];
        int base_0 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_1 = 4;
        {
            unsigned int _vec_load_30[4];
            {
                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0))) + 0);
                _vec_load_30[0 + 0] = _uv4_12.x;
                _vec_load_30[0 + 1] = _uv4_12.y;
                _vec_load_30[0 + 2] = _uv4_12.z;
                _vec_load_30[0 + 3] = _uv4_12.w;
            }
            words[woff_1] = _vec_load_30[0];
            words[woff_1 + 1] = _vec_load_30[1];
            words[woff_1 + 2] = _vec_load_30[2];
            words[woff_1 + 3] = _vec_load_30[3];
        }
        {
            unsigned int _vec_load_33[4];
            {
                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_33[0 + 0] = _uv4_13.x;
                _vec_load_33[0 + 1] = _uv4_13.y;
                _vec_load_33[0 + 2] = _uv4_13.z;
                _vec_load_33[0 + 3] = _uv4_13.w;
            }
            words[14 + woff_1] = _vec_load_33[0];
            words[14 + woff_1 + 1] = _vec_load_33[1];
            words[14 + woff_1 + 2] = _vec_load_33[2];
            words[14 + woff_1 + 3] = _vec_load_33[3];
        }
        {
            unsigned int _vec_load_36[4];
            {
                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_36[0 + 0] = _uv4_14.x;
                _vec_load_36[0 + 1] = _uv4_14.y;
                _vec_load_36[0 + 2] = _uv4_14.z;
                _vec_load_36[0 + 3] = _uv4_14.w;
            }
            words[28 + woff_1] = _vec_load_36[0];
            words[28 + woff_1 + 1] = _vec_load_36[1];
            words[28 + woff_1 + 2] = _vec_load_36[2];
            words[28 + woff_1 + 3] = _vec_load_36[3];
        }
        {
            unsigned int _vec_load_39[4];
            {
                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_39[0 + 0] = _uv4_15.x;
                _vec_load_39[0 + 1] = _uv4_15.y;
                _vec_load_39[0 + 2] = _uv4_15.z;
                _vec_load_39[0 + 3] = _uv4_15.w;
            }
            words[42 + woff_1] = _vec_load_39[0];
            words[42 + woff_1 + 1] = _vec_load_39[1];
            words[42 + woff_1 + 2] = _vec_load_39[2];
            words[42 + woff_1 + 3] = _vec_load_39[3];
        }
        {
            unsigned int _vec_load_42[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_42[0 + 0] = _uv4_16.x;
                _vec_load_42[0 + 1] = _uv4_16.y;
                _vec_load_42[0 + 2] = _uv4_16.z;
                _vec_load_42[0 + 3] = _uv4_16.w;
            }
            words[56 + woff_1] = _vec_load_42[0];
            words[56 + woff_1 + 1] = _vec_load_42[1];
            words[56 + woff_1 + 2] = _vec_load_42[2];
            words[56 + woff_1 + 3] = _vec_load_42[3];
        }
        {
            unsigned int _vec_load_45[4];
            {
                uint4 _uv4_17 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_45[0 + 0] = _uv4_17.x;
                _vec_load_45[0 + 1] = _uv4_17.y;
                _vec_load_45[0 + 2] = _uv4_17.z;
                _vec_load_45[0 + 3] = _uv4_17.w;
            }
            words[70 + woff_1] = _vec_load_45[0];
            words[70 + woff_1 + 1] = _vec_load_45[1];
            words[70 + woff_1 + 2] = _vec_load_45[2];
            words[70 + woff_1 + 3] = _vec_load_45[3];
        }
        {
            unsigned int _vec_load_48[4];
            {
                uint4 _uv4_18 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_48[0 + 0] = _uv4_18.x;
                _vec_load_48[0 + 1] = _uv4_18.y;
                _vec_load_48[0 + 2] = _uv4_18.z;
                _vec_load_48[0 + 3] = _uv4_18.w;
            }
            words[84 + woff_1] = _vec_load_48[0];
            words[84 + woff_1 + 1] = _vec_load_48[1];
            words[84 + woff_1 + 2] = _vec_load_48[2];
            words[84 + woff_1 + 3] = _vec_load_48[3];
        }
        {
            unsigned int _vec_load_51[4];
            {
                uint4 _uv4_19 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_0))) + 0);
                _vec_load_51[0 + 0] = _uv4_19.x;
                _vec_load_51[0 + 1] = _uv4_19.y;
                _vec_load_51[0 + 2] = _uv4_19.z;
                _vec_load_51[0 + 3] = _uv4_19.w;
            }
            words[98 + woff_1] = _vec_load_51[0];
            words[98 + woff_1 + 1] = _vec_load_51[1];
            words[98 + woff_1 + 2] = _vec_load_51[2];
            words[98 + woff_1 + 3] = _vec_load_51[3];
        }
        {
            unsigned int _vec_load_54[4];
            {
                uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
                _vec_load_54[0 + 0] = _uv4_20.x;
                _vec_load_54[0 + 1] = _uv4_20.y;
                _vec_load_54[0 + 2] = _uv4_20.z;
                _vec_load_54[0 + 3] = _uv4_20.w;
            }
            dwords[woff_1] = _vec_load_54[0];
            dwords[woff_1 + 1] = _vec_load_54[1];
            dwords[woff_1 + 2] = _vec_load_54[2];
            dwords[woff_1 + 3] = _vec_load_54[3];
        }
        float _vec_load_57[8];
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
                        : "=f"((&_vec_load_57[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_57[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_21[_pair]));
                }
            }
        }
        float _vec_load_58[8];
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
                        : "=f"((&_vec_load_58[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_58[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_22[_pair]));
                }
            }
        }
        q[8] = _vec_load_57[0] * _vec_load_58[0];
        q[9] = _vec_load_57[1] * _vec_load_58[1];
        q[10] = _vec_load_57[2] * _vec_load_58[2];
        q[11] = _vec_load_57[3] * _vec_load_58[3];
        q[12] = _vec_load_57[4] * _vec_load_58[4];
        q[13] = _vec_load_57[5] * _vec_load_58[5];
        q[14] = _vec_load_57[6] * _vec_load_58[6];
        q[15] = _vec_load_57[7] * _vec_load_58[7];
        float _vec_load_59[8];
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
                        : "=f"((&_vec_load_59[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_59[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_23[_pair]));
                }
            }
        }
        wout[8] = _vec_load_59[0];
        wout[9] = _vec_load_59[1];
        wout[10] = _vec_load_59[2];
        wout[11] = _vec_load_59[3];
        wout[12] = _vec_load_59[4];
        wout[13] = _vec_load_59[5];
        wout[14] = _vec_load_59[6];
        wout[15] = _vec_load_59[7];
        int base_2 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_3 = 8;
        {
            unsigned int _vec_load_60[4];
            {
                uint4 _uv4_24 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2))) + 0);
                _vec_load_60[0 + 0] = _uv4_24.x;
                _vec_load_60[0 + 1] = _uv4_24.y;
                _vec_load_60[0 + 2] = _uv4_24.z;
                _vec_load_60[0 + 3] = _uv4_24.w;
            }
            words[woff_3] = _vec_load_60[0];
            words[woff_3 + 1] = _vec_load_60[1];
            words[woff_3 + 2] = _vec_load_60[2];
            words[woff_3 + 3] = _vec_load_60[3];
        }
        {
            unsigned int _vec_load_63[4];
            {
                uint4 _uv4_25 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_63[0 + 0] = _uv4_25.x;
                _vec_load_63[0 + 1] = _uv4_25.y;
                _vec_load_63[0 + 2] = _uv4_25.z;
                _vec_load_63[0 + 3] = _uv4_25.w;
            }
            words[14 + woff_3] = _vec_load_63[0];
            words[14 + woff_3 + 1] = _vec_load_63[1];
            words[14 + woff_3 + 2] = _vec_load_63[2];
            words[14 + woff_3 + 3] = _vec_load_63[3];
        }
        {
            unsigned int _vec_load_66[4];
            {
                uint4 _uv4_26 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_66[0 + 0] = _uv4_26.x;
                _vec_load_66[0 + 1] = _uv4_26.y;
                _vec_load_66[0 + 2] = _uv4_26.z;
                _vec_load_66[0 + 3] = _uv4_26.w;
            }
            words[28 + woff_3] = _vec_load_66[0];
            words[28 + woff_3 + 1] = _vec_load_66[1];
            words[28 + woff_3 + 2] = _vec_load_66[2];
            words[28 + woff_3 + 3] = _vec_load_66[3];
        }
        {
            unsigned int _vec_load_69[4];
            {
                uint4 _uv4_27 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_69[0 + 0] = _uv4_27.x;
                _vec_load_69[0 + 1] = _uv4_27.y;
                _vec_load_69[0 + 2] = _uv4_27.z;
                _vec_load_69[0 + 3] = _uv4_27.w;
            }
            words[42 + woff_3] = _vec_load_69[0];
            words[42 + woff_3 + 1] = _vec_load_69[1];
            words[42 + woff_3 + 2] = _vec_load_69[2];
            words[42 + woff_3 + 3] = _vec_load_69[3];
        }
        {
            unsigned int _vec_load_72[4];
            {
                uint4 _uv4_28 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_72[0 + 0] = _uv4_28.x;
                _vec_load_72[0 + 1] = _uv4_28.y;
                _vec_load_72[0 + 2] = _uv4_28.z;
                _vec_load_72[0 + 3] = _uv4_28.w;
            }
            words[56 + woff_3] = _vec_load_72[0];
            words[56 + woff_3 + 1] = _vec_load_72[1];
            words[56 + woff_3 + 2] = _vec_load_72[2];
            words[56 + woff_3 + 3] = _vec_load_72[3];
        }
        {
            unsigned int _vec_load_75[4];
            {
                uint4 _uv4_29 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_75[0 + 0] = _uv4_29.x;
                _vec_load_75[0 + 1] = _uv4_29.y;
                _vec_load_75[0 + 2] = _uv4_29.z;
                _vec_load_75[0 + 3] = _uv4_29.w;
            }
            words[70 + woff_3] = _vec_load_75[0];
            words[70 + woff_3 + 1] = _vec_load_75[1];
            words[70 + woff_3 + 2] = _vec_load_75[2];
            words[70 + woff_3 + 3] = _vec_load_75[3];
        }
        {
            unsigned int _vec_load_78[4];
            {
                uint4 _uv4_30 = *reinterpret_cast<const uint4*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_78[0 + 0] = _uv4_30.x;
                _vec_load_78[0 + 1] = _uv4_30.y;
                _vec_load_78[0 + 2] = _uv4_30.z;
                _vec_load_78[0 + 3] = _uv4_30.w;
            }
            words[84 + woff_3] = _vec_load_78[0];
            words[84 + woff_3 + 1] = _vec_load_78[1];
            words[84 + woff_3 + 2] = _vec_load_78[2];
            words[84 + woff_3 + 3] = _vec_load_78[3];
        }
        {
            unsigned int _vec_load_81[4];
            {
                uint4 _uv4_31 = *reinterpret_cast<const uint4*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_2))) + 0);
                _vec_load_81[0 + 0] = _uv4_31.x;
                _vec_load_81[0 + 1] = _uv4_31.y;
                _vec_load_81[0 + 2] = _uv4_31.z;
                _vec_load_81[0 + 3] = _uv4_31.w;
            }
            words[98 + woff_3] = _vec_load_81[0];
            words[98 + woff_3 + 1] = _vec_load_81[1];
            words[98 + woff_3 + 2] = _vec_load_81[2];
            words[98 + woff_3 + 3] = _vec_load_81[3];
        }
        {
            unsigned int _vec_load_84[4];
            {
                uint4 _uv4_32 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
                _vec_load_84[0 + 0] = _uv4_32.x;
                _vec_load_84[0 + 1] = _uv4_32.y;
                _vec_load_84[0 + 2] = _uv4_32.z;
                _vec_load_84[0 + 3] = _uv4_32.w;
            }
            dwords[woff_3] = _vec_load_84[0];
            dwords[woff_3 + 1] = _vec_load_84[1];
            dwords[woff_3 + 2] = _vec_load_84[2];
            dwords[woff_3 + 3] = _vec_load_84[3];
        }
        float _vec_load_87[8];
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
                        : "=f"((&_vec_load_87[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_87[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_33[_pair]));
                }
            }
        }
        float _vec_load_88[8];
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
                        : "=f"((&_vec_load_88[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_88[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_34[_pair]));
                }
            }
        }
        q[16] = _vec_load_87[0] * _vec_load_88[0];
        q[17] = _vec_load_87[1] * _vec_load_88[1];
        q[18] = _vec_load_87[2] * _vec_load_88[2];
        q[19] = _vec_load_87[3] * _vec_load_88[3];
        q[20] = _vec_load_87[4] * _vec_load_88[4];
        q[21] = _vec_load_87[5] * _vec_load_88[5];
        q[22] = _vec_load_87[6] * _vec_load_88[6];
        q[23] = _vec_load_87[7] * _vec_load_88[7];
        float _vec_load_89[8];
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
                        : "=f"((&_vec_load_89[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_89[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_35[_pair]));
                }
            }
        }
        wout[16] = _vec_load_89[0];
        wout[17] = _vec_load_89[1];
        wout[18] = _vec_load_89[2];
        wout[19] = _vec_load_89[3];
        wout[20] = _vec_load_89[4];
        wout[21] = _vec_load_89[5];
        wout[22] = _vec_load_89[6];
        wout[23] = _vec_load_89[7];
        int base_4 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_5 = 12;
        {
            unsigned int _vec_load_91[1];
            {
                _vec_load_91[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 0);
            }
            words[woff_5] = _vec_load_91[0];
            unsigned int _vec_load_92[1];
            {
                _vec_load_92[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4))) + 1);
            }
            words[woff_5 + 1] = _vec_load_92[0];
        }
        {
            unsigned int _vec_load_94[1];
            {
                _vec_load_94[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[14 + woff_5] = _vec_load_94[0];
            unsigned int _vec_load_95[1];
            {
                _vec_load_95[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[14 + woff_5 + 1] = _vec_load_95[0];
        }
        {
            unsigned int _vec_load_97[1];
            {
                _vec_load_97[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[28 + woff_5] = _vec_load_97[0];
            unsigned int _vec_load_98[1];
            {
                _vec_load_98[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[28 + woff_5 + 1] = _vec_load_98[0];
        }
        {
            unsigned int _vec_load_100[1];
            {
                _vec_load_100[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[42 + woff_5] = _vec_load_100[0];
            unsigned int _vec_load_101[1];
            {
                _vec_load_101[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[42 + woff_5 + 1] = _vec_load_101[0];
        }
        {
            unsigned int _vec_load_103[1];
            {
                _vec_load_103[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[56 + woff_5] = _vec_load_103[0];
            unsigned int _vec_load_104[1];
            {
                _vec_load_104[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[56 + woff_5 + 1] = _vec_load_104[0];
        }
        {
            unsigned int _vec_load_106[1];
            {
                _vec_load_106[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[70 + woff_5] = _vec_load_106[0];
            unsigned int _vec_load_107[1];
            {
                _vec_load_107[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[70 + woff_5 + 1] = _vec_load_107[0];
        }
        {
            unsigned int _vec_load_109[1];
            {
                _vec_load_109[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[84 + woff_5] = _vec_load_109[0];
            unsigned int _vec_load_110[1];
            {
                _vec_load_110[0] = *reinterpret_cast<const unsigned int*>(((0) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[84 + woff_5 + 1] = _vec_load_110[0];
        }
        {
            unsigned int _vec_load_112[1];
            {
                _vec_load_112[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_4))) + 0);
            }
            words[98 + woff_5] = _vec_load_112[0];
            unsigned int _vec_load_113[1];
            {
                _vec_load_113[0] = *reinterpret_cast<const unsigned int*>(((1) ? reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) : reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_4))) + 1);
            }
            words[98 + woff_5 + 1] = _vec_load_113[0];
        }
        {
            unsigned int _vec_load_115[1];
            {
                _vec_load_115[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
            }
            dwords[woff_5] = _vec_load_115[0];
            unsigned int _vec_load_116[1];
            {
                _vec_load_116[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
            }
            dwords[woff_5 + 1] = _vec_load_116[0];
        }
        float _vec_load_117[4];
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
                    : "=f"((&_vec_load_117[0 + _pair * 2])[0]), "=f"((&_vec_load_117[0 + _pair * 2])[1])
                    : "r"(_vpairs_36[_pair]));
            }
        }
        float _vec_load_118[4];
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
                    : "=f"((&_vec_load_118[0 + _pair * 2])[0]), "=f"((&_vec_load_118[0 + _pair * 2])[1])
                    : "r"(_vpairs_37[_pair]));
            }
        }
        q[24] = _vec_load_117[0] * _vec_load_118[0];
        q[25] = _vec_load_117[1] * _vec_load_118[1];
        q[26] = _vec_load_117[2] * _vec_load_118[2];
        q[27] = _vec_load_117[3] * _vec_load_118[3];
        float _vec_load_119[4];
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
                    : "=f"((&_vec_load_119[0 + _pair * 2])[0]), "=f"((&_vec_load_119[0 + _pair * 2])[1])
                    : "r"(_vpairs_38[_pair]));
            }
        }
        wout[24] = _vec_load_119[0];
        wout[25] = _vec_load_119[1];
        wout[26] = _vec_load_119[2];
        wout[27] = _vec_load_119[3];
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
            float2 _f2_22 = make_float2(sw_f32[0], sw_f32[1]);
            float2 v = _f2_22;
            float2 _f2_23 = make_float2(q[0], q[1]);
            float2 qp = _f2_23;
            sq[0] = fma_f32x2_rn_noftz(v, v, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v, qp, dot[0]);
            float2 _f2_24 = make_float2(sw_f32[2], sw_f32[3]);
            float2 v_0 = _f2_24;
            float2 _f2_25 = make_float2(q[2], q[3]);
            float2 qp_1 = _f2_25;
            sq[0] = fma_f32x2_rn_noftz(v_0, v_0, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0, qp_1, dot[0]);
            float2 _f2_26 = make_float2(sw_f32[4], sw_f32[5]);
            float2 v_2 = _f2_26;
            float2 _f2_27 = make_float2(q[4], q[5]);
            float2 qp_3 = _f2_27;
            sq[0] = fma_f32x2_rn_noftz(v_2, v_2, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2, qp_3, dot[0]);
            float2 _f2_28 = make_float2(sw_f32[6], sw_f32[7]);
            float2 v_4 = _f2_28;
            float2 _f2_29 = make_float2(q[6], q[7]);
            float2 qp_5 = _f2_29;
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
            float2 _f2_36 = make_float2(sw_8_f32[0], sw_8_f32[1]);
            float2 v_1 = _f2_36;
            float2 _f2_37 = make_float2(q[0], q[1]);
            float2 qp_2 = _f2_37;
            sq[1] = fma_f32x2_rn_noftz(v_1, v_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_1, qp_2, dot[1]);
            float2 _f2_38 = make_float2(sw_8_f32[2], sw_8_f32[3]);
            float2 v_0_1 = _f2_38;
            float2 _f2_39 = make_float2(q[2], q[3]);
            float2 qp_1_1 = _f2_39;
            sq[1] = fma_f32x2_rn_noftz(v_0_1, v_0_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_1, qp_1_1, dot[1]);
            float2 _f2_40 = make_float2(sw_8_f32[4], sw_8_f32[5]);
            float2 v_2_1 = _f2_40;
            float2 _f2_41 = make_float2(q[4], q[5]);
            float2 qp_3_1 = _f2_41;
            sq[1] = fma_f32x2_rn_noftz(v_2_1, v_2_1, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_1, qp_3_1, dot[1]);
            float2 _f2_42 = make_float2(sw_8_f32[6], sw_8_f32[7]);
            float2 v_4_1 = _f2_42;
            float2 _f2_43 = make_float2(q[6], q[7]);
            float2 qp_5_1 = _f2_43;
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
            float2 _f2_50 = make_float2(sw_9_f32[0], sw_9_f32[1]);
            float2 v_3 = _f2_50;
            float2 _f2_51 = make_float2(q[0], q[1]);
            float2 qp_4 = _f2_51;
            sq[2] = fma_f32x2_rn_noftz(v_3, v_3, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_3, qp_4, dot[2]);
            float2 _f2_52 = make_float2(sw_9_f32[2], sw_9_f32[3]);
            float2 v_0_2 = _f2_52;
            float2 _f2_53 = make_float2(q[2], q[3]);
            float2 qp_1_2 = _f2_53;
            sq[2] = fma_f32x2_rn_noftz(v_0_2, v_0_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_2, qp_1_2, dot[2]);
            float2 _f2_54 = make_float2(sw_9_f32[4], sw_9_f32[5]);
            float2 v_2_2 = _f2_54;
            float2 _f2_55 = make_float2(q[4], q[5]);
            float2 qp_3_2 = _f2_55;
            sq[2] = fma_f32x2_rn_noftz(v_2_2, v_2_2, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_2, qp_3_2, dot[2]);
            float2 _f2_56 = make_float2(sw_9_f32[6], sw_9_f32[7]);
            float2 v_4_2 = _f2_56;
            float2 _f2_57 = make_float2(q[6], q[7]);
            float2 qp_5_2 = _f2_57;
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
            float2 _f2_64 = make_float2(sw_10_f32[0], sw_10_f32[1]);
            float2 v_5 = _f2_64;
            float2 _f2_65 = make_float2(q[0], q[1]);
            float2 qp_6 = _f2_65;
            sq[3] = fma_f32x2_rn_noftz(v_5, v_5, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_5, qp_6, dot[3]);
            float2 _f2_66 = make_float2(sw_10_f32[2], sw_10_f32[3]);
            float2 v_0_3 = _f2_66;
            float2 _f2_67 = make_float2(q[2], q[3]);
            float2 qp_1_3 = _f2_67;
            sq[3] = fma_f32x2_rn_noftz(v_0_3, v_0_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_3, qp_1_3, dot[3]);
            float2 _f2_68 = make_float2(sw_10_f32[4], sw_10_f32[5]);
            float2 v_2_3 = _f2_68;
            float2 _f2_69 = make_float2(q[4], q[5]);
            float2 qp_3_3 = _f2_69;
            sq[3] = fma_f32x2_rn_noftz(v_2_3, v_2_3, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_3, qp_3_3, dot[3]);
            float2 _f2_70 = make_float2(sw_10_f32[6], sw_10_f32[7]);
            float2 v_4_3 = _f2_70;
            float2 _f2_71 = make_float2(q[6], q[7]);
            float2 qp_5_3 = _f2_71;
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
            float2 _f2_78 = make_float2(sw_11_f32[0], sw_11_f32[1]);
            float2 v_6 = _f2_78;
            float2 _f2_79 = make_float2(q[0], q[1]);
            float2 qp_7 = _f2_79;
            sq[4] = fma_f32x2_rn_noftz(v_6, v_6, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_6, qp_7, dot[4]);
            float2 _f2_80 = make_float2(sw_11_f32[2], sw_11_f32[3]);
            float2 v_0_4 = _f2_80;
            float2 _f2_81 = make_float2(q[2], q[3]);
            float2 qp_1_4 = _f2_81;
            sq[4] = fma_f32x2_rn_noftz(v_0_4, v_0_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_4, qp_1_4, dot[4]);
            float2 _f2_82 = make_float2(sw_11_f32[4], sw_11_f32[5]);
            float2 v_2_4 = _f2_82;
            float2 _f2_83 = make_float2(q[4], q[5]);
            float2 qp_3_4 = _f2_83;
            sq[4] = fma_f32x2_rn_noftz(v_2_4, v_2_4, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_4, qp_3_4, dot[4]);
            float2 _f2_84 = make_float2(sw_11_f32[6], sw_11_f32[7]);
            float2 v_4_4 = _f2_84;
            float2 _f2_85 = make_float2(q[6], q[7]);
            float2 qp_5_4 = _f2_85;
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
            float2 _f2_92 = make_float2(sw_12_f32[0], sw_12_f32[1]);
            float2 v_7 = _f2_92;
            float2 _f2_93 = make_float2(q[0], q[1]);
            float2 qp_8 = _f2_93;
            sq[5] = fma_f32x2_rn_noftz(v_7, v_7, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_7, qp_8, dot[5]);
            float2 _f2_94 = make_float2(sw_12_f32[2], sw_12_f32[3]);
            float2 v_0_5 = _f2_94;
            float2 _f2_95 = make_float2(q[2], q[3]);
            float2 qp_1_5 = _f2_95;
            sq[5] = fma_f32x2_rn_noftz(v_0_5, v_0_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_5, qp_1_5, dot[5]);
            float2 _f2_96 = make_float2(sw_12_f32[4], sw_12_f32[5]);
            float2 v_2_5 = _f2_96;
            float2 _f2_97 = make_float2(q[4], q[5]);
            float2 qp_3_5 = _f2_97;
            sq[5] = fma_f32x2_rn_noftz(v_2_5, v_2_5, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_5, qp_3_5, dot[5]);
            float2 _f2_98 = make_float2(sw_12_f32[6], sw_12_f32[7]);
            float2 v_4_5 = _f2_98;
            float2 _f2_99 = make_float2(q[6], q[7]);
            float2 qp_5_5 = _f2_99;
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
            float2 _f2_106 = make_float2(sw_13_f32[0], sw_13_f32[1]);
            float2 v_8 = _f2_106;
            float2 _f2_107 = make_float2(q[0], q[1]);
            float2 qp_9 = _f2_107;
            sq[6] = fma_f32x2_rn_noftz(v_8, v_8, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_8, qp_9, dot[6]);
            float2 _f2_108 = make_float2(sw_13_f32[2], sw_13_f32[3]);
            float2 v_0_6 = _f2_108;
            float2 _f2_109 = make_float2(q[2], q[3]);
            float2 qp_1_6 = _f2_109;
            sq[6] = fma_f32x2_rn_noftz(v_0_6, v_0_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_6, qp_1_6, dot[6]);
            float2 _f2_110 = make_float2(sw_13_f32[4], sw_13_f32[5]);
            float2 v_2_6 = _f2_110;
            float2 _f2_111 = make_float2(q[4], q[5]);
            float2 qp_3_6 = _f2_111;
            sq[6] = fma_f32x2_rn_noftz(v_2_6, v_2_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_6, qp_3_6, dot[6]);
            float2 _f2_112 = make_float2(sw_13_f32[6], sw_13_f32[7]);
            float2 v_4_6 = _f2_112;
            float2 _f2_113 = make_float2(q[6], q[7]);
            float2 qp_5_6 = _f2_113;
            sq[6] = fma_f32x2_rn_noftz(v_4_6, v_4_6, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_6, qp_5_6, dot[6]);
        }
        unsigned int sw_14[4];
        sw_14[0] = words[98 + woff_7];
        sw_14[1] = words[98 + woff_7 + 1];
        sw_14[2] = words[98 + woff_7 + 2];
        sw_14[3] = words[98 + woff_7 + 3];
        {
            __nv_bfloat162 a = __as_bf16x2(sw_14[0]);
            __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
            __nv_bfloat162 mixed = a + d;
            sw_14[0] = __as_u32(mixed);
            words[98 + woff_7] = sw_14[0];
            __nv_bfloat162 a_0 = __as_bf16x2(sw_14[1]);
            __nv_bfloat162 d_1 = __as_bf16x2(dwords[woff_7 + 1]);
            __nv_bfloat162 mixed_2 = a_0 + d_1;
            sw_14[1] = __as_u32(mixed_2);
            words[98 + woff_7 + 1] = sw_14[1];
            __nv_bfloat162 a_3 = __as_bf16x2(sw_14[2]);
            __nv_bfloat162 d_4 = __as_bf16x2(dwords[woff_7 + 2]);
            __nv_bfloat162 mixed_5 = a_3 + d_4;
            sw_14[2] = __as_u32(mixed_5);
            words[98 + woff_7 + 2] = sw_14[2];
            __nv_bfloat162 a_6 = __as_bf16x2(sw_14[3]);
            __nv_bfloat162 d_7 = __as_bf16x2(dwords[woff_7 + 3]);
            __nv_bfloat162 mixed_8 = a_6 + d_7;
            sw_14[3] = __as_u32(mixed_8);
            words[98 + woff_7 + 3] = sw_14[3];
            {
                int4 _iv4 = make_int4(sw_14[0 + 0], sw_14[0 + 1], sw_14[0 + 2], sw_14[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
            }
        }
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
            float2 _f2_120 = make_float2(sw_14_f32[0], sw_14_f32[1]);
            float2 v_9 = _f2_120;
            float2 _f2_121 = make_float2(q[0], q[1]);
            float2 qp_10 = _f2_121;
            sq[7] = fma_f32x2_rn_noftz(v_9, v_9, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_9, qp_10, dot[7]);
            float2 _f2_122 = make_float2(sw_14_f32[2], sw_14_f32[3]);
            float2 v_0_7 = _f2_122;
            float2 _f2_123 = make_float2(q[2], q[3]);
            float2 qp_1_7 = _f2_123;
            sq[7] = fma_f32x2_rn_noftz(v_0_7, v_0_7, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_0_7, qp_1_7, dot[7]);
            float2 _f2_124 = make_float2(sw_14_f32[4], sw_14_f32[5]);
            float2 v_2_7 = _f2_124;
            float2 _f2_125 = make_float2(q[4], q[5]);
            float2 qp_3_7 = _f2_125;
            sq[7] = fma_f32x2_rn_noftz(v_2_7, v_2_7, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_2_7, qp_3_7, dot[7]);
            float2 _f2_126 = make_float2(sw_14_f32[6], sw_14_f32[7]);
            float2 v_4_7 = _f2_126;
            float2 _f2_127 = make_float2(q[6], q[7]);
            float2 qp_5_7 = _f2_127;
            sq[7] = fma_f32x2_rn_noftz(v_4_7, v_4_7, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_4_7, qp_5_7, dot[7]);
        }
        int base_15 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        const int woff_16 = 4;
        unsigned int sw_17[4];
        sw_17[0] = words[woff_16];
        sw_17[1] = words[woff_16 + 1];
        sw_17[2] = words[woff_16 + 2];
        sw_17[3] = words[woff_16 + 3];
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
            fsrc[8] = sw_17_f32[0];
            fsrc[9] = sw_17_f32[1];
            fsrc[10] = sw_17_f32[2];
            fsrc[11] = sw_17_f32[3];
            fsrc[12] = sw_17_f32[4];
            fsrc[13] = sw_17_f32[5];
            fsrc[14] = sw_17_f32[6];
            fsrc[15] = sw_17_f32[7];
        }
        {
            float2 _f2_134 = make_float2(sw_17_f32[0], sw_17_f32[1]);
            float2 v_10 = _f2_134;
            float2 _f2_135 = make_float2(q[8], q[9]);
            float2 qp_11 = _f2_135;
            sq[0] = fma_f32x2_rn_noftz(v_10, v_10, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_10, qp_11, dot[0]);
            float2 _f2_136 = make_float2(sw_17_f32[2], sw_17_f32[3]);
            float2 v_0_8 = _f2_136;
            float2 _f2_137 = make_float2(q[10], q[11]);
            float2 qp_1_8 = _f2_137;
            sq[0] = fma_f32x2_rn_noftz(v_0_8, v_0_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_8, qp_1_8, dot[0]);
            float2 _f2_138 = make_float2(sw_17_f32[4], sw_17_f32[5]);
            float2 v_2_8 = _f2_138;
            float2 _f2_139 = make_float2(q[12], q[13]);
            float2 qp_3_8 = _f2_139;
            sq[0] = fma_f32x2_rn_noftz(v_2_8, v_2_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_8, qp_3_8, dot[0]);
            float2 _f2_140 = make_float2(sw_17_f32[6], sw_17_f32[7]);
            float2 v_4_8 = _f2_140;
            float2 _f2_141 = make_float2(q[14], q[15]);
            float2 qp_5_8 = _f2_141;
            sq[0] = fma_f32x2_rn_noftz(v_4_8, v_4_8, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_8, qp_5_8, dot[0]);
        }
        unsigned int sw_18[4];
        sw_18[0] = words[14 + woff_16];
        sw_18[1] = words[14 + woff_16 + 1];
        sw_18[2] = words[14 + woff_16 + 2];
        sw_18[3] = words[14 + woff_16 + 3];
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
            fsrc[36] = sw_18_f32[0];
            fsrc[37] = sw_18_f32[1];
            fsrc[38] = sw_18_f32[2];
            fsrc[39] = sw_18_f32[3];
            fsrc[40] = sw_18_f32[4];
            fsrc[41] = sw_18_f32[5];
            fsrc[42] = sw_18_f32[6];
            fsrc[43] = sw_18_f32[7];
        }
        {
            float2 _f2_148 = make_float2(sw_18_f32[0], sw_18_f32[1]);
            float2 v_11 = _f2_148;
            float2 _f2_149 = make_float2(q[8], q[9]);
            float2 qp_12 = _f2_149;
            sq[1] = fma_f32x2_rn_noftz(v_11, v_11, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_11, qp_12, dot[1]);
            float2 _f2_150 = make_float2(sw_18_f32[2], sw_18_f32[3]);
            float2 v_0_9 = _f2_150;
            float2 _f2_151 = make_float2(q[10], q[11]);
            float2 qp_1_9 = _f2_151;
            sq[1] = fma_f32x2_rn_noftz(v_0_9, v_0_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_9, qp_1_9, dot[1]);
            float2 _f2_152 = make_float2(sw_18_f32[4], sw_18_f32[5]);
            float2 v_2_9 = _f2_152;
            float2 _f2_153 = make_float2(q[12], q[13]);
            float2 qp_3_9 = _f2_153;
            sq[1] = fma_f32x2_rn_noftz(v_2_9, v_2_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_9, qp_3_9, dot[1]);
            float2 _f2_154 = make_float2(sw_18_f32[6], sw_18_f32[7]);
            float2 v_4_9 = _f2_154;
            float2 _f2_155 = make_float2(q[14], q[15]);
            float2 qp_5_9 = _f2_155;
            sq[1] = fma_f32x2_rn_noftz(v_4_9, v_4_9, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_9, qp_5_9, dot[1]);
        }
        unsigned int sw_19[4];
        sw_19[0] = words[28 + woff_16];
        sw_19[1] = words[28 + woff_16 + 1];
        sw_19[2] = words[28 + woff_16 + 2];
        sw_19[3] = words[28 + woff_16 + 3];
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
            fsrc[64] = sw_19_f32[0];
            fsrc[65] = sw_19_f32[1];
            fsrc[66] = sw_19_f32[2];
            fsrc[67] = sw_19_f32[3];
            fsrc[68] = sw_19_f32[4];
            fsrc[69] = sw_19_f32[5];
            fsrc[70] = sw_19_f32[6];
            fsrc[71] = sw_19_f32[7];
        }
        {
            float2 _f2_162 = make_float2(sw_19_f32[0], sw_19_f32[1]);
            float2 v_12 = _f2_162;
            float2 _f2_163 = make_float2(q[8], q[9]);
            float2 qp_13 = _f2_163;
            sq[2] = fma_f32x2_rn_noftz(v_12, v_12, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_12, qp_13, dot[2]);
            float2 _f2_164 = make_float2(sw_19_f32[2], sw_19_f32[3]);
            float2 v_0_10 = _f2_164;
            float2 _f2_165 = make_float2(q[10], q[11]);
            float2 qp_1_10 = _f2_165;
            sq[2] = fma_f32x2_rn_noftz(v_0_10, v_0_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_10, qp_1_10, dot[2]);
            float2 _f2_166 = make_float2(sw_19_f32[4], sw_19_f32[5]);
            float2 v_2_10 = _f2_166;
            float2 _f2_167 = make_float2(q[12], q[13]);
            float2 qp_3_10 = _f2_167;
            sq[2] = fma_f32x2_rn_noftz(v_2_10, v_2_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_10, qp_3_10, dot[2]);
            float2 _f2_168 = make_float2(sw_19_f32[6], sw_19_f32[7]);
            float2 v_4_10 = _f2_168;
            float2 _f2_169 = make_float2(q[14], q[15]);
            float2 qp_5_10 = _f2_169;
            sq[2] = fma_f32x2_rn_noftz(v_4_10, v_4_10, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_10, qp_5_10, dot[2]);
        }
        unsigned int sw_20[4];
        sw_20[0] = words[42 + woff_16];
        sw_20[1] = words[42 + woff_16 + 1];
        sw_20[2] = words[42 + woff_16 + 2];
        sw_20[3] = words[42 + woff_16 + 3];
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
            float2 _f2_176 = make_float2(sw_20_f32[0], sw_20_f32[1]);
            float2 v_13 = _f2_176;
            float2 _f2_177 = make_float2(q[8], q[9]);
            float2 qp_14 = _f2_177;
            sq[3] = fma_f32x2_rn_noftz(v_13, v_13, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_13, qp_14, dot[3]);
            float2 _f2_178 = make_float2(sw_20_f32[2], sw_20_f32[3]);
            float2 v_0_11 = _f2_178;
            float2 _f2_179 = make_float2(q[10], q[11]);
            float2 qp_1_11 = _f2_179;
            sq[3] = fma_f32x2_rn_noftz(v_0_11, v_0_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_11, qp_1_11, dot[3]);
            float2 _f2_180 = make_float2(sw_20_f32[4], sw_20_f32[5]);
            float2 v_2_11 = _f2_180;
            float2 _f2_181 = make_float2(q[12], q[13]);
            float2 qp_3_11 = _f2_181;
            sq[3] = fma_f32x2_rn_noftz(v_2_11, v_2_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_11, qp_3_11, dot[3]);
            float2 _f2_182 = make_float2(sw_20_f32[6], sw_20_f32[7]);
            float2 v_4_11 = _f2_182;
            float2 _f2_183 = make_float2(q[14], q[15]);
            float2 qp_5_11 = _f2_183;
            sq[3] = fma_f32x2_rn_noftz(v_4_11, v_4_11, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_11, qp_5_11, dot[3]);
        }
        unsigned int sw_21[4];
        sw_21[0] = words[56 + woff_16];
        sw_21[1] = words[56 + woff_16 + 1];
        sw_21[2] = words[56 + woff_16 + 2];
        sw_21[3] = words[56 + woff_16 + 3];
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
            float2 _f2_190 = make_float2(sw_21_f32[0], sw_21_f32[1]);
            float2 v_14 = _f2_190;
            float2 _f2_191 = make_float2(q[8], q[9]);
            float2 qp_15 = _f2_191;
            sq[4] = fma_f32x2_rn_noftz(v_14, v_14, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_14, qp_15, dot[4]);
            float2 _f2_192 = make_float2(sw_21_f32[2], sw_21_f32[3]);
            float2 v_0_12 = _f2_192;
            float2 _f2_193 = make_float2(q[10], q[11]);
            float2 qp_1_12 = _f2_193;
            sq[4] = fma_f32x2_rn_noftz(v_0_12, v_0_12, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_12, qp_1_12, dot[4]);
            float2 _f2_194 = make_float2(sw_21_f32[4], sw_21_f32[5]);
            float2 v_2_12 = _f2_194;
            float2 _f2_195 = make_float2(q[12], q[13]);
            float2 qp_3_12 = _f2_195;
            sq[4] = fma_f32x2_rn_noftz(v_2_12, v_2_12, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_12, qp_3_12, dot[4]);
            float2 _f2_196 = make_float2(sw_21_f32[6], sw_21_f32[7]);
            float2 v_4_12 = _f2_196;
            float2 _f2_197 = make_float2(q[14], q[15]);
            float2 qp_5_12 = _f2_197;
            sq[4] = fma_f32x2_rn_noftz(v_4_12, v_4_12, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_12, qp_5_12, dot[4]);
        }
        unsigned int sw_22[4];
        sw_22[0] = words[70 + woff_16];
        sw_22[1] = words[70 + woff_16 + 1];
        sw_22[2] = words[70 + woff_16 + 2];
        sw_22[3] = words[70 + woff_16 + 3];
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
            float2 _f2_204 = make_float2(sw_22_f32[0], sw_22_f32[1]);
            float2 v_15 = _f2_204;
            float2 _f2_205 = make_float2(q[8], q[9]);
            float2 qp_16 = _f2_205;
            sq[5] = fma_f32x2_rn_noftz(v_15, v_15, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_15, qp_16, dot[5]);
            float2 _f2_206 = make_float2(sw_22_f32[2], sw_22_f32[3]);
            float2 v_0_13 = _f2_206;
            float2 _f2_207 = make_float2(q[10], q[11]);
            float2 qp_1_13 = _f2_207;
            sq[5] = fma_f32x2_rn_noftz(v_0_13, v_0_13, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_13, qp_1_13, dot[5]);
            float2 _f2_208 = make_float2(sw_22_f32[4], sw_22_f32[5]);
            float2 v_2_13 = _f2_208;
            float2 _f2_209 = make_float2(q[12], q[13]);
            float2 qp_3_13 = _f2_209;
            sq[5] = fma_f32x2_rn_noftz(v_2_13, v_2_13, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_13, qp_3_13, dot[5]);
            float2 _f2_210 = make_float2(sw_22_f32[6], sw_22_f32[7]);
            float2 v_4_13 = _f2_210;
            float2 _f2_211 = make_float2(q[14], q[15]);
            float2 qp_5_13 = _f2_211;
            sq[5] = fma_f32x2_rn_noftz(v_4_13, v_4_13, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_13, qp_5_13, dot[5]);
        }
        unsigned int sw_23[4];
        sw_23[0] = words[84 + woff_16];
        sw_23[1] = words[84 + woff_16 + 1];
        sw_23[2] = words[84 + woff_16 + 2];
        sw_23[3] = words[84 + woff_16 + 3];
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
            float2 _f2_218 = make_float2(sw_23_f32[0], sw_23_f32[1]);
            float2 v_16 = _f2_218;
            float2 _f2_219 = make_float2(q[8], q[9]);
            float2 qp_17 = _f2_219;
            sq[6] = fma_f32x2_rn_noftz(v_16, v_16, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_16, qp_17, dot[6]);
            float2 _f2_220 = make_float2(sw_23_f32[2], sw_23_f32[3]);
            float2 v_0_14 = _f2_220;
            float2 _f2_221 = make_float2(q[10], q[11]);
            float2 qp_1_14 = _f2_221;
            sq[6] = fma_f32x2_rn_noftz(v_0_14, v_0_14, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_14, qp_1_14, dot[6]);
            float2 _f2_222 = make_float2(sw_23_f32[4], sw_23_f32[5]);
            float2 v_2_14 = _f2_222;
            float2 _f2_223 = make_float2(q[12], q[13]);
            float2 qp_3_14 = _f2_223;
            sq[6] = fma_f32x2_rn_noftz(v_2_14, v_2_14, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_14, qp_3_14, dot[6]);
            float2 _f2_224 = make_float2(sw_23_f32[6], sw_23_f32[7]);
            float2 v_4_14 = _f2_224;
            float2 _f2_225 = make_float2(q[14], q[15]);
            float2 qp_5_14 = _f2_225;
            sq[6] = fma_f32x2_rn_noftz(v_4_14, v_4_14, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_14, qp_5_14, dot[6]);
        }
        unsigned int sw_24[4];
        sw_24[0] = words[98 + woff_16];
        sw_24[1] = words[98 + woff_16 + 1];
        sw_24[2] = words[98 + woff_16 + 2];
        sw_24[3] = words[98 + woff_16 + 3];
        {
            __nv_bfloat162 a_1 = __as_bf16x2(sw_24[0]);
            __nv_bfloat162 d_2 = __as_bf16x2(dwords[woff_16]);
            __nv_bfloat162 mixed_1 = a_1 + d_2;
            sw_24[0] = __as_u32(mixed_1);
            words[98 + woff_16] = sw_24[0];
            __nv_bfloat162 a_0_1 = __as_bf16x2(sw_24[1]);
            __nv_bfloat162 d_1_1 = __as_bf16x2(dwords[woff_16 + 1]);
            __nv_bfloat162 mixed_2_1 = a_0_1 + d_1_1;
            sw_24[1] = __as_u32(mixed_2_1);
            words[98 + woff_16 + 1] = sw_24[1];
            __nv_bfloat162 a_3_1 = __as_bf16x2(sw_24[2]);
            __nv_bfloat162 d_4_1 = __as_bf16x2(dwords[woff_16 + 2]);
            __nv_bfloat162 mixed_5_1 = a_3_1 + d_4_1;
            sw_24[2] = __as_u32(mixed_5_1);
            words[98 + woff_16 + 2] = sw_24[2];
            __nv_bfloat162 a_6_1 = __as_bf16x2(sw_24[3]);
            __nv_bfloat162 d_7_1 = __as_bf16x2(dwords[woff_16 + 3]);
            __nv_bfloat162 mixed_8_1 = a_6_1 + d_7_1;
            sw_24[3] = __as_u32(mixed_8_1);
            words[98 + woff_16 + 3] = sw_24[3];
            {
                int4 _iv4 = make_int4(sw_24[0 + 0], sw_24[0 + 1], sw_24[0 + 2], sw_24[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_15)) + 0) = _iv4;
            }
        }
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
            float2 _f2_232 = make_float2(sw_24_f32[0], sw_24_f32[1]);
            float2 v_17 = _f2_232;
            float2 _f2_233 = make_float2(q[8], q[9]);
            float2 qp_18 = _f2_233;
            sq[7] = fma_f32x2_rn_noftz(v_17, v_17, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_17, qp_18, dot[7]);
            float2 _f2_234 = make_float2(sw_24_f32[2], sw_24_f32[3]);
            float2 v_0_15 = _f2_234;
            float2 _f2_235 = make_float2(q[10], q[11]);
            float2 qp_1_15 = _f2_235;
            sq[7] = fma_f32x2_rn_noftz(v_0_15, v_0_15, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_0_15, qp_1_15, dot[7]);
            float2 _f2_236 = make_float2(sw_24_f32[4], sw_24_f32[5]);
            float2 v_2_15 = _f2_236;
            float2 _f2_237 = make_float2(q[12], q[13]);
            float2 qp_3_15 = _f2_237;
            sq[7] = fma_f32x2_rn_noftz(v_2_15, v_2_15, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_2_15, qp_3_15, dot[7]);
            float2 _f2_238 = make_float2(sw_24_f32[6], sw_24_f32[7]);
            float2 v_4_15 = _f2_238;
            float2 _f2_239 = make_float2(q[14], q[15]);
            float2 qp_5_15 = _f2_239;
            sq[7] = fma_f32x2_rn_noftz(v_4_15, v_4_15, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_4_15, qp_5_15, dot[7]);
        }
        int base_25 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        const int woff_26 = 8;
        unsigned int sw_27[4];
        sw_27[0] = words[woff_26];
        sw_27[1] = words[woff_26 + 1];
        sw_27[2] = words[woff_26 + 2];
        sw_27[3] = words[woff_26 + 3];
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
            fsrc[16] = sw_27_f32[0];
            fsrc[17] = sw_27_f32[1];
            fsrc[18] = sw_27_f32[2];
            fsrc[19] = sw_27_f32[3];
            fsrc[20] = sw_27_f32[4];
            fsrc[21] = sw_27_f32[5];
            fsrc[22] = sw_27_f32[6];
            fsrc[23] = sw_27_f32[7];
        }
        {
            float2 _f2_246 = make_float2(sw_27_f32[0], sw_27_f32[1]);
            float2 v_18 = _f2_246;
            float2 _f2_247 = make_float2(q[16], q[17]);
            float2 qp_19 = _f2_247;
            sq[0] = fma_f32x2_rn_noftz(v_18, v_18, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_18, qp_19, dot[0]);
            float2 _f2_248 = make_float2(sw_27_f32[2], sw_27_f32[3]);
            float2 v_0_16 = _f2_248;
            float2 _f2_249 = make_float2(q[18], q[19]);
            float2 qp_1_16 = _f2_249;
            sq[0] = fma_f32x2_rn_noftz(v_0_16, v_0_16, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_0_16, qp_1_16, dot[0]);
            float2 _f2_250 = make_float2(sw_27_f32[4], sw_27_f32[5]);
            float2 v_2_16 = _f2_250;
            float2 _f2_251 = make_float2(q[20], q[21]);
            float2 qp_3_16 = _f2_251;
            sq[0] = fma_f32x2_rn_noftz(v_2_16, v_2_16, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_2_16, qp_3_16, dot[0]);
            float2 _f2_252 = make_float2(sw_27_f32[6], sw_27_f32[7]);
            float2 v_4_16 = _f2_252;
            float2 _f2_253 = make_float2(q[22], q[23]);
            float2 qp_5_16 = _f2_253;
            sq[0] = fma_f32x2_rn_noftz(v_4_16, v_4_16, sq[0]);
            dot[0] = fma_f32x2_rn_noftz(v_4_16, qp_5_16, dot[0]);
        }
        unsigned int sw_28[4];
        sw_28[0] = words[14 + woff_26];
        sw_28[1] = words[14 + woff_26 + 1];
        sw_28[2] = words[14 + woff_26 + 2];
        sw_28[3] = words[14 + woff_26 + 3];
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
            fsrc[44] = sw_28_f32[0];
            fsrc[45] = sw_28_f32[1];
            fsrc[46] = sw_28_f32[2];
            fsrc[47] = sw_28_f32[3];
            fsrc[48] = sw_28_f32[4];
            fsrc[49] = sw_28_f32[5];
            fsrc[50] = sw_28_f32[6];
            fsrc[51] = sw_28_f32[7];
        }
        {
            float2 _f2_260 = make_float2(sw_28_f32[0], sw_28_f32[1]);
            float2 v_19 = _f2_260;
            float2 _f2_261 = make_float2(q[16], q[17]);
            float2 qp_20 = _f2_261;
            sq[1] = fma_f32x2_rn_noftz(v_19, v_19, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_19, qp_20, dot[1]);
            float2 _f2_262 = make_float2(sw_28_f32[2], sw_28_f32[3]);
            float2 v_0_17 = _f2_262;
            float2 _f2_263 = make_float2(q[18], q[19]);
            float2 qp_1_17 = _f2_263;
            sq[1] = fma_f32x2_rn_noftz(v_0_17, v_0_17, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_0_17, qp_1_17, dot[1]);
            float2 _f2_264 = make_float2(sw_28_f32[4], sw_28_f32[5]);
            float2 v_2_17 = _f2_264;
            float2 _f2_265 = make_float2(q[20], q[21]);
            float2 qp_3_17 = _f2_265;
            sq[1] = fma_f32x2_rn_noftz(v_2_17, v_2_17, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_2_17, qp_3_17, dot[1]);
            float2 _f2_266 = make_float2(sw_28_f32[6], sw_28_f32[7]);
            float2 v_4_17 = _f2_266;
            float2 _f2_267 = make_float2(q[22], q[23]);
            float2 qp_5_17 = _f2_267;
            sq[1] = fma_f32x2_rn_noftz(v_4_17, v_4_17, sq[1]);
            dot[1] = fma_f32x2_rn_noftz(v_4_17, qp_5_17, dot[1]);
        }
        unsigned int sw_29[4];
        sw_29[0] = words[28 + woff_26];
        sw_29[1] = words[28 + woff_26 + 1];
        sw_29[2] = words[28 + woff_26 + 2];
        sw_29[3] = words[28 + woff_26 + 3];
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
            fsrc[72] = sw_29_f32[0];
            fsrc[73] = sw_29_f32[1];
            fsrc[74] = sw_29_f32[2];
            fsrc[75] = sw_29_f32[3];
            fsrc[76] = sw_29_f32[4];
            fsrc[77] = sw_29_f32[5];
            fsrc[78] = sw_29_f32[6];
            fsrc[79] = sw_29_f32[7];
        }
        {
            float2 _f2_274 = make_float2(sw_29_f32[0], sw_29_f32[1]);
            float2 v_20 = _f2_274;
            float2 _f2_275 = make_float2(q[16], q[17]);
            float2 qp_21 = _f2_275;
            sq[2] = fma_f32x2_rn_noftz(v_20, v_20, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_20, qp_21, dot[2]);
            float2 _f2_276 = make_float2(sw_29_f32[2], sw_29_f32[3]);
            float2 v_0_18 = _f2_276;
            float2 _f2_277 = make_float2(q[18], q[19]);
            float2 qp_1_18 = _f2_277;
            sq[2] = fma_f32x2_rn_noftz(v_0_18, v_0_18, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_0_18, qp_1_18, dot[2]);
            float2 _f2_278 = make_float2(sw_29_f32[4], sw_29_f32[5]);
            float2 v_2_18 = _f2_278;
            float2 _f2_279 = make_float2(q[20], q[21]);
            float2 qp_3_18 = _f2_279;
            sq[2] = fma_f32x2_rn_noftz(v_2_18, v_2_18, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_2_18, qp_3_18, dot[2]);
            float2 _f2_280 = make_float2(sw_29_f32[6], sw_29_f32[7]);
            float2 v_4_18 = _f2_280;
            float2 _f2_281 = make_float2(q[22], q[23]);
            float2 qp_5_18 = _f2_281;
            sq[2] = fma_f32x2_rn_noftz(v_4_18, v_4_18, sq[2]);
            dot[2] = fma_f32x2_rn_noftz(v_4_18, qp_5_18, dot[2]);
        }
        unsigned int sw_30[4];
        sw_30[0] = words[42 + woff_26];
        sw_30[1] = words[42 + woff_26 + 1];
        sw_30[2] = words[42 + woff_26 + 2];
        sw_30[3] = words[42 + woff_26 + 3];
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
            float2 _f2_288 = make_float2(sw_30_f32[0], sw_30_f32[1]);
            float2 v_21 = _f2_288;
            float2 _f2_289 = make_float2(q[16], q[17]);
            float2 qp_22 = _f2_289;
            sq[3] = fma_f32x2_rn_noftz(v_21, v_21, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_21, qp_22, dot[3]);
            float2 _f2_290 = make_float2(sw_30_f32[2], sw_30_f32[3]);
            float2 v_0_19 = _f2_290;
            float2 _f2_291 = make_float2(q[18], q[19]);
            float2 qp_1_19 = _f2_291;
            sq[3] = fma_f32x2_rn_noftz(v_0_19, v_0_19, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_0_19, qp_1_19, dot[3]);
            float2 _f2_292 = make_float2(sw_30_f32[4], sw_30_f32[5]);
            float2 v_2_19 = _f2_292;
            float2 _f2_293 = make_float2(q[20], q[21]);
            float2 qp_3_19 = _f2_293;
            sq[3] = fma_f32x2_rn_noftz(v_2_19, v_2_19, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_2_19, qp_3_19, dot[3]);
            float2 _f2_294 = make_float2(sw_30_f32[6], sw_30_f32[7]);
            float2 v_4_19 = _f2_294;
            float2 _f2_295 = make_float2(q[22], q[23]);
            float2 qp_5_19 = _f2_295;
            sq[3] = fma_f32x2_rn_noftz(v_4_19, v_4_19, sq[3]);
            dot[3] = fma_f32x2_rn_noftz(v_4_19, qp_5_19, dot[3]);
        }
        unsigned int sw_31[4];
        sw_31[0] = words[56 + woff_26];
        sw_31[1] = words[56 + woff_26 + 1];
        sw_31[2] = words[56 + woff_26 + 2];
        sw_31[3] = words[56 + woff_26 + 3];
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
            float2 _f2_302 = make_float2(sw_31_f32[0], sw_31_f32[1]);
            float2 v_22 = _f2_302;
            float2 _f2_303 = make_float2(q[16], q[17]);
            float2 qp_23 = _f2_303;
            sq[4] = fma_f32x2_rn_noftz(v_22, v_22, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_22, qp_23, dot[4]);
            float2 _f2_304 = make_float2(sw_31_f32[2], sw_31_f32[3]);
            float2 v_0_20 = _f2_304;
            float2 _f2_305 = make_float2(q[18], q[19]);
            float2 qp_1_20 = _f2_305;
            sq[4] = fma_f32x2_rn_noftz(v_0_20, v_0_20, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_0_20, qp_1_20, dot[4]);
            float2 _f2_306 = make_float2(sw_31_f32[4], sw_31_f32[5]);
            float2 v_2_20 = _f2_306;
            float2 _f2_307 = make_float2(q[20], q[21]);
            float2 qp_3_20 = _f2_307;
            sq[4] = fma_f32x2_rn_noftz(v_2_20, v_2_20, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_2_20, qp_3_20, dot[4]);
            float2 _f2_308 = make_float2(sw_31_f32[6], sw_31_f32[7]);
            float2 v_4_20 = _f2_308;
            float2 _f2_309 = make_float2(q[22], q[23]);
            float2 qp_5_20 = _f2_309;
            sq[4] = fma_f32x2_rn_noftz(v_4_20, v_4_20, sq[4]);
            dot[4] = fma_f32x2_rn_noftz(v_4_20, qp_5_20, dot[4]);
        }
        unsigned int sw_32[4];
        sw_32[0] = words[70 + woff_26];
        sw_32[1] = words[70 + woff_26 + 1];
        sw_32[2] = words[70 + woff_26 + 2];
        sw_32[3] = words[70 + woff_26 + 3];
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
            float2 _f2_316 = make_float2(sw_32_f32[0], sw_32_f32[1]);
            float2 v_23 = _f2_316;
            float2 _f2_317 = make_float2(q[16], q[17]);
            float2 qp_24 = _f2_317;
            sq[5] = fma_f32x2_rn_noftz(v_23, v_23, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_23, qp_24, dot[5]);
            float2 _f2_318 = make_float2(sw_32_f32[2], sw_32_f32[3]);
            float2 v_0_21 = _f2_318;
            float2 _f2_319 = make_float2(q[18], q[19]);
            float2 qp_1_21 = _f2_319;
            sq[5] = fma_f32x2_rn_noftz(v_0_21, v_0_21, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_0_21, qp_1_21, dot[5]);
            float2 _f2_320 = make_float2(sw_32_f32[4], sw_32_f32[5]);
            float2 v_2_21 = _f2_320;
            float2 _f2_321 = make_float2(q[20], q[21]);
            float2 qp_3_21 = _f2_321;
            sq[5] = fma_f32x2_rn_noftz(v_2_21, v_2_21, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_2_21, qp_3_21, dot[5]);
            float2 _f2_322 = make_float2(sw_32_f32[6], sw_32_f32[7]);
            float2 v_4_21 = _f2_322;
            float2 _f2_323 = make_float2(q[22], q[23]);
            float2 qp_5_21 = _f2_323;
            sq[5] = fma_f32x2_rn_noftz(v_4_21, v_4_21, sq[5]);
            dot[5] = fma_f32x2_rn_noftz(v_4_21, qp_5_21, dot[5]);
        }
        unsigned int sw_33[4];
        sw_33[0] = words[84 + woff_26];
        sw_33[1] = words[84 + woff_26 + 1];
        sw_33[2] = words[84 + woff_26 + 2];
        sw_33[3] = words[84 + woff_26 + 3];
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
            float2 _f2_330 = make_float2(sw_33_f32[0], sw_33_f32[1]);
            float2 v_24 = _f2_330;
            float2 _f2_331 = make_float2(q[16], q[17]);
            float2 qp_25 = _f2_331;
            sq[6] = fma_f32x2_rn_noftz(v_24, v_24, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_24, qp_25, dot[6]);
            float2 _f2_332 = make_float2(sw_33_f32[2], sw_33_f32[3]);
            float2 v_0_22 = _f2_332;
            float2 _f2_333 = make_float2(q[18], q[19]);
            float2 qp_1_22 = _f2_333;
            sq[6] = fma_f32x2_rn_noftz(v_0_22, v_0_22, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_0_22, qp_1_22, dot[6]);
            float2 _f2_334 = make_float2(sw_33_f32[4], sw_33_f32[5]);
            float2 v_2_22 = _f2_334;
            float2 _f2_335 = make_float2(q[20], q[21]);
            float2 qp_3_22 = _f2_335;
            sq[6] = fma_f32x2_rn_noftz(v_2_22, v_2_22, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_2_22, qp_3_22, dot[6]);
            float2 _f2_336 = make_float2(sw_33_f32[6], sw_33_f32[7]);
            float2 v_4_22 = _f2_336;
            float2 _f2_337 = make_float2(q[22], q[23]);
            float2 qp_5_22 = _f2_337;
            sq[6] = fma_f32x2_rn_noftz(v_4_22, v_4_22, sq[6]);
            dot[6] = fma_f32x2_rn_noftz(v_4_22, qp_5_22, dot[6]);
        }
        unsigned int sw_34[4];
        sw_34[0] = words[98 + woff_26];
        sw_34[1] = words[98 + woff_26 + 1];
        sw_34[2] = words[98 + woff_26 + 2];
        sw_34[3] = words[98 + woff_26 + 3];
        {
            __nv_bfloat162 a_2 = __as_bf16x2(sw_34[0]);
            __nv_bfloat162 d_3 = __as_bf16x2(dwords[woff_26]);
            __nv_bfloat162 mixed_3 = a_2 + d_3;
            sw_34[0] = __as_u32(mixed_3);
            words[98 + woff_26] = sw_34[0];
            __nv_bfloat162 a_0_2 = __as_bf16x2(sw_34[1]);
            __nv_bfloat162 d_1_2 = __as_bf16x2(dwords[woff_26 + 1]);
            __nv_bfloat162 mixed_2_2 = a_0_2 + d_1_2;
            sw_34[1] = __as_u32(mixed_2_2);
            words[98 + woff_26 + 1] = sw_34[1];
            __nv_bfloat162 a_3_2 = __as_bf16x2(sw_34[2]);
            __nv_bfloat162 d_4_2 = __as_bf16x2(dwords[woff_26 + 2]);
            __nv_bfloat162 mixed_5_2 = a_3_2 + d_4_2;
            sw_34[2] = __as_u32(mixed_5_2);
            words[98 + woff_26 + 2] = sw_34[2];
            __nv_bfloat162 a_6_2 = __as_bf16x2(sw_34[3]);
            __nv_bfloat162 d_7_2 = __as_bf16x2(dwords[woff_26 + 3]);
            __nv_bfloat162 mixed_8_2 = a_6_2 + d_7_2;
            sw_34[3] = __as_u32(mixed_8_2);
            words[98 + woff_26 + 3] = sw_34[3];
            {
                int4 _iv4 = make_int4(sw_34[0 + 0], sw_34[0 + 1], sw_34[0 + 2], sw_34[0 + 3]);
                *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_25)) + 0) = _iv4;
            }
        }
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
            float2 _f2_344 = make_float2(sw_34_f32[0], sw_34_f32[1]);
            float2 v_25 = _f2_344;
            float2 _f2_345 = make_float2(q[16], q[17]);
            float2 qp_26 = _f2_345;
            sq[7] = fma_f32x2_rn_noftz(v_25, v_25, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_25, qp_26, dot[7]);
            float2 _f2_346 = make_float2(sw_34_f32[2], sw_34_f32[3]);
            float2 v_0_23 = _f2_346;
            float2 _f2_347 = make_float2(q[18], q[19]);
            float2 qp_1_23 = _f2_347;
            sq[7] = fma_f32x2_rn_noftz(v_0_23, v_0_23, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_0_23, qp_1_23, dot[7]);
            float2 _f2_348 = make_float2(sw_34_f32[4], sw_34_f32[5]);
            float2 v_2_23 = _f2_348;
            float2 _f2_349 = make_float2(q[20], q[21]);
            float2 qp_3_23 = _f2_349;
            sq[7] = fma_f32x2_rn_noftz(v_2_23, v_2_23, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_2_23, qp_3_23, dot[7]);
            float2 _f2_350 = make_float2(sw_34_f32[6], sw_34_f32[7]);
            float2 v_4_23 = _f2_350;
            float2 _f2_351 = make_float2(q[22], q[23]);
            float2 qp_5_23 = _f2_351;
            sq[7] = fma_f32x2_rn_noftz(v_4_23, v_4_23, sq[7]);
            dot[7] = fma_f32x2_rn_noftz(v_4_23, qp_5_23, dot[7]);
        }
        int base_35 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        const int woff_36 = 12;
        unsigned int sw_37[4];
        sw_37[0] = words[woff_36];
        sw_37[1] = words[woff_36 + 1];
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
            fsrc[24] = sw_37_f32[0];
            fsrc[25] = sw_37_f32[1];
            fsrc[26] = sw_37_f32[2];
            fsrc[27] = sw_37_f32[3];
        }
        {
            float2 _f2_352 = make_float2(sw_37_f32[0], sw_37_f32[1]);
            float2 v_26 = _f2_352;
            sq[0] = fma_f32x2_rn_noftz(v_26, v_26, sq[0]);
            float2 _f2_353 = make_float2(sw_37_f32[2], sw_37_f32[3]);
            float2 v_0_24 = _f2_353;
            sq[0] = fma_f32x2_rn_noftz(v_0_24, v_0_24, sq[0]);
            float2 _f2_354 = make_float2(sw_37_f32[0], sw_37_f32[1]);
            float2 v_1_1 = _f2_354;
            float2 _f2_355 = make_float2(q[24], q[25]);
            float2 qp_27 = _f2_355;
            dot[0] = fma_f32x2_rn_noftz(v_1_1, qp_27, dot[0]);
            float2 _f2_356 = make_float2(sw_37_f32[2], sw_37_f32[3]);
            float2 v_2_24 = _f2_356;
            float2 _f2_357 = make_float2(q[26], q[27]);
            float2 qp_3_24 = _f2_357;
            dot[0] = fma_f32x2_rn_noftz(v_2_24, qp_3_24, dot[0]);
        }
        unsigned int sw_38[4];
        sw_38[0] = words[14 + woff_36];
        sw_38[1] = words[14 + woff_36 + 1];
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
            fsrc[52] = sw_38_f32[0];
            fsrc[53] = sw_38_f32[1];
            fsrc[54] = sw_38_f32[2];
            fsrc[55] = sw_38_f32[3];
        }
        {
            float2 _f2_366 = make_float2(sw_38_f32[0], sw_38_f32[1]);
            float2 v_27 = _f2_366;
            sq[1] = fma_f32x2_rn_noftz(v_27, v_27, sq[1]);
            float2 _f2_367 = make_float2(sw_38_f32[2], sw_38_f32[3]);
            float2 v_0_25 = _f2_367;
            sq[1] = fma_f32x2_rn_noftz(v_0_25, v_0_25, sq[1]);
            float2 _f2_368 = make_float2(sw_38_f32[0], sw_38_f32[1]);
            float2 v_1_2 = _f2_368;
            float2 _f2_369 = make_float2(q[24], q[25]);
            float2 qp_28 = _f2_369;
            dot[1] = fma_f32x2_rn_noftz(v_1_2, qp_28, dot[1]);
            float2 _f2_370 = make_float2(sw_38_f32[2], sw_38_f32[3]);
            float2 v_2_25 = _f2_370;
            float2 _f2_371 = make_float2(q[26], q[27]);
            float2 qp_3_25 = _f2_371;
            dot[1] = fma_f32x2_rn_noftz(v_2_25, qp_3_25, dot[1]);
        }
        unsigned int sw_39[4];
        sw_39[0] = words[28 + woff_36];
        sw_39[1] = words[28 + woff_36 + 1];
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
            fsrc[80] = sw_39_f32[0];
            fsrc[81] = sw_39_f32[1];
            fsrc[82] = sw_39_f32[2];
            fsrc[83] = sw_39_f32[3];
        }
        {
            float2 _f2_380 = make_float2(sw_39_f32[0], sw_39_f32[1]);
            float2 v_28 = _f2_380;
            sq[2] = fma_f32x2_rn_noftz(v_28, v_28, sq[2]);
            float2 _f2_381 = make_float2(sw_39_f32[2], sw_39_f32[3]);
            float2 v_0_26 = _f2_381;
            sq[2] = fma_f32x2_rn_noftz(v_0_26, v_0_26, sq[2]);
            float2 _f2_382 = make_float2(sw_39_f32[0], sw_39_f32[1]);
            float2 v_1_3 = _f2_382;
            float2 _f2_383 = make_float2(q[24], q[25]);
            float2 qp_29 = _f2_383;
            dot[2] = fma_f32x2_rn_noftz(v_1_3, qp_29, dot[2]);
            float2 _f2_384 = make_float2(sw_39_f32[2], sw_39_f32[3]);
            float2 v_2_26 = _f2_384;
            float2 _f2_385 = make_float2(q[26], q[27]);
            float2 qp_3_26 = _f2_385;
            dot[2] = fma_f32x2_rn_noftz(v_2_26, qp_3_26, dot[2]);
        }
        unsigned int sw_40[4];
        sw_40[0] = words[42 + woff_36];
        sw_40[1] = words[42 + woff_36 + 1];
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
            float2 _f2_394 = make_float2(sw_40_f32[0], sw_40_f32[1]);
            float2 v_29 = _f2_394;
            sq[3] = fma_f32x2_rn_noftz(v_29, v_29, sq[3]);
            float2 _f2_395 = make_float2(sw_40_f32[2], sw_40_f32[3]);
            float2 v_0_27 = _f2_395;
            sq[3] = fma_f32x2_rn_noftz(v_0_27, v_0_27, sq[3]);
            float2 _f2_396 = make_float2(sw_40_f32[0], sw_40_f32[1]);
            float2 v_1_4 = _f2_396;
            float2 _f2_397 = make_float2(q[24], q[25]);
            float2 qp_30 = _f2_397;
            dot[3] = fma_f32x2_rn_noftz(v_1_4, qp_30, dot[3]);
            float2 _f2_398 = make_float2(sw_40_f32[2], sw_40_f32[3]);
            float2 v_2_27 = _f2_398;
            float2 _f2_399 = make_float2(q[26], q[27]);
            float2 qp_3_27 = _f2_399;
            dot[3] = fma_f32x2_rn_noftz(v_2_27, qp_3_27, dot[3]);
        }
        unsigned int sw_41[4];
        sw_41[0] = words[56 + woff_36];
        sw_41[1] = words[56 + woff_36 + 1];
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
            float2 _f2_408 = make_float2(sw_41_f32[0], sw_41_f32[1]);
            float2 v_30 = _f2_408;
            sq[4] = fma_f32x2_rn_noftz(v_30, v_30, sq[4]);
            float2 _f2_409 = make_float2(sw_41_f32[2], sw_41_f32[3]);
            float2 v_0_28 = _f2_409;
            sq[4] = fma_f32x2_rn_noftz(v_0_28, v_0_28, sq[4]);
            float2 _f2_410 = make_float2(sw_41_f32[0], sw_41_f32[1]);
            float2 v_1_5 = _f2_410;
            float2 _f2_411 = make_float2(q[24], q[25]);
            float2 qp_31 = _f2_411;
            dot[4] = fma_f32x2_rn_noftz(v_1_5, qp_31, dot[4]);
            float2 _f2_412 = make_float2(sw_41_f32[2], sw_41_f32[3]);
            float2 v_2_28 = _f2_412;
            float2 _f2_413 = make_float2(q[26], q[27]);
            float2 qp_3_28 = _f2_413;
            dot[4] = fma_f32x2_rn_noftz(v_2_28, qp_3_28, dot[4]);
        }
        unsigned int sw_42[4];
        sw_42[0] = words[70 + woff_36];
        sw_42[1] = words[70 + woff_36 + 1];
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
            float2 _f2_422 = make_float2(sw_42_f32[0], sw_42_f32[1]);
            float2 v_31 = _f2_422;
            sq[5] = fma_f32x2_rn_noftz(v_31, v_31, sq[5]);
            float2 _f2_423 = make_float2(sw_42_f32[2], sw_42_f32[3]);
            float2 v_0_29 = _f2_423;
            sq[5] = fma_f32x2_rn_noftz(v_0_29, v_0_29, sq[5]);
            float2 _f2_424 = make_float2(sw_42_f32[0], sw_42_f32[1]);
            float2 v_1_6 = _f2_424;
            float2 _f2_425 = make_float2(q[24], q[25]);
            float2 qp_32 = _f2_425;
            dot[5] = fma_f32x2_rn_noftz(v_1_6, qp_32, dot[5]);
            float2 _f2_426 = make_float2(sw_42_f32[2], sw_42_f32[3]);
            float2 v_2_29 = _f2_426;
            float2 _f2_427 = make_float2(q[26], q[27]);
            float2 qp_3_29 = _f2_427;
            dot[5] = fma_f32x2_rn_noftz(v_2_29, qp_3_29, dot[5]);
        }
        unsigned int sw_43[4];
        sw_43[0] = words[84 + woff_36];
        sw_43[1] = words[84 + woff_36 + 1];
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
            float2 _f2_436 = make_float2(sw_43_f32[0], sw_43_f32[1]);
            float2 v_32 = _f2_436;
            sq[6] = fma_f32x2_rn_noftz(v_32, v_32, sq[6]);
            float2 _f2_437 = make_float2(sw_43_f32[2], sw_43_f32[3]);
            float2 v_0_30 = _f2_437;
            sq[6] = fma_f32x2_rn_noftz(v_0_30, v_0_30, sq[6]);
            float2 _f2_438 = make_float2(sw_43_f32[0], sw_43_f32[1]);
            float2 v_1_7 = _f2_438;
            float2 _f2_439 = make_float2(q[24], q[25]);
            float2 qp_33 = _f2_439;
            dot[6] = fma_f32x2_rn_noftz(v_1_7, qp_33, dot[6]);
            float2 _f2_440 = make_float2(sw_43_f32[2], sw_43_f32[3]);
            float2 v_2_30 = _f2_440;
            float2 _f2_441 = make_float2(q[26], q[27]);
            float2 qp_3_30 = _f2_441;
            dot[6] = fma_f32x2_rn_noftz(v_2_30, qp_3_30, dot[6]);
        }
        unsigned int sw_44[4];
        sw_44[0] = words[98 + woff_36];
        sw_44[1] = words[98 + woff_36 + 1];
        {
            __nv_bfloat162 a_4 = __as_bf16x2(sw_44[0]);
            __nv_bfloat162 d_5 = __as_bf16x2(dwords[woff_36]);
            __nv_bfloat162 mixed_4 = a_4 + d_5;
            sw_44[0] = __as_u32(mixed_4);
            words[98 + woff_36] = sw_44[0];
            __nv_bfloat162 a_0_3 = __as_bf16x2(sw_44[1]);
            __nv_bfloat162 d_1_3 = __as_bf16x2(dwords[woff_36 + 1]);
            __nv_bfloat162 mixed_2_3 = a_0_3 + d_1_3;
            sw_44[1] = __as_u32(mixed_2_3);
            words[98 + woff_36 + 1] = sw_44[1];
            {
                int2 _iv2 = make_int2(sw_44[0 + 0], sw_44[0 + 1]);
                *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_35)) + 0) = _iv2;
            }
        }
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
            float2 _f2_450 = make_float2(sw_44_f32[0], sw_44_f32[1]);
            float2 v_33 = _f2_450;
            sq[7] = fma_f32x2_rn_noftz(v_33, v_33, sq[7]);
            float2 _f2_451 = make_float2(sw_44_f32[2], sw_44_f32[3]);
            float2 v_0_31 = _f2_451;
            sq[7] = fma_f32x2_rn_noftz(v_0_31, v_0_31, sq[7]);
            float2 _f2_452 = make_float2(sw_44_f32[0], sw_44_f32[1]);
            float2 v_1_8 = _f2_452;
            float2 _f2_453 = make_float2(q[24], q[25]);
            float2 qp_34 = _f2_453;
            dot[7] = fma_f32x2_rn_noftz(v_1_8, qp_34, dot[7]);
            float2 _f2_454 = make_float2(sw_44_f32[2], sw_44_f32[3]);
            float2 v_2_31 = _f2_454;
            float2 _f2_455 = make_float2(q[26], q[27]);
            float2 qp_3_31 = _f2_455;
            dot[7] = fma_f32x2_rn_noftz(v_2_31, qp_3_31, dot[7]);
        }
        float2 pairs[8];
        float2 _f2_464 = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[0] = _f2_464;
        float2 _f2_465 = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[1] = _f2_465;
        float2 _f2_466 = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[2] = _f2_466;
        float2 _f2_467 = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[3] = _f2_467;
        float2 _f2_468 = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[4] = _f2_468;
        float2 _f2_469 = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[5] = _f2_469;
        float2 _f2_470 = make_float2(sq[6].x + sq[6].y, dot[6].x + dot[6].y);
        pairs[6] = _f2_470;
        float2 _f2_471 = make_float2(sq[7].x + sq[7].y, dot[7].x + dot[7].y);
        pairs[7] = _f2_471;
        unsigned long long bits = 0;
        bits = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        unsigned long long peerbits = _shfl_xor_0;
        float2 _f2_472 = make_float2(0.0f, 0.0f);
        float2 peer = _f2_472;
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_45 = 0;
        bits_45 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bits_45, 16);
        unsigned long long peerbits_46 = _shfl_xor_1;
        float2 _f2_473 = make_float2(0.0f, 0.0f);
        float2 peer_47 = _f2_473;
        peer_47 = reinterpret_cast<float2*>(&peerbits_46)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_47);
        unsigned long long bits_48 = 0;
        bits_48 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bits_48, 16);
        unsigned long long peerbits_49 = _shfl_xor_2;
        float2 _f2_474 = make_float2(0.0f, 0.0f);
        float2 peer_50 = _f2_474;
        peer_50 = reinterpret_cast<float2*>(&peerbits_49)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_50);
        unsigned long long bits_51 = 0;
        bits_51 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bits_51, 16);
        unsigned long long peerbits_52 = _shfl_xor_3;
        float2 _f2_475 = make_float2(0.0f, 0.0f);
        float2 peer_53 = _f2_475;
        peer_53 = reinterpret_cast<float2*>(&peerbits_52)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_53);
        unsigned long long bits_54 = 0;
        bits_54 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bits_54, 16);
        unsigned long long peerbits_55 = _shfl_xor_4;
        float2 _f2_476 = make_float2(0.0f, 0.0f);
        float2 peer_56 = _f2_476;
        peer_56 = reinterpret_cast<float2*>(&peerbits_55)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_56);
        unsigned long long bits_57 = 0;
        bits_57 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bits_57, 16);
        unsigned long long peerbits_58 = _shfl_xor_5;
        float2 _f2_477 = make_float2(0.0f, 0.0f);
        float2 peer_59 = _f2_477;
        peer_59 = reinterpret_cast<float2*>(&peerbits_58)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_59);
        unsigned long long bits_60 = 0;
        bits_60 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, bits_60, 16);
        unsigned long long peerbits_61 = _shfl_xor_6;
        float2 _f2_478 = make_float2(0.0f, 0.0f);
        float2 peer_62 = _f2_478;
        peer_62 = reinterpret_cast<float2*>(&peerbits_61)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_62);
        unsigned long long bits_63 = 0;
        bits_63 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, bits_63, 16);
        unsigned long long peerbits_64 = _shfl_xor_7;
        float2 _f2_479 = make_float2(0.0f, 0.0f);
        float2 peer_65 = _f2_479;
        peer_65 = reinterpret_cast<float2*>(&peerbits_64)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_65);
        unsigned long long bits_66 = 0;
        bits_66 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, bits_66, 8);
        unsigned long long peerbits_67 = _shfl_xor_8;
        float2 _f2_480 = make_float2(0.0f, 0.0f);
        float2 peer_68 = _f2_480;
        peer_68 = reinterpret_cast<float2*>(&peerbits_67)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_68);
        unsigned long long bits_69 = 0;
        bits_69 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, bits_69, 8);
        unsigned long long peerbits_70 = _shfl_xor_9;
        float2 _f2_481 = make_float2(0.0f, 0.0f);
        float2 peer_71 = _f2_481;
        peer_71 = reinterpret_cast<float2*>(&peerbits_70)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_71);
        unsigned long long bits_72 = 0;
        bits_72 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, bits_72, 8);
        unsigned long long peerbits_73 = _shfl_xor_10;
        float2 _f2_482 = make_float2(0.0f, 0.0f);
        float2 peer_74 = _f2_482;
        peer_74 = reinterpret_cast<float2*>(&peerbits_73)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_74);
        unsigned long long bits_75 = 0;
        bits_75 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, bits_75, 8);
        unsigned long long peerbits_76 = _shfl_xor_11;
        float2 _f2_483 = make_float2(0.0f, 0.0f);
        float2 peer_77 = _f2_483;
        peer_77 = reinterpret_cast<float2*>(&peerbits_76)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_77);
        unsigned long long bits_78 = 0;
        bits_78 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, bits_78, 8);
        unsigned long long peerbits_79 = _shfl_xor_12;
        float2 _f2_484 = make_float2(0.0f, 0.0f);
        float2 peer_80 = _f2_484;
        peer_80 = reinterpret_cast<float2*>(&peerbits_79)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_80);
        unsigned long long bits_81 = 0;
        bits_81 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, bits_81, 8);
        unsigned long long peerbits_82 = _shfl_xor_13;
        float2 _f2_485 = make_float2(0.0f, 0.0f);
        float2 peer_83 = _f2_485;
        peer_83 = reinterpret_cast<float2*>(&peerbits_82)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_83);
        unsigned long long bits_84 = 0;
        bits_84 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, bits_84, 8);
        unsigned long long peerbits_85 = _shfl_xor_14;
        float2 _f2_486 = make_float2(0.0f, 0.0f);
        float2 peer_86 = _f2_486;
        peer_86 = reinterpret_cast<float2*>(&peerbits_85)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_86);
        unsigned long long bits_87 = 0;
        bits_87 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, bits_87, 8);
        unsigned long long peerbits_88 = _shfl_xor_15;
        float2 _f2_487 = make_float2(0.0f, 0.0f);
        float2 peer_89 = _f2_487;
        peer_89 = reinterpret_cast<float2*>(&peerbits_88)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_89);
        unsigned long long bits_90 = 0;
        bits_90 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, bits_90, 4);
        unsigned long long peerbits_91 = _shfl_xor_16;
        float2 _f2_488 = make_float2(0.0f, 0.0f);
        float2 peer_92 = _f2_488;
        peer_92 = reinterpret_cast<float2*>(&peerbits_91)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_92);
        unsigned long long bits_93 = 0;
        bits_93 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, bits_93, 4);
        unsigned long long peerbits_94 = _shfl_xor_17;
        float2 _f2_489 = make_float2(0.0f, 0.0f);
        float2 peer_95 = _f2_489;
        peer_95 = reinterpret_cast<float2*>(&peerbits_94)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_95);
        unsigned long long bits_96 = 0;
        bits_96 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, bits_96, 4);
        unsigned long long peerbits_97 = _shfl_xor_18;
        float2 _f2_490 = make_float2(0.0f, 0.0f);
        float2 peer_98 = _f2_490;
        peer_98 = reinterpret_cast<float2*>(&peerbits_97)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_98);
        unsigned long long bits_99 = 0;
        bits_99 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, bits_99, 4);
        unsigned long long peerbits_100 = _shfl_xor_19;
        float2 _f2_491 = make_float2(0.0f, 0.0f);
        float2 peer_101 = _f2_491;
        peer_101 = reinterpret_cast<float2*>(&peerbits_100)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_101);
        unsigned long long bits_102 = 0;
        bits_102 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, bits_102, 4);
        unsigned long long peerbits_103 = _shfl_xor_20;
        float2 _f2_492 = make_float2(0.0f, 0.0f);
        float2 peer_104 = _f2_492;
        peer_104 = reinterpret_cast<float2*>(&peerbits_103)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_104);
        unsigned long long bits_105 = 0;
        bits_105 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, bits_105, 4);
        unsigned long long peerbits_106 = _shfl_xor_21;
        float2 _f2_493 = make_float2(0.0f, 0.0f);
        float2 peer_107 = _f2_493;
        peer_107 = reinterpret_cast<float2*>(&peerbits_106)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_107);
        unsigned long long bits_108 = 0;
        bits_108 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, bits_108, 4);
        unsigned long long peerbits_109 = _shfl_xor_22;
        float2 _f2_494 = make_float2(0.0f, 0.0f);
        float2 peer_110 = _f2_494;
        peer_110 = reinterpret_cast<float2*>(&peerbits_109)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_110);
        unsigned long long bits_111 = 0;
        bits_111 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, bits_111, 4);
        unsigned long long peerbits_112 = _shfl_xor_23;
        float2 _f2_495 = make_float2(0.0f, 0.0f);
        float2 peer_113 = _f2_495;
        peer_113 = reinterpret_cast<float2*>(&peerbits_112)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_113);
        unsigned long long bits_114 = 0;
        bits_114 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, bits_114, 2);
        unsigned long long peerbits_115 = _shfl_xor_24;
        float2 _f2_496 = make_float2(0.0f, 0.0f);
        float2 peer_116 = _f2_496;
        peer_116 = reinterpret_cast<float2*>(&peerbits_115)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_116);
        unsigned long long bits_117 = 0;
        bits_117 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, bits_117, 2);
        unsigned long long peerbits_118 = _shfl_xor_25;
        float2 _f2_497 = make_float2(0.0f, 0.0f);
        float2 peer_119 = _f2_497;
        peer_119 = reinterpret_cast<float2*>(&peerbits_118)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_119);
        unsigned long long bits_120 = 0;
        bits_120 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, bits_120, 2);
        unsigned long long peerbits_121 = _shfl_xor_26;
        float2 _f2_498 = make_float2(0.0f, 0.0f);
        float2 peer_122 = _f2_498;
        peer_122 = reinterpret_cast<float2*>(&peerbits_121)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_122);
        unsigned long long bits_123 = 0;
        bits_123 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, bits_123, 2);
        unsigned long long peerbits_124 = _shfl_xor_27;
        float2 _f2_499 = make_float2(0.0f, 0.0f);
        float2 peer_125 = _f2_499;
        peer_125 = reinterpret_cast<float2*>(&peerbits_124)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_125);
        unsigned long long bits_126 = 0;
        bits_126 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, bits_126, 2);
        unsigned long long peerbits_127 = _shfl_xor_28;
        float2 _f2_500 = make_float2(0.0f, 0.0f);
        float2 peer_128 = _f2_500;
        peer_128 = reinterpret_cast<float2*>(&peerbits_127)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_128);
        unsigned long long bits_129 = 0;
        bits_129 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, bits_129, 2);
        unsigned long long peerbits_130 = _shfl_xor_29;
        float2 _f2_501 = make_float2(0.0f, 0.0f);
        float2 peer_131 = _f2_501;
        peer_131 = reinterpret_cast<float2*>(&peerbits_130)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_131);
        unsigned long long bits_132 = 0;
        bits_132 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, bits_132, 2);
        unsigned long long peerbits_133 = _shfl_xor_30;
        float2 _f2_502 = make_float2(0.0f, 0.0f);
        float2 peer_134 = _f2_502;
        peer_134 = reinterpret_cast<float2*>(&peerbits_133)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_134);
        unsigned long long bits_135 = 0;
        bits_135 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, bits_135, 2);
        unsigned long long peerbits_136 = _shfl_xor_31;
        float2 _f2_503 = make_float2(0.0f, 0.0f);
        float2 peer_137 = _f2_503;
        peer_137 = reinterpret_cast<float2*>(&peerbits_136)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_137);
        unsigned long long bits_138 = 0;
        bits_138 = reinterpret_cast<unsigned long long*>(&pairs[0])[0];
        unsigned long long _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, bits_138, 1);
        unsigned long long peerbits_139 = _shfl_xor_32;
        float2 _f2_504 = make_float2(0.0f, 0.0f);
        float2 peer_140 = _f2_504;
        peer_140 = reinterpret_cast<float2*>(&peerbits_139)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_140);
        unsigned long long bits_141 = 0;
        bits_141 = reinterpret_cast<unsigned long long*>(&pairs[1])[0];
        unsigned long long _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, bits_141, 1);
        unsigned long long peerbits_142 = _shfl_xor_33;
        float2 _f2_505 = make_float2(0.0f, 0.0f);
        float2 peer_143 = _f2_505;
        peer_143 = reinterpret_cast<float2*>(&peerbits_142)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_143);
        unsigned long long bits_144 = 0;
        bits_144 = reinterpret_cast<unsigned long long*>(&pairs[2])[0];
        unsigned long long _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, bits_144, 1);
        unsigned long long peerbits_145 = _shfl_xor_34;
        float2 _f2_506 = make_float2(0.0f, 0.0f);
        float2 peer_146 = _f2_506;
        peer_146 = reinterpret_cast<float2*>(&peerbits_145)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_146);
        unsigned long long bits_147 = 0;
        bits_147 = reinterpret_cast<unsigned long long*>(&pairs[3])[0];
        unsigned long long _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, bits_147, 1);
        unsigned long long peerbits_148 = _shfl_xor_35;
        float2 _f2_507 = make_float2(0.0f, 0.0f);
        float2 peer_149 = _f2_507;
        peer_149 = reinterpret_cast<float2*>(&peerbits_148)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_149);
        unsigned long long bits_150 = 0;
        bits_150 = reinterpret_cast<unsigned long long*>(&pairs[4])[0];
        unsigned long long _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, bits_150, 1);
        unsigned long long peerbits_151 = _shfl_xor_36;
        float2 _f2_508 = make_float2(0.0f, 0.0f);
        float2 peer_152 = _f2_508;
        peer_152 = reinterpret_cast<float2*>(&peerbits_151)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_152);
        unsigned long long bits_153 = 0;
        bits_153 = reinterpret_cast<unsigned long long*>(&pairs[5])[0];
        unsigned long long _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, bits_153, 1);
        unsigned long long peerbits_154 = _shfl_xor_37;
        float2 _f2_509 = make_float2(0.0f, 0.0f);
        float2 peer_155 = _f2_509;
        peer_155 = reinterpret_cast<float2*>(&peerbits_154)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_155);
        unsigned long long bits_156 = 0;
        bits_156 = reinterpret_cast<unsigned long long*>(&pairs[6])[0];
        unsigned long long _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, bits_156, 1);
        unsigned long long peerbits_157 = _shfl_xor_38;
        float2 _f2_510 = make_float2(0.0f, 0.0f);
        float2 peer_158 = _f2_510;
        peer_158 = reinterpret_cast<float2*>(&peerbits_157)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_158);
        unsigned long long bits_159 = 0;
        bits_159 = reinterpret_cast<unsigned long long*>(&pairs[7])[0];
        unsigned long long _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, bits_159, 1);
        unsigned long long peerbits_160 = _shfl_xor_39;
        float2 _f2_511 = make_float2(0.0f, 0.0f);
        float2 peer_161 = _f2_511;
        peer_161 = reinterpret_cast<float2*>(&peerbits_160)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_161);
        if (lane == 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(stats_addr + (unsigned int)(warp_0 * 8 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_0), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(stats_addr + (unsigned int)((warp_0 * 8 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_1), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_2), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_3), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_4;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_4) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_4), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_5;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_5) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_5), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_6;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_6) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_6), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_7;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_7) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_7), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_8;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_8) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_8), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_9;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_9) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_9), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_10;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_10) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_10), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_11;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_11) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_11), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_12;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_12) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 6) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_12), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_13;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_13) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 6) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_13), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_14;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_14) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 7) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_14), "f"(pairs[7].x) : "memory");
            uint32_t _mapa_15;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_15) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 7) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_15), "f"(pairs[7].y) : "memory");
            uint32_t _mapa_16;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_16) : "r"(stats_addr + (unsigned int)(warp_0 * 8 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_16), "f"(pairs[0].x) : "memory");
            uint32_t _mapa_17;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_17) : "r"(stats_addr + (unsigned int)((warp_0 * 8 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_17), "f"(pairs[0].y) : "memory");
            uint32_t _mapa_18;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_18) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_18), "f"(pairs[1].x) : "memory");
            uint32_t _mapa_19;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_19) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_19), "f"(pairs[1].y) : "memory");
            uint32_t _mapa_20;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_20) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_20), "f"(pairs[2].x) : "memory");
            uint32_t _mapa_21;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_21) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_21), "f"(pairs[2].y) : "memory");
            uint32_t _mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_22) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_22), "f"(pairs[3].x) : "memory");
            uint32_t _mapa_23;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_23) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_23), "f"(pairs[3].y) : "memory");
            uint32_t _mapa_24;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_24) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_24), "f"(pairs[4].x) : "memory");
            uint32_t _mapa_25;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_25) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_25), "f"(pairs[4].y) : "memory");
            uint32_t _mapa_26;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_26) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_26), "f"(pairs[5].x) : "memory");
            uint32_t _mapa_27;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_27) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_27), "f"(pairs[5].y) : "memory");
            uint32_t _mapa_28;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_28) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 6) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_28), "f"(pairs[6].x) : "memory");
            uint32_t _mapa_29;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_29) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 6) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_29), "f"(pairs[6].y) : "memory");
            uint32_t _mapa_30;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_30) : "r"(stats_addr + (unsigned int)((warp_0 * 8 + 7) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_30), "f"(pairs[7].x) : "memory");
            uint32_t _mapa_31;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_31) : "r"(stats_addr + (unsigned int)(((warp_0 * 8 + 7) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_31), "f"(pairs[7].y) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        int stat_n = lane / 8;
        int stat_w = lane % 8;
        float total_sq = 0.0f;
        float total_dot = 0.0f;
        if (stat_n < 8) {
            total_sq = stats[(stat_w * 8 + stat_n) * 2];
            total_dot = stats[(stat_w * 8 + stat_n) * 2 + 1];
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
        if (stat_n < 8 && stat_w == 0) {
            float _rsqrt_0 = rsqrtf(total_sq / 7168.0f + eps);
            float sigma = _rsqrt_0;
            logit = total_dot * sigma;
        }
        float logits[8];
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
        if (stat_n < 4) {
            total_sq2 = stats[(stat_w * 8 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 8 + 4 + stat_n) * 2 + 1];
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
        float _shfl_4 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_4;
        float _shfl_5 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_5;
        float _shfl_6 = __shfl_sync(0xFFFFFFFF, logit2, 16);
        logits[6] = _shfl_6;
        float _shfl_7 = __shfl_sync(0xFFFFFFFF, logit2, 24);
        logits[7] = _shfl_7;
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
        float weights[8];
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
        float2 _f2_512 = make_float2(correction, correction);
        float2 corr = _f2_512;
        const int woff_162 = 0;
        int base_163 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_5[4];
        float2 _f2_513 = make_float2(acc[0], acc[1]);
        float2 previous = _f2_513;
        a_5[0] = mul_f32x2_noftz(previous, corr);
        float2 _f2_514 = make_float2(acc[2], acc[3]);
        float2 previous_164 = _f2_514;
        a_5[1] = mul_f32x2_noftz(previous_164, corr);
        float2 _f2_515 = make_float2(acc[4], acc[5]);
        float2 previous_165 = _f2_515;
        a_5[2] = mul_f32x2_noftz(previous_165, corr);
        float2 _f2_516 = make_float2(acc[6], acc[7]);
        float2 previous_166 = _f2_516;
        a_5[3] = mul_f32x2_noftz(previous_166, corr);
        float2 _f2_517 = make_float2(weights[0], weights[0]);
        float2 weight = _f2_517;
        {
            float2 _f2_518 = make_float2(fsrc[0], fsrc[1]);
            float2 v_34 = _f2_518;
            a_5[0] = fma_f32x2_rn_noftz(weight, v_34, a_5[0]);
            float2 _f2_519 = make_float2(fsrc[2], fsrc[3]);
            float2 v_0_32 = _f2_519;
            a_5[1] = fma_f32x2_rn_noftz(weight, v_0_32, a_5[1]);
            float2 _f2_520 = make_float2(fsrc[4], fsrc[5]);
            float2 v_1_9 = _f2_520;
            a_5[2] = fma_f32x2_rn_noftz(weight, v_1_9, a_5[2]);
            float2 _f2_521 = make_float2(fsrc[6], fsrc[7]);
            float2 v_2_32 = _f2_521;
            a_5[3] = fma_f32x2_rn_noftz(weight, v_2_32, a_5[3]);
        }
        float2 _f2_526 = make_float2(weights[1], weights[1]);
        float2 weight_167 = _f2_526;
        {
            float2 _f2_527 = make_float2(fsrc[28], fsrc[29]);
            float2 v_35 = _f2_527;
            a_5[0] = fma_f32x2_rn_noftz(weight_167, v_35, a_5[0]);
            float2 _f2_528 = make_float2(fsrc[30], fsrc[31]);
            float2 v_0_33 = _f2_528;
            a_5[1] = fma_f32x2_rn_noftz(weight_167, v_0_33, a_5[1]);
            float2 _f2_529 = make_float2(fsrc[32], fsrc[33]);
            float2 v_1_10 = _f2_529;
            a_5[2] = fma_f32x2_rn_noftz(weight_167, v_1_10, a_5[2]);
            float2 _f2_530 = make_float2(fsrc[34], fsrc[35]);
            float2 v_2_33 = _f2_530;
            a_5[3] = fma_f32x2_rn_noftz(weight_167, v_2_33, a_5[3]);
        }
        float2 _f2_535 = make_float2(weights[2], weights[2]);
        float2 weight_168 = _f2_535;
        {
            float2 _f2_536 = make_float2(fsrc[56], fsrc[57]);
            float2 v_36 = _f2_536;
            a_5[0] = fma_f32x2_rn_noftz(weight_168, v_36, a_5[0]);
            float2 _f2_537 = make_float2(fsrc[58], fsrc[59]);
            float2 v_0_34 = _f2_537;
            a_5[1] = fma_f32x2_rn_noftz(weight_168, v_0_34, a_5[1]);
            float2 _f2_538 = make_float2(fsrc[60], fsrc[61]);
            float2 v_1_11 = _f2_538;
            a_5[2] = fma_f32x2_rn_noftz(weight_168, v_1_11, a_5[2]);
            float2 _f2_539 = make_float2(fsrc[62], fsrc[63]);
            float2 v_2_34 = _f2_539;
            a_5[3] = fma_f32x2_rn_noftz(weight_168, v_2_34, a_5[3]);
        }
        acc[0] = a_5[0].x;
        acc[1] = a_5[0].y;
        acc[2] = a_5[1].x;
        acc[3] = a_5[1].y;
        acc[4] = a_5[2].x;
        acc[5] = a_5[2].y;
        acc[6] = a_5[3].x;
        acc[7] = a_5[3].y;
        const int woff_169 = 4;
        int base_170 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_171[4];
        float2 _f2_544 = make_float2(acc[8], acc[9]);
        float2 previous_172 = _f2_544;
        a_171[0] = mul_f32x2_noftz(previous_172, corr);
        float2 _f2_545 = make_float2(acc[10], acc[11]);
        float2 previous_173 = _f2_545;
        a_171[1] = mul_f32x2_noftz(previous_173, corr);
        float2 _f2_546 = make_float2(acc[12], acc[13]);
        float2 previous_174 = _f2_546;
        a_171[2] = mul_f32x2_noftz(previous_174, corr);
        float2 _f2_547 = make_float2(acc[14], acc[15]);
        float2 previous_175 = _f2_547;
        a_171[3] = mul_f32x2_noftz(previous_175, corr);
        float2 _f2_548 = make_float2(weights[0], weights[0]);
        float2 weight_176 = _f2_548;
        {
            float2 _f2_549 = make_float2(fsrc[8], fsrc[9]);
            float2 v_37 = _f2_549;
            a_171[0] = fma_f32x2_rn_noftz(weight_176, v_37, a_171[0]);
            float2 _f2_550 = make_float2(fsrc[10], fsrc[11]);
            float2 v_0_35 = _f2_550;
            a_171[1] = fma_f32x2_rn_noftz(weight_176, v_0_35, a_171[1]);
            float2 _f2_551 = make_float2(fsrc[12], fsrc[13]);
            float2 v_1_12 = _f2_551;
            a_171[2] = fma_f32x2_rn_noftz(weight_176, v_1_12, a_171[2]);
            float2 _f2_552 = make_float2(fsrc[14], fsrc[15]);
            float2 v_2_35 = _f2_552;
            a_171[3] = fma_f32x2_rn_noftz(weight_176, v_2_35, a_171[3]);
        }
        float2 _f2_557 = make_float2(weights[1], weights[1]);
        float2 weight_177 = _f2_557;
        {
            float2 _f2_558 = make_float2(fsrc[36], fsrc[37]);
            float2 v_38 = _f2_558;
            a_171[0] = fma_f32x2_rn_noftz(weight_177, v_38, a_171[0]);
            float2 _f2_559 = make_float2(fsrc[38], fsrc[39]);
            float2 v_0_36 = _f2_559;
            a_171[1] = fma_f32x2_rn_noftz(weight_177, v_0_36, a_171[1]);
            float2 _f2_560 = make_float2(fsrc[40], fsrc[41]);
            float2 v_1_13 = _f2_560;
            a_171[2] = fma_f32x2_rn_noftz(weight_177, v_1_13, a_171[2]);
            float2 _f2_561 = make_float2(fsrc[42], fsrc[43]);
            float2 v_2_36 = _f2_561;
            a_171[3] = fma_f32x2_rn_noftz(weight_177, v_2_36, a_171[3]);
        }
        float2 _f2_566 = make_float2(weights[2], weights[2]);
        float2 weight_178 = _f2_566;
        {
            float2 _f2_567 = make_float2(fsrc[64], fsrc[65]);
            float2 v_39 = _f2_567;
            a_171[0] = fma_f32x2_rn_noftz(weight_178, v_39, a_171[0]);
            float2 _f2_568 = make_float2(fsrc[66], fsrc[67]);
            float2 v_0_37 = _f2_568;
            a_171[1] = fma_f32x2_rn_noftz(weight_178, v_0_37, a_171[1]);
            float2 _f2_569 = make_float2(fsrc[68], fsrc[69]);
            float2 v_1_14 = _f2_569;
            a_171[2] = fma_f32x2_rn_noftz(weight_178, v_1_14, a_171[2]);
            float2 _f2_570 = make_float2(fsrc[70], fsrc[71]);
            float2 v_2_37 = _f2_570;
            a_171[3] = fma_f32x2_rn_noftz(weight_178, v_2_37, a_171[3]);
        }
        acc[8] = a_171[0].x;
        acc[9] = a_171[0].y;
        acc[10] = a_171[1].x;
        acc[11] = a_171[1].y;
        acc[12] = a_171[2].x;
        acc[13] = a_171[2].y;
        acc[14] = a_171[3].x;
        acc[15] = a_171[3].y;
        const int woff_179 = 8;
        int base_180 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_181[4];
        float2 _f2_575 = make_float2(acc[16], acc[17]);
        float2 previous_182 = _f2_575;
        a_181[0] = mul_f32x2_noftz(previous_182, corr);
        float2 _f2_576 = make_float2(acc[18], acc[19]);
        float2 previous_183 = _f2_576;
        a_181[1] = mul_f32x2_noftz(previous_183, corr);
        float2 _f2_577 = make_float2(acc[20], acc[21]);
        float2 previous_184 = _f2_577;
        a_181[2] = mul_f32x2_noftz(previous_184, corr);
        float2 _f2_578 = make_float2(acc[22], acc[23]);
        float2 previous_185 = _f2_578;
        a_181[3] = mul_f32x2_noftz(previous_185, corr);
        float2 _f2_579 = make_float2(weights[0], weights[0]);
        float2 weight_186 = _f2_579;
        {
            float2 _f2_580 = make_float2(fsrc[16], fsrc[17]);
            float2 v_40 = _f2_580;
            a_181[0] = fma_f32x2_rn_noftz(weight_186, v_40, a_181[0]);
            float2 _f2_581 = make_float2(fsrc[18], fsrc[19]);
            float2 v_0_38 = _f2_581;
            a_181[1] = fma_f32x2_rn_noftz(weight_186, v_0_38, a_181[1]);
            float2 _f2_582 = make_float2(fsrc[20], fsrc[21]);
            float2 v_1_15 = _f2_582;
            a_181[2] = fma_f32x2_rn_noftz(weight_186, v_1_15, a_181[2]);
            float2 _f2_583 = make_float2(fsrc[22], fsrc[23]);
            float2 v_2_38 = _f2_583;
            a_181[3] = fma_f32x2_rn_noftz(weight_186, v_2_38, a_181[3]);
        }
        float2 _f2_588 = make_float2(weights[1], weights[1]);
        float2 weight_187 = _f2_588;
        {
            float2 _f2_589 = make_float2(fsrc[44], fsrc[45]);
            float2 v_41 = _f2_589;
            a_181[0] = fma_f32x2_rn_noftz(weight_187, v_41, a_181[0]);
            float2 _f2_590 = make_float2(fsrc[46], fsrc[47]);
            float2 v_0_39 = _f2_590;
            a_181[1] = fma_f32x2_rn_noftz(weight_187, v_0_39, a_181[1]);
            float2 _f2_591 = make_float2(fsrc[48], fsrc[49]);
            float2 v_1_16 = _f2_591;
            a_181[2] = fma_f32x2_rn_noftz(weight_187, v_1_16, a_181[2]);
            float2 _f2_592 = make_float2(fsrc[50], fsrc[51]);
            float2 v_2_39 = _f2_592;
            a_181[3] = fma_f32x2_rn_noftz(weight_187, v_2_39, a_181[3]);
        }
        float2 _f2_597 = make_float2(weights[2], weights[2]);
        float2 weight_188 = _f2_597;
        {
            float2 _f2_598 = make_float2(fsrc[72], fsrc[73]);
            float2 v_42 = _f2_598;
            a_181[0] = fma_f32x2_rn_noftz(weight_188, v_42, a_181[0]);
            float2 _f2_599 = make_float2(fsrc[74], fsrc[75]);
            float2 v_0_40 = _f2_599;
            a_181[1] = fma_f32x2_rn_noftz(weight_188, v_0_40, a_181[1]);
            float2 _f2_600 = make_float2(fsrc[76], fsrc[77]);
            float2 v_1_17 = _f2_600;
            a_181[2] = fma_f32x2_rn_noftz(weight_188, v_1_17, a_181[2]);
            float2 _f2_601 = make_float2(fsrc[78], fsrc[79]);
            float2 v_2_40 = _f2_601;
            a_181[3] = fma_f32x2_rn_noftz(weight_188, v_2_40, a_181[3]);
        }
        acc[16] = a_181[0].x;
        acc[17] = a_181[0].y;
        acc[18] = a_181[1].x;
        acc[19] = a_181[1].y;
        acc[20] = a_181[2].x;
        acc[21] = a_181[2].y;
        acc[22] = a_181[3].x;
        acc[23] = a_181[3].y;
        const int woff_189 = 12;
        int base_190 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_191[4];
        float2 _f2_606 = make_float2(acc[24], acc[25]);
        float2 previous_192 = _f2_606;
        a_191[0] = mul_f32x2_noftz(previous_192, corr);
        float2 _f2_607 = make_float2(acc[26], acc[27]);
        float2 previous_193 = _f2_607;
        a_191[1] = mul_f32x2_noftz(previous_193, corr);
        float2 _f2_608 = make_float2(weights[0], weights[0]);
        float2 weight_194 = _f2_608;
        {
            float2 _f2_609 = make_float2(fsrc[24], fsrc[25]);
            float2 v_43 = _f2_609;
            a_191[0] = fma_f32x2_rn_noftz(weight_194, v_43, a_191[0]);
            float2 _f2_610 = make_float2(fsrc[26], fsrc[27]);
            float2 v_0_41 = _f2_610;
            a_191[1] = fma_f32x2_rn_noftz(weight_194, v_0_41, a_191[1]);
        }
        float2 _f2_613 = make_float2(weights[1], weights[1]);
        float2 weight_195 = _f2_613;
        {
            float2 _f2_614 = make_float2(fsrc[52], fsrc[53]);
            float2 v_44 = _f2_614;
            a_191[0] = fma_f32x2_rn_noftz(weight_195, v_44, a_191[0]);
            float2 _f2_615 = make_float2(fsrc[54], fsrc[55]);
            float2 v_0_42 = _f2_615;
            a_191[1] = fma_f32x2_rn_noftz(weight_195, v_0_42, a_191[1]);
        }
        float2 _f2_618 = make_float2(weights[2], weights[2]);
        float2 weight_196 = _f2_618;
        {
            float2 _f2_619 = make_float2(fsrc[80], fsrc[81]);
            float2 v_45 = _f2_619;
            a_191[0] = fma_f32x2_rn_noftz(weight_196, v_45, a_191[0]);
            float2 _f2_620 = make_float2(fsrc[82], fsrc[83]);
            float2 v_0_43 = _f2_620;
            a_191[1] = fma_f32x2_rn_noftz(weight_196, v_0_43, a_191[1]);
        }
        acc[24] = a_191[0].x;
        acc[25] = a_191[0].y;
        acc[26] = a_191[1].x;
        acc[27] = a_191[1].y;
        sum_running = sum_running * correction + sum_weights;
        max_running = max_new;
        float max_chunk_197 = -3.4028234663852886e+38f;
        float _fmax_4 = fmaxf(max_chunk_197, logits[3]);
        max_chunk_197 = _fmax_4;
        float _fmax_5 = fmaxf(max_chunk_197, logits[4]);
        max_chunk_197 = _fmax_5;
        float _fmax_6 = fmaxf(max_chunk_197, logits[5]);
        max_chunk_197 = _fmax_6;
        float _fmax_7 = fmaxf(max_running, max_chunk_197);
        float max_new_198 = _fmax_7;
        float _exp2_4 = approx_exp2((max_running - max_new_198) * 1.4426950408889634f);
        float correction_199 = _exp2_4;
        float weights_200[8];
        float sum_weights_201 = 0.0f;
        float _exp2_5 = approx_exp2((logits[3] - max_new_198) * 1.4426950408889634f);
        weights_200[3] = _exp2_5;
        sum_weights_201 += weights_200[3];
        float _exp2_6 = approx_exp2((logits[4] - max_new_198) * 1.4426950408889634f);
        weights_200[4] = _exp2_6;
        sum_weights_201 += weights_200[4];
        float _exp2_7 = approx_exp2((logits[5] - max_new_198) * 1.4426950408889634f);
        weights_200[5] = _exp2_7;
        sum_weights_201 += weights_200[5];
        float2 _f2_623 = make_float2(correction_199, correction_199);
        float2 corr_202 = _f2_623;
        const int woff_203 = 0;
        int base_204 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_205[4];
        float2 _f2_624 = make_float2(acc[0], acc[1]);
        float2 previous_206 = _f2_624;
        a_205[0] = mul_f32x2_noftz(previous_206, corr_202);
        float2 _f2_625 = make_float2(acc[2], acc[3]);
        float2 previous_207 = _f2_625;
        a_205[1] = mul_f32x2_noftz(previous_207, corr_202);
        float2 _f2_626 = make_float2(acc[4], acc[5]);
        float2 previous_208 = _f2_626;
        a_205[2] = mul_f32x2_noftz(previous_208, corr_202);
        float2 _f2_627 = make_float2(acc[6], acc[7]);
        float2 previous_209 = _f2_627;
        a_205[3] = mul_f32x2_noftz(previous_209, corr_202);
        float2 _f2_628 = make_float2(weights_200[3], weights_200[3]);
        float2 weight_210 = _f2_628;
        {
            unsigned int sw2[4];
            sw2[0] = words[42 + woff_203];
            sw2[1] = words[42 + woff_203 + 1];
            sw2[2] = words[42 + woff_203 + 2];
            sw2[3] = words[42 + woff_203 + 3];
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
            float2 _f2_633 = make_float2(sw2_f32[0], sw2_f32[1]);
            float2 v_46 = _f2_633;
            a_205[0] = fma_f32x2_rn_noftz(weight_210, v_46, a_205[0]);
            float2 _f2_634 = make_float2(sw2_f32[2], sw2_f32[3]);
            float2 v_0_44 = _f2_634;
            a_205[1] = fma_f32x2_rn_noftz(weight_210, v_0_44, a_205[1]);
            float2 _f2_635 = make_float2(sw2_f32[4], sw2_f32[5]);
            float2 v_1_18 = _f2_635;
            a_205[2] = fma_f32x2_rn_noftz(weight_210, v_1_18, a_205[2]);
            float2 _f2_636 = make_float2(sw2_f32[6], sw2_f32[7]);
            float2 v_2_41 = _f2_636;
            a_205[3] = fma_f32x2_rn_noftz(weight_210, v_2_41, a_205[3]);
        }
        float2 _f2_637 = make_float2(weights_200[4], weights_200[4]);
        float2 weight_211 = _f2_637;
        {
            unsigned int sw2_1[4];
            sw2_1[0] = words[56 + woff_203];
            sw2_1[1] = words[56 + woff_203 + 1];
            sw2_1[2] = words[56 + woff_203 + 2];
            sw2_1[3] = words[56 + woff_203 + 3];
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
            float2 _f2_642 = make_float2(sw2_f32_1[0], sw2_f32_1[1]);
            float2 v_47 = _f2_642;
            a_205[0] = fma_f32x2_rn_noftz(weight_211, v_47, a_205[0]);
            float2 _f2_643 = make_float2(sw2_f32_1[2], sw2_f32_1[3]);
            float2 v_0_45 = _f2_643;
            a_205[1] = fma_f32x2_rn_noftz(weight_211, v_0_45, a_205[1]);
            float2 _f2_644 = make_float2(sw2_f32_1[4], sw2_f32_1[5]);
            float2 v_1_19 = _f2_644;
            a_205[2] = fma_f32x2_rn_noftz(weight_211, v_1_19, a_205[2]);
            float2 _f2_645 = make_float2(sw2_f32_1[6], sw2_f32_1[7]);
            float2 v_2_42 = _f2_645;
            a_205[3] = fma_f32x2_rn_noftz(weight_211, v_2_42, a_205[3]);
        }
        float2 _f2_646 = make_float2(weights_200[5], weights_200[5]);
        float2 weight_212 = _f2_646;
        {
            unsigned int sw2_2[4];
            sw2_2[0] = words[70 + woff_203];
            sw2_2[1] = words[70 + woff_203 + 1];
            sw2_2[2] = words[70 + woff_203 + 2];
            sw2_2[3] = words[70 + woff_203 + 3];
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
            float2 _f2_651 = make_float2(sw2_f32_2[0], sw2_f32_2[1]);
            float2 v_48 = _f2_651;
            a_205[0] = fma_f32x2_rn_noftz(weight_212, v_48, a_205[0]);
            float2 _f2_652 = make_float2(sw2_f32_2[2], sw2_f32_2[3]);
            float2 v_0_46 = _f2_652;
            a_205[1] = fma_f32x2_rn_noftz(weight_212, v_0_46, a_205[1]);
            float2 _f2_653 = make_float2(sw2_f32_2[4], sw2_f32_2[5]);
            float2 v_1_20 = _f2_653;
            a_205[2] = fma_f32x2_rn_noftz(weight_212, v_1_20, a_205[2]);
            float2 _f2_654 = make_float2(sw2_f32_2[6], sw2_f32_2[7]);
            float2 v_2_43 = _f2_654;
            a_205[3] = fma_f32x2_rn_noftz(weight_212, v_2_43, a_205[3]);
        }
        acc[0] = a_205[0].x;
        acc[1] = a_205[0].y;
        acc[2] = a_205[1].x;
        acc[3] = a_205[1].y;
        acc[4] = a_205[2].x;
        acc[5] = a_205[2].y;
        acc[6] = a_205[3].x;
        acc[7] = a_205[3].y;
        const int woff_213 = 4;
        int base_214 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_215[4];
        float2 _f2_655 = make_float2(acc[8], acc[9]);
        float2 previous_216 = _f2_655;
        a_215[0] = mul_f32x2_noftz(previous_216, corr_202);
        float2 _f2_656 = make_float2(acc[10], acc[11]);
        float2 previous_217 = _f2_656;
        a_215[1] = mul_f32x2_noftz(previous_217, corr_202);
        float2 _f2_657 = make_float2(acc[12], acc[13]);
        float2 previous_218 = _f2_657;
        a_215[2] = mul_f32x2_noftz(previous_218, corr_202);
        float2 _f2_658 = make_float2(acc[14], acc[15]);
        float2 previous_219 = _f2_658;
        a_215[3] = mul_f32x2_noftz(previous_219, corr_202);
        float2 _f2_659 = make_float2(weights_200[3], weights_200[3]);
        float2 weight_220 = _f2_659;
        {
            unsigned int sw2_3[4];
            sw2_3[0] = words[42 + woff_213];
            sw2_3[1] = words[42 + woff_213 + 1];
            sw2_3[2] = words[42 + woff_213 + 2];
            sw2_3[3] = words[42 + woff_213 + 3];
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
            float2 _f2_664 = make_float2(sw2_f32_3[0], sw2_f32_3[1]);
            float2 v_49 = _f2_664;
            a_215[0] = fma_f32x2_rn_noftz(weight_220, v_49, a_215[0]);
            float2 _f2_665 = make_float2(sw2_f32_3[2], sw2_f32_3[3]);
            float2 v_0_47 = _f2_665;
            a_215[1] = fma_f32x2_rn_noftz(weight_220, v_0_47, a_215[1]);
            float2 _f2_666 = make_float2(sw2_f32_3[4], sw2_f32_3[5]);
            float2 v_1_21 = _f2_666;
            a_215[2] = fma_f32x2_rn_noftz(weight_220, v_1_21, a_215[2]);
            float2 _f2_667 = make_float2(sw2_f32_3[6], sw2_f32_3[7]);
            float2 v_2_44 = _f2_667;
            a_215[3] = fma_f32x2_rn_noftz(weight_220, v_2_44, a_215[3]);
        }
        float2 _f2_668 = make_float2(weights_200[4], weights_200[4]);
        float2 weight_221 = _f2_668;
        {
            unsigned int sw2_4[4];
            sw2_4[0] = words[56 + woff_213];
            sw2_4[1] = words[56 + woff_213 + 1];
            sw2_4[2] = words[56 + woff_213 + 2];
            sw2_4[3] = words[56 + woff_213 + 3];
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
            float2 _f2_673 = make_float2(sw2_f32_4[0], sw2_f32_4[1]);
            float2 v_50 = _f2_673;
            a_215[0] = fma_f32x2_rn_noftz(weight_221, v_50, a_215[0]);
            float2 _f2_674 = make_float2(sw2_f32_4[2], sw2_f32_4[3]);
            float2 v_0_48 = _f2_674;
            a_215[1] = fma_f32x2_rn_noftz(weight_221, v_0_48, a_215[1]);
            float2 _f2_675 = make_float2(sw2_f32_4[4], sw2_f32_4[5]);
            float2 v_1_22 = _f2_675;
            a_215[2] = fma_f32x2_rn_noftz(weight_221, v_1_22, a_215[2]);
            float2 _f2_676 = make_float2(sw2_f32_4[6], sw2_f32_4[7]);
            float2 v_2_45 = _f2_676;
            a_215[3] = fma_f32x2_rn_noftz(weight_221, v_2_45, a_215[3]);
        }
        float2 _f2_677 = make_float2(weights_200[5], weights_200[5]);
        float2 weight_222 = _f2_677;
        {
            unsigned int sw2_5[4];
            sw2_5[0] = words[70 + woff_213];
            sw2_5[1] = words[70 + woff_213 + 1];
            sw2_5[2] = words[70 + woff_213 + 2];
            sw2_5[3] = words[70 + woff_213 + 3];
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
            float2 _f2_682 = make_float2(sw2_f32_5[0], sw2_f32_5[1]);
            float2 v_51 = _f2_682;
            a_215[0] = fma_f32x2_rn_noftz(weight_222, v_51, a_215[0]);
            float2 _f2_683 = make_float2(sw2_f32_5[2], sw2_f32_5[3]);
            float2 v_0_49 = _f2_683;
            a_215[1] = fma_f32x2_rn_noftz(weight_222, v_0_49, a_215[1]);
            float2 _f2_684 = make_float2(sw2_f32_5[4], sw2_f32_5[5]);
            float2 v_1_23 = _f2_684;
            a_215[2] = fma_f32x2_rn_noftz(weight_222, v_1_23, a_215[2]);
            float2 _f2_685 = make_float2(sw2_f32_5[6], sw2_f32_5[7]);
            float2 v_2_46 = _f2_685;
            a_215[3] = fma_f32x2_rn_noftz(weight_222, v_2_46, a_215[3]);
        }
        acc[8] = a_215[0].x;
        acc[9] = a_215[0].y;
        acc[10] = a_215[1].x;
        acc[11] = a_215[1].y;
        acc[12] = a_215[2].x;
        acc[13] = a_215[2].y;
        acc[14] = a_215[3].x;
        acc[15] = a_215[3].y;
        const int woff_223 = 8;
        int base_224 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_225[4];
        float2 _f2_686 = make_float2(acc[16], acc[17]);
        float2 previous_226 = _f2_686;
        a_225[0] = mul_f32x2_noftz(previous_226, corr_202);
        float2 _f2_687 = make_float2(acc[18], acc[19]);
        float2 previous_227 = _f2_687;
        a_225[1] = mul_f32x2_noftz(previous_227, corr_202);
        float2 _f2_688 = make_float2(acc[20], acc[21]);
        float2 previous_228 = _f2_688;
        a_225[2] = mul_f32x2_noftz(previous_228, corr_202);
        float2 _f2_689 = make_float2(acc[22], acc[23]);
        float2 previous_229 = _f2_689;
        a_225[3] = mul_f32x2_noftz(previous_229, corr_202);
        float2 _f2_690 = make_float2(weights_200[3], weights_200[3]);
        float2 weight_230 = _f2_690;
        {
            unsigned int sw2_6[4];
            sw2_6[0] = words[42 + woff_223];
            sw2_6[1] = words[42 + woff_223 + 1];
            sw2_6[2] = words[42 + woff_223 + 2];
            sw2_6[3] = words[42 + woff_223 + 3];
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
            float2 _f2_695 = make_float2(sw2_f32_6[0], sw2_f32_6[1]);
            float2 v_52 = _f2_695;
            a_225[0] = fma_f32x2_rn_noftz(weight_230, v_52, a_225[0]);
            float2 _f2_696 = make_float2(sw2_f32_6[2], sw2_f32_6[3]);
            float2 v_0_50 = _f2_696;
            a_225[1] = fma_f32x2_rn_noftz(weight_230, v_0_50, a_225[1]);
            float2 _f2_697 = make_float2(sw2_f32_6[4], sw2_f32_6[5]);
            float2 v_1_24 = _f2_697;
            a_225[2] = fma_f32x2_rn_noftz(weight_230, v_1_24, a_225[2]);
            float2 _f2_698 = make_float2(sw2_f32_6[6], sw2_f32_6[7]);
            float2 v_2_47 = _f2_698;
            a_225[3] = fma_f32x2_rn_noftz(weight_230, v_2_47, a_225[3]);
        }
        float2 _f2_699 = make_float2(weights_200[4], weights_200[4]);
        float2 weight_231 = _f2_699;
        {
            unsigned int sw2_7[4];
            sw2_7[0] = words[56 + woff_223];
            sw2_7[1] = words[56 + woff_223 + 1];
            sw2_7[2] = words[56 + woff_223 + 2];
            sw2_7[3] = words[56 + woff_223 + 3];
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
            float2 _f2_704 = make_float2(sw2_f32_7[0], sw2_f32_7[1]);
            float2 v_53 = _f2_704;
            a_225[0] = fma_f32x2_rn_noftz(weight_231, v_53, a_225[0]);
            float2 _f2_705 = make_float2(sw2_f32_7[2], sw2_f32_7[3]);
            float2 v_0_51 = _f2_705;
            a_225[1] = fma_f32x2_rn_noftz(weight_231, v_0_51, a_225[1]);
            float2 _f2_706 = make_float2(sw2_f32_7[4], sw2_f32_7[5]);
            float2 v_1_25 = _f2_706;
            a_225[2] = fma_f32x2_rn_noftz(weight_231, v_1_25, a_225[2]);
            float2 _f2_707 = make_float2(sw2_f32_7[6], sw2_f32_7[7]);
            float2 v_2_48 = _f2_707;
            a_225[3] = fma_f32x2_rn_noftz(weight_231, v_2_48, a_225[3]);
        }
        float2 _f2_708 = make_float2(weights_200[5], weights_200[5]);
        float2 weight_232 = _f2_708;
        {
            unsigned int sw2_8[4];
            sw2_8[0] = words[70 + woff_223];
            sw2_8[1] = words[70 + woff_223 + 1];
            sw2_8[2] = words[70 + woff_223 + 2];
            sw2_8[3] = words[70 + woff_223 + 3];
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
            float2 _f2_713 = make_float2(sw2_f32_8[0], sw2_f32_8[1]);
            float2 v_54 = _f2_713;
            a_225[0] = fma_f32x2_rn_noftz(weight_232, v_54, a_225[0]);
            float2 _f2_714 = make_float2(sw2_f32_8[2], sw2_f32_8[3]);
            float2 v_0_52 = _f2_714;
            a_225[1] = fma_f32x2_rn_noftz(weight_232, v_0_52, a_225[1]);
            float2 _f2_715 = make_float2(sw2_f32_8[4], sw2_f32_8[5]);
            float2 v_1_26 = _f2_715;
            a_225[2] = fma_f32x2_rn_noftz(weight_232, v_1_26, a_225[2]);
            float2 _f2_716 = make_float2(sw2_f32_8[6], sw2_f32_8[7]);
            float2 v_2_49 = _f2_716;
            a_225[3] = fma_f32x2_rn_noftz(weight_232, v_2_49, a_225[3]);
        }
        acc[16] = a_225[0].x;
        acc[17] = a_225[0].y;
        acc[18] = a_225[1].x;
        acc[19] = a_225[1].y;
        acc[20] = a_225[2].x;
        acc[21] = a_225[2].y;
        acc[22] = a_225[3].x;
        acc[23] = a_225[3].y;
        const int woff_233 = 12;
        int base_234 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_235[4];
        float2 _f2_717 = make_float2(acc[24], acc[25]);
        float2 previous_236 = _f2_717;
        a_235[0] = mul_f32x2_noftz(previous_236, corr_202);
        float2 _f2_718 = make_float2(acc[26], acc[27]);
        float2 previous_237 = _f2_718;
        a_235[1] = mul_f32x2_noftz(previous_237, corr_202);
        float2 _f2_719 = make_float2(weights_200[3], weights_200[3]);
        float2 weight_238 = _f2_719;
        {
            unsigned int sw2_9[4];
            sw2_9[0] = words[42 + woff_233];
            sw2_9[1] = words[42 + woff_233 + 1];
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
            float2 _f2_722 = make_float2(sw2_f32_9[0], sw2_f32_9[1]);
            float2 v_55 = _f2_722;
            a_235[0] = fma_f32x2_rn_noftz(weight_238, v_55, a_235[0]);
            float2 _f2_723 = make_float2(sw2_f32_9[2], sw2_f32_9[3]);
            float2 v_0_53 = _f2_723;
            a_235[1] = fma_f32x2_rn_noftz(weight_238, v_0_53, a_235[1]);
        }
        float2 _f2_724 = make_float2(weights_200[4], weights_200[4]);
        float2 weight_239 = _f2_724;
        {
            unsigned int sw2_10[4];
            sw2_10[0] = words[56 + woff_233];
            sw2_10[1] = words[56 + woff_233 + 1];
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
            float2 _f2_727 = make_float2(sw2_f32_10[0], sw2_f32_10[1]);
            float2 v_56 = _f2_727;
            a_235[0] = fma_f32x2_rn_noftz(weight_239, v_56, a_235[0]);
            float2 _f2_728 = make_float2(sw2_f32_10[2], sw2_f32_10[3]);
            float2 v_0_54 = _f2_728;
            a_235[1] = fma_f32x2_rn_noftz(weight_239, v_0_54, a_235[1]);
        }
        float2 _f2_729 = make_float2(weights_200[5], weights_200[5]);
        float2 weight_240 = _f2_729;
        {
            unsigned int sw2_11[4];
            sw2_11[0] = words[70 + woff_233];
            sw2_11[1] = words[70 + woff_233 + 1];
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
            float2 _f2_732 = make_float2(sw2_f32_11[0], sw2_f32_11[1]);
            float2 v_57 = _f2_732;
            a_235[0] = fma_f32x2_rn_noftz(weight_240, v_57, a_235[0]);
            float2 _f2_733 = make_float2(sw2_f32_11[2], sw2_f32_11[3]);
            float2 v_0_55 = _f2_733;
            a_235[1] = fma_f32x2_rn_noftz(weight_240, v_0_55, a_235[1]);
        }
        acc[24] = a_235[0].x;
        acc[25] = a_235[0].y;
        acc[26] = a_235[1].x;
        acc[27] = a_235[1].y;
        sum_running = sum_running * correction_199 + sum_weights_201;
        max_running = max_new_198;
        float max_chunk_241 = -3.4028234663852886e+38f;
        float _fmax_8 = fmaxf(max_chunk_241, logits[6]);
        max_chunk_241 = _fmax_8;
        float _fmax_9 = fmaxf(max_chunk_241, logits[7]);
        max_chunk_241 = _fmax_9;
        float _fmax_10 = fmaxf(max_running, max_chunk_241);
        float max_new_242 = _fmax_10;
        float _exp2_8 = approx_exp2((max_running - max_new_242) * 1.4426950408889634f);
        float correction_243 = _exp2_8;
        float weights_244[8];
        float sum_weights_245 = 0.0f;
        float _exp2_9 = approx_exp2((logits[6] - max_new_242) * 1.4426950408889634f);
        weights_244[6] = _exp2_9;
        sum_weights_245 += weights_244[6];
        float _exp2_10 = approx_exp2((logits[7] - max_new_242) * 1.4426950408889634f);
        weights_244[7] = _exp2_10;
        sum_weights_245 += weights_244[7];
        float2 _f2_734 = make_float2(correction_243, correction_243);
        float2 corr_246 = _f2_734;
        const int woff_247 = 0;
        int base_248 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float2 a_249[4];
        float2 _f2_735 = make_float2(acc[0], acc[1]);
        float2 previous_250 = _f2_735;
        a_249[0] = mul_f32x2_noftz(previous_250, corr_246);
        float2 _f2_736 = make_float2(acc[2], acc[3]);
        float2 previous_251 = _f2_736;
        a_249[1] = mul_f32x2_noftz(previous_251, corr_246);
        float2 _f2_737 = make_float2(acc[4], acc[5]);
        float2 previous_252 = _f2_737;
        a_249[2] = mul_f32x2_noftz(previous_252, corr_246);
        float2 _f2_738 = make_float2(acc[6], acc[7]);
        float2 previous_253 = _f2_738;
        a_249[3] = mul_f32x2_noftz(previous_253, corr_246);
        float2 _f2_739 = make_float2(weights_244[6], weights_244[6]);
        float2 weight_254 = _f2_739;
        {
            unsigned int sw2_12[4];
            sw2_12[0] = words[84 + woff_247];
            sw2_12[1] = words[84 + woff_247 + 1];
            sw2_12[2] = words[84 + woff_247 + 2];
            sw2_12[3] = words[84 + woff_247 + 3];
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
            float2 _f2_744 = make_float2(sw2_f32_12[0], sw2_f32_12[1]);
            float2 v_58 = _f2_744;
            a_249[0] = fma_f32x2_rn_noftz(weight_254, v_58, a_249[0]);
            float2 _f2_745 = make_float2(sw2_f32_12[2], sw2_f32_12[3]);
            float2 v_0_56 = _f2_745;
            a_249[1] = fma_f32x2_rn_noftz(weight_254, v_0_56, a_249[1]);
            float2 _f2_746 = make_float2(sw2_f32_12[4], sw2_f32_12[5]);
            float2 v_1_27 = _f2_746;
            a_249[2] = fma_f32x2_rn_noftz(weight_254, v_1_27, a_249[2]);
            float2 _f2_747 = make_float2(sw2_f32_12[6], sw2_f32_12[7]);
            float2 v_2_50 = _f2_747;
            a_249[3] = fma_f32x2_rn_noftz(weight_254, v_2_50, a_249[3]);
        }
        float2 _f2_748 = make_float2(weights_244[7], weights_244[7]);
        float2 weight_255 = _f2_748;
        {
            unsigned int sw2_13[4];
            sw2_13[0] = words[98 + woff_247];
            sw2_13[1] = words[98 + woff_247 + 1];
            sw2_13[2] = words[98 + woff_247 + 2];
            sw2_13[3] = words[98 + woff_247 + 3];
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
            float2 _f2_753 = make_float2(sw2_f32_13[0], sw2_f32_13[1]);
            float2 v_59 = _f2_753;
            a_249[0] = fma_f32x2_rn_noftz(weight_255, v_59, a_249[0]);
            float2 _f2_754 = make_float2(sw2_f32_13[2], sw2_f32_13[3]);
            float2 v_0_57 = _f2_754;
            a_249[1] = fma_f32x2_rn_noftz(weight_255, v_0_57, a_249[1]);
            float2 _f2_755 = make_float2(sw2_f32_13[4], sw2_f32_13[5]);
            float2 v_1_28 = _f2_755;
            a_249[2] = fma_f32x2_rn_noftz(weight_255, v_1_28, a_249[2]);
            float2 _f2_756 = make_float2(sw2_f32_13[6], sw2_f32_13[7]);
            float2 v_2_51 = _f2_756;
            a_249[3] = fma_f32x2_rn_noftz(weight_255, v_2_51, a_249[3]);
        }
        acc[0] = a_249[0].x;
        acc[1] = a_249[0].y;
        acc[2] = a_249[1].x;
        acc[3] = a_249[1].y;
        acc[4] = a_249[2].x;
        acc[5] = a_249[2].y;
        acc[6] = a_249[3].x;
        acc[7] = a_249[3].y;
        const int woff_256 = 4;
        int base_257 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float2 a_258[4];
        float2 _f2_757 = make_float2(acc[8], acc[9]);
        float2 previous_259 = _f2_757;
        a_258[0] = mul_f32x2_noftz(previous_259, corr_246);
        float2 _f2_758 = make_float2(acc[10], acc[11]);
        float2 previous_260 = _f2_758;
        a_258[1] = mul_f32x2_noftz(previous_260, corr_246);
        float2 _f2_759 = make_float2(acc[12], acc[13]);
        float2 previous_261 = _f2_759;
        a_258[2] = mul_f32x2_noftz(previous_261, corr_246);
        float2 _f2_760 = make_float2(acc[14], acc[15]);
        float2 previous_262 = _f2_760;
        a_258[3] = mul_f32x2_noftz(previous_262, corr_246);
        float2 _f2_761 = make_float2(weights_244[6], weights_244[6]);
        float2 weight_263 = _f2_761;
        {
            unsigned int sw2_14[4];
            sw2_14[0] = words[84 + woff_256];
            sw2_14[1] = words[84 + woff_256 + 1];
            sw2_14[2] = words[84 + woff_256 + 2];
            sw2_14[3] = words[84 + woff_256 + 3];
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
            float2 _f2_766 = make_float2(sw2_f32_14[0], sw2_f32_14[1]);
            float2 v_60 = _f2_766;
            a_258[0] = fma_f32x2_rn_noftz(weight_263, v_60, a_258[0]);
            float2 _f2_767 = make_float2(sw2_f32_14[2], sw2_f32_14[3]);
            float2 v_0_58 = _f2_767;
            a_258[1] = fma_f32x2_rn_noftz(weight_263, v_0_58, a_258[1]);
            float2 _f2_768 = make_float2(sw2_f32_14[4], sw2_f32_14[5]);
            float2 v_1_29 = _f2_768;
            a_258[2] = fma_f32x2_rn_noftz(weight_263, v_1_29, a_258[2]);
            float2 _f2_769 = make_float2(sw2_f32_14[6], sw2_f32_14[7]);
            float2 v_2_52 = _f2_769;
            a_258[3] = fma_f32x2_rn_noftz(weight_263, v_2_52, a_258[3]);
        }
        float2 _f2_770 = make_float2(weights_244[7], weights_244[7]);
        float2 weight_264 = _f2_770;
        {
            unsigned int sw2_15[4];
            sw2_15[0] = words[98 + woff_256];
            sw2_15[1] = words[98 + woff_256 + 1];
            sw2_15[2] = words[98 + woff_256 + 2];
            sw2_15[3] = words[98 + woff_256 + 3];
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
            float2 _f2_775 = make_float2(sw2_f32_15[0], sw2_f32_15[1]);
            float2 v_61 = _f2_775;
            a_258[0] = fma_f32x2_rn_noftz(weight_264, v_61, a_258[0]);
            float2 _f2_776 = make_float2(sw2_f32_15[2], sw2_f32_15[3]);
            float2 v_0_59 = _f2_776;
            a_258[1] = fma_f32x2_rn_noftz(weight_264, v_0_59, a_258[1]);
            float2 _f2_777 = make_float2(sw2_f32_15[4], sw2_f32_15[5]);
            float2 v_1_30 = _f2_777;
            a_258[2] = fma_f32x2_rn_noftz(weight_264, v_1_30, a_258[2]);
            float2 _f2_778 = make_float2(sw2_f32_15[6], sw2_f32_15[7]);
            float2 v_2_53 = _f2_778;
            a_258[3] = fma_f32x2_rn_noftz(weight_264, v_2_53, a_258[3]);
        }
        acc[8] = a_258[0].x;
        acc[9] = a_258[0].y;
        acc[10] = a_258[1].x;
        acc[11] = a_258[1].y;
        acc[12] = a_258[2].x;
        acc[13] = a_258[2].y;
        acc[14] = a_258[3].x;
        acc[15] = a_258[3].y;
        const int woff_265 = 8;
        int base_266 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float2 a_267[4];
        float2 _f2_779 = make_float2(acc[16], acc[17]);
        float2 previous_268 = _f2_779;
        a_267[0] = mul_f32x2_noftz(previous_268, corr_246);
        float2 _f2_780 = make_float2(acc[18], acc[19]);
        float2 previous_269 = _f2_780;
        a_267[1] = mul_f32x2_noftz(previous_269, corr_246);
        float2 _f2_781 = make_float2(acc[20], acc[21]);
        float2 previous_270 = _f2_781;
        a_267[2] = mul_f32x2_noftz(previous_270, corr_246);
        float2 _f2_782 = make_float2(acc[22], acc[23]);
        float2 previous_271 = _f2_782;
        a_267[3] = mul_f32x2_noftz(previous_271, corr_246);
        float2 _f2_783 = make_float2(weights_244[6], weights_244[6]);
        float2 weight_272 = _f2_783;
        {
            unsigned int sw2_16[4];
            sw2_16[0] = words[84 + woff_265];
            sw2_16[1] = words[84 + woff_265 + 1];
            sw2_16[2] = words[84 + woff_265 + 2];
            sw2_16[3] = words[84 + woff_265 + 3];
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
            float2 _f2_788 = make_float2(sw2_f32_16[0], sw2_f32_16[1]);
            float2 v_62 = _f2_788;
            a_267[0] = fma_f32x2_rn_noftz(weight_272, v_62, a_267[0]);
            float2 _f2_789 = make_float2(sw2_f32_16[2], sw2_f32_16[3]);
            float2 v_0_60 = _f2_789;
            a_267[1] = fma_f32x2_rn_noftz(weight_272, v_0_60, a_267[1]);
            float2 _f2_790 = make_float2(sw2_f32_16[4], sw2_f32_16[5]);
            float2 v_1_31 = _f2_790;
            a_267[2] = fma_f32x2_rn_noftz(weight_272, v_1_31, a_267[2]);
            float2 _f2_791 = make_float2(sw2_f32_16[6], sw2_f32_16[7]);
            float2 v_2_54 = _f2_791;
            a_267[3] = fma_f32x2_rn_noftz(weight_272, v_2_54, a_267[3]);
        }
        float2 _f2_792 = make_float2(weights_244[7], weights_244[7]);
        float2 weight_273 = _f2_792;
        {
            unsigned int sw2_17[4];
            sw2_17[0] = words[98 + woff_265];
            sw2_17[1] = words[98 + woff_265 + 1];
            sw2_17[2] = words[98 + woff_265 + 2];
            sw2_17[3] = words[98 + woff_265 + 3];
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
            float2 _f2_797 = make_float2(sw2_f32_17[0], sw2_f32_17[1]);
            float2 v_63 = _f2_797;
            a_267[0] = fma_f32x2_rn_noftz(weight_273, v_63, a_267[0]);
            float2 _f2_798 = make_float2(sw2_f32_17[2], sw2_f32_17[3]);
            float2 v_0_61 = _f2_798;
            a_267[1] = fma_f32x2_rn_noftz(weight_273, v_0_61, a_267[1]);
            float2 _f2_799 = make_float2(sw2_f32_17[4], sw2_f32_17[5]);
            float2 v_1_32 = _f2_799;
            a_267[2] = fma_f32x2_rn_noftz(weight_273, v_1_32, a_267[2]);
            float2 _f2_800 = make_float2(sw2_f32_17[6], sw2_f32_17[7]);
            float2 v_2_55 = _f2_800;
            a_267[3] = fma_f32x2_rn_noftz(weight_273, v_2_55, a_267[3]);
        }
        acc[16] = a_267[0].x;
        acc[17] = a_267[0].y;
        acc[18] = a_267[1].x;
        acc[19] = a_267[1].y;
        acc[20] = a_267[2].x;
        acc[21] = a_267[2].y;
        acc[22] = a_267[3].x;
        acc[23] = a_267[3].y;
        const int woff_274 = 12;
        int base_275 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float2 a_276[4];
        float2 _f2_801 = make_float2(acc[24], acc[25]);
        float2 previous_277 = _f2_801;
        a_276[0] = mul_f32x2_noftz(previous_277, corr_246);
        float2 _f2_802 = make_float2(acc[26], acc[27]);
        float2 previous_278 = _f2_802;
        a_276[1] = mul_f32x2_noftz(previous_278, corr_246);
        float2 _f2_803 = make_float2(weights_244[6], weights_244[6]);
        float2 weight_279 = _f2_803;
        {
            unsigned int sw2_18[4];
            sw2_18[0] = words[84 + woff_274];
            sw2_18[1] = words[84 + woff_274 + 1];
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
            float2 _f2_806 = make_float2(sw2_f32_18[0], sw2_f32_18[1]);
            float2 v_64 = _f2_806;
            a_276[0] = fma_f32x2_rn_noftz(weight_279, v_64, a_276[0]);
            float2 _f2_807 = make_float2(sw2_f32_18[2], sw2_f32_18[3]);
            float2 v_0_62 = _f2_807;
            a_276[1] = fma_f32x2_rn_noftz(weight_279, v_0_62, a_276[1]);
        }
        float2 _f2_808 = make_float2(weights_244[7], weights_244[7]);
        float2 weight_280 = _f2_808;
        {
            unsigned int sw2_19[4];
            sw2_19[0] = words[98 + woff_274];
            sw2_19[1] = words[98 + woff_274 + 1];
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
            float2 _f2_811 = make_float2(sw2_f32_19[0], sw2_f32_19[1]);
            float2 v_65 = _f2_811;
            a_276[0] = fma_f32x2_rn_noftz(weight_280, v_65, a_276[0]);
            float2 _f2_812 = make_float2(sw2_f32_19[2], sw2_f32_19[3]);
            float2 v_0_63 = _f2_812;
            a_276[1] = fma_f32x2_rn_noftz(weight_280, v_0_63, a_276[1]);
        }
        acc[24] = a_276[0].x;
        acc[25] = a_276[0].y;
        acc[26] = a_276[1].x;
        acc[27] = a_276[1].y;
        sum_running = sum_running * correction_243 + sum_weights_245;
        max_running = max_new_242;
        float2 _f2_813 = make_float2(0.0f, 0.0f);
        float2 output_sq_pair = _f2_813;
        float2 _f2_814 = make_float2(acc[0], acc[1]);
        float2 v_66 = _f2_814;
        output_sq_pair = fma_f32x2_rn_noftz(v_66, v_66, output_sq_pair);
        float2 _f2_815 = make_float2(acc[2], acc[3]);
        float2 v_281 = _f2_815;
        output_sq_pair = fma_f32x2_rn_noftz(v_281, v_281, output_sq_pair);
        float2 _f2_816 = make_float2(acc[4], acc[5]);
        float2 v_282 = _f2_816;
        output_sq_pair = fma_f32x2_rn_noftz(v_282, v_282, output_sq_pair);
        float2 _f2_817 = make_float2(acc[6], acc[7]);
        float2 v_283 = _f2_817;
        output_sq_pair = fma_f32x2_rn_noftz(v_283, v_283, output_sq_pair);
        float2 _f2_818 = make_float2(acc[8], acc[9]);
        float2 v_284 = _f2_818;
        output_sq_pair = fma_f32x2_rn_noftz(v_284, v_284, output_sq_pair);
        float2 _f2_819 = make_float2(acc[10], acc[11]);
        float2 v_285 = _f2_819;
        output_sq_pair = fma_f32x2_rn_noftz(v_285, v_285, output_sq_pair);
        float2 _f2_820 = make_float2(acc[12], acc[13]);
        float2 v_286 = _f2_820;
        output_sq_pair = fma_f32x2_rn_noftz(v_286, v_286, output_sq_pair);
        float2 _f2_821 = make_float2(acc[14], acc[15]);
        float2 v_287 = _f2_821;
        output_sq_pair = fma_f32x2_rn_noftz(v_287, v_287, output_sq_pair);
        float2 _f2_822 = make_float2(acc[16], acc[17]);
        float2 v_288 = _f2_822;
        output_sq_pair = fma_f32x2_rn_noftz(v_288, v_288, output_sq_pair);
        float2 _f2_823 = make_float2(acc[18], acc[19]);
        float2 v_289 = _f2_823;
        output_sq_pair = fma_f32x2_rn_noftz(v_289, v_289, output_sq_pair);
        float2 _f2_824 = make_float2(acc[20], acc[21]);
        float2 v_290 = _f2_824;
        output_sq_pair = fma_f32x2_rn_noftz(v_290, v_290, output_sq_pair);
        float2 _f2_825 = make_float2(acc[22], acc[23]);
        float2 v_291 = _f2_825;
        output_sq_pair = fma_f32x2_rn_noftz(v_291, v_291, output_sq_pair);
        float2 _f2_826 = make_float2(acc[24], acc[25]);
        float2 v_292 = _f2_826;
        output_sq_pair = fma_f32x2_rn_noftz(v_292, v_292, output_sq_pair);
        float2 _f2_827 = make_float2(acc[26], acc[27]);
        float2 v_293 = _f2_827;
        output_sq_pair = fma_f32x2_rn_noftz(v_293, v_293, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_40;
        float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_41;
        float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_42;
        float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_43;
        float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_44;
        if (lane == 0) {
            uint32_t _mapa_32;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_32) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_32), "f"(output_sq) : "memory");
            uint32_t _mapa_33;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_33) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_33), "f"(output_sq) : "memory");
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
        float _shfl_8 = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float rsigma = _shfl_8;
        float2 _f2_828 = make_float2(rsigma, rsigma);
        float2 rsigma_pair = _f2_828;
        int base_294 = ((0) ? 6144 + group * 512 + thread * 4 : group * 1024 + thread * 8);
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f2_829 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair = mul_f32x2_noftz(_f2_829, rsigma_pair);
        float2 _f2_830 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair = mul_f32x2_noftz(scaled_pair, _f2_830);
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_295 = 2;
        const int acc_idx_296 = value_idx_295;
        float2 _f2_831 = make_float2(acc[acc_idx_296], acc[acc_idx_296 + 1]);
        float2 scaled_pair_297 = mul_f32x2_noftz(_f2_831, rsigma_pair);
        float2 _f2_832 = make_float2(wout[acc_idx_296], wout[acc_idx_296 + 1]);
        float2 normalized_pair_298 = mul_f32x2_noftz(scaled_pair_297, _f2_832);
        output_values[value_idx_295] = normalized_pair_298.x;
        output_values[value_idx_295 + 1] = normalized_pair_298.y;
        const int value_idx_299 = 4;
        const int acc_idx_300 = value_idx_299;
        float2 _f2_833 = make_float2(acc[acc_idx_300], acc[acc_idx_300 + 1]);
        float2 scaled_pair_301 = mul_f32x2_noftz(_f2_833, rsigma_pair);
        float2 _f2_834 = make_float2(wout[acc_idx_300], wout[acc_idx_300 + 1]);
        float2 normalized_pair_302 = mul_f32x2_noftz(scaled_pair_301, _f2_834);
        output_values[value_idx_299] = normalized_pair_302.x;
        output_values[value_idx_299 + 1] = normalized_pair_302.y;
        const int value_idx_303 = 6;
        const int acc_idx_304 = value_idx_303;
        float2 _f2_835 = make_float2(acc[acc_idx_304], acc[acc_idx_304 + 1]);
        float2 scaled_pair_305 = mul_f32x2_noftz(_f2_835, rsigma_pair);
        float2 _f2_836 = make_float2(wout[acc_idx_304], wout[acc_idx_304 + 1]);
        float2 normalized_pair_306 = mul_f32x2_noftz(scaled_pair_305, _f2_836);
        output_values[value_idx_303] = normalized_pair_306.x;
        output_values[value_idx_303 + 1] = normalized_pair_306.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values[0 + 0], output_values[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values[0 + 2], output_values[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values[0 + 4], output_values[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values[0 + 6], output_values[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_294 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_307 = ((0) ? 6144 + group * 512 + thread * 4 : (2 + group) * 1024 + thread * 8);
        float output_values_308[8];
        const int value_idx_309 = 0;
        const int acc_idx_310 = 8 + value_idx_309;
        float2 _f2_837 = make_float2(acc[acc_idx_310], acc[acc_idx_310 + 1]);
        float2 scaled_pair_311 = mul_f32x2_noftz(_f2_837, rsigma_pair);
        float2 _f2_838 = make_float2(wout[acc_idx_310], wout[acc_idx_310 + 1]);
        float2 normalized_pair_312 = mul_f32x2_noftz(scaled_pair_311, _f2_838);
        output_values_308[value_idx_309] = normalized_pair_312.x;
        output_values_308[value_idx_309 + 1] = normalized_pair_312.y;
        const int value_idx_313 = 2;
        const int acc_idx_314 = 8 + value_idx_313;
        float2 _f2_839 = make_float2(acc[acc_idx_314], acc[acc_idx_314 + 1]);
        float2 scaled_pair_315 = mul_f32x2_noftz(_f2_839, rsigma_pair);
        float2 _f2_840 = make_float2(wout[acc_idx_314], wout[acc_idx_314 + 1]);
        float2 normalized_pair_316 = mul_f32x2_noftz(scaled_pair_315, _f2_840);
        output_values_308[value_idx_313] = normalized_pair_316.x;
        output_values_308[value_idx_313 + 1] = normalized_pair_316.y;
        const int value_idx_317 = 4;
        const int acc_idx_318 = 8 + value_idx_317;
        float2 _f2_841 = make_float2(acc[acc_idx_318], acc[acc_idx_318 + 1]);
        float2 scaled_pair_319 = mul_f32x2_noftz(_f2_841, rsigma_pair);
        float2 _f2_842 = make_float2(wout[acc_idx_318], wout[acc_idx_318 + 1]);
        float2 normalized_pair_320 = mul_f32x2_noftz(scaled_pair_319, _f2_842);
        output_values_308[value_idx_317] = normalized_pair_320.x;
        output_values_308[value_idx_317 + 1] = normalized_pair_320.y;
        const int value_idx_321 = 6;
        const int acc_idx_322 = 8 + value_idx_321;
        float2 _f2_843 = make_float2(acc[acc_idx_322], acc[acc_idx_322 + 1]);
        float2 scaled_pair_323 = mul_f32x2_noftz(_f2_843, rsigma_pair);
        float2 _f2_844 = make_float2(wout[acc_idx_322], wout[acc_idx_322 + 1]);
        float2 normalized_pair_324 = mul_f32x2_noftz(scaled_pair_323, _f2_844);
        output_values_308[value_idx_321] = normalized_pair_324.x;
        output_values_308[value_idx_321 + 1] = normalized_pair_324.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_308[0 + 0], output_values_308[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_308[0 + 2], output_values_308[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_308[0 + 4], output_values_308[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_308[0 + 6], output_values_308[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_307 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_325 = ((0) ? 6144 + group * 512 + thread * 4 : (4 + group) * 1024 + thread * 8);
        float output_values_326[8];
        const int value_idx_327 = 0;
        const int acc_idx_328 = 16 + value_idx_327;
        float2 _f2_845 = make_float2(acc[acc_idx_328], acc[acc_idx_328 + 1]);
        float2 scaled_pair_329 = mul_f32x2_noftz(_f2_845, rsigma_pair);
        float2 _f2_846 = make_float2(wout[acc_idx_328], wout[acc_idx_328 + 1]);
        float2 normalized_pair_330 = mul_f32x2_noftz(scaled_pair_329, _f2_846);
        output_values_326[value_idx_327] = normalized_pair_330.x;
        output_values_326[value_idx_327 + 1] = normalized_pair_330.y;
        const int value_idx_331 = 2;
        const int acc_idx_332 = 16 + value_idx_331;
        float2 _f2_847 = make_float2(acc[acc_idx_332], acc[acc_idx_332 + 1]);
        float2 scaled_pair_333 = mul_f32x2_noftz(_f2_847, rsigma_pair);
        float2 _f2_848 = make_float2(wout[acc_idx_332], wout[acc_idx_332 + 1]);
        float2 normalized_pair_334 = mul_f32x2_noftz(scaled_pair_333, _f2_848);
        output_values_326[value_idx_331] = normalized_pair_334.x;
        output_values_326[value_idx_331 + 1] = normalized_pair_334.y;
        const int value_idx_335 = 4;
        const int acc_idx_336 = 16 + value_idx_335;
        float2 _f2_849 = make_float2(acc[acc_idx_336], acc[acc_idx_336 + 1]);
        float2 scaled_pair_337 = mul_f32x2_noftz(_f2_849, rsigma_pair);
        float2 _f2_850 = make_float2(wout[acc_idx_336], wout[acc_idx_336 + 1]);
        float2 normalized_pair_338 = mul_f32x2_noftz(scaled_pair_337, _f2_850);
        output_values_326[value_idx_335] = normalized_pair_338.x;
        output_values_326[value_idx_335 + 1] = normalized_pair_338.y;
        const int value_idx_339 = 6;
        const int acc_idx_340 = 16 + value_idx_339;
        float2 _f2_851 = make_float2(acc[acc_idx_340], acc[acc_idx_340 + 1]);
        float2 scaled_pair_341 = mul_f32x2_noftz(_f2_851, rsigma_pair);
        float2 _f2_852 = make_float2(wout[acc_idx_340], wout[acc_idx_340 + 1]);
        float2 normalized_pair_342 = mul_f32x2_noftz(scaled_pair_341, _f2_852);
        output_values_326[value_idx_339] = normalized_pair_342.x;
        output_values_326[value_idx_339 + 1] = normalized_pair_342.y;
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(output_values_326[0 + 0], output_values_326[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_326[0 + 2], output_values_326[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(output_values_326[0 + 4], output_values_326[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(output_values_326[0 + 6], output_values_326[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_325 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        int base_343 = ((1) ? 6144 + group * 512 + thread * 4 : (6 + group) * 1024 + thread * 8);
        float output_values_344[8];
        const int value_idx_345 = 0;
        const int acc_idx_346 = 24 + value_idx_345;
        float2 _f2_853 = make_float2(acc[acc_idx_346], acc[acc_idx_346 + 1]);
        float2 scaled_pair_347 = mul_f32x2_noftz(_f2_853, rsigma_pair);
        float2 _f2_854 = make_float2(wout[acc_idx_346], wout[acc_idx_346 + 1]);
        float2 normalized_pair_348 = mul_f32x2_noftz(scaled_pair_347, _f2_854);
        output_values_344[value_idx_345] = normalized_pair_348.x;
        output_values_344[value_idx_345 + 1] = normalized_pair_348.y;
        const int value_idx_349 = 2;
        const int acc_idx_350 = 24 + value_idx_349;
        float2 _f2_855 = make_float2(acc[acc_idx_350], acc[acc_idx_350 + 1]);
        float2 scaled_pair_351 = mul_f32x2_noftz(_f2_855, rsigma_pair);
        float2 _f2_856 = make_float2(wout[acc_idx_350], wout[acc_idx_350 + 1]);
        float2 normalized_pair_352 = mul_f32x2_noftz(scaled_pair_351, _f2_856);
        output_values_344[value_idx_349] = normalized_pair_352.x;
        output_values_344[value_idx_349 + 1] = normalized_pair_352.y;
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(output_values_344[0 + 0], output_values_344[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(output_values_344[0 + 2], output_values_344[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(out))[row_base + (unsigned long long)base_343]) = _pk2;
        }
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
