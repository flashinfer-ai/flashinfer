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
#define SMEM_STATS_STAGE_BYTES 448
#define SMEM_STATS_STRIDE 448
#define SMEM_OUT_STATS_OFF 448
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
kernel_cake_kimi_k3_attn_res_80bceeb850a2f55bbd9d(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 448);
    const int out_stats_addr = smem + 448;

    // === Task calls (declaration order) ===
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
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
        unsigned int words[98];
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
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_7[0 + 0] = _uv4_7.x;
            _vec_load_7[0 + 1] = _uv4_7.y;
            _vec_load_7[0 + 2] = _uv4_7.z;
            _vec_load_7[0 + 3] = _uv4_7.w;
        }
        dwords[woff] = _vec_load_7[0];
        dwords[woff + 1] = _vec_load_7[1];
        dwords[woff + 2] = _vec_load_7[2];
        dwords[woff + 3] = _vec_load_7[3];
        float _vec_load_8[8];
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
                        : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        float _vec_load_9[8];
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
                        : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        q[0] = _vec_load_8[0] * _vec_load_9[0];
        q[1] = _vec_load_8[1] * _vec_load_9[1];
        q[2] = _vec_load_8[2] * _vec_load_9[2];
        q[3] = _vec_load_8[3] * _vec_load_9[3];
        q[4] = _vec_load_8[4] * _vec_load_9[4];
        q[5] = _vec_load_8[5] * _vec_load_9[5];
        q[6] = _vec_load_8[6] * _vec_load_9[6];
        q[7] = _vec_load_8[7] * _vec_load_9[7];
        float _vec_load_10[8];
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
                        : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        wout[0] = _vec_load_10[0];
        wout[1] = _vec_load_10[1];
        wout[2] = _vec_load_10[2];
        wout[3] = _vec_load_10[3];
        wout[4] = _vec_load_10[4];
        wout[5] = _vec_load_10[5];
        wout[6] = _vec_load_10[6];
        wout[7] = _vec_load_10[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_11[4];
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_11[0 + 0] = _uv4_11.x;
            _vec_load_11[0 + 1] = _uv4_11.y;
            _vec_load_11[0 + 2] = _uv4_11.z;
            _vec_load_11[0 + 3] = _uv4_11.w;
        }
        words[woff_1] = _vec_load_11[0];
        words[woff_1 + 1] = _vec_load_11[1];
        words[woff_1 + 2] = _vec_load_11[2];
        words[woff_1 + 3] = _vec_load_11[3];
        unsigned int _vec_load_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_12[0 + 0] = _uv4_12.x;
            _vec_load_12[0 + 1] = _uv4_12.y;
            _vec_load_12[0 + 2] = _uv4_12.z;
            _vec_load_12[0 + 3] = _uv4_12.w;
        }
        words[14 + woff_1] = _vec_load_12[0];
        words[14 + woff_1 + 1] = _vec_load_12[1];
        words[14 + woff_1 + 2] = _vec_load_12[2];
        words[14 + woff_1 + 3] = _vec_load_12[3];
        unsigned int _vec_load_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_13[0 + 0] = _uv4_13.x;
            _vec_load_13[0 + 1] = _uv4_13.y;
            _vec_load_13[0 + 2] = _uv4_13.z;
            _vec_load_13[0 + 3] = _uv4_13.w;
        }
        words[28 + woff_1] = _vec_load_13[0];
        words[28 + woff_1 + 1] = _vec_load_13[1];
        words[28 + woff_1 + 2] = _vec_load_13[2];
        words[28 + woff_1 + 3] = _vec_load_13[3];
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        words[42 + woff_1] = _vec_load_14[0];
        words[42 + woff_1 + 1] = _vec_load_14[1];
        words[42 + woff_1 + 2] = _vec_load_14[2];
        words[42 + woff_1 + 3] = _vec_load_14[3];
        unsigned int _vec_load_15[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_15[0 + 0] = _uv4_15.x;
            _vec_load_15[0 + 1] = _uv4_15.y;
            _vec_load_15[0 + 2] = _uv4_15.z;
            _vec_load_15[0 + 3] = _uv4_15.w;
        }
        words[56 + woff_1] = _vec_load_15[0];
        words[56 + woff_1 + 1] = _vec_load_15[1];
        words[56 + woff_1 + 2] = _vec_load_15[2];
        words[56 + woff_1 + 3] = _vec_load_15[3];
        unsigned int _vec_load_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_16[0 + 0] = _uv4_16.x;
            _vec_load_16[0 + 1] = _uv4_16.y;
            _vec_load_16[0 + 2] = _uv4_16.z;
            _vec_load_16[0 + 3] = _uv4_16.w;
        }
        words[70 + woff_1] = _vec_load_16[0];
        words[70 + woff_1 + 1] = _vec_load_16[1];
        words[70 + woff_1 + 2] = _vec_load_16[2];
        words[70 + woff_1 + 3] = _vec_load_16[3];
        unsigned int _vec_load_17[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_17[0 + 0] = _uv4_17.x;
            _vec_load_17[0 + 1] = _uv4_17.y;
            _vec_load_17[0 + 2] = _uv4_17.z;
            _vec_load_17[0 + 3] = _uv4_17.w;
        }
        words[84 + woff_1] = _vec_load_17[0];
        words[84 + woff_1 + 1] = _vec_load_17[1];
        words[84 + woff_1 + 2] = _vec_load_17[2];
        words[84 + woff_1 + 3] = _vec_load_17[3];
        unsigned int _vec_load_18[4];
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_18[0 + 0] = _uv4_18.x;
            _vec_load_18[0 + 1] = _uv4_18.y;
            _vec_load_18[0 + 2] = _uv4_18.z;
            _vec_load_18[0 + 3] = _uv4_18.w;
        }
        dwords[woff_1] = _vec_load_18[0];
        dwords[woff_1 + 1] = _vec_load_18[1];
        dwords[woff_1 + 2] = _vec_load_18[2];
        dwords[woff_1 + 3] = _vec_load_18[3];
        float _vec_load_19[8];
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
                        : "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        float _vec_load_20[8];
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
                        : "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_20[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_20[_pair]));
                }
            }
        }
        q[8] = _vec_load_19[0] * _vec_load_20[0];
        q[9] = _vec_load_19[1] * _vec_load_20[1];
        q[10] = _vec_load_19[2] * _vec_load_20[2];
        q[11] = _vec_load_19[3] * _vec_load_20[3];
        q[12] = _vec_load_19[4] * _vec_load_20[4];
        q[13] = _vec_load_19[5] * _vec_load_20[5];
        q[14] = _vec_load_19[6] * _vec_load_20[6];
        q[15] = _vec_load_19[7] * _vec_load_20[7];
        float _vec_load_21[8];
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
                        : "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_21[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_21[_pair]));
                }
            }
        }
        wout[8] = _vec_load_21[0];
        wout[9] = _vec_load_21[1];
        wout[10] = _vec_load_21[2];
        wout[11] = _vec_load_21[3];
        wout[12] = _vec_load_21[4];
        wout[13] = _vec_load_21[5];
        wout[14] = _vec_load_21[6];
        wout[15] = _vec_load_21[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_22[4];
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_22[0 + 0] = _uv4_22.x;
            _vec_load_22[0 + 1] = _uv4_22.y;
            _vec_load_22[0 + 2] = _uv4_22.z;
            _vec_load_22[0 + 3] = _uv4_22.w;
        }
        words[woff_3] = _vec_load_22[0];
        words[woff_3 + 1] = _vec_load_22[1];
        words[woff_3 + 2] = _vec_load_22[2];
        words[woff_3 + 3] = _vec_load_22[3];
        unsigned int _vec_load_23[4];
        {
            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_23[0 + 0] = _uv4_23.x;
            _vec_load_23[0 + 1] = _uv4_23.y;
            _vec_load_23[0 + 2] = _uv4_23.z;
            _vec_load_23[0 + 3] = _uv4_23.w;
        }
        words[14 + woff_3] = _vec_load_23[0];
        words[14 + woff_3 + 1] = _vec_load_23[1];
        words[14 + woff_3 + 2] = _vec_load_23[2];
        words[14 + woff_3 + 3] = _vec_load_23[3];
        unsigned int _vec_load_24[4];
        {
            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_24[0 + 0] = _uv4_24.x;
            _vec_load_24[0 + 1] = _uv4_24.y;
            _vec_load_24[0 + 2] = _uv4_24.z;
            _vec_load_24[0 + 3] = _uv4_24.w;
        }
        words[28 + woff_3] = _vec_load_24[0];
        words[28 + woff_3 + 1] = _vec_load_24[1];
        words[28 + woff_3 + 2] = _vec_load_24[2];
        words[28 + woff_3 + 3] = _vec_load_24[3];
        unsigned int _vec_load_25[4];
        {
            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_25[0 + 0] = _uv4_25.x;
            _vec_load_25[0 + 1] = _uv4_25.y;
            _vec_load_25[0 + 2] = _uv4_25.z;
            _vec_load_25[0 + 3] = _uv4_25.w;
        }
        words[42 + woff_3] = _vec_load_25[0];
        words[42 + woff_3 + 1] = _vec_load_25[1];
        words[42 + woff_3 + 2] = _vec_load_25[2];
        words[42 + woff_3 + 3] = _vec_load_25[3];
        unsigned int _vec_load_26[4];
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_26[0 + 0] = _uv4_26.x;
            _vec_load_26[0 + 1] = _uv4_26.y;
            _vec_load_26[0 + 2] = _uv4_26.z;
            _vec_load_26[0 + 3] = _uv4_26.w;
        }
        words[56 + woff_3] = _vec_load_26[0];
        words[56 + woff_3 + 1] = _vec_load_26[1];
        words[56 + woff_3 + 2] = _vec_load_26[2];
        words[56 + woff_3 + 3] = _vec_load_26[3];
        unsigned int _vec_load_27[4];
        {
            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_27[0 + 0] = _uv4_27.x;
            _vec_load_27[0 + 1] = _uv4_27.y;
            _vec_load_27[0 + 2] = _uv4_27.z;
            _vec_load_27[0 + 3] = _uv4_27.w;
        }
        words[70 + woff_3] = _vec_load_27[0];
        words[70 + woff_3 + 1] = _vec_load_27[1];
        words[70 + woff_3 + 2] = _vec_load_27[2];
        words[70 + woff_3 + 3] = _vec_load_27[3];
        unsigned int _vec_load_28[4];
        {
            uint4 _uv4_28 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_28[0 + 0] = _uv4_28.x;
            _vec_load_28[0 + 1] = _uv4_28.y;
            _vec_load_28[0 + 2] = _uv4_28.z;
            _vec_load_28[0 + 3] = _uv4_28.w;
        }
        words[84 + woff_3] = _vec_load_28[0];
        words[84 + woff_3 + 1] = _vec_load_28[1];
        words[84 + woff_3 + 2] = _vec_load_28[2];
        words[84 + woff_3 + 3] = _vec_load_28[3];
        unsigned int _vec_load_29[4];
        {
            uint4 _uv4_29 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_29[0 + 0] = _uv4_29.x;
            _vec_load_29[0 + 1] = _uv4_29.y;
            _vec_load_29[0 + 2] = _uv4_29.z;
            _vec_load_29[0 + 3] = _uv4_29.w;
        }
        dwords[woff_3] = _vec_load_29[0];
        dwords[woff_3 + 1] = _vec_load_29[1];
        dwords[woff_3 + 2] = _vec_load_29[2];
        dwords[woff_3 + 3] = _vec_load_29[3];
        float _vec_load_30[8];
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
                        : "=f"((&_vec_load_30[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_30[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_30[_pair]));
                }
            }
        }
        float _vec_load_31[8];
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
                        : "=f"((&_vec_load_31[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_31[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_31[_pair]));
                }
            }
        }
        q[16] = _vec_load_30[0] * _vec_load_31[0];
        q[17] = _vec_load_30[1] * _vec_load_31[1];
        q[18] = _vec_load_30[2] * _vec_load_31[2];
        q[19] = _vec_load_30[3] * _vec_load_31[3];
        q[20] = _vec_load_30[4] * _vec_load_31[4];
        q[21] = _vec_load_30[5] * _vec_load_31[5];
        q[22] = _vec_load_30[6] * _vec_load_31[6];
        q[23] = _vec_load_30[7] * _vec_load_31[7];
        float _vec_load_32[8];
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
                        : "=f"((&_vec_load_32[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_32[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_32[_pair]));
                }
            }
        }
        wout[16] = _vec_load_32[0];
        wout[17] = _vec_load_32[1];
        wout[18] = _vec_load_32[2];
        wout[19] = _vec_load_32[3];
        wout[20] = _vec_load_32[4];
        wout[21] = _vec_load_32[5];
        wout[22] = _vec_load_32[6];
        wout[23] = _vec_load_32[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_33[1];
        {
            _vec_load_33[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_33[0];
        unsigned int _vec_load_34[1];
        {
            _vec_load_34[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_34[0];
        unsigned int _vec_load_35[1];
        {
            _vec_load_35[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_35[0];
        unsigned int _vec_load_36[1];
        {
            _vec_load_36[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_36[0];
        unsigned int _vec_load_37[1];
        {
            _vec_load_37[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_37[0];
        unsigned int _vec_load_38[1];
        {
            _vec_load_38[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_38[0];
        unsigned int _vec_load_39[1];
        {
            _vec_load_39[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[42 + woff_5] = _vec_load_39[0];
        unsigned int _vec_load_40[1];
        {
            _vec_load_40[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[42 + woff_5 + 1] = _vec_load_40[0];
        unsigned int _vec_load_41[1];
        {
            _vec_load_41[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[56 + woff_5] = _vec_load_41[0];
        unsigned int _vec_load_42[1];
        {
            _vec_load_42[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[56 + woff_5 + 1] = _vec_load_42[0];
        unsigned int _vec_load_43[1];
        {
            _vec_load_43[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[70 + woff_5] = _vec_load_43[0];
        unsigned int _vec_load_44[1];
        {
            _vec_load_44[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[70 + woff_5 + 1] = _vec_load_44[0];
        unsigned int _vec_load_45[1];
        {
            _vec_load_45[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[84 + woff_5] = _vec_load_45[0];
        unsigned int _vec_load_46[1];
        {
            _vec_load_46[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[84 + woff_5 + 1] = _vec_load_46[0];
        unsigned int _vec_load_47[1];
        {
            _vec_load_47[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_47[0];
        unsigned int _vec_load_48[1];
        {
            _vec_load_48[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_48[0];
        float _vec_load_49[4];
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
                    : "=f"((&_vec_load_49[0 + _pair * 2])[0]), "=f"((&_vec_load_49[0 + _pair * 2])[1])
                    : "r"(_vpairs_33[_pair]));
            }
        }
        float _vec_load_50[4];
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
                    : "=f"((&_vec_load_50[0 + _pair * 2])[0]), "=f"((&_vec_load_50[0 + _pair * 2])[1])
                    : "r"(_vpairs_34[_pair]));
            }
        }
        q[24] = _vec_load_49[0] * _vec_load_50[0];
        q[25] = _vec_load_49[1] * _vec_load_50[1];
        q[26] = _vec_load_49[2] * _vec_load_50[2];
        q[27] = _vec_load_49[3] * _vec_load_50[3];
        float _vec_load_51[4];
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
                    : "=f"((&_vec_load_51[0 + _pair * 2])[0]), "=f"((&_vec_load_51[0 + _pair * 2])[1])
                    : "r"(_vpairs_35[_pair]));
            }
        }
        wout[24] = _vec_load_51[0];
        wout[25] = _vec_load_51[1];
        wout[26] = _vec_load_51[2];
        wout[27] = _vec_load_51[3];
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
        __nv_bfloat162 a = __as_bf16x2(sw_59[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_59[0] = __as_u32(mixed);
        words[84 + woff_7] = sw_59[0];
        __nv_bfloat162 a_60 = __as_bf16x2(sw_59[1]);
        __nv_bfloat162 d_61 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_62 = a_60 + d_61;
        sw_59[1] = __as_u32(mixed_62);
        words[84 + woff_7 + 1] = sw_59[1];
        __nv_bfloat162 a_63 = __as_bf16x2(sw_59[2]);
        __nv_bfloat162 d_64 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_65 = a_63 + d_64;
        sw_59[2] = __as_u32(mixed_65);
        words[84 + woff_7 + 2] = sw_59[2];
        __nv_bfloat162 a_66 = __as_bf16x2(sw_59[3]);
        __nv_bfloat162 d_67 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_68 = a_66 + d_67;
        sw_59[3] = __as_u32(mixed_68);
        words[84 + woff_7 + 3] = sw_59[3];
        {
            int4 _iv4 = make_int4(sw_59[0 + 0], sw_59[0 + 1], sw_59[0 + 2], sw_59[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
        }
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
        float2 v_69 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 qp_70 = make_float2(q[0], q[1]);
        sq[6] = fma_f32x2_rn_noftz(v_69, v_69, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_69, qp_70, dot[6]);
        float2 v_71 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 qp_72 = make_float2(q[2], q[3]);
        sq[6] = fma_f32x2_rn_noftz(v_71, v_71, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_71, qp_72, dot[6]);
        float2 v_73 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 qp_74 = make_float2(q[4], q[5]);
        sq[6] = fma_f32x2_rn_noftz(v_73, v_73, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_73, qp_74, dot[6]);
        float2 v_75 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 qp_76 = make_float2(q[6], q[7]);
        sq[6] = fma_f32x2_rn_noftz(v_75, v_75, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_75, qp_76, dot[6]);
        int base_77 = (2 + group) * 1024 + thread * 8;
        const int woff_78 = 4;
        unsigned int sw_79[4];
        sw_79[0] = words[woff_78];
        sw_79[1] = words[woff_78 + 1];
        sw_79[2] = words[woff_78 + 2];
        sw_79[3] = words[woff_78 + 3];
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
        fsrc[8] = _unpack_bf16_7[0];
        fsrc[9] = _unpack_bf16_7[1];
        fsrc[10] = _unpack_bf16_7[2];
        fsrc[11] = _unpack_bf16_7[3];
        fsrc[12] = _unpack_bf16_7[4];
        fsrc[13] = _unpack_bf16_7[5];
        fsrc[14] = _unpack_bf16_7[6];
        fsrc[15] = _unpack_bf16_7[7];
        float2 v_80 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 qp_81 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_80, v_80, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_80, qp_81, dot[0]);
        float2 v_82 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 qp_83 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_82, v_82, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_82, qp_83, dot[0]);
        float2 v_84 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 qp_85 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_84, v_84, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_84, qp_85, dot[0]);
        float2 v_86 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 qp_87 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_86, v_86, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_86, qp_87, dot[0]);
        unsigned int sw_88[4];
        sw_88[0] = words[14 + woff_78];
        sw_88[1] = words[14 + woff_78 + 1];
        sw_88[2] = words[14 + woff_78 + 2];
        sw_88[3] = words[14 + woff_78 + 3];
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
        fsrc[36] = _unpack_bf16_8[0];
        fsrc[37] = _unpack_bf16_8[1];
        fsrc[38] = _unpack_bf16_8[2];
        fsrc[39] = _unpack_bf16_8[3];
        fsrc[40] = _unpack_bf16_8[4];
        fsrc[41] = _unpack_bf16_8[5];
        fsrc[42] = _unpack_bf16_8[6];
        fsrc[43] = _unpack_bf16_8[7];
        float2 v_89 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 qp_90 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_89, v_89, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_89, qp_90, dot[1]);
        float2 v_91 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 qp_92 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_91, v_91, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_91, qp_92, dot[1]);
        float2 v_93 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 qp_94 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_93, v_93, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_93, qp_94, dot[1]);
        float2 v_95 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 qp_96 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_95, v_95, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_95, qp_96, dot[1]);
        unsigned int sw_97[4];
        sw_97[0] = words[28 + woff_78];
        sw_97[1] = words[28 + woff_78 + 1];
        sw_97[2] = words[28 + woff_78 + 2];
        sw_97[3] = words[28 + woff_78 + 3];
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
        fsrc[64] = _unpack_bf16_9[0];
        fsrc[65] = _unpack_bf16_9[1];
        fsrc[66] = _unpack_bf16_9[2];
        fsrc[67] = _unpack_bf16_9[3];
        fsrc[68] = _unpack_bf16_9[4];
        fsrc[69] = _unpack_bf16_9[5];
        fsrc[70] = _unpack_bf16_9[6];
        fsrc[71] = _unpack_bf16_9[7];
        float2 v_98 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 qp_99 = make_float2(q[8], q[9]);
        sq[2] = fma_f32x2_rn_noftz(v_98, v_98, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_98, qp_99, dot[2]);
        float2 v_100 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 qp_101 = make_float2(q[10], q[11]);
        sq[2] = fma_f32x2_rn_noftz(v_100, v_100, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_100, qp_101, dot[2]);
        float2 v_102 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 qp_103 = make_float2(q[12], q[13]);
        sq[2] = fma_f32x2_rn_noftz(v_102, v_102, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_102, qp_103, dot[2]);
        float2 v_104 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 qp_105 = make_float2(q[14], q[15]);
        sq[2] = fma_f32x2_rn_noftz(v_104, v_104, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_104, qp_105, dot[2]);
        unsigned int sw_106[4];
        sw_106[0] = words[42 + woff_78];
        sw_106[1] = words[42 + woff_78 + 1];
        sw_106[2] = words[42 + woff_78 + 2];
        sw_106[3] = words[42 + woff_78 + 3];
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
        float2 v_107 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 qp_108 = make_float2(q[8], q[9]);
        sq[3] = fma_f32x2_rn_noftz(v_107, v_107, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_107, qp_108, dot[3]);
        float2 v_109 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 qp_110 = make_float2(q[10], q[11]);
        sq[3] = fma_f32x2_rn_noftz(v_109, v_109, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_109, qp_110, dot[3]);
        float2 v_111 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 qp_112 = make_float2(q[12], q[13]);
        sq[3] = fma_f32x2_rn_noftz(v_111, v_111, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_111, qp_112, dot[3]);
        float2 v_113 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 qp_114 = make_float2(q[14], q[15]);
        sq[3] = fma_f32x2_rn_noftz(v_113, v_113, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_113, qp_114, dot[3]);
        unsigned int sw_115[4];
        sw_115[0] = words[56 + woff_78];
        sw_115[1] = words[56 + woff_78 + 1];
        sw_115[2] = words[56 + woff_78 + 2];
        sw_115[3] = words[56 + woff_78 + 3];
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
        sq[4] = fma_f32x2_rn_noftz(v_116, v_116, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_116, qp_117, dot[4]);
        float2 v_118 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 qp_119 = make_float2(q[10], q[11]);
        sq[4] = fma_f32x2_rn_noftz(v_118, v_118, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_118, qp_119, dot[4]);
        float2 v_120 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 qp_121 = make_float2(q[12], q[13]);
        sq[4] = fma_f32x2_rn_noftz(v_120, v_120, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_120, qp_121, dot[4]);
        float2 v_122 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 qp_123 = make_float2(q[14], q[15]);
        sq[4] = fma_f32x2_rn_noftz(v_122, v_122, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_122, qp_123, dot[4]);
        unsigned int sw_124[4];
        sw_124[0] = words[70 + woff_78];
        sw_124[1] = words[70 + woff_78 + 1];
        sw_124[2] = words[70 + woff_78 + 2];
        sw_124[3] = words[70 + woff_78 + 3];
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
        sq[5] = fma_f32x2_rn_noftz(v_125, v_125, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_125, qp_126, dot[5]);
        float2 v_127 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 qp_128 = make_float2(q[10], q[11]);
        sq[5] = fma_f32x2_rn_noftz(v_127, v_127, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_127, qp_128, dot[5]);
        float2 v_129 = make_float2(_unpack_bf16_12[4], _unpack_bf16_12[5]);
        float2 qp_130 = make_float2(q[12], q[13]);
        sq[5] = fma_f32x2_rn_noftz(v_129, v_129, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_129, qp_130, dot[5]);
        float2 v_131 = make_float2(_unpack_bf16_12[6], _unpack_bf16_12[7]);
        float2 qp_132 = make_float2(q[14], q[15]);
        sq[5] = fma_f32x2_rn_noftz(v_131, v_131, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_131, qp_132, dot[5]);
        unsigned int sw_133[4];
        sw_133[0] = words[84 + woff_78];
        sw_133[1] = words[84 + woff_78 + 1];
        sw_133[2] = words[84 + woff_78 + 2];
        sw_133[3] = words[84 + woff_78 + 3];
        __nv_bfloat162 a_134 = __as_bf16x2(sw_133[0]);
        __nv_bfloat162 d_135 = __as_bf16x2(dwords[woff_78]);
        __nv_bfloat162 mixed_136 = a_134 + d_135;
        sw_133[0] = __as_u32(mixed_136);
        words[84 + woff_78] = sw_133[0];
        __nv_bfloat162 a_137 = __as_bf16x2(sw_133[1]);
        __nv_bfloat162 d_138 = __as_bf16x2(dwords[woff_78 + 1]);
        __nv_bfloat162 mixed_139 = a_137 + d_138;
        sw_133[1] = __as_u32(mixed_139);
        words[84 + woff_78 + 1] = sw_133[1];
        __nv_bfloat162 a_140 = __as_bf16x2(sw_133[2]);
        __nv_bfloat162 d_141 = __as_bf16x2(dwords[woff_78 + 2]);
        __nv_bfloat162 mixed_142 = a_140 + d_141;
        sw_133[2] = __as_u32(mixed_142);
        words[84 + woff_78 + 2] = sw_133[2];
        __nv_bfloat162 a_143 = __as_bf16x2(sw_133[3]);
        __nv_bfloat162 d_144 = __as_bf16x2(dwords[woff_78 + 3]);
        __nv_bfloat162 mixed_145 = a_143 + d_144;
        sw_133[3] = __as_u32(mixed_145);
        words[84 + woff_78 + 3] = sw_133[3];
        {
            int4 _iv4 = make_int4(sw_133[0 + 0], sw_133[0 + 1], sw_133[0 + 2], sw_133[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_77)) + 0) = _iv4;
        }
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
        float2 v_146 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 qp_147 = make_float2(q[8], q[9]);
        sq[6] = fma_f32x2_rn_noftz(v_146, v_146, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_146, qp_147, dot[6]);
        float2 v_148 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 qp_149 = make_float2(q[10], q[11]);
        sq[6] = fma_f32x2_rn_noftz(v_148, v_148, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_148, qp_149, dot[6]);
        float2 v_150 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
        float2 qp_151 = make_float2(q[12], q[13]);
        sq[6] = fma_f32x2_rn_noftz(v_150, v_150, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_150, qp_151, dot[6]);
        float2 v_152 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
        float2 qp_153 = make_float2(q[14], q[15]);
        sq[6] = fma_f32x2_rn_noftz(v_152, v_152, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_152, qp_153, dot[6]);
        int base_154 = (4 + group) * 1024 + thread * 8;
        const int woff_155 = 8;
        unsigned int sw_156[4];
        sw_156[0] = words[woff_155];
        sw_156[1] = words[woff_155 + 1];
        sw_156[2] = words[woff_155 + 2];
        sw_156[3] = words[woff_155 + 3];
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                : "r"(sw_156[_pair]));
        }
        fsrc[16] = _unpack_bf16_14[0];
        fsrc[17] = _unpack_bf16_14[1];
        fsrc[18] = _unpack_bf16_14[2];
        fsrc[19] = _unpack_bf16_14[3];
        fsrc[20] = _unpack_bf16_14[4];
        fsrc[21] = _unpack_bf16_14[5];
        fsrc[22] = _unpack_bf16_14[6];
        fsrc[23] = _unpack_bf16_14[7];
        float2 v_157 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 qp_158 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_157, v_157, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_157, qp_158, dot[0]);
        float2 v_159 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 qp_160 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_159, v_159, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_159, qp_160, dot[0]);
        float2 v_161 = make_float2(_unpack_bf16_14[4], _unpack_bf16_14[5]);
        float2 qp_162 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_161, v_161, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_161, qp_162, dot[0]);
        float2 v_163 = make_float2(_unpack_bf16_14[6], _unpack_bf16_14[7]);
        float2 qp_164 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_163, v_163, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_163, qp_164, dot[0]);
        unsigned int sw_165[4];
        sw_165[0] = words[14 + woff_155];
        sw_165[1] = words[14 + woff_155 + 1];
        sw_165[2] = words[14 + woff_155 + 2];
        sw_165[3] = words[14 + woff_155 + 3];
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                : "r"(sw_165[_pair]));
        }
        fsrc[44] = _unpack_bf16_15[0];
        fsrc[45] = _unpack_bf16_15[1];
        fsrc[46] = _unpack_bf16_15[2];
        fsrc[47] = _unpack_bf16_15[3];
        fsrc[48] = _unpack_bf16_15[4];
        fsrc[49] = _unpack_bf16_15[5];
        fsrc[50] = _unpack_bf16_15[6];
        fsrc[51] = _unpack_bf16_15[7];
        float2 v_166 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_167 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_166, v_166, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_166, qp_167, dot[1]);
        float2 v_168 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_169 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_168, v_168, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_168, qp_169, dot[1]);
        float2 v_170 = make_float2(_unpack_bf16_15[4], _unpack_bf16_15[5]);
        float2 qp_171 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_170, v_170, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_170, qp_171, dot[1]);
        float2 v_172 = make_float2(_unpack_bf16_15[6], _unpack_bf16_15[7]);
        float2 qp_173 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_172, v_172, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_172, qp_173, dot[1]);
        unsigned int sw_174[4];
        sw_174[0] = words[28 + woff_155];
        sw_174[1] = words[28 + woff_155 + 1];
        sw_174[2] = words[28 + woff_155 + 2];
        sw_174[3] = words[28 + woff_155 + 3];
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
        fsrc[72] = _unpack_bf16_16[0];
        fsrc[73] = _unpack_bf16_16[1];
        fsrc[74] = _unpack_bf16_16[2];
        fsrc[75] = _unpack_bf16_16[3];
        fsrc[76] = _unpack_bf16_16[4];
        fsrc[77] = _unpack_bf16_16[5];
        fsrc[78] = _unpack_bf16_16[6];
        fsrc[79] = _unpack_bf16_16[7];
        float2 v_175 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_176 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_175, v_175, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_175, qp_176, dot[2]);
        float2 v_177 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_178 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_177, v_177, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_177, qp_178, dot[2]);
        float2 v_179 = make_float2(_unpack_bf16_16[4], _unpack_bf16_16[5]);
        float2 qp_180 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_179, v_179, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_179, qp_180, dot[2]);
        float2 v_181 = make_float2(_unpack_bf16_16[6], _unpack_bf16_16[7]);
        float2 qp_182 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_181, v_181, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_181, qp_182, dot[2]);
        unsigned int sw_183[4];
        sw_183[0] = words[42 + woff_155];
        sw_183[1] = words[42 + woff_155 + 1];
        sw_183[2] = words[42 + woff_155 + 2];
        sw_183[3] = words[42 + woff_155 + 3];
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
        float2 v_184 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_185 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_184, v_184, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_184, qp_185, dot[3]);
        float2 v_186 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_187 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_186, v_186, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_186, qp_187, dot[3]);
        float2 v_188 = make_float2(_unpack_bf16_17[4], _unpack_bf16_17[5]);
        float2 qp_189 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_188, v_188, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_188, qp_189, dot[3]);
        float2 v_190 = make_float2(_unpack_bf16_17[6], _unpack_bf16_17[7]);
        float2 qp_191 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_190, v_190, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_190, qp_191, dot[3]);
        unsigned int sw_192[4];
        sw_192[0] = words[56 + woff_155];
        sw_192[1] = words[56 + woff_155 + 1];
        sw_192[2] = words[56 + woff_155 + 2];
        sw_192[3] = words[56 + woff_155 + 3];
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
        float2 v_193 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_194 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_193, v_193, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_193, qp_194, dot[4]);
        float2 v_195 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_196 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_195, v_195, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_195, qp_196, dot[4]);
        float2 v_197 = make_float2(_unpack_bf16_18[4], _unpack_bf16_18[5]);
        float2 qp_198 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_197, v_197, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_197, qp_198, dot[4]);
        float2 v_199 = make_float2(_unpack_bf16_18[6], _unpack_bf16_18[7]);
        float2 qp_200 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_199, v_199, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_199, qp_200, dot[4]);
        unsigned int sw_201[4];
        sw_201[0] = words[70 + woff_155];
        sw_201[1] = words[70 + woff_155 + 1];
        sw_201[2] = words[70 + woff_155 + 2];
        sw_201[3] = words[70 + woff_155 + 3];
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
        sq[5] = fma_f32x2_rn_noftz(v_202, v_202, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_202, qp_203, dot[5]);
        float2 v_204 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_205 = make_float2(q[18], q[19]);
        sq[5] = fma_f32x2_rn_noftz(v_204, v_204, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_204, qp_205, dot[5]);
        float2 v_206 = make_float2(_unpack_bf16_19[4], _unpack_bf16_19[5]);
        float2 qp_207 = make_float2(q[20], q[21]);
        sq[5] = fma_f32x2_rn_noftz(v_206, v_206, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_206, qp_207, dot[5]);
        float2 v_208 = make_float2(_unpack_bf16_19[6], _unpack_bf16_19[7]);
        float2 qp_209 = make_float2(q[22], q[23]);
        sq[5] = fma_f32x2_rn_noftz(v_208, v_208, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_208, qp_209, dot[5]);
        unsigned int sw_210[4];
        sw_210[0] = words[84 + woff_155];
        sw_210[1] = words[84 + woff_155 + 1];
        sw_210[2] = words[84 + woff_155 + 2];
        sw_210[3] = words[84 + woff_155 + 3];
        __nv_bfloat162 a_211 = __as_bf16x2(sw_210[0]);
        __nv_bfloat162 d_212 = __as_bf16x2(dwords[woff_155]);
        __nv_bfloat162 mixed_213 = a_211 + d_212;
        sw_210[0] = __as_u32(mixed_213);
        words[84 + woff_155] = sw_210[0];
        __nv_bfloat162 a_214 = __as_bf16x2(sw_210[1]);
        __nv_bfloat162 d_215 = __as_bf16x2(dwords[woff_155 + 1]);
        __nv_bfloat162 mixed_216 = a_214 + d_215;
        sw_210[1] = __as_u32(mixed_216);
        words[84 + woff_155 + 1] = sw_210[1];
        __nv_bfloat162 a_217 = __as_bf16x2(sw_210[2]);
        __nv_bfloat162 d_218 = __as_bf16x2(dwords[woff_155 + 2]);
        __nv_bfloat162 mixed_219 = a_217 + d_218;
        sw_210[2] = __as_u32(mixed_219);
        words[84 + woff_155 + 2] = sw_210[2];
        __nv_bfloat162 a_220 = __as_bf16x2(sw_210[3]);
        __nv_bfloat162 d_221 = __as_bf16x2(dwords[woff_155 + 3]);
        __nv_bfloat162 mixed_222 = a_220 + d_221;
        sw_210[3] = __as_u32(mixed_222);
        words[84 + woff_155 + 3] = sw_210[3];
        {
            int4 _iv4 = make_int4(sw_210[0 + 0], sw_210[0 + 1], sw_210[0 + 2], sw_210[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_154)) + 0) = _iv4;
        }
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
        float2 v_223 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        float2 qp_224 = make_float2(q[16], q[17]);
        sq[6] = fma_f32x2_rn_noftz(v_223, v_223, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_223, qp_224, dot[6]);
        float2 v_225 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        float2 qp_226 = make_float2(q[18], q[19]);
        sq[6] = fma_f32x2_rn_noftz(v_225, v_225, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_225, qp_226, dot[6]);
        float2 v_227 = make_float2(_unpack_bf16_20[4], _unpack_bf16_20[5]);
        float2 qp_228 = make_float2(q[20], q[21]);
        sq[6] = fma_f32x2_rn_noftz(v_227, v_227, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_227, qp_228, dot[6]);
        float2 v_229 = make_float2(_unpack_bf16_20[6], _unpack_bf16_20[7]);
        float2 qp_230 = make_float2(q[22], q[23]);
        sq[6] = fma_f32x2_rn_noftz(v_229, v_229, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_229, qp_230, dot[6]);
        int base_231 = 6144 + group * 512 + thread * 4;
        const int woff_232 = 12;
        unsigned int sw_233[4];
        sw_233[0] = words[woff_232];
        sw_233[1] = words[woff_232 + 1];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw_233[_pair]));
        }
        fsrc[24] = _unpack_bf16_21[0];
        fsrc[25] = _unpack_bf16_21[1];
        fsrc[26] = _unpack_bf16_21[2];
        fsrc[27] = _unpack_bf16_21[3];
        float2 v_234 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        sq[0] = fma_f32x2_rn_noftz(v_234, v_234, sq[0]);
        float2 v_235 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        sq[0] = fma_f32x2_rn_noftz(v_235, v_235, sq[0]);
        float2 v_236 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        float2 qp_237 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_236, qp_237, dot[0]);
        float2 v_238 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        float2 qp_239 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_238, qp_239, dot[0]);
        unsigned int sw_240[4];
        sw_240[0] = words[14 + woff_232];
        sw_240[1] = words[14 + woff_232 + 1];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw_240[_pair]));
        }
        fsrc[52] = _unpack_bf16_22[0];
        fsrc[53] = _unpack_bf16_22[1];
        fsrc[54] = _unpack_bf16_22[2];
        fsrc[55] = _unpack_bf16_22[3];
        float2 v_241 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        sq[1] = fma_f32x2_rn_noftz(v_241, v_241, sq[1]);
        float2 v_242 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        sq[1] = fma_f32x2_rn_noftz(v_242, v_242, sq[1]);
        float2 v_243 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        float2 qp_244 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_243, qp_244, dot[1]);
        float2 v_245 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        float2 qp_246 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_245, qp_246, dot[1]);
        unsigned int sw_247[4];
        sw_247[0] = words[28 + woff_232];
        sw_247[1] = words[28 + woff_232 + 1];
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                : "r"(sw_247[_pair]));
        }
        fsrc[80] = _unpack_bf16_23[0];
        fsrc[81] = _unpack_bf16_23[1];
        fsrc[82] = _unpack_bf16_23[2];
        fsrc[83] = _unpack_bf16_23[3];
        float2 v_248 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        sq[2] = fma_f32x2_rn_noftz(v_248, v_248, sq[2]);
        float2 v_249 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        sq[2] = fma_f32x2_rn_noftz(v_249, v_249, sq[2]);
        float2 v_250 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        float2 qp_251 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_250, qp_251, dot[2]);
        float2 v_252 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        float2 qp_253 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_252, qp_253, dot[2]);
        unsigned int sw_254[4];
        sw_254[0] = words[42 + woff_232];
        sw_254[1] = words[42 + woff_232 + 1];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw_254[_pair]));
        }
        float2 v_255 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        sq[3] = fma_f32x2_rn_noftz(v_255, v_255, sq[3]);
        float2 v_256 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        sq[3] = fma_f32x2_rn_noftz(v_256, v_256, sq[3]);
        float2 v_257 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        float2 qp_258 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_257, qp_258, dot[3]);
        float2 v_259 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        float2 qp_260 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_259, qp_260, dot[3]);
        unsigned int sw_261[4];
        sw_261[0] = words[56 + woff_232];
        sw_261[1] = words[56 + woff_232 + 1];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw_261[_pair]));
        }
        float2 v_262 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        sq[4] = fma_f32x2_rn_noftz(v_262, v_262, sq[4]);
        float2 v_263 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        sq[4] = fma_f32x2_rn_noftz(v_263, v_263, sq[4]);
        float2 v_264 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        float2 qp_265 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_264, qp_265, dot[4]);
        float2 v_266 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        float2 qp_267 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_266, qp_267, dot[4]);
        unsigned int sw_268[4];
        sw_268[0] = words[70 + woff_232];
        sw_268[1] = words[70 + woff_232 + 1];
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw_268[_pair]));
        }
        float2 v_269 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        sq[5] = fma_f32x2_rn_noftz(v_269, v_269, sq[5]);
        float2 v_270 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        sq[5] = fma_f32x2_rn_noftz(v_270, v_270, sq[5]);
        float2 v_271 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        float2 qp_272 = make_float2(q[24], q[25]);
        dot[5] = fma_f32x2_rn_noftz(v_271, qp_272, dot[5]);
        float2 v_273 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        float2 qp_274 = make_float2(q[26], q[27]);
        dot[5] = fma_f32x2_rn_noftz(v_273, qp_274, dot[5]);
        unsigned int sw_275[4];
        sw_275[0] = words[84 + woff_232];
        sw_275[1] = words[84 + woff_232 + 1];
        __nv_bfloat162 a_276 = __as_bf16x2(sw_275[0]);
        __nv_bfloat162 d_277 = __as_bf16x2(dwords[woff_232]);
        __nv_bfloat162 mixed_278 = a_276 + d_277;
        sw_275[0] = __as_u32(mixed_278);
        words[84 + woff_232] = sw_275[0];
        __nv_bfloat162 a_279 = __as_bf16x2(sw_275[1]);
        __nv_bfloat162 d_280 = __as_bf16x2(dwords[woff_232 + 1]);
        __nv_bfloat162 mixed_281 = a_279 + d_280;
        sw_275[1] = __as_u32(mixed_281);
        words[84 + woff_232 + 1] = sw_275[1];
        {
            int2 _iv2 = make_int2(sw_275[0 + 0], sw_275[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_231)) + 0) = _iv2;
        }
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw_275[_pair]));
        }
        float2 v_282 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        sq[6] = fma_f32x2_rn_noftz(v_282, v_282, sq[6]);
        float2 v_283 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        sq[6] = fma_f32x2_rn_noftz(v_283, v_283, sq[6]);
        float2 v_284 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        float2 qp_285 = make_float2(q[24], q[25]);
        dot[6] = fma_f32x2_rn_noftz(v_284, qp_285, dot[6]);
        float2 v_286 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        float2 qp_287 = make_float2(q[26], q[27]);
        dot[6] = fma_f32x2_rn_noftz(v_286, qp_287, dot[6]);
        float2 pairs[7];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[2] = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[3] = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[4] = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[5] = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[6] = make_float2(sq[6].x + sq[6].y, dot[6].x + dot[6].y);
        unsigned long long bits = 0;
        float2 _bit_cast_src_36 = pairs[0];
        bits = reinterpret_cast<unsigned long long*>(&_bit_cast_src_36)[0];
        unsigned long long peerbits = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        float2 peer = make_float2(0.0f, 0.0f);
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_288 = 0;
        float2 _bit_cast_src_37 = pairs[1];
        bits_288 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_37)[0];
        unsigned long long peerbits_289 = __shfl_xor_sync(0xFFFFFFFF, bits_288, 16);
        float2 peer_290 = make_float2(0.0f, 0.0f);
        peer_290 = reinterpret_cast<float2*>(&peerbits_289)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_290);
        unsigned long long bits_291 = 0;
        float2 _bit_cast_src_38 = pairs[2];
        bits_291 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_38)[0];
        unsigned long long peerbits_292 = __shfl_xor_sync(0xFFFFFFFF, bits_291, 16);
        float2 peer_293 = make_float2(0.0f, 0.0f);
        peer_293 = reinterpret_cast<float2*>(&peerbits_292)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_293);
        unsigned long long bits_294 = 0;
        float2 _bit_cast_src_39 = pairs[3];
        bits_294 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_39)[0];
        unsigned long long peerbits_295 = __shfl_xor_sync(0xFFFFFFFF, bits_294, 16);
        float2 peer_296 = make_float2(0.0f, 0.0f);
        peer_296 = reinterpret_cast<float2*>(&peerbits_295)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_296);
        unsigned long long bits_297 = 0;
        float2 _bit_cast_src_40 = pairs[4];
        bits_297 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_40)[0];
        unsigned long long peerbits_298 = __shfl_xor_sync(0xFFFFFFFF, bits_297, 16);
        float2 peer_299 = make_float2(0.0f, 0.0f);
        peer_299 = reinterpret_cast<float2*>(&peerbits_298)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_299);
        unsigned long long bits_300 = 0;
        float2 _bit_cast_src_41 = pairs[5];
        bits_300 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_41)[0];
        unsigned long long peerbits_301 = __shfl_xor_sync(0xFFFFFFFF, bits_300, 16);
        float2 peer_302 = make_float2(0.0f, 0.0f);
        peer_302 = reinterpret_cast<float2*>(&peerbits_301)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_302);
        unsigned long long bits_303 = 0;
        float2 _bit_cast_src_42 = pairs[6];
        bits_303 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits_304 = __shfl_xor_sync(0xFFFFFFFF, bits_303, 16);
        float2 peer_305 = make_float2(0.0f, 0.0f);
        peer_305 = reinterpret_cast<float2*>(&peerbits_304)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_305);
        unsigned long long bits_306 = 0;
        float2 _bit_cast_src_43 = pairs[0];
        bits_306 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_307 = __shfl_xor_sync(0xFFFFFFFF, bits_306, 8);
        float2 peer_308 = make_float2(0.0f, 0.0f);
        peer_308 = reinterpret_cast<float2*>(&peerbits_307)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_308);
        unsigned long long bits_309 = 0;
        float2 _bit_cast_src_44 = pairs[1];
        bits_309 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_310 = __shfl_xor_sync(0xFFFFFFFF, bits_309, 8);
        float2 peer_311 = make_float2(0.0f, 0.0f);
        peer_311 = reinterpret_cast<float2*>(&peerbits_310)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_311);
        unsigned long long bits_312 = 0;
        float2 _bit_cast_src_45 = pairs[2];
        bits_312 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_313 = __shfl_xor_sync(0xFFFFFFFF, bits_312, 8);
        float2 peer_314 = make_float2(0.0f, 0.0f);
        peer_314 = reinterpret_cast<float2*>(&peerbits_313)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_314);
        unsigned long long bits_315 = 0;
        float2 _bit_cast_src_46 = pairs[3];
        bits_315 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_316 = __shfl_xor_sync(0xFFFFFFFF, bits_315, 8);
        float2 peer_317 = make_float2(0.0f, 0.0f);
        peer_317 = reinterpret_cast<float2*>(&peerbits_316)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_317);
        unsigned long long bits_318 = 0;
        float2 _bit_cast_src_47 = pairs[4];
        bits_318 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_319 = __shfl_xor_sync(0xFFFFFFFF, bits_318, 8);
        float2 peer_320 = make_float2(0.0f, 0.0f);
        peer_320 = reinterpret_cast<float2*>(&peerbits_319)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_320);
        unsigned long long bits_321 = 0;
        float2 _bit_cast_src_48 = pairs[5];
        bits_321 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_322 = __shfl_xor_sync(0xFFFFFFFF, bits_321, 8);
        float2 peer_323 = make_float2(0.0f, 0.0f);
        peer_323 = reinterpret_cast<float2*>(&peerbits_322)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_323);
        unsigned long long bits_324 = 0;
        float2 _bit_cast_src_49 = pairs[6];
        bits_324 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_325 = __shfl_xor_sync(0xFFFFFFFF, bits_324, 8);
        float2 peer_326 = make_float2(0.0f, 0.0f);
        peer_326 = reinterpret_cast<float2*>(&peerbits_325)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_326);
        unsigned long long bits_327 = 0;
        float2 _bit_cast_src_50 = pairs[0];
        bits_327 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_328 = __shfl_xor_sync(0xFFFFFFFF, bits_327, 4);
        float2 peer_329 = make_float2(0.0f, 0.0f);
        peer_329 = reinterpret_cast<float2*>(&peerbits_328)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_329);
        unsigned long long bits_330 = 0;
        float2 _bit_cast_src_51 = pairs[1];
        bits_330 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_331 = __shfl_xor_sync(0xFFFFFFFF, bits_330, 4);
        float2 peer_332 = make_float2(0.0f, 0.0f);
        peer_332 = reinterpret_cast<float2*>(&peerbits_331)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_332);
        unsigned long long bits_333 = 0;
        float2 _bit_cast_src_52 = pairs[2];
        bits_333 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_334 = __shfl_xor_sync(0xFFFFFFFF, bits_333, 4);
        float2 peer_335 = make_float2(0.0f, 0.0f);
        peer_335 = reinterpret_cast<float2*>(&peerbits_334)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_335);
        unsigned long long bits_336 = 0;
        float2 _bit_cast_src_53 = pairs[3];
        bits_336 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_337 = __shfl_xor_sync(0xFFFFFFFF, bits_336, 4);
        float2 peer_338 = make_float2(0.0f, 0.0f);
        peer_338 = reinterpret_cast<float2*>(&peerbits_337)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_338);
        unsigned long long bits_339 = 0;
        float2 _bit_cast_src_54 = pairs[4];
        bits_339 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_340 = __shfl_xor_sync(0xFFFFFFFF, bits_339, 4);
        float2 peer_341 = make_float2(0.0f, 0.0f);
        peer_341 = reinterpret_cast<float2*>(&peerbits_340)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_341);
        unsigned long long bits_342 = 0;
        float2 _bit_cast_src_55 = pairs[5];
        bits_342 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_55)[0];
        unsigned long long peerbits_343 = __shfl_xor_sync(0xFFFFFFFF, bits_342, 4);
        float2 peer_344 = make_float2(0.0f, 0.0f);
        peer_344 = reinterpret_cast<float2*>(&peerbits_343)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_344);
        unsigned long long bits_345 = 0;
        float2 _bit_cast_src_56 = pairs[6];
        bits_345 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_56)[0];
        unsigned long long peerbits_346 = __shfl_xor_sync(0xFFFFFFFF, bits_345, 4);
        float2 peer_347 = make_float2(0.0f, 0.0f);
        peer_347 = reinterpret_cast<float2*>(&peerbits_346)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_347);
        unsigned long long bits_348 = 0;
        float2 _bit_cast_src_57 = pairs[0];
        bits_348 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_57)[0];
        unsigned long long peerbits_349 = __shfl_xor_sync(0xFFFFFFFF, bits_348, 2);
        float2 peer_350 = make_float2(0.0f, 0.0f);
        peer_350 = reinterpret_cast<float2*>(&peerbits_349)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_350);
        unsigned long long bits_351 = 0;
        float2 _bit_cast_src_58 = pairs[1];
        bits_351 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_58)[0];
        unsigned long long peerbits_352 = __shfl_xor_sync(0xFFFFFFFF, bits_351, 2);
        float2 peer_353 = make_float2(0.0f, 0.0f);
        peer_353 = reinterpret_cast<float2*>(&peerbits_352)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_353);
        unsigned long long bits_354 = 0;
        float2 _bit_cast_src_59 = pairs[2];
        bits_354 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_59)[0];
        unsigned long long peerbits_355 = __shfl_xor_sync(0xFFFFFFFF, bits_354, 2);
        float2 peer_356 = make_float2(0.0f, 0.0f);
        peer_356 = reinterpret_cast<float2*>(&peerbits_355)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_356);
        unsigned long long bits_357 = 0;
        float2 _bit_cast_src_60 = pairs[3];
        bits_357 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_60)[0];
        unsigned long long peerbits_358 = __shfl_xor_sync(0xFFFFFFFF, bits_357, 2);
        float2 peer_359 = make_float2(0.0f, 0.0f);
        peer_359 = reinterpret_cast<float2*>(&peerbits_358)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_359);
        unsigned long long bits_360 = 0;
        float2 _bit_cast_src_61 = pairs[4];
        bits_360 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_61)[0];
        unsigned long long peerbits_361 = __shfl_xor_sync(0xFFFFFFFF, bits_360, 2);
        float2 peer_362 = make_float2(0.0f, 0.0f);
        peer_362 = reinterpret_cast<float2*>(&peerbits_361)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_362);
        unsigned long long bits_363 = 0;
        float2 _bit_cast_src_62 = pairs[5];
        bits_363 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_62)[0];
        unsigned long long peerbits_364 = __shfl_xor_sync(0xFFFFFFFF, bits_363, 2);
        float2 peer_365 = make_float2(0.0f, 0.0f);
        peer_365 = reinterpret_cast<float2*>(&peerbits_364)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_365);
        unsigned long long bits_366 = 0;
        float2 _bit_cast_src_63 = pairs[6];
        bits_366 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_63)[0];
        unsigned long long peerbits_367 = __shfl_xor_sync(0xFFFFFFFF, bits_366, 2);
        float2 peer_368 = make_float2(0.0f, 0.0f);
        peer_368 = reinterpret_cast<float2*>(&peerbits_367)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_368);
        unsigned long long bits_369 = 0;
        float2 _bit_cast_src_64 = pairs[0];
        bits_369 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_64)[0];
        unsigned long long peerbits_370 = __shfl_xor_sync(0xFFFFFFFF, bits_369, 1);
        float2 peer_371 = make_float2(0.0f, 0.0f);
        peer_371 = reinterpret_cast<float2*>(&peerbits_370)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_371);
        unsigned long long bits_372 = 0;
        float2 _bit_cast_src_65 = pairs[1];
        bits_372 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_65)[0];
        unsigned long long peerbits_373 = __shfl_xor_sync(0xFFFFFFFF, bits_372, 1);
        float2 peer_374 = make_float2(0.0f, 0.0f);
        peer_374 = reinterpret_cast<float2*>(&peerbits_373)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_374);
        unsigned long long bits_375 = 0;
        float2 _bit_cast_src_66 = pairs[2];
        bits_375 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_66)[0];
        unsigned long long peerbits_376 = __shfl_xor_sync(0xFFFFFFFF, bits_375, 1);
        float2 peer_377 = make_float2(0.0f, 0.0f);
        peer_377 = reinterpret_cast<float2*>(&peerbits_376)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_377);
        unsigned long long bits_378 = 0;
        float2 _bit_cast_src_67 = pairs[3];
        bits_378 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_67)[0];
        unsigned long long peerbits_379 = __shfl_xor_sync(0xFFFFFFFF, bits_378, 1);
        float2 peer_380 = make_float2(0.0f, 0.0f);
        peer_380 = reinterpret_cast<float2*>(&peerbits_379)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_380);
        unsigned long long bits_381 = 0;
        float2 _bit_cast_src_68 = pairs[4];
        bits_381 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_68)[0];
        unsigned long long peerbits_382 = __shfl_xor_sync(0xFFFFFFFF, bits_381, 1);
        float2 peer_383 = make_float2(0.0f, 0.0f);
        peer_383 = reinterpret_cast<float2*>(&peerbits_382)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_383);
        unsigned long long bits_384 = 0;
        float2 _bit_cast_src_69 = pairs[5];
        bits_384 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_69)[0];
        unsigned long long peerbits_385 = __shfl_xor_sync(0xFFFFFFFF, bits_384, 1);
        float2 peer_386 = make_float2(0.0f, 0.0f);
        peer_386 = reinterpret_cast<float2*>(&peerbits_385)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_386);
        unsigned long long bits_387 = 0;
        float2 _bit_cast_src_70 = pairs[6];
        bits_387 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_70)[0];
        unsigned long long peerbits_388 = __shfl_xor_sync(0xFFFFFFFF, bits_387, 1);
        float2 peer_389 = make_float2(0.0f, 0.0f);
        peer_389 = reinterpret_cast<float2*>(&peerbits_388)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_389);
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (lane == 0) {
            uint32_t _cluster_mapa_71;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_71) : "r"(stats_addr + (unsigned int)(warp_0 * 7 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_71), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_72;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_72) : "r"(stats_addr + (unsigned int)((warp_0 * 7 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_72), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_73;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_73) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_73), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_74;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_74) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_74), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_75;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_75) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_75), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_76;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_76) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_76), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_77;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_77) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_77), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_78;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_78) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_78), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_79;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_79) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_79), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_80;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_80) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_80), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_81;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_81) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_81), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_82;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_82) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_82), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_83;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_83) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 6) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_83), "f"(pairs[6].x) : "memory");
            uint32_t _cluster_mapa_84;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_84) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 6) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_84), "f"(pairs[6].y) : "memory");
            uint32_t _cluster_mapa_85;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_85) : "r"(stats_addr + (unsigned int)(warp_0 * 7 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_85), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_86;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_86) : "r"(stats_addr + (unsigned int)((warp_0 * 7 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_86), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_87;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_87) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_87), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_88;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_88) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_88), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_89;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_89) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_89), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_90;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_90) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_90), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_91;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_91) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_91), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_92;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_92) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_92), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_93;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_93) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_93), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_94;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_94) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_94), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_95;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_95) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_95), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_96;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_96) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_96), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_97;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_97) : "r"(stats_addr + (unsigned int)((warp_0 * 7 + 6) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_97), "f"(pairs[6].x) : "memory");
            uint32_t _cluster_mapa_98;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_98) : "r"(stats_addr + (unsigned int)(((warp_0 * 7 + 6) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_98), "f"(pairs[6].y) : "memory");
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
        float _shfl_down_sync_99 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_99;
        float _shfl_down_sync_100 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_100;
        float _shfl_down_sync_101 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_101;
        float _shfl_down_sync_102 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_102;
        float _shfl_down_sync_103 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_103;
        float _shfl_down_sync_104 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_104;
        float logit = 0.0f;
        if (stat_n < 7 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[7];
        float _shfl_sync_105 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_105;
        float _shfl_sync_106 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_106;
        float _shfl_sync_107 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_107;
        float _shfl_sync_108 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_sync_108;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 3) {
            total_sq2 = stats[(stat_w * 7 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 7 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_109 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_109;
        float _shfl_down_sync_110 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_110;
        float _shfl_down_sync_111 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_111;
        float _shfl_down_sync_112 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_112;
        float _shfl_down_sync_113 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_113;
        float _shfl_down_sync_114 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_114;
        logit2 = 0.0f;
        if (stat_n < 3 && stat_w == 0) {
            float sigma2 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_sync_115 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_sync_115;
        float _shfl_sync_116 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_sync_116;
        float _shfl_sync_117 = __shfl_sync(0xFFFFFFFF, logit2, 16);
        logits[6] = _shfl_sync_117;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028235e+38f;
        max_chunk = fmaxf(max_chunk, logits[0]);
        max_chunk = fmaxf(max_chunk, logits[1]);
        max_chunk = fmaxf(max_chunk, logits[2]);
        max_chunk = fmaxf(max_chunk, logits[3]);
        float max_new = fmaxf(max_running, max_chunk);
        float correction = approx_exp2((max_running - max_new) * 1.442695f);
        float weights[7];
        float sum_weights = 0.0f;
        float _exp2_118 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_118;
        sum_weights += weights[0];
        float _exp2_119 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_119;
        sum_weights += weights[1];
        float _exp2_120 = approx_exp2((logits[2] - max_new) * 1.442695f);
        weights[2] = _exp2_120;
        sum_weights += weights[2];
        float _exp2_121 = approx_exp2((logits[3] - max_new) * 1.442695f);
        weights[3] = _exp2_121;
        sum_weights += weights[3];
        float2 corr = make_float2(correction, correction);
        const int woff_390 = 0;
        int base_391 = group * 1024 + thread * 8;
        float2 a_392[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_392[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_393 = make_float2(acc[2], acc[3]);
        a_392[1] = mul_f32x2_noftz(previous_393, corr);
        float2 previous_394 = make_float2(acc[4], acc[5]);
        a_392[2] = mul_f32x2_noftz(previous_394, corr);
        float2 previous_395 = make_float2(acc[6], acc[7]);
        a_392[3] = mul_f32x2_noftz(previous_395, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_396 = make_float2(fsrc[0], fsrc[1]);
        a_392[0] = fma_f32x2_rn_noftz(weight, v_396, a_392[0]);
        float2 v_397 = make_float2(fsrc[2], fsrc[3]);
        a_392[1] = fma_f32x2_rn_noftz(weight, v_397, a_392[1]);
        float2 v_398 = make_float2(fsrc[4], fsrc[5]);
        a_392[2] = fma_f32x2_rn_noftz(weight, v_398, a_392[2]);
        float2 v_399 = make_float2(fsrc[6], fsrc[7]);
        a_392[3] = fma_f32x2_rn_noftz(weight, v_399, a_392[3]);
        float2 weight_400 = make_float2(weights[1], weights[1]);
        float2 v_401 = make_float2(fsrc[28], fsrc[29]);
        a_392[0] = fma_f32x2_rn_noftz(weight_400, v_401, a_392[0]);
        float2 v_402 = make_float2(fsrc[30], fsrc[31]);
        a_392[1] = fma_f32x2_rn_noftz(weight_400, v_402, a_392[1]);
        float2 v_403 = make_float2(fsrc[32], fsrc[33]);
        a_392[2] = fma_f32x2_rn_noftz(weight_400, v_403, a_392[2]);
        float2 v_404 = make_float2(fsrc[34], fsrc[35]);
        a_392[3] = fma_f32x2_rn_noftz(weight_400, v_404, a_392[3]);
        float2 weight_405 = make_float2(weights[2], weights[2]);
        float2 v_406 = make_float2(fsrc[56], fsrc[57]);
        a_392[0] = fma_f32x2_rn_noftz(weight_405, v_406, a_392[0]);
        float2 v_407 = make_float2(fsrc[58], fsrc[59]);
        a_392[1] = fma_f32x2_rn_noftz(weight_405, v_407, a_392[1]);
        float2 v_408 = make_float2(fsrc[60], fsrc[61]);
        a_392[2] = fma_f32x2_rn_noftz(weight_405, v_408, a_392[2]);
        float2 v_409 = make_float2(fsrc[62], fsrc[63]);
        a_392[3] = fma_f32x2_rn_noftz(weight_405, v_409, a_392[3]);
        float2 weight_410 = make_float2(weights[3], weights[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_390];
        sw2[1] = words[42 + woff_390 + 1];
        sw2[2] = words[42 + woff_390 + 2];
        sw2[3] = words[42 + woff_390 + 3];
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                : "r"(sw2[_pair]));
        }
        float2 v_411 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        a_392[0] = fma_f32x2_rn_noftz(weight_410, v_411, a_392[0]);
        float2 v_412 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        a_392[1] = fma_f32x2_rn_noftz(weight_410, v_412, a_392[1]);
        float2 v_413 = make_float2(_unpack_bf16_28[4], _unpack_bf16_28[5]);
        a_392[2] = fma_f32x2_rn_noftz(weight_410, v_413, a_392[2]);
        float2 v_414 = make_float2(_unpack_bf16_28[6], _unpack_bf16_28[7]);
        a_392[3] = fma_f32x2_rn_noftz(weight_410, v_414, a_392[3]);
        acc[0] = a_392[0].x;
        acc[1] = a_392[0].y;
        acc[2] = a_392[1].x;
        acc[3] = a_392[1].y;
        acc[4] = a_392[2].x;
        acc[5] = a_392[2].y;
        acc[6] = a_392[3].x;
        acc[7] = a_392[3].y;
        const int woff_415 = 4;
        int base_416 = (2 + group) * 1024 + thread * 8;
        float2 a_417[4];
        float2 previous_418 = make_float2(acc[8], acc[9]);
        a_417[0] = mul_f32x2_noftz(previous_418, corr);
        float2 previous_419 = make_float2(acc[10], acc[11]);
        a_417[1] = mul_f32x2_noftz(previous_419, corr);
        float2 previous_420 = make_float2(acc[12], acc[13]);
        a_417[2] = mul_f32x2_noftz(previous_420, corr);
        float2 previous_421 = make_float2(acc[14], acc[15]);
        a_417[3] = mul_f32x2_noftz(previous_421, corr);
        float2 weight_422 = make_float2(weights[0], weights[0]);
        float2 v_423 = make_float2(fsrc[8], fsrc[9]);
        a_417[0] = fma_f32x2_rn_noftz(weight_422, v_423, a_417[0]);
        float2 v_424 = make_float2(fsrc[10], fsrc[11]);
        a_417[1] = fma_f32x2_rn_noftz(weight_422, v_424, a_417[1]);
        float2 v_425 = make_float2(fsrc[12], fsrc[13]);
        a_417[2] = fma_f32x2_rn_noftz(weight_422, v_425, a_417[2]);
        float2 v_426 = make_float2(fsrc[14], fsrc[15]);
        a_417[3] = fma_f32x2_rn_noftz(weight_422, v_426, a_417[3]);
        float2 weight_427 = make_float2(weights[1], weights[1]);
        float2 v_428 = make_float2(fsrc[36], fsrc[37]);
        a_417[0] = fma_f32x2_rn_noftz(weight_427, v_428, a_417[0]);
        float2 v_429 = make_float2(fsrc[38], fsrc[39]);
        a_417[1] = fma_f32x2_rn_noftz(weight_427, v_429, a_417[1]);
        float2 v_430 = make_float2(fsrc[40], fsrc[41]);
        a_417[2] = fma_f32x2_rn_noftz(weight_427, v_430, a_417[2]);
        float2 v_431 = make_float2(fsrc[42], fsrc[43]);
        a_417[3] = fma_f32x2_rn_noftz(weight_427, v_431, a_417[3]);
        float2 weight_432 = make_float2(weights[2], weights[2]);
        float2 v_433 = make_float2(fsrc[64], fsrc[65]);
        a_417[0] = fma_f32x2_rn_noftz(weight_432, v_433, a_417[0]);
        float2 v_434 = make_float2(fsrc[66], fsrc[67]);
        a_417[1] = fma_f32x2_rn_noftz(weight_432, v_434, a_417[1]);
        float2 v_435 = make_float2(fsrc[68], fsrc[69]);
        a_417[2] = fma_f32x2_rn_noftz(weight_432, v_435, a_417[2]);
        float2 v_436 = make_float2(fsrc[70], fsrc[71]);
        a_417[3] = fma_f32x2_rn_noftz(weight_432, v_436, a_417[3]);
        float2 weight_437 = make_float2(weights[3], weights[3]);
        unsigned int sw2_438[4];
        sw2_438[0] = words[42 + woff_415];
        sw2_438[1] = words[42 + woff_415 + 1];
        sw2_438[2] = words[42 + woff_415 + 2];
        sw2_438[3] = words[42 + woff_415 + 3];
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                : "r"(sw2_438[_pair]));
        }
        float2 v_439 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        a_417[0] = fma_f32x2_rn_noftz(weight_437, v_439, a_417[0]);
        float2 v_440 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        a_417[1] = fma_f32x2_rn_noftz(weight_437, v_440, a_417[1]);
        float2 v_441 = make_float2(_unpack_bf16_29[4], _unpack_bf16_29[5]);
        a_417[2] = fma_f32x2_rn_noftz(weight_437, v_441, a_417[2]);
        float2 v_442 = make_float2(_unpack_bf16_29[6], _unpack_bf16_29[7]);
        a_417[3] = fma_f32x2_rn_noftz(weight_437, v_442, a_417[3]);
        acc[8] = a_417[0].x;
        acc[9] = a_417[0].y;
        acc[10] = a_417[1].x;
        acc[11] = a_417[1].y;
        acc[12] = a_417[2].x;
        acc[13] = a_417[2].y;
        acc[14] = a_417[3].x;
        acc[15] = a_417[3].y;
        const int woff_443 = 8;
        int base_444 = (4 + group) * 1024 + thread * 8;
        float2 a_445[4];
        float2 previous_446 = make_float2(acc[16], acc[17]);
        a_445[0] = mul_f32x2_noftz(previous_446, corr);
        float2 previous_447 = make_float2(acc[18], acc[19]);
        a_445[1] = mul_f32x2_noftz(previous_447, corr);
        float2 previous_448 = make_float2(acc[20], acc[21]);
        a_445[2] = mul_f32x2_noftz(previous_448, corr);
        float2 previous_449 = make_float2(acc[22], acc[23]);
        a_445[3] = mul_f32x2_noftz(previous_449, corr);
        float2 weight_450 = make_float2(weights[0], weights[0]);
        float2 v_451 = make_float2(fsrc[16], fsrc[17]);
        a_445[0] = fma_f32x2_rn_noftz(weight_450, v_451, a_445[0]);
        float2 v_452 = make_float2(fsrc[18], fsrc[19]);
        a_445[1] = fma_f32x2_rn_noftz(weight_450, v_452, a_445[1]);
        float2 v_453 = make_float2(fsrc[20], fsrc[21]);
        a_445[2] = fma_f32x2_rn_noftz(weight_450, v_453, a_445[2]);
        float2 v_454 = make_float2(fsrc[22], fsrc[23]);
        a_445[3] = fma_f32x2_rn_noftz(weight_450, v_454, a_445[3]);
        float2 weight_455 = make_float2(weights[1], weights[1]);
        float2 v_456 = make_float2(fsrc[44], fsrc[45]);
        a_445[0] = fma_f32x2_rn_noftz(weight_455, v_456, a_445[0]);
        float2 v_457 = make_float2(fsrc[46], fsrc[47]);
        a_445[1] = fma_f32x2_rn_noftz(weight_455, v_457, a_445[1]);
        float2 v_458 = make_float2(fsrc[48], fsrc[49]);
        a_445[2] = fma_f32x2_rn_noftz(weight_455, v_458, a_445[2]);
        float2 v_459 = make_float2(fsrc[50], fsrc[51]);
        a_445[3] = fma_f32x2_rn_noftz(weight_455, v_459, a_445[3]);
        float2 weight_460 = make_float2(weights[2], weights[2]);
        float2 v_461 = make_float2(fsrc[72], fsrc[73]);
        a_445[0] = fma_f32x2_rn_noftz(weight_460, v_461, a_445[0]);
        float2 v_462 = make_float2(fsrc[74], fsrc[75]);
        a_445[1] = fma_f32x2_rn_noftz(weight_460, v_462, a_445[1]);
        float2 v_463 = make_float2(fsrc[76], fsrc[77]);
        a_445[2] = fma_f32x2_rn_noftz(weight_460, v_463, a_445[2]);
        float2 v_464 = make_float2(fsrc[78], fsrc[79]);
        a_445[3] = fma_f32x2_rn_noftz(weight_460, v_464, a_445[3]);
        float2 weight_465 = make_float2(weights[3], weights[3]);
        unsigned int sw2_466[4];
        sw2_466[0] = words[42 + woff_443];
        sw2_466[1] = words[42 + woff_443 + 1];
        sw2_466[2] = words[42 + woff_443 + 2];
        sw2_466[3] = words[42 + woff_443 + 3];
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                : "r"(sw2_466[_pair]));
        }
        float2 v_467 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        a_445[0] = fma_f32x2_rn_noftz(weight_465, v_467, a_445[0]);
        float2 v_468 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        a_445[1] = fma_f32x2_rn_noftz(weight_465, v_468, a_445[1]);
        float2 v_469 = make_float2(_unpack_bf16_30[4], _unpack_bf16_30[5]);
        a_445[2] = fma_f32x2_rn_noftz(weight_465, v_469, a_445[2]);
        float2 v_470 = make_float2(_unpack_bf16_30[6], _unpack_bf16_30[7]);
        a_445[3] = fma_f32x2_rn_noftz(weight_465, v_470, a_445[3]);
        acc[16] = a_445[0].x;
        acc[17] = a_445[0].y;
        acc[18] = a_445[1].x;
        acc[19] = a_445[1].y;
        acc[20] = a_445[2].x;
        acc[21] = a_445[2].y;
        acc[22] = a_445[3].x;
        acc[23] = a_445[3].y;
        const int woff_471 = 12;
        int base_472 = 6144 + group * 512 + thread * 4;
        float2 a_473[4];
        float2 previous_474 = make_float2(acc[24], acc[25]);
        a_473[0] = mul_f32x2_noftz(previous_474, corr);
        float2 previous_475 = make_float2(acc[26], acc[27]);
        a_473[1] = mul_f32x2_noftz(previous_475, corr);
        float2 weight_476 = make_float2(weights[0], weights[0]);
        float2 v_477 = make_float2(fsrc[24], fsrc[25]);
        a_473[0] = fma_f32x2_rn_noftz(weight_476, v_477, a_473[0]);
        float2 v_478 = make_float2(fsrc[26], fsrc[27]);
        a_473[1] = fma_f32x2_rn_noftz(weight_476, v_478, a_473[1]);
        float2 weight_479 = make_float2(weights[1], weights[1]);
        float2 v_480 = make_float2(fsrc[52], fsrc[53]);
        a_473[0] = fma_f32x2_rn_noftz(weight_479, v_480, a_473[0]);
        float2 v_481 = make_float2(fsrc[54], fsrc[55]);
        a_473[1] = fma_f32x2_rn_noftz(weight_479, v_481, a_473[1]);
        float2 weight_482 = make_float2(weights[2], weights[2]);
        float2 v_483 = make_float2(fsrc[80], fsrc[81]);
        a_473[0] = fma_f32x2_rn_noftz(weight_482, v_483, a_473[0]);
        float2 v_484 = make_float2(fsrc[82], fsrc[83]);
        a_473[1] = fma_f32x2_rn_noftz(weight_482, v_484, a_473[1]);
        float2 weight_485 = make_float2(weights[3], weights[3]);
        unsigned int sw2_486[4];
        sw2_486[0] = words[42 + woff_471];
        sw2_486[1] = words[42 + woff_471 + 1];
        float _unpack_bf16_31[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                : "r"(sw2_486[_pair]));
        }
        float2 v_487 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        a_473[0] = fma_f32x2_rn_noftz(weight_485, v_487, a_473[0]);
        float2 v_488 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        a_473[1] = fma_f32x2_rn_noftz(weight_485, v_488, a_473[1]);
        acc[24] = a_473[0].x;
        acc[25] = a_473[0].y;
        acc[26] = a_473[1].x;
        acc[27] = a_473[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_489 = -3.4028235e+38f;
        max_chunk_489 = fmaxf(max_chunk_489, logits[4]);
        max_chunk_489 = fmaxf(max_chunk_489, logits[5]);
        max_chunk_489 = fmaxf(max_chunk_489, logits[6]);
        float max_new_490 = fmaxf(max_running, max_chunk_489);
        float correction_491 = approx_exp2((max_running - max_new_490) * 1.442695f);
        float weights_492[7];
        float sum_weights_493 = 0.0f;
        float _exp2_122 = approx_exp2((logits[4] - max_new_490) * 1.442695f);
        weights_492[4] = _exp2_122;
        sum_weights_493 += weights_492[4];
        float _exp2_123 = approx_exp2((logits[5] - max_new_490) * 1.442695f);
        weights_492[5] = _exp2_123;
        sum_weights_493 += weights_492[5];
        float _exp2_124 = approx_exp2((logits[6] - max_new_490) * 1.442695f);
        weights_492[6] = _exp2_124;
        sum_weights_493 += weights_492[6];
        float2 corr_494 = make_float2(correction_491, correction_491);
        const int woff_495 = 0;
        int base_496 = group * 1024 + thread * 8;
        float2 a_497[4];
        float2 previous_498 = make_float2(acc[0], acc[1]);
        a_497[0] = mul_f32x2_noftz(previous_498, corr_494);
        float2 previous_499 = make_float2(acc[2], acc[3]);
        a_497[1] = mul_f32x2_noftz(previous_499, corr_494);
        float2 previous_500 = make_float2(acc[4], acc[5]);
        a_497[2] = mul_f32x2_noftz(previous_500, corr_494);
        float2 previous_501 = make_float2(acc[6], acc[7]);
        a_497[3] = mul_f32x2_noftz(previous_501, corr_494);
        float2 weight_502 = make_float2(weights_492[4], weights_492[4]);
        unsigned int sw2_503[4];
        sw2_503[0] = words[56 + woff_495];
        sw2_503[1] = words[56 + woff_495 + 1];
        sw2_503[2] = words[56 + woff_495 + 2];
        sw2_503[3] = words[56 + woff_495 + 3];
        float _unpack_bf16_32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                : "r"(sw2_503[_pair]));
        }
        float2 v_504 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
        a_497[0] = fma_f32x2_rn_noftz(weight_502, v_504, a_497[0]);
        float2 v_505 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
        a_497[1] = fma_f32x2_rn_noftz(weight_502, v_505, a_497[1]);
        float2 v_506 = make_float2(_unpack_bf16_32[4], _unpack_bf16_32[5]);
        a_497[2] = fma_f32x2_rn_noftz(weight_502, v_506, a_497[2]);
        float2 v_507 = make_float2(_unpack_bf16_32[6], _unpack_bf16_32[7]);
        a_497[3] = fma_f32x2_rn_noftz(weight_502, v_507, a_497[3]);
        float2 weight_508 = make_float2(weights_492[5], weights_492[5]);
        unsigned int sw2_509[4];
        sw2_509[0] = words[70 + woff_495];
        sw2_509[1] = words[70 + woff_495 + 1];
        sw2_509[2] = words[70 + woff_495 + 2];
        sw2_509[3] = words[70 + woff_495 + 3];
        float _unpack_bf16_33[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                : "r"(sw2_509[_pair]));
        }
        float2 v_510 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
        a_497[0] = fma_f32x2_rn_noftz(weight_508, v_510, a_497[0]);
        float2 v_511 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
        a_497[1] = fma_f32x2_rn_noftz(weight_508, v_511, a_497[1]);
        float2 v_512 = make_float2(_unpack_bf16_33[4], _unpack_bf16_33[5]);
        a_497[2] = fma_f32x2_rn_noftz(weight_508, v_512, a_497[2]);
        float2 v_513 = make_float2(_unpack_bf16_33[6], _unpack_bf16_33[7]);
        a_497[3] = fma_f32x2_rn_noftz(weight_508, v_513, a_497[3]);
        float2 weight_514 = make_float2(weights_492[6], weights_492[6]);
        unsigned int sw2_515[4];
        sw2_515[0] = words[84 + woff_495];
        sw2_515[1] = words[84 + woff_495 + 1];
        sw2_515[2] = words[84 + woff_495 + 2];
        sw2_515[3] = words[84 + woff_495 + 3];
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
        a_497[0] = fma_f32x2_rn_noftz(weight_514, v_516, a_497[0]);
        float2 v_517 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
        a_497[1] = fma_f32x2_rn_noftz(weight_514, v_517, a_497[1]);
        float2 v_518 = make_float2(_unpack_bf16_34[4], _unpack_bf16_34[5]);
        a_497[2] = fma_f32x2_rn_noftz(weight_514, v_518, a_497[2]);
        float2 v_519 = make_float2(_unpack_bf16_34[6], _unpack_bf16_34[7]);
        a_497[3] = fma_f32x2_rn_noftz(weight_514, v_519, a_497[3]);
        acc[0] = a_497[0].x;
        acc[1] = a_497[0].y;
        acc[2] = a_497[1].x;
        acc[3] = a_497[1].y;
        acc[4] = a_497[2].x;
        acc[5] = a_497[2].y;
        acc[6] = a_497[3].x;
        acc[7] = a_497[3].y;
        const int woff_520 = 4;
        int base_521 = (2 + group) * 1024 + thread * 8;
        float2 a_522[4];
        float2 previous_523 = make_float2(acc[8], acc[9]);
        a_522[0] = mul_f32x2_noftz(previous_523, corr_494);
        float2 previous_524 = make_float2(acc[10], acc[11]);
        a_522[1] = mul_f32x2_noftz(previous_524, corr_494);
        float2 previous_525 = make_float2(acc[12], acc[13]);
        a_522[2] = mul_f32x2_noftz(previous_525, corr_494);
        float2 previous_526 = make_float2(acc[14], acc[15]);
        a_522[3] = mul_f32x2_noftz(previous_526, corr_494);
        float2 weight_527 = make_float2(weights_492[4], weights_492[4]);
        unsigned int sw2_528[4];
        sw2_528[0] = words[56 + woff_520];
        sw2_528[1] = words[56 + woff_520 + 1];
        sw2_528[2] = words[56 + woff_520 + 2];
        sw2_528[3] = words[56 + woff_520 + 3];
        float _unpack_bf16_35[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                : "r"(sw2_528[_pair]));
        }
        float2 v_529 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
        a_522[0] = fma_f32x2_rn_noftz(weight_527, v_529, a_522[0]);
        float2 v_530 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
        a_522[1] = fma_f32x2_rn_noftz(weight_527, v_530, a_522[1]);
        float2 v_531 = make_float2(_unpack_bf16_35[4], _unpack_bf16_35[5]);
        a_522[2] = fma_f32x2_rn_noftz(weight_527, v_531, a_522[2]);
        float2 v_532 = make_float2(_unpack_bf16_35[6], _unpack_bf16_35[7]);
        a_522[3] = fma_f32x2_rn_noftz(weight_527, v_532, a_522[3]);
        float2 weight_533 = make_float2(weights_492[5], weights_492[5]);
        unsigned int sw2_534[4];
        sw2_534[0] = words[70 + woff_520];
        sw2_534[1] = words[70 + woff_520 + 1];
        sw2_534[2] = words[70 + woff_520 + 2];
        sw2_534[3] = words[70 + woff_520 + 3];
        float _unpack_bf16_36[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                : "r"(sw2_534[_pair]));
        }
        float2 v_535 = make_float2(_unpack_bf16_36[0], _unpack_bf16_36[1]);
        a_522[0] = fma_f32x2_rn_noftz(weight_533, v_535, a_522[0]);
        float2 v_536 = make_float2(_unpack_bf16_36[2], _unpack_bf16_36[3]);
        a_522[1] = fma_f32x2_rn_noftz(weight_533, v_536, a_522[1]);
        float2 v_537 = make_float2(_unpack_bf16_36[4], _unpack_bf16_36[5]);
        a_522[2] = fma_f32x2_rn_noftz(weight_533, v_537, a_522[2]);
        float2 v_538 = make_float2(_unpack_bf16_36[6], _unpack_bf16_36[7]);
        a_522[3] = fma_f32x2_rn_noftz(weight_533, v_538, a_522[3]);
        float2 weight_539 = make_float2(weights_492[6], weights_492[6]);
        unsigned int sw2_540[4];
        sw2_540[0] = words[84 + woff_520];
        sw2_540[1] = words[84 + woff_520 + 1];
        sw2_540[2] = words[84 + woff_520 + 2];
        sw2_540[3] = words[84 + woff_520 + 3];
        float _unpack_bf16_37[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                : "r"(sw2_540[_pair]));
        }
        float2 v_541 = make_float2(_unpack_bf16_37[0], _unpack_bf16_37[1]);
        a_522[0] = fma_f32x2_rn_noftz(weight_539, v_541, a_522[0]);
        float2 v_542 = make_float2(_unpack_bf16_37[2], _unpack_bf16_37[3]);
        a_522[1] = fma_f32x2_rn_noftz(weight_539, v_542, a_522[1]);
        float2 v_543 = make_float2(_unpack_bf16_37[4], _unpack_bf16_37[5]);
        a_522[2] = fma_f32x2_rn_noftz(weight_539, v_543, a_522[2]);
        float2 v_544 = make_float2(_unpack_bf16_37[6], _unpack_bf16_37[7]);
        a_522[3] = fma_f32x2_rn_noftz(weight_539, v_544, a_522[3]);
        acc[8] = a_522[0].x;
        acc[9] = a_522[0].y;
        acc[10] = a_522[1].x;
        acc[11] = a_522[1].y;
        acc[12] = a_522[2].x;
        acc[13] = a_522[2].y;
        acc[14] = a_522[3].x;
        acc[15] = a_522[3].y;
        const int woff_545 = 8;
        int base_546 = (4 + group) * 1024 + thread * 8;
        float2 a_547[4];
        float2 previous_548 = make_float2(acc[16], acc[17]);
        a_547[0] = mul_f32x2_noftz(previous_548, corr_494);
        float2 previous_549 = make_float2(acc[18], acc[19]);
        a_547[1] = mul_f32x2_noftz(previous_549, corr_494);
        float2 previous_550 = make_float2(acc[20], acc[21]);
        a_547[2] = mul_f32x2_noftz(previous_550, corr_494);
        float2 previous_551 = make_float2(acc[22], acc[23]);
        a_547[3] = mul_f32x2_noftz(previous_551, corr_494);
        float2 weight_552 = make_float2(weights_492[4], weights_492[4]);
        unsigned int sw2_553[4];
        sw2_553[0] = words[56 + woff_545];
        sw2_553[1] = words[56 + woff_545 + 1];
        sw2_553[2] = words[56 + woff_545 + 2];
        sw2_553[3] = words[56 + woff_545 + 3];
        float _unpack_bf16_38[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                : "r"(sw2_553[_pair]));
        }
        float2 v_554 = make_float2(_unpack_bf16_38[0], _unpack_bf16_38[1]);
        a_547[0] = fma_f32x2_rn_noftz(weight_552, v_554, a_547[0]);
        float2 v_555 = make_float2(_unpack_bf16_38[2], _unpack_bf16_38[3]);
        a_547[1] = fma_f32x2_rn_noftz(weight_552, v_555, a_547[1]);
        float2 v_556 = make_float2(_unpack_bf16_38[4], _unpack_bf16_38[5]);
        a_547[2] = fma_f32x2_rn_noftz(weight_552, v_556, a_547[2]);
        float2 v_557 = make_float2(_unpack_bf16_38[6], _unpack_bf16_38[7]);
        a_547[3] = fma_f32x2_rn_noftz(weight_552, v_557, a_547[3]);
        float2 weight_558 = make_float2(weights_492[5], weights_492[5]);
        unsigned int sw2_559[4];
        sw2_559[0] = words[70 + woff_545];
        sw2_559[1] = words[70 + woff_545 + 1];
        sw2_559[2] = words[70 + woff_545 + 2];
        sw2_559[3] = words[70 + woff_545 + 3];
        float _unpack_bf16_39[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                : "r"(sw2_559[_pair]));
        }
        float2 v_560 = make_float2(_unpack_bf16_39[0], _unpack_bf16_39[1]);
        a_547[0] = fma_f32x2_rn_noftz(weight_558, v_560, a_547[0]);
        float2 v_561 = make_float2(_unpack_bf16_39[2], _unpack_bf16_39[3]);
        a_547[1] = fma_f32x2_rn_noftz(weight_558, v_561, a_547[1]);
        float2 v_562 = make_float2(_unpack_bf16_39[4], _unpack_bf16_39[5]);
        a_547[2] = fma_f32x2_rn_noftz(weight_558, v_562, a_547[2]);
        float2 v_563 = make_float2(_unpack_bf16_39[6], _unpack_bf16_39[7]);
        a_547[3] = fma_f32x2_rn_noftz(weight_558, v_563, a_547[3]);
        float2 weight_564 = make_float2(weights_492[6], weights_492[6]);
        unsigned int sw2_565[4];
        sw2_565[0] = words[84 + woff_545];
        sw2_565[1] = words[84 + woff_545 + 1];
        sw2_565[2] = words[84 + woff_545 + 2];
        sw2_565[3] = words[84 + woff_545 + 3];
        float _unpack_bf16_40[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                : "r"(sw2_565[_pair]));
        }
        float2 v_566 = make_float2(_unpack_bf16_40[0], _unpack_bf16_40[1]);
        a_547[0] = fma_f32x2_rn_noftz(weight_564, v_566, a_547[0]);
        float2 v_567 = make_float2(_unpack_bf16_40[2], _unpack_bf16_40[3]);
        a_547[1] = fma_f32x2_rn_noftz(weight_564, v_567, a_547[1]);
        float2 v_568 = make_float2(_unpack_bf16_40[4], _unpack_bf16_40[5]);
        a_547[2] = fma_f32x2_rn_noftz(weight_564, v_568, a_547[2]);
        float2 v_569 = make_float2(_unpack_bf16_40[6], _unpack_bf16_40[7]);
        a_547[3] = fma_f32x2_rn_noftz(weight_564, v_569, a_547[3]);
        acc[16] = a_547[0].x;
        acc[17] = a_547[0].y;
        acc[18] = a_547[1].x;
        acc[19] = a_547[1].y;
        acc[20] = a_547[2].x;
        acc[21] = a_547[2].y;
        acc[22] = a_547[3].x;
        acc[23] = a_547[3].y;
        const int woff_570 = 12;
        int base_571 = 6144 + group * 512 + thread * 4;
        float2 a_572[4];
        float2 previous_573 = make_float2(acc[24], acc[25]);
        a_572[0] = mul_f32x2_noftz(previous_573, corr_494);
        float2 previous_574 = make_float2(acc[26], acc[27]);
        a_572[1] = mul_f32x2_noftz(previous_574, corr_494);
        float2 weight_575 = make_float2(weights_492[4], weights_492[4]);
        unsigned int sw2_576[4];
        sw2_576[0] = words[56 + woff_570];
        sw2_576[1] = words[56 + woff_570 + 1];
        float _unpack_bf16_41[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                : "r"(sw2_576[_pair]));
        }
        float2 v_577 = make_float2(_unpack_bf16_41[0], _unpack_bf16_41[1]);
        a_572[0] = fma_f32x2_rn_noftz(weight_575, v_577, a_572[0]);
        float2 v_578 = make_float2(_unpack_bf16_41[2], _unpack_bf16_41[3]);
        a_572[1] = fma_f32x2_rn_noftz(weight_575, v_578, a_572[1]);
        float2 weight_579 = make_float2(weights_492[5], weights_492[5]);
        unsigned int sw2_580[4];
        sw2_580[0] = words[70 + woff_570];
        sw2_580[1] = words[70 + woff_570 + 1];
        float _unpack_bf16_42[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                : "r"(sw2_580[_pair]));
        }
        float2 v_581 = make_float2(_unpack_bf16_42[0], _unpack_bf16_42[1]);
        a_572[0] = fma_f32x2_rn_noftz(weight_579, v_581, a_572[0]);
        float2 v_582 = make_float2(_unpack_bf16_42[2], _unpack_bf16_42[3]);
        a_572[1] = fma_f32x2_rn_noftz(weight_579, v_582, a_572[1]);
        float2 weight_583 = make_float2(weights_492[6], weights_492[6]);
        unsigned int sw2_584[4];
        sw2_584[0] = words[84 + woff_570];
        sw2_584[1] = words[84 + woff_570 + 1];
        float _unpack_bf16_43[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                : "r"(sw2_584[_pair]));
        }
        float2 v_585 = make_float2(_unpack_bf16_43[0], _unpack_bf16_43[1]);
        a_572[0] = fma_f32x2_rn_noftz(weight_583, v_585, a_572[0]);
        float2 v_586 = make_float2(_unpack_bf16_43[2], _unpack_bf16_43[3]);
        a_572[1] = fma_f32x2_rn_noftz(weight_583, v_586, a_572[1]);
        acc[24] = a_572[0].x;
        acc[25] = a_572[0].y;
        acc[26] = a_572[1].x;
        acc[27] = a_572[1].y;
        sum_running = __fmaf_rn(sum_running, correction_491, sum_weights_493);
        max_running = max_new_490;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_587 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_587, v_587, output_sq_pair);
        float2 v_588 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_588, v_588, output_sq_pair);
        float2 v_589 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_589, v_589, output_sq_pair);
        float2 v_590 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_590, v_590, output_sq_pair);
        float2 v_591 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_591, v_591, output_sq_pair);
        float2 v_592 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_592, v_592, output_sq_pair);
        float2 v_593 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_593, v_593, output_sq_pair);
        float2 v_594 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_594, v_594, output_sq_pair);
        float2 v_595 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_595, v_595, output_sq_pair);
        float2 v_596 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_596, v_596, output_sq_pair);
        float2 v_597 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_597, v_597, output_sq_pair);
        float2 v_598 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_598, v_598, output_sq_pair);
        float2 v_599 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_599, v_599, output_sq_pair);
        float2 v_600 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_600, v_600, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_125;
        float _shfl_xor_sync_126 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_126;
        float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_127;
        float _shfl_xor_sync_128 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_128;
        float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_129;
        if (lane == 0) {
            uint32_t _cluster_mapa_130;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_130) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_130), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_131;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_131) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_131), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_132 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_132;
        float _shfl_down_sync_133 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_133;
        float _shfl_down_sync_134 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_134;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_135 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_135;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_601 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_136 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_137 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_602 = 2;
        const int acc_idx_603 = value_idx_602;
        float2 _f32x2_mul_a_138 = make_float2(acc[acc_idx_603], acc[acc_idx_603 + 1]);
        float2 scaled_pair_604;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_604) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_139 = make_float2(wout[acc_idx_603], wout[acc_idx_603 + 1]);
        float2 normalized_pair_605;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_605) : "l"(*(const unsigned long long*)&scaled_pair_604), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
        output_values[value_idx_602] = normalized_pair_605.x;
        output_values[value_idx_602 + 1] = normalized_pair_605.y;
        const int value_idx_606 = 4;
        const int acc_idx_607 = value_idx_606;
        float2 _f32x2_mul_a_140 = make_float2(acc[acc_idx_607], acc[acc_idx_607 + 1]);
        float2 scaled_pair_608;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_608) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_141 = make_float2(wout[acc_idx_607], wout[acc_idx_607 + 1]);
        float2 normalized_pair_609;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_609) : "l"(*(const unsigned long long*)&scaled_pair_608), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
        output_values[value_idx_606] = normalized_pair_609.x;
        output_values[value_idx_606 + 1] = normalized_pair_609.y;
        const int value_idx_610 = 6;
        const int acc_idx_611 = value_idx_610;
        float2 _f32x2_mul_a_142 = make_float2(acc[acc_idx_611], acc[acc_idx_611 + 1]);
        float2 scaled_pair_612;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_612) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_143 = make_float2(wout[acc_idx_611], wout[acc_idx_611 + 1]);
        float2 normalized_pair_613;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_613) : "l"(*(const unsigned long long*)&scaled_pair_612), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
        output_values[value_idx_610] = normalized_pair_613.x;
        output_values[value_idx_610 + 1] = normalized_pair_613.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_601))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_614 = (2 + group) * 1024 + thread * 8;
        float output_values_615[8];
        const int value_idx_616 = 0;
        const int acc_idx_617 = 8 + value_idx_616;
        float2 _f32x2_mul_a_144 = make_float2(acc[acc_idx_617], acc[acc_idx_617 + 1]);
        float2 scaled_pair_618;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_618) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_145 = make_float2(wout[acc_idx_617], wout[acc_idx_617 + 1]);
        float2 normalized_pair_619;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_619) : "l"(*(const unsigned long long*)&scaled_pair_618), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
        output_values_615[value_idx_616] = normalized_pair_619.x;
        output_values_615[value_idx_616 + 1] = normalized_pair_619.y;
        const int value_idx_620 = 2;
        const int acc_idx_621 = 8 + value_idx_620;
        float2 _f32x2_mul_a_146 = make_float2(acc[acc_idx_621], acc[acc_idx_621 + 1]);
        float2 scaled_pair_622;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_622) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_147 = make_float2(wout[acc_idx_621], wout[acc_idx_621 + 1]);
        float2 normalized_pair_623;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_623) : "l"(*(const unsigned long long*)&scaled_pair_622), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
        output_values_615[value_idx_620] = normalized_pair_623.x;
        output_values_615[value_idx_620 + 1] = normalized_pair_623.y;
        const int value_idx_624 = 4;
        const int acc_idx_625 = 8 + value_idx_624;
        float2 _f32x2_mul_a_148 = make_float2(acc[acc_idx_625], acc[acc_idx_625 + 1]);
        float2 scaled_pair_626;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_626) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_149 = make_float2(wout[acc_idx_625], wout[acc_idx_625 + 1]);
        float2 normalized_pair_627;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_627) : "l"(*(const unsigned long long*)&scaled_pair_626), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
        output_values_615[value_idx_624] = normalized_pair_627.x;
        output_values_615[value_idx_624 + 1] = normalized_pair_627.y;
        const int value_idx_628 = 6;
        const int acc_idx_629 = 8 + value_idx_628;
        float2 _f32x2_mul_a_150 = make_float2(acc[acc_idx_629], acc[acc_idx_629 + 1]);
        float2 scaled_pair_630;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_630) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_151 = make_float2(wout[acc_idx_629], wout[acc_idx_629 + 1]);
        float2 normalized_pair_631;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_631) : "l"(*(const unsigned long long*)&scaled_pair_630), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
        output_values_615[value_idx_628] = normalized_pair_631.x;
        output_values_615[value_idx_628 + 1] = normalized_pair_631.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_615[_lp*2 + 0], output_values_615[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_615[_lp*2 + 2], output_values_615[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_615[_lp*2 + 4], output_values_615[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_615[_lp*2 + 6], output_values_615[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_614))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_632 = (4 + group) * 1024 + thread * 8;
        float output_values_633[8];
        const int value_idx_634 = 0;
        const int acc_idx_635 = 16 + value_idx_634;
        float2 _f32x2_mul_a_152 = make_float2(acc[acc_idx_635], acc[acc_idx_635 + 1]);
        float2 scaled_pair_636;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_636) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_153 = make_float2(wout[acc_idx_635], wout[acc_idx_635 + 1]);
        float2 normalized_pair_637;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_637) : "l"(*(const unsigned long long*)&scaled_pair_636), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
        output_values_633[value_idx_634] = normalized_pair_637.x;
        output_values_633[value_idx_634 + 1] = normalized_pair_637.y;
        const int value_idx_638 = 2;
        const int acc_idx_639 = 16 + value_idx_638;
        float2 _f32x2_mul_a_154 = make_float2(acc[acc_idx_639], acc[acc_idx_639 + 1]);
        float2 scaled_pair_640;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_640) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_155 = make_float2(wout[acc_idx_639], wout[acc_idx_639 + 1]);
        float2 normalized_pair_641;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_641) : "l"(*(const unsigned long long*)&scaled_pair_640), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
        output_values_633[value_idx_638] = normalized_pair_641.x;
        output_values_633[value_idx_638 + 1] = normalized_pair_641.y;
        const int value_idx_642 = 4;
        const int acc_idx_643 = 16 + value_idx_642;
        float2 _f32x2_mul_a_156 = make_float2(acc[acc_idx_643], acc[acc_idx_643 + 1]);
        float2 scaled_pair_644;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_644) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_157 = make_float2(wout[acc_idx_643], wout[acc_idx_643 + 1]);
        float2 normalized_pair_645;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_645) : "l"(*(const unsigned long long*)&scaled_pair_644), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
        output_values_633[value_idx_642] = normalized_pair_645.x;
        output_values_633[value_idx_642 + 1] = normalized_pair_645.y;
        const int value_idx_646 = 6;
        const int acc_idx_647 = 16 + value_idx_646;
        float2 _f32x2_mul_a_158 = make_float2(acc[acc_idx_647], acc[acc_idx_647 + 1]);
        float2 scaled_pair_648;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_648) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_159 = make_float2(wout[acc_idx_647], wout[acc_idx_647 + 1]);
        float2 normalized_pair_649;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_649) : "l"(*(const unsigned long long*)&scaled_pair_648), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
        output_values_633[value_idx_646] = normalized_pair_649.x;
        output_values_633[value_idx_646 + 1] = normalized_pair_649.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_633[_lp*2 + 0], output_values_633[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_633[_lp*2 + 2], output_values_633[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_633[_lp*2 + 4], output_values_633[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_633[_lp*2 + 6], output_values_633[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_632))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_650 = 6144 + group * 512 + thread * 4;
        float output_values_651[8];
        const int value_idx_652 = 0;
        const int acc_idx_653 = 24 + value_idx_652;
        float2 _f32x2_mul_a_160 = make_float2(acc[acc_idx_653], acc[acc_idx_653 + 1]);
        float2 scaled_pair_654;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_654) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_161 = make_float2(wout[acc_idx_653], wout[acc_idx_653 + 1]);
        float2 normalized_pair_655;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_655) : "l"(*(const unsigned long long*)&scaled_pair_654), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
        output_values_651[value_idx_652] = normalized_pair_655.x;
        output_values_651[value_idx_652 + 1] = normalized_pair_655.y;
        const int value_idx_656 = 2;
        const int acc_idx_657 = 24 + value_idx_656;
        float2 _f32x2_mul_a_162 = make_float2(acc[acc_idx_657], acc[acc_idx_657 + 1]);
        float2 scaled_pair_658;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_658) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_163 = make_float2(wout[acc_idx_657], wout[acc_idx_657 + 1]);
        float2 normalized_pair_659;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_659) : "l"(*(const unsigned long long*)&scaled_pair_658), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
        output_values_651[value_idx_656] = normalized_pair_659.x;
        output_values_651[value_idx_656 + 1] = normalized_pair_659.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_651[_lp*2 + 0], output_values_651[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_651[_lp*2 + 2], output_values_651[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_650))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
