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
#define SMEM_STATS_STAGE_BYTES 576
#define SMEM_STATS_STRIDE 576
#define SMEM_OUT_STATS_OFF 576
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 640
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
kernel_cake_kimi_k3_attn_res_55c951a57dd705a8a578(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 576);
    const int out_stats_addr = smem + 576;

    // === Task calls (declaration order) ===
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
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
        unsigned int words[126];
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
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base)) + 0);
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
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base)) + 0);
            _vec_load_7[0 + 0] = _uv4_7.x;
            _vec_load_7[0 + 1] = _uv4_7.y;
            _vec_load_7[0 + 2] = _uv4_7.z;
            _vec_load_7[0 + 3] = _uv4_7.w;
        }
        words[98 + woff] = _vec_load_7[0];
        words[98 + woff + 1] = _vec_load_7[1];
        words[98 + woff + 2] = _vec_load_7[2];
        words[98 + woff + 3] = _vec_load_7[3];
        unsigned int _vec_load_8[4];
        {
            uint4 _uv4_8 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
            _vec_load_8[0 + 0] = _uv4_8.x;
            _vec_load_8[0 + 1] = _uv4_8.y;
            _vec_load_8[0 + 2] = _uv4_8.z;
            _vec_load_8[0 + 3] = _uv4_8.w;
        }
        words[112 + woff] = _vec_load_8[0];
        words[112 + woff + 1] = _vec_load_8[1];
        words[112 + woff + 2] = _vec_load_8[2];
        words[112 + woff + 3] = _vec_load_8[3];
        unsigned int _vec_load_9[4];
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_9[0 + 0] = _uv4_9.x;
            _vec_load_9[0 + 1] = _uv4_9.y;
            _vec_load_9[0 + 2] = _uv4_9.z;
            _vec_load_9[0 + 3] = _uv4_9.w;
        }
        dwords[woff] = _vec_load_9[0];
        dwords[woff + 1] = _vec_load_9[1];
        dwords[woff + 2] = _vec_load_9[2];
        dwords[woff + 3] = _vec_load_9[3];
        float _vec_load_10[8];
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
                        : "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_10[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_10[_pair]));
                }
            }
        }
        float _vec_load_11[8];
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
                        : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_11[_pair]));
                }
            }
        }
        q[0] = _vec_load_10[0] * _vec_load_11[0];
        q[1] = _vec_load_10[1] * _vec_load_11[1];
        q[2] = _vec_load_10[2] * _vec_load_11[2];
        q[3] = _vec_load_10[3] * _vec_load_11[3];
        q[4] = _vec_load_10[4] * _vec_load_11[4];
        q[5] = _vec_load_10[5] * _vec_load_11[5];
        q[6] = _vec_load_10[6] * _vec_load_11[6];
        q[7] = _vec_load_10[7] * _vec_load_11[7];
        float _vec_load_12[8];
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
                        : "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_12[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_12[_pair]));
                }
            }
        }
        wout[0] = _vec_load_12[0];
        wout[1] = _vec_load_12[1];
        wout[2] = _vec_load_12[2];
        wout[3] = _vec_load_12[3];
        wout[4] = _vec_load_12[4];
        wout[5] = _vec_load_12[5];
        wout[6] = _vec_load_12[6];
        wout[7] = _vec_load_12[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_13[0 + 0] = _uv4_13.x;
            _vec_load_13[0 + 1] = _uv4_13.y;
            _vec_load_13[0 + 2] = _uv4_13.z;
            _vec_load_13[0 + 3] = _uv4_13.w;
        }
        words[woff_1] = _vec_load_13[0];
        words[woff_1 + 1] = _vec_load_13[1];
        words[woff_1 + 2] = _vec_load_13[2];
        words[woff_1 + 3] = _vec_load_13[3];
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        words[14 + woff_1] = _vec_load_14[0];
        words[14 + woff_1 + 1] = _vec_load_14[1];
        words[14 + woff_1 + 2] = _vec_load_14[2];
        words[14 + woff_1 + 3] = _vec_load_14[3];
        unsigned int _vec_load_15[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_15[0 + 0] = _uv4_15.x;
            _vec_load_15[0 + 1] = _uv4_15.y;
            _vec_load_15[0 + 2] = _uv4_15.z;
            _vec_load_15[0 + 3] = _uv4_15.w;
        }
        words[28 + woff_1] = _vec_load_15[0];
        words[28 + woff_1 + 1] = _vec_load_15[1];
        words[28 + woff_1 + 2] = _vec_load_15[2];
        words[28 + woff_1 + 3] = _vec_load_15[3];
        unsigned int _vec_load_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_16[0 + 0] = _uv4_16.x;
            _vec_load_16[0 + 1] = _uv4_16.y;
            _vec_load_16[0 + 2] = _uv4_16.z;
            _vec_load_16[0 + 3] = _uv4_16.w;
        }
        words[42 + woff_1] = _vec_load_16[0];
        words[42 + woff_1 + 1] = _vec_load_16[1];
        words[42 + woff_1 + 2] = _vec_load_16[2];
        words[42 + woff_1 + 3] = _vec_load_16[3];
        unsigned int _vec_load_17[4];
        {
            uint4 _uv4_17 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_17[0 + 0] = _uv4_17.x;
            _vec_load_17[0 + 1] = _uv4_17.y;
            _vec_load_17[0 + 2] = _uv4_17.z;
            _vec_load_17[0 + 3] = _uv4_17.w;
        }
        words[56 + woff_1] = _vec_load_17[0];
        words[56 + woff_1 + 1] = _vec_load_17[1];
        words[56 + woff_1 + 2] = _vec_load_17[2];
        words[56 + woff_1 + 3] = _vec_load_17[3];
        unsigned int _vec_load_18[4];
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_18[0 + 0] = _uv4_18.x;
            _vec_load_18[0 + 1] = _uv4_18.y;
            _vec_load_18[0 + 2] = _uv4_18.z;
            _vec_load_18[0 + 3] = _uv4_18.w;
        }
        words[70 + woff_1] = _vec_load_18[0];
        words[70 + woff_1 + 1] = _vec_load_18[1];
        words[70 + woff_1 + 2] = _vec_load_18[2];
        words[70 + woff_1 + 3] = _vec_load_18[3];
        unsigned int _vec_load_19[4];
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_19[0 + 0] = _uv4_19.x;
            _vec_load_19[0 + 1] = _uv4_19.y;
            _vec_load_19[0 + 2] = _uv4_19.z;
            _vec_load_19[0 + 3] = _uv4_19.w;
        }
        words[84 + woff_1] = _vec_load_19[0];
        words[84 + woff_1 + 1] = _vec_load_19[1];
        words[84 + woff_1 + 2] = _vec_load_19[2];
        words[84 + woff_1 + 3] = _vec_load_19[3];
        unsigned int _vec_load_20[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_20[0 + 0] = _uv4_20.x;
            _vec_load_20[0 + 1] = _uv4_20.y;
            _vec_load_20[0 + 2] = _uv4_20.z;
            _vec_load_20[0 + 3] = _uv4_20.w;
        }
        words[98 + woff_1] = _vec_load_20[0];
        words[98 + woff_1 + 1] = _vec_load_20[1];
        words[98 + woff_1 + 2] = _vec_load_20[2];
        words[98 + woff_1 + 3] = _vec_load_20[3];
        unsigned int _vec_load_21[4];
        {
            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_21[0 + 0] = _uv4_21.x;
            _vec_load_21[0 + 1] = _uv4_21.y;
            _vec_load_21[0 + 2] = _uv4_21.z;
            _vec_load_21[0 + 3] = _uv4_21.w;
        }
        words[112 + woff_1] = _vec_load_21[0];
        words[112 + woff_1 + 1] = _vec_load_21[1];
        words[112 + woff_1 + 2] = _vec_load_21[2];
        words[112 + woff_1 + 3] = _vec_load_21[3];
        unsigned int _vec_load_22[4];
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_22[0 + 0] = _uv4_22.x;
            _vec_load_22[0 + 1] = _uv4_22.y;
            _vec_load_22[0 + 2] = _uv4_22.z;
            _vec_load_22[0 + 3] = _uv4_22.w;
        }
        dwords[woff_1] = _vec_load_22[0];
        dwords[woff_1 + 1] = _vec_load_22[1];
        dwords[woff_1 + 2] = _vec_load_22[2];
        dwords[woff_1 + 3] = _vec_load_22[3];
        float _vec_load_23[8];
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
                        : "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_23[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_23[_pair]));
                }
            }
        }
        float _vec_load_24[8];
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
                        : "=f"((&_vec_load_24[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_24[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_24[_pair]));
                }
            }
        }
        q[8] = _vec_load_23[0] * _vec_load_24[0];
        q[9] = _vec_load_23[1] * _vec_load_24[1];
        q[10] = _vec_load_23[2] * _vec_load_24[2];
        q[11] = _vec_load_23[3] * _vec_load_24[3];
        q[12] = _vec_load_23[4] * _vec_load_24[4];
        q[13] = _vec_load_23[5] * _vec_load_24[5];
        q[14] = _vec_load_23[6] * _vec_load_24[6];
        q[15] = _vec_load_23[7] * _vec_load_24[7];
        float _vec_load_25[8];
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
                        : "=f"((&_vec_load_25[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_25[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_25[_pair]));
                }
            }
        }
        wout[8] = _vec_load_25[0];
        wout[9] = _vec_load_25[1];
        wout[10] = _vec_load_25[2];
        wout[11] = _vec_load_25[3];
        wout[12] = _vec_load_25[4];
        wout[13] = _vec_load_25[5];
        wout[14] = _vec_load_25[6];
        wout[15] = _vec_load_25[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_26[4];
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_26[0 + 0] = _uv4_26.x;
            _vec_load_26[0 + 1] = _uv4_26.y;
            _vec_load_26[0 + 2] = _uv4_26.z;
            _vec_load_26[0 + 3] = _uv4_26.w;
        }
        words[woff_3] = _vec_load_26[0];
        words[woff_3 + 1] = _vec_load_26[1];
        words[woff_3 + 2] = _vec_load_26[2];
        words[woff_3 + 3] = _vec_load_26[3];
        unsigned int _vec_load_27[4];
        {
            uint4 _uv4_27 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_27[0 + 0] = _uv4_27.x;
            _vec_load_27[0 + 1] = _uv4_27.y;
            _vec_load_27[0 + 2] = _uv4_27.z;
            _vec_load_27[0 + 3] = _uv4_27.w;
        }
        words[14 + woff_3] = _vec_load_27[0];
        words[14 + woff_3 + 1] = _vec_load_27[1];
        words[14 + woff_3 + 2] = _vec_load_27[2];
        words[14 + woff_3 + 3] = _vec_load_27[3];
        unsigned int _vec_load_28[4];
        {
            uint4 _uv4_28 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_28[0 + 0] = _uv4_28.x;
            _vec_load_28[0 + 1] = _uv4_28.y;
            _vec_load_28[0 + 2] = _uv4_28.z;
            _vec_load_28[0 + 3] = _uv4_28.w;
        }
        words[28 + woff_3] = _vec_load_28[0];
        words[28 + woff_3 + 1] = _vec_load_28[1];
        words[28 + woff_3 + 2] = _vec_load_28[2];
        words[28 + woff_3 + 3] = _vec_load_28[3];
        unsigned int _vec_load_29[4];
        {
            uint4 _uv4_29 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_29[0 + 0] = _uv4_29.x;
            _vec_load_29[0 + 1] = _uv4_29.y;
            _vec_load_29[0 + 2] = _uv4_29.z;
            _vec_load_29[0 + 3] = _uv4_29.w;
        }
        words[42 + woff_3] = _vec_load_29[0];
        words[42 + woff_3 + 1] = _vec_load_29[1];
        words[42 + woff_3 + 2] = _vec_load_29[2];
        words[42 + woff_3 + 3] = _vec_load_29[3];
        unsigned int _vec_load_30[4];
        {
            uint4 _uv4_30 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_30[0 + 0] = _uv4_30.x;
            _vec_load_30[0 + 1] = _uv4_30.y;
            _vec_load_30[0 + 2] = _uv4_30.z;
            _vec_load_30[0 + 3] = _uv4_30.w;
        }
        words[56 + woff_3] = _vec_load_30[0];
        words[56 + woff_3 + 1] = _vec_load_30[1];
        words[56 + woff_3 + 2] = _vec_load_30[2];
        words[56 + woff_3 + 3] = _vec_load_30[3];
        unsigned int _vec_load_31[4];
        {
            uint4 _uv4_31 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_31[0 + 0] = _uv4_31.x;
            _vec_load_31[0 + 1] = _uv4_31.y;
            _vec_load_31[0 + 2] = _uv4_31.z;
            _vec_load_31[0 + 3] = _uv4_31.w;
        }
        words[70 + woff_3] = _vec_load_31[0];
        words[70 + woff_3 + 1] = _vec_load_31[1];
        words[70 + woff_3 + 2] = _vec_load_31[2];
        words[70 + woff_3 + 3] = _vec_load_31[3];
        unsigned int _vec_load_32[4];
        {
            uint4 _uv4_32 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_32[0 + 0] = _uv4_32.x;
            _vec_load_32[0 + 1] = _uv4_32.y;
            _vec_load_32[0 + 2] = _uv4_32.z;
            _vec_load_32[0 + 3] = _uv4_32.w;
        }
        words[84 + woff_3] = _vec_load_32[0];
        words[84 + woff_3 + 1] = _vec_load_32[1];
        words[84 + woff_3 + 2] = _vec_load_32[2];
        words[84 + woff_3 + 3] = _vec_load_32[3];
        unsigned int _vec_load_33[4];
        {
            uint4 _uv4_33 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_33[0 + 0] = _uv4_33.x;
            _vec_load_33[0 + 1] = _uv4_33.y;
            _vec_load_33[0 + 2] = _uv4_33.z;
            _vec_load_33[0 + 3] = _uv4_33.w;
        }
        words[98 + woff_3] = _vec_load_33[0];
        words[98 + woff_3 + 1] = _vec_load_33[1];
        words[98 + woff_3 + 2] = _vec_load_33[2];
        words[98 + woff_3 + 3] = _vec_load_33[3];
        unsigned int _vec_load_34[4];
        {
            uint4 _uv4_34 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_34[0 + 0] = _uv4_34.x;
            _vec_load_34[0 + 1] = _uv4_34.y;
            _vec_load_34[0 + 2] = _uv4_34.z;
            _vec_load_34[0 + 3] = _uv4_34.w;
        }
        words[112 + woff_3] = _vec_load_34[0];
        words[112 + woff_3 + 1] = _vec_load_34[1];
        words[112 + woff_3 + 2] = _vec_load_34[2];
        words[112 + woff_3 + 3] = _vec_load_34[3];
        unsigned int _vec_load_35[4];
        {
            uint4 _uv4_35 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_35[0 + 0] = _uv4_35.x;
            _vec_load_35[0 + 1] = _uv4_35.y;
            _vec_load_35[0 + 2] = _uv4_35.z;
            _vec_load_35[0 + 3] = _uv4_35.w;
        }
        dwords[woff_3] = _vec_load_35[0];
        dwords[woff_3 + 1] = _vec_load_35[1];
        dwords[woff_3 + 2] = _vec_load_35[2];
        dwords[woff_3 + 3] = _vec_load_35[3];
        float _vec_load_36[8];
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
                        : "=f"((&_vec_load_36[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_36[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_36[_pair]));
                }
            }
        }
        float _vec_load_37[8];
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
                        : "=f"((&_vec_load_37[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_37[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_37[_pair]));
                }
            }
        }
        q[16] = _vec_load_36[0] * _vec_load_37[0];
        q[17] = _vec_load_36[1] * _vec_load_37[1];
        q[18] = _vec_load_36[2] * _vec_load_37[2];
        q[19] = _vec_load_36[3] * _vec_load_37[3];
        q[20] = _vec_load_36[4] * _vec_load_37[4];
        q[21] = _vec_load_36[5] * _vec_load_37[5];
        q[22] = _vec_load_36[6] * _vec_load_37[6];
        q[23] = _vec_load_36[7] * _vec_load_37[7];
        float _vec_load_38[8];
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
                        : "=f"((&_vec_load_38[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_38[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_38[_pair]));
                }
            }
        }
        wout[16] = _vec_load_38[0];
        wout[17] = _vec_load_38[1];
        wout[18] = _vec_load_38[2];
        wout[19] = _vec_load_38[3];
        wout[20] = _vec_load_38[4];
        wout[21] = _vec_load_38[5];
        wout[22] = _vec_load_38[6];
        wout[23] = _vec_load_38[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_39[1];
        {
            _vec_load_39[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_39[0];
        unsigned int _vec_load_40[1];
        {
            _vec_load_40[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_40[0];
        unsigned int _vec_load_41[1];
        {
            _vec_load_41[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_41[0];
        unsigned int _vec_load_42[1];
        {
            _vec_load_42[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_42[0];
        unsigned int _vec_load_43[1];
        {
            _vec_load_43[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_43[0];
        unsigned int _vec_load_44[1];
        {
            _vec_load_44[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_44[0];
        unsigned int _vec_load_45[1];
        {
            _vec_load_45[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[42 + woff_5] = _vec_load_45[0];
        unsigned int _vec_load_46[1];
        {
            _vec_load_46[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[42 + woff_5 + 1] = _vec_load_46[0];
        unsigned int _vec_load_47[1];
        {
            _vec_load_47[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[56 + woff_5] = _vec_load_47[0];
        unsigned int _vec_load_48[1];
        {
            _vec_load_48[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[56 + woff_5 + 1] = _vec_load_48[0];
        unsigned int _vec_load_49[1];
        {
            _vec_load_49[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[70 + woff_5] = _vec_load_49[0];
        unsigned int _vec_load_50[1];
        {
            _vec_load_50[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 5 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[70 + woff_5 + 1] = _vec_load_50[0];
        unsigned int _vec_load_51[1];
        {
            _vec_load_51[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[84 + woff_5] = _vec_load_51[0];
        unsigned int _vec_load_52[1];
        {
            _vec_load_52[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 6 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[84 + woff_5 + 1] = _vec_load_52[0];
        unsigned int _vec_load_53[1];
        {
            _vec_load_53[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[98 + woff_5] = _vec_load_53[0];
        unsigned int _vec_load_54[1];
        {
            _vec_load_54[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 7 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[98 + woff_5 + 1] = _vec_load_54[0];
        unsigned int _vec_load_55[1];
        {
            _vec_load_55[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[112 + woff_5] = _vec_load_55[0];
        unsigned int _vec_load_56[1];
        {
            _vec_load_56[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[112 + woff_5 + 1] = _vec_load_56[0];
        unsigned int _vec_load_57[1];
        {
            _vec_load_57[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_57[0];
        unsigned int _vec_load_58[1];
        {
            _vec_load_58[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_58[0];
        float _vec_load_59[4];
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
                    : "=f"((&_vec_load_59[0 + _pair * 2])[0]), "=f"((&_vec_load_59[0 + _pair * 2])[1])
                    : "r"(_vpairs_39[_pair]));
            }
        }
        float _vec_load_60[4];
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
                    : "=f"((&_vec_load_60[0 + _pair * 2])[0]), "=f"((&_vec_load_60[0 + _pair * 2])[1])
                    : "r"(_vpairs_40[_pair]));
            }
        }
        q[24] = _vec_load_59[0] * _vec_load_60[0];
        q[25] = _vec_load_59[1] * _vec_load_60[1];
        q[26] = _vec_load_59[2] * _vec_load_60[2];
        q[27] = _vec_load_59[3] * _vec_load_60[3];
        float _vec_load_61[4];
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
                    : "=f"((&_vec_load_61[0 + _pair * 2])[0]), "=f"((&_vec_load_61[0 + _pair * 2])[1])
                    : "r"(_vpairs_41[_pair]));
            }
        }
        wout[24] = _vec_load_61[0];
        wout[25] = _vec_load_61[1];
        wout[26] = _vec_load_61[2];
        wout[27] = _vec_load_61[3];
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
        sq[7] = make_float2(0.0f, 0.0f);
        dot[7] = make_float2(0.0f, 0.0f);
        sq[8] = make_float2(0.0f, 0.0f);
        dot[8] = make_float2(0.0f, 0.0f);
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
        float2 v_60 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 qp_61 = make_float2(q[0], q[1]);
        sq[6] = fma_f32x2_rn_noftz(v_60, v_60, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_60, qp_61, dot[6]);
        float2 v_62 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 qp_63 = make_float2(q[2], q[3]);
        sq[6] = fma_f32x2_rn_noftz(v_62, v_62, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_62, qp_63, dot[6]);
        float2 v_64 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 qp_65 = make_float2(q[4], q[5]);
        sq[6] = fma_f32x2_rn_noftz(v_64, v_64, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_64, qp_65, dot[6]);
        float2 v_66 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 qp_67 = make_float2(q[6], q[7]);
        sq[6] = fma_f32x2_rn_noftz(v_66, v_66, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_66, qp_67, dot[6]);
        unsigned int sw_68[4];
        sw_68[0] = words[98 + woff_7];
        sw_68[1] = words[98 + woff_7 + 1];
        sw_68[2] = words[98 + woff_7 + 2];
        sw_68[3] = words[98 + woff_7 + 3];
        float _unpack_bf16_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_7[_pair * 2])[0]), "=f"((&_unpack_bf16_7[_pair * 2])[1])
                : "r"(sw_68[_pair]));
        }
        float2 v_69 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 qp_70 = make_float2(q[0], q[1]);
        sq[7] = fma_f32x2_rn_noftz(v_69, v_69, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_69, qp_70, dot[7]);
        float2 v_71 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 qp_72 = make_float2(q[2], q[3]);
        sq[7] = fma_f32x2_rn_noftz(v_71, v_71, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_71, qp_72, dot[7]);
        float2 v_73 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 qp_74 = make_float2(q[4], q[5]);
        sq[7] = fma_f32x2_rn_noftz(v_73, v_73, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_73, qp_74, dot[7]);
        float2 v_75 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 qp_76 = make_float2(q[6], q[7]);
        sq[7] = fma_f32x2_rn_noftz(v_75, v_75, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_75, qp_76, dot[7]);
        unsigned int sw_77[4];
        sw_77[0] = words[112 + woff_7];
        sw_77[1] = words[112 + woff_7 + 1];
        sw_77[2] = words[112 + woff_7 + 2];
        sw_77[3] = words[112 + woff_7 + 3];
        __nv_bfloat162 a = __as_bf16x2(sw_77[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_77[0] = __as_u32(mixed);
        words[112 + woff_7] = sw_77[0];
        __nv_bfloat162 a_78 = __as_bf16x2(sw_77[1]);
        __nv_bfloat162 d_79 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_80 = a_78 + d_79;
        sw_77[1] = __as_u32(mixed_80);
        words[112 + woff_7 + 1] = sw_77[1];
        __nv_bfloat162 a_81 = __as_bf16x2(sw_77[2]);
        __nv_bfloat162 d_82 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_83 = a_81 + d_82;
        sw_77[2] = __as_u32(mixed_83);
        words[112 + woff_7 + 2] = sw_77[2];
        __nv_bfloat162 a_84 = __as_bf16x2(sw_77[3]);
        __nv_bfloat162 d_85 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_86 = a_84 + d_85;
        sw_77[3] = __as_u32(mixed_86);
        words[112 + woff_7 + 3] = sw_77[3];
        {
            int4 _iv4 = make_int4(sw_77[0 + 0], sw_77[0 + 1], sw_77[0 + 2], sw_77[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
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
                : "r"(sw_77[_pair]));
        }
        float2 v_87 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 qp_88 = make_float2(q[0], q[1]);
        sq[8] = fma_f32x2_rn_noftz(v_87, v_87, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_87, qp_88, dot[8]);
        float2 v_89 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 qp_90 = make_float2(q[2], q[3]);
        sq[8] = fma_f32x2_rn_noftz(v_89, v_89, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_89, qp_90, dot[8]);
        float2 v_91 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 qp_92 = make_float2(q[4], q[5]);
        sq[8] = fma_f32x2_rn_noftz(v_91, v_91, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_91, qp_92, dot[8]);
        float2 v_93 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 qp_94 = make_float2(q[6], q[7]);
        sq[8] = fma_f32x2_rn_noftz(v_93, v_93, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_93, qp_94, dot[8]);
        int base_95 = (2 + group) * 1024 + thread * 8;
        const int woff_96 = 4;
        unsigned int sw_97[4];
        sw_97[0] = words[woff_96];
        sw_97[1] = words[woff_96 + 1];
        sw_97[2] = words[woff_96 + 2];
        sw_97[3] = words[woff_96 + 3];
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
        fsrc[8] = _unpack_bf16_9[0];
        fsrc[9] = _unpack_bf16_9[1];
        fsrc[10] = _unpack_bf16_9[2];
        fsrc[11] = _unpack_bf16_9[3];
        fsrc[12] = _unpack_bf16_9[4];
        fsrc[13] = _unpack_bf16_9[5];
        fsrc[14] = _unpack_bf16_9[6];
        fsrc[15] = _unpack_bf16_9[7];
        float2 v_98 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 qp_99 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_98, v_98, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_98, qp_99, dot[0]);
        float2 v_100 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 qp_101 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_100, v_100, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_100, qp_101, dot[0]);
        float2 v_102 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 qp_103 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_102, v_102, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_102, qp_103, dot[0]);
        float2 v_104 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 qp_105 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_104, v_104, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_104, qp_105, dot[0]);
        unsigned int sw_106[4];
        sw_106[0] = words[14 + woff_96];
        sw_106[1] = words[14 + woff_96 + 1];
        sw_106[2] = words[14 + woff_96 + 2];
        sw_106[3] = words[14 + woff_96 + 3];
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
        fsrc[36] = _unpack_bf16_10[0];
        fsrc[37] = _unpack_bf16_10[1];
        fsrc[38] = _unpack_bf16_10[2];
        fsrc[39] = _unpack_bf16_10[3];
        fsrc[40] = _unpack_bf16_10[4];
        fsrc[41] = _unpack_bf16_10[5];
        fsrc[42] = _unpack_bf16_10[6];
        fsrc[43] = _unpack_bf16_10[7];
        float2 v_107 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 qp_108 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_107, v_107, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_107, qp_108, dot[1]);
        float2 v_109 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 qp_110 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_109, v_109, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_109, qp_110, dot[1]);
        float2 v_111 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 qp_112 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_111, v_111, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_111, qp_112, dot[1]);
        float2 v_113 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 qp_114 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_113, v_113, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_113, qp_114, dot[1]);
        unsigned int sw_115[4];
        sw_115[0] = words[28 + woff_96];
        sw_115[1] = words[28 + woff_96 + 1];
        sw_115[2] = words[28 + woff_96 + 2];
        sw_115[3] = words[28 + woff_96 + 3];
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
        fsrc[64] = _unpack_bf16_11[0];
        fsrc[65] = _unpack_bf16_11[1];
        fsrc[66] = _unpack_bf16_11[2];
        fsrc[67] = _unpack_bf16_11[3];
        fsrc[68] = _unpack_bf16_11[4];
        fsrc[69] = _unpack_bf16_11[5];
        fsrc[70] = _unpack_bf16_11[6];
        fsrc[71] = _unpack_bf16_11[7];
        float2 v_116 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 qp_117 = make_float2(q[8], q[9]);
        sq[2] = fma_f32x2_rn_noftz(v_116, v_116, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_116, qp_117, dot[2]);
        float2 v_118 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 qp_119 = make_float2(q[10], q[11]);
        sq[2] = fma_f32x2_rn_noftz(v_118, v_118, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_118, qp_119, dot[2]);
        float2 v_120 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 qp_121 = make_float2(q[12], q[13]);
        sq[2] = fma_f32x2_rn_noftz(v_120, v_120, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_120, qp_121, dot[2]);
        float2 v_122 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 qp_123 = make_float2(q[14], q[15]);
        sq[2] = fma_f32x2_rn_noftz(v_122, v_122, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_122, qp_123, dot[2]);
        unsigned int sw_124[4];
        sw_124[0] = words[42 + woff_96];
        sw_124[1] = words[42 + woff_96 + 1];
        sw_124[2] = words[42 + woff_96 + 2];
        sw_124[3] = words[42 + woff_96 + 3];
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
        sq[3] = fma_f32x2_rn_noftz(v_125, v_125, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_125, qp_126, dot[3]);
        float2 v_127 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 qp_128 = make_float2(q[10], q[11]);
        sq[3] = fma_f32x2_rn_noftz(v_127, v_127, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_127, qp_128, dot[3]);
        float2 v_129 = make_float2(_unpack_bf16_12[4], _unpack_bf16_12[5]);
        float2 qp_130 = make_float2(q[12], q[13]);
        sq[3] = fma_f32x2_rn_noftz(v_129, v_129, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_129, qp_130, dot[3]);
        float2 v_131 = make_float2(_unpack_bf16_12[6], _unpack_bf16_12[7]);
        float2 qp_132 = make_float2(q[14], q[15]);
        sq[3] = fma_f32x2_rn_noftz(v_131, v_131, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_131, qp_132, dot[3]);
        unsigned int sw_133[4];
        sw_133[0] = words[56 + woff_96];
        sw_133[1] = words[56 + woff_96 + 1];
        sw_133[2] = words[56 + woff_96 + 2];
        sw_133[3] = words[56 + woff_96 + 3];
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
        float2 v_134 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 qp_135 = make_float2(q[8], q[9]);
        sq[4] = fma_f32x2_rn_noftz(v_134, v_134, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_134, qp_135, dot[4]);
        float2 v_136 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 qp_137 = make_float2(q[10], q[11]);
        sq[4] = fma_f32x2_rn_noftz(v_136, v_136, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_136, qp_137, dot[4]);
        float2 v_138 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
        float2 qp_139 = make_float2(q[12], q[13]);
        sq[4] = fma_f32x2_rn_noftz(v_138, v_138, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_138, qp_139, dot[4]);
        float2 v_140 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
        float2 qp_141 = make_float2(q[14], q[15]);
        sq[4] = fma_f32x2_rn_noftz(v_140, v_140, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_140, qp_141, dot[4]);
        unsigned int sw_142[4];
        sw_142[0] = words[70 + woff_96];
        sw_142[1] = words[70 + woff_96 + 1];
        sw_142[2] = words[70 + woff_96 + 2];
        sw_142[3] = words[70 + woff_96 + 3];
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                : "r"(sw_142[_pair]));
        }
        float2 v_143 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 qp_144 = make_float2(q[8], q[9]);
        sq[5] = fma_f32x2_rn_noftz(v_143, v_143, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_143, qp_144, dot[5]);
        float2 v_145 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 qp_146 = make_float2(q[10], q[11]);
        sq[5] = fma_f32x2_rn_noftz(v_145, v_145, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_145, qp_146, dot[5]);
        float2 v_147 = make_float2(_unpack_bf16_14[4], _unpack_bf16_14[5]);
        float2 qp_148 = make_float2(q[12], q[13]);
        sq[5] = fma_f32x2_rn_noftz(v_147, v_147, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_147, qp_148, dot[5]);
        float2 v_149 = make_float2(_unpack_bf16_14[6], _unpack_bf16_14[7]);
        float2 qp_150 = make_float2(q[14], q[15]);
        sq[5] = fma_f32x2_rn_noftz(v_149, v_149, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_149, qp_150, dot[5]);
        unsigned int sw_151[4];
        sw_151[0] = words[84 + woff_96];
        sw_151[1] = words[84 + woff_96 + 1];
        sw_151[2] = words[84 + woff_96 + 2];
        sw_151[3] = words[84 + woff_96 + 3];
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                : "r"(sw_151[_pair]));
        }
        float2 v_152 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_153 = make_float2(q[8], q[9]);
        sq[6] = fma_f32x2_rn_noftz(v_152, v_152, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_152, qp_153, dot[6]);
        float2 v_154 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_155 = make_float2(q[10], q[11]);
        sq[6] = fma_f32x2_rn_noftz(v_154, v_154, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_154, qp_155, dot[6]);
        float2 v_156 = make_float2(_unpack_bf16_15[4], _unpack_bf16_15[5]);
        float2 qp_157 = make_float2(q[12], q[13]);
        sq[6] = fma_f32x2_rn_noftz(v_156, v_156, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_156, qp_157, dot[6]);
        float2 v_158 = make_float2(_unpack_bf16_15[6], _unpack_bf16_15[7]);
        float2 qp_159 = make_float2(q[14], q[15]);
        sq[6] = fma_f32x2_rn_noftz(v_158, v_158, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_158, qp_159, dot[6]);
        unsigned int sw_160[4];
        sw_160[0] = words[98 + woff_96];
        sw_160[1] = words[98 + woff_96 + 1];
        sw_160[2] = words[98 + woff_96 + 2];
        sw_160[3] = words[98 + woff_96 + 3];
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(sw_160[_pair]));
        }
        float2 v_161 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_162 = make_float2(q[8], q[9]);
        sq[7] = fma_f32x2_rn_noftz(v_161, v_161, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_161, qp_162, dot[7]);
        float2 v_163 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_164 = make_float2(q[10], q[11]);
        sq[7] = fma_f32x2_rn_noftz(v_163, v_163, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_163, qp_164, dot[7]);
        float2 v_165 = make_float2(_unpack_bf16_16[4], _unpack_bf16_16[5]);
        float2 qp_166 = make_float2(q[12], q[13]);
        sq[7] = fma_f32x2_rn_noftz(v_165, v_165, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_165, qp_166, dot[7]);
        float2 v_167 = make_float2(_unpack_bf16_16[6], _unpack_bf16_16[7]);
        float2 qp_168 = make_float2(q[14], q[15]);
        sq[7] = fma_f32x2_rn_noftz(v_167, v_167, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_167, qp_168, dot[7]);
        unsigned int sw_169[4];
        sw_169[0] = words[112 + woff_96];
        sw_169[1] = words[112 + woff_96 + 1];
        sw_169[2] = words[112 + woff_96 + 2];
        sw_169[3] = words[112 + woff_96 + 3];
        __nv_bfloat162 a_170 = __as_bf16x2(sw_169[0]);
        __nv_bfloat162 d_171 = __as_bf16x2(dwords[woff_96]);
        __nv_bfloat162 mixed_172 = a_170 + d_171;
        sw_169[0] = __as_u32(mixed_172);
        words[112 + woff_96] = sw_169[0];
        __nv_bfloat162 a_173 = __as_bf16x2(sw_169[1]);
        __nv_bfloat162 d_174 = __as_bf16x2(dwords[woff_96 + 1]);
        __nv_bfloat162 mixed_175 = a_173 + d_174;
        sw_169[1] = __as_u32(mixed_175);
        words[112 + woff_96 + 1] = sw_169[1];
        __nv_bfloat162 a_176 = __as_bf16x2(sw_169[2]);
        __nv_bfloat162 d_177 = __as_bf16x2(dwords[woff_96 + 2]);
        __nv_bfloat162 mixed_178 = a_176 + d_177;
        sw_169[2] = __as_u32(mixed_178);
        words[112 + woff_96 + 2] = sw_169[2];
        __nv_bfloat162 a_179 = __as_bf16x2(sw_169[3]);
        __nv_bfloat162 d_180 = __as_bf16x2(dwords[woff_96 + 3]);
        __nv_bfloat162 mixed_181 = a_179 + d_180;
        sw_169[3] = __as_u32(mixed_181);
        words[112 + woff_96 + 3] = sw_169[3];
        {
            int4 _iv4 = make_int4(sw_169[0 + 0], sw_169[0 + 1], sw_169[0 + 2], sw_169[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_95)) + 0) = _iv4;
        }
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(sw_169[_pair]));
        }
        float2 v_182 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_183 = make_float2(q[8], q[9]);
        sq[8] = fma_f32x2_rn_noftz(v_182, v_182, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_182, qp_183, dot[8]);
        float2 v_184 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_185 = make_float2(q[10], q[11]);
        sq[8] = fma_f32x2_rn_noftz(v_184, v_184, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_184, qp_185, dot[8]);
        float2 v_186 = make_float2(_unpack_bf16_17[4], _unpack_bf16_17[5]);
        float2 qp_187 = make_float2(q[12], q[13]);
        sq[8] = fma_f32x2_rn_noftz(v_186, v_186, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_186, qp_187, dot[8]);
        float2 v_188 = make_float2(_unpack_bf16_17[6], _unpack_bf16_17[7]);
        float2 qp_189 = make_float2(q[14], q[15]);
        sq[8] = fma_f32x2_rn_noftz(v_188, v_188, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_188, qp_189, dot[8]);
        int base_190 = (4 + group) * 1024 + thread * 8;
        const int woff_191 = 8;
        unsigned int sw_192[4];
        sw_192[0] = words[woff_191];
        sw_192[1] = words[woff_191 + 1];
        sw_192[2] = words[woff_191 + 2];
        sw_192[3] = words[woff_191 + 3];
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
        fsrc[16] = _unpack_bf16_18[0];
        fsrc[17] = _unpack_bf16_18[1];
        fsrc[18] = _unpack_bf16_18[2];
        fsrc[19] = _unpack_bf16_18[3];
        fsrc[20] = _unpack_bf16_18[4];
        fsrc[21] = _unpack_bf16_18[5];
        fsrc[22] = _unpack_bf16_18[6];
        fsrc[23] = _unpack_bf16_18[7];
        float2 v_193 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_194 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_193, v_193, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_193, qp_194, dot[0]);
        float2 v_195 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_196 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_195, v_195, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_195, qp_196, dot[0]);
        float2 v_197 = make_float2(_unpack_bf16_18[4], _unpack_bf16_18[5]);
        float2 qp_198 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_197, v_197, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_197, qp_198, dot[0]);
        float2 v_199 = make_float2(_unpack_bf16_18[6], _unpack_bf16_18[7]);
        float2 qp_200 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_199, v_199, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_199, qp_200, dot[0]);
        unsigned int sw_201[4];
        sw_201[0] = words[14 + woff_191];
        sw_201[1] = words[14 + woff_191 + 1];
        sw_201[2] = words[14 + woff_191 + 2];
        sw_201[3] = words[14 + woff_191 + 3];
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
        fsrc[44] = _unpack_bf16_19[0];
        fsrc[45] = _unpack_bf16_19[1];
        fsrc[46] = _unpack_bf16_19[2];
        fsrc[47] = _unpack_bf16_19[3];
        fsrc[48] = _unpack_bf16_19[4];
        fsrc[49] = _unpack_bf16_19[5];
        fsrc[50] = _unpack_bf16_19[6];
        fsrc[51] = _unpack_bf16_19[7];
        float2 v_202 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 qp_203 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_202, v_202, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_202, qp_203, dot[1]);
        float2 v_204 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_205 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_204, v_204, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_204, qp_205, dot[1]);
        float2 v_206 = make_float2(_unpack_bf16_19[4], _unpack_bf16_19[5]);
        float2 qp_207 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_206, v_206, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_206, qp_207, dot[1]);
        float2 v_208 = make_float2(_unpack_bf16_19[6], _unpack_bf16_19[7]);
        float2 qp_209 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_208, v_208, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_208, qp_209, dot[1]);
        unsigned int sw_210[4];
        sw_210[0] = words[28 + woff_191];
        sw_210[1] = words[28 + woff_191 + 1];
        sw_210[2] = words[28 + woff_191 + 2];
        sw_210[3] = words[28 + woff_191 + 3];
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
        fsrc[72] = _unpack_bf16_20[0];
        fsrc[73] = _unpack_bf16_20[1];
        fsrc[74] = _unpack_bf16_20[2];
        fsrc[75] = _unpack_bf16_20[3];
        fsrc[76] = _unpack_bf16_20[4];
        fsrc[77] = _unpack_bf16_20[5];
        fsrc[78] = _unpack_bf16_20[6];
        fsrc[79] = _unpack_bf16_20[7];
        float2 v_211 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        float2 qp_212 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_211, v_211, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_211, qp_212, dot[2]);
        float2 v_213 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        float2 qp_214 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_213, v_213, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_213, qp_214, dot[2]);
        float2 v_215 = make_float2(_unpack_bf16_20[4], _unpack_bf16_20[5]);
        float2 qp_216 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_215, v_215, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_215, qp_216, dot[2]);
        float2 v_217 = make_float2(_unpack_bf16_20[6], _unpack_bf16_20[7]);
        float2 qp_218 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_217, v_217, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_217, qp_218, dot[2]);
        unsigned int sw_219[4];
        sw_219[0] = words[42 + woff_191];
        sw_219[1] = words[42 + woff_191 + 1];
        sw_219[2] = words[42 + woff_191 + 2];
        sw_219[3] = words[42 + woff_191 + 3];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw_219[_pair]));
        }
        float2 v_220 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        float2 qp_221 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_220, v_220, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_220, qp_221, dot[3]);
        float2 v_222 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        float2 qp_223 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_222, v_222, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_222, qp_223, dot[3]);
        float2 v_224 = make_float2(_unpack_bf16_21[4], _unpack_bf16_21[5]);
        float2 qp_225 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_224, v_224, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_224, qp_225, dot[3]);
        float2 v_226 = make_float2(_unpack_bf16_21[6], _unpack_bf16_21[7]);
        float2 qp_227 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_226, v_226, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_226, qp_227, dot[3]);
        unsigned int sw_228[4];
        sw_228[0] = words[56 + woff_191];
        sw_228[1] = words[56 + woff_191 + 1];
        sw_228[2] = words[56 + woff_191 + 2];
        sw_228[3] = words[56 + woff_191 + 3];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw_228[_pair]));
        }
        float2 v_229 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        float2 qp_230 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_229, v_229, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_229, qp_230, dot[4]);
        float2 v_231 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        float2 qp_232 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_231, v_231, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_231, qp_232, dot[4]);
        float2 v_233 = make_float2(_unpack_bf16_22[4], _unpack_bf16_22[5]);
        float2 qp_234 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_233, v_233, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_233, qp_234, dot[4]);
        float2 v_235 = make_float2(_unpack_bf16_22[6], _unpack_bf16_22[7]);
        float2 qp_236 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_235, v_235, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_235, qp_236, dot[4]);
        unsigned int sw_237[4];
        sw_237[0] = words[70 + woff_191];
        sw_237[1] = words[70 + woff_191 + 1];
        sw_237[2] = words[70 + woff_191 + 2];
        sw_237[3] = words[70 + woff_191 + 3];
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                : "r"(sw_237[_pair]));
        }
        float2 v_238 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        float2 qp_239 = make_float2(q[16], q[17]);
        sq[5] = fma_f32x2_rn_noftz(v_238, v_238, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_238, qp_239, dot[5]);
        float2 v_240 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        float2 qp_241 = make_float2(q[18], q[19]);
        sq[5] = fma_f32x2_rn_noftz(v_240, v_240, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_240, qp_241, dot[5]);
        float2 v_242 = make_float2(_unpack_bf16_23[4], _unpack_bf16_23[5]);
        float2 qp_243 = make_float2(q[20], q[21]);
        sq[5] = fma_f32x2_rn_noftz(v_242, v_242, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_242, qp_243, dot[5]);
        float2 v_244 = make_float2(_unpack_bf16_23[6], _unpack_bf16_23[7]);
        float2 qp_245 = make_float2(q[22], q[23]);
        sq[5] = fma_f32x2_rn_noftz(v_244, v_244, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_244, qp_245, dot[5]);
        unsigned int sw_246[4];
        sw_246[0] = words[84 + woff_191];
        sw_246[1] = words[84 + woff_191 + 1];
        sw_246[2] = words[84 + woff_191 + 2];
        sw_246[3] = words[84 + woff_191 + 3];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw_246[_pair]));
        }
        float2 v_247 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        float2 qp_248 = make_float2(q[16], q[17]);
        sq[6] = fma_f32x2_rn_noftz(v_247, v_247, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_247, qp_248, dot[6]);
        float2 v_249 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        float2 qp_250 = make_float2(q[18], q[19]);
        sq[6] = fma_f32x2_rn_noftz(v_249, v_249, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_249, qp_250, dot[6]);
        float2 v_251 = make_float2(_unpack_bf16_24[4], _unpack_bf16_24[5]);
        float2 qp_252 = make_float2(q[20], q[21]);
        sq[6] = fma_f32x2_rn_noftz(v_251, v_251, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_251, qp_252, dot[6]);
        float2 v_253 = make_float2(_unpack_bf16_24[6], _unpack_bf16_24[7]);
        float2 qp_254 = make_float2(q[22], q[23]);
        sq[6] = fma_f32x2_rn_noftz(v_253, v_253, sq[6]);
        dot[6] = fma_f32x2_rn_noftz(v_253, qp_254, dot[6]);
        unsigned int sw_255[4];
        sw_255[0] = words[98 + woff_191];
        sw_255[1] = words[98 + woff_191 + 1];
        sw_255[2] = words[98 + woff_191 + 2];
        sw_255[3] = words[98 + woff_191 + 3];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw_255[_pair]));
        }
        float2 v_256 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        float2 qp_257 = make_float2(q[16], q[17]);
        sq[7] = fma_f32x2_rn_noftz(v_256, v_256, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_256, qp_257, dot[7]);
        float2 v_258 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        float2 qp_259 = make_float2(q[18], q[19]);
        sq[7] = fma_f32x2_rn_noftz(v_258, v_258, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_258, qp_259, dot[7]);
        float2 v_260 = make_float2(_unpack_bf16_25[4], _unpack_bf16_25[5]);
        float2 qp_261 = make_float2(q[20], q[21]);
        sq[7] = fma_f32x2_rn_noftz(v_260, v_260, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_260, qp_261, dot[7]);
        float2 v_262 = make_float2(_unpack_bf16_25[6], _unpack_bf16_25[7]);
        float2 qp_263 = make_float2(q[22], q[23]);
        sq[7] = fma_f32x2_rn_noftz(v_262, v_262, sq[7]);
        dot[7] = fma_f32x2_rn_noftz(v_262, qp_263, dot[7]);
        unsigned int sw_264[4];
        sw_264[0] = words[112 + woff_191];
        sw_264[1] = words[112 + woff_191 + 1];
        sw_264[2] = words[112 + woff_191 + 2];
        sw_264[3] = words[112 + woff_191 + 3];
        __nv_bfloat162 a_265 = __as_bf16x2(sw_264[0]);
        __nv_bfloat162 d_266 = __as_bf16x2(dwords[woff_191]);
        __nv_bfloat162 mixed_267 = a_265 + d_266;
        sw_264[0] = __as_u32(mixed_267);
        words[112 + woff_191] = sw_264[0];
        __nv_bfloat162 a_268 = __as_bf16x2(sw_264[1]);
        __nv_bfloat162 d_269 = __as_bf16x2(dwords[woff_191 + 1]);
        __nv_bfloat162 mixed_270 = a_268 + d_269;
        sw_264[1] = __as_u32(mixed_270);
        words[112 + woff_191 + 1] = sw_264[1];
        __nv_bfloat162 a_271 = __as_bf16x2(sw_264[2]);
        __nv_bfloat162 d_272 = __as_bf16x2(dwords[woff_191 + 2]);
        __nv_bfloat162 mixed_273 = a_271 + d_272;
        sw_264[2] = __as_u32(mixed_273);
        words[112 + woff_191 + 2] = sw_264[2];
        __nv_bfloat162 a_274 = __as_bf16x2(sw_264[3]);
        __nv_bfloat162 d_275 = __as_bf16x2(dwords[woff_191 + 3]);
        __nv_bfloat162 mixed_276 = a_274 + d_275;
        sw_264[3] = __as_u32(mixed_276);
        words[112 + woff_191 + 3] = sw_264[3];
        {
            int4 _iv4 = make_int4(sw_264[0 + 0], sw_264[0 + 1], sw_264[0 + 2], sw_264[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_190)) + 0) = _iv4;
        }
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw_264[_pair]));
        }
        float2 v_277 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        float2 qp_278 = make_float2(q[16], q[17]);
        sq[8] = fma_f32x2_rn_noftz(v_277, v_277, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_277, qp_278, dot[8]);
        float2 v_279 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        float2 qp_280 = make_float2(q[18], q[19]);
        sq[8] = fma_f32x2_rn_noftz(v_279, v_279, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_279, qp_280, dot[8]);
        float2 v_281 = make_float2(_unpack_bf16_26[4], _unpack_bf16_26[5]);
        float2 qp_282 = make_float2(q[20], q[21]);
        sq[8] = fma_f32x2_rn_noftz(v_281, v_281, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_281, qp_282, dot[8]);
        float2 v_283 = make_float2(_unpack_bf16_26[6], _unpack_bf16_26[7]);
        float2 qp_284 = make_float2(q[22], q[23]);
        sq[8] = fma_f32x2_rn_noftz(v_283, v_283, sq[8]);
        dot[8] = fma_f32x2_rn_noftz(v_283, qp_284, dot[8]);
        int base_285 = 6144 + group * 512 + thread * 4;
        const int woff_286 = 12;
        unsigned int sw_287[4];
        sw_287[0] = words[woff_286];
        sw_287[1] = words[woff_286 + 1];
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw_287[_pair]));
        }
        fsrc[24] = _unpack_bf16_27[0];
        fsrc[25] = _unpack_bf16_27[1];
        fsrc[26] = _unpack_bf16_27[2];
        fsrc[27] = _unpack_bf16_27[3];
        float2 v_288 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        sq[0] = fma_f32x2_rn_noftz(v_288, v_288, sq[0]);
        float2 v_289 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        sq[0] = fma_f32x2_rn_noftz(v_289, v_289, sq[0]);
        float2 v_290 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        float2 qp_291 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_290, qp_291, dot[0]);
        float2 v_292 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        float2 qp_293 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_292, qp_293, dot[0]);
        unsigned int sw_294[4];
        sw_294[0] = words[14 + woff_286];
        sw_294[1] = words[14 + woff_286 + 1];
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                : "r"(sw_294[_pair]));
        }
        fsrc[52] = _unpack_bf16_28[0];
        fsrc[53] = _unpack_bf16_28[1];
        fsrc[54] = _unpack_bf16_28[2];
        fsrc[55] = _unpack_bf16_28[3];
        float2 v_295 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        sq[1] = fma_f32x2_rn_noftz(v_295, v_295, sq[1]);
        float2 v_296 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        sq[1] = fma_f32x2_rn_noftz(v_296, v_296, sq[1]);
        float2 v_297 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        float2 qp_298 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_297, qp_298, dot[1]);
        float2 v_299 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        float2 qp_300 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_299, qp_300, dot[1]);
        unsigned int sw_301[4];
        sw_301[0] = words[28 + woff_286];
        sw_301[1] = words[28 + woff_286 + 1];
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                : "r"(sw_301[_pair]));
        }
        fsrc[80] = _unpack_bf16_29[0];
        fsrc[81] = _unpack_bf16_29[1];
        fsrc[82] = _unpack_bf16_29[2];
        fsrc[83] = _unpack_bf16_29[3];
        float2 v_302 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        sq[2] = fma_f32x2_rn_noftz(v_302, v_302, sq[2]);
        float2 v_303 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        sq[2] = fma_f32x2_rn_noftz(v_303, v_303, sq[2]);
        float2 v_304 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        float2 qp_305 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_304, qp_305, dot[2]);
        float2 v_306 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        float2 qp_307 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_306, qp_307, dot[2]);
        unsigned int sw_308[4];
        sw_308[0] = words[42 + woff_286];
        sw_308[1] = words[42 + woff_286 + 1];
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                : "r"(sw_308[_pair]));
        }
        float2 v_309 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        sq[3] = fma_f32x2_rn_noftz(v_309, v_309, sq[3]);
        float2 v_310 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        sq[3] = fma_f32x2_rn_noftz(v_310, v_310, sq[3]);
        float2 v_311 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        float2 qp_312 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_311, qp_312, dot[3]);
        float2 v_313 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        float2 qp_314 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_313, qp_314, dot[3]);
        unsigned int sw_315[4];
        sw_315[0] = words[56 + woff_286];
        sw_315[1] = words[56 + woff_286 + 1];
        float _unpack_bf16_31[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                : "r"(sw_315[_pair]));
        }
        float2 v_316 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        sq[4] = fma_f32x2_rn_noftz(v_316, v_316, sq[4]);
        float2 v_317 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        sq[4] = fma_f32x2_rn_noftz(v_317, v_317, sq[4]);
        float2 v_318 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        float2 qp_319 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_318, qp_319, dot[4]);
        float2 v_320 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        float2 qp_321 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_320, qp_321, dot[4]);
        unsigned int sw_322[4];
        sw_322[0] = words[70 + woff_286];
        sw_322[1] = words[70 + woff_286 + 1];
        float _unpack_bf16_32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                : "r"(sw_322[_pair]));
        }
        float2 v_323 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
        sq[5] = fma_f32x2_rn_noftz(v_323, v_323, sq[5]);
        float2 v_324 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
        sq[5] = fma_f32x2_rn_noftz(v_324, v_324, sq[5]);
        float2 v_325 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
        float2 qp_326 = make_float2(q[24], q[25]);
        dot[5] = fma_f32x2_rn_noftz(v_325, qp_326, dot[5]);
        float2 v_327 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
        float2 qp_328 = make_float2(q[26], q[27]);
        dot[5] = fma_f32x2_rn_noftz(v_327, qp_328, dot[5]);
        unsigned int sw_329[4];
        sw_329[0] = words[84 + woff_286];
        sw_329[1] = words[84 + woff_286 + 1];
        float _unpack_bf16_33[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                : "r"(sw_329[_pair]));
        }
        float2 v_330 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
        sq[6] = fma_f32x2_rn_noftz(v_330, v_330, sq[6]);
        float2 v_331 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
        sq[6] = fma_f32x2_rn_noftz(v_331, v_331, sq[6]);
        float2 v_332 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
        float2 qp_333 = make_float2(q[24], q[25]);
        dot[6] = fma_f32x2_rn_noftz(v_332, qp_333, dot[6]);
        float2 v_334 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
        float2 qp_335 = make_float2(q[26], q[27]);
        dot[6] = fma_f32x2_rn_noftz(v_334, qp_335, dot[6]);
        unsigned int sw_336[4];
        sw_336[0] = words[98 + woff_286];
        sw_336[1] = words[98 + woff_286 + 1];
        float _unpack_bf16_34[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                : "r"(sw_336[_pair]));
        }
        float2 v_337 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
        sq[7] = fma_f32x2_rn_noftz(v_337, v_337, sq[7]);
        float2 v_338 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
        sq[7] = fma_f32x2_rn_noftz(v_338, v_338, sq[7]);
        float2 v_339 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
        float2 qp_340 = make_float2(q[24], q[25]);
        dot[7] = fma_f32x2_rn_noftz(v_339, qp_340, dot[7]);
        float2 v_341 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
        float2 qp_342 = make_float2(q[26], q[27]);
        dot[7] = fma_f32x2_rn_noftz(v_341, qp_342, dot[7]);
        unsigned int sw_343[4];
        sw_343[0] = words[112 + woff_286];
        sw_343[1] = words[112 + woff_286 + 1];
        __nv_bfloat162 a_344 = __as_bf16x2(sw_343[0]);
        __nv_bfloat162 d_345 = __as_bf16x2(dwords[woff_286]);
        __nv_bfloat162 mixed_346 = a_344 + d_345;
        sw_343[0] = __as_u32(mixed_346);
        words[112 + woff_286] = sw_343[0];
        __nv_bfloat162 a_347 = __as_bf16x2(sw_343[1]);
        __nv_bfloat162 d_348 = __as_bf16x2(dwords[woff_286 + 1]);
        __nv_bfloat162 mixed_349 = a_347 + d_348;
        sw_343[1] = __as_u32(mixed_349);
        words[112 + woff_286 + 1] = sw_343[1];
        {
            int2 _iv2 = make_int2(sw_343[0 + 0], sw_343[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_285)) + 0) = _iv2;
        }
        float _unpack_bf16_35[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                : "r"(sw_343[_pair]));
        }
        float2 v_350 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
        sq[8] = fma_f32x2_rn_noftz(v_350, v_350, sq[8]);
        float2 v_351 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
        sq[8] = fma_f32x2_rn_noftz(v_351, v_351, sq[8]);
        float2 v_352 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
        float2 qp_353 = make_float2(q[24], q[25]);
        dot[8] = fma_f32x2_rn_noftz(v_352, qp_353, dot[8]);
        float2 v_354 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
        float2 qp_355 = make_float2(q[26], q[27]);
        dot[8] = fma_f32x2_rn_noftz(v_354, qp_355, dot[8]);
        float2 pairs[9];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[2] = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[3] = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[4] = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[5] = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        pairs[6] = make_float2(sq[6].x + sq[6].y, dot[6].x + dot[6].y);
        pairs[7] = make_float2(sq[7].x + sq[7].y, dot[7].x + dot[7].y);
        pairs[8] = make_float2(sq[8].x + sq[8].y, dot[8].x + dot[8].y);
        unsigned long long bits = 0;
        float2 _bit_cast_src_42 = pairs[0];
        bits = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        float2 peer = make_float2(0.0f, 0.0f);
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_356 = 0;
        float2 _bit_cast_src_43 = pairs[1];
        bits_356 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_357 = __shfl_xor_sync(0xFFFFFFFF, bits_356, 16);
        float2 peer_358 = make_float2(0.0f, 0.0f);
        peer_358 = reinterpret_cast<float2*>(&peerbits_357)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_358);
        unsigned long long bits_359 = 0;
        float2 _bit_cast_src_44 = pairs[2];
        bits_359 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_360 = __shfl_xor_sync(0xFFFFFFFF, bits_359, 16);
        float2 peer_361 = make_float2(0.0f, 0.0f);
        peer_361 = reinterpret_cast<float2*>(&peerbits_360)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_361);
        unsigned long long bits_362 = 0;
        float2 _bit_cast_src_45 = pairs[3];
        bits_362 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_363 = __shfl_xor_sync(0xFFFFFFFF, bits_362, 16);
        float2 peer_364 = make_float2(0.0f, 0.0f);
        peer_364 = reinterpret_cast<float2*>(&peerbits_363)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_364);
        unsigned long long bits_365 = 0;
        float2 _bit_cast_src_46 = pairs[4];
        bits_365 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_366 = __shfl_xor_sync(0xFFFFFFFF, bits_365, 16);
        float2 peer_367 = make_float2(0.0f, 0.0f);
        peer_367 = reinterpret_cast<float2*>(&peerbits_366)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_367);
        unsigned long long bits_368 = 0;
        float2 _bit_cast_src_47 = pairs[5];
        bits_368 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_369 = __shfl_xor_sync(0xFFFFFFFF, bits_368, 16);
        float2 peer_370 = make_float2(0.0f, 0.0f);
        peer_370 = reinterpret_cast<float2*>(&peerbits_369)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_370);
        unsigned long long bits_371 = 0;
        float2 _bit_cast_src_48 = pairs[6];
        bits_371 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_372 = __shfl_xor_sync(0xFFFFFFFF, bits_371, 16);
        float2 peer_373 = make_float2(0.0f, 0.0f);
        peer_373 = reinterpret_cast<float2*>(&peerbits_372)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_373);
        unsigned long long bits_374 = 0;
        float2 _bit_cast_src_49 = pairs[7];
        bits_374 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_375 = __shfl_xor_sync(0xFFFFFFFF, bits_374, 16);
        float2 peer_376 = make_float2(0.0f, 0.0f);
        peer_376 = reinterpret_cast<float2*>(&peerbits_375)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_376);
        unsigned long long bits_377 = 0;
        float2 _bit_cast_src_50 = pairs[8];
        bits_377 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_378 = __shfl_xor_sync(0xFFFFFFFF, bits_377, 16);
        float2 peer_379 = make_float2(0.0f, 0.0f);
        peer_379 = reinterpret_cast<float2*>(&peerbits_378)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_379);
        unsigned long long bits_380 = 0;
        float2 _bit_cast_src_51 = pairs[0];
        bits_380 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_381 = __shfl_xor_sync(0xFFFFFFFF, bits_380, 8);
        float2 peer_382 = make_float2(0.0f, 0.0f);
        peer_382 = reinterpret_cast<float2*>(&peerbits_381)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_382);
        unsigned long long bits_383 = 0;
        float2 _bit_cast_src_52 = pairs[1];
        bits_383 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_384 = __shfl_xor_sync(0xFFFFFFFF, bits_383, 8);
        float2 peer_385 = make_float2(0.0f, 0.0f);
        peer_385 = reinterpret_cast<float2*>(&peerbits_384)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_385);
        unsigned long long bits_386 = 0;
        float2 _bit_cast_src_53 = pairs[2];
        bits_386 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_387 = __shfl_xor_sync(0xFFFFFFFF, bits_386, 8);
        float2 peer_388 = make_float2(0.0f, 0.0f);
        peer_388 = reinterpret_cast<float2*>(&peerbits_387)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_388);
        unsigned long long bits_389 = 0;
        float2 _bit_cast_src_54 = pairs[3];
        bits_389 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_390 = __shfl_xor_sync(0xFFFFFFFF, bits_389, 8);
        float2 peer_391 = make_float2(0.0f, 0.0f);
        peer_391 = reinterpret_cast<float2*>(&peerbits_390)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_391);
        unsigned long long bits_392 = 0;
        float2 _bit_cast_src_55 = pairs[4];
        bits_392 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_55)[0];
        unsigned long long peerbits_393 = __shfl_xor_sync(0xFFFFFFFF, bits_392, 8);
        float2 peer_394 = make_float2(0.0f, 0.0f);
        peer_394 = reinterpret_cast<float2*>(&peerbits_393)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_394);
        unsigned long long bits_395 = 0;
        float2 _bit_cast_src_56 = pairs[5];
        bits_395 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_56)[0];
        unsigned long long peerbits_396 = __shfl_xor_sync(0xFFFFFFFF, bits_395, 8);
        float2 peer_397 = make_float2(0.0f, 0.0f);
        peer_397 = reinterpret_cast<float2*>(&peerbits_396)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_397);
        unsigned long long bits_398 = 0;
        float2 _bit_cast_src_57 = pairs[6];
        bits_398 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_57)[0];
        unsigned long long peerbits_399 = __shfl_xor_sync(0xFFFFFFFF, bits_398, 8);
        float2 peer_400 = make_float2(0.0f, 0.0f);
        peer_400 = reinterpret_cast<float2*>(&peerbits_399)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_400);
        unsigned long long bits_401 = 0;
        float2 _bit_cast_src_58 = pairs[7];
        bits_401 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_58)[0];
        unsigned long long peerbits_402 = __shfl_xor_sync(0xFFFFFFFF, bits_401, 8);
        float2 peer_403 = make_float2(0.0f, 0.0f);
        peer_403 = reinterpret_cast<float2*>(&peerbits_402)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_403);
        unsigned long long bits_404 = 0;
        float2 _bit_cast_src_59 = pairs[8];
        bits_404 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_59)[0];
        unsigned long long peerbits_405 = __shfl_xor_sync(0xFFFFFFFF, bits_404, 8);
        float2 peer_406 = make_float2(0.0f, 0.0f);
        peer_406 = reinterpret_cast<float2*>(&peerbits_405)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_406);
        unsigned long long bits_407 = 0;
        float2 _bit_cast_src_60 = pairs[0];
        bits_407 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_60)[0];
        unsigned long long peerbits_408 = __shfl_xor_sync(0xFFFFFFFF, bits_407, 4);
        float2 peer_409 = make_float2(0.0f, 0.0f);
        peer_409 = reinterpret_cast<float2*>(&peerbits_408)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_409);
        unsigned long long bits_410 = 0;
        float2 _bit_cast_src_61 = pairs[1];
        bits_410 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_61)[0];
        unsigned long long peerbits_411 = __shfl_xor_sync(0xFFFFFFFF, bits_410, 4);
        float2 peer_412 = make_float2(0.0f, 0.0f);
        peer_412 = reinterpret_cast<float2*>(&peerbits_411)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_412);
        unsigned long long bits_413 = 0;
        float2 _bit_cast_src_62 = pairs[2];
        bits_413 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_62)[0];
        unsigned long long peerbits_414 = __shfl_xor_sync(0xFFFFFFFF, bits_413, 4);
        float2 peer_415 = make_float2(0.0f, 0.0f);
        peer_415 = reinterpret_cast<float2*>(&peerbits_414)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_415);
        unsigned long long bits_416 = 0;
        float2 _bit_cast_src_63 = pairs[3];
        bits_416 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_63)[0];
        unsigned long long peerbits_417 = __shfl_xor_sync(0xFFFFFFFF, bits_416, 4);
        float2 peer_418 = make_float2(0.0f, 0.0f);
        peer_418 = reinterpret_cast<float2*>(&peerbits_417)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_418);
        unsigned long long bits_419 = 0;
        float2 _bit_cast_src_64 = pairs[4];
        bits_419 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_64)[0];
        unsigned long long peerbits_420 = __shfl_xor_sync(0xFFFFFFFF, bits_419, 4);
        float2 peer_421 = make_float2(0.0f, 0.0f);
        peer_421 = reinterpret_cast<float2*>(&peerbits_420)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_421);
        unsigned long long bits_422 = 0;
        float2 _bit_cast_src_65 = pairs[5];
        bits_422 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_65)[0];
        unsigned long long peerbits_423 = __shfl_xor_sync(0xFFFFFFFF, bits_422, 4);
        float2 peer_424 = make_float2(0.0f, 0.0f);
        peer_424 = reinterpret_cast<float2*>(&peerbits_423)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_424);
        unsigned long long bits_425 = 0;
        float2 _bit_cast_src_66 = pairs[6];
        bits_425 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_66)[0];
        unsigned long long peerbits_426 = __shfl_xor_sync(0xFFFFFFFF, bits_425, 4);
        float2 peer_427 = make_float2(0.0f, 0.0f);
        peer_427 = reinterpret_cast<float2*>(&peerbits_426)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_427);
        unsigned long long bits_428 = 0;
        float2 _bit_cast_src_67 = pairs[7];
        bits_428 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_67)[0];
        unsigned long long peerbits_429 = __shfl_xor_sync(0xFFFFFFFF, bits_428, 4);
        float2 peer_430 = make_float2(0.0f, 0.0f);
        peer_430 = reinterpret_cast<float2*>(&peerbits_429)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_430);
        unsigned long long bits_431 = 0;
        float2 _bit_cast_src_68 = pairs[8];
        bits_431 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_68)[0];
        unsigned long long peerbits_432 = __shfl_xor_sync(0xFFFFFFFF, bits_431, 4);
        float2 peer_433 = make_float2(0.0f, 0.0f);
        peer_433 = reinterpret_cast<float2*>(&peerbits_432)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_433);
        unsigned long long bits_434 = 0;
        float2 _bit_cast_src_69 = pairs[0];
        bits_434 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_69)[0];
        unsigned long long peerbits_435 = __shfl_xor_sync(0xFFFFFFFF, bits_434, 2);
        float2 peer_436 = make_float2(0.0f, 0.0f);
        peer_436 = reinterpret_cast<float2*>(&peerbits_435)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_436);
        unsigned long long bits_437 = 0;
        float2 _bit_cast_src_70 = pairs[1];
        bits_437 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_70)[0];
        unsigned long long peerbits_438 = __shfl_xor_sync(0xFFFFFFFF, bits_437, 2);
        float2 peer_439 = make_float2(0.0f, 0.0f);
        peer_439 = reinterpret_cast<float2*>(&peerbits_438)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_439);
        unsigned long long bits_440 = 0;
        float2 _bit_cast_src_71 = pairs[2];
        bits_440 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_71)[0];
        unsigned long long peerbits_441 = __shfl_xor_sync(0xFFFFFFFF, bits_440, 2);
        float2 peer_442 = make_float2(0.0f, 0.0f);
        peer_442 = reinterpret_cast<float2*>(&peerbits_441)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_442);
        unsigned long long bits_443 = 0;
        float2 _bit_cast_src_72 = pairs[3];
        bits_443 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_72)[0];
        unsigned long long peerbits_444 = __shfl_xor_sync(0xFFFFFFFF, bits_443, 2);
        float2 peer_445 = make_float2(0.0f, 0.0f);
        peer_445 = reinterpret_cast<float2*>(&peerbits_444)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_445);
        unsigned long long bits_446 = 0;
        float2 _bit_cast_src_73 = pairs[4];
        bits_446 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_73)[0];
        unsigned long long peerbits_447 = __shfl_xor_sync(0xFFFFFFFF, bits_446, 2);
        float2 peer_448 = make_float2(0.0f, 0.0f);
        peer_448 = reinterpret_cast<float2*>(&peerbits_447)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_448);
        unsigned long long bits_449 = 0;
        float2 _bit_cast_src_74 = pairs[5];
        bits_449 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_74)[0];
        unsigned long long peerbits_450 = __shfl_xor_sync(0xFFFFFFFF, bits_449, 2);
        float2 peer_451 = make_float2(0.0f, 0.0f);
        peer_451 = reinterpret_cast<float2*>(&peerbits_450)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_451);
        unsigned long long bits_452 = 0;
        float2 _bit_cast_src_75 = pairs[6];
        bits_452 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_75)[0];
        unsigned long long peerbits_453 = __shfl_xor_sync(0xFFFFFFFF, bits_452, 2);
        float2 peer_454 = make_float2(0.0f, 0.0f);
        peer_454 = reinterpret_cast<float2*>(&peerbits_453)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_454);
        unsigned long long bits_455 = 0;
        float2 _bit_cast_src_76 = pairs[7];
        bits_455 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_76)[0];
        unsigned long long peerbits_456 = __shfl_xor_sync(0xFFFFFFFF, bits_455, 2);
        float2 peer_457 = make_float2(0.0f, 0.0f);
        peer_457 = reinterpret_cast<float2*>(&peerbits_456)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_457);
        unsigned long long bits_458 = 0;
        float2 _bit_cast_src_77 = pairs[8];
        bits_458 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_77)[0];
        unsigned long long peerbits_459 = __shfl_xor_sync(0xFFFFFFFF, bits_458, 2);
        float2 peer_460 = make_float2(0.0f, 0.0f);
        peer_460 = reinterpret_cast<float2*>(&peerbits_459)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_460);
        unsigned long long bits_461 = 0;
        float2 _bit_cast_src_78 = pairs[0];
        bits_461 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_78)[0];
        unsigned long long peerbits_462 = __shfl_xor_sync(0xFFFFFFFF, bits_461, 1);
        float2 peer_463 = make_float2(0.0f, 0.0f);
        peer_463 = reinterpret_cast<float2*>(&peerbits_462)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_463);
        unsigned long long bits_464 = 0;
        float2 _bit_cast_src_79 = pairs[1];
        bits_464 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_79)[0];
        unsigned long long peerbits_465 = __shfl_xor_sync(0xFFFFFFFF, bits_464, 1);
        float2 peer_466 = make_float2(0.0f, 0.0f);
        peer_466 = reinterpret_cast<float2*>(&peerbits_465)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_466);
        unsigned long long bits_467 = 0;
        float2 _bit_cast_src_80 = pairs[2];
        bits_467 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_80)[0];
        unsigned long long peerbits_468 = __shfl_xor_sync(0xFFFFFFFF, bits_467, 1);
        float2 peer_469 = make_float2(0.0f, 0.0f);
        peer_469 = reinterpret_cast<float2*>(&peerbits_468)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_469);
        unsigned long long bits_470 = 0;
        float2 _bit_cast_src_81 = pairs[3];
        bits_470 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_81)[0];
        unsigned long long peerbits_471 = __shfl_xor_sync(0xFFFFFFFF, bits_470, 1);
        float2 peer_472 = make_float2(0.0f, 0.0f);
        peer_472 = reinterpret_cast<float2*>(&peerbits_471)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_472);
        unsigned long long bits_473 = 0;
        float2 _bit_cast_src_82 = pairs[4];
        bits_473 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_82)[0];
        unsigned long long peerbits_474 = __shfl_xor_sync(0xFFFFFFFF, bits_473, 1);
        float2 peer_475 = make_float2(0.0f, 0.0f);
        peer_475 = reinterpret_cast<float2*>(&peerbits_474)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_475);
        unsigned long long bits_476 = 0;
        float2 _bit_cast_src_83 = pairs[5];
        bits_476 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_83)[0];
        unsigned long long peerbits_477 = __shfl_xor_sync(0xFFFFFFFF, bits_476, 1);
        float2 peer_478 = make_float2(0.0f, 0.0f);
        peer_478 = reinterpret_cast<float2*>(&peerbits_477)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_478);
        unsigned long long bits_479 = 0;
        float2 _bit_cast_src_84 = pairs[6];
        bits_479 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_84)[0];
        unsigned long long peerbits_480 = __shfl_xor_sync(0xFFFFFFFF, bits_479, 1);
        float2 peer_481 = make_float2(0.0f, 0.0f);
        peer_481 = reinterpret_cast<float2*>(&peerbits_480)[0];
        pairs[6] = add_f32x2_noftz(pairs[6], peer_481);
        unsigned long long bits_482 = 0;
        float2 _bit_cast_src_85 = pairs[7];
        bits_482 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_85)[0];
        unsigned long long peerbits_483 = __shfl_xor_sync(0xFFFFFFFF, bits_482, 1);
        float2 peer_484 = make_float2(0.0f, 0.0f);
        peer_484 = reinterpret_cast<float2*>(&peerbits_483)[0];
        pairs[7] = add_f32x2_noftz(pairs[7], peer_484);
        unsigned long long bits_485 = 0;
        float2 _bit_cast_src_86 = pairs[8];
        bits_485 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_86)[0];
        unsigned long long peerbits_486 = __shfl_xor_sync(0xFFFFFFFF, bits_485, 1);
        float2 peer_487 = make_float2(0.0f, 0.0f);
        peer_487 = reinterpret_cast<float2*>(&peerbits_486)[0];
        pairs[8] = add_f32x2_noftz(pairs[8], peer_487);
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (lane == 0) {
            uint32_t _cluster_mapa_87;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_87) : "r"(stats_addr + (unsigned int)(warp_0 * 9 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_87), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_88;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_88) : "r"(stats_addr + (unsigned int)((warp_0 * 9 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_88), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_89;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_89) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_89), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_90;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_90) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_90), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_91;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_91) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_91), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_92;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_92) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_92), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_93;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_93) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_93), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_94;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_94) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_94), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_95;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_95) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_95), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_96;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_96) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_96), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_97;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_97) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_97), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_98;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_98) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_98), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_99;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_99) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 6) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_99), "f"(pairs[6].x) : "memory");
            uint32_t _cluster_mapa_100;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_100) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 6) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_100), "f"(pairs[6].y) : "memory");
            uint32_t _cluster_mapa_101;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_101) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 7) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_101), "f"(pairs[7].x) : "memory");
            uint32_t _cluster_mapa_102;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_102) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 7) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_102), "f"(pairs[7].y) : "memory");
            uint32_t _cluster_mapa_103;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_103) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 8) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_103), "f"(pairs[8].x) : "memory");
            uint32_t _cluster_mapa_104;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_104) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 8) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_104), "f"(pairs[8].y) : "memory");
            uint32_t _cluster_mapa_105;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_105) : "r"(stats_addr + (unsigned int)(warp_0 * 9 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_105), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_106;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_106) : "r"(stats_addr + (unsigned int)((warp_0 * 9 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_106), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_107;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_107) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_107), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_108;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_108) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_108), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_109;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_109) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_109), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_110;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_110) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_110), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_111;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_111) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_111), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_112;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_112) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_112), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_113;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_113) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_113), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_114;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_114) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_114), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_115;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_115) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_115), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_116;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_116) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_116), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_117;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_117) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 6) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_117), "f"(pairs[6].x) : "memory");
            uint32_t _cluster_mapa_118;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_118) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 6) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_118), "f"(pairs[6].y) : "memory");
            uint32_t _cluster_mapa_119;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_119) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 7) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_119), "f"(pairs[7].x) : "memory");
            uint32_t _cluster_mapa_120;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_120) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 7) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_120), "f"(pairs[7].y) : "memory");
            uint32_t _cluster_mapa_121;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_121) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 8) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_121), "f"(pairs[8].x) : "memory");
            uint32_t _cluster_mapa_122;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_122) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 8) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_122), "f"(pairs[8].y) : "memory");
            uint32_t _cluster_mapa_123;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_123) : "r"(stats_addr + (unsigned int)(warp_0 * 9 * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_123), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_124;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_124) : "r"(stats_addr + (unsigned int)((warp_0 * 9 * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_124), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_125;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_125) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 1) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_125), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_126;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_126) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 1) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_126), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_127;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_127) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 2) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_127), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_128;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_128) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 2) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_128), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_129;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_129) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 3) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_129), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_130;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_130) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 3) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_130), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_131;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_131) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 4) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_131), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_132;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_132) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 4) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_132), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_133;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_133) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 5) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_133), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_134;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_134) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 5) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_134), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_135;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_135) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 6) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_135), "f"(pairs[6].x) : "memory");
            uint32_t _cluster_mapa_136;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_136) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 6) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_136), "f"(pairs[6].y) : "memory");
            uint32_t _cluster_mapa_137;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_137) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 7) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_137), "f"(pairs[7].x) : "memory");
            uint32_t _cluster_mapa_138;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_138) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 7) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_138), "f"(pairs[7].y) : "memory");
            uint32_t _cluster_mapa_139;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_139) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 8) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_139), "f"(pairs[8].x) : "memory");
            uint32_t _cluster_mapa_140;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_140) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 8) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_140), "f"(pairs[8].y) : "memory");
            uint32_t _cluster_mapa_141;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_141) : "r"(stats_addr + (unsigned int)(warp_0 * 9 * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_141), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_142;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_142) : "r"(stats_addr + (unsigned int)((warp_0 * 9 * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_142), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_143;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_143) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 1) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_143), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_144;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_144) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 1) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_144), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_145;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_145) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 2) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_145), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_146;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_146) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 2) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_146), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_147;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_147) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 3) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_147), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_148;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_148) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 3) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_148), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_149;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_149) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 4) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_149), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_150;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_150) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 4) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_150), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_151;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_151) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 5) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_151), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_152;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_152) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 5) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_152), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_153;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_153) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 6) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_153), "f"(pairs[6].x) : "memory");
            uint32_t _cluster_mapa_154;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_154) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 6) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_154), "f"(pairs[6].y) : "memory");
            uint32_t _cluster_mapa_155;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_155) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 7) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_155), "f"(pairs[7].x) : "memory");
            uint32_t _cluster_mapa_156;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_156) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 7) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_156), "f"(pairs[7].y) : "memory");
            uint32_t _cluster_mapa_157;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_157) : "r"(stats_addr + (unsigned int)((warp_0 * 9 + 8) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_157), "f"(pairs[8].x) : "memory");
            uint32_t _cluster_mapa_158;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_158) : "r"(stats_addr + (unsigned int)(((warp_0 * 9 + 8) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_158), "f"(pairs[8].y) : "memory");
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
        float _shfl_down_sync_159 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_159;
        float _shfl_down_sync_160 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_160;
        float _shfl_down_sync_161 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_161;
        float _shfl_down_sync_162 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_162;
        float _shfl_down_sync_163 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_163;
        float _shfl_down_sync_164 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_164;
        float logit = 0.0f;
        if (stat_n < 9 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[9];
        float _shfl_sync_165 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_165;
        float _shfl_sync_166 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_166;
        float _shfl_sync_167 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_167;
        float _shfl_sync_168 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_sync_168;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 4) {
            total_sq2 = stats[(stat_w * 9 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 9 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_169 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_169;
        float _shfl_down_sync_170 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_170;
        float _shfl_down_sync_171 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_171;
        float _shfl_down_sync_172 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_172;
        float _shfl_down_sync_173 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_173;
        float _shfl_down_sync_174 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_174;
        logit2 = 0.0f;
        if (stat_n < 4 && stat_w == 0) {
            float sigma2 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_sync_175 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_sync_175;
        float _shfl_sync_176 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_sync_176;
        float _shfl_sync_177 = __shfl_sync(0xFFFFFFFF, logit2, 16);
        logits[6] = _shfl_sync_177;
        float _shfl_sync_178 = __shfl_sync(0xFFFFFFFF, logit2, 24);
        logits[7] = _shfl_sync_178;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 1) {
            total_sq2 = stats[(stat_w * 9 + 8 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 9 + 8 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_179 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_179;
        float _shfl_down_sync_180 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_180;
        float _shfl_down_sync_181 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_181;
        float _shfl_down_sync_182 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_182;
        float _shfl_down_sync_183 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_183;
        float _shfl_down_sync_184 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_184;
        logit2 = 0.0f;
        if (stat_n < 1 && stat_w == 0) {
            float sigma2_1 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2_1;
        }
        float _shfl_sync_185 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[8] = _shfl_sync_185;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028235e+38f;
        max_chunk = fmaxf(max_chunk, logits[0]);
        max_chunk = fmaxf(max_chunk, logits[1]);
        max_chunk = fmaxf(max_chunk, logits[2]);
        float max_new = fmaxf(max_running, max_chunk);
        float correction = approx_exp2((max_running - max_new) * 1.442695f);
        float weights[9];
        float sum_weights = 0.0f;
        float _exp2_186 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_186;
        sum_weights += weights[0];
        float _exp2_187 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_187;
        sum_weights += weights[1];
        float _exp2_188 = approx_exp2((logits[2] - max_new) * 1.442695f);
        weights[2] = _exp2_188;
        sum_weights += weights[2];
        float2 corr = make_float2(correction, correction);
        const int woff_488 = 0;
        int base_489 = group * 1024 + thread * 8;
        float2 a_490[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_490[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_491 = make_float2(acc[2], acc[3]);
        a_490[1] = mul_f32x2_noftz(previous_491, corr);
        float2 previous_492 = make_float2(acc[4], acc[5]);
        a_490[2] = mul_f32x2_noftz(previous_492, corr);
        float2 previous_493 = make_float2(acc[6], acc[7]);
        a_490[3] = mul_f32x2_noftz(previous_493, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_494 = make_float2(fsrc[0], fsrc[1]);
        a_490[0] = fma_f32x2_rn_noftz(weight, v_494, a_490[0]);
        float2 v_495 = make_float2(fsrc[2], fsrc[3]);
        a_490[1] = fma_f32x2_rn_noftz(weight, v_495, a_490[1]);
        float2 v_496 = make_float2(fsrc[4], fsrc[5]);
        a_490[2] = fma_f32x2_rn_noftz(weight, v_496, a_490[2]);
        float2 v_497 = make_float2(fsrc[6], fsrc[7]);
        a_490[3] = fma_f32x2_rn_noftz(weight, v_497, a_490[3]);
        float2 weight_498 = make_float2(weights[1], weights[1]);
        float2 v_499 = make_float2(fsrc[28], fsrc[29]);
        a_490[0] = fma_f32x2_rn_noftz(weight_498, v_499, a_490[0]);
        float2 v_500 = make_float2(fsrc[30], fsrc[31]);
        a_490[1] = fma_f32x2_rn_noftz(weight_498, v_500, a_490[1]);
        float2 v_501 = make_float2(fsrc[32], fsrc[33]);
        a_490[2] = fma_f32x2_rn_noftz(weight_498, v_501, a_490[2]);
        float2 v_502 = make_float2(fsrc[34], fsrc[35]);
        a_490[3] = fma_f32x2_rn_noftz(weight_498, v_502, a_490[3]);
        float2 weight_503 = make_float2(weights[2], weights[2]);
        float2 v_504 = make_float2(fsrc[56], fsrc[57]);
        a_490[0] = fma_f32x2_rn_noftz(weight_503, v_504, a_490[0]);
        float2 v_505 = make_float2(fsrc[58], fsrc[59]);
        a_490[1] = fma_f32x2_rn_noftz(weight_503, v_505, a_490[1]);
        float2 v_506 = make_float2(fsrc[60], fsrc[61]);
        a_490[2] = fma_f32x2_rn_noftz(weight_503, v_506, a_490[2]);
        float2 v_507 = make_float2(fsrc[62], fsrc[63]);
        a_490[3] = fma_f32x2_rn_noftz(weight_503, v_507, a_490[3]);
        acc[0] = a_490[0].x;
        acc[1] = a_490[0].y;
        acc[2] = a_490[1].x;
        acc[3] = a_490[1].y;
        acc[4] = a_490[2].x;
        acc[5] = a_490[2].y;
        acc[6] = a_490[3].x;
        acc[7] = a_490[3].y;
        const int woff_508 = 4;
        int base_509 = (2 + group) * 1024 + thread * 8;
        float2 a_510[4];
        float2 previous_511 = make_float2(acc[8], acc[9]);
        a_510[0] = mul_f32x2_noftz(previous_511, corr);
        float2 previous_512 = make_float2(acc[10], acc[11]);
        a_510[1] = mul_f32x2_noftz(previous_512, corr);
        float2 previous_513 = make_float2(acc[12], acc[13]);
        a_510[2] = mul_f32x2_noftz(previous_513, corr);
        float2 previous_514 = make_float2(acc[14], acc[15]);
        a_510[3] = mul_f32x2_noftz(previous_514, corr);
        float2 weight_515 = make_float2(weights[0], weights[0]);
        float2 v_516 = make_float2(fsrc[8], fsrc[9]);
        a_510[0] = fma_f32x2_rn_noftz(weight_515, v_516, a_510[0]);
        float2 v_517 = make_float2(fsrc[10], fsrc[11]);
        a_510[1] = fma_f32x2_rn_noftz(weight_515, v_517, a_510[1]);
        float2 v_518 = make_float2(fsrc[12], fsrc[13]);
        a_510[2] = fma_f32x2_rn_noftz(weight_515, v_518, a_510[2]);
        float2 v_519 = make_float2(fsrc[14], fsrc[15]);
        a_510[3] = fma_f32x2_rn_noftz(weight_515, v_519, a_510[3]);
        float2 weight_520 = make_float2(weights[1], weights[1]);
        float2 v_521 = make_float2(fsrc[36], fsrc[37]);
        a_510[0] = fma_f32x2_rn_noftz(weight_520, v_521, a_510[0]);
        float2 v_522 = make_float2(fsrc[38], fsrc[39]);
        a_510[1] = fma_f32x2_rn_noftz(weight_520, v_522, a_510[1]);
        float2 v_523 = make_float2(fsrc[40], fsrc[41]);
        a_510[2] = fma_f32x2_rn_noftz(weight_520, v_523, a_510[2]);
        float2 v_524 = make_float2(fsrc[42], fsrc[43]);
        a_510[3] = fma_f32x2_rn_noftz(weight_520, v_524, a_510[3]);
        float2 weight_525 = make_float2(weights[2], weights[2]);
        float2 v_526 = make_float2(fsrc[64], fsrc[65]);
        a_510[0] = fma_f32x2_rn_noftz(weight_525, v_526, a_510[0]);
        float2 v_527 = make_float2(fsrc[66], fsrc[67]);
        a_510[1] = fma_f32x2_rn_noftz(weight_525, v_527, a_510[1]);
        float2 v_528 = make_float2(fsrc[68], fsrc[69]);
        a_510[2] = fma_f32x2_rn_noftz(weight_525, v_528, a_510[2]);
        float2 v_529 = make_float2(fsrc[70], fsrc[71]);
        a_510[3] = fma_f32x2_rn_noftz(weight_525, v_529, a_510[3]);
        acc[8] = a_510[0].x;
        acc[9] = a_510[0].y;
        acc[10] = a_510[1].x;
        acc[11] = a_510[1].y;
        acc[12] = a_510[2].x;
        acc[13] = a_510[2].y;
        acc[14] = a_510[3].x;
        acc[15] = a_510[3].y;
        const int woff_530 = 8;
        int base_531 = (4 + group) * 1024 + thread * 8;
        float2 a_532[4];
        float2 previous_533 = make_float2(acc[16], acc[17]);
        a_532[0] = mul_f32x2_noftz(previous_533, corr);
        float2 previous_534 = make_float2(acc[18], acc[19]);
        a_532[1] = mul_f32x2_noftz(previous_534, corr);
        float2 previous_535 = make_float2(acc[20], acc[21]);
        a_532[2] = mul_f32x2_noftz(previous_535, corr);
        float2 previous_536 = make_float2(acc[22], acc[23]);
        a_532[3] = mul_f32x2_noftz(previous_536, corr);
        float2 weight_537 = make_float2(weights[0], weights[0]);
        float2 v_538 = make_float2(fsrc[16], fsrc[17]);
        a_532[0] = fma_f32x2_rn_noftz(weight_537, v_538, a_532[0]);
        float2 v_539 = make_float2(fsrc[18], fsrc[19]);
        a_532[1] = fma_f32x2_rn_noftz(weight_537, v_539, a_532[1]);
        float2 v_540 = make_float2(fsrc[20], fsrc[21]);
        a_532[2] = fma_f32x2_rn_noftz(weight_537, v_540, a_532[2]);
        float2 v_541 = make_float2(fsrc[22], fsrc[23]);
        a_532[3] = fma_f32x2_rn_noftz(weight_537, v_541, a_532[3]);
        float2 weight_542 = make_float2(weights[1], weights[1]);
        float2 v_543 = make_float2(fsrc[44], fsrc[45]);
        a_532[0] = fma_f32x2_rn_noftz(weight_542, v_543, a_532[0]);
        float2 v_544 = make_float2(fsrc[46], fsrc[47]);
        a_532[1] = fma_f32x2_rn_noftz(weight_542, v_544, a_532[1]);
        float2 v_545 = make_float2(fsrc[48], fsrc[49]);
        a_532[2] = fma_f32x2_rn_noftz(weight_542, v_545, a_532[2]);
        float2 v_546 = make_float2(fsrc[50], fsrc[51]);
        a_532[3] = fma_f32x2_rn_noftz(weight_542, v_546, a_532[3]);
        float2 weight_547 = make_float2(weights[2], weights[2]);
        float2 v_548 = make_float2(fsrc[72], fsrc[73]);
        a_532[0] = fma_f32x2_rn_noftz(weight_547, v_548, a_532[0]);
        float2 v_549 = make_float2(fsrc[74], fsrc[75]);
        a_532[1] = fma_f32x2_rn_noftz(weight_547, v_549, a_532[1]);
        float2 v_550 = make_float2(fsrc[76], fsrc[77]);
        a_532[2] = fma_f32x2_rn_noftz(weight_547, v_550, a_532[2]);
        float2 v_551 = make_float2(fsrc[78], fsrc[79]);
        a_532[3] = fma_f32x2_rn_noftz(weight_547, v_551, a_532[3]);
        acc[16] = a_532[0].x;
        acc[17] = a_532[0].y;
        acc[18] = a_532[1].x;
        acc[19] = a_532[1].y;
        acc[20] = a_532[2].x;
        acc[21] = a_532[2].y;
        acc[22] = a_532[3].x;
        acc[23] = a_532[3].y;
        const int woff_552 = 12;
        int base_553 = 6144 + group * 512 + thread * 4;
        float2 a_554[4];
        float2 previous_555 = make_float2(acc[24], acc[25]);
        a_554[0] = mul_f32x2_noftz(previous_555, corr);
        float2 previous_556 = make_float2(acc[26], acc[27]);
        a_554[1] = mul_f32x2_noftz(previous_556, corr);
        float2 weight_557 = make_float2(weights[0], weights[0]);
        float2 v_558 = make_float2(fsrc[24], fsrc[25]);
        a_554[0] = fma_f32x2_rn_noftz(weight_557, v_558, a_554[0]);
        float2 v_559 = make_float2(fsrc[26], fsrc[27]);
        a_554[1] = fma_f32x2_rn_noftz(weight_557, v_559, a_554[1]);
        float2 weight_560 = make_float2(weights[1], weights[1]);
        float2 v_561 = make_float2(fsrc[52], fsrc[53]);
        a_554[0] = fma_f32x2_rn_noftz(weight_560, v_561, a_554[0]);
        float2 v_562 = make_float2(fsrc[54], fsrc[55]);
        a_554[1] = fma_f32x2_rn_noftz(weight_560, v_562, a_554[1]);
        float2 weight_563 = make_float2(weights[2], weights[2]);
        float2 v_564 = make_float2(fsrc[80], fsrc[81]);
        a_554[0] = fma_f32x2_rn_noftz(weight_563, v_564, a_554[0]);
        float2 v_565 = make_float2(fsrc[82], fsrc[83]);
        a_554[1] = fma_f32x2_rn_noftz(weight_563, v_565, a_554[1]);
        acc[24] = a_554[0].x;
        acc[25] = a_554[0].y;
        acc[26] = a_554[1].x;
        acc[27] = a_554[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_566 = -3.4028235e+38f;
        max_chunk_566 = fmaxf(max_chunk_566, logits[3]);
        max_chunk_566 = fmaxf(max_chunk_566, logits[4]);
        max_chunk_566 = fmaxf(max_chunk_566, logits[5]);
        float max_new_567 = fmaxf(max_running, max_chunk_566);
        float correction_568 = approx_exp2((max_running - max_new_567) * 1.442695f);
        float weights_569[9];
        float sum_weights_570 = 0.0f;
        float _exp2_189 = approx_exp2((logits[3] - max_new_567) * 1.442695f);
        weights_569[3] = _exp2_189;
        sum_weights_570 += weights_569[3];
        float _exp2_190 = approx_exp2((logits[4] - max_new_567) * 1.442695f);
        weights_569[4] = _exp2_190;
        sum_weights_570 += weights_569[4];
        float _exp2_191 = approx_exp2((logits[5] - max_new_567) * 1.442695f);
        weights_569[5] = _exp2_191;
        sum_weights_570 += weights_569[5];
        float2 corr_571 = make_float2(correction_568, correction_568);
        const int woff_572 = 0;
        int base_573 = group * 1024 + thread * 8;
        float2 a_574[4];
        float2 previous_575 = make_float2(acc[0], acc[1]);
        a_574[0] = mul_f32x2_noftz(previous_575, corr_571);
        float2 previous_576 = make_float2(acc[2], acc[3]);
        a_574[1] = mul_f32x2_noftz(previous_576, corr_571);
        float2 previous_577 = make_float2(acc[4], acc[5]);
        a_574[2] = mul_f32x2_noftz(previous_577, corr_571);
        float2 previous_578 = make_float2(acc[6], acc[7]);
        a_574[3] = mul_f32x2_noftz(previous_578, corr_571);
        float2 weight_579 = make_float2(weights_569[3], weights_569[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_572];
        sw2[1] = words[42 + woff_572 + 1];
        sw2[2] = words[42 + woff_572 + 2];
        sw2[3] = words[42 + woff_572 + 3];
        float _unpack_bf16_36[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_36[_pair * 2])[0]), "=f"((&_unpack_bf16_36[_pair * 2])[1])
                : "r"(sw2[_pair]));
        }
        float2 v_580 = make_float2(_unpack_bf16_36[0], _unpack_bf16_36[1]);
        a_574[0] = fma_f32x2_rn_noftz(weight_579, v_580, a_574[0]);
        float2 v_581 = make_float2(_unpack_bf16_36[2], _unpack_bf16_36[3]);
        a_574[1] = fma_f32x2_rn_noftz(weight_579, v_581, a_574[1]);
        float2 v_582 = make_float2(_unpack_bf16_36[4], _unpack_bf16_36[5]);
        a_574[2] = fma_f32x2_rn_noftz(weight_579, v_582, a_574[2]);
        float2 v_583 = make_float2(_unpack_bf16_36[6], _unpack_bf16_36[7]);
        a_574[3] = fma_f32x2_rn_noftz(weight_579, v_583, a_574[3]);
        float2 weight_584 = make_float2(weights_569[4], weights_569[4]);
        unsigned int sw2_585[4];
        sw2_585[0] = words[56 + woff_572];
        sw2_585[1] = words[56 + woff_572 + 1];
        sw2_585[2] = words[56 + woff_572 + 2];
        sw2_585[3] = words[56 + woff_572 + 3];
        float _unpack_bf16_37[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_37[_pair * 2])[0]), "=f"((&_unpack_bf16_37[_pair * 2])[1])
                : "r"(sw2_585[_pair]));
        }
        float2 v_586 = make_float2(_unpack_bf16_37[0], _unpack_bf16_37[1]);
        a_574[0] = fma_f32x2_rn_noftz(weight_584, v_586, a_574[0]);
        float2 v_587 = make_float2(_unpack_bf16_37[2], _unpack_bf16_37[3]);
        a_574[1] = fma_f32x2_rn_noftz(weight_584, v_587, a_574[1]);
        float2 v_588 = make_float2(_unpack_bf16_37[4], _unpack_bf16_37[5]);
        a_574[2] = fma_f32x2_rn_noftz(weight_584, v_588, a_574[2]);
        float2 v_589 = make_float2(_unpack_bf16_37[6], _unpack_bf16_37[7]);
        a_574[3] = fma_f32x2_rn_noftz(weight_584, v_589, a_574[3]);
        float2 weight_590 = make_float2(weights_569[5], weights_569[5]);
        unsigned int sw2_591[4];
        sw2_591[0] = words[70 + woff_572];
        sw2_591[1] = words[70 + woff_572 + 1];
        sw2_591[2] = words[70 + woff_572 + 2];
        sw2_591[3] = words[70 + woff_572 + 3];
        float _unpack_bf16_38[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_38[_pair * 2])[0]), "=f"((&_unpack_bf16_38[_pair * 2])[1])
                : "r"(sw2_591[_pair]));
        }
        float2 v_592 = make_float2(_unpack_bf16_38[0], _unpack_bf16_38[1]);
        a_574[0] = fma_f32x2_rn_noftz(weight_590, v_592, a_574[0]);
        float2 v_593 = make_float2(_unpack_bf16_38[2], _unpack_bf16_38[3]);
        a_574[1] = fma_f32x2_rn_noftz(weight_590, v_593, a_574[1]);
        float2 v_594 = make_float2(_unpack_bf16_38[4], _unpack_bf16_38[5]);
        a_574[2] = fma_f32x2_rn_noftz(weight_590, v_594, a_574[2]);
        float2 v_595 = make_float2(_unpack_bf16_38[6], _unpack_bf16_38[7]);
        a_574[3] = fma_f32x2_rn_noftz(weight_590, v_595, a_574[3]);
        acc[0] = a_574[0].x;
        acc[1] = a_574[0].y;
        acc[2] = a_574[1].x;
        acc[3] = a_574[1].y;
        acc[4] = a_574[2].x;
        acc[5] = a_574[2].y;
        acc[6] = a_574[3].x;
        acc[7] = a_574[3].y;
        const int woff_596 = 4;
        int base_597 = (2 + group) * 1024 + thread * 8;
        float2 a_598[4];
        float2 previous_599 = make_float2(acc[8], acc[9]);
        a_598[0] = mul_f32x2_noftz(previous_599, corr_571);
        float2 previous_600 = make_float2(acc[10], acc[11]);
        a_598[1] = mul_f32x2_noftz(previous_600, corr_571);
        float2 previous_601 = make_float2(acc[12], acc[13]);
        a_598[2] = mul_f32x2_noftz(previous_601, corr_571);
        float2 previous_602 = make_float2(acc[14], acc[15]);
        a_598[3] = mul_f32x2_noftz(previous_602, corr_571);
        float2 weight_603 = make_float2(weights_569[3], weights_569[3]);
        unsigned int sw2_604[4];
        sw2_604[0] = words[42 + woff_596];
        sw2_604[1] = words[42 + woff_596 + 1];
        sw2_604[2] = words[42 + woff_596 + 2];
        sw2_604[3] = words[42 + woff_596 + 3];
        float _unpack_bf16_39[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_39[_pair * 2])[0]), "=f"((&_unpack_bf16_39[_pair * 2])[1])
                : "r"(sw2_604[_pair]));
        }
        float2 v_605 = make_float2(_unpack_bf16_39[0], _unpack_bf16_39[1]);
        a_598[0] = fma_f32x2_rn_noftz(weight_603, v_605, a_598[0]);
        float2 v_606 = make_float2(_unpack_bf16_39[2], _unpack_bf16_39[3]);
        a_598[1] = fma_f32x2_rn_noftz(weight_603, v_606, a_598[1]);
        float2 v_607 = make_float2(_unpack_bf16_39[4], _unpack_bf16_39[5]);
        a_598[2] = fma_f32x2_rn_noftz(weight_603, v_607, a_598[2]);
        float2 v_608 = make_float2(_unpack_bf16_39[6], _unpack_bf16_39[7]);
        a_598[3] = fma_f32x2_rn_noftz(weight_603, v_608, a_598[3]);
        float2 weight_609 = make_float2(weights_569[4], weights_569[4]);
        unsigned int sw2_610[4];
        sw2_610[0] = words[56 + woff_596];
        sw2_610[1] = words[56 + woff_596 + 1];
        sw2_610[2] = words[56 + woff_596 + 2];
        sw2_610[3] = words[56 + woff_596 + 3];
        float _unpack_bf16_40[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_40[_pair * 2])[0]), "=f"((&_unpack_bf16_40[_pair * 2])[1])
                : "r"(sw2_610[_pair]));
        }
        float2 v_611 = make_float2(_unpack_bf16_40[0], _unpack_bf16_40[1]);
        a_598[0] = fma_f32x2_rn_noftz(weight_609, v_611, a_598[0]);
        float2 v_612 = make_float2(_unpack_bf16_40[2], _unpack_bf16_40[3]);
        a_598[1] = fma_f32x2_rn_noftz(weight_609, v_612, a_598[1]);
        float2 v_613 = make_float2(_unpack_bf16_40[4], _unpack_bf16_40[5]);
        a_598[2] = fma_f32x2_rn_noftz(weight_609, v_613, a_598[2]);
        float2 v_614 = make_float2(_unpack_bf16_40[6], _unpack_bf16_40[7]);
        a_598[3] = fma_f32x2_rn_noftz(weight_609, v_614, a_598[3]);
        float2 weight_615 = make_float2(weights_569[5], weights_569[5]);
        unsigned int sw2_616[4];
        sw2_616[0] = words[70 + woff_596];
        sw2_616[1] = words[70 + woff_596 + 1];
        sw2_616[2] = words[70 + woff_596 + 2];
        sw2_616[3] = words[70 + woff_596 + 3];
        float _unpack_bf16_41[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_41[_pair * 2])[0]), "=f"((&_unpack_bf16_41[_pair * 2])[1])
                : "r"(sw2_616[_pair]));
        }
        float2 v_617 = make_float2(_unpack_bf16_41[0], _unpack_bf16_41[1]);
        a_598[0] = fma_f32x2_rn_noftz(weight_615, v_617, a_598[0]);
        float2 v_618 = make_float2(_unpack_bf16_41[2], _unpack_bf16_41[3]);
        a_598[1] = fma_f32x2_rn_noftz(weight_615, v_618, a_598[1]);
        float2 v_619 = make_float2(_unpack_bf16_41[4], _unpack_bf16_41[5]);
        a_598[2] = fma_f32x2_rn_noftz(weight_615, v_619, a_598[2]);
        float2 v_620 = make_float2(_unpack_bf16_41[6], _unpack_bf16_41[7]);
        a_598[3] = fma_f32x2_rn_noftz(weight_615, v_620, a_598[3]);
        acc[8] = a_598[0].x;
        acc[9] = a_598[0].y;
        acc[10] = a_598[1].x;
        acc[11] = a_598[1].y;
        acc[12] = a_598[2].x;
        acc[13] = a_598[2].y;
        acc[14] = a_598[3].x;
        acc[15] = a_598[3].y;
        const int woff_621 = 8;
        int base_622 = (4 + group) * 1024 + thread * 8;
        float2 a_623[4];
        float2 previous_624 = make_float2(acc[16], acc[17]);
        a_623[0] = mul_f32x2_noftz(previous_624, corr_571);
        float2 previous_625 = make_float2(acc[18], acc[19]);
        a_623[1] = mul_f32x2_noftz(previous_625, corr_571);
        float2 previous_626 = make_float2(acc[20], acc[21]);
        a_623[2] = mul_f32x2_noftz(previous_626, corr_571);
        float2 previous_627 = make_float2(acc[22], acc[23]);
        a_623[3] = mul_f32x2_noftz(previous_627, corr_571);
        float2 weight_628 = make_float2(weights_569[3], weights_569[3]);
        unsigned int sw2_629[4];
        sw2_629[0] = words[42 + woff_621];
        sw2_629[1] = words[42 + woff_621 + 1];
        sw2_629[2] = words[42 + woff_621 + 2];
        sw2_629[3] = words[42 + woff_621 + 3];
        float _unpack_bf16_42[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_42[_pair * 2])[0]), "=f"((&_unpack_bf16_42[_pair * 2])[1])
                : "r"(sw2_629[_pair]));
        }
        float2 v_630 = make_float2(_unpack_bf16_42[0], _unpack_bf16_42[1]);
        a_623[0] = fma_f32x2_rn_noftz(weight_628, v_630, a_623[0]);
        float2 v_631 = make_float2(_unpack_bf16_42[2], _unpack_bf16_42[3]);
        a_623[1] = fma_f32x2_rn_noftz(weight_628, v_631, a_623[1]);
        float2 v_632 = make_float2(_unpack_bf16_42[4], _unpack_bf16_42[5]);
        a_623[2] = fma_f32x2_rn_noftz(weight_628, v_632, a_623[2]);
        float2 v_633 = make_float2(_unpack_bf16_42[6], _unpack_bf16_42[7]);
        a_623[3] = fma_f32x2_rn_noftz(weight_628, v_633, a_623[3]);
        float2 weight_634 = make_float2(weights_569[4], weights_569[4]);
        unsigned int sw2_635[4];
        sw2_635[0] = words[56 + woff_621];
        sw2_635[1] = words[56 + woff_621 + 1];
        sw2_635[2] = words[56 + woff_621 + 2];
        sw2_635[3] = words[56 + woff_621 + 3];
        float _unpack_bf16_43[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_43[_pair * 2])[0]), "=f"((&_unpack_bf16_43[_pair * 2])[1])
                : "r"(sw2_635[_pair]));
        }
        float2 v_636 = make_float2(_unpack_bf16_43[0], _unpack_bf16_43[1]);
        a_623[0] = fma_f32x2_rn_noftz(weight_634, v_636, a_623[0]);
        float2 v_637 = make_float2(_unpack_bf16_43[2], _unpack_bf16_43[3]);
        a_623[1] = fma_f32x2_rn_noftz(weight_634, v_637, a_623[1]);
        float2 v_638 = make_float2(_unpack_bf16_43[4], _unpack_bf16_43[5]);
        a_623[2] = fma_f32x2_rn_noftz(weight_634, v_638, a_623[2]);
        float2 v_639 = make_float2(_unpack_bf16_43[6], _unpack_bf16_43[7]);
        a_623[3] = fma_f32x2_rn_noftz(weight_634, v_639, a_623[3]);
        float2 weight_640 = make_float2(weights_569[5], weights_569[5]);
        unsigned int sw2_641[4];
        sw2_641[0] = words[70 + woff_621];
        sw2_641[1] = words[70 + woff_621 + 1];
        sw2_641[2] = words[70 + woff_621 + 2];
        sw2_641[3] = words[70 + woff_621 + 3];
        float _unpack_bf16_44[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_44[_pair * 2])[0]), "=f"((&_unpack_bf16_44[_pair * 2])[1])
                : "r"(sw2_641[_pair]));
        }
        float2 v_642 = make_float2(_unpack_bf16_44[0], _unpack_bf16_44[1]);
        a_623[0] = fma_f32x2_rn_noftz(weight_640, v_642, a_623[0]);
        float2 v_643 = make_float2(_unpack_bf16_44[2], _unpack_bf16_44[3]);
        a_623[1] = fma_f32x2_rn_noftz(weight_640, v_643, a_623[1]);
        float2 v_644 = make_float2(_unpack_bf16_44[4], _unpack_bf16_44[5]);
        a_623[2] = fma_f32x2_rn_noftz(weight_640, v_644, a_623[2]);
        float2 v_645 = make_float2(_unpack_bf16_44[6], _unpack_bf16_44[7]);
        a_623[3] = fma_f32x2_rn_noftz(weight_640, v_645, a_623[3]);
        acc[16] = a_623[0].x;
        acc[17] = a_623[0].y;
        acc[18] = a_623[1].x;
        acc[19] = a_623[1].y;
        acc[20] = a_623[2].x;
        acc[21] = a_623[2].y;
        acc[22] = a_623[3].x;
        acc[23] = a_623[3].y;
        const int woff_646 = 12;
        int base_647 = 6144 + group * 512 + thread * 4;
        float2 a_648[4];
        float2 previous_649 = make_float2(acc[24], acc[25]);
        a_648[0] = mul_f32x2_noftz(previous_649, corr_571);
        float2 previous_650 = make_float2(acc[26], acc[27]);
        a_648[1] = mul_f32x2_noftz(previous_650, corr_571);
        float2 weight_651 = make_float2(weights_569[3], weights_569[3]);
        unsigned int sw2_652[4];
        sw2_652[0] = words[42 + woff_646];
        sw2_652[1] = words[42 + woff_646 + 1];
        float _unpack_bf16_45[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_45[_pair * 2])[0]), "=f"((&_unpack_bf16_45[_pair * 2])[1])
                : "r"(sw2_652[_pair]));
        }
        float2 v_653 = make_float2(_unpack_bf16_45[0], _unpack_bf16_45[1]);
        a_648[0] = fma_f32x2_rn_noftz(weight_651, v_653, a_648[0]);
        float2 v_654 = make_float2(_unpack_bf16_45[2], _unpack_bf16_45[3]);
        a_648[1] = fma_f32x2_rn_noftz(weight_651, v_654, a_648[1]);
        float2 weight_655 = make_float2(weights_569[4], weights_569[4]);
        unsigned int sw2_656[4];
        sw2_656[0] = words[56 + woff_646];
        sw2_656[1] = words[56 + woff_646 + 1];
        float _unpack_bf16_46[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_46[_pair * 2])[0]), "=f"((&_unpack_bf16_46[_pair * 2])[1])
                : "r"(sw2_656[_pair]));
        }
        float2 v_657 = make_float2(_unpack_bf16_46[0], _unpack_bf16_46[1]);
        a_648[0] = fma_f32x2_rn_noftz(weight_655, v_657, a_648[0]);
        float2 v_658 = make_float2(_unpack_bf16_46[2], _unpack_bf16_46[3]);
        a_648[1] = fma_f32x2_rn_noftz(weight_655, v_658, a_648[1]);
        float2 weight_659 = make_float2(weights_569[5], weights_569[5]);
        unsigned int sw2_660[4];
        sw2_660[0] = words[70 + woff_646];
        sw2_660[1] = words[70 + woff_646 + 1];
        float _unpack_bf16_47[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_47[_pair * 2])[0]), "=f"((&_unpack_bf16_47[_pair * 2])[1])
                : "r"(sw2_660[_pair]));
        }
        float2 v_661 = make_float2(_unpack_bf16_47[0], _unpack_bf16_47[1]);
        a_648[0] = fma_f32x2_rn_noftz(weight_659, v_661, a_648[0]);
        float2 v_662 = make_float2(_unpack_bf16_47[2], _unpack_bf16_47[3]);
        a_648[1] = fma_f32x2_rn_noftz(weight_659, v_662, a_648[1]);
        acc[24] = a_648[0].x;
        acc[25] = a_648[0].y;
        acc[26] = a_648[1].x;
        acc[27] = a_648[1].y;
        sum_running = __fmaf_rn(sum_running, correction_568, sum_weights_570);
        max_running = max_new_567;
        float max_chunk_663 = -3.4028235e+38f;
        max_chunk_663 = fmaxf(max_chunk_663, logits[6]);
        max_chunk_663 = fmaxf(max_chunk_663, logits[7]);
        max_chunk_663 = fmaxf(max_chunk_663, logits[8]);
        float max_new_664 = fmaxf(max_running, max_chunk_663);
        float correction_665 = approx_exp2((max_running - max_new_664) * 1.442695f);
        float weights_666[9];
        float sum_weights_667 = 0.0f;
        float _exp2_192 = approx_exp2((logits[6] - max_new_664) * 1.442695f);
        weights_666[6] = _exp2_192;
        sum_weights_667 += weights_666[6];
        float _exp2_193 = approx_exp2((logits[7] - max_new_664) * 1.442695f);
        weights_666[7] = _exp2_193;
        sum_weights_667 += weights_666[7];
        float _exp2_194 = approx_exp2((logits[8] - max_new_664) * 1.442695f);
        weights_666[8] = _exp2_194;
        sum_weights_667 += weights_666[8];
        float2 corr_668 = make_float2(correction_665, correction_665);
        const int woff_669 = 0;
        int base_670 = group * 1024 + thread * 8;
        float2 a_671[4];
        float2 previous_672 = make_float2(acc[0], acc[1]);
        a_671[0] = mul_f32x2_noftz(previous_672, corr_668);
        float2 previous_673 = make_float2(acc[2], acc[3]);
        a_671[1] = mul_f32x2_noftz(previous_673, corr_668);
        float2 previous_674 = make_float2(acc[4], acc[5]);
        a_671[2] = mul_f32x2_noftz(previous_674, corr_668);
        float2 previous_675 = make_float2(acc[6], acc[7]);
        a_671[3] = mul_f32x2_noftz(previous_675, corr_668);
        float2 weight_676 = make_float2(weights_666[6], weights_666[6]);
        unsigned int sw2_677[4];
        sw2_677[0] = words[84 + woff_669];
        sw2_677[1] = words[84 + woff_669 + 1];
        sw2_677[2] = words[84 + woff_669 + 2];
        sw2_677[3] = words[84 + woff_669 + 3];
        float _unpack_bf16_48[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_48[_pair * 2])[0]), "=f"((&_unpack_bf16_48[_pair * 2])[1])
                : "r"(sw2_677[_pair]));
        }
        float2 v_678 = make_float2(_unpack_bf16_48[0], _unpack_bf16_48[1]);
        a_671[0] = fma_f32x2_rn_noftz(weight_676, v_678, a_671[0]);
        float2 v_679 = make_float2(_unpack_bf16_48[2], _unpack_bf16_48[3]);
        a_671[1] = fma_f32x2_rn_noftz(weight_676, v_679, a_671[1]);
        float2 v_680 = make_float2(_unpack_bf16_48[4], _unpack_bf16_48[5]);
        a_671[2] = fma_f32x2_rn_noftz(weight_676, v_680, a_671[2]);
        float2 v_681 = make_float2(_unpack_bf16_48[6], _unpack_bf16_48[7]);
        a_671[3] = fma_f32x2_rn_noftz(weight_676, v_681, a_671[3]);
        float2 weight_682 = make_float2(weights_666[7], weights_666[7]);
        unsigned int sw2_683[4];
        sw2_683[0] = words[98 + woff_669];
        sw2_683[1] = words[98 + woff_669 + 1];
        sw2_683[2] = words[98 + woff_669 + 2];
        sw2_683[3] = words[98 + woff_669 + 3];
        float _unpack_bf16_49[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_49[_pair * 2])[0]), "=f"((&_unpack_bf16_49[_pair * 2])[1])
                : "r"(sw2_683[_pair]));
        }
        float2 v_684 = make_float2(_unpack_bf16_49[0], _unpack_bf16_49[1]);
        a_671[0] = fma_f32x2_rn_noftz(weight_682, v_684, a_671[0]);
        float2 v_685 = make_float2(_unpack_bf16_49[2], _unpack_bf16_49[3]);
        a_671[1] = fma_f32x2_rn_noftz(weight_682, v_685, a_671[1]);
        float2 v_686 = make_float2(_unpack_bf16_49[4], _unpack_bf16_49[5]);
        a_671[2] = fma_f32x2_rn_noftz(weight_682, v_686, a_671[2]);
        float2 v_687 = make_float2(_unpack_bf16_49[6], _unpack_bf16_49[7]);
        a_671[3] = fma_f32x2_rn_noftz(weight_682, v_687, a_671[3]);
        float2 weight_688 = make_float2(weights_666[8], weights_666[8]);
        unsigned int sw2_689[4];
        sw2_689[0] = words[112 + woff_669];
        sw2_689[1] = words[112 + woff_669 + 1];
        sw2_689[2] = words[112 + woff_669 + 2];
        sw2_689[3] = words[112 + woff_669 + 3];
        float _unpack_bf16_50[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_50[_pair * 2])[0]), "=f"((&_unpack_bf16_50[_pair * 2])[1])
                : "r"(sw2_689[_pair]));
        }
        float2 v_690 = make_float2(_unpack_bf16_50[0], _unpack_bf16_50[1]);
        a_671[0] = fma_f32x2_rn_noftz(weight_688, v_690, a_671[0]);
        float2 v_691 = make_float2(_unpack_bf16_50[2], _unpack_bf16_50[3]);
        a_671[1] = fma_f32x2_rn_noftz(weight_688, v_691, a_671[1]);
        float2 v_692 = make_float2(_unpack_bf16_50[4], _unpack_bf16_50[5]);
        a_671[2] = fma_f32x2_rn_noftz(weight_688, v_692, a_671[2]);
        float2 v_693 = make_float2(_unpack_bf16_50[6], _unpack_bf16_50[7]);
        a_671[3] = fma_f32x2_rn_noftz(weight_688, v_693, a_671[3]);
        acc[0] = a_671[0].x;
        acc[1] = a_671[0].y;
        acc[2] = a_671[1].x;
        acc[3] = a_671[1].y;
        acc[4] = a_671[2].x;
        acc[5] = a_671[2].y;
        acc[6] = a_671[3].x;
        acc[7] = a_671[3].y;
        const int woff_694 = 4;
        int base_695 = (2 + group) * 1024 + thread * 8;
        float2 a_696[4];
        float2 previous_697 = make_float2(acc[8], acc[9]);
        a_696[0] = mul_f32x2_noftz(previous_697, corr_668);
        float2 previous_698 = make_float2(acc[10], acc[11]);
        a_696[1] = mul_f32x2_noftz(previous_698, corr_668);
        float2 previous_699 = make_float2(acc[12], acc[13]);
        a_696[2] = mul_f32x2_noftz(previous_699, corr_668);
        float2 previous_700 = make_float2(acc[14], acc[15]);
        a_696[3] = mul_f32x2_noftz(previous_700, corr_668);
        float2 weight_701 = make_float2(weights_666[6], weights_666[6]);
        unsigned int sw2_702[4];
        sw2_702[0] = words[84 + woff_694];
        sw2_702[1] = words[84 + woff_694 + 1];
        sw2_702[2] = words[84 + woff_694 + 2];
        sw2_702[3] = words[84 + woff_694 + 3];
        float _unpack_bf16_51[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_51[_pair * 2])[0]), "=f"((&_unpack_bf16_51[_pair * 2])[1])
                : "r"(sw2_702[_pair]));
        }
        float2 v_703 = make_float2(_unpack_bf16_51[0], _unpack_bf16_51[1]);
        a_696[0] = fma_f32x2_rn_noftz(weight_701, v_703, a_696[0]);
        float2 v_704 = make_float2(_unpack_bf16_51[2], _unpack_bf16_51[3]);
        a_696[1] = fma_f32x2_rn_noftz(weight_701, v_704, a_696[1]);
        float2 v_705 = make_float2(_unpack_bf16_51[4], _unpack_bf16_51[5]);
        a_696[2] = fma_f32x2_rn_noftz(weight_701, v_705, a_696[2]);
        float2 v_706 = make_float2(_unpack_bf16_51[6], _unpack_bf16_51[7]);
        a_696[3] = fma_f32x2_rn_noftz(weight_701, v_706, a_696[3]);
        float2 weight_707 = make_float2(weights_666[7], weights_666[7]);
        unsigned int sw2_708[4];
        sw2_708[0] = words[98 + woff_694];
        sw2_708[1] = words[98 + woff_694 + 1];
        sw2_708[2] = words[98 + woff_694 + 2];
        sw2_708[3] = words[98 + woff_694 + 3];
        float _unpack_bf16_52[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_52[_pair * 2])[0]), "=f"((&_unpack_bf16_52[_pair * 2])[1])
                : "r"(sw2_708[_pair]));
        }
        float2 v_709 = make_float2(_unpack_bf16_52[0], _unpack_bf16_52[1]);
        a_696[0] = fma_f32x2_rn_noftz(weight_707, v_709, a_696[0]);
        float2 v_710 = make_float2(_unpack_bf16_52[2], _unpack_bf16_52[3]);
        a_696[1] = fma_f32x2_rn_noftz(weight_707, v_710, a_696[1]);
        float2 v_711 = make_float2(_unpack_bf16_52[4], _unpack_bf16_52[5]);
        a_696[2] = fma_f32x2_rn_noftz(weight_707, v_711, a_696[2]);
        float2 v_712 = make_float2(_unpack_bf16_52[6], _unpack_bf16_52[7]);
        a_696[3] = fma_f32x2_rn_noftz(weight_707, v_712, a_696[3]);
        float2 weight_713 = make_float2(weights_666[8], weights_666[8]);
        unsigned int sw2_714[4];
        sw2_714[0] = words[112 + woff_694];
        sw2_714[1] = words[112 + woff_694 + 1];
        sw2_714[2] = words[112 + woff_694 + 2];
        sw2_714[3] = words[112 + woff_694 + 3];
        float _unpack_bf16_53[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_53[_pair * 2])[0]), "=f"((&_unpack_bf16_53[_pair * 2])[1])
                : "r"(sw2_714[_pair]));
        }
        float2 v_715 = make_float2(_unpack_bf16_53[0], _unpack_bf16_53[1]);
        a_696[0] = fma_f32x2_rn_noftz(weight_713, v_715, a_696[0]);
        float2 v_716 = make_float2(_unpack_bf16_53[2], _unpack_bf16_53[3]);
        a_696[1] = fma_f32x2_rn_noftz(weight_713, v_716, a_696[1]);
        float2 v_717 = make_float2(_unpack_bf16_53[4], _unpack_bf16_53[5]);
        a_696[2] = fma_f32x2_rn_noftz(weight_713, v_717, a_696[2]);
        float2 v_718 = make_float2(_unpack_bf16_53[6], _unpack_bf16_53[7]);
        a_696[3] = fma_f32x2_rn_noftz(weight_713, v_718, a_696[3]);
        acc[8] = a_696[0].x;
        acc[9] = a_696[0].y;
        acc[10] = a_696[1].x;
        acc[11] = a_696[1].y;
        acc[12] = a_696[2].x;
        acc[13] = a_696[2].y;
        acc[14] = a_696[3].x;
        acc[15] = a_696[3].y;
        const int woff_719 = 8;
        int base_720 = (4 + group) * 1024 + thread * 8;
        float2 a_721[4];
        float2 previous_722 = make_float2(acc[16], acc[17]);
        a_721[0] = mul_f32x2_noftz(previous_722, corr_668);
        float2 previous_723 = make_float2(acc[18], acc[19]);
        a_721[1] = mul_f32x2_noftz(previous_723, corr_668);
        float2 previous_724 = make_float2(acc[20], acc[21]);
        a_721[2] = mul_f32x2_noftz(previous_724, corr_668);
        float2 previous_725 = make_float2(acc[22], acc[23]);
        a_721[3] = mul_f32x2_noftz(previous_725, corr_668);
        float2 weight_726 = make_float2(weights_666[6], weights_666[6]);
        unsigned int sw2_727[4];
        sw2_727[0] = words[84 + woff_719];
        sw2_727[1] = words[84 + woff_719 + 1];
        sw2_727[2] = words[84 + woff_719 + 2];
        sw2_727[3] = words[84 + woff_719 + 3];
        float _unpack_bf16_54[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_54[_pair * 2])[0]), "=f"((&_unpack_bf16_54[_pair * 2])[1])
                : "r"(sw2_727[_pair]));
        }
        float2 v_728 = make_float2(_unpack_bf16_54[0], _unpack_bf16_54[1]);
        a_721[0] = fma_f32x2_rn_noftz(weight_726, v_728, a_721[0]);
        float2 v_729 = make_float2(_unpack_bf16_54[2], _unpack_bf16_54[3]);
        a_721[1] = fma_f32x2_rn_noftz(weight_726, v_729, a_721[1]);
        float2 v_730 = make_float2(_unpack_bf16_54[4], _unpack_bf16_54[5]);
        a_721[2] = fma_f32x2_rn_noftz(weight_726, v_730, a_721[2]);
        float2 v_731 = make_float2(_unpack_bf16_54[6], _unpack_bf16_54[7]);
        a_721[3] = fma_f32x2_rn_noftz(weight_726, v_731, a_721[3]);
        float2 weight_732 = make_float2(weights_666[7], weights_666[7]);
        unsigned int sw2_733[4];
        sw2_733[0] = words[98 + woff_719];
        sw2_733[1] = words[98 + woff_719 + 1];
        sw2_733[2] = words[98 + woff_719 + 2];
        sw2_733[3] = words[98 + woff_719 + 3];
        float _unpack_bf16_55[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_55[_pair * 2])[0]), "=f"((&_unpack_bf16_55[_pair * 2])[1])
                : "r"(sw2_733[_pair]));
        }
        float2 v_734 = make_float2(_unpack_bf16_55[0], _unpack_bf16_55[1]);
        a_721[0] = fma_f32x2_rn_noftz(weight_732, v_734, a_721[0]);
        float2 v_735 = make_float2(_unpack_bf16_55[2], _unpack_bf16_55[3]);
        a_721[1] = fma_f32x2_rn_noftz(weight_732, v_735, a_721[1]);
        float2 v_736 = make_float2(_unpack_bf16_55[4], _unpack_bf16_55[5]);
        a_721[2] = fma_f32x2_rn_noftz(weight_732, v_736, a_721[2]);
        float2 v_737 = make_float2(_unpack_bf16_55[6], _unpack_bf16_55[7]);
        a_721[3] = fma_f32x2_rn_noftz(weight_732, v_737, a_721[3]);
        float2 weight_738 = make_float2(weights_666[8], weights_666[8]);
        unsigned int sw2_739[4];
        sw2_739[0] = words[112 + woff_719];
        sw2_739[1] = words[112 + woff_719 + 1];
        sw2_739[2] = words[112 + woff_719 + 2];
        sw2_739[3] = words[112 + woff_719 + 3];
        float _unpack_bf16_56[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_56[_pair * 2])[0]), "=f"((&_unpack_bf16_56[_pair * 2])[1])
                : "r"(sw2_739[_pair]));
        }
        float2 v_740 = make_float2(_unpack_bf16_56[0], _unpack_bf16_56[1]);
        a_721[0] = fma_f32x2_rn_noftz(weight_738, v_740, a_721[0]);
        float2 v_741 = make_float2(_unpack_bf16_56[2], _unpack_bf16_56[3]);
        a_721[1] = fma_f32x2_rn_noftz(weight_738, v_741, a_721[1]);
        float2 v_742 = make_float2(_unpack_bf16_56[4], _unpack_bf16_56[5]);
        a_721[2] = fma_f32x2_rn_noftz(weight_738, v_742, a_721[2]);
        float2 v_743 = make_float2(_unpack_bf16_56[6], _unpack_bf16_56[7]);
        a_721[3] = fma_f32x2_rn_noftz(weight_738, v_743, a_721[3]);
        acc[16] = a_721[0].x;
        acc[17] = a_721[0].y;
        acc[18] = a_721[1].x;
        acc[19] = a_721[1].y;
        acc[20] = a_721[2].x;
        acc[21] = a_721[2].y;
        acc[22] = a_721[3].x;
        acc[23] = a_721[3].y;
        const int woff_744 = 12;
        int base_745 = 6144 + group * 512 + thread * 4;
        float2 a_746[4];
        float2 previous_747 = make_float2(acc[24], acc[25]);
        a_746[0] = mul_f32x2_noftz(previous_747, corr_668);
        float2 previous_748 = make_float2(acc[26], acc[27]);
        a_746[1] = mul_f32x2_noftz(previous_748, corr_668);
        float2 weight_749 = make_float2(weights_666[6], weights_666[6]);
        unsigned int sw2_750[4];
        sw2_750[0] = words[84 + woff_744];
        sw2_750[1] = words[84 + woff_744 + 1];
        float _unpack_bf16_57[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_57[_pair * 2])[0]), "=f"((&_unpack_bf16_57[_pair * 2])[1])
                : "r"(sw2_750[_pair]));
        }
        float2 v_751 = make_float2(_unpack_bf16_57[0], _unpack_bf16_57[1]);
        a_746[0] = fma_f32x2_rn_noftz(weight_749, v_751, a_746[0]);
        float2 v_752 = make_float2(_unpack_bf16_57[2], _unpack_bf16_57[3]);
        a_746[1] = fma_f32x2_rn_noftz(weight_749, v_752, a_746[1]);
        float2 weight_753 = make_float2(weights_666[7], weights_666[7]);
        unsigned int sw2_754[4];
        sw2_754[0] = words[98 + woff_744];
        sw2_754[1] = words[98 + woff_744 + 1];
        float _unpack_bf16_58[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_58[_pair * 2])[0]), "=f"((&_unpack_bf16_58[_pair * 2])[1])
                : "r"(sw2_754[_pair]));
        }
        float2 v_755 = make_float2(_unpack_bf16_58[0], _unpack_bf16_58[1]);
        a_746[0] = fma_f32x2_rn_noftz(weight_753, v_755, a_746[0]);
        float2 v_756 = make_float2(_unpack_bf16_58[2], _unpack_bf16_58[3]);
        a_746[1] = fma_f32x2_rn_noftz(weight_753, v_756, a_746[1]);
        float2 weight_757 = make_float2(weights_666[8], weights_666[8]);
        unsigned int sw2_758[4];
        sw2_758[0] = words[112 + woff_744];
        sw2_758[1] = words[112 + woff_744 + 1];
        float _unpack_bf16_59[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_59[_pair * 2])[0]), "=f"((&_unpack_bf16_59[_pair * 2])[1])
                : "r"(sw2_758[_pair]));
        }
        float2 v_759 = make_float2(_unpack_bf16_59[0], _unpack_bf16_59[1]);
        a_746[0] = fma_f32x2_rn_noftz(weight_757, v_759, a_746[0]);
        float2 v_760 = make_float2(_unpack_bf16_59[2], _unpack_bf16_59[3]);
        a_746[1] = fma_f32x2_rn_noftz(weight_757, v_760, a_746[1]);
        acc[24] = a_746[0].x;
        acc[25] = a_746[0].y;
        acc[26] = a_746[1].x;
        acc[27] = a_746[1].y;
        sum_running = __fmaf_rn(sum_running, correction_665, sum_weights_667);
        max_running = max_new_664;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_761 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_761, v_761, output_sq_pair);
        float2 v_762 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_762, v_762, output_sq_pair);
        float2 v_763 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_763, v_763, output_sq_pair);
        float2 v_764 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_764, v_764, output_sq_pair);
        float2 v_765 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_765, v_765, output_sq_pair);
        float2 v_766 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_766, v_766, output_sq_pair);
        float2 v_767 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_767, v_767, output_sq_pair);
        float2 v_768 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_768, v_768, output_sq_pair);
        float2 v_769 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_769, v_769, output_sq_pair);
        float2 v_770 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_770, v_770, output_sq_pair);
        float2 v_771 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_771, v_771, output_sq_pair);
        float2 v_772 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_772, v_772, output_sq_pair);
        float2 v_773 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_773, v_773, output_sq_pair);
        float2 v_774 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_774, v_774, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_195;
        float _shfl_xor_sync_196 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_196;
        float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_197;
        float _shfl_xor_sync_198 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_198;
        float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_199;
        if (lane == 0) {
            uint32_t _cluster_mapa_200;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_200) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_200), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_201;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_201) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_201), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_202;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_202) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_202), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_203;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_203) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_203), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_204 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_204;
        float _shfl_down_sync_205 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_205;
        float _shfl_down_sync_206 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_206;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_207 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_207;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_775 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_208 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_208), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_209 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_209));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_776 = 2;
        const int acc_idx_777 = value_idx_776;
        float2 _f32x2_mul_a_210 = make_float2(acc[acc_idx_777], acc[acc_idx_777 + 1]);
        float2 scaled_pair_778;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_778) : "l"(*(const unsigned long long*)&_f32x2_mul_a_210), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_211 = make_float2(wout[acc_idx_777], wout[acc_idx_777 + 1]);
        float2 normalized_pair_779;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_779) : "l"(*(const unsigned long long*)&scaled_pair_778), "l"(*(const unsigned long long*)&_f32x2_mul_b_211));
        output_values[value_idx_776] = normalized_pair_779.x;
        output_values[value_idx_776 + 1] = normalized_pair_779.y;
        const int value_idx_780 = 4;
        const int acc_idx_781 = value_idx_780;
        float2 _f32x2_mul_a_212 = make_float2(acc[acc_idx_781], acc[acc_idx_781 + 1]);
        float2 scaled_pair_782;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_782) : "l"(*(const unsigned long long*)&_f32x2_mul_a_212), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_213 = make_float2(wout[acc_idx_781], wout[acc_idx_781 + 1]);
        float2 normalized_pair_783;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_783) : "l"(*(const unsigned long long*)&scaled_pair_782), "l"(*(const unsigned long long*)&_f32x2_mul_b_213));
        output_values[value_idx_780] = normalized_pair_783.x;
        output_values[value_idx_780 + 1] = normalized_pair_783.y;
        const int value_idx_784 = 6;
        const int acc_idx_785 = value_idx_784;
        float2 _f32x2_mul_a_214 = make_float2(acc[acc_idx_785], acc[acc_idx_785 + 1]);
        float2 scaled_pair_786;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_786) : "l"(*(const unsigned long long*)&_f32x2_mul_a_214), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_215 = make_float2(wout[acc_idx_785], wout[acc_idx_785 + 1]);
        float2 normalized_pair_787;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_787) : "l"(*(const unsigned long long*)&scaled_pair_786), "l"(*(const unsigned long long*)&_f32x2_mul_b_215));
        output_values[value_idx_784] = normalized_pair_787.x;
        output_values[value_idx_784 + 1] = normalized_pair_787.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_775))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_788 = (2 + group) * 1024 + thread * 8;
        float output_values_789[8];
        const int value_idx_790 = 0;
        const int acc_idx_791 = 8 + value_idx_790;
        float2 _f32x2_mul_a_216 = make_float2(acc[acc_idx_791], acc[acc_idx_791 + 1]);
        float2 scaled_pair_792;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_792) : "l"(*(const unsigned long long*)&_f32x2_mul_a_216), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_217 = make_float2(wout[acc_idx_791], wout[acc_idx_791 + 1]);
        float2 normalized_pair_793;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_793) : "l"(*(const unsigned long long*)&scaled_pair_792), "l"(*(const unsigned long long*)&_f32x2_mul_b_217));
        output_values_789[value_idx_790] = normalized_pair_793.x;
        output_values_789[value_idx_790 + 1] = normalized_pair_793.y;
        const int value_idx_794 = 2;
        const int acc_idx_795 = 8 + value_idx_794;
        float2 _f32x2_mul_a_218 = make_float2(acc[acc_idx_795], acc[acc_idx_795 + 1]);
        float2 scaled_pair_796;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_796) : "l"(*(const unsigned long long*)&_f32x2_mul_a_218), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_219 = make_float2(wout[acc_idx_795], wout[acc_idx_795 + 1]);
        float2 normalized_pair_797;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_797) : "l"(*(const unsigned long long*)&scaled_pair_796), "l"(*(const unsigned long long*)&_f32x2_mul_b_219));
        output_values_789[value_idx_794] = normalized_pair_797.x;
        output_values_789[value_idx_794 + 1] = normalized_pair_797.y;
        const int value_idx_798 = 4;
        const int acc_idx_799 = 8 + value_idx_798;
        float2 _f32x2_mul_a_220 = make_float2(acc[acc_idx_799], acc[acc_idx_799 + 1]);
        float2 scaled_pair_800;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_800) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_221 = make_float2(wout[acc_idx_799], wout[acc_idx_799 + 1]);
        float2 normalized_pair_801;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_801) : "l"(*(const unsigned long long*)&scaled_pair_800), "l"(*(const unsigned long long*)&_f32x2_mul_b_221));
        output_values_789[value_idx_798] = normalized_pair_801.x;
        output_values_789[value_idx_798 + 1] = normalized_pair_801.y;
        const int value_idx_802 = 6;
        const int acc_idx_803 = 8 + value_idx_802;
        float2 _f32x2_mul_a_222 = make_float2(acc[acc_idx_803], acc[acc_idx_803 + 1]);
        float2 scaled_pair_804;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_804) : "l"(*(const unsigned long long*)&_f32x2_mul_a_222), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_223 = make_float2(wout[acc_idx_803], wout[acc_idx_803 + 1]);
        float2 normalized_pair_805;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_805) : "l"(*(const unsigned long long*)&scaled_pair_804), "l"(*(const unsigned long long*)&_f32x2_mul_b_223));
        output_values_789[value_idx_802] = normalized_pair_805.x;
        output_values_789[value_idx_802 + 1] = normalized_pair_805.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_789[_lp*2 + 0], output_values_789[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_789[_lp*2 + 2], output_values_789[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_789[_lp*2 + 4], output_values_789[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_789[_lp*2 + 6], output_values_789[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_788))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_806 = (4 + group) * 1024 + thread * 8;
        float output_values_807[8];
        const int value_idx_808 = 0;
        const int acc_idx_809 = 16 + value_idx_808;
        float2 _f32x2_mul_a_224 = make_float2(acc[acc_idx_809], acc[acc_idx_809 + 1]);
        float2 scaled_pair_810;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_810) : "l"(*(const unsigned long long*)&_f32x2_mul_a_224), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_225 = make_float2(wout[acc_idx_809], wout[acc_idx_809 + 1]);
        float2 normalized_pair_811;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_811) : "l"(*(const unsigned long long*)&scaled_pair_810), "l"(*(const unsigned long long*)&_f32x2_mul_b_225));
        output_values_807[value_idx_808] = normalized_pair_811.x;
        output_values_807[value_idx_808 + 1] = normalized_pair_811.y;
        const int value_idx_812 = 2;
        const int acc_idx_813 = 16 + value_idx_812;
        float2 _f32x2_mul_a_226 = make_float2(acc[acc_idx_813], acc[acc_idx_813 + 1]);
        float2 scaled_pair_814;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_814) : "l"(*(const unsigned long long*)&_f32x2_mul_a_226), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_227 = make_float2(wout[acc_idx_813], wout[acc_idx_813 + 1]);
        float2 normalized_pair_815;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_815) : "l"(*(const unsigned long long*)&scaled_pair_814), "l"(*(const unsigned long long*)&_f32x2_mul_b_227));
        output_values_807[value_idx_812] = normalized_pair_815.x;
        output_values_807[value_idx_812 + 1] = normalized_pair_815.y;
        const int value_idx_816 = 4;
        const int acc_idx_817 = 16 + value_idx_816;
        float2 _f32x2_mul_a_228 = make_float2(acc[acc_idx_817], acc[acc_idx_817 + 1]);
        float2 scaled_pair_818;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_818) : "l"(*(const unsigned long long*)&_f32x2_mul_a_228), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_229 = make_float2(wout[acc_idx_817], wout[acc_idx_817 + 1]);
        float2 normalized_pair_819;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_819) : "l"(*(const unsigned long long*)&scaled_pair_818), "l"(*(const unsigned long long*)&_f32x2_mul_b_229));
        output_values_807[value_idx_816] = normalized_pair_819.x;
        output_values_807[value_idx_816 + 1] = normalized_pair_819.y;
        const int value_idx_820 = 6;
        const int acc_idx_821 = 16 + value_idx_820;
        float2 _f32x2_mul_a_230 = make_float2(acc[acc_idx_821], acc[acc_idx_821 + 1]);
        float2 scaled_pair_822;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_822) : "l"(*(const unsigned long long*)&_f32x2_mul_a_230), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_231 = make_float2(wout[acc_idx_821], wout[acc_idx_821 + 1]);
        float2 normalized_pair_823;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_823) : "l"(*(const unsigned long long*)&scaled_pair_822), "l"(*(const unsigned long long*)&_f32x2_mul_b_231));
        output_values_807[value_idx_820] = normalized_pair_823.x;
        output_values_807[value_idx_820 + 1] = normalized_pair_823.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_807[_lp*2 + 0], output_values_807[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_807[_lp*2 + 2], output_values_807[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_807[_lp*2 + 4], output_values_807[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_807[_lp*2 + 6], output_values_807[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_806))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_824 = 6144 + group * 512 + thread * 4;
        float output_values_825[8];
        const int value_idx_826 = 0;
        const int acc_idx_827 = 24 + value_idx_826;
        float2 _f32x2_mul_a_232 = make_float2(acc[acc_idx_827], acc[acc_idx_827 + 1]);
        float2 scaled_pair_828;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_828) : "l"(*(const unsigned long long*)&_f32x2_mul_a_232), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_233 = make_float2(wout[acc_idx_827], wout[acc_idx_827 + 1]);
        float2 normalized_pair_829;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_829) : "l"(*(const unsigned long long*)&scaled_pair_828), "l"(*(const unsigned long long*)&_f32x2_mul_b_233));
        output_values_825[value_idx_826] = normalized_pair_829.x;
        output_values_825[value_idx_826 + 1] = normalized_pair_829.y;
        const int value_idx_830 = 2;
        const int acc_idx_831 = 24 + value_idx_830;
        float2 _f32x2_mul_a_234 = make_float2(acc[acc_idx_831], acc[acc_idx_831 + 1]);
        float2 scaled_pair_832;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_832) : "l"(*(const unsigned long long*)&_f32x2_mul_a_234), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_235 = make_float2(wout[acc_idx_831], wout[acc_idx_831 + 1]);
        float2 normalized_pair_833;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_833) : "l"(*(const unsigned long long*)&scaled_pair_832), "l"(*(const unsigned long long*)&_f32x2_mul_b_235));
        output_values_825[value_idx_830] = normalized_pair_833.x;
        output_values_825[value_idx_830 + 1] = normalized_pair_833.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_825[_lp*2 + 0], output_values_825[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_825[_lp*2 + 2], output_values_825[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_824))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
