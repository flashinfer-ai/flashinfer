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
#define SMEM_STATS_STAGE_BYTES 384
#define SMEM_STATS_STRIDE 384
#define SMEM_OUT_STATS_OFF 384
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
kernel_cake_kimi_k3_attn_res_dbabf3db08f8c1b27616(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 384);
    const int out_stats_addr = smem + 384;

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
        unsigned int words[84];
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
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_6 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_6[0 + 0] = _uv4_6.x;
            _vec_load_6[0 + 1] = _uv4_6.y;
            _vec_load_6[0 + 2] = _uv4_6.z;
            _vec_load_6[0 + 3] = _uv4_6.w;
        }
        dwords[woff] = _vec_load_6[0];
        dwords[woff + 1] = _vec_load_6[1];
        dwords[woff + 2] = _vec_load_6[2];
        dwords[woff + 3] = _vec_load_6[3];
        float _vec_load_7[8];
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
                        : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
        }
        float _vec_load_8[8];
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
                        : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        q[0] = _vec_load_7[0] * _vec_load_8[0];
        q[1] = _vec_load_7[1] * _vec_load_8[1];
        q[2] = _vec_load_7[2] * _vec_load_8[2];
        q[3] = _vec_load_7[3] * _vec_load_8[3];
        q[4] = _vec_load_7[4] * _vec_load_8[4];
        q[5] = _vec_load_7[5] * _vec_load_8[5];
        q[6] = _vec_load_7[6] * _vec_load_8[6];
        q[7] = _vec_load_7[7] * _vec_load_8[7];
        float _vec_load_9[8];
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
                        : "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_9[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_9[_pair]));
                }
            }
        }
        wout[0] = _vec_load_9[0];
        wout[1] = _vec_load_9[1];
        wout[2] = _vec_load_9[2];
        wout[3] = _vec_load_9[3];
        wout[4] = _vec_load_9[4];
        wout[5] = _vec_load_9[5];
        wout[6] = _vec_load_9[6];
        wout[7] = _vec_load_9[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_10[4];
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_10[0 + 0] = _uv4_10.x;
            _vec_load_10[0 + 1] = _uv4_10.y;
            _vec_load_10[0 + 2] = _uv4_10.z;
            _vec_load_10[0 + 3] = _uv4_10.w;
        }
        words[woff_1] = _vec_load_10[0];
        words[woff_1 + 1] = _vec_load_10[1];
        words[woff_1 + 2] = _vec_load_10[2];
        words[woff_1 + 3] = _vec_load_10[3];
        unsigned int _vec_load_11[4];
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_11[0 + 0] = _uv4_11.x;
            _vec_load_11[0 + 1] = _uv4_11.y;
            _vec_load_11[0 + 2] = _uv4_11.z;
            _vec_load_11[0 + 3] = _uv4_11.w;
        }
        words[14 + woff_1] = _vec_load_11[0];
        words[14 + woff_1 + 1] = _vec_load_11[1];
        words[14 + woff_1 + 2] = _vec_load_11[2];
        words[14 + woff_1 + 3] = _vec_load_11[3];
        unsigned int _vec_load_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_12[0 + 0] = _uv4_12.x;
            _vec_load_12[0 + 1] = _uv4_12.y;
            _vec_load_12[0 + 2] = _uv4_12.z;
            _vec_load_12[0 + 3] = _uv4_12.w;
        }
        words[28 + woff_1] = _vec_load_12[0];
        words[28 + woff_1 + 1] = _vec_load_12[1];
        words[28 + woff_1 + 2] = _vec_load_12[2];
        words[28 + woff_1 + 3] = _vec_load_12[3];
        unsigned int _vec_load_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_13[0 + 0] = _uv4_13.x;
            _vec_load_13[0 + 1] = _uv4_13.y;
            _vec_load_13[0 + 2] = _uv4_13.z;
            _vec_load_13[0 + 3] = _uv4_13.w;
        }
        words[42 + woff_1] = _vec_load_13[0];
        words[42 + woff_1 + 1] = _vec_load_13[1];
        words[42 + woff_1 + 2] = _vec_load_13[2];
        words[42 + woff_1 + 3] = _vec_load_13[3];
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        words[56 + woff_1] = _vec_load_14[0];
        words[56 + woff_1 + 1] = _vec_load_14[1];
        words[56 + woff_1 + 2] = _vec_load_14[2];
        words[56 + woff_1 + 3] = _vec_load_14[3];
        unsigned int _vec_load_15[4];
        {
            uint4 _uv4_15 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_15[0 + 0] = _uv4_15.x;
            _vec_load_15[0 + 1] = _uv4_15.y;
            _vec_load_15[0 + 2] = _uv4_15.z;
            _vec_load_15[0 + 3] = _uv4_15.w;
        }
        words[70 + woff_1] = _vec_load_15[0];
        words[70 + woff_1 + 1] = _vec_load_15[1];
        words[70 + woff_1 + 2] = _vec_load_15[2];
        words[70 + woff_1 + 3] = _vec_load_15[3];
        unsigned int _vec_load_16[4];
        {
            uint4 _uv4_16 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_16[0 + 0] = _uv4_16.x;
            _vec_load_16[0 + 1] = _uv4_16.y;
            _vec_load_16[0 + 2] = _uv4_16.z;
            _vec_load_16[0 + 3] = _uv4_16.w;
        }
        dwords[woff_1] = _vec_load_16[0];
        dwords[woff_1 + 1] = _vec_load_16[1];
        dwords[woff_1 + 2] = _vec_load_16[2];
        dwords[woff_1 + 3] = _vec_load_16[3];
        float _vec_load_17[8];
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
                        : "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_17[_pair]));
                }
            }
        }
        float _vec_load_18[8];
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
                        : "=f"((&_vec_load_18[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_18[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_18[_pair]));
                }
            }
        }
        q[8] = _vec_load_17[0] * _vec_load_18[0];
        q[9] = _vec_load_17[1] * _vec_load_18[1];
        q[10] = _vec_load_17[2] * _vec_load_18[2];
        q[11] = _vec_load_17[3] * _vec_load_18[3];
        q[12] = _vec_load_17[4] * _vec_load_18[4];
        q[13] = _vec_load_17[5] * _vec_load_18[5];
        q[14] = _vec_load_17[6] * _vec_load_18[6];
        q[15] = _vec_load_17[7] * _vec_load_18[7];
        float _vec_load_19[8];
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
                        : "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_19[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_19[_pair]));
                }
            }
        }
        wout[8] = _vec_load_19[0];
        wout[9] = _vec_load_19[1];
        wout[10] = _vec_load_19[2];
        wout[11] = _vec_load_19[3];
        wout[12] = _vec_load_19[4];
        wout[13] = _vec_load_19[5];
        wout[14] = _vec_load_19[6];
        wout[15] = _vec_load_19[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_20[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_20[0 + 0] = _uv4_20.x;
            _vec_load_20[0 + 1] = _uv4_20.y;
            _vec_load_20[0 + 2] = _uv4_20.z;
            _vec_load_20[0 + 3] = _uv4_20.w;
        }
        words[woff_3] = _vec_load_20[0];
        words[woff_3 + 1] = _vec_load_20[1];
        words[woff_3 + 2] = _vec_load_20[2];
        words[woff_3 + 3] = _vec_load_20[3];
        unsigned int _vec_load_21[4];
        {
            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_21[0 + 0] = _uv4_21.x;
            _vec_load_21[0 + 1] = _uv4_21.y;
            _vec_load_21[0 + 2] = _uv4_21.z;
            _vec_load_21[0 + 3] = _uv4_21.w;
        }
        words[14 + woff_3] = _vec_load_21[0];
        words[14 + woff_3 + 1] = _vec_load_21[1];
        words[14 + woff_3 + 2] = _vec_load_21[2];
        words[14 + woff_3 + 3] = _vec_load_21[3];
        unsigned int _vec_load_22[4];
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_22[0 + 0] = _uv4_22.x;
            _vec_load_22[0 + 1] = _uv4_22.y;
            _vec_load_22[0 + 2] = _uv4_22.z;
            _vec_load_22[0 + 3] = _uv4_22.w;
        }
        words[28 + woff_3] = _vec_load_22[0];
        words[28 + woff_3 + 1] = _vec_load_22[1];
        words[28 + woff_3 + 2] = _vec_load_22[2];
        words[28 + woff_3 + 3] = _vec_load_22[3];
        unsigned int _vec_load_23[4];
        {
            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_23[0 + 0] = _uv4_23.x;
            _vec_load_23[0 + 1] = _uv4_23.y;
            _vec_load_23[0 + 2] = _uv4_23.z;
            _vec_load_23[0 + 3] = _uv4_23.w;
        }
        words[42 + woff_3] = _vec_load_23[0];
        words[42 + woff_3 + 1] = _vec_load_23[1];
        words[42 + woff_3 + 2] = _vec_load_23[2];
        words[42 + woff_3 + 3] = _vec_load_23[3];
        unsigned int _vec_load_24[4];
        {
            uint4 _uv4_24 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_24[0 + 0] = _uv4_24.x;
            _vec_load_24[0 + 1] = _uv4_24.y;
            _vec_load_24[0 + 2] = _uv4_24.z;
            _vec_load_24[0 + 3] = _uv4_24.w;
        }
        words[56 + woff_3] = _vec_load_24[0];
        words[56 + woff_3 + 1] = _vec_load_24[1];
        words[56 + woff_3 + 2] = _vec_load_24[2];
        words[56 + woff_3 + 3] = _vec_load_24[3];
        unsigned int _vec_load_25[4];
        {
            uint4 _uv4_25 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_25[0 + 0] = _uv4_25.x;
            _vec_load_25[0 + 1] = _uv4_25.y;
            _vec_load_25[0 + 2] = _uv4_25.z;
            _vec_load_25[0 + 3] = _uv4_25.w;
        }
        words[70 + woff_3] = _vec_load_25[0];
        words[70 + woff_3 + 1] = _vec_load_25[1];
        words[70 + woff_3 + 2] = _vec_load_25[2];
        words[70 + woff_3 + 3] = _vec_load_25[3];
        unsigned int _vec_load_26[4];
        {
            uint4 _uv4_26 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_26[0 + 0] = _uv4_26.x;
            _vec_load_26[0 + 1] = _uv4_26.y;
            _vec_load_26[0 + 2] = _uv4_26.z;
            _vec_load_26[0 + 3] = _uv4_26.w;
        }
        dwords[woff_3] = _vec_load_26[0];
        dwords[woff_3 + 1] = _vec_load_26[1];
        dwords[woff_3 + 2] = _vec_load_26[2];
        dwords[woff_3 + 3] = _vec_load_26[3];
        float _vec_load_27[8];
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
                        : "=f"((&_vec_load_27[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_27[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_27[_pair]));
                }
            }
        }
        float _vec_load_28[8];
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
                        : "=f"((&_vec_load_28[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_28[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_28[_pair]));
                }
            }
        }
        q[16] = _vec_load_27[0] * _vec_load_28[0];
        q[17] = _vec_load_27[1] * _vec_load_28[1];
        q[18] = _vec_load_27[2] * _vec_load_28[2];
        q[19] = _vec_load_27[3] * _vec_load_28[3];
        q[20] = _vec_load_27[4] * _vec_load_28[4];
        q[21] = _vec_load_27[5] * _vec_load_28[5];
        q[22] = _vec_load_27[6] * _vec_load_28[6];
        q[23] = _vec_load_27[7] * _vec_load_28[7];
        float _vec_load_29[8];
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
                        : "=f"((&_vec_load_29[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_29[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_29[_pair]));
                }
            }
        }
        wout[16] = _vec_load_29[0];
        wout[17] = _vec_load_29[1];
        wout[18] = _vec_load_29[2];
        wout[19] = _vec_load_29[3];
        wout[20] = _vec_load_29[4];
        wout[21] = _vec_load_29[5];
        wout[22] = _vec_load_29[6];
        wout[23] = _vec_load_29[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_30[1];
        {
            _vec_load_30[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_30[0];
        unsigned int _vec_load_31[1];
        {
            _vec_load_31[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_31[0];
        unsigned int _vec_load_32[1];
        {
            _vec_load_32[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_32[0];
        unsigned int _vec_load_33[1];
        {
            _vec_load_33[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_33[0];
        unsigned int _vec_load_34[1];
        {
            _vec_load_34[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_34[0];
        unsigned int _vec_load_35[1];
        {
            _vec_load_35[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_35[0];
        unsigned int _vec_load_36[1];
        {
            _vec_load_36[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[42 + woff_5] = _vec_load_36[0];
        unsigned int _vec_load_37[1];
        {
            _vec_load_37[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[42 + woff_5 + 1] = _vec_load_37[0];
        unsigned int _vec_load_38[1];
        {
            _vec_load_38[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[56 + woff_5] = _vec_load_38[0];
        unsigned int _vec_load_39[1];
        {
            _vec_load_39[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 4 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[56 + woff_5 + 1] = _vec_load_39[0];
        unsigned int _vec_load_40[1];
        {
            _vec_load_40[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[70 + woff_5] = _vec_load_40[0];
        unsigned int _vec_load_41[1];
        {
            _vec_load_41[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[70 + woff_5 + 1] = _vec_load_41[0];
        unsigned int _vec_load_42[1];
        {
            _vec_load_42[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_42[0];
        unsigned int _vec_load_43[1];
        {
            _vec_load_43[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_43[0];
        float _vec_load_44[4];
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
                    : "=f"((&_vec_load_44[0 + _pair * 2])[0]), "=f"((&_vec_load_44[0 + _pair * 2])[1])
                    : "r"(_vpairs_30[_pair]));
            }
        }
        float _vec_load_45[4];
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
                    : "=f"((&_vec_load_45[0 + _pair * 2])[0]), "=f"((&_vec_load_45[0 + _pair * 2])[1])
                    : "r"(_vpairs_31[_pair]));
            }
        }
        q[24] = _vec_load_44[0] * _vec_load_45[0];
        q[25] = _vec_load_44[1] * _vec_load_45[1];
        q[26] = _vec_load_44[2] * _vec_load_45[2];
        q[27] = _vec_load_44[3] * _vec_load_45[3];
        float _vec_load_46[4];
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
                    : "=f"((&_vec_load_46[0 + _pair * 2])[0]), "=f"((&_vec_load_46[0 + _pair * 2])[1])
                    : "r"(_vpairs_32[_pair]));
            }
        }
        wout[24] = _vec_load_46[0];
        wout[25] = _vec_load_46[1];
        wout[26] = _vec_load_46[2];
        wout[27] = _vec_load_46[3];
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
        __nv_bfloat162 a = __as_bf16x2(sw_50[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_50[0] = __as_u32(mixed);
        words[70 + woff_7] = sw_50[0];
        __nv_bfloat162 a_51 = __as_bf16x2(sw_50[1]);
        __nv_bfloat162 d_52 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_53 = a_51 + d_52;
        sw_50[1] = __as_u32(mixed_53);
        words[70 + woff_7 + 1] = sw_50[1];
        __nv_bfloat162 a_54 = __as_bf16x2(sw_50[2]);
        __nv_bfloat162 d_55 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_56 = a_54 + d_55;
        sw_50[2] = __as_u32(mixed_56);
        words[70 + woff_7 + 2] = sw_50[2];
        __nv_bfloat162 a_57 = __as_bf16x2(sw_50[3]);
        __nv_bfloat162 d_58 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_59 = a_57 + d_58;
        sw_50[3] = __as_u32(mixed_59);
        words[70 + woff_7 + 3] = sw_50[3];
        {
            int4 _iv4 = make_int4(sw_50[0 + 0], sw_50[0 + 1], sw_50[0 + 2], sw_50[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
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
                : "r"(sw_50[_pair]));
        }
        float2 v_60 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 qp_61 = make_float2(q[0], q[1]);
        sq[5] = fma_f32x2_rn_noftz(v_60, v_60, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_60, qp_61, dot[5]);
        float2 v_62 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 qp_63 = make_float2(q[2], q[3]);
        sq[5] = fma_f32x2_rn_noftz(v_62, v_62, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_62, qp_63, dot[5]);
        float2 v_64 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 qp_65 = make_float2(q[4], q[5]);
        sq[5] = fma_f32x2_rn_noftz(v_64, v_64, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_64, qp_65, dot[5]);
        float2 v_66 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 qp_67 = make_float2(q[6], q[7]);
        sq[5] = fma_f32x2_rn_noftz(v_66, v_66, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_66, qp_67, dot[5]);
        int base_68 = (2 + group) * 1024 + thread * 8;
        const int woff_69 = 4;
        unsigned int sw_70[4];
        sw_70[0] = words[woff_69];
        sw_70[1] = words[woff_69 + 1];
        sw_70[2] = words[woff_69 + 2];
        sw_70[3] = words[woff_69 + 3];
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_6[_pair * 2])[0]), "=f"((&_unpack_bf16_6[_pair * 2])[1])
                : "r"(sw_70[_pair]));
        }
        fsrc[8] = _unpack_bf16_6[0];
        fsrc[9] = _unpack_bf16_6[1];
        fsrc[10] = _unpack_bf16_6[2];
        fsrc[11] = _unpack_bf16_6[3];
        fsrc[12] = _unpack_bf16_6[4];
        fsrc[13] = _unpack_bf16_6[5];
        fsrc[14] = _unpack_bf16_6[6];
        fsrc[15] = _unpack_bf16_6[7];
        float2 v_71 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 qp_72 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_71, v_71, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_71, qp_72, dot[0]);
        float2 v_73 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 qp_74 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_73, v_73, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_73, qp_74, dot[0]);
        float2 v_75 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 qp_76 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_75, v_75, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_75, qp_76, dot[0]);
        float2 v_77 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 qp_78 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_77, v_77, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_77, qp_78, dot[0]);
        unsigned int sw_79[4];
        sw_79[0] = words[14 + woff_69];
        sw_79[1] = words[14 + woff_69 + 1];
        sw_79[2] = words[14 + woff_69 + 2];
        sw_79[3] = words[14 + woff_69 + 3];
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
        fsrc[36] = _unpack_bf16_7[0];
        fsrc[37] = _unpack_bf16_7[1];
        fsrc[38] = _unpack_bf16_7[2];
        fsrc[39] = _unpack_bf16_7[3];
        fsrc[40] = _unpack_bf16_7[4];
        fsrc[41] = _unpack_bf16_7[5];
        fsrc[42] = _unpack_bf16_7[6];
        fsrc[43] = _unpack_bf16_7[7];
        float2 v_80 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 qp_81 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_80, v_80, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_80, qp_81, dot[1]);
        float2 v_82 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 qp_83 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_82, v_82, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_82, qp_83, dot[1]);
        float2 v_84 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 qp_85 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_84, v_84, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_84, qp_85, dot[1]);
        float2 v_86 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 qp_87 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_86, v_86, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_86, qp_87, dot[1]);
        unsigned int sw_88[4];
        sw_88[0] = words[28 + woff_69];
        sw_88[1] = words[28 + woff_69 + 1];
        sw_88[2] = words[28 + woff_69 + 2];
        sw_88[3] = words[28 + woff_69 + 3];
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
        fsrc[64] = _unpack_bf16_8[0];
        fsrc[65] = _unpack_bf16_8[1];
        fsrc[66] = _unpack_bf16_8[2];
        fsrc[67] = _unpack_bf16_8[3];
        fsrc[68] = _unpack_bf16_8[4];
        fsrc[69] = _unpack_bf16_8[5];
        fsrc[70] = _unpack_bf16_8[6];
        fsrc[71] = _unpack_bf16_8[7];
        float2 v_89 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 qp_90 = make_float2(q[8], q[9]);
        sq[2] = fma_f32x2_rn_noftz(v_89, v_89, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_89, qp_90, dot[2]);
        float2 v_91 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 qp_92 = make_float2(q[10], q[11]);
        sq[2] = fma_f32x2_rn_noftz(v_91, v_91, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_91, qp_92, dot[2]);
        float2 v_93 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 qp_94 = make_float2(q[12], q[13]);
        sq[2] = fma_f32x2_rn_noftz(v_93, v_93, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_93, qp_94, dot[2]);
        float2 v_95 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 qp_96 = make_float2(q[14], q[15]);
        sq[2] = fma_f32x2_rn_noftz(v_95, v_95, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_95, qp_96, dot[2]);
        unsigned int sw_97[4];
        sw_97[0] = words[42 + woff_69];
        sw_97[1] = words[42 + woff_69 + 1];
        sw_97[2] = words[42 + woff_69 + 2];
        sw_97[3] = words[42 + woff_69 + 3];
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
        float2 v_98 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 qp_99 = make_float2(q[8], q[9]);
        sq[3] = fma_f32x2_rn_noftz(v_98, v_98, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_98, qp_99, dot[3]);
        float2 v_100 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 qp_101 = make_float2(q[10], q[11]);
        sq[3] = fma_f32x2_rn_noftz(v_100, v_100, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_100, qp_101, dot[3]);
        float2 v_102 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 qp_103 = make_float2(q[12], q[13]);
        sq[3] = fma_f32x2_rn_noftz(v_102, v_102, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_102, qp_103, dot[3]);
        float2 v_104 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 qp_105 = make_float2(q[14], q[15]);
        sq[3] = fma_f32x2_rn_noftz(v_104, v_104, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_104, qp_105, dot[3]);
        unsigned int sw_106[4];
        sw_106[0] = words[56 + woff_69];
        sw_106[1] = words[56 + woff_69 + 1];
        sw_106[2] = words[56 + woff_69 + 2];
        sw_106[3] = words[56 + woff_69 + 3];
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
        sq[4] = fma_f32x2_rn_noftz(v_107, v_107, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_107, qp_108, dot[4]);
        float2 v_109 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 qp_110 = make_float2(q[10], q[11]);
        sq[4] = fma_f32x2_rn_noftz(v_109, v_109, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_109, qp_110, dot[4]);
        float2 v_111 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 qp_112 = make_float2(q[12], q[13]);
        sq[4] = fma_f32x2_rn_noftz(v_111, v_111, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_111, qp_112, dot[4]);
        float2 v_113 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 qp_114 = make_float2(q[14], q[15]);
        sq[4] = fma_f32x2_rn_noftz(v_113, v_113, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_113, qp_114, dot[4]);
        unsigned int sw_115[4];
        sw_115[0] = words[70 + woff_69];
        sw_115[1] = words[70 + woff_69 + 1];
        sw_115[2] = words[70 + woff_69 + 2];
        sw_115[3] = words[70 + woff_69 + 3];
        __nv_bfloat162 a_116 = __as_bf16x2(sw_115[0]);
        __nv_bfloat162 d_117 = __as_bf16x2(dwords[woff_69]);
        __nv_bfloat162 mixed_118 = a_116 + d_117;
        sw_115[0] = __as_u32(mixed_118);
        words[70 + woff_69] = sw_115[0];
        __nv_bfloat162 a_119 = __as_bf16x2(sw_115[1]);
        __nv_bfloat162 d_120 = __as_bf16x2(dwords[woff_69 + 1]);
        __nv_bfloat162 mixed_121 = a_119 + d_120;
        sw_115[1] = __as_u32(mixed_121);
        words[70 + woff_69 + 1] = sw_115[1];
        __nv_bfloat162 a_122 = __as_bf16x2(sw_115[2]);
        __nv_bfloat162 d_123 = __as_bf16x2(dwords[woff_69 + 2]);
        __nv_bfloat162 mixed_124 = a_122 + d_123;
        sw_115[2] = __as_u32(mixed_124);
        words[70 + woff_69 + 2] = sw_115[2];
        __nv_bfloat162 a_125 = __as_bf16x2(sw_115[3]);
        __nv_bfloat162 d_126 = __as_bf16x2(dwords[woff_69 + 3]);
        __nv_bfloat162 mixed_127 = a_125 + d_126;
        sw_115[3] = __as_u32(mixed_127);
        words[70 + woff_69 + 3] = sw_115[3];
        {
            int4 _iv4 = make_int4(sw_115[0 + 0], sw_115[0 + 1], sw_115[0 + 2], sw_115[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_68)) + 0) = _iv4;
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
                : "r"(sw_115[_pair]));
        }
        float2 v_128 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 qp_129 = make_float2(q[8], q[9]);
        sq[5] = fma_f32x2_rn_noftz(v_128, v_128, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_128, qp_129, dot[5]);
        float2 v_130 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 qp_131 = make_float2(q[10], q[11]);
        sq[5] = fma_f32x2_rn_noftz(v_130, v_130, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_130, qp_131, dot[5]);
        float2 v_132 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 qp_133 = make_float2(q[12], q[13]);
        sq[5] = fma_f32x2_rn_noftz(v_132, v_132, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_132, qp_133, dot[5]);
        float2 v_134 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 qp_135 = make_float2(q[14], q[15]);
        sq[5] = fma_f32x2_rn_noftz(v_134, v_134, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_134, qp_135, dot[5]);
        int base_136 = (4 + group) * 1024 + thread * 8;
        const int woff_137 = 8;
        unsigned int sw_138[4];
        sw_138[0] = words[woff_137];
        sw_138[1] = words[woff_137 + 1];
        sw_138[2] = words[woff_137 + 2];
        sw_138[3] = words[woff_137 + 3];
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                : "r"(sw_138[_pair]));
        }
        fsrc[16] = _unpack_bf16_12[0];
        fsrc[17] = _unpack_bf16_12[1];
        fsrc[18] = _unpack_bf16_12[2];
        fsrc[19] = _unpack_bf16_12[3];
        fsrc[20] = _unpack_bf16_12[4];
        fsrc[21] = _unpack_bf16_12[5];
        fsrc[22] = _unpack_bf16_12[6];
        fsrc[23] = _unpack_bf16_12[7];
        float2 v_139 = make_float2(_unpack_bf16_12[0], _unpack_bf16_12[1]);
        float2 qp_140 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_139, v_139, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_139, qp_140, dot[0]);
        float2 v_141 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 qp_142 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_141, v_141, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_141, qp_142, dot[0]);
        float2 v_143 = make_float2(_unpack_bf16_12[4], _unpack_bf16_12[5]);
        float2 qp_144 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_143, v_143, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_143, qp_144, dot[0]);
        float2 v_145 = make_float2(_unpack_bf16_12[6], _unpack_bf16_12[7]);
        float2 qp_146 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_145, v_145, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_145, qp_146, dot[0]);
        unsigned int sw_147[4];
        sw_147[0] = words[14 + woff_137];
        sw_147[1] = words[14 + woff_137 + 1];
        sw_147[2] = words[14 + woff_137 + 2];
        sw_147[3] = words[14 + woff_137 + 3];
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                : "r"(sw_147[_pair]));
        }
        fsrc[44] = _unpack_bf16_13[0];
        fsrc[45] = _unpack_bf16_13[1];
        fsrc[46] = _unpack_bf16_13[2];
        fsrc[47] = _unpack_bf16_13[3];
        fsrc[48] = _unpack_bf16_13[4];
        fsrc[49] = _unpack_bf16_13[5];
        fsrc[50] = _unpack_bf16_13[6];
        fsrc[51] = _unpack_bf16_13[7];
        float2 v_148 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 qp_149 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_148, v_148, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_148, qp_149, dot[1]);
        float2 v_150 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 qp_151 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_150, v_150, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_150, qp_151, dot[1]);
        float2 v_152 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
        float2 qp_153 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_152, v_152, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_152, qp_153, dot[1]);
        float2 v_154 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
        float2 qp_155 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_154, v_154, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_154, qp_155, dot[1]);
        unsigned int sw_156[4];
        sw_156[0] = words[28 + woff_137];
        sw_156[1] = words[28 + woff_137 + 1];
        sw_156[2] = words[28 + woff_137 + 2];
        sw_156[3] = words[28 + woff_137 + 3];
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
        fsrc[72] = _unpack_bf16_14[0];
        fsrc[73] = _unpack_bf16_14[1];
        fsrc[74] = _unpack_bf16_14[2];
        fsrc[75] = _unpack_bf16_14[3];
        fsrc[76] = _unpack_bf16_14[4];
        fsrc[77] = _unpack_bf16_14[5];
        fsrc[78] = _unpack_bf16_14[6];
        fsrc[79] = _unpack_bf16_14[7];
        float2 v_157 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 qp_158 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_157, v_157, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_157, qp_158, dot[2]);
        float2 v_159 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 qp_160 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_159, v_159, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_159, qp_160, dot[2]);
        float2 v_161 = make_float2(_unpack_bf16_14[4], _unpack_bf16_14[5]);
        float2 qp_162 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_161, v_161, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_161, qp_162, dot[2]);
        float2 v_163 = make_float2(_unpack_bf16_14[6], _unpack_bf16_14[7]);
        float2 qp_164 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_163, v_163, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_163, qp_164, dot[2]);
        unsigned int sw_165[4];
        sw_165[0] = words[42 + woff_137];
        sw_165[1] = words[42 + woff_137 + 1];
        sw_165[2] = words[42 + woff_137 + 2];
        sw_165[3] = words[42 + woff_137 + 3];
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
        float2 v_166 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_167 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_166, v_166, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_166, qp_167, dot[3]);
        float2 v_168 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_169 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_168, v_168, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_168, qp_169, dot[3]);
        float2 v_170 = make_float2(_unpack_bf16_15[4], _unpack_bf16_15[5]);
        float2 qp_171 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_170, v_170, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_170, qp_171, dot[3]);
        float2 v_172 = make_float2(_unpack_bf16_15[6], _unpack_bf16_15[7]);
        float2 qp_173 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_172, v_172, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_172, qp_173, dot[3]);
        unsigned int sw_174[4];
        sw_174[0] = words[56 + woff_137];
        sw_174[1] = words[56 + woff_137 + 1];
        sw_174[2] = words[56 + woff_137 + 2];
        sw_174[3] = words[56 + woff_137 + 3];
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
        float2 v_175 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_176 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_175, v_175, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_175, qp_176, dot[4]);
        float2 v_177 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_178 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_177, v_177, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_177, qp_178, dot[4]);
        float2 v_179 = make_float2(_unpack_bf16_16[4], _unpack_bf16_16[5]);
        float2 qp_180 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_179, v_179, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_179, qp_180, dot[4]);
        float2 v_181 = make_float2(_unpack_bf16_16[6], _unpack_bf16_16[7]);
        float2 qp_182 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_181, v_181, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_181, qp_182, dot[4]);
        unsigned int sw_183[4];
        sw_183[0] = words[70 + woff_137];
        sw_183[1] = words[70 + woff_137 + 1];
        sw_183[2] = words[70 + woff_137 + 2];
        sw_183[3] = words[70 + woff_137 + 3];
        __nv_bfloat162 a_184 = __as_bf16x2(sw_183[0]);
        __nv_bfloat162 d_185 = __as_bf16x2(dwords[woff_137]);
        __nv_bfloat162 mixed_186 = a_184 + d_185;
        sw_183[0] = __as_u32(mixed_186);
        words[70 + woff_137] = sw_183[0];
        __nv_bfloat162 a_187 = __as_bf16x2(sw_183[1]);
        __nv_bfloat162 d_188 = __as_bf16x2(dwords[woff_137 + 1]);
        __nv_bfloat162 mixed_189 = a_187 + d_188;
        sw_183[1] = __as_u32(mixed_189);
        words[70 + woff_137 + 1] = sw_183[1];
        __nv_bfloat162 a_190 = __as_bf16x2(sw_183[2]);
        __nv_bfloat162 d_191 = __as_bf16x2(dwords[woff_137 + 2]);
        __nv_bfloat162 mixed_192 = a_190 + d_191;
        sw_183[2] = __as_u32(mixed_192);
        words[70 + woff_137 + 2] = sw_183[2];
        __nv_bfloat162 a_193 = __as_bf16x2(sw_183[3]);
        __nv_bfloat162 d_194 = __as_bf16x2(dwords[woff_137 + 3]);
        __nv_bfloat162 mixed_195 = a_193 + d_194;
        sw_183[3] = __as_u32(mixed_195);
        words[70 + woff_137 + 3] = sw_183[3];
        {
            int4 _iv4 = make_int4(sw_183[0 + 0], sw_183[0 + 1], sw_183[0 + 2], sw_183[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_136)) + 0) = _iv4;
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
                : "r"(sw_183[_pair]));
        }
        float2 v_196 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_197 = make_float2(q[16], q[17]);
        sq[5] = fma_f32x2_rn_noftz(v_196, v_196, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_196, qp_197, dot[5]);
        float2 v_198 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_199 = make_float2(q[18], q[19]);
        sq[5] = fma_f32x2_rn_noftz(v_198, v_198, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_198, qp_199, dot[5]);
        float2 v_200 = make_float2(_unpack_bf16_17[4], _unpack_bf16_17[5]);
        float2 qp_201 = make_float2(q[20], q[21]);
        sq[5] = fma_f32x2_rn_noftz(v_200, v_200, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_200, qp_201, dot[5]);
        float2 v_202 = make_float2(_unpack_bf16_17[6], _unpack_bf16_17[7]);
        float2 qp_203 = make_float2(q[22], q[23]);
        sq[5] = fma_f32x2_rn_noftz(v_202, v_202, sq[5]);
        dot[5] = fma_f32x2_rn_noftz(v_202, qp_203, dot[5]);
        int base_204 = 6144 + group * 512 + thread * 4;
        const int woff_205 = 12;
        unsigned int sw_206[4];
        sw_206[0] = words[woff_205];
        sw_206[1] = words[woff_205 + 1];
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                : "r"(sw_206[_pair]));
        }
        fsrc[24] = _unpack_bf16_18[0];
        fsrc[25] = _unpack_bf16_18[1];
        fsrc[26] = _unpack_bf16_18[2];
        fsrc[27] = _unpack_bf16_18[3];
        float2 v_207 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        sq[0] = fma_f32x2_rn_noftz(v_207, v_207, sq[0]);
        float2 v_208 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        sq[0] = fma_f32x2_rn_noftz(v_208, v_208, sq[0]);
        float2 v_209 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_210 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_209, qp_210, dot[0]);
        float2 v_211 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_212 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_211, qp_212, dot[0]);
        unsigned int sw_213[4];
        sw_213[0] = words[14 + woff_205];
        sw_213[1] = words[14 + woff_205 + 1];
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                : "r"(sw_213[_pair]));
        }
        fsrc[52] = _unpack_bf16_19[0];
        fsrc[53] = _unpack_bf16_19[1];
        fsrc[54] = _unpack_bf16_19[2];
        fsrc[55] = _unpack_bf16_19[3];
        float2 v_214 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        sq[1] = fma_f32x2_rn_noftz(v_214, v_214, sq[1]);
        float2 v_215 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        sq[1] = fma_f32x2_rn_noftz(v_215, v_215, sq[1]);
        float2 v_216 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 qp_217 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_216, qp_217, dot[1]);
        float2 v_218 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_219 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_218, qp_219, dot[1]);
        unsigned int sw_220[4];
        sw_220[0] = words[28 + woff_205];
        sw_220[1] = words[28 + woff_205 + 1];
        float _unpack_bf16_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                : "r"(sw_220[_pair]));
        }
        fsrc[80] = _unpack_bf16_20[0];
        fsrc[81] = _unpack_bf16_20[1];
        fsrc[82] = _unpack_bf16_20[2];
        fsrc[83] = _unpack_bf16_20[3];
        float2 v_221 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        sq[2] = fma_f32x2_rn_noftz(v_221, v_221, sq[2]);
        float2 v_222 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        sq[2] = fma_f32x2_rn_noftz(v_222, v_222, sq[2]);
        float2 v_223 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        float2 qp_224 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_223, qp_224, dot[2]);
        float2 v_225 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        float2 qp_226 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_225, qp_226, dot[2]);
        unsigned int sw_227[4];
        sw_227[0] = words[42 + woff_205];
        sw_227[1] = words[42 + woff_205 + 1];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw_227[_pair]));
        }
        float2 v_228 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        sq[3] = fma_f32x2_rn_noftz(v_228, v_228, sq[3]);
        float2 v_229 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        sq[3] = fma_f32x2_rn_noftz(v_229, v_229, sq[3]);
        float2 v_230 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        float2 qp_231 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_230, qp_231, dot[3]);
        float2 v_232 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        float2 qp_233 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_232, qp_233, dot[3]);
        unsigned int sw_234[4];
        sw_234[0] = words[56 + woff_205];
        sw_234[1] = words[56 + woff_205 + 1];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw_234[_pair]));
        }
        float2 v_235 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        sq[4] = fma_f32x2_rn_noftz(v_235, v_235, sq[4]);
        float2 v_236 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        sq[4] = fma_f32x2_rn_noftz(v_236, v_236, sq[4]);
        float2 v_237 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        float2 qp_238 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_237, qp_238, dot[4]);
        float2 v_239 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        float2 qp_240 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_239, qp_240, dot[4]);
        unsigned int sw_241[4];
        sw_241[0] = words[70 + woff_205];
        sw_241[1] = words[70 + woff_205 + 1];
        __nv_bfloat162 a_242 = __as_bf16x2(sw_241[0]);
        __nv_bfloat162 d_243 = __as_bf16x2(dwords[woff_205]);
        __nv_bfloat162 mixed_244 = a_242 + d_243;
        sw_241[0] = __as_u32(mixed_244);
        words[70 + woff_205] = sw_241[0];
        __nv_bfloat162 a_245 = __as_bf16x2(sw_241[1]);
        __nv_bfloat162 d_246 = __as_bf16x2(dwords[woff_205 + 1]);
        __nv_bfloat162 mixed_247 = a_245 + d_246;
        sw_241[1] = __as_u32(mixed_247);
        words[70 + woff_205 + 1] = sw_241[1];
        {
            int2 _iv2 = make_int2(sw_241[0 + 0], sw_241[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_204)) + 0) = _iv2;
        }
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                : "r"(sw_241[_pair]));
        }
        float2 v_248 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        sq[5] = fma_f32x2_rn_noftz(v_248, v_248, sq[5]);
        float2 v_249 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        sq[5] = fma_f32x2_rn_noftz(v_249, v_249, sq[5]);
        float2 v_250 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        float2 qp_251 = make_float2(q[24], q[25]);
        dot[5] = fma_f32x2_rn_noftz(v_250, qp_251, dot[5]);
        float2 v_252 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        float2 qp_253 = make_float2(q[26], q[27]);
        dot[5] = fma_f32x2_rn_noftz(v_252, qp_253, dot[5]);
        float2 pairs[6];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[2] = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[3] = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[4] = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        pairs[5] = make_float2(sq[5].x + sq[5].y, dot[5].x + dot[5].y);
        unsigned long long bits = 0;
        float2 _bit_cast_src_33 = pairs[0];
        bits = reinterpret_cast<unsigned long long*>(&_bit_cast_src_33)[0];
        unsigned long long peerbits = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        float2 peer = make_float2(0.0f, 0.0f);
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_254 = 0;
        float2 _bit_cast_src_34 = pairs[1];
        bits_254 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_34)[0];
        unsigned long long peerbits_255 = __shfl_xor_sync(0xFFFFFFFF, bits_254, 16);
        float2 peer_256 = make_float2(0.0f, 0.0f);
        peer_256 = reinterpret_cast<float2*>(&peerbits_255)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_256);
        unsigned long long bits_257 = 0;
        float2 _bit_cast_src_35 = pairs[2];
        bits_257 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_35)[0];
        unsigned long long peerbits_258 = __shfl_xor_sync(0xFFFFFFFF, bits_257, 16);
        float2 peer_259 = make_float2(0.0f, 0.0f);
        peer_259 = reinterpret_cast<float2*>(&peerbits_258)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_259);
        unsigned long long bits_260 = 0;
        float2 _bit_cast_src_36 = pairs[3];
        bits_260 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_36)[0];
        unsigned long long peerbits_261 = __shfl_xor_sync(0xFFFFFFFF, bits_260, 16);
        float2 peer_262 = make_float2(0.0f, 0.0f);
        peer_262 = reinterpret_cast<float2*>(&peerbits_261)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_262);
        unsigned long long bits_263 = 0;
        float2 _bit_cast_src_37 = pairs[4];
        bits_263 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_37)[0];
        unsigned long long peerbits_264 = __shfl_xor_sync(0xFFFFFFFF, bits_263, 16);
        float2 peer_265 = make_float2(0.0f, 0.0f);
        peer_265 = reinterpret_cast<float2*>(&peerbits_264)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_265);
        unsigned long long bits_266 = 0;
        float2 _bit_cast_src_38 = pairs[5];
        bits_266 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_38)[0];
        unsigned long long peerbits_267 = __shfl_xor_sync(0xFFFFFFFF, bits_266, 16);
        float2 peer_268 = make_float2(0.0f, 0.0f);
        peer_268 = reinterpret_cast<float2*>(&peerbits_267)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_268);
        unsigned long long bits_269 = 0;
        float2 _bit_cast_src_39 = pairs[0];
        bits_269 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_39)[0];
        unsigned long long peerbits_270 = __shfl_xor_sync(0xFFFFFFFF, bits_269, 8);
        float2 peer_271 = make_float2(0.0f, 0.0f);
        peer_271 = reinterpret_cast<float2*>(&peerbits_270)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_271);
        unsigned long long bits_272 = 0;
        float2 _bit_cast_src_40 = pairs[1];
        bits_272 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_40)[0];
        unsigned long long peerbits_273 = __shfl_xor_sync(0xFFFFFFFF, bits_272, 8);
        float2 peer_274 = make_float2(0.0f, 0.0f);
        peer_274 = reinterpret_cast<float2*>(&peerbits_273)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_274);
        unsigned long long bits_275 = 0;
        float2 _bit_cast_src_41 = pairs[2];
        bits_275 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_41)[0];
        unsigned long long peerbits_276 = __shfl_xor_sync(0xFFFFFFFF, bits_275, 8);
        float2 peer_277 = make_float2(0.0f, 0.0f);
        peer_277 = reinterpret_cast<float2*>(&peerbits_276)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_277);
        unsigned long long bits_278 = 0;
        float2 _bit_cast_src_42 = pairs[3];
        bits_278 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits_279 = __shfl_xor_sync(0xFFFFFFFF, bits_278, 8);
        float2 peer_280 = make_float2(0.0f, 0.0f);
        peer_280 = reinterpret_cast<float2*>(&peerbits_279)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_280);
        unsigned long long bits_281 = 0;
        float2 _bit_cast_src_43 = pairs[4];
        bits_281 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_282 = __shfl_xor_sync(0xFFFFFFFF, bits_281, 8);
        float2 peer_283 = make_float2(0.0f, 0.0f);
        peer_283 = reinterpret_cast<float2*>(&peerbits_282)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_283);
        unsigned long long bits_284 = 0;
        float2 _bit_cast_src_44 = pairs[5];
        bits_284 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_285 = __shfl_xor_sync(0xFFFFFFFF, bits_284, 8);
        float2 peer_286 = make_float2(0.0f, 0.0f);
        peer_286 = reinterpret_cast<float2*>(&peerbits_285)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_286);
        unsigned long long bits_287 = 0;
        float2 _bit_cast_src_45 = pairs[0];
        bits_287 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_288 = __shfl_xor_sync(0xFFFFFFFF, bits_287, 4);
        float2 peer_289 = make_float2(0.0f, 0.0f);
        peer_289 = reinterpret_cast<float2*>(&peerbits_288)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_289);
        unsigned long long bits_290 = 0;
        float2 _bit_cast_src_46 = pairs[1];
        bits_290 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_291 = __shfl_xor_sync(0xFFFFFFFF, bits_290, 4);
        float2 peer_292 = make_float2(0.0f, 0.0f);
        peer_292 = reinterpret_cast<float2*>(&peerbits_291)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_292);
        unsigned long long bits_293 = 0;
        float2 _bit_cast_src_47 = pairs[2];
        bits_293 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_294 = __shfl_xor_sync(0xFFFFFFFF, bits_293, 4);
        float2 peer_295 = make_float2(0.0f, 0.0f);
        peer_295 = reinterpret_cast<float2*>(&peerbits_294)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_295);
        unsigned long long bits_296 = 0;
        float2 _bit_cast_src_48 = pairs[3];
        bits_296 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_297 = __shfl_xor_sync(0xFFFFFFFF, bits_296, 4);
        float2 peer_298 = make_float2(0.0f, 0.0f);
        peer_298 = reinterpret_cast<float2*>(&peerbits_297)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_298);
        unsigned long long bits_299 = 0;
        float2 _bit_cast_src_49 = pairs[4];
        bits_299 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_300 = __shfl_xor_sync(0xFFFFFFFF, bits_299, 4);
        float2 peer_301 = make_float2(0.0f, 0.0f);
        peer_301 = reinterpret_cast<float2*>(&peerbits_300)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_301);
        unsigned long long bits_302 = 0;
        float2 _bit_cast_src_50 = pairs[5];
        bits_302 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_303 = __shfl_xor_sync(0xFFFFFFFF, bits_302, 4);
        float2 peer_304 = make_float2(0.0f, 0.0f);
        peer_304 = reinterpret_cast<float2*>(&peerbits_303)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_304);
        unsigned long long bits_305 = 0;
        float2 _bit_cast_src_51 = pairs[0];
        bits_305 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_306 = __shfl_xor_sync(0xFFFFFFFF, bits_305, 2);
        float2 peer_307 = make_float2(0.0f, 0.0f);
        peer_307 = reinterpret_cast<float2*>(&peerbits_306)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_307);
        unsigned long long bits_308 = 0;
        float2 _bit_cast_src_52 = pairs[1];
        bits_308 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_309 = __shfl_xor_sync(0xFFFFFFFF, bits_308, 2);
        float2 peer_310 = make_float2(0.0f, 0.0f);
        peer_310 = reinterpret_cast<float2*>(&peerbits_309)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_310);
        unsigned long long bits_311 = 0;
        float2 _bit_cast_src_53 = pairs[2];
        bits_311 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_312 = __shfl_xor_sync(0xFFFFFFFF, bits_311, 2);
        float2 peer_313 = make_float2(0.0f, 0.0f);
        peer_313 = reinterpret_cast<float2*>(&peerbits_312)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_313);
        unsigned long long bits_314 = 0;
        float2 _bit_cast_src_54 = pairs[3];
        bits_314 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_315 = __shfl_xor_sync(0xFFFFFFFF, bits_314, 2);
        float2 peer_316 = make_float2(0.0f, 0.0f);
        peer_316 = reinterpret_cast<float2*>(&peerbits_315)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_316);
        unsigned long long bits_317 = 0;
        float2 _bit_cast_src_55 = pairs[4];
        bits_317 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_55)[0];
        unsigned long long peerbits_318 = __shfl_xor_sync(0xFFFFFFFF, bits_317, 2);
        float2 peer_319 = make_float2(0.0f, 0.0f);
        peer_319 = reinterpret_cast<float2*>(&peerbits_318)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_319);
        unsigned long long bits_320 = 0;
        float2 _bit_cast_src_56 = pairs[5];
        bits_320 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_56)[0];
        unsigned long long peerbits_321 = __shfl_xor_sync(0xFFFFFFFF, bits_320, 2);
        float2 peer_322 = make_float2(0.0f, 0.0f);
        peer_322 = reinterpret_cast<float2*>(&peerbits_321)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_322);
        unsigned long long bits_323 = 0;
        float2 _bit_cast_src_57 = pairs[0];
        bits_323 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_57)[0];
        unsigned long long peerbits_324 = __shfl_xor_sync(0xFFFFFFFF, bits_323, 1);
        float2 peer_325 = make_float2(0.0f, 0.0f);
        peer_325 = reinterpret_cast<float2*>(&peerbits_324)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_325);
        unsigned long long bits_326 = 0;
        float2 _bit_cast_src_58 = pairs[1];
        bits_326 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_58)[0];
        unsigned long long peerbits_327 = __shfl_xor_sync(0xFFFFFFFF, bits_326, 1);
        float2 peer_328 = make_float2(0.0f, 0.0f);
        peer_328 = reinterpret_cast<float2*>(&peerbits_327)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_328);
        unsigned long long bits_329 = 0;
        float2 _bit_cast_src_59 = pairs[2];
        bits_329 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_59)[0];
        unsigned long long peerbits_330 = __shfl_xor_sync(0xFFFFFFFF, bits_329, 1);
        float2 peer_331 = make_float2(0.0f, 0.0f);
        peer_331 = reinterpret_cast<float2*>(&peerbits_330)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_331);
        unsigned long long bits_332 = 0;
        float2 _bit_cast_src_60 = pairs[3];
        bits_332 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_60)[0];
        unsigned long long peerbits_333 = __shfl_xor_sync(0xFFFFFFFF, bits_332, 1);
        float2 peer_334 = make_float2(0.0f, 0.0f);
        peer_334 = reinterpret_cast<float2*>(&peerbits_333)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_334);
        unsigned long long bits_335 = 0;
        float2 _bit_cast_src_61 = pairs[4];
        bits_335 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_61)[0];
        unsigned long long peerbits_336 = __shfl_xor_sync(0xFFFFFFFF, bits_335, 1);
        float2 peer_337 = make_float2(0.0f, 0.0f);
        peer_337 = reinterpret_cast<float2*>(&peerbits_336)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_337);
        unsigned long long bits_338 = 0;
        float2 _bit_cast_src_62 = pairs[5];
        bits_338 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_62)[0];
        unsigned long long peerbits_339 = __shfl_xor_sync(0xFFFFFFFF, bits_338, 1);
        float2 peer_340 = make_float2(0.0f, 0.0f);
        peer_340 = reinterpret_cast<float2*>(&peerbits_339)[0];
        pairs[5] = add_f32x2_noftz(pairs[5], peer_340);
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (lane == 0) {
            uint32_t _cluster_mapa_63;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_63) : "r"(stats_addr + (unsigned int)(warp_0 * 6 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_63), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_64;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_64) : "r"(stats_addr + (unsigned int)((warp_0 * 6 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_64), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_65;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_65) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_65), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_66;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_66) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_66), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_67;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_67) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_67), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_68;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_68) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_68), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_69;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_69) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_69), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_70;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_70) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_70), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_71;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_71) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_71), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_72;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_72) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_72), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_73;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_73) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 5) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_73), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_74;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_74) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 5) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_74), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_75;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_75) : "r"(stats_addr + (unsigned int)(warp_0 * 6 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_75), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_76;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_76) : "r"(stats_addr + (unsigned int)((warp_0 * 6 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_76), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_77;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_77) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_77), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_78;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_78) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_78), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_79;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_79) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_79), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_80;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_80) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_80), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_81;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_81) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_81), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_82;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_82) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_82), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_83;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_83) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_83), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_84;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_84) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_84), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_85;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_85) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 5) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_85), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_86;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_86) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 5) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_86), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_87;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_87) : "r"(stats_addr + (unsigned int)(warp_0 * 6 * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_87), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_88;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_88) : "r"(stats_addr + (unsigned int)((warp_0 * 6 * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_88), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_89;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_89) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 1) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_89), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_90;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_90) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 1) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_90), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_91;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_91) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 2) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_91), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_92;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_92) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 2) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_92), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_93;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_93) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 3) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_93), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_94;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_94) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 3) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_94), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_95;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_95) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 4) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_95), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_96;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_96) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 4) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_96), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_97;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_97) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 5) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_97), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_98;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_98) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 5) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_98), "f"(pairs[5].y) : "memory");
            uint32_t _cluster_mapa_99;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_99) : "r"(stats_addr + (unsigned int)(warp_0 * 6 * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_99), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_100;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_100) : "r"(stats_addr + (unsigned int)((warp_0 * 6 * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_100), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_101;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_101) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 1) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_101), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_102;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_102) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 1) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_102), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_103;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_103) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 2) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_103), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_104;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_104) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 2) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_104), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_105;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_105) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 3) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_105), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_106;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_106) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 3) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_106), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_107;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_107) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 4) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_107), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_108;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_108) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 4) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_108), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_109;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_109) : "r"(stats_addr + (unsigned int)((warp_0 * 6 + 5) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_109), "f"(pairs[5].x) : "memory");
            uint32_t _cluster_mapa_110;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_110) : "r"(stats_addr + (unsigned int)(((warp_0 * 6 + 5) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_110), "f"(pairs[5].y) : "memory");
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
        float _shfl_down_sync_111 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_111;
        float _shfl_down_sync_112 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_112;
        float _shfl_down_sync_113 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_113;
        float _shfl_down_sync_114 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_114;
        float _shfl_down_sync_115 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_115;
        float _shfl_down_sync_116 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_116;
        float logit = 0.0f;
        if (stat_n < 6 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[6];
        float _shfl_sync_117 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_117;
        float _shfl_sync_118 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_118;
        float _shfl_sync_119 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_119;
        float _shfl_sync_120 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_sync_120;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 2) {
            total_sq2 = stats[(stat_w * 6 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 6 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_121 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_121;
        float _shfl_down_sync_122 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_122;
        float _shfl_down_sync_123 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_123;
        float _shfl_down_sync_124 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_124;
        float _shfl_down_sync_125 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_125;
        float _shfl_down_sync_126 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_126;
        logit2 = 0.0f;
        if (stat_n < 2 && stat_w == 0) {
            float sigma2 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_sync_127 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_sync_127;
        float _shfl_sync_128 = __shfl_sync(0xFFFFFFFF, logit2, 8);
        logits[5] = _shfl_sync_128;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028235e+38f;
        max_chunk = fmaxf(max_chunk, logits[0]);
        max_chunk = fmaxf(max_chunk, logits[1]);
        max_chunk = fmaxf(max_chunk, logits[2]);
        float max_new = fmaxf(max_running, max_chunk);
        float correction = approx_exp2((max_running - max_new) * 1.442695f);
        float weights[6];
        float sum_weights = 0.0f;
        float _exp2_129 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_129;
        sum_weights += weights[0];
        float _exp2_130 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_130;
        sum_weights += weights[1];
        float _exp2_131 = approx_exp2((logits[2] - max_new) * 1.442695f);
        weights[2] = _exp2_131;
        sum_weights += weights[2];
        float2 corr = make_float2(correction, correction);
        const int woff_341 = 0;
        int base_342 = group * 1024 + thread * 8;
        float2 a_343[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_343[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_344 = make_float2(acc[2], acc[3]);
        a_343[1] = mul_f32x2_noftz(previous_344, corr);
        float2 previous_345 = make_float2(acc[4], acc[5]);
        a_343[2] = mul_f32x2_noftz(previous_345, corr);
        float2 previous_346 = make_float2(acc[6], acc[7]);
        a_343[3] = mul_f32x2_noftz(previous_346, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_347 = make_float2(fsrc[0], fsrc[1]);
        a_343[0] = fma_f32x2_rn_noftz(weight, v_347, a_343[0]);
        float2 v_348 = make_float2(fsrc[2], fsrc[3]);
        a_343[1] = fma_f32x2_rn_noftz(weight, v_348, a_343[1]);
        float2 v_349 = make_float2(fsrc[4], fsrc[5]);
        a_343[2] = fma_f32x2_rn_noftz(weight, v_349, a_343[2]);
        float2 v_350 = make_float2(fsrc[6], fsrc[7]);
        a_343[3] = fma_f32x2_rn_noftz(weight, v_350, a_343[3]);
        float2 weight_351 = make_float2(weights[1], weights[1]);
        float2 v_352 = make_float2(fsrc[28], fsrc[29]);
        a_343[0] = fma_f32x2_rn_noftz(weight_351, v_352, a_343[0]);
        float2 v_353 = make_float2(fsrc[30], fsrc[31]);
        a_343[1] = fma_f32x2_rn_noftz(weight_351, v_353, a_343[1]);
        float2 v_354 = make_float2(fsrc[32], fsrc[33]);
        a_343[2] = fma_f32x2_rn_noftz(weight_351, v_354, a_343[2]);
        float2 v_355 = make_float2(fsrc[34], fsrc[35]);
        a_343[3] = fma_f32x2_rn_noftz(weight_351, v_355, a_343[3]);
        float2 weight_356 = make_float2(weights[2], weights[2]);
        float2 v_357 = make_float2(fsrc[56], fsrc[57]);
        a_343[0] = fma_f32x2_rn_noftz(weight_356, v_357, a_343[0]);
        float2 v_358 = make_float2(fsrc[58], fsrc[59]);
        a_343[1] = fma_f32x2_rn_noftz(weight_356, v_358, a_343[1]);
        float2 v_359 = make_float2(fsrc[60], fsrc[61]);
        a_343[2] = fma_f32x2_rn_noftz(weight_356, v_359, a_343[2]);
        float2 v_360 = make_float2(fsrc[62], fsrc[63]);
        a_343[3] = fma_f32x2_rn_noftz(weight_356, v_360, a_343[3]);
        acc[0] = a_343[0].x;
        acc[1] = a_343[0].y;
        acc[2] = a_343[1].x;
        acc[3] = a_343[1].y;
        acc[4] = a_343[2].x;
        acc[5] = a_343[2].y;
        acc[6] = a_343[3].x;
        acc[7] = a_343[3].y;
        const int woff_361 = 4;
        int base_362 = (2 + group) * 1024 + thread * 8;
        float2 a_363[4];
        float2 previous_364 = make_float2(acc[8], acc[9]);
        a_363[0] = mul_f32x2_noftz(previous_364, corr);
        float2 previous_365 = make_float2(acc[10], acc[11]);
        a_363[1] = mul_f32x2_noftz(previous_365, corr);
        float2 previous_366 = make_float2(acc[12], acc[13]);
        a_363[2] = mul_f32x2_noftz(previous_366, corr);
        float2 previous_367 = make_float2(acc[14], acc[15]);
        a_363[3] = mul_f32x2_noftz(previous_367, corr);
        float2 weight_368 = make_float2(weights[0], weights[0]);
        float2 v_369 = make_float2(fsrc[8], fsrc[9]);
        a_363[0] = fma_f32x2_rn_noftz(weight_368, v_369, a_363[0]);
        float2 v_370 = make_float2(fsrc[10], fsrc[11]);
        a_363[1] = fma_f32x2_rn_noftz(weight_368, v_370, a_363[1]);
        float2 v_371 = make_float2(fsrc[12], fsrc[13]);
        a_363[2] = fma_f32x2_rn_noftz(weight_368, v_371, a_363[2]);
        float2 v_372 = make_float2(fsrc[14], fsrc[15]);
        a_363[3] = fma_f32x2_rn_noftz(weight_368, v_372, a_363[3]);
        float2 weight_373 = make_float2(weights[1], weights[1]);
        float2 v_374 = make_float2(fsrc[36], fsrc[37]);
        a_363[0] = fma_f32x2_rn_noftz(weight_373, v_374, a_363[0]);
        float2 v_375 = make_float2(fsrc[38], fsrc[39]);
        a_363[1] = fma_f32x2_rn_noftz(weight_373, v_375, a_363[1]);
        float2 v_376 = make_float2(fsrc[40], fsrc[41]);
        a_363[2] = fma_f32x2_rn_noftz(weight_373, v_376, a_363[2]);
        float2 v_377 = make_float2(fsrc[42], fsrc[43]);
        a_363[3] = fma_f32x2_rn_noftz(weight_373, v_377, a_363[3]);
        float2 weight_378 = make_float2(weights[2], weights[2]);
        float2 v_379 = make_float2(fsrc[64], fsrc[65]);
        a_363[0] = fma_f32x2_rn_noftz(weight_378, v_379, a_363[0]);
        float2 v_380 = make_float2(fsrc[66], fsrc[67]);
        a_363[1] = fma_f32x2_rn_noftz(weight_378, v_380, a_363[1]);
        float2 v_381 = make_float2(fsrc[68], fsrc[69]);
        a_363[2] = fma_f32x2_rn_noftz(weight_378, v_381, a_363[2]);
        float2 v_382 = make_float2(fsrc[70], fsrc[71]);
        a_363[3] = fma_f32x2_rn_noftz(weight_378, v_382, a_363[3]);
        acc[8] = a_363[0].x;
        acc[9] = a_363[0].y;
        acc[10] = a_363[1].x;
        acc[11] = a_363[1].y;
        acc[12] = a_363[2].x;
        acc[13] = a_363[2].y;
        acc[14] = a_363[3].x;
        acc[15] = a_363[3].y;
        const int woff_383 = 8;
        int base_384 = (4 + group) * 1024 + thread * 8;
        float2 a_385[4];
        float2 previous_386 = make_float2(acc[16], acc[17]);
        a_385[0] = mul_f32x2_noftz(previous_386, corr);
        float2 previous_387 = make_float2(acc[18], acc[19]);
        a_385[1] = mul_f32x2_noftz(previous_387, corr);
        float2 previous_388 = make_float2(acc[20], acc[21]);
        a_385[2] = mul_f32x2_noftz(previous_388, corr);
        float2 previous_389 = make_float2(acc[22], acc[23]);
        a_385[3] = mul_f32x2_noftz(previous_389, corr);
        float2 weight_390 = make_float2(weights[0], weights[0]);
        float2 v_391 = make_float2(fsrc[16], fsrc[17]);
        a_385[0] = fma_f32x2_rn_noftz(weight_390, v_391, a_385[0]);
        float2 v_392 = make_float2(fsrc[18], fsrc[19]);
        a_385[1] = fma_f32x2_rn_noftz(weight_390, v_392, a_385[1]);
        float2 v_393 = make_float2(fsrc[20], fsrc[21]);
        a_385[2] = fma_f32x2_rn_noftz(weight_390, v_393, a_385[2]);
        float2 v_394 = make_float2(fsrc[22], fsrc[23]);
        a_385[3] = fma_f32x2_rn_noftz(weight_390, v_394, a_385[3]);
        float2 weight_395 = make_float2(weights[1], weights[1]);
        float2 v_396 = make_float2(fsrc[44], fsrc[45]);
        a_385[0] = fma_f32x2_rn_noftz(weight_395, v_396, a_385[0]);
        float2 v_397 = make_float2(fsrc[46], fsrc[47]);
        a_385[1] = fma_f32x2_rn_noftz(weight_395, v_397, a_385[1]);
        float2 v_398 = make_float2(fsrc[48], fsrc[49]);
        a_385[2] = fma_f32x2_rn_noftz(weight_395, v_398, a_385[2]);
        float2 v_399 = make_float2(fsrc[50], fsrc[51]);
        a_385[3] = fma_f32x2_rn_noftz(weight_395, v_399, a_385[3]);
        float2 weight_400 = make_float2(weights[2], weights[2]);
        float2 v_401 = make_float2(fsrc[72], fsrc[73]);
        a_385[0] = fma_f32x2_rn_noftz(weight_400, v_401, a_385[0]);
        float2 v_402 = make_float2(fsrc[74], fsrc[75]);
        a_385[1] = fma_f32x2_rn_noftz(weight_400, v_402, a_385[1]);
        float2 v_403 = make_float2(fsrc[76], fsrc[77]);
        a_385[2] = fma_f32x2_rn_noftz(weight_400, v_403, a_385[2]);
        float2 v_404 = make_float2(fsrc[78], fsrc[79]);
        a_385[3] = fma_f32x2_rn_noftz(weight_400, v_404, a_385[3]);
        acc[16] = a_385[0].x;
        acc[17] = a_385[0].y;
        acc[18] = a_385[1].x;
        acc[19] = a_385[1].y;
        acc[20] = a_385[2].x;
        acc[21] = a_385[2].y;
        acc[22] = a_385[3].x;
        acc[23] = a_385[3].y;
        const int woff_405 = 12;
        int base_406 = 6144 + group * 512 + thread * 4;
        float2 a_407[4];
        float2 previous_408 = make_float2(acc[24], acc[25]);
        a_407[0] = mul_f32x2_noftz(previous_408, corr);
        float2 previous_409 = make_float2(acc[26], acc[27]);
        a_407[1] = mul_f32x2_noftz(previous_409, corr);
        float2 weight_410 = make_float2(weights[0], weights[0]);
        float2 v_411 = make_float2(fsrc[24], fsrc[25]);
        a_407[0] = fma_f32x2_rn_noftz(weight_410, v_411, a_407[0]);
        float2 v_412 = make_float2(fsrc[26], fsrc[27]);
        a_407[1] = fma_f32x2_rn_noftz(weight_410, v_412, a_407[1]);
        float2 weight_413 = make_float2(weights[1], weights[1]);
        float2 v_414 = make_float2(fsrc[52], fsrc[53]);
        a_407[0] = fma_f32x2_rn_noftz(weight_413, v_414, a_407[0]);
        float2 v_415 = make_float2(fsrc[54], fsrc[55]);
        a_407[1] = fma_f32x2_rn_noftz(weight_413, v_415, a_407[1]);
        float2 weight_416 = make_float2(weights[2], weights[2]);
        float2 v_417 = make_float2(fsrc[80], fsrc[81]);
        a_407[0] = fma_f32x2_rn_noftz(weight_416, v_417, a_407[0]);
        float2 v_418 = make_float2(fsrc[82], fsrc[83]);
        a_407[1] = fma_f32x2_rn_noftz(weight_416, v_418, a_407[1]);
        acc[24] = a_407[0].x;
        acc[25] = a_407[0].y;
        acc[26] = a_407[1].x;
        acc[27] = a_407[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_419 = -3.4028235e+38f;
        max_chunk_419 = fmaxf(max_chunk_419, logits[3]);
        max_chunk_419 = fmaxf(max_chunk_419, logits[4]);
        max_chunk_419 = fmaxf(max_chunk_419, logits[5]);
        float max_new_420 = fmaxf(max_running, max_chunk_419);
        float correction_421 = approx_exp2((max_running - max_new_420) * 1.442695f);
        float weights_422[6];
        float sum_weights_423 = 0.0f;
        float _exp2_132 = approx_exp2((logits[3] - max_new_420) * 1.442695f);
        weights_422[3] = _exp2_132;
        sum_weights_423 += weights_422[3];
        float _exp2_133 = approx_exp2((logits[4] - max_new_420) * 1.442695f);
        weights_422[4] = _exp2_133;
        sum_weights_423 += weights_422[4];
        float _exp2_134 = approx_exp2((logits[5] - max_new_420) * 1.442695f);
        weights_422[5] = _exp2_134;
        sum_weights_423 += weights_422[5];
        float2 corr_424 = make_float2(correction_421, correction_421);
        const int woff_425 = 0;
        int base_426 = group * 1024 + thread * 8;
        float2 a_427[4];
        float2 previous_428 = make_float2(acc[0], acc[1]);
        a_427[0] = mul_f32x2_noftz(previous_428, corr_424);
        float2 previous_429 = make_float2(acc[2], acc[3]);
        a_427[1] = mul_f32x2_noftz(previous_429, corr_424);
        float2 previous_430 = make_float2(acc[4], acc[5]);
        a_427[2] = mul_f32x2_noftz(previous_430, corr_424);
        float2 previous_431 = make_float2(acc[6], acc[7]);
        a_427[3] = mul_f32x2_noftz(previous_431, corr_424);
        float2 weight_432 = make_float2(weights_422[3], weights_422[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_425];
        sw2[1] = words[42 + woff_425 + 1];
        sw2[2] = words[42 + woff_425 + 2];
        sw2[3] = words[42 + woff_425 + 3];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw2[_pair]));
        }
        float2 v_433 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        a_427[0] = fma_f32x2_rn_noftz(weight_432, v_433, a_427[0]);
        float2 v_434 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        a_427[1] = fma_f32x2_rn_noftz(weight_432, v_434, a_427[1]);
        float2 v_435 = make_float2(_unpack_bf16_24[4], _unpack_bf16_24[5]);
        a_427[2] = fma_f32x2_rn_noftz(weight_432, v_435, a_427[2]);
        float2 v_436 = make_float2(_unpack_bf16_24[6], _unpack_bf16_24[7]);
        a_427[3] = fma_f32x2_rn_noftz(weight_432, v_436, a_427[3]);
        float2 weight_437 = make_float2(weights_422[4], weights_422[4]);
        unsigned int sw2_438[4];
        sw2_438[0] = words[56 + woff_425];
        sw2_438[1] = words[56 + woff_425 + 1];
        sw2_438[2] = words[56 + woff_425 + 2];
        sw2_438[3] = words[56 + woff_425 + 3];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw2_438[_pair]));
        }
        float2 v_439 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        a_427[0] = fma_f32x2_rn_noftz(weight_437, v_439, a_427[0]);
        float2 v_440 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        a_427[1] = fma_f32x2_rn_noftz(weight_437, v_440, a_427[1]);
        float2 v_441 = make_float2(_unpack_bf16_25[4], _unpack_bf16_25[5]);
        a_427[2] = fma_f32x2_rn_noftz(weight_437, v_441, a_427[2]);
        float2 v_442 = make_float2(_unpack_bf16_25[6], _unpack_bf16_25[7]);
        a_427[3] = fma_f32x2_rn_noftz(weight_437, v_442, a_427[3]);
        float2 weight_443 = make_float2(weights_422[5], weights_422[5]);
        unsigned int sw2_444[4];
        sw2_444[0] = words[70 + woff_425];
        sw2_444[1] = words[70 + woff_425 + 1];
        sw2_444[2] = words[70 + woff_425 + 2];
        sw2_444[3] = words[70 + woff_425 + 3];
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw2_444[_pair]));
        }
        float2 v_445 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        a_427[0] = fma_f32x2_rn_noftz(weight_443, v_445, a_427[0]);
        float2 v_446 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        a_427[1] = fma_f32x2_rn_noftz(weight_443, v_446, a_427[1]);
        float2 v_447 = make_float2(_unpack_bf16_26[4], _unpack_bf16_26[5]);
        a_427[2] = fma_f32x2_rn_noftz(weight_443, v_447, a_427[2]);
        float2 v_448 = make_float2(_unpack_bf16_26[6], _unpack_bf16_26[7]);
        a_427[3] = fma_f32x2_rn_noftz(weight_443, v_448, a_427[3]);
        acc[0] = a_427[0].x;
        acc[1] = a_427[0].y;
        acc[2] = a_427[1].x;
        acc[3] = a_427[1].y;
        acc[4] = a_427[2].x;
        acc[5] = a_427[2].y;
        acc[6] = a_427[3].x;
        acc[7] = a_427[3].y;
        const int woff_449 = 4;
        int base_450 = (2 + group) * 1024 + thread * 8;
        float2 a_451[4];
        float2 previous_452 = make_float2(acc[8], acc[9]);
        a_451[0] = mul_f32x2_noftz(previous_452, corr_424);
        float2 previous_453 = make_float2(acc[10], acc[11]);
        a_451[1] = mul_f32x2_noftz(previous_453, corr_424);
        float2 previous_454 = make_float2(acc[12], acc[13]);
        a_451[2] = mul_f32x2_noftz(previous_454, corr_424);
        float2 previous_455 = make_float2(acc[14], acc[15]);
        a_451[3] = mul_f32x2_noftz(previous_455, corr_424);
        float2 weight_456 = make_float2(weights_422[3], weights_422[3]);
        unsigned int sw2_457[4];
        sw2_457[0] = words[42 + woff_449];
        sw2_457[1] = words[42 + woff_449 + 1];
        sw2_457[2] = words[42 + woff_449 + 2];
        sw2_457[3] = words[42 + woff_449 + 3];
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw2_457[_pair]));
        }
        float2 v_458 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        a_451[0] = fma_f32x2_rn_noftz(weight_456, v_458, a_451[0]);
        float2 v_459 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        a_451[1] = fma_f32x2_rn_noftz(weight_456, v_459, a_451[1]);
        float2 v_460 = make_float2(_unpack_bf16_27[4], _unpack_bf16_27[5]);
        a_451[2] = fma_f32x2_rn_noftz(weight_456, v_460, a_451[2]);
        float2 v_461 = make_float2(_unpack_bf16_27[6], _unpack_bf16_27[7]);
        a_451[3] = fma_f32x2_rn_noftz(weight_456, v_461, a_451[3]);
        float2 weight_462 = make_float2(weights_422[4], weights_422[4]);
        unsigned int sw2_463[4];
        sw2_463[0] = words[56 + woff_449];
        sw2_463[1] = words[56 + woff_449 + 1];
        sw2_463[2] = words[56 + woff_449 + 2];
        sw2_463[3] = words[56 + woff_449 + 3];
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_28[_pair * 2])[0]), "=f"((&_unpack_bf16_28[_pair * 2])[1])
                : "r"(sw2_463[_pair]));
        }
        float2 v_464 = make_float2(_unpack_bf16_28[0], _unpack_bf16_28[1]);
        a_451[0] = fma_f32x2_rn_noftz(weight_462, v_464, a_451[0]);
        float2 v_465 = make_float2(_unpack_bf16_28[2], _unpack_bf16_28[3]);
        a_451[1] = fma_f32x2_rn_noftz(weight_462, v_465, a_451[1]);
        float2 v_466 = make_float2(_unpack_bf16_28[4], _unpack_bf16_28[5]);
        a_451[2] = fma_f32x2_rn_noftz(weight_462, v_466, a_451[2]);
        float2 v_467 = make_float2(_unpack_bf16_28[6], _unpack_bf16_28[7]);
        a_451[3] = fma_f32x2_rn_noftz(weight_462, v_467, a_451[3]);
        float2 weight_468 = make_float2(weights_422[5], weights_422[5]);
        unsigned int sw2_469[4];
        sw2_469[0] = words[70 + woff_449];
        sw2_469[1] = words[70 + woff_449 + 1];
        sw2_469[2] = words[70 + woff_449 + 2];
        sw2_469[3] = words[70 + woff_449 + 3];
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_29[_pair * 2])[0]), "=f"((&_unpack_bf16_29[_pair * 2])[1])
                : "r"(sw2_469[_pair]));
        }
        float2 v_470 = make_float2(_unpack_bf16_29[0], _unpack_bf16_29[1]);
        a_451[0] = fma_f32x2_rn_noftz(weight_468, v_470, a_451[0]);
        float2 v_471 = make_float2(_unpack_bf16_29[2], _unpack_bf16_29[3]);
        a_451[1] = fma_f32x2_rn_noftz(weight_468, v_471, a_451[1]);
        float2 v_472 = make_float2(_unpack_bf16_29[4], _unpack_bf16_29[5]);
        a_451[2] = fma_f32x2_rn_noftz(weight_468, v_472, a_451[2]);
        float2 v_473 = make_float2(_unpack_bf16_29[6], _unpack_bf16_29[7]);
        a_451[3] = fma_f32x2_rn_noftz(weight_468, v_473, a_451[3]);
        acc[8] = a_451[0].x;
        acc[9] = a_451[0].y;
        acc[10] = a_451[1].x;
        acc[11] = a_451[1].y;
        acc[12] = a_451[2].x;
        acc[13] = a_451[2].y;
        acc[14] = a_451[3].x;
        acc[15] = a_451[3].y;
        const int woff_474 = 8;
        int base_475 = (4 + group) * 1024 + thread * 8;
        float2 a_476[4];
        float2 previous_477 = make_float2(acc[16], acc[17]);
        a_476[0] = mul_f32x2_noftz(previous_477, corr_424);
        float2 previous_478 = make_float2(acc[18], acc[19]);
        a_476[1] = mul_f32x2_noftz(previous_478, corr_424);
        float2 previous_479 = make_float2(acc[20], acc[21]);
        a_476[2] = mul_f32x2_noftz(previous_479, corr_424);
        float2 previous_480 = make_float2(acc[22], acc[23]);
        a_476[3] = mul_f32x2_noftz(previous_480, corr_424);
        float2 weight_481 = make_float2(weights_422[3], weights_422[3]);
        unsigned int sw2_482[4];
        sw2_482[0] = words[42 + woff_474];
        sw2_482[1] = words[42 + woff_474 + 1];
        sw2_482[2] = words[42 + woff_474 + 2];
        sw2_482[3] = words[42 + woff_474 + 3];
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_30[_pair * 2])[0]), "=f"((&_unpack_bf16_30[_pair * 2])[1])
                : "r"(sw2_482[_pair]));
        }
        float2 v_483 = make_float2(_unpack_bf16_30[0], _unpack_bf16_30[1]);
        a_476[0] = fma_f32x2_rn_noftz(weight_481, v_483, a_476[0]);
        float2 v_484 = make_float2(_unpack_bf16_30[2], _unpack_bf16_30[3]);
        a_476[1] = fma_f32x2_rn_noftz(weight_481, v_484, a_476[1]);
        float2 v_485 = make_float2(_unpack_bf16_30[4], _unpack_bf16_30[5]);
        a_476[2] = fma_f32x2_rn_noftz(weight_481, v_485, a_476[2]);
        float2 v_486 = make_float2(_unpack_bf16_30[6], _unpack_bf16_30[7]);
        a_476[3] = fma_f32x2_rn_noftz(weight_481, v_486, a_476[3]);
        float2 weight_487 = make_float2(weights_422[4], weights_422[4]);
        unsigned int sw2_488[4];
        sw2_488[0] = words[56 + woff_474];
        sw2_488[1] = words[56 + woff_474 + 1];
        sw2_488[2] = words[56 + woff_474 + 2];
        sw2_488[3] = words[56 + woff_474 + 3];
        float _unpack_bf16_31[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_31[_pair * 2])[0]), "=f"((&_unpack_bf16_31[_pair * 2])[1])
                : "r"(sw2_488[_pair]));
        }
        float2 v_489 = make_float2(_unpack_bf16_31[0], _unpack_bf16_31[1]);
        a_476[0] = fma_f32x2_rn_noftz(weight_487, v_489, a_476[0]);
        float2 v_490 = make_float2(_unpack_bf16_31[2], _unpack_bf16_31[3]);
        a_476[1] = fma_f32x2_rn_noftz(weight_487, v_490, a_476[1]);
        float2 v_491 = make_float2(_unpack_bf16_31[4], _unpack_bf16_31[5]);
        a_476[2] = fma_f32x2_rn_noftz(weight_487, v_491, a_476[2]);
        float2 v_492 = make_float2(_unpack_bf16_31[6], _unpack_bf16_31[7]);
        a_476[3] = fma_f32x2_rn_noftz(weight_487, v_492, a_476[3]);
        float2 weight_493 = make_float2(weights_422[5], weights_422[5]);
        unsigned int sw2_494[4];
        sw2_494[0] = words[70 + woff_474];
        sw2_494[1] = words[70 + woff_474 + 1];
        sw2_494[2] = words[70 + woff_474 + 2];
        sw2_494[3] = words[70 + woff_474 + 3];
        float _unpack_bf16_32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_32[_pair * 2])[0]), "=f"((&_unpack_bf16_32[_pair * 2])[1])
                : "r"(sw2_494[_pair]));
        }
        float2 v_495 = make_float2(_unpack_bf16_32[0], _unpack_bf16_32[1]);
        a_476[0] = fma_f32x2_rn_noftz(weight_493, v_495, a_476[0]);
        float2 v_496 = make_float2(_unpack_bf16_32[2], _unpack_bf16_32[3]);
        a_476[1] = fma_f32x2_rn_noftz(weight_493, v_496, a_476[1]);
        float2 v_497 = make_float2(_unpack_bf16_32[4], _unpack_bf16_32[5]);
        a_476[2] = fma_f32x2_rn_noftz(weight_493, v_497, a_476[2]);
        float2 v_498 = make_float2(_unpack_bf16_32[6], _unpack_bf16_32[7]);
        a_476[3] = fma_f32x2_rn_noftz(weight_493, v_498, a_476[3]);
        acc[16] = a_476[0].x;
        acc[17] = a_476[0].y;
        acc[18] = a_476[1].x;
        acc[19] = a_476[1].y;
        acc[20] = a_476[2].x;
        acc[21] = a_476[2].y;
        acc[22] = a_476[3].x;
        acc[23] = a_476[3].y;
        const int woff_499 = 12;
        int base_500 = 6144 + group * 512 + thread * 4;
        float2 a_501[4];
        float2 previous_502 = make_float2(acc[24], acc[25]);
        a_501[0] = mul_f32x2_noftz(previous_502, corr_424);
        float2 previous_503 = make_float2(acc[26], acc[27]);
        a_501[1] = mul_f32x2_noftz(previous_503, corr_424);
        float2 weight_504 = make_float2(weights_422[3], weights_422[3]);
        unsigned int sw2_505[4];
        sw2_505[0] = words[42 + woff_499];
        sw2_505[1] = words[42 + woff_499 + 1];
        float _unpack_bf16_33[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_33[_pair * 2])[0]), "=f"((&_unpack_bf16_33[_pair * 2])[1])
                : "r"(sw2_505[_pair]));
        }
        float2 v_506 = make_float2(_unpack_bf16_33[0], _unpack_bf16_33[1]);
        a_501[0] = fma_f32x2_rn_noftz(weight_504, v_506, a_501[0]);
        float2 v_507 = make_float2(_unpack_bf16_33[2], _unpack_bf16_33[3]);
        a_501[1] = fma_f32x2_rn_noftz(weight_504, v_507, a_501[1]);
        float2 weight_508 = make_float2(weights_422[4], weights_422[4]);
        unsigned int sw2_509[4];
        sw2_509[0] = words[56 + woff_499];
        sw2_509[1] = words[56 + woff_499 + 1];
        float _unpack_bf16_34[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_34[_pair * 2])[0]), "=f"((&_unpack_bf16_34[_pair * 2])[1])
                : "r"(sw2_509[_pair]));
        }
        float2 v_510 = make_float2(_unpack_bf16_34[0], _unpack_bf16_34[1]);
        a_501[0] = fma_f32x2_rn_noftz(weight_508, v_510, a_501[0]);
        float2 v_511 = make_float2(_unpack_bf16_34[2], _unpack_bf16_34[3]);
        a_501[1] = fma_f32x2_rn_noftz(weight_508, v_511, a_501[1]);
        float2 weight_512 = make_float2(weights_422[5], weights_422[5]);
        unsigned int sw2_513[4];
        sw2_513[0] = words[70 + woff_499];
        sw2_513[1] = words[70 + woff_499 + 1];
        float _unpack_bf16_35[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_35[_pair * 2])[0]), "=f"((&_unpack_bf16_35[_pair * 2])[1])
                : "r"(sw2_513[_pair]));
        }
        float2 v_514 = make_float2(_unpack_bf16_35[0], _unpack_bf16_35[1]);
        a_501[0] = fma_f32x2_rn_noftz(weight_512, v_514, a_501[0]);
        float2 v_515 = make_float2(_unpack_bf16_35[2], _unpack_bf16_35[3]);
        a_501[1] = fma_f32x2_rn_noftz(weight_512, v_515, a_501[1]);
        acc[24] = a_501[0].x;
        acc[25] = a_501[0].y;
        acc[26] = a_501[1].x;
        acc[27] = a_501[1].y;
        sum_running = __fmaf_rn(sum_running, correction_421, sum_weights_423);
        max_running = max_new_420;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_516 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_516, v_516, output_sq_pair);
        float2 v_517 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_517, v_517, output_sq_pair);
        float2 v_518 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_518, v_518, output_sq_pair);
        float2 v_519 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_519, v_519, output_sq_pair);
        float2 v_520 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_520, v_520, output_sq_pair);
        float2 v_521 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_521, v_521, output_sq_pair);
        float2 v_522 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_522, v_522, output_sq_pair);
        float2 v_523 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_523, v_523, output_sq_pair);
        float2 v_524 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_524, v_524, output_sq_pair);
        float2 v_525 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_525, v_525, output_sq_pair);
        float2 v_526 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_526, v_526, output_sq_pair);
        float2 v_527 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_527, v_527, output_sq_pair);
        float2 v_528 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_528, v_528, output_sq_pair);
        float2 v_529 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_529, v_529, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_135 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_135;
        float _shfl_xor_sync_136 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_136;
        float _shfl_xor_sync_137 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_137;
        float _shfl_xor_sync_138 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_138;
        float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_139;
        if (lane == 0) {
            uint32_t _cluster_mapa_140;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_140) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_140), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_141;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_141) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_141), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_142;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_142) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_142), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_143;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_143) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_143), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_144 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_144;
        float _shfl_down_sync_145 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_145;
        float _shfl_down_sync_146 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_146;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_147 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_147;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_530 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_148 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_149 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_531 = 2;
        const int acc_idx_532 = value_idx_531;
        float2 _f32x2_mul_a_150 = make_float2(acc[acc_idx_532], acc[acc_idx_532 + 1]);
        float2 scaled_pair_533;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_533) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_151 = make_float2(wout[acc_idx_532], wout[acc_idx_532 + 1]);
        float2 normalized_pair_534;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_534) : "l"(*(const unsigned long long*)&scaled_pair_533), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
        output_values[value_idx_531] = normalized_pair_534.x;
        output_values[value_idx_531 + 1] = normalized_pair_534.y;
        const int value_idx_535 = 4;
        const int acc_idx_536 = value_idx_535;
        float2 _f32x2_mul_a_152 = make_float2(acc[acc_idx_536], acc[acc_idx_536 + 1]);
        float2 scaled_pair_537;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_537) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_153 = make_float2(wout[acc_idx_536], wout[acc_idx_536 + 1]);
        float2 normalized_pair_538;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_538) : "l"(*(const unsigned long long*)&scaled_pair_537), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
        output_values[value_idx_535] = normalized_pair_538.x;
        output_values[value_idx_535 + 1] = normalized_pair_538.y;
        const int value_idx_539 = 6;
        const int acc_idx_540 = value_idx_539;
        float2 _f32x2_mul_a_154 = make_float2(acc[acc_idx_540], acc[acc_idx_540 + 1]);
        float2 scaled_pair_541;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_541) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_155 = make_float2(wout[acc_idx_540], wout[acc_idx_540 + 1]);
        float2 normalized_pair_542;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_542) : "l"(*(const unsigned long long*)&scaled_pair_541), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
        output_values[value_idx_539] = normalized_pair_542.x;
        output_values[value_idx_539 + 1] = normalized_pair_542.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_530))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_543 = (2 + group) * 1024 + thread * 8;
        float output_values_544[8];
        const int value_idx_545 = 0;
        const int acc_idx_546 = 8 + value_idx_545;
        float2 _f32x2_mul_a_156 = make_float2(acc[acc_idx_546], acc[acc_idx_546 + 1]);
        float2 scaled_pair_547;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_547) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_157 = make_float2(wout[acc_idx_546], wout[acc_idx_546 + 1]);
        float2 normalized_pair_548;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_548) : "l"(*(const unsigned long long*)&scaled_pair_547), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
        output_values_544[value_idx_545] = normalized_pair_548.x;
        output_values_544[value_idx_545 + 1] = normalized_pair_548.y;
        const int value_idx_549 = 2;
        const int acc_idx_550 = 8 + value_idx_549;
        float2 _f32x2_mul_a_158 = make_float2(acc[acc_idx_550], acc[acc_idx_550 + 1]);
        float2 scaled_pair_551;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_551) : "l"(*(const unsigned long long*)&_f32x2_mul_a_158), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_159 = make_float2(wout[acc_idx_550], wout[acc_idx_550 + 1]);
        float2 normalized_pair_552;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_552) : "l"(*(const unsigned long long*)&scaled_pair_551), "l"(*(const unsigned long long*)&_f32x2_mul_b_159));
        output_values_544[value_idx_549] = normalized_pair_552.x;
        output_values_544[value_idx_549 + 1] = normalized_pair_552.y;
        const int value_idx_553 = 4;
        const int acc_idx_554 = 8 + value_idx_553;
        float2 _f32x2_mul_a_160 = make_float2(acc[acc_idx_554], acc[acc_idx_554 + 1]);
        float2 scaled_pair_555;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_555) : "l"(*(const unsigned long long*)&_f32x2_mul_a_160), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_161 = make_float2(wout[acc_idx_554], wout[acc_idx_554 + 1]);
        float2 normalized_pair_556;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_556) : "l"(*(const unsigned long long*)&scaled_pair_555), "l"(*(const unsigned long long*)&_f32x2_mul_b_161));
        output_values_544[value_idx_553] = normalized_pair_556.x;
        output_values_544[value_idx_553 + 1] = normalized_pair_556.y;
        const int value_idx_557 = 6;
        const int acc_idx_558 = 8 + value_idx_557;
        float2 _f32x2_mul_a_162 = make_float2(acc[acc_idx_558], acc[acc_idx_558 + 1]);
        float2 scaled_pair_559;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_559) : "l"(*(const unsigned long long*)&_f32x2_mul_a_162), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_163 = make_float2(wout[acc_idx_558], wout[acc_idx_558 + 1]);
        float2 normalized_pair_560;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_560) : "l"(*(const unsigned long long*)&scaled_pair_559), "l"(*(const unsigned long long*)&_f32x2_mul_b_163));
        output_values_544[value_idx_557] = normalized_pair_560.x;
        output_values_544[value_idx_557 + 1] = normalized_pair_560.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_544[_lp*2 + 0], output_values_544[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_544[_lp*2 + 2], output_values_544[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_544[_lp*2 + 4], output_values_544[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_544[_lp*2 + 6], output_values_544[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_543))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_561 = (4 + group) * 1024 + thread * 8;
        float output_values_562[8];
        const int value_idx_563 = 0;
        const int acc_idx_564 = 16 + value_idx_563;
        float2 _f32x2_mul_a_164 = make_float2(acc[acc_idx_564], acc[acc_idx_564 + 1]);
        float2 scaled_pair_565;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_565) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_165 = make_float2(wout[acc_idx_564], wout[acc_idx_564 + 1]);
        float2 normalized_pair_566;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_566) : "l"(*(const unsigned long long*)&scaled_pair_565), "l"(*(const unsigned long long*)&_f32x2_mul_b_165));
        output_values_562[value_idx_563] = normalized_pair_566.x;
        output_values_562[value_idx_563 + 1] = normalized_pair_566.y;
        const int value_idx_567 = 2;
        const int acc_idx_568 = 16 + value_idx_567;
        float2 _f32x2_mul_a_166 = make_float2(acc[acc_idx_568], acc[acc_idx_568 + 1]);
        float2 scaled_pair_569;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_569) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_167 = make_float2(wout[acc_idx_568], wout[acc_idx_568 + 1]);
        float2 normalized_pair_570;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_570) : "l"(*(const unsigned long long*)&scaled_pair_569), "l"(*(const unsigned long long*)&_f32x2_mul_b_167));
        output_values_562[value_idx_567] = normalized_pair_570.x;
        output_values_562[value_idx_567 + 1] = normalized_pair_570.y;
        const int value_idx_571 = 4;
        const int acc_idx_572 = 16 + value_idx_571;
        float2 _f32x2_mul_a_168 = make_float2(acc[acc_idx_572], acc[acc_idx_572 + 1]);
        float2 scaled_pair_573;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_573) : "l"(*(const unsigned long long*)&_f32x2_mul_a_168), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_169 = make_float2(wout[acc_idx_572], wout[acc_idx_572 + 1]);
        float2 normalized_pair_574;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_574) : "l"(*(const unsigned long long*)&scaled_pair_573), "l"(*(const unsigned long long*)&_f32x2_mul_b_169));
        output_values_562[value_idx_571] = normalized_pair_574.x;
        output_values_562[value_idx_571 + 1] = normalized_pair_574.y;
        const int value_idx_575 = 6;
        const int acc_idx_576 = 16 + value_idx_575;
        float2 _f32x2_mul_a_170 = make_float2(acc[acc_idx_576], acc[acc_idx_576 + 1]);
        float2 scaled_pair_577;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_577) : "l"(*(const unsigned long long*)&_f32x2_mul_a_170), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_171 = make_float2(wout[acc_idx_576], wout[acc_idx_576 + 1]);
        float2 normalized_pair_578;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_578) : "l"(*(const unsigned long long*)&scaled_pair_577), "l"(*(const unsigned long long*)&_f32x2_mul_b_171));
        output_values_562[value_idx_575] = normalized_pair_578.x;
        output_values_562[value_idx_575 + 1] = normalized_pair_578.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_562[_lp*2 + 0], output_values_562[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_562[_lp*2 + 2], output_values_562[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_562[_lp*2 + 4], output_values_562[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_562[_lp*2 + 6], output_values_562[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_561))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_579 = 6144 + group * 512 + thread * 4;
        float output_values_580[8];
        const int value_idx_581 = 0;
        const int acc_idx_582 = 24 + value_idx_581;
        float2 _f32x2_mul_a_172 = make_float2(acc[acc_idx_582], acc[acc_idx_582 + 1]);
        float2 scaled_pair_583;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_583) : "l"(*(const unsigned long long*)&_f32x2_mul_a_172), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_173 = make_float2(wout[acc_idx_582], wout[acc_idx_582 + 1]);
        float2 normalized_pair_584;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_584) : "l"(*(const unsigned long long*)&scaled_pair_583), "l"(*(const unsigned long long*)&_f32x2_mul_b_173));
        output_values_580[value_idx_581] = normalized_pair_584.x;
        output_values_580[value_idx_581 + 1] = normalized_pair_584.y;
        const int value_idx_585 = 2;
        const int acc_idx_586 = 24 + value_idx_585;
        float2 _f32x2_mul_a_174 = make_float2(acc[acc_idx_586], acc[acc_idx_586 + 1]);
        float2 scaled_pair_587;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_587) : "l"(*(const unsigned long long*)&_f32x2_mul_a_174), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_175 = make_float2(wout[acc_idx_586], wout[acc_idx_586 + 1]);
        float2 normalized_pair_588;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_588) : "l"(*(const unsigned long long*)&scaled_pair_587), "l"(*(const unsigned long long*)&_f32x2_mul_b_175));
        output_values_580[value_idx_585] = normalized_pair_588.x;
        output_values_580[value_idx_585 + 1] = normalized_pair_588.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_580[_lp*2 + 0], output_values_580[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_580[_lp*2 + 2], output_values_580[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_579))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
