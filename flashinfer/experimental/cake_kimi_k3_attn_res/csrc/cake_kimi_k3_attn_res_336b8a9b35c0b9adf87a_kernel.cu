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
#define SMEM_STATS_STAGE_BYTES 320
#define SMEM_STATS_STRIDE 320
#define SMEM_OUT_STATS_OFF 320
#define SMEM_OUT_STATS_STAGE_BYTES 32
#define SMEM_OUT_STATS_STRIDE 32
#define SMEM_TOTAL 384
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
kernel_cake_kimi_k3_attn_res_336b8a9b35c0b9adf87a(__nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ delta, __nv_bfloat16* __restrict__ blocks, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ qk_weight, __nv_bfloat16* __restrict__ output_norm_weight, unsigned long long blocks_m_stride, unsigned long long blocks_k_stride, float eps, float output_norm_eps, int M)
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
    float* out_stats = reinterpret_cast<float*>(smem_raw + 320);
    const int out_stats_addr = smem + 320;

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
        unsigned int words[70];
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
            uint4 _uv4_4 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base)) + 0);
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
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base)) + 0);
            _vec_load_5[0 + 0] = _uv4_5.x;
            _vec_load_5[0 + 1] = _uv4_5.y;
            _vec_load_5[0 + 2] = _uv4_5.z;
            _vec_load_5[0 + 3] = _uv4_5.w;
        }
        dwords[woff] = _vec_load_5[0];
        dwords[woff + 1] = _vec_load_5[1];
        dwords[woff + 2] = _vec_load_5[2];
        dwords[woff + 3] = _vec_load_5[3];
        float _vec_load_6[8];
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
                        : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        float _vec_load_7[8];
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
                        : "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_7[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_7[_pair]));
                }
            }
        }
        q[0] = _vec_load_6[0] * _vec_load_7[0];
        q[1] = _vec_load_6[1] * _vec_load_7[1];
        q[2] = _vec_load_6[2] * _vec_load_7[2];
        q[3] = _vec_load_6[3] * _vec_load_7[3];
        q[4] = _vec_load_6[4] * _vec_load_7[4];
        q[5] = _vec_load_6[5] * _vec_load_7[5];
        q[6] = _vec_load_6[6] * _vec_load_7[6];
        q[7] = _vec_load_6[7] * _vec_load_7[7];
        float _vec_load_8[8];
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
                        : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_8[_pair]));
                }
            }
        }
        wout[0] = _vec_load_8[0];
        wout[1] = _vec_load_8[1];
        wout[2] = _vec_load_8[2];
        wout[3] = _vec_load_8[3];
        wout[4] = _vec_load_8[4];
        wout[5] = _vec_load_8[5];
        wout[6] = _vec_load_8[6];
        wout[7] = _vec_load_8[7];
        int base_0 = (2 + group) * 1024 + thread * 8;
        const int woff_1 = 4;
        unsigned int _vec_load_9[4];
        {
            uint4 _uv4_9 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_0)) + 0);
            _vec_load_9[0 + 0] = _uv4_9.x;
            _vec_load_9[0 + 1] = _uv4_9.y;
            _vec_load_9[0 + 2] = _uv4_9.z;
            _vec_load_9[0 + 3] = _uv4_9.w;
        }
        words[woff_1] = _vec_load_9[0];
        words[woff_1 + 1] = _vec_load_9[1];
        words[woff_1 + 2] = _vec_load_9[2];
        words[woff_1 + 3] = _vec_load_9[3];
        unsigned int _vec_load_10[4];
        {
            uint4 _uv4_10 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_10[0 + 0] = _uv4_10.x;
            _vec_load_10[0 + 1] = _uv4_10.y;
            _vec_load_10[0 + 2] = _uv4_10.z;
            _vec_load_10[0 + 3] = _uv4_10.w;
        }
        words[14 + woff_1] = _vec_load_10[0];
        words[14 + woff_1 + 1] = _vec_load_10[1];
        words[14 + woff_1 + 2] = _vec_load_10[2];
        words[14 + woff_1 + 3] = _vec_load_10[3];
        unsigned int _vec_load_11[4];
        {
            uint4 _uv4_11 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_11[0 + 0] = _uv4_11.x;
            _vec_load_11[0 + 1] = _uv4_11.y;
            _vec_load_11[0 + 2] = _uv4_11.z;
            _vec_load_11[0 + 3] = _uv4_11.w;
        }
        words[28 + woff_1] = _vec_load_11[0];
        words[28 + woff_1 + 1] = _vec_load_11[1];
        words[28 + woff_1 + 2] = _vec_load_11[2];
        words[28 + woff_1 + 3] = _vec_load_11[3];
        unsigned int _vec_load_12[4];
        {
            uint4 _uv4_12 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_0)) + 0);
            _vec_load_12[0 + 0] = _uv4_12.x;
            _vec_load_12[0 + 1] = _uv4_12.y;
            _vec_load_12[0 + 2] = _uv4_12.z;
            _vec_load_12[0 + 3] = _uv4_12.w;
        }
        words[42 + woff_1] = _vec_load_12[0];
        words[42 + woff_1 + 1] = _vec_load_12[1];
        words[42 + woff_1 + 2] = _vec_load_12[2];
        words[42 + woff_1 + 3] = _vec_load_12[3];
        unsigned int _vec_load_13[4];
        {
            uint4 _uv4_13 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_13[0 + 0] = _uv4_13.x;
            _vec_load_13[0 + 1] = _uv4_13.y;
            _vec_load_13[0 + 2] = _uv4_13.z;
            _vec_load_13[0 + 3] = _uv4_13.w;
        }
        words[56 + woff_1] = _vec_load_13[0];
        words[56 + woff_1 + 1] = _vec_load_13[1];
        words[56 + woff_1 + 2] = _vec_load_13[2];
        words[56 + woff_1 + 3] = _vec_load_13[3];
        unsigned int _vec_load_14[4];
        {
            uint4 _uv4_14 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_0)) + 0);
            _vec_load_14[0 + 0] = _uv4_14.x;
            _vec_load_14[0 + 1] = _uv4_14.y;
            _vec_load_14[0 + 2] = _uv4_14.z;
            _vec_load_14[0 + 3] = _uv4_14.w;
        }
        dwords[woff_1] = _vec_load_14[0];
        dwords[woff_1 + 1] = _vec_load_14[1];
        dwords[woff_1 + 2] = _vec_load_14[2];
        dwords[woff_1 + 3] = _vec_load_14[3];
        float _vec_load_15[8];
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
                        : "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_15[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_15[_pair]));
                }
            }
        }
        float _vec_load_16[8];
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
                        : "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_16[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_16[_pair]));
                }
            }
        }
        q[8] = _vec_load_15[0] * _vec_load_16[0];
        q[9] = _vec_load_15[1] * _vec_load_16[1];
        q[10] = _vec_load_15[2] * _vec_load_16[2];
        q[11] = _vec_load_15[3] * _vec_load_16[3];
        q[12] = _vec_load_15[4] * _vec_load_16[4];
        q[13] = _vec_load_15[5] * _vec_load_16[5];
        q[14] = _vec_load_15[6] * _vec_load_16[6];
        q[15] = _vec_load_15[7] * _vec_load_16[7];
        float _vec_load_17[8];
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
                        : "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_17[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_17[_pair]));
                }
            }
        }
        wout[8] = _vec_load_17[0];
        wout[9] = _vec_load_17[1];
        wout[10] = _vec_load_17[2];
        wout[11] = _vec_load_17[3];
        wout[12] = _vec_load_17[4];
        wout[13] = _vec_load_17[5];
        wout[14] = _vec_load_17[6];
        wout[15] = _vec_load_17[7];
        int base_2 = (4 + group) * 1024 + thread * 8;
        const int woff_3 = 8;
        unsigned int _vec_load_18[4];
        {
            uint4 _uv4_18 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_2)) + 0);
            _vec_load_18[0 + 0] = _uv4_18.x;
            _vec_load_18[0 + 1] = _uv4_18.y;
            _vec_load_18[0 + 2] = _uv4_18.z;
            _vec_load_18[0 + 3] = _uv4_18.w;
        }
        words[woff_3] = _vec_load_18[0];
        words[woff_3 + 1] = _vec_load_18[1];
        words[woff_3 + 2] = _vec_load_18[2];
        words[woff_3 + 3] = _vec_load_18[3];
        unsigned int _vec_load_19[4];
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_19[0 + 0] = _uv4_19.x;
            _vec_load_19[0 + 1] = _uv4_19.y;
            _vec_load_19[0 + 2] = _uv4_19.z;
            _vec_load_19[0 + 3] = _uv4_19.w;
        }
        words[14 + woff_3] = _vec_load_19[0];
        words[14 + woff_3 + 1] = _vec_load_19[1];
        words[14 + woff_3 + 2] = _vec_load_19[2];
        words[14 + woff_3 + 3] = _vec_load_19[3];
        unsigned int _vec_load_20[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_20[0 + 0] = _uv4_20.x;
            _vec_load_20[0 + 1] = _uv4_20.y;
            _vec_load_20[0 + 2] = _uv4_20.z;
            _vec_load_20[0 + 3] = _uv4_20.w;
        }
        words[28 + woff_3] = _vec_load_20[0];
        words[28 + woff_3 + 1] = _vec_load_20[1];
        words[28 + woff_3 + 2] = _vec_load_20[2];
        words[28 + woff_3 + 3] = _vec_load_20[3];
        unsigned int _vec_load_21[4];
        {
            uint4 _uv4_21 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_2)) + 0);
            _vec_load_21[0 + 0] = _uv4_21.x;
            _vec_load_21[0 + 1] = _uv4_21.y;
            _vec_load_21[0 + 2] = _uv4_21.z;
            _vec_load_21[0 + 3] = _uv4_21.w;
        }
        words[42 + woff_3] = _vec_load_21[0];
        words[42 + woff_3 + 1] = _vec_load_21[1];
        words[42 + woff_3 + 2] = _vec_load_21[2];
        words[42 + woff_3 + 3] = _vec_load_21[3];
        unsigned int _vec_load_22[4];
        {
            uint4 _uv4_22 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_22[0 + 0] = _uv4_22.x;
            _vec_load_22[0 + 1] = _uv4_22.y;
            _vec_load_22[0 + 2] = _uv4_22.z;
            _vec_load_22[0 + 3] = _uv4_22.w;
        }
        words[56 + woff_3] = _vec_load_22[0];
        words[56 + woff_3 + 1] = _vec_load_22[1];
        words[56 + woff_3 + 2] = _vec_load_22[2];
        words[56 + woff_3 + 3] = _vec_load_22[3];
        unsigned int _vec_load_23[4];
        {
            uint4 _uv4_23 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_2)) + 0);
            _vec_load_23[0 + 0] = _uv4_23.x;
            _vec_load_23[0 + 1] = _uv4_23.y;
            _vec_load_23[0 + 2] = _uv4_23.z;
            _vec_load_23[0 + 3] = _uv4_23.w;
        }
        dwords[woff_3] = _vec_load_23[0];
        dwords[woff_3 + 1] = _vec_load_23[1];
        dwords[woff_3 + 2] = _vec_load_23[2];
        dwords[woff_3 + 3] = _vec_load_23[3];
        float _vec_load_24[8];
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
                        : "=f"((&_vec_load_24[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_24[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_24[_pair]));
                }
            }
        }
        float _vec_load_25[8];
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
                        : "=f"((&_vec_load_25[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_25[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_25[_pair]));
                }
            }
        }
        q[16] = _vec_load_24[0] * _vec_load_25[0];
        q[17] = _vec_load_24[1] * _vec_load_25[1];
        q[18] = _vec_load_24[2] * _vec_load_25[2];
        q[19] = _vec_load_24[3] * _vec_load_25[3];
        q[20] = _vec_load_24[4] * _vec_load_25[4];
        q[21] = _vec_load_24[5] * _vec_load_25[5];
        q[22] = _vec_load_24[6] * _vec_load_25[6];
        q[23] = _vec_load_24[7] * _vec_load_25[7];
        float _vec_load_26[8];
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
                        : "=f"((&_vec_load_26[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_26[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_26[_pair]));
                }
            }
        }
        wout[16] = _vec_load_26[0];
        wout[17] = _vec_load_26[1];
        wout[18] = _vec_load_26[2];
        wout[19] = _vec_load_26[3];
        wout[20] = _vec_load_26[4];
        wout[21] = _vec_load_26[5];
        wout[22] = _vec_load_26[6];
        wout[23] = _vec_load_26[7];
        int base_4 = 6144 + group * 512 + thread * 4;
        const int woff_5 = 12;
        unsigned int _vec_load_27[1];
        {
            _vec_load_27[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 0);
        }
        words[woff_5] = _vec_load_27[0];
        unsigned int _vec_load_28[1];
        {
            _vec_load_28[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + (unsigned long long)base_4)) + 1);
        }
        words[woff_5 + 1] = _vec_load_28[0];
        unsigned int _vec_load_29[1];
        {
            _vec_load_29[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[14 + woff_5] = _vec_load_29[0];
        unsigned int _vec_load_30[1];
        {
            _vec_load_30[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[14 + woff_5 + 1] = _vec_load_30[0];
        unsigned int _vec_load_31[1];
        {
            _vec_load_31[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[28 + woff_5] = _vec_load_31[0];
        unsigned int _vec_load_32[1];
        {
            _vec_load_32[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 2 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[28 + woff_5 + 1] = _vec_load_32[0];
        unsigned int _vec_load_33[1];
        {
            _vec_load_33[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 0);
        }
        words[42 + woff_5] = _vec_load_33[0];
        unsigned int _vec_load_34[1];
        {
            _vec_load_34[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(blocks + (block_base + 3 * blocks_k_stride + (unsigned long long)base_4)) + 1);
        }
        words[42 + woff_5 + 1] = _vec_load_34[0];
        unsigned int _vec_load_35[1];
        {
            _vec_load_35[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 0);
        }
        words[56 + woff_5] = _vec_load_35[0];
        unsigned int _vec_load_36[1];
        {
            _vec_load_36[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(prefix + (row_base + (unsigned long long)base_4)) + 1);
        }
        words[56 + woff_5 + 1] = _vec_load_36[0];
        unsigned int _vec_load_37[1];
        {
            _vec_load_37[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 0);
        }
        dwords[woff_5] = _vec_load_37[0];
        unsigned int _vec_load_38[1];
        {
            _vec_load_38[0] = *reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(delta + (row_base + (unsigned long long)base_4)) + 1);
        }
        dwords[woff_5 + 1] = _vec_load_38[0];
        float _vec_load_39[4];
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
                    : "=f"((&_vec_load_39[0 + _pair * 2])[0]), "=f"((&_vec_load_39[0 + _pair * 2])[1])
                    : "r"(_vpairs_27[_pair]));
            }
        }
        float _vec_load_40[4];
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
                    : "=f"((&_vec_load_40[0 + _pair * 2])[0]), "=f"((&_vec_load_40[0 + _pair * 2])[1])
                    : "r"(_vpairs_28[_pair]));
            }
        }
        q[24] = _vec_load_39[0] * _vec_load_40[0];
        q[25] = _vec_load_39[1] * _vec_load_40[1];
        q[26] = _vec_load_39[2] * _vec_load_40[2];
        q[27] = _vec_load_39[3] * _vec_load_40[3];
        float _vec_load_41[4];
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
                    : "=f"((&_vec_load_41[0 + _pair * 2])[0]), "=f"((&_vec_load_41[0 + _pair * 2])[1])
                    : "r"(_vpairs_29[_pair]));
            }
        }
        wout[24] = _vec_load_41[0];
        wout[25] = _vec_load_41[1];
        wout[26] = _vec_load_41[2];
        wout[27] = _vec_load_41[3];
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
        __nv_bfloat162 a = __as_bf16x2(sw_41[0]);
        __nv_bfloat162 d = __as_bf16x2(dwords[woff_7]);
        __nv_bfloat162 mixed = a + d;
        sw_41[0] = __as_u32(mixed);
        words[56 + woff_7] = sw_41[0];
        __nv_bfloat162 a_42 = __as_bf16x2(sw_41[1]);
        __nv_bfloat162 d_43 = __as_bf16x2(dwords[woff_7 + 1]);
        __nv_bfloat162 mixed_44 = a_42 + d_43;
        sw_41[1] = __as_u32(mixed_44);
        words[56 + woff_7 + 1] = sw_41[1];
        __nv_bfloat162 a_45 = __as_bf16x2(sw_41[2]);
        __nv_bfloat162 d_46 = __as_bf16x2(dwords[woff_7 + 2]);
        __nv_bfloat162 mixed_47 = a_45 + d_46;
        sw_41[2] = __as_u32(mixed_47);
        words[56 + woff_7 + 2] = sw_41[2];
        __nv_bfloat162 a_48 = __as_bf16x2(sw_41[3]);
        __nv_bfloat162 d_49 = __as_bf16x2(dwords[woff_7 + 3]);
        __nv_bfloat162 mixed_50 = a_48 + d_49;
        sw_41[3] = __as_u32(mixed_50);
        words[56 + woff_7 + 3] = sw_41[3];
        {
            int4 _iv4 = make_int4(sw_41[0 + 0], sw_41[0 + 1], sw_41[0 + 2], sw_41[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_6)) + 0) = _iv4;
        }
        unsigned long long write_base = block_base + 4 * blocks_k_stride + (unsigned long long)base_6;
        {
            int4 _iv4 = make_int4(sw_41[0 + 0], sw_41[0 + 1], sw_41[0 + 2], sw_41[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base) + 0) = _iv4;
        }
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
        float2 v_51 = make_float2(_unpack_bf16_4[0], _unpack_bf16_4[1]);
        float2 qp_52 = make_float2(q[0], q[1]);
        sq[4] = fma_f32x2_rn_noftz(v_51, v_51, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_51, qp_52, dot[4]);
        float2 v_53 = make_float2(_unpack_bf16_4[2], _unpack_bf16_4[3]);
        float2 qp_54 = make_float2(q[2], q[3]);
        sq[4] = fma_f32x2_rn_noftz(v_53, v_53, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_53, qp_54, dot[4]);
        float2 v_55 = make_float2(_unpack_bf16_4[4], _unpack_bf16_4[5]);
        float2 qp_56 = make_float2(q[4], q[5]);
        sq[4] = fma_f32x2_rn_noftz(v_55, v_55, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_55, qp_56, dot[4]);
        float2 v_57 = make_float2(_unpack_bf16_4[6], _unpack_bf16_4[7]);
        float2 qp_58 = make_float2(q[6], q[7]);
        sq[4] = fma_f32x2_rn_noftz(v_57, v_57, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_57, qp_58, dot[4]);
        int base_59 = (2 + group) * 1024 + thread * 8;
        const int woff_60 = 4;
        unsigned int sw_61[4];
        sw_61[0] = words[woff_60];
        sw_61[1] = words[woff_60 + 1];
        sw_61[2] = words[woff_60 + 2];
        sw_61[3] = words[woff_60 + 3];
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                : "r"(sw_61[_pair]));
        }
        fsrc[8] = _unpack_bf16_5[0];
        fsrc[9] = _unpack_bf16_5[1];
        fsrc[10] = _unpack_bf16_5[2];
        fsrc[11] = _unpack_bf16_5[3];
        fsrc[12] = _unpack_bf16_5[4];
        fsrc[13] = _unpack_bf16_5[5];
        fsrc[14] = _unpack_bf16_5[6];
        fsrc[15] = _unpack_bf16_5[7];
        float2 v_62 = make_float2(_unpack_bf16_5[0], _unpack_bf16_5[1]);
        float2 qp_63 = make_float2(q[8], q[9]);
        sq[0] = fma_f32x2_rn_noftz(v_62, v_62, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_62, qp_63, dot[0]);
        float2 v_64 = make_float2(_unpack_bf16_5[2], _unpack_bf16_5[3]);
        float2 qp_65 = make_float2(q[10], q[11]);
        sq[0] = fma_f32x2_rn_noftz(v_64, v_64, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_64, qp_65, dot[0]);
        float2 v_66 = make_float2(_unpack_bf16_5[4], _unpack_bf16_5[5]);
        float2 qp_67 = make_float2(q[12], q[13]);
        sq[0] = fma_f32x2_rn_noftz(v_66, v_66, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_66, qp_67, dot[0]);
        float2 v_68 = make_float2(_unpack_bf16_5[6], _unpack_bf16_5[7]);
        float2 qp_69 = make_float2(q[14], q[15]);
        sq[0] = fma_f32x2_rn_noftz(v_68, v_68, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_68, qp_69, dot[0]);
        unsigned int sw_70[4];
        sw_70[0] = words[14 + woff_60];
        sw_70[1] = words[14 + woff_60 + 1];
        sw_70[2] = words[14 + woff_60 + 2];
        sw_70[3] = words[14 + woff_60 + 3];
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
        fsrc[36] = _unpack_bf16_6[0];
        fsrc[37] = _unpack_bf16_6[1];
        fsrc[38] = _unpack_bf16_6[2];
        fsrc[39] = _unpack_bf16_6[3];
        fsrc[40] = _unpack_bf16_6[4];
        fsrc[41] = _unpack_bf16_6[5];
        fsrc[42] = _unpack_bf16_6[6];
        fsrc[43] = _unpack_bf16_6[7];
        float2 v_71 = make_float2(_unpack_bf16_6[0], _unpack_bf16_6[1]);
        float2 qp_72 = make_float2(q[8], q[9]);
        sq[1] = fma_f32x2_rn_noftz(v_71, v_71, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_71, qp_72, dot[1]);
        float2 v_73 = make_float2(_unpack_bf16_6[2], _unpack_bf16_6[3]);
        float2 qp_74 = make_float2(q[10], q[11]);
        sq[1] = fma_f32x2_rn_noftz(v_73, v_73, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_73, qp_74, dot[1]);
        float2 v_75 = make_float2(_unpack_bf16_6[4], _unpack_bf16_6[5]);
        float2 qp_76 = make_float2(q[12], q[13]);
        sq[1] = fma_f32x2_rn_noftz(v_75, v_75, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_75, qp_76, dot[1]);
        float2 v_77 = make_float2(_unpack_bf16_6[6], _unpack_bf16_6[7]);
        float2 qp_78 = make_float2(q[14], q[15]);
        sq[1] = fma_f32x2_rn_noftz(v_77, v_77, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_77, qp_78, dot[1]);
        unsigned int sw_79[4];
        sw_79[0] = words[28 + woff_60];
        sw_79[1] = words[28 + woff_60 + 1];
        sw_79[2] = words[28 + woff_60 + 2];
        sw_79[3] = words[28 + woff_60 + 3];
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
        fsrc[64] = _unpack_bf16_7[0];
        fsrc[65] = _unpack_bf16_7[1];
        fsrc[66] = _unpack_bf16_7[2];
        fsrc[67] = _unpack_bf16_7[3];
        fsrc[68] = _unpack_bf16_7[4];
        fsrc[69] = _unpack_bf16_7[5];
        fsrc[70] = _unpack_bf16_7[6];
        fsrc[71] = _unpack_bf16_7[7];
        float2 v_80 = make_float2(_unpack_bf16_7[0], _unpack_bf16_7[1]);
        float2 qp_81 = make_float2(q[8], q[9]);
        sq[2] = fma_f32x2_rn_noftz(v_80, v_80, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_80, qp_81, dot[2]);
        float2 v_82 = make_float2(_unpack_bf16_7[2], _unpack_bf16_7[3]);
        float2 qp_83 = make_float2(q[10], q[11]);
        sq[2] = fma_f32x2_rn_noftz(v_82, v_82, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_82, qp_83, dot[2]);
        float2 v_84 = make_float2(_unpack_bf16_7[4], _unpack_bf16_7[5]);
        float2 qp_85 = make_float2(q[12], q[13]);
        sq[2] = fma_f32x2_rn_noftz(v_84, v_84, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_84, qp_85, dot[2]);
        float2 v_86 = make_float2(_unpack_bf16_7[6], _unpack_bf16_7[7]);
        float2 qp_87 = make_float2(q[14], q[15]);
        sq[2] = fma_f32x2_rn_noftz(v_86, v_86, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_86, qp_87, dot[2]);
        unsigned int sw_88[4];
        sw_88[0] = words[42 + woff_60];
        sw_88[1] = words[42 + woff_60 + 1];
        sw_88[2] = words[42 + woff_60 + 2];
        sw_88[3] = words[42 + woff_60 + 3];
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
        float2 v_89 = make_float2(_unpack_bf16_8[0], _unpack_bf16_8[1]);
        float2 qp_90 = make_float2(q[8], q[9]);
        sq[3] = fma_f32x2_rn_noftz(v_89, v_89, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_89, qp_90, dot[3]);
        float2 v_91 = make_float2(_unpack_bf16_8[2], _unpack_bf16_8[3]);
        float2 qp_92 = make_float2(q[10], q[11]);
        sq[3] = fma_f32x2_rn_noftz(v_91, v_91, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_91, qp_92, dot[3]);
        float2 v_93 = make_float2(_unpack_bf16_8[4], _unpack_bf16_8[5]);
        float2 qp_94 = make_float2(q[12], q[13]);
        sq[3] = fma_f32x2_rn_noftz(v_93, v_93, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_93, qp_94, dot[3]);
        float2 v_95 = make_float2(_unpack_bf16_8[6], _unpack_bf16_8[7]);
        float2 qp_96 = make_float2(q[14], q[15]);
        sq[3] = fma_f32x2_rn_noftz(v_95, v_95, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_95, qp_96, dot[3]);
        unsigned int sw_97[4];
        sw_97[0] = words[56 + woff_60];
        sw_97[1] = words[56 + woff_60 + 1];
        sw_97[2] = words[56 + woff_60 + 2];
        sw_97[3] = words[56 + woff_60 + 3];
        __nv_bfloat162 a_98 = __as_bf16x2(sw_97[0]);
        __nv_bfloat162 d_99 = __as_bf16x2(dwords[woff_60]);
        __nv_bfloat162 mixed_100 = a_98 + d_99;
        sw_97[0] = __as_u32(mixed_100);
        words[56 + woff_60] = sw_97[0];
        __nv_bfloat162 a_101 = __as_bf16x2(sw_97[1]);
        __nv_bfloat162 d_102 = __as_bf16x2(dwords[woff_60 + 1]);
        __nv_bfloat162 mixed_103 = a_101 + d_102;
        sw_97[1] = __as_u32(mixed_103);
        words[56 + woff_60 + 1] = sw_97[1];
        __nv_bfloat162 a_104 = __as_bf16x2(sw_97[2]);
        __nv_bfloat162 d_105 = __as_bf16x2(dwords[woff_60 + 2]);
        __nv_bfloat162 mixed_106 = a_104 + d_105;
        sw_97[2] = __as_u32(mixed_106);
        words[56 + woff_60 + 2] = sw_97[2];
        __nv_bfloat162 a_107 = __as_bf16x2(sw_97[3]);
        __nv_bfloat162 d_108 = __as_bf16x2(dwords[woff_60 + 3]);
        __nv_bfloat162 mixed_109 = a_107 + d_108;
        sw_97[3] = __as_u32(mixed_109);
        words[56 + woff_60 + 3] = sw_97[3];
        {
            int4 _iv4 = make_int4(sw_97[0 + 0], sw_97[0 + 1], sw_97[0 + 2], sw_97[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_59)) + 0) = _iv4;
        }
        unsigned long long write_base_110 = block_base + 4 * blocks_k_stride + (unsigned long long)base_59;
        {
            int4 _iv4 = make_int4(sw_97[0 + 0], sw_97[0 + 1], sw_97[0 + 2], sw_97[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_110) + 0) = _iv4;
        }
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
        float2 v_111 = make_float2(_unpack_bf16_9[0], _unpack_bf16_9[1]);
        float2 qp_112 = make_float2(q[8], q[9]);
        sq[4] = fma_f32x2_rn_noftz(v_111, v_111, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_111, qp_112, dot[4]);
        float2 v_113 = make_float2(_unpack_bf16_9[2], _unpack_bf16_9[3]);
        float2 qp_114 = make_float2(q[10], q[11]);
        sq[4] = fma_f32x2_rn_noftz(v_113, v_113, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_113, qp_114, dot[4]);
        float2 v_115 = make_float2(_unpack_bf16_9[4], _unpack_bf16_9[5]);
        float2 qp_116 = make_float2(q[12], q[13]);
        sq[4] = fma_f32x2_rn_noftz(v_115, v_115, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_115, qp_116, dot[4]);
        float2 v_117 = make_float2(_unpack_bf16_9[6], _unpack_bf16_9[7]);
        float2 qp_118 = make_float2(q[14], q[15]);
        sq[4] = fma_f32x2_rn_noftz(v_117, v_117, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_117, qp_118, dot[4]);
        int base_119 = (4 + group) * 1024 + thread * 8;
        const int woff_120 = 8;
        unsigned int sw_121[4];
        sw_121[0] = words[woff_120];
        sw_121[1] = words[woff_120 + 1];
        sw_121[2] = words[woff_120 + 2];
        sw_121[3] = words[woff_120 + 3];
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_10[_pair * 2])[0]), "=f"((&_unpack_bf16_10[_pair * 2])[1])
                : "r"(sw_121[_pair]));
        }
        fsrc[16] = _unpack_bf16_10[0];
        fsrc[17] = _unpack_bf16_10[1];
        fsrc[18] = _unpack_bf16_10[2];
        fsrc[19] = _unpack_bf16_10[3];
        fsrc[20] = _unpack_bf16_10[4];
        fsrc[21] = _unpack_bf16_10[5];
        fsrc[22] = _unpack_bf16_10[6];
        fsrc[23] = _unpack_bf16_10[7];
        float2 v_122 = make_float2(_unpack_bf16_10[0], _unpack_bf16_10[1]);
        float2 qp_123 = make_float2(q[16], q[17]);
        sq[0] = fma_f32x2_rn_noftz(v_122, v_122, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_122, qp_123, dot[0]);
        float2 v_124 = make_float2(_unpack_bf16_10[2], _unpack_bf16_10[3]);
        float2 qp_125 = make_float2(q[18], q[19]);
        sq[0] = fma_f32x2_rn_noftz(v_124, v_124, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_124, qp_125, dot[0]);
        float2 v_126 = make_float2(_unpack_bf16_10[4], _unpack_bf16_10[5]);
        float2 qp_127 = make_float2(q[20], q[21]);
        sq[0] = fma_f32x2_rn_noftz(v_126, v_126, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_126, qp_127, dot[0]);
        float2 v_128 = make_float2(_unpack_bf16_10[6], _unpack_bf16_10[7]);
        float2 qp_129 = make_float2(q[22], q[23]);
        sq[0] = fma_f32x2_rn_noftz(v_128, v_128, sq[0]);
        dot[0] = fma_f32x2_rn_noftz(v_128, qp_129, dot[0]);
        unsigned int sw_130[4];
        sw_130[0] = words[14 + woff_120];
        sw_130[1] = words[14 + woff_120 + 1];
        sw_130[2] = words[14 + woff_120 + 2];
        sw_130[3] = words[14 + woff_120 + 3];
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_11[_pair * 2])[0]), "=f"((&_unpack_bf16_11[_pair * 2])[1])
                : "r"(sw_130[_pair]));
        }
        fsrc[44] = _unpack_bf16_11[0];
        fsrc[45] = _unpack_bf16_11[1];
        fsrc[46] = _unpack_bf16_11[2];
        fsrc[47] = _unpack_bf16_11[3];
        fsrc[48] = _unpack_bf16_11[4];
        fsrc[49] = _unpack_bf16_11[5];
        fsrc[50] = _unpack_bf16_11[6];
        fsrc[51] = _unpack_bf16_11[7];
        float2 v_131 = make_float2(_unpack_bf16_11[0], _unpack_bf16_11[1]);
        float2 qp_132 = make_float2(q[16], q[17]);
        sq[1] = fma_f32x2_rn_noftz(v_131, v_131, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_131, qp_132, dot[1]);
        float2 v_133 = make_float2(_unpack_bf16_11[2], _unpack_bf16_11[3]);
        float2 qp_134 = make_float2(q[18], q[19]);
        sq[1] = fma_f32x2_rn_noftz(v_133, v_133, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_133, qp_134, dot[1]);
        float2 v_135 = make_float2(_unpack_bf16_11[4], _unpack_bf16_11[5]);
        float2 qp_136 = make_float2(q[20], q[21]);
        sq[1] = fma_f32x2_rn_noftz(v_135, v_135, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_135, qp_136, dot[1]);
        float2 v_137 = make_float2(_unpack_bf16_11[6], _unpack_bf16_11[7]);
        float2 qp_138 = make_float2(q[22], q[23]);
        sq[1] = fma_f32x2_rn_noftz(v_137, v_137, sq[1]);
        dot[1] = fma_f32x2_rn_noftz(v_137, qp_138, dot[1]);
        unsigned int sw_139[4];
        sw_139[0] = words[28 + woff_120];
        sw_139[1] = words[28 + woff_120 + 1];
        sw_139[2] = words[28 + woff_120 + 2];
        sw_139[3] = words[28 + woff_120 + 3];
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_12[_pair * 2])[0]), "=f"((&_unpack_bf16_12[_pair * 2])[1])
                : "r"(sw_139[_pair]));
        }
        fsrc[72] = _unpack_bf16_12[0];
        fsrc[73] = _unpack_bf16_12[1];
        fsrc[74] = _unpack_bf16_12[2];
        fsrc[75] = _unpack_bf16_12[3];
        fsrc[76] = _unpack_bf16_12[4];
        fsrc[77] = _unpack_bf16_12[5];
        fsrc[78] = _unpack_bf16_12[6];
        fsrc[79] = _unpack_bf16_12[7];
        float2 v_140 = make_float2(_unpack_bf16_12[0], _unpack_bf16_12[1]);
        float2 qp_141 = make_float2(q[16], q[17]);
        sq[2] = fma_f32x2_rn_noftz(v_140, v_140, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_140, qp_141, dot[2]);
        float2 v_142 = make_float2(_unpack_bf16_12[2], _unpack_bf16_12[3]);
        float2 qp_143 = make_float2(q[18], q[19]);
        sq[2] = fma_f32x2_rn_noftz(v_142, v_142, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_142, qp_143, dot[2]);
        float2 v_144 = make_float2(_unpack_bf16_12[4], _unpack_bf16_12[5]);
        float2 qp_145 = make_float2(q[20], q[21]);
        sq[2] = fma_f32x2_rn_noftz(v_144, v_144, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_144, qp_145, dot[2]);
        float2 v_146 = make_float2(_unpack_bf16_12[6], _unpack_bf16_12[7]);
        float2 qp_147 = make_float2(q[22], q[23]);
        sq[2] = fma_f32x2_rn_noftz(v_146, v_146, sq[2]);
        dot[2] = fma_f32x2_rn_noftz(v_146, qp_147, dot[2]);
        unsigned int sw_148[4];
        sw_148[0] = words[42 + woff_120];
        sw_148[1] = words[42 + woff_120 + 1];
        sw_148[2] = words[42 + woff_120 + 2];
        sw_148[3] = words[42 + woff_120 + 3];
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_13[_pair * 2])[0]), "=f"((&_unpack_bf16_13[_pair * 2])[1])
                : "r"(sw_148[_pair]));
        }
        float2 v_149 = make_float2(_unpack_bf16_13[0], _unpack_bf16_13[1]);
        float2 qp_150 = make_float2(q[16], q[17]);
        sq[3] = fma_f32x2_rn_noftz(v_149, v_149, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_149, qp_150, dot[3]);
        float2 v_151 = make_float2(_unpack_bf16_13[2], _unpack_bf16_13[3]);
        float2 qp_152 = make_float2(q[18], q[19]);
        sq[3] = fma_f32x2_rn_noftz(v_151, v_151, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_151, qp_152, dot[3]);
        float2 v_153 = make_float2(_unpack_bf16_13[4], _unpack_bf16_13[5]);
        float2 qp_154 = make_float2(q[20], q[21]);
        sq[3] = fma_f32x2_rn_noftz(v_153, v_153, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_153, qp_154, dot[3]);
        float2 v_155 = make_float2(_unpack_bf16_13[6], _unpack_bf16_13[7]);
        float2 qp_156 = make_float2(q[22], q[23]);
        sq[3] = fma_f32x2_rn_noftz(v_155, v_155, sq[3]);
        dot[3] = fma_f32x2_rn_noftz(v_155, qp_156, dot[3]);
        unsigned int sw_157[4];
        sw_157[0] = words[56 + woff_120];
        sw_157[1] = words[56 + woff_120 + 1];
        sw_157[2] = words[56 + woff_120 + 2];
        sw_157[3] = words[56 + woff_120 + 3];
        __nv_bfloat162 a_158 = __as_bf16x2(sw_157[0]);
        __nv_bfloat162 d_159 = __as_bf16x2(dwords[woff_120]);
        __nv_bfloat162 mixed_160 = a_158 + d_159;
        sw_157[0] = __as_u32(mixed_160);
        words[56 + woff_120] = sw_157[0];
        __nv_bfloat162 a_161 = __as_bf16x2(sw_157[1]);
        __nv_bfloat162 d_162 = __as_bf16x2(dwords[woff_120 + 1]);
        __nv_bfloat162 mixed_163 = a_161 + d_162;
        sw_157[1] = __as_u32(mixed_163);
        words[56 + woff_120 + 1] = sw_157[1];
        __nv_bfloat162 a_164 = __as_bf16x2(sw_157[2]);
        __nv_bfloat162 d_165 = __as_bf16x2(dwords[woff_120 + 2]);
        __nv_bfloat162 mixed_166 = a_164 + d_165;
        sw_157[2] = __as_u32(mixed_166);
        words[56 + woff_120 + 2] = sw_157[2];
        __nv_bfloat162 a_167 = __as_bf16x2(sw_157[3]);
        __nv_bfloat162 d_168 = __as_bf16x2(dwords[woff_120 + 3]);
        __nv_bfloat162 mixed_169 = a_167 + d_168;
        sw_157[3] = __as_u32(mixed_169);
        words[56 + woff_120 + 3] = sw_157[3];
        {
            int4 _iv4 = make_int4(sw_157[0 + 0], sw_157[0 + 1], sw_157[0 + 2], sw_157[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_119)) + 0) = _iv4;
        }
        unsigned long long write_base_170 = block_base + 4 * blocks_k_stride + (unsigned long long)base_119;
        {
            int4 _iv4 = make_int4(sw_157[0 + 0], sw_157[0 + 1], sw_157[0 + 2], sw_157[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(blocks + write_base_170) + 0) = _iv4;
        }
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_14[_pair * 2])[0]), "=f"((&_unpack_bf16_14[_pair * 2])[1])
                : "r"(sw_157[_pair]));
        }
        float2 v_171 = make_float2(_unpack_bf16_14[0], _unpack_bf16_14[1]);
        float2 qp_172 = make_float2(q[16], q[17]);
        sq[4] = fma_f32x2_rn_noftz(v_171, v_171, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_171, qp_172, dot[4]);
        float2 v_173 = make_float2(_unpack_bf16_14[2], _unpack_bf16_14[3]);
        float2 qp_174 = make_float2(q[18], q[19]);
        sq[4] = fma_f32x2_rn_noftz(v_173, v_173, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_173, qp_174, dot[4]);
        float2 v_175 = make_float2(_unpack_bf16_14[4], _unpack_bf16_14[5]);
        float2 qp_176 = make_float2(q[20], q[21]);
        sq[4] = fma_f32x2_rn_noftz(v_175, v_175, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_175, qp_176, dot[4]);
        float2 v_177 = make_float2(_unpack_bf16_14[6], _unpack_bf16_14[7]);
        float2 qp_178 = make_float2(q[22], q[23]);
        sq[4] = fma_f32x2_rn_noftz(v_177, v_177, sq[4]);
        dot[4] = fma_f32x2_rn_noftz(v_177, qp_178, dot[4]);
        int base_179 = 6144 + group * 512 + thread * 4;
        const int woff_180 = 12;
        unsigned int sw_181[4];
        sw_181[0] = words[woff_180];
        sw_181[1] = words[woff_180 + 1];
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_15[_pair * 2])[0]), "=f"((&_unpack_bf16_15[_pair * 2])[1])
                : "r"(sw_181[_pair]));
        }
        fsrc[24] = _unpack_bf16_15[0];
        fsrc[25] = _unpack_bf16_15[1];
        fsrc[26] = _unpack_bf16_15[2];
        fsrc[27] = _unpack_bf16_15[3];
        float2 v_182 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        sq[0] = fma_f32x2_rn_noftz(v_182, v_182, sq[0]);
        float2 v_183 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        sq[0] = fma_f32x2_rn_noftz(v_183, v_183, sq[0]);
        float2 v_184 = make_float2(_unpack_bf16_15[0], _unpack_bf16_15[1]);
        float2 qp_185 = make_float2(q[24], q[25]);
        dot[0] = fma_f32x2_rn_noftz(v_184, qp_185, dot[0]);
        float2 v_186 = make_float2(_unpack_bf16_15[2], _unpack_bf16_15[3]);
        float2 qp_187 = make_float2(q[26], q[27]);
        dot[0] = fma_f32x2_rn_noftz(v_186, qp_187, dot[0]);
        unsigned int sw_188[4];
        sw_188[0] = words[14 + woff_180];
        sw_188[1] = words[14 + woff_180 + 1];
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_16[_pair * 2])[0]), "=f"((&_unpack_bf16_16[_pair * 2])[1])
                : "r"(sw_188[_pair]));
        }
        fsrc[52] = _unpack_bf16_16[0];
        fsrc[53] = _unpack_bf16_16[1];
        fsrc[54] = _unpack_bf16_16[2];
        fsrc[55] = _unpack_bf16_16[3];
        float2 v_189 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        sq[1] = fma_f32x2_rn_noftz(v_189, v_189, sq[1]);
        float2 v_190 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        sq[1] = fma_f32x2_rn_noftz(v_190, v_190, sq[1]);
        float2 v_191 = make_float2(_unpack_bf16_16[0], _unpack_bf16_16[1]);
        float2 qp_192 = make_float2(q[24], q[25]);
        dot[1] = fma_f32x2_rn_noftz(v_191, qp_192, dot[1]);
        float2 v_193 = make_float2(_unpack_bf16_16[2], _unpack_bf16_16[3]);
        float2 qp_194 = make_float2(q[26], q[27]);
        dot[1] = fma_f32x2_rn_noftz(v_193, qp_194, dot[1]);
        unsigned int sw_195[4];
        sw_195[0] = words[28 + woff_180];
        sw_195[1] = words[28 + woff_180 + 1];
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_17[_pair * 2])[0]), "=f"((&_unpack_bf16_17[_pair * 2])[1])
                : "r"(sw_195[_pair]));
        }
        fsrc[80] = _unpack_bf16_17[0];
        fsrc[81] = _unpack_bf16_17[1];
        fsrc[82] = _unpack_bf16_17[2];
        fsrc[83] = _unpack_bf16_17[3];
        float2 v_196 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        sq[2] = fma_f32x2_rn_noftz(v_196, v_196, sq[2]);
        float2 v_197 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        sq[2] = fma_f32x2_rn_noftz(v_197, v_197, sq[2]);
        float2 v_198 = make_float2(_unpack_bf16_17[0], _unpack_bf16_17[1]);
        float2 qp_199 = make_float2(q[24], q[25]);
        dot[2] = fma_f32x2_rn_noftz(v_198, qp_199, dot[2]);
        float2 v_200 = make_float2(_unpack_bf16_17[2], _unpack_bf16_17[3]);
        float2 qp_201 = make_float2(q[26], q[27]);
        dot[2] = fma_f32x2_rn_noftz(v_200, qp_201, dot[2]);
        unsigned int sw_202[4];
        sw_202[0] = words[42 + woff_180];
        sw_202[1] = words[42 + woff_180 + 1];
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_18[_pair * 2])[0]), "=f"((&_unpack_bf16_18[_pair * 2])[1])
                : "r"(sw_202[_pair]));
        }
        float2 v_203 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        sq[3] = fma_f32x2_rn_noftz(v_203, v_203, sq[3]);
        float2 v_204 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        sq[3] = fma_f32x2_rn_noftz(v_204, v_204, sq[3]);
        float2 v_205 = make_float2(_unpack_bf16_18[0], _unpack_bf16_18[1]);
        float2 qp_206 = make_float2(q[24], q[25]);
        dot[3] = fma_f32x2_rn_noftz(v_205, qp_206, dot[3]);
        float2 v_207 = make_float2(_unpack_bf16_18[2], _unpack_bf16_18[3]);
        float2 qp_208 = make_float2(q[26], q[27]);
        dot[3] = fma_f32x2_rn_noftz(v_207, qp_208, dot[3]);
        unsigned int sw_209[4];
        sw_209[0] = words[56 + woff_180];
        sw_209[1] = words[56 + woff_180 + 1];
        __nv_bfloat162 a_210 = __as_bf16x2(sw_209[0]);
        __nv_bfloat162 d_211 = __as_bf16x2(dwords[woff_180]);
        __nv_bfloat162 mixed_212 = a_210 + d_211;
        sw_209[0] = __as_u32(mixed_212);
        words[56 + woff_180] = sw_209[0];
        __nv_bfloat162 a_213 = __as_bf16x2(sw_209[1]);
        __nv_bfloat162 d_214 = __as_bf16x2(dwords[woff_180 + 1]);
        __nv_bfloat162 mixed_215 = a_213 + d_214;
        sw_209[1] = __as_u32(mixed_215);
        words[56 + woff_180 + 1] = sw_209[1];
        {
            int2 _iv2 = make_int2(sw_209[0 + 0], sw_209[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(prefix + (row_base + (unsigned long long)base_179)) + 0) = _iv2;
        }
        unsigned long long write_base_216 = block_base + 4 * blocks_k_stride + (unsigned long long)base_179;
        {
            int2 _iv2 = make_int2(sw_209[0 + 0], sw_209[0 + 1]);
            *reinterpret_cast<int2*>(reinterpret_cast<int*>(blocks + write_base_216) + 0) = _iv2;
        }
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_19[_pair * 2])[0]), "=f"((&_unpack_bf16_19[_pair * 2])[1])
                : "r"(sw_209[_pair]));
        }
        float2 v_217 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        sq[4] = fma_f32x2_rn_noftz(v_217, v_217, sq[4]);
        float2 v_218 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        sq[4] = fma_f32x2_rn_noftz(v_218, v_218, sq[4]);
        float2 v_219 = make_float2(_unpack_bf16_19[0], _unpack_bf16_19[1]);
        float2 qp_220 = make_float2(q[24], q[25]);
        dot[4] = fma_f32x2_rn_noftz(v_219, qp_220, dot[4]);
        float2 v_221 = make_float2(_unpack_bf16_19[2], _unpack_bf16_19[3]);
        float2 qp_222 = make_float2(q[26], q[27]);
        dot[4] = fma_f32x2_rn_noftz(v_221, qp_222, dot[4]);
        float2 pairs[5];
        pairs[0] = make_float2(sq[0].x + sq[0].y, dot[0].x + dot[0].y);
        pairs[1] = make_float2(sq[1].x + sq[1].y, dot[1].x + dot[1].y);
        pairs[2] = make_float2(sq[2].x + sq[2].y, dot[2].x + dot[2].y);
        pairs[3] = make_float2(sq[3].x + sq[3].y, dot[3].x + dot[3].y);
        pairs[4] = make_float2(sq[4].x + sq[4].y, dot[4].x + dot[4].y);
        unsigned long long bits = 0;
        float2 _bit_cast_src_30 = pairs[0];
        bits = reinterpret_cast<unsigned long long*>(&_bit_cast_src_30)[0];
        unsigned long long peerbits = __shfl_xor_sync(0xFFFFFFFF, bits, 16);
        float2 peer = make_float2(0.0f, 0.0f);
        peer = reinterpret_cast<float2*>(&peerbits)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer);
        unsigned long long bits_223 = 0;
        float2 _bit_cast_src_31 = pairs[1];
        bits_223 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_31)[0];
        unsigned long long peerbits_224 = __shfl_xor_sync(0xFFFFFFFF, bits_223, 16);
        float2 peer_225 = make_float2(0.0f, 0.0f);
        peer_225 = reinterpret_cast<float2*>(&peerbits_224)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_225);
        unsigned long long bits_226 = 0;
        float2 _bit_cast_src_32 = pairs[2];
        bits_226 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_32)[0];
        unsigned long long peerbits_227 = __shfl_xor_sync(0xFFFFFFFF, bits_226, 16);
        float2 peer_228 = make_float2(0.0f, 0.0f);
        peer_228 = reinterpret_cast<float2*>(&peerbits_227)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_228);
        unsigned long long bits_229 = 0;
        float2 _bit_cast_src_33 = pairs[3];
        bits_229 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_33)[0];
        unsigned long long peerbits_230 = __shfl_xor_sync(0xFFFFFFFF, bits_229, 16);
        float2 peer_231 = make_float2(0.0f, 0.0f);
        peer_231 = reinterpret_cast<float2*>(&peerbits_230)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_231);
        unsigned long long bits_232 = 0;
        float2 _bit_cast_src_34 = pairs[4];
        bits_232 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_34)[0];
        unsigned long long peerbits_233 = __shfl_xor_sync(0xFFFFFFFF, bits_232, 16);
        float2 peer_234 = make_float2(0.0f, 0.0f);
        peer_234 = reinterpret_cast<float2*>(&peerbits_233)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_234);
        unsigned long long bits_235 = 0;
        float2 _bit_cast_src_35 = pairs[0];
        bits_235 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_35)[0];
        unsigned long long peerbits_236 = __shfl_xor_sync(0xFFFFFFFF, bits_235, 8);
        float2 peer_237 = make_float2(0.0f, 0.0f);
        peer_237 = reinterpret_cast<float2*>(&peerbits_236)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_237);
        unsigned long long bits_238 = 0;
        float2 _bit_cast_src_36 = pairs[1];
        bits_238 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_36)[0];
        unsigned long long peerbits_239 = __shfl_xor_sync(0xFFFFFFFF, bits_238, 8);
        float2 peer_240 = make_float2(0.0f, 0.0f);
        peer_240 = reinterpret_cast<float2*>(&peerbits_239)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_240);
        unsigned long long bits_241 = 0;
        float2 _bit_cast_src_37 = pairs[2];
        bits_241 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_37)[0];
        unsigned long long peerbits_242 = __shfl_xor_sync(0xFFFFFFFF, bits_241, 8);
        float2 peer_243 = make_float2(0.0f, 0.0f);
        peer_243 = reinterpret_cast<float2*>(&peerbits_242)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_243);
        unsigned long long bits_244 = 0;
        float2 _bit_cast_src_38 = pairs[3];
        bits_244 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_38)[0];
        unsigned long long peerbits_245 = __shfl_xor_sync(0xFFFFFFFF, bits_244, 8);
        float2 peer_246 = make_float2(0.0f, 0.0f);
        peer_246 = reinterpret_cast<float2*>(&peerbits_245)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_246);
        unsigned long long bits_247 = 0;
        float2 _bit_cast_src_39 = pairs[4];
        bits_247 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_39)[0];
        unsigned long long peerbits_248 = __shfl_xor_sync(0xFFFFFFFF, bits_247, 8);
        float2 peer_249 = make_float2(0.0f, 0.0f);
        peer_249 = reinterpret_cast<float2*>(&peerbits_248)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_249);
        unsigned long long bits_250 = 0;
        float2 _bit_cast_src_40 = pairs[0];
        bits_250 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_40)[0];
        unsigned long long peerbits_251 = __shfl_xor_sync(0xFFFFFFFF, bits_250, 4);
        float2 peer_252 = make_float2(0.0f, 0.0f);
        peer_252 = reinterpret_cast<float2*>(&peerbits_251)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_252);
        unsigned long long bits_253 = 0;
        float2 _bit_cast_src_41 = pairs[1];
        bits_253 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_41)[0];
        unsigned long long peerbits_254 = __shfl_xor_sync(0xFFFFFFFF, bits_253, 4);
        float2 peer_255 = make_float2(0.0f, 0.0f);
        peer_255 = reinterpret_cast<float2*>(&peerbits_254)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_255);
        unsigned long long bits_256 = 0;
        float2 _bit_cast_src_42 = pairs[2];
        bits_256 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_42)[0];
        unsigned long long peerbits_257 = __shfl_xor_sync(0xFFFFFFFF, bits_256, 4);
        float2 peer_258 = make_float2(0.0f, 0.0f);
        peer_258 = reinterpret_cast<float2*>(&peerbits_257)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_258);
        unsigned long long bits_259 = 0;
        float2 _bit_cast_src_43 = pairs[3];
        bits_259 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_43)[0];
        unsigned long long peerbits_260 = __shfl_xor_sync(0xFFFFFFFF, bits_259, 4);
        float2 peer_261 = make_float2(0.0f, 0.0f);
        peer_261 = reinterpret_cast<float2*>(&peerbits_260)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_261);
        unsigned long long bits_262 = 0;
        float2 _bit_cast_src_44 = pairs[4];
        bits_262 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_44)[0];
        unsigned long long peerbits_263 = __shfl_xor_sync(0xFFFFFFFF, bits_262, 4);
        float2 peer_264 = make_float2(0.0f, 0.0f);
        peer_264 = reinterpret_cast<float2*>(&peerbits_263)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_264);
        unsigned long long bits_265 = 0;
        float2 _bit_cast_src_45 = pairs[0];
        bits_265 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_45)[0];
        unsigned long long peerbits_266 = __shfl_xor_sync(0xFFFFFFFF, bits_265, 2);
        float2 peer_267 = make_float2(0.0f, 0.0f);
        peer_267 = reinterpret_cast<float2*>(&peerbits_266)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_267);
        unsigned long long bits_268 = 0;
        float2 _bit_cast_src_46 = pairs[1];
        bits_268 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_46)[0];
        unsigned long long peerbits_269 = __shfl_xor_sync(0xFFFFFFFF, bits_268, 2);
        float2 peer_270 = make_float2(0.0f, 0.0f);
        peer_270 = reinterpret_cast<float2*>(&peerbits_269)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_270);
        unsigned long long bits_271 = 0;
        float2 _bit_cast_src_47 = pairs[2];
        bits_271 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_47)[0];
        unsigned long long peerbits_272 = __shfl_xor_sync(0xFFFFFFFF, bits_271, 2);
        float2 peer_273 = make_float2(0.0f, 0.0f);
        peer_273 = reinterpret_cast<float2*>(&peerbits_272)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_273);
        unsigned long long bits_274 = 0;
        float2 _bit_cast_src_48 = pairs[3];
        bits_274 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_48)[0];
        unsigned long long peerbits_275 = __shfl_xor_sync(0xFFFFFFFF, bits_274, 2);
        float2 peer_276 = make_float2(0.0f, 0.0f);
        peer_276 = reinterpret_cast<float2*>(&peerbits_275)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_276);
        unsigned long long bits_277 = 0;
        float2 _bit_cast_src_49 = pairs[4];
        bits_277 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_49)[0];
        unsigned long long peerbits_278 = __shfl_xor_sync(0xFFFFFFFF, bits_277, 2);
        float2 peer_279 = make_float2(0.0f, 0.0f);
        peer_279 = reinterpret_cast<float2*>(&peerbits_278)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_279);
        unsigned long long bits_280 = 0;
        float2 _bit_cast_src_50 = pairs[0];
        bits_280 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_50)[0];
        unsigned long long peerbits_281 = __shfl_xor_sync(0xFFFFFFFF, bits_280, 1);
        float2 peer_282 = make_float2(0.0f, 0.0f);
        peer_282 = reinterpret_cast<float2*>(&peerbits_281)[0];
        pairs[0] = add_f32x2_noftz(pairs[0], peer_282);
        unsigned long long bits_283 = 0;
        float2 _bit_cast_src_51 = pairs[1];
        bits_283 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_51)[0];
        unsigned long long peerbits_284 = __shfl_xor_sync(0xFFFFFFFF, bits_283, 1);
        float2 peer_285 = make_float2(0.0f, 0.0f);
        peer_285 = reinterpret_cast<float2*>(&peerbits_284)[0];
        pairs[1] = add_f32x2_noftz(pairs[1], peer_285);
        unsigned long long bits_286 = 0;
        float2 _bit_cast_src_52 = pairs[2];
        bits_286 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_52)[0];
        unsigned long long peerbits_287 = __shfl_xor_sync(0xFFFFFFFF, bits_286, 1);
        float2 peer_288 = make_float2(0.0f, 0.0f);
        peer_288 = reinterpret_cast<float2*>(&peerbits_287)[0];
        pairs[2] = add_f32x2_noftz(pairs[2], peer_288);
        unsigned long long bits_289 = 0;
        float2 _bit_cast_src_53 = pairs[3];
        bits_289 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_53)[0];
        unsigned long long peerbits_290 = __shfl_xor_sync(0xFFFFFFFF, bits_289, 1);
        float2 peer_291 = make_float2(0.0f, 0.0f);
        peer_291 = reinterpret_cast<float2*>(&peerbits_290)[0];
        pairs[3] = add_f32x2_noftz(pairs[3], peer_291);
        unsigned long long bits_292 = 0;
        float2 _bit_cast_src_54 = pairs[4];
        bits_292 = reinterpret_cast<unsigned long long*>(&_bit_cast_src_54)[0];
        unsigned long long peerbits_293 = __shfl_xor_sync(0xFFFFFFFF, bits_292, 1);
        float2 peer_294 = make_float2(0.0f, 0.0f);
        peer_294 = reinterpret_cast<float2*>(&peerbits_293)[0];
        pairs[4] = add_f32x2_noftz(pairs[4], peer_294);
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (lane == 0) {
            uint32_t _cluster_mapa_55;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_55) : "r"(stats_addr + (unsigned int)(warp_0 * 5 * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_55), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_56;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_56) : "r"(stats_addr + (unsigned int)((warp_0 * 5 * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_56), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_57;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_57) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 1) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_57), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_58;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_58) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 1) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_58), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_59;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_59) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 2) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_59), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_60;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_60) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 2) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_60), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_61;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_61) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 3) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_61), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_62;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_62) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 3) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_62), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_63;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_63) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 4) * 2 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_63), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_64;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_64) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 4) * 2 + 1) * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_64), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_65;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_65) : "r"(stats_addr + (unsigned int)(warp_0 * 5 * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_65), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_66;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_66) : "r"(stats_addr + (unsigned int)((warp_0 * 5 * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_66), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_67;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_67) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 1) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_67), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_68;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_68) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 1) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_68), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_69;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_69) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 2) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_69), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_70;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_70) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 2) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_70), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_71;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_71) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 3) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_71), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_72;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_72) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 3) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_72), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_73;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_73) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 4) * 2 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_73), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_74;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_74) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 4) * 2 + 1) * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_74), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_75;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_75) : "r"(stats_addr + (unsigned int)(warp_0 * 5 * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_75), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_76;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_76) : "r"(stats_addr + (unsigned int)((warp_0 * 5 * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_76), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_77;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_77) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 1) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_77), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_78;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_78) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 1) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_78), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_79;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_79) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 2) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_79), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_80;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_80) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 2) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_80), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_81;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_81) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 3) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_81), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_82;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_82) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 3) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_82), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_83;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_83) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 4) * 2 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_83), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_84;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_84) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 4) * 2 + 1) * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_84), "f"(pairs[4].y) : "memory");
            uint32_t _cluster_mapa_85;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_85) : "r"(stats_addr + (unsigned int)(warp_0 * 5 * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_85), "f"(pairs[0].x) : "memory");
            uint32_t _cluster_mapa_86;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_86) : "r"(stats_addr + (unsigned int)((warp_0 * 5 * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_86), "f"(pairs[0].y) : "memory");
            uint32_t _cluster_mapa_87;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_87) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 1) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_87), "f"(pairs[1].x) : "memory");
            uint32_t _cluster_mapa_88;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_88) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 1) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_88), "f"(pairs[1].y) : "memory");
            uint32_t _cluster_mapa_89;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_89) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 2) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_89), "f"(pairs[2].x) : "memory");
            uint32_t _cluster_mapa_90;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_90) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 2) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_90), "f"(pairs[2].y) : "memory");
            uint32_t _cluster_mapa_91;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_91) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 3) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_91), "f"(pairs[3].x) : "memory");
            uint32_t _cluster_mapa_92;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_92) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 3) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_92), "f"(pairs[3].y) : "memory");
            uint32_t _cluster_mapa_93;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_93) : "r"(stats_addr + (unsigned int)((warp_0 * 5 + 4) * 2 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_93), "f"(pairs[4].x) : "memory");
            uint32_t _cluster_mapa_94;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_94) : "r"(stats_addr + (unsigned int)(((warp_0 * 5 + 4) * 2 + 1) * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_94), "f"(pairs[4].y) : "memory");
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
        float _shfl_down_sync_95 = __shfl_down_sync(0xFFFFFFFF, total_sq, 4, 8);
        total_sq += _shfl_down_sync_95;
        float _shfl_down_sync_96 = __shfl_down_sync(0xFFFFFFFF, total_dot, 4, 8);
        total_dot += _shfl_down_sync_96;
        float _shfl_down_sync_97 = __shfl_down_sync(0xFFFFFFFF, total_sq, 2, 8);
        total_sq += _shfl_down_sync_97;
        float _shfl_down_sync_98 = __shfl_down_sync(0xFFFFFFFF, total_dot, 2, 8);
        total_dot += _shfl_down_sync_98;
        float _shfl_down_sync_99 = __shfl_down_sync(0xFFFFFFFF, total_sq, 1, 8);
        total_sq += _shfl_down_sync_99;
        float _shfl_down_sync_100 = __shfl_down_sync(0xFFFFFFFF, total_dot, 1, 8);
        total_dot += _shfl_down_sync_100;
        float logit = 0.0f;
        if (stat_n < 5 && stat_w == 0) {
            float sigma = rsqrtf(total_sq / 7168.0f + eps);
            logit = total_dot * sigma;
        }
        float logits[5];
        float _shfl_sync_101 = __shfl_sync(0xFFFFFFFF, logit, 0);
        logits[0] = _shfl_sync_101;
        float _shfl_sync_102 = __shfl_sync(0xFFFFFFFF, logit, 8);
        logits[1] = _shfl_sync_102;
        float _shfl_sync_103 = __shfl_sync(0xFFFFFFFF, logit, 16);
        logits[2] = _shfl_sync_103;
        float _shfl_sync_104 = __shfl_sync(0xFFFFFFFF, logit, 24);
        logits[3] = _shfl_sync_104;
        float total_sq2 = 0.0f;
        float total_dot2 = 0.0f;
        float logit2 = 0.0f;
        total_sq2 = 0.0f;
        total_dot2 = 0.0f;
        if (stat_n < 1) {
            total_sq2 = stats[(stat_w * 5 + 4 + stat_n) * 2];
            total_dot2 = stats[(stat_w * 5 + 4 + stat_n) * 2 + 1];
        }
        float _shfl_down_sync_105 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 4, 8);
        total_sq2 += _shfl_down_sync_105;
        float _shfl_down_sync_106 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 4, 8);
        total_dot2 += _shfl_down_sync_106;
        float _shfl_down_sync_107 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 2, 8);
        total_sq2 += _shfl_down_sync_107;
        float _shfl_down_sync_108 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 2, 8);
        total_dot2 += _shfl_down_sync_108;
        float _shfl_down_sync_109 = __shfl_down_sync(0xFFFFFFFF, total_sq2, 1, 8);
        total_sq2 += _shfl_down_sync_109;
        float _shfl_down_sync_110 = __shfl_down_sync(0xFFFFFFFF, total_dot2, 1, 8);
        total_dot2 += _shfl_down_sync_110;
        logit2 = 0.0f;
        if (stat_n < 1 && stat_w == 0) {
            float sigma2 = rsqrtf(total_sq2 / 7168.0f + eps);
            logit2 = total_dot2 * sigma2;
        }
        float _shfl_sync_111 = __shfl_sync(0xFFFFFFFF, logit2, 0);
        logits[4] = _shfl_sync_111;
        float max_running = -3.4028235e+38f;
        float sum_running = 0.0f;
        float max_chunk = -3.4028235e+38f;
        max_chunk = fmaxf(max_chunk, logits[0]);
        max_chunk = fmaxf(max_chunk, logits[1]);
        max_chunk = fmaxf(max_chunk, logits[2]);
        float max_new = fmaxf(max_running, max_chunk);
        float correction = approx_exp2((max_running - max_new) * 1.442695f);
        float weights[5];
        float sum_weights = 0.0f;
        float _exp2_112 = approx_exp2((logits[0] - max_new) * 1.442695f);
        weights[0] = _exp2_112;
        sum_weights += weights[0];
        float _exp2_113 = approx_exp2((logits[1] - max_new) * 1.442695f);
        weights[1] = _exp2_113;
        sum_weights += weights[1];
        float _exp2_114 = approx_exp2((logits[2] - max_new) * 1.442695f);
        weights[2] = _exp2_114;
        sum_weights += weights[2];
        float2 corr = make_float2(correction, correction);
        const int woff_295 = 0;
        int base_296 = group * 1024 + thread * 8;
        float2 a_297[4];
        float2 previous = make_float2(acc[0], acc[1]);
        a_297[0] = mul_f32x2_noftz(previous, corr);
        float2 previous_298 = make_float2(acc[2], acc[3]);
        a_297[1] = mul_f32x2_noftz(previous_298, corr);
        float2 previous_299 = make_float2(acc[4], acc[5]);
        a_297[2] = mul_f32x2_noftz(previous_299, corr);
        float2 previous_300 = make_float2(acc[6], acc[7]);
        a_297[3] = mul_f32x2_noftz(previous_300, corr);
        float2 weight = make_float2(weights[0], weights[0]);
        float2 v_301 = make_float2(fsrc[0], fsrc[1]);
        a_297[0] = fma_f32x2_rn_noftz(weight, v_301, a_297[0]);
        float2 v_302 = make_float2(fsrc[2], fsrc[3]);
        a_297[1] = fma_f32x2_rn_noftz(weight, v_302, a_297[1]);
        float2 v_303 = make_float2(fsrc[4], fsrc[5]);
        a_297[2] = fma_f32x2_rn_noftz(weight, v_303, a_297[2]);
        float2 v_304 = make_float2(fsrc[6], fsrc[7]);
        a_297[3] = fma_f32x2_rn_noftz(weight, v_304, a_297[3]);
        float2 weight_305 = make_float2(weights[1], weights[1]);
        float2 v_306 = make_float2(fsrc[28], fsrc[29]);
        a_297[0] = fma_f32x2_rn_noftz(weight_305, v_306, a_297[0]);
        float2 v_307 = make_float2(fsrc[30], fsrc[31]);
        a_297[1] = fma_f32x2_rn_noftz(weight_305, v_307, a_297[1]);
        float2 v_308 = make_float2(fsrc[32], fsrc[33]);
        a_297[2] = fma_f32x2_rn_noftz(weight_305, v_308, a_297[2]);
        float2 v_309 = make_float2(fsrc[34], fsrc[35]);
        a_297[3] = fma_f32x2_rn_noftz(weight_305, v_309, a_297[3]);
        float2 weight_310 = make_float2(weights[2], weights[2]);
        float2 v_311 = make_float2(fsrc[56], fsrc[57]);
        a_297[0] = fma_f32x2_rn_noftz(weight_310, v_311, a_297[0]);
        float2 v_312 = make_float2(fsrc[58], fsrc[59]);
        a_297[1] = fma_f32x2_rn_noftz(weight_310, v_312, a_297[1]);
        float2 v_313 = make_float2(fsrc[60], fsrc[61]);
        a_297[2] = fma_f32x2_rn_noftz(weight_310, v_313, a_297[2]);
        float2 v_314 = make_float2(fsrc[62], fsrc[63]);
        a_297[3] = fma_f32x2_rn_noftz(weight_310, v_314, a_297[3]);
        acc[0] = a_297[0].x;
        acc[1] = a_297[0].y;
        acc[2] = a_297[1].x;
        acc[3] = a_297[1].y;
        acc[4] = a_297[2].x;
        acc[5] = a_297[2].y;
        acc[6] = a_297[3].x;
        acc[7] = a_297[3].y;
        const int woff_315 = 4;
        int base_316 = (2 + group) * 1024 + thread * 8;
        float2 a_317[4];
        float2 previous_318 = make_float2(acc[8], acc[9]);
        a_317[0] = mul_f32x2_noftz(previous_318, corr);
        float2 previous_319 = make_float2(acc[10], acc[11]);
        a_317[1] = mul_f32x2_noftz(previous_319, corr);
        float2 previous_320 = make_float2(acc[12], acc[13]);
        a_317[2] = mul_f32x2_noftz(previous_320, corr);
        float2 previous_321 = make_float2(acc[14], acc[15]);
        a_317[3] = mul_f32x2_noftz(previous_321, corr);
        float2 weight_322 = make_float2(weights[0], weights[0]);
        float2 v_323 = make_float2(fsrc[8], fsrc[9]);
        a_317[0] = fma_f32x2_rn_noftz(weight_322, v_323, a_317[0]);
        float2 v_324 = make_float2(fsrc[10], fsrc[11]);
        a_317[1] = fma_f32x2_rn_noftz(weight_322, v_324, a_317[1]);
        float2 v_325 = make_float2(fsrc[12], fsrc[13]);
        a_317[2] = fma_f32x2_rn_noftz(weight_322, v_325, a_317[2]);
        float2 v_326 = make_float2(fsrc[14], fsrc[15]);
        a_317[3] = fma_f32x2_rn_noftz(weight_322, v_326, a_317[3]);
        float2 weight_327 = make_float2(weights[1], weights[1]);
        float2 v_328 = make_float2(fsrc[36], fsrc[37]);
        a_317[0] = fma_f32x2_rn_noftz(weight_327, v_328, a_317[0]);
        float2 v_329 = make_float2(fsrc[38], fsrc[39]);
        a_317[1] = fma_f32x2_rn_noftz(weight_327, v_329, a_317[1]);
        float2 v_330 = make_float2(fsrc[40], fsrc[41]);
        a_317[2] = fma_f32x2_rn_noftz(weight_327, v_330, a_317[2]);
        float2 v_331 = make_float2(fsrc[42], fsrc[43]);
        a_317[3] = fma_f32x2_rn_noftz(weight_327, v_331, a_317[3]);
        float2 weight_332 = make_float2(weights[2], weights[2]);
        float2 v_333 = make_float2(fsrc[64], fsrc[65]);
        a_317[0] = fma_f32x2_rn_noftz(weight_332, v_333, a_317[0]);
        float2 v_334 = make_float2(fsrc[66], fsrc[67]);
        a_317[1] = fma_f32x2_rn_noftz(weight_332, v_334, a_317[1]);
        float2 v_335 = make_float2(fsrc[68], fsrc[69]);
        a_317[2] = fma_f32x2_rn_noftz(weight_332, v_335, a_317[2]);
        float2 v_336 = make_float2(fsrc[70], fsrc[71]);
        a_317[3] = fma_f32x2_rn_noftz(weight_332, v_336, a_317[3]);
        acc[8] = a_317[0].x;
        acc[9] = a_317[0].y;
        acc[10] = a_317[1].x;
        acc[11] = a_317[1].y;
        acc[12] = a_317[2].x;
        acc[13] = a_317[2].y;
        acc[14] = a_317[3].x;
        acc[15] = a_317[3].y;
        const int woff_337 = 8;
        int base_338 = (4 + group) * 1024 + thread * 8;
        float2 a_339[4];
        float2 previous_340 = make_float2(acc[16], acc[17]);
        a_339[0] = mul_f32x2_noftz(previous_340, corr);
        float2 previous_341 = make_float2(acc[18], acc[19]);
        a_339[1] = mul_f32x2_noftz(previous_341, corr);
        float2 previous_342 = make_float2(acc[20], acc[21]);
        a_339[2] = mul_f32x2_noftz(previous_342, corr);
        float2 previous_343 = make_float2(acc[22], acc[23]);
        a_339[3] = mul_f32x2_noftz(previous_343, corr);
        float2 weight_344 = make_float2(weights[0], weights[0]);
        float2 v_345 = make_float2(fsrc[16], fsrc[17]);
        a_339[0] = fma_f32x2_rn_noftz(weight_344, v_345, a_339[0]);
        float2 v_346 = make_float2(fsrc[18], fsrc[19]);
        a_339[1] = fma_f32x2_rn_noftz(weight_344, v_346, a_339[1]);
        float2 v_347 = make_float2(fsrc[20], fsrc[21]);
        a_339[2] = fma_f32x2_rn_noftz(weight_344, v_347, a_339[2]);
        float2 v_348 = make_float2(fsrc[22], fsrc[23]);
        a_339[3] = fma_f32x2_rn_noftz(weight_344, v_348, a_339[3]);
        float2 weight_349 = make_float2(weights[1], weights[1]);
        float2 v_350 = make_float2(fsrc[44], fsrc[45]);
        a_339[0] = fma_f32x2_rn_noftz(weight_349, v_350, a_339[0]);
        float2 v_351 = make_float2(fsrc[46], fsrc[47]);
        a_339[1] = fma_f32x2_rn_noftz(weight_349, v_351, a_339[1]);
        float2 v_352 = make_float2(fsrc[48], fsrc[49]);
        a_339[2] = fma_f32x2_rn_noftz(weight_349, v_352, a_339[2]);
        float2 v_353 = make_float2(fsrc[50], fsrc[51]);
        a_339[3] = fma_f32x2_rn_noftz(weight_349, v_353, a_339[3]);
        float2 weight_354 = make_float2(weights[2], weights[2]);
        float2 v_355 = make_float2(fsrc[72], fsrc[73]);
        a_339[0] = fma_f32x2_rn_noftz(weight_354, v_355, a_339[0]);
        float2 v_356 = make_float2(fsrc[74], fsrc[75]);
        a_339[1] = fma_f32x2_rn_noftz(weight_354, v_356, a_339[1]);
        float2 v_357 = make_float2(fsrc[76], fsrc[77]);
        a_339[2] = fma_f32x2_rn_noftz(weight_354, v_357, a_339[2]);
        float2 v_358 = make_float2(fsrc[78], fsrc[79]);
        a_339[3] = fma_f32x2_rn_noftz(weight_354, v_358, a_339[3]);
        acc[16] = a_339[0].x;
        acc[17] = a_339[0].y;
        acc[18] = a_339[1].x;
        acc[19] = a_339[1].y;
        acc[20] = a_339[2].x;
        acc[21] = a_339[2].y;
        acc[22] = a_339[3].x;
        acc[23] = a_339[3].y;
        const int woff_359 = 12;
        int base_360 = 6144 + group * 512 + thread * 4;
        float2 a_361[4];
        float2 previous_362 = make_float2(acc[24], acc[25]);
        a_361[0] = mul_f32x2_noftz(previous_362, corr);
        float2 previous_363 = make_float2(acc[26], acc[27]);
        a_361[1] = mul_f32x2_noftz(previous_363, corr);
        float2 weight_364 = make_float2(weights[0], weights[0]);
        float2 v_365 = make_float2(fsrc[24], fsrc[25]);
        a_361[0] = fma_f32x2_rn_noftz(weight_364, v_365, a_361[0]);
        float2 v_366 = make_float2(fsrc[26], fsrc[27]);
        a_361[1] = fma_f32x2_rn_noftz(weight_364, v_366, a_361[1]);
        float2 weight_367 = make_float2(weights[1], weights[1]);
        float2 v_368 = make_float2(fsrc[52], fsrc[53]);
        a_361[0] = fma_f32x2_rn_noftz(weight_367, v_368, a_361[0]);
        float2 v_369 = make_float2(fsrc[54], fsrc[55]);
        a_361[1] = fma_f32x2_rn_noftz(weight_367, v_369, a_361[1]);
        float2 weight_370 = make_float2(weights[2], weights[2]);
        float2 v_371 = make_float2(fsrc[80], fsrc[81]);
        a_361[0] = fma_f32x2_rn_noftz(weight_370, v_371, a_361[0]);
        float2 v_372 = make_float2(fsrc[82], fsrc[83]);
        a_361[1] = fma_f32x2_rn_noftz(weight_370, v_372, a_361[1]);
        acc[24] = a_361[0].x;
        acc[25] = a_361[0].y;
        acc[26] = a_361[1].x;
        acc[27] = a_361[1].y;
        sum_running = __fmaf_rn(sum_running, correction, sum_weights);
        max_running = max_new;
        float max_chunk_373 = -3.4028235e+38f;
        max_chunk_373 = fmaxf(max_chunk_373, logits[3]);
        max_chunk_373 = fmaxf(max_chunk_373, logits[4]);
        float max_new_374 = fmaxf(max_running, max_chunk_373);
        float correction_375 = approx_exp2((max_running - max_new_374) * 1.442695f);
        float weights_376[5];
        float sum_weights_377 = 0.0f;
        float _exp2_115 = approx_exp2((logits[3] - max_new_374) * 1.442695f);
        weights_376[3] = _exp2_115;
        sum_weights_377 += weights_376[3];
        float _exp2_116 = approx_exp2((logits[4] - max_new_374) * 1.442695f);
        weights_376[4] = _exp2_116;
        sum_weights_377 += weights_376[4];
        float2 corr_378 = make_float2(correction_375, correction_375);
        const int woff_379 = 0;
        int base_380 = group * 1024 + thread * 8;
        float2 a_381[4];
        float2 previous_382 = make_float2(acc[0], acc[1]);
        a_381[0] = mul_f32x2_noftz(previous_382, corr_378);
        float2 previous_383 = make_float2(acc[2], acc[3]);
        a_381[1] = mul_f32x2_noftz(previous_383, corr_378);
        float2 previous_384 = make_float2(acc[4], acc[5]);
        a_381[2] = mul_f32x2_noftz(previous_384, corr_378);
        float2 previous_385 = make_float2(acc[6], acc[7]);
        a_381[3] = mul_f32x2_noftz(previous_385, corr_378);
        float2 weight_386 = make_float2(weights_376[3], weights_376[3]);
        unsigned int sw2[4];
        sw2[0] = words[42 + woff_379];
        sw2[1] = words[42 + woff_379 + 1];
        sw2[2] = words[42 + woff_379 + 2];
        sw2[3] = words[42 + woff_379 + 3];
        float _unpack_bf16_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_20[_pair * 2])[0]), "=f"((&_unpack_bf16_20[_pair * 2])[1])
                : "r"(sw2[_pair]));
        }
        float2 v_387 = make_float2(_unpack_bf16_20[0], _unpack_bf16_20[1]);
        a_381[0] = fma_f32x2_rn_noftz(weight_386, v_387, a_381[0]);
        float2 v_388 = make_float2(_unpack_bf16_20[2], _unpack_bf16_20[3]);
        a_381[1] = fma_f32x2_rn_noftz(weight_386, v_388, a_381[1]);
        float2 v_389 = make_float2(_unpack_bf16_20[4], _unpack_bf16_20[5]);
        a_381[2] = fma_f32x2_rn_noftz(weight_386, v_389, a_381[2]);
        float2 v_390 = make_float2(_unpack_bf16_20[6], _unpack_bf16_20[7]);
        a_381[3] = fma_f32x2_rn_noftz(weight_386, v_390, a_381[3]);
        float2 weight_391 = make_float2(weights_376[4], weights_376[4]);
        unsigned int sw2_392[4];
        sw2_392[0] = words[56 + woff_379];
        sw2_392[1] = words[56 + woff_379 + 1];
        sw2_392[2] = words[56 + woff_379 + 2];
        sw2_392[3] = words[56 + woff_379 + 3];
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_21[_pair * 2])[0]), "=f"((&_unpack_bf16_21[_pair * 2])[1])
                : "r"(sw2_392[_pair]));
        }
        float2 v_393 = make_float2(_unpack_bf16_21[0], _unpack_bf16_21[1]);
        a_381[0] = fma_f32x2_rn_noftz(weight_391, v_393, a_381[0]);
        float2 v_394 = make_float2(_unpack_bf16_21[2], _unpack_bf16_21[3]);
        a_381[1] = fma_f32x2_rn_noftz(weight_391, v_394, a_381[1]);
        float2 v_395 = make_float2(_unpack_bf16_21[4], _unpack_bf16_21[5]);
        a_381[2] = fma_f32x2_rn_noftz(weight_391, v_395, a_381[2]);
        float2 v_396 = make_float2(_unpack_bf16_21[6], _unpack_bf16_21[7]);
        a_381[3] = fma_f32x2_rn_noftz(weight_391, v_396, a_381[3]);
        acc[0] = a_381[0].x;
        acc[1] = a_381[0].y;
        acc[2] = a_381[1].x;
        acc[3] = a_381[1].y;
        acc[4] = a_381[2].x;
        acc[5] = a_381[2].y;
        acc[6] = a_381[3].x;
        acc[7] = a_381[3].y;
        const int woff_397 = 4;
        int base_398 = (2 + group) * 1024 + thread * 8;
        float2 a_399[4];
        float2 previous_400 = make_float2(acc[8], acc[9]);
        a_399[0] = mul_f32x2_noftz(previous_400, corr_378);
        float2 previous_401 = make_float2(acc[10], acc[11]);
        a_399[1] = mul_f32x2_noftz(previous_401, corr_378);
        float2 previous_402 = make_float2(acc[12], acc[13]);
        a_399[2] = mul_f32x2_noftz(previous_402, corr_378);
        float2 previous_403 = make_float2(acc[14], acc[15]);
        a_399[3] = mul_f32x2_noftz(previous_403, corr_378);
        float2 weight_404 = make_float2(weights_376[3], weights_376[3]);
        unsigned int sw2_405[4];
        sw2_405[0] = words[42 + woff_397];
        sw2_405[1] = words[42 + woff_397 + 1];
        sw2_405[2] = words[42 + woff_397 + 2];
        sw2_405[3] = words[42 + woff_397 + 3];
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_22[_pair * 2])[0]), "=f"((&_unpack_bf16_22[_pair * 2])[1])
                : "r"(sw2_405[_pair]));
        }
        float2 v_406 = make_float2(_unpack_bf16_22[0], _unpack_bf16_22[1]);
        a_399[0] = fma_f32x2_rn_noftz(weight_404, v_406, a_399[0]);
        float2 v_407 = make_float2(_unpack_bf16_22[2], _unpack_bf16_22[3]);
        a_399[1] = fma_f32x2_rn_noftz(weight_404, v_407, a_399[1]);
        float2 v_408 = make_float2(_unpack_bf16_22[4], _unpack_bf16_22[5]);
        a_399[2] = fma_f32x2_rn_noftz(weight_404, v_408, a_399[2]);
        float2 v_409 = make_float2(_unpack_bf16_22[6], _unpack_bf16_22[7]);
        a_399[3] = fma_f32x2_rn_noftz(weight_404, v_409, a_399[3]);
        float2 weight_410 = make_float2(weights_376[4], weights_376[4]);
        unsigned int sw2_411[4];
        sw2_411[0] = words[56 + woff_397];
        sw2_411[1] = words[56 + woff_397 + 1];
        sw2_411[2] = words[56 + woff_397 + 2];
        sw2_411[3] = words[56 + woff_397 + 3];
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_23[_pair * 2])[0]), "=f"((&_unpack_bf16_23[_pair * 2])[1])
                : "r"(sw2_411[_pair]));
        }
        float2 v_412 = make_float2(_unpack_bf16_23[0], _unpack_bf16_23[1]);
        a_399[0] = fma_f32x2_rn_noftz(weight_410, v_412, a_399[0]);
        float2 v_413 = make_float2(_unpack_bf16_23[2], _unpack_bf16_23[3]);
        a_399[1] = fma_f32x2_rn_noftz(weight_410, v_413, a_399[1]);
        float2 v_414 = make_float2(_unpack_bf16_23[4], _unpack_bf16_23[5]);
        a_399[2] = fma_f32x2_rn_noftz(weight_410, v_414, a_399[2]);
        float2 v_415 = make_float2(_unpack_bf16_23[6], _unpack_bf16_23[7]);
        a_399[3] = fma_f32x2_rn_noftz(weight_410, v_415, a_399[3]);
        acc[8] = a_399[0].x;
        acc[9] = a_399[0].y;
        acc[10] = a_399[1].x;
        acc[11] = a_399[1].y;
        acc[12] = a_399[2].x;
        acc[13] = a_399[2].y;
        acc[14] = a_399[3].x;
        acc[15] = a_399[3].y;
        const int woff_416 = 8;
        int base_417 = (4 + group) * 1024 + thread * 8;
        float2 a_418[4];
        float2 previous_419 = make_float2(acc[16], acc[17]);
        a_418[0] = mul_f32x2_noftz(previous_419, corr_378);
        float2 previous_420 = make_float2(acc[18], acc[19]);
        a_418[1] = mul_f32x2_noftz(previous_420, corr_378);
        float2 previous_421 = make_float2(acc[20], acc[21]);
        a_418[2] = mul_f32x2_noftz(previous_421, corr_378);
        float2 previous_422 = make_float2(acc[22], acc[23]);
        a_418[3] = mul_f32x2_noftz(previous_422, corr_378);
        float2 weight_423 = make_float2(weights_376[3], weights_376[3]);
        unsigned int sw2_424[4];
        sw2_424[0] = words[42 + woff_416];
        sw2_424[1] = words[42 + woff_416 + 1];
        sw2_424[2] = words[42 + woff_416 + 2];
        sw2_424[3] = words[42 + woff_416 + 3];
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_24[_pair * 2])[0]), "=f"((&_unpack_bf16_24[_pair * 2])[1])
                : "r"(sw2_424[_pair]));
        }
        float2 v_425 = make_float2(_unpack_bf16_24[0], _unpack_bf16_24[1]);
        a_418[0] = fma_f32x2_rn_noftz(weight_423, v_425, a_418[0]);
        float2 v_426 = make_float2(_unpack_bf16_24[2], _unpack_bf16_24[3]);
        a_418[1] = fma_f32x2_rn_noftz(weight_423, v_426, a_418[1]);
        float2 v_427 = make_float2(_unpack_bf16_24[4], _unpack_bf16_24[5]);
        a_418[2] = fma_f32x2_rn_noftz(weight_423, v_427, a_418[2]);
        float2 v_428 = make_float2(_unpack_bf16_24[6], _unpack_bf16_24[7]);
        a_418[3] = fma_f32x2_rn_noftz(weight_423, v_428, a_418[3]);
        float2 weight_429 = make_float2(weights_376[4], weights_376[4]);
        unsigned int sw2_430[4];
        sw2_430[0] = words[56 + woff_416];
        sw2_430[1] = words[56 + woff_416 + 1];
        sw2_430[2] = words[56 + woff_416 + 2];
        sw2_430[3] = words[56 + woff_416 + 3];
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_25[_pair * 2])[0]), "=f"((&_unpack_bf16_25[_pair * 2])[1])
                : "r"(sw2_430[_pair]));
        }
        float2 v_431 = make_float2(_unpack_bf16_25[0], _unpack_bf16_25[1]);
        a_418[0] = fma_f32x2_rn_noftz(weight_429, v_431, a_418[0]);
        float2 v_432 = make_float2(_unpack_bf16_25[2], _unpack_bf16_25[3]);
        a_418[1] = fma_f32x2_rn_noftz(weight_429, v_432, a_418[1]);
        float2 v_433 = make_float2(_unpack_bf16_25[4], _unpack_bf16_25[5]);
        a_418[2] = fma_f32x2_rn_noftz(weight_429, v_433, a_418[2]);
        float2 v_434 = make_float2(_unpack_bf16_25[6], _unpack_bf16_25[7]);
        a_418[3] = fma_f32x2_rn_noftz(weight_429, v_434, a_418[3]);
        acc[16] = a_418[0].x;
        acc[17] = a_418[0].y;
        acc[18] = a_418[1].x;
        acc[19] = a_418[1].y;
        acc[20] = a_418[2].x;
        acc[21] = a_418[2].y;
        acc[22] = a_418[3].x;
        acc[23] = a_418[3].y;
        const int woff_435 = 12;
        int base_436 = 6144 + group * 512 + thread * 4;
        float2 a_437[4];
        float2 previous_438 = make_float2(acc[24], acc[25]);
        a_437[0] = mul_f32x2_noftz(previous_438, corr_378);
        float2 previous_439 = make_float2(acc[26], acc[27]);
        a_437[1] = mul_f32x2_noftz(previous_439, corr_378);
        float2 weight_440 = make_float2(weights_376[3], weights_376[3]);
        unsigned int sw2_441[4];
        sw2_441[0] = words[42 + woff_435];
        sw2_441[1] = words[42 + woff_435 + 1];
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_26[_pair * 2])[0]), "=f"((&_unpack_bf16_26[_pair * 2])[1])
                : "r"(sw2_441[_pair]));
        }
        float2 v_442 = make_float2(_unpack_bf16_26[0], _unpack_bf16_26[1]);
        a_437[0] = fma_f32x2_rn_noftz(weight_440, v_442, a_437[0]);
        float2 v_443 = make_float2(_unpack_bf16_26[2], _unpack_bf16_26[3]);
        a_437[1] = fma_f32x2_rn_noftz(weight_440, v_443, a_437[1]);
        float2 weight_444 = make_float2(weights_376[4], weights_376[4]);
        unsigned int sw2_445[4];
        sw2_445[0] = words[56 + woff_435];
        sw2_445[1] = words[56 + woff_435 + 1];
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_unpack_bf16_27[_pair * 2])[0]), "=f"((&_unpack_bf16_27[_pair * 2])[1])
                : "r"(sw2_445[_pair]));
        }
        float2 v_446 = make_float2(_unpack_bf16_27[0], _unpack_bf16_27[1]);
        a_437[0] = fma_f32x2_rn_noftz(weight_444, v_446, a_437[0]);
        float2 v_447 = make_float2(_unpack_bf16_27[2], _unpack_bf16_27[3]);
        a_437[1] = fma_f32x2_rn_noftz(weight_444, v_447, a_437[1]);
        acc[24] = a_437[0].x;
        acc[25] = a_437[0].y;
        acc[26] = a_437[1].x;
        acc[27] = a_437[1].y;
        sum_running = __fmaf_rn(sum_running, correction_375, sum_weights_377);
        max_running = max_new_374;
        float2 output_sq_pair = make_float2(0.0f, 0.0f);
        float2 v_448 = make_float2(acc[0], acc[1]);
        output_sq_pair = fma_f32x2_rn_noftz(v_448, v_448, output_sq_pair);
        float2 v_449 = make_float2(acc[2], acc[3]);
        output_sq_pair = fma_f32x2_rn_noftz(v_449, v_449, output_sq_pair);
        float2 v_450 = make_float2(acc[4], acc[5]);
        output_sq_pair = fma_f32x2_rn_noftz(v_450, v_450, output_sq_pair);
        float2 v_451 = make_float2(acc[6], acc[7]);
        output_sq_pair = fma_f32x2_rn_noftz(v_451, v_451, output_sq_pair);
        float2 v_452 = make_float2(acc[8], acc[9]);
        output_sq_pair = fma_f32x2_rn_noftz(v_452, v_452, output_sq_pair);
        float2 v_453 = make_float2(acc[10], acc[11]);
        output_sq_pair = fma_f32x2_rn_noftz(v_453, v_453, output_sq_pair);
        float2 v_454 = make_float2(acc[12], acc[13]);
        output_sq_pair = fma_f32x2_rn_noftz(v_454, v_454, output_sq_pair);
        float2 v_455 = make_float2(acc[14], acc[15]);
        output_sq_pair = fma_f32x2_rn_noftz(v_455, v_455, output_sq_pair);
        float2 v_456 = make_float2(acc[16], acc[17]);
        output_sq_pair = fma_f32x2_rn_noftz(v_456, v_456, output_sq_pair);
        float2 v_457 = make_float2(acc[18], acc[19]);
        output_sq_pair = fma_f32x2_rn_noftz(v_457, v_457, output_sq_pair);
        float2 v_458 = make_float2(acc[20], acc[21]);
        output_sq_pair = fma_f32x2_rn_noftz(v_458, v_458, output_sq_pair);
        float2 v_459 = make_float2(acc[22], acc[23]);
        output_sq_pair = fma_f32x2_rn_noftz(v_459, v_459, output_sq_pair);
        float2 v_460 = make_float2(acc[24], acc[25]);
        output_sq_pair = fma_f32x2_rn_noftz(v_460, v_460, output_sq_pair);
        float2 v_461 = make_float2(acc[26], acc[27]);
        output_sq_pair = fma_f32x2_rn_noftz(v_461, v_461, output_sq_pair);
        float output_sq = output_sq_pair.x + output_sq_pair.y;
        float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 16);
        output_sq += _shfl_xor_sync_117;
        float _shfl_xor_sync_118 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 8);
        output_sq += _shfl_xor_sync_118;
        float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 4);
        output_sq += _shfl_xor_sync_119;
        float _shfl_xor_sync_120 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 2);
        output_sq += _shfl_xor_sync_120;
        float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, output_sq, 1);
        output_sq += _shfl_xor_sync_121;
        if (lane == 0) {
            uint32_t _cluster_mapa_122;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_122) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_122), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_123;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_123) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_123), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_124;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_124) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_124), "f"(output_sq) : "memory");
            uint32_t _cluster_mapa_125;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_125) : "r"(out_stats_addr + (unsigned int)(warp_0 * 4)), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_cluster_mapa_125), "f"(output_sq) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float output_total = ((lane < 8) ? out_stats[lane] : 0.0f);
        float _shfl_down_sync_126 = __shfl_down_sync(0xFFFFFFFF, output_total, 4, 8);
        output_total += _shfl_down_sync_126;
        float _shfl_down_sync_127 = __shfl_down_sync(0xFFFFFFFF, output_total, 2, 8);
        output_total += _shfl_down_sync_127;
        float _shfl_down_sync_128 = __shfl_down_sync(0xFFFFFFFF, output_total, 1, 8);
        output_total += _shfl_down_sync_128;
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        float rsigma_lane = 0.0f;
        if (lane == 0) {
            float _rsqrt_129 = rsqrtf(__fmaf_rn(output_norm_eps * sum_running, sum_running, output_total / 7168.0f));
            rsigma_lane = _rsqrt_129;
        }
        float rsigma = __shfl_sync(0xFFFFFFFF, rsigma_lane, 0);
        float2 rsigma_pair = make_float2(rsigma, rsigma);
        int base_462 = group * 1024 + thread * 8;
        float output_values[8];
        const int value_idx = 0;
        const int acc_idx = value_idx;
        float2 _f32x2_mul_a_130 = make_float2(acc[acc_idx], acc[acc_idx + 1]);
        float2 scaled_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_131 = make_float2(wout[acc_idx], wout[acc_idx + 1]);
        float2 normalized_pair;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair) : "l"(*(const unsigned long long*)&scaled_pair), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
        output_values[value_idx] = normalized_pair.x;
        output_values[value_idx + 1] = normalized_pair.y;
        const int value_idx_463 = 2;
        const int acc_idx_464 = value_idx_463;
        float2 _f32x2_mul_a_132 = make_float2(acc[acc_idx_464], acc[acc_idx_464 + 1]);
        float2 scaled_pair_465;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_465) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_133 = make_float2(wout[acc_idx_464], wout[acc_idx_464 + 1]);
        float2 normalized_pair_466;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_466) : "l"(*(const unsigned long long*)&scaled_pair_465), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
        output_values[value_idx_463] = normalized_pair_466.x;
        output_values[value_idx_463 + 1] = normalized_pair_466.y;
        const int value_idx_467 = 4;
        const int acc_idx_468 = value_idx_467;
        float2 _f32x2_mul_a_134 = make_float2(acc[acc_idx_468], acc[acc_idx_468 + 1]);
        float2 scaled_pair_469;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_469) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_135 = make_float2(wout[acc_idx_468], wout[acc_idx_468 + 1]);
        float2 normalized_pair_470;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_470) : "l"(*(const unsigned long long*)&scaled_pair_469), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
        output_values[value_idx_467] = normalized_pair_470.x;
        output_values[value_idx_467 + 1] = normalized_pair_470.y;
        const int value_idx_471 = 6;
        const int acc_idx_472 = value_idx_471;
        float2 _f32x2_mul_a_136 = make_float2(acc[acc_idx_472], acc[acc_idx_472 + 1]);
        float2 scaled_pair_473;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_473) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_137 = make_float2(wout[acc_idx_472], wout[acc_idx_472 + 1]);
        float2 normalized_pair_474;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_474) : "l"(*(const unsigned long long*)&scaled_pair_473), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
        output_values[value_idx_471] = normalized_pair_474.x;
        output_values[value_idx_471 + 1] = normalized_pair_474.y;
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
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_462))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        int base_475 = (2 + group) * 1024 + thread * 8;
        float output_values_476[8];
        const int value_idx_477 = 0;
        const int acc_idx_478 = 8 + value_idx_477;
        float2 _f32x2_mul_a_138 = make_float2(acc[acc_idx_478], acc[acc_idx_478 + 1]);
        float2 scaled_pair_479;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_479) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_139 = make_float2(wout[acc_idx_478], wout[acc_idx_478 + 1]);
        float2 normalized_pair_480;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_480) : "l"(*(const unsigned long long*)&scaled_pair_479), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
        output_values_476[value_idx_477] = normalized_pair_480.x;
        output_values_476[value_idx_477 + 1] = normalized_pair_480.y;
        const int value_idx_481 = 2;
        const int acc_idx_482 = 8 + value_idx_481;
        float2 _f32x2_mul_a_140 = make_float2(acc[acc_idx_482], acc[acc_idx_482 + 1]);
        float2 scaled_pair_483;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_483) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_141 = make_float2(wout[acc_idx_482], wout[acc_idx_482 + 1]);
        float2 normalized_pair_484;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_484) : "l"(*(const unsigned long long*)&scaled_pair_483), "l"(*(const unsigned long long*)&_f32x2_mul_b_141));
        output_values_476[value_idx_481] = normalized_pair_484.x;
        output_values_476[value_idx_481 + 1] = normalized_pair_484.y;
        const int value_idx_485 = 4;
        const int acc_idx_486 = 8 + value_idx_485;
        float2 _f32x2_mul_a_142 = make_float2(acc[acc_idx_486], acc[acc_idx_486 + 1]);
        float2 scaled_pair_487;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_487) : "l"(*(const unsigned long long*)&_f32x2_mul_a_142), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_143 = make_float2(wout[acc_idx_486], wout[acc_idx_486 + 1]);
        float2 normalized_pair_488;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_488) : "l"(*(const unsigned long long*)&scaled_pair_487), "l"(*(const unsigned long long*)&_f32x2_mul_b_143));
        output_values_476[value_idx_485] = normalized_pair_488.x;
        output_values_476[value_idx_485 + 1] = normalized_pair_488.y;
        const int value_idx_489 = 6;
        const int acc_idx_490 = 8 + value_idx_489;
        float2 _f32x2_mul_a_144 = make_float2(acc[acc_idx_490], acc[acc_idx_490 + 1]);
        float2 scaled_pair_491;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_491) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_145 = make_float2(wout[acc_idx_490], wout[acc_idx_490 + 1]);
        float2 normalized_pair_492;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_492) : "l"(*(const unsigned long long*)&scaled_pair_491), "l"(*(const unsigned long long*)&_f32x2_mul_b_145));
        output_values_476[value_idx_489] = normalized_pair_492.x;
        output_values_476[value_idx_489 + 1] = normalized_pair_492.y;
        unsigned int _packed_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_476[_lp*2 + 0], output_values_476[_lp*2+1 + 0]));
            _packed_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_476[_lp*2 + 2], output_values_476[_lp*2+1 + 2]));
            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_476[_lp*2 + 4], output_values_476[_lp*2+1 + 4]));
            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_476[_lp*2 + 6], output_values_476[_lp*2+1 + 6]));
            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_475))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        int base_493 = (4 + group) * 1024 + thread * 8;
        float output_values_494[8];
        const int value_idx_495 = 0;
        const int acc_idx_496 = 16 + value_idx_495;
        float2 _f32x2_mul_a_146 = make_float2(acc[acc_idx_496], acc[acc_idx_496 + 1]);
        float2 scaled_pair_497;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_497) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_147 = make_float2(wout[acc_idx_496], wout[acc_idx_496 + 1]);
        float2 normalized_pair_498;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_498) : "l"(*(const unsigned long long*)&scaled_pair_497), "l"(*(const unsigned long long*)&_f32x2_mul_b_147));
        output_values_494[value_idx_495] = normalized_pair_498.x;
        output_values_494[value_idx_495 + 1] = normalized_pair_498.y;
        const int value_idx_499 = 2;
        const int acc_idx_500 = 16 + value_idx_499;
        float2 _f32x2_mul_a_148 = make_float2(acc[acc_idx_500], acc[acc_idx_500 + 1]);
        float2 scaled_pair_501;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_501) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_149 = make_float2(wout[acc_idx_500], wout[acc_idx_500 + 1]);
        float2 normalized_pair_502;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_502) : "l"(*(const unsigned long long*)&scaled_pair_501), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
        output_values_494[value_idx_499] = normalized_pair_502.x;
        output_values_494[value_idx_499 + 1] = normalized_pair_502.y;
        const int value_idx_503 = 4;
        const int acc_idx_504 = 16 + value_idx_503;
        float2 _f32x2_mul_a_150 = make_float2(acc[acc_idx_504], acc[acc_idx_504 + 1]);
        float2 scaled_pair_505;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_505) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_151 = make_float2(wout[acc_idx_504], wout[acc_idx_504 + 1]);
        float2 normalized_pair_506;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_506) : "l"(*(const unsigned long long*)&scaled_pair_505), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
        output_values_494[value_idx_503] = normalized_pair_506.x;
        output_values_494[value_idx_503 + 1] = normalized_pair_506.y;
        const int value_idx_507 = 6;
        const int acc_idx_508 = 16 + value_idx_507;
        float2 _f32x2_mul_a_152 = make_float2(acc[acc_idx_508], acc[acc_idx_508 + 1]);
        float2 scaled_pair_509;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_509) : "l"(*(const unsigned long long*)&_f32x2_mul_a_152), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_153 = make_float2(wout[acc_idx_508], wout[acc_idx_508 + 1]);
        float2 normalized_pair_510;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_510) : "l"(*(const unsigned long long*)&scaled_pair_509), "l"(*(const unsigned long long*)&_f32x2_mul_b_153));
        output_values_494[value_idx_507] = normalized_pair_510.x;
        output_values_494[value_idx_507 + 1] = normalized_pair_510.y;
        unsigned int _packed_2[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_494[_lp*2 + 0], output_values_494[_lp*2+1 + 0]));
            _packed_2[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_494[_lp*2 + 2], output_values_494[_lp*2+1 + 2]));
            _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_494[_lp*2 + 4], output_values_494[_lp*2+1 + 4]));
            _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_494[_lp*2 + 6], output_values_494[_lp*2+1 + 6]));
            _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(out + (row_base + (unsigned long long)base_493))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        int base_511 = 6144 + group * 512 + thread * 4;
        float output_values_512[8];
        const int value_idx_513 = 0;
        const int acc_idx_514 = 24 + value_idx_513;
        float2 _f32x2_mul_a_154 = make_float2(acc[acc_idx_514], acc[acc_idx_514 + 1]);
        float2 scaled_pair_515;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_515) : "l"(*(const unsigned long long*)&_f32x2_mul_a_154), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_155 = make_float2(wout[acc_idx_514], wout[acc_idx_514 + 1]);
        float2 normalized_pair_516;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_516) : "l"(*(const unsigned long long*)&scaled_pair_515), "l"(*(const unsigned long long*)&_f32x2_mul_b_155));
        output_values_512[value_idx_513] = normalized_pair_516.x;
        output_values_512[value_idx_513 + 1] = normalized_pair_516.y;
        const int value_idx_517 = 2;
        const int acc_idx_518 = 24 + value_idx_517;
        float2 _f32x2_mul_a_156 = make_float2(acc[acc_idx_518], acc[acc_idx_518 + 1]);
        float2 scaled_pair_519;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_pair_519) : "l"(*(const unsigned long long*)&_f32x2_mul_a_156), "l"(*(const unsigned long long*)&rsigma_pair));
        float2 _f32x2_mul_b_157 = make_float2(wout[acc_idx_518], wout[acc_idx_518 + 1]);
        float2 normalized_pair_520;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&normalized_pair_520) : "l"(*(const unsigned long long*)&scaled_pair_519), "l"(*(const unsigned long long*)&_f32x2_mul_b_157));
        output_values_512[value_idx_517] = normalized_pair_520.x;
        output_values_512[value_idx_517 + 1] = normalized_pair_520.y;
        unsigned int _packed_3[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_512[_lp*2 + 0], output_values_512[_lp*2+1 + 0]));
            _packed_3[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_512[_lp*2 + 2], output_values_512[_lp*2+1 + 2]));
            _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(out + (row_base + (unsigned long long)base_511))[0] = reinterpret_cast<int2*>(_packed_3)[0];
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

} // extern "C"
